from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BoundaryProbeArtifact:
    """Serialized native-dimension linear boundary probe for online inference."""

    representation: str
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    threshold: float

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32).reshape(-1)
        scale = np.asarray(self.scale, dtype=np.float32).reshape(-1)
        coefficient = np.asarray(self.coefficient, dtype=np.float32).reshape(-1)
        if not self.representation:
            raise ValueError("representation must not be empty")
        if mean.shape != scale.shape:
            raise ValueError("mean and scale must have the same shape")
        if coefficient.size != mean.size * 2:
            raise ValueError("coefficient must match concat(z_t, z_t - z_t-1)")
        if np.any(scale <= 0):
            raise ValueError("scale values must be positive")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must be between zero and one")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "coefficient", coefficient)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            representation=np.asarray(self.representation),
            mean=self.mean,
            scale=self.scale,
            coefficient=self.coefficient,
            intercept=np.asarray(self.intercept, dtype=np.float64),
            threshold=np.asarray(self.threshold, dtype=np.float64),
        )

    @classmethod
    def load(cls, path: Path) -> BoundaryProbeArtifact:
        with np.load(Path(path), allow_pickle=False) as archive:
            return cls(
                representation=str(archive["representation"].item()),
                mean=archive["mean"],
                scale=archive["scale"],
                coefficient=archive["coefficient"],
                intercept=float(archive["intercept"].item()),
                threshold=float(archive["threshold"].item()),
            )


@dataclass(frozen=True)
class BoundaryEvent:
    env_step: int
    probability: float
    representation: str


class OnlineBoundaryDetector:
    """Emit the first rising boundary event of each episode."""

    def __init__(self, artifact: BoundaryProbeArtifact):
        self.artifact = artifact
        self.reset()

    @property
    def representation(self) -> str:
        return self.artifact.representation

    def reset(self) -> None:
        self._previous_z: np.ndarray | None = None
        self._previous_above = False
        self._handled = False

    def observe(self, feature: np.ndarray, *, env_step: int) -> BoundaryEvent | None:
        value = np.asarray(feature, dtype=np.float32).reshape(-1)
        if value.shape != self.artifact.mean.shape:
            raise ValueError(
                f"{self.representation} expected {self.artifact.mean.size} values, got {value.size}"
            )
        z = (value - self.artifact.mean) / self.artifact.scale
        delta = np.zeros_like(z) if self._previous_z is None else z - self._previous_z
        probe_input = np.concatenate([z, delta])
        logit = float(probe_input @ self.artifact.coefficient + self.artifact.intercept)
        if logit >= 0:
            probability = 1.0 / (1.0 + float(np.exp(-logit)))
        else:
            exp_logit = float(np.exp(logit))
            probability = exp_logit / (1.0 + exp_logit)
        above = probability >= self.artifact.threshold
        rising = above and not self._previous_above
        self._previous_z = z
        self._previous_above = above
        if self._handled or not rising:
            return None
        self._handled = True
        return BoundaryEvent(
            env_step=int(env_step),
            probability=probability,
            representation=self.representation,
        )


@dataclass(frozen=True)
class PauseConfig:
    seconds: float = 10.0
    control_hz: float = 20.0

    def __post_init__(self) -> None:
        if self.seconds < 0:
            raise ValueError("seconds must be non-negative")
        if self.control_hz <= 0:
            raise ValueError("control_hz must be positive")


@dataclass(frozen=True)
class PauseResult:
    env_step: int
    probability: float
    observation: Any
    physics_steps: int
    macro_steps: int
    simulated_seconds: float
    wall_seconds: float
    terminated: bool
    truncated: bool
    infos: tuple[dict[str, Any], ...]


class OnlinePauseController:
    """Advance physics with hold actions while policy inference is suspended."""

    def __init__(
        self,
        config: PauseConfig | None = None,
    ) -> None:
        self.config = PauseConfig() if config is None else config

    def pause_environment(
        self,
        env: Any,
        *,
        observation: Any,
        hold_action: dict[str, np.ndarray],
        env_step: int,
        probability: float,
        reset_after_resume: Iterable[object] = (),
    ) -> PauseResult:
        action_steps = min(np.asarray(value).shape[0] for value in hold_action.values())
        chunk_steps = int(getattr(env, "n_action_steps", action_steps))
        if chunk_steps <= 0:
            raise ValueError("hold_action must contain at least one physical step")
        if chunk_steps > action_steps:
            raise ValueError("environment action horizon exceeds the hold action length")
        target_steps = math.ceil(self.config.seconds * self.config.control_hz)
        target_macro_steps = math.ceil(target_steps / chunk_steps) if target_steps else 0
        started = time.perf_counter()
        latest_observation = observation
        infos: list[dict[str, Any]] = []
        physics_steps = 0
        terminated = False
        truncated = False
        for _ in range(target_macro_steps):
            latest_observation, _, terminated, truncated, info = env.step(hold_action)
            infos.append(info)
            physics_steps += int(info.get("n_env_steps", chunk_steps))
            success = bool(np.asarray(info.get("success", [False]), dtype=bool).any())
            if terminated or truncated or success:
                break
        for component in reset_after_resume:
            reset = getattr(component, "reset", None)
            if callable(reset):
                reset()
        return PauseResult(
            env_step=int(env_step),
            probability=float(probability),
            observation=latest_observation,
            physics_steps=physics_steps,
            macro_steps=len(infos),
            simulated_seconds=physics_steps / self.config.control_hz,
            wall_seconds=time.perf_counter() - started,
            terminated=bool(terminated),
            truncated=bool(truncated),
            infos=tuple(infos),
        )


class GrootOnlinePauseGate:
    """Detect one boundary and advance the simulator with hold actions."""

    def __init__(
        self,
        detector: OnlineBoundaryDetector,
        controller: OnlinePauseController,
    ) -> None:
        self.detector = detector
        self.controller = controller

    @property
    def representation(self) -> str:
        return self.detector.representation

    def reset(self) -> None:
        self.detector.reset()

    def after_inference(
        self,
        feature: np.ndarray,
        *,
        env_step: int,
    ) -> BoundaryEvent | None:
        return self.detector.observe(feature, env_step=env_step)

    def pause_environment(
        self,
        env: Any,
        *,
        observation: Any,
        hold_action: dict[str, np.ndarray],
        event: BoundaryEvent,
        reset_after_resume: Iterable[object] = (),
    ) -> PauseResult:
        return self.controller.pause_environment(
            env,
            observation=observation,
            hold_action=hold_action,
            env_step=event.env_step,
            probability=event.probability,
            reset_after_resume=reset_after_resume,
        )


def fit_online_boundary_probe(
    episodes_list: list[Any], representation: str, seed: int
) -> BoundaryProbeArtifact:
    """Fit the native probe and select C/threshold on episode-level validation data."""

    from .evaluate import (
        BOUNDARY_THRESHOLDS,
        C_GRID,
        Transform,
        _boundary_labels,
        _boundary_probabilities,
        _episode_map,
        _event_metrics,
        _fit_logistic,
        _stage_xy,
        split_episode_ids,
    )

    episodes = _episode_map(episodes_list)
    train_ids, validation_ids, _ = split_episode_ids(episodes_list, seed)
    stage_train_x, _ = _stage_xy(episodes, train_ids, representation)
    transform = Transform(common_256=False, seed=seed).fit(stage_train_x)

    xs, ys = [], []
    for episode_id in sorted(train_ids):
        episode = episodes[episode_id]
        valid = episode.terminal == 0
        z = transform.apply(episode.features[representation][valid])
        delta = np.vstack([np.zeros((1, z.shape[1]), dtype=z.dtype), np.diff(z, axis=0)])
        xs.append(np.concatenate([z, delta], axis=1))
        ys.append(_boundary_labels(episode)[valid])
    train_x = np.concatenate(xs)
    train_y = np.concatenate(ys)

    best: tuple[tuple[float, float, float], float, Any] | None = None
    for c in C_GRID:
        model = _fit_logistic(train_x, train_y, c)
        validation_probabilities = _boundary_probabilities(
            model, transform, episodes, validation_ids, representation
        )
        for threshold in BOUNDARY_THRESHOLDS:
            metrics = _event_metrics(
                episodes, validation_ids, validation_probabilities, float(threshold)
            )
            score = (
                metrics["boundary_event_f1_at_5"],
                -metrics["false_boundaries_per_episode"],
                -abs(float(threshold) - 0.5),
            )
            if best is None or score > best[0]:
                best = (score, float(threshold), model)
    if best is None:
        raise RuntimeError(f"Could not fit online boundary probe for {representation}")

    return BoundaryProbeArtifact(
        representation=representation,
        mean=transform.scaler.mean_,
        scale=transform.scaler.scale_,
        coefficient=best[2].coef_[0],
        intercept=float(best[2].intercept_[0]),
        threshold=best[1],
    )


def export_main() -> None:
    from .evaluate import load_episodes

    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/online_boundary_probe.npz"))
    parser.add_argument("--representation", default="R0_raw_state")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    artifact = fit_online_boundary_probe(
        load_episodes(args.artifacts), args.representation, args.seed
    )
    artifact.save(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "representation": artifact.representation,
                "feature_dim": int(artifact.mean.size),
                "threshold": artifact.threshold,
                "seed": args.seed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    export_main()
