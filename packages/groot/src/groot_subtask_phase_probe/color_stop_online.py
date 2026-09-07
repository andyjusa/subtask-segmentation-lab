from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .evaluate import Episode, Transform


@dataclass(frozen=True)
class ColorStopArtifact:
    """Portable common-256 MLP used by the causal color-stop detector."""

    representation: str
    seed: int
    threshold: float
    confirm_steps: int
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    coefficients: tuple[np.ndarray, ...]
    intercepts: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if self.confirm_steps < 1:
            raise ValueError("confirm_steps must be positive")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must be between zero and one")
        if len(self.coefficients) != len(self.intercepts) or not self.coefficients:
            raise ValueError("MLP coefficients and intercepts must have matching layers")

    def transform(self, feature: np.ndarray) -> np.ndarray:
        value = np.asarray(feature, dtype=np.float32).reshape(1, -1)
        if value.shape[1] != self.scaler_mean.size:
            raise ValueError(
                f"{self.representation} expected {self.scaler_mean.size} values, "
                f"got {value.shape[1]}"
            )
        scaled = (value - self.scaler_mean) / self.scaler_scale
        projected = (scaled - self.pca_mean) @ self.pca_components.T
        if projected.shape[1] < 256:
            projected = np.pad(projected, ((0, 0), (0, 256 - projected.shape[1])))
        return projected[0].astype(np.float32, copy=False)

    def probability(self, current: np.ndarray, previous: np.ndarray | None) -> float:
        delta = np.zeros_like(current) if previous is None else current - previous
        hidden = np.concatenate([current, delta]).reshape(1, -1)
        for coefficient, intercept in zip(self.coefficients[:-1], self.intercepts[:-1]):
            hidden = np.maximum(0.0, hidden @ coefficient + intercept)
        logit = float((hidden @ self.coefficients[-1] + self.intercepts[-1]).item())
        if logit >= 0:
            return 1.0 / (1.0 + float(np.exp(-logit)))
        exp_logit = float(np.exp(logit))
        return exp_logit / (1.0 + exp_logit)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "metadata": np.asarray(
                json.dumps(
                    {
                        "representation": self.representation,
                        "seed": self.seed,
                        "threshold": self.threshold,
                        "confirm_steps": self.confirm_steps,
                        "layers": len(self.coefficients),
                    }
                )
            ),
            "scaler_mean": self.scaler_mean,
            "scaler_scale": self.scaler_scale,
            "pca_mean": self.pca_mean,
            "pca_components": self.pca_components,
        }
        for index, (coefficient, intercept) in enumerate(
            zip(self.coefficients, self.intercepts)
        ):
            payload[f"coefficient_{index}"] = coefficient
            payload[f"intercept_{index}"] = intercept
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: Path) -> ColorStopArtifact:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            layers = int(metadata["layers"])
            return cls(
                representation=str(metadata["representation"]),
                seed=int(metadata["seed"]),
                threshold=float(metadata["threshold"]),
                confirm_steps=int(metadata["confirm_steps"]),
                scaler_mean=archive["scaler_mean"].astype(np.float32),
                scaler_scale=archive["scaler_scale"].astype(np.float32),
                pca_mean=archive["pca_mean"].astype(np.float32),
                pca_components=archive["pca_components"].astype(np.float32),
                coefficients=tuple(
                    archive[f"coefficient_{index}"].astype(np.float32)
                    for index in range(layers)
                ),
                intercepts=tuple(
                    archive[f"intercept_{index}"].astype(np.float32)
                    for index in range(layers)
                ),
            )


class ColorStopDetector:
    """Causal one-shot detector with validation-selected temporal confirmation."""

    def __init__(self, artifact: ColorStopArtifact):
        self.artifact = artifact
        self.reset()

    def reset(self) -> None:
        self.previous: np.ndarray | None = None
        self.consecutive = 0
        self.handled = False

    def observe(self, feature: np.ndarray, step: int) -> tuple[int, float] | None:
        current = self.artifact.transform(feature)
        probability = self.artifact.probability(current, self.previous)
        self.previous = current
        self.consecutive = self.consecutive + 1 if probability >= self.artifact.threshold else 0
        if self.handled or self.consecutive < self.artifact.confirm_steps:
            return None
        self.handled = True
        return int(step), probability


def _first_confirmed(probabilities: np.ndarray, threshold: float, confirm_steps: int) -> int | None:
    consecutive = 0
    for index, probability in enumerate(probabilities):
        consecutive = consecutive + 1 if probability >= threshold else 0
        if consecutive >= confirm_steps:
            return index
    return None


def _event_metrics(
    episodes: dict[str, Episode],
    ids: set[str],
    probabilities: dict[str, np.ndarray],
    threshold: float,
    confirm_steps: int,
) -> dict[str, float]:
    counts = {5: [0, 0, 0], 10: [0, 0, 0]}
    errors: list[float] = []
    false_events = 0
    no_change = 0
    no_change_fp = 0
    for episode_id in sorted(ids):
        episode = episodes[episode_id]
        index = _first_confirmed(probabilities[episode_id], threshold, confirm_steps)
        valid_steps = episode.env_step[episode.terminal == 0]
        prediction = None if index is None else int(valid_steps[index])
        false_events += int(
            prediction is not None
            and (
                episode.boundary_step is None
                or abs(prediction - episode.boundary_step) > 5
            )
        )
        if episode.boundary_step is None:
            no_change += 1
            no_change_fp += int(prediction is not None)
        if prediction is not None and episode.boundary_step is not None:
            errors.append(abs(prediction - episode.boundary_step))
        for tolerance, (tp, fp, fn) in counts.items():
            if episode.boundary_step is None:
                fp += int(prediction is not None)
            elif prediction is None:
                fn += 1
            elif abs(prediction - episode.boundary_step) <= tolerance:
                tp += 1
            else:
                fp += 1
                fn += 1
            counts[tolerance] = [tp, fp, fn]
    result: dict[str, float] = {}
    for tolerance, (tp, fp, fn) in counts.items():
        denominator = 2 * tp + fp + fn
        result[f"boundary_event_f1_at_{tolerance}"] = (
            0.0 if denominator == 0 else 2 * tp / denominator
        )
    result["boundary_median_abs_error"] = (
        float(np.median(errors)) if errors else float("nan")
    )
    result["false_stops_per_episode"] = false_events / max(1, len(ids))
    result["no_change_fpr"] = no_change_fp / max(1, no_change)
    return result


def _probabilities(
    model: Any,
    transform: Transform,
    episodes: dict[str, Episode],
    ids: set[str],
    representation: str,
) -> dict[str, np.ndarray]:
    result = {}
    for episode_id in ids:
        episode = episodes[episode_id]
        valid = episode.terminal == 0
        z = transform.apply(episode.features[representation][valid])
        delta = np.vstack([np.zeros((1, z.shape[1]), dtype=z.dtype), np.diff(z, axis=0)])
        result[episode_id] = model.predict_proba(np.concatenate([z, delta], axis=1))[:, 1]
    return result


def fit_color_stop_artifact(
    artifacts: Path,
    *,
    representation: str = "R3b_vision_tokens",
    seed: int = 17,
) -> tuple[ColorStopArtifact, dict[str, Any]]:
    from .color_stop_probe import paired_split_ids
    from .evaluate import BOUNDARY_THRESHOLDS, Transform, load_episodes
    from .modality_probe import (
        _boundary_xy,
        _fit_classifier,
        _stage_xy,
        add_modality_representations,
        calibration_from_artifacts,
    )

    episodes_list = load_episodes(artifacts)
    calibration = calibration_from_artifacts(artifacts)
    for episode in episodes_list:
        add_modality_representations(episode, calibration)
    episodes = {episode.episode_id: episode for episode in episodes_list}
    train_ids, validation_ids, test_ids = paired_split_ids(episodes_list, artifacts, seed)

    train_x, _ = _stage_xy(episodes, train_ids, representation)
    transform = Transform(common_256=True, seed=seed).fit(train_x)
    boundary_x, boundary_y = _boundary_xy(episodes, train_ids, representation, transform)
    model = _fit_classifier("mlp", boundary_x, boundary_y, 1e-3, seed)
    validation_probabilities = _probabilities(
        model, transform, episodes, validation_ids, representation
    )

    best: tuple[tuple[float, ...], float, int, dict[str, float]] | None = None
    for threshold in BOUNDARY_THRESHOLDS:
        for confirm_steps in (1, 2, 3):
            metrics = _event_metrics(
                episodes,
                validation_ids,
                validation_probabilities,
                float(threshold),
                confirm_steps,
            )
            score = (
                metrics["boundary_event_f1_at_5"],
                -metrics["no_change_fpr"],
                -metrics["false_stops_per_episode"],
                -confirm_steps,
                -abs(float(threshold) - 0.5),
            )
            if best is None or score > best[0]:
                best = (score, float(threshold), confirm_steps, metrics)
    if best is None or transform.pca is None:
        raise RuntimeError("Could not fit color-stop artifact")

    artifact = ColorStopArtifact(
        representation=representation,
        seed=seed,
        threshold=best[1],
        confirm_steps=best[2],
        scaler_mean=transform.scaler.mean_.astype(np.float32),
        scaler_scale=transform.scaler.scale_.astype(np.float32),
        pca_mean=transform.pca.mean_.astype(np.float32),
        pca_components=transform.pca.components_.astype(np.float32),
        coefficients=tuple(value.astype(np.float32) for value in model.coefs_),
        intercepts=tuple(value.astype(np.float32) for value in model.intercepts_),
    )
    test_probabilities = _probabilities(model, transform, episodes, test_ids, representation)
    test_metrics = _event_metrics(
        episodes, test_ids, test_probabilities, artifact.threshold, artifact.confirm_steps
    )
    report = {
        "representation": representation,
        "classifier": "MLP(128,64), 10 epochs",
        "projection": "train-only PCA common-256",
        "seed": seed,
        "threshold": artifact.threshold,
        "confirm_steps": artifact.confirm_steps,
        "validation_metrics": best[3],
        "held_out_test_metrics": test_metrics,
        "split": {
            "train": sorted(train_ids),
            "validation": sorted(validation_ids),
            "test": sorted(test_ids),
        },
    }
    return artifact, report


def _annotate(frame: np.ndarray, *, step: int, stop_step: int | None, probability: float) -> np.ndarray:
    import cv2

    image = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
    stopped = stop_step is not None and step >= stop_step
    status = "STOPPED - inference suspended" if stopped else "RUNNING"
    color = (40, 40, 230) if stopped else (40, 210, 40)
    cv2.rectangle(image, (8, 8), (image.shape[1] - 8, 58), (12, 12, 12), -1)
    cv2.putText(image, status, (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
    cv2.putText(
        image,
        f"step={step:02d}  boundary-prob={probability:.3f}",
        (18, 51),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (235, 235, 235),
        1,
    )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _write_h264(path: Path, frames: list[np.ndarray], fps: int) -> None:
    from .color_stop import _save_video

    temporary = path.with_suffix(".temporary.mp4")
    _save_video(temporary, frames, fps)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(temporary),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        check=True,
    )
    temporary.unlink()


def run_live_demo(
    artifacts: Path,
    artifact: ColorStopArtifact,
    report: dict[str, Any],
    output_dir: Path,
    model_path: str,
) -> list[dict[str, Any]]:
    """Re-run GR00T causally; after stop, camera frames continue but inference does not."""

    import gymnasium as gym
    import torch

    from .collect import _batch_observation, _decoded_action_chunk, _raw_state
    from .color_stop import (
        ENV_NAME,
        _add_history_axis,
        _replace_instruction,
        render_beaker,
    )
    from .features import GrootFeatureCapture

    isaac_root = Path(os.environ.get("ISAAC_GROOT_ROOT", "/root/projects/Isaac-GR00T"))
    if str(isaac_root) not in sys.path:
        sys.path.insert(0, str(isaac_root))
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    register_libero_envs()
    environment = gym.make(ENV_NAME)
    try:
        base_observation, _ = environment.reset(seed=7300)
    finally:
        environment.close()
    resolved_model_path = Path(model_path)
    if not resolved_model_path.is_absolute():
        resolved_model_path = isaac_root / resolved_model_path
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.LIBERO_PANDA,
        model_path=str(resolved_model_path),
        device="cuda",
    )
    sim_policy = Gr00tSimPolicyWrapper(policy)
    capture = GrootFeatureCapture(policy)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    try:
        test_ids = report["split"]["test"]
        selected = {}
        for episode_id in test_ids:
            metadata_path = artifacts / episode_id / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            selected.setdefault(metadata["condition"], (episode_id, metadata))
        for condition in ("change", "no_change"):
            episode_id, metadata = selected[condition]
            detector = ColorStopDetector(artifact)
            reset = getattr(sim_policy, "reset", None)
            if callable(reset):
                reset()
            frames: list[np.ndarray] = []
            probabilities: list[float] = []
            stop_step: int | None = None
            inference_calls = 0
            inference_calls_after_stop = 0
            freeze_phase: float | None = None
            observation = _add_history_axis(_replace_instruction(base_observation))
            steps = int(metadata["n_policy_samples"])
            fps = 5
            for step in range(steps):
                changed = condition == "change" and step >= int(metadata["exact_boundary_env_step"])
                color = tuple(metadata["changed_color_rgb"] if changed else metadata["initial_color_rgb"])
                phase = freeze_phase if freeze_phase is not None else 2 * np.pi * step / fps
                frame = render_beaker(base_observation["video.image"], color, phase)
                wrist = render_beaker(base_observation["video.wrist_image"], color, phase)
                probability = probabilities[-1] if probabilities else 0.0
                if stop_step is None:
                    step_observation = dict(observation)
                    step_observation["video.image"] = np.expand_dims(frame, axis=0)
                    step_observation["video.wrist_image"] = np.expand_dims(wrist, axis=0)
                    capture.begin_step()
                    actions, _ = sim_policy.get_action(_batch_observation(step_observation))
                    inference_calls += 1
                    snapshot = capture.snapshot(
                        raw_state=_raw_state(step_observation),
                        decoded_action_chunk=_decoded_action_chunk(actions),
                    )
                    current = artifact.transform(snapshot.arrays[artifact.representation])
                    probability = artifact.probability(current, detector.previous)
                    event = detector.observe(snapshot.arrays[artifact.representation], step)
                    if event is not None:
                        stop_step = event[0]
                        probability = event[1]
                        freeze_phase = phase
                else:
                    inference_calls_after_stop += 0
                probabilities.append(probability)
                frames.append(
                    _annotate(frame, step=step, stop_step=stop_step, probability=probability)
                )
            video_path = output_dir / f"color_stop_online_{condition}.mp4"
            _write_h264(video_path, frames, fps)
            ground_truth = metadata["exact_boundary_env_step"]
            summaries.append(
                {
                    "episode_id": episode_id,
                    "condition": condition,
                    "ground_truth_boundary": ground_truth,
                    "predicted_stop_step": stop_step,
                    "absolute_timing_error": (
                        None if ground_truth is None or stop_step is None else abs(stop_step - ground_truth)
                    ),
                    "inference_calls": inference_calls,
                    "inference_calls_after_stop": inference_calls_after_stop,
                    "video": str(video_path),
                }
            )
    finally:
        capture.close()
        del sim_policy, policy
        gc.collect()
        torch.cuda.empty_cache()
    (output_dir / "live_demo_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n"
    )
    return summaries


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.artifact:
        artifact = ColorStopArtifact.load(Path(args.artifact))
        if not args.report:
            raise ValueError("--report is required with --artifact")
        report = json.loads(Path(args.report).read_text())
    else:
        artifact, report = fit_color_stop_artifact(
            Path(args.artifacts), representation=args.representation, seed=args.seed
        )
        artifact_path = output_dir / "color_stop_online_probe.npz"
        artifact.save(artifact_path)
        (output_dir / "offline_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
    if args.live:
        report["live_demo"] = run_live_demo(
            Path(args.artifacts), artifact, report, output_dir, args.model_path
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts_color_stop")
    parser.add_argument("--output-dir", default="reports/color_stop_online")
    parser.add_argument("--representation", default="R3b_vision_tokens")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--artifact")
    parser.add_argument("--report")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model-path", default="checkpoints/GR00T-N1.7-LIBERO/libero_10")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
