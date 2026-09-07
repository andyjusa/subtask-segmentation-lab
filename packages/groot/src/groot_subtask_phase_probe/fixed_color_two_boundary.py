from __future__ import annotations

import argparse
import gc
import json
import subprocess
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .color_stop import PALETTE, render_beaker
from .predicates import PredicateInfoWrapper
from .tasks import TASKS

INITIAL_COLOR = (42, 111, 219)
CHANGED_COLOR = (224, 180, 52)


def schedule_color_change(
    *,
    predicate_seen: bool,
    post_stove_samples: int,
    delay_samples: int,
    change_enabled: bool,
) -> bool:
    return bool(change_enabled and predicate_seen and post_stove_samples >= delay_samples)


def render_history_observation(
    observation: np.ndarray,
    color: tuple[int, int, int],
    phase: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(observation, dtype=np.uint8)
    if values.ndim == 3:
        frame = render_beaker(values, color, phase)
        return frame, frame
    if values.ndim != 4:
        raise ValueError(f"Expected camera observation [H,W,C] or [T,H,W,C], got {values.shape}")
    rendered = np.stack([render_beaker(frame, color, phase) for frame in values])
    return rendered, rendered[-1]


def _save_h264(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        raise ValueError("Cannot save an empty video")
    height, width = frames[0].shape[:2]
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin is unavailable")
    try:
        for frame in frames:
            process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed to write {path}")


def _build_environment(contract: Any, max_steps: int):
    import gymnasium as gym
    from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs
    from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper

    task = TASKS["stove_moka"]
    if task.env_name not in gym.registry:
        register_libero_envs()
    env = PredicateInfoWrapper(gym.make(task.env_name), task)
    return MultiStepWrapper(
        env,
        contract=contract,
        max_episode_steps=max_steps,
        terminate_on_success=False,
    )


def _save_episode(
    episode_dir: Path,
    arrays: dict[str, list[np.ndarray]],
    labels: dict[str, list[int]],
    metadata: dict[str, Any],
    frames: list[np.ndarray],
    fps: int,
) -> None:
    episode_dir.mkdir(parents=True, exist_ok=True)
    payload = {name: np.stack(values).astype(np.float32) for name, values in arrays.items()}
    payload.update({name: np.asarray(values, dtype=np.int64) for name, values in labels.items()})
    np.savez_compressed(episode_dir / "features.npz", **payload)
    (episode_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    _save_h264(episode_dir / "rollout.mp4", frames, fps)


def collect_episode(
    *,
    episode_index: int,
    seed: int,
    pair_id: int,
    color_mode: str,
    change_enabled: bool,
    delay_samples: int,
    initial_color: tuple[int, int, int],
    changed_color: tuple[int, int, int],
    output_dir: Path,
    sim_policy: Any,
    capture: Any,
    contract: Any,
    max_steps: int,
    post_stove_samples_target: int,
    fps: int,
) -> dict[str, Any]:
    import torch

    from .collect import (
        _batch_observation,
        _decoded_action_chunk,
        _libero_hold_action,
        _raw_state,
        _seed_everything,
        _unbatch_actions,
    )

    _seed_everything(seed)
    env = _build_environment(contract, max_steps)
    arrays: dict[str, list[np.ndarray]] = {}
    labels = {
        "stage": [],
        "terminal": [],
        "boundary": [],
        "stove_boundary": [],
        "color_boundary": [],
        "env_step": [],
        "sample_index": [],
    }
    frames: list[np.ndarray] = []
    inference_ms: list[float] = []
    predicate_seen = False
    stove_boundary_pending = False
    stove_boundary_sample: int | None = None
    stove_boundary_env_step: int | None = None
    color_boundary_sample: int | None = None
    color_boundary_env_step: int | None = None
    post_stove_samples = 0
    env_step = 0
    last_executed_action: dict[str, np.ndarray] | None = None
    stopped_by_env = False
    sample_index = 0
    token_counts: dict[str, int] = {}
    shapes: dict[str, list[int]] = {}

    try:
        observation, _ = env.reset(seed=seed)
        while env_step < max_steps:
            color_changed = schedule_color_change(
                predicate_seen=predicate_seen,
                post_stove_samples=post_stove_samples,
                delay_samples=delay_samples,
                change_enabled=change_enabled,
            )
            is_color_boundary = bool(color_changed and color_boundary_sample is None)
            if is_color_boundary:
                color_boundary_sample = sample_index
                color_boundary_env_step = env_step

            color = changed_color if color_changed else initial_color
            phase = 2 * np.pi * sample_index / max(1, fps)
            policy_observation = dict(observation)
            front_history, front_frame = render_history_observation(
                observation["video.image"], color, phase
            )
            wrist_history, _ = render_history_observation(
                observation["video.wrist_image"], color, phase
            )
            policy_observation["video.image"] = front_history
            policy_observation["video.wrist_image"] = wrist_history

            control_actions = None
            if not predicate_seen:
                control_actions, _ = sim_policy.get_action(_batch_observation(observation))

            capture.begin_step()
            started = perf_counter()
            batched_actions, _ = sim_policy.get_action(_batch_observation(policy_observation))
            torch.cuda.synchronize()
            inference_ms.append((perf_counter() - started) * 1000)
            snapshot = capture.snapshot(
                raw_state=_raw_state(policy_observation),
                decoded_action_chunk=_decoded_action_chunk(batched_actions),
            )
            token_counts = snapshot.token_counts
            shapes = snapshot.shapes
            for name, value in snapshot.arrays.items():
                arrays.setdefault(name, []).append(value)

            is_stove_boundary = stove_boundary_pending
            if is_stove_boundary:
                stove_boundary_pending = False
                stove_boundary_sample = sample_index
            stage = 2 if color_changed else (1 if predicate_seen else 0)
            labels["stage"].append(stage)
            labels["terminal"].append(0)
            labels["boundary"].append(int(is_stove_boundary or is_color_boundary))
            labels["stove_boundary"].append(int(is_stove_boundary))
            labels["color_boundary"].append(int(is_color_boundary))
            labels["env_step"].append(env_step)
            labels["sample_index"].append(sample_index)
            frames.append(front_frame)

            action = _unbatch_actions(
                control_actions if control_actions is not None else batched_actions
            )
            if predicate_seen:
                executed_action_steps = int(
                    getattr(env, "n_action_steps", len(next(iter(action.values()))))
                )
                action = _libero_hold_action(action, last_executed_action, executed_action_steps)
            next_observation, _, terminated, truncated, info = env.step(action)
            if not predicate_seen:
                last_executed_action = action
            predicate_steps = np.asarray(info.get("probe_predicate_1", []), dtype=bool)
            first_true = np.flatnonzero(predicate_steps)
            if not predicate_seen and first_true.size:
                predicate_seen = True
                stove_boundary_pending = True
                stove_boundary_env_step = env_step + int(first_true[0]) + 1
            env_step += int(info.get("n_env_steps", len(predicate_steps)))
            observation = next_observation
            sample_index += 1
            if predicate_seen and not stove_boundary_pending:
                post_stove_samples += 1
            if predicate_seen and post_stove_samples >= post_stove_samples_target:
                break
            if terminated or truncated:
                stopped_by_env = True
                break
    finally:
        env.close()

    metadata = {
        "task": "fixed_color_two_boundary",
        "env_name": TASKS["stove_moka"].env_name,
        "episode_index": episode_index,
        "pair_id": pair_id,
        "seed": seed,
        "color_mode": color_mode,
        "stove_success": stove_boundary_sample is not None,
        "color_change_enabled": change_enabled,
        "initial_color_rgb": list(initial_color),
        "changed_color_rgb": list(changed_color),
        "delay_samples": delay_samples if change_enabled else None,
        "stove_boundary_sample": stove_boundary_sample,
        "stove_boundary_env_step": stove_boundary_env_step,
        "color_boundary_sample": color_boundary_sample,
        "color_boundary_env_step": color_boundary_env_step,
        "n_policy_samples": len(labels["sample_index"]),
        "env_steps": env_step,
        "post_stove_samples_target": post_stove_samples_target,
        "stopped_by_env": stopped_by_env,
        "backbone_token_counts": token_counts,
        "tensor_shapes": shapes,
        "inference_latency_ms": inference_ms,
        "action_after_stove": "discarded; zero-motion hold applied",
        "controller_observation": "raw simulator cameras without synthetic beaker",
        "probe_observation": "same simulator state with synthetic beaker overlay",
        "label_source": {
            "stove": "LIBERO turnon flat_stove_1 predicate",
            "color": "pre-sampled fixed blue-to-yellow schedule",
        },
    }
    episode_dir = (
        output_dir
        / "fixed_color_two_boundary"
        / color_mode
        / f"episode_{episode_index:04d}"
    )
    _save_episode(episode_dir, arrays, labels, metadata, frames, fps)
    return metadata


def run(args: argparse.Namespace) -> None:
    import torch

    from .collect import _add_isaac_root
    from .features import GrootFeatureCapture

    isaac_root = _add_isaac_root()
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval._horizon_contract import PolicyHorizonSpec
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = isaac_root / model_path
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.LIBERO_PANDA,
        model_path=str(model_path),
        device="cuda",
    )
    sim_policy = Gr00tSimPolicyWrapper(policy)
    contract = PolicyHorizonSpec.from_policy(sim_policy, n_action_steps=args.action_steps)
    capture = GrootFeatureCapture(policy)
    output_dir = Path(args.output_dir)
    rng = np.random.default_rng(args.schedule_seed)
    summaries = []
    try:
        for pair_id in range(args.episodes_per_mode):
            change_enabled = pair_id % args.no_change_every != 0
            delay_samples = int(rng.integers(args.min_delay_samples, args.max_delay_samples + 1))
            color_indices = rng.choice(len(PALETTE), size=2, replace=False)
            random_initial = PALETTE[int(color_indices[0])]
            random_changed = PALETTE[int(color_indices[1])]
            for mode_index, color_mode in enumerate(args.color_modes):
                initial_color = INITIAL_COLOR if color_mode == "fixed" else random_initial
                changed_color = CHANGED_COLOR if color_mode == "fixed" else random_changed
                episode_index = pair_id * len(args.color_modes) + mode_index
                summary = collect_episode(
                    episode_index=episode_index,
                    pair_id=pair_id,
                    color_mode=color_mode,
                    seed=args.seed + pair_id,
                    change_enabled=change_enabled,
                    delay_samples=delay_samples,
                    initial_color=initial_color,
                    changed_color=changed_color,
                    output_dir=output_dir,
                    sim_policy=sim_policy,
                    capture=capture,
                    contract=contract,
                    max_steps=args.max_steps,
                    post_stove_samples_target=args.post_stove_samples,
                    fps=args.fps,
                )
                summaries.append(summary)
                print(
                    f"pair={pair_id} mode={color_mode} stove={summary['stove_boundary_sample']} "
                    f"color={summary['color_boundary_sample']} delay={summary['delay_samples']}"
                )
    finally:
        capture.close()
        del sim_policy, policy
        gc.collect()
        torch.cuda.empty_cache()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "collection_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-mode", type=int, default=24)
    parser.add_argument(
        "--color-modes", nargs="+", choices=("fixed", "random"), default=["fixed", "random"]
    )
    parser.add_argument("--seed", type=int, default=8100)
    parser.add_argument("--schedule-seed", type=int, default=9401)
    parser.add_argument("--action-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--post-stove-samples", type=int, default=10)
    parser.add_argument("--min-delay-samples", type=int, default=2)
    parser.add_argument("--max-delay-samples", type=int, default=6)
    parser.add_argument("--no-change-every", type=int, default=4)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--output-dir", default="artifacts_fixed_color_two_boundary")
    parser.add_argument("--model-path", default="checkpoints/GR00T-N1.7-LIBERO/libero_10")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
