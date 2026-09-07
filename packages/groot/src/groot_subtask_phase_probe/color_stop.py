from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

PALETTE = (
    (42, 111, 219),
    (225, 70, 70),
    (46, 170, 92),
    (224, 180, 52),
    (150, 82, 210),
)
INSTRUCTION = "Keep stirring and stop immediately when the liquid changes color."
ENV_NAME = "libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"


def render_beaker(frame: np.ndarray, liquid_rgb: tuple[int, int, int], phase: float) -> np.ndarray:
    """Overlay a deterministic beaker and stirring rod on a real simulator frame."""

    image = np.asarray(frame, dtype=np.uint8).copy()
    height, width = image.shape[:2]
    cx, top = int(width * 0.76), int(height * 0.26)
    half_width, bottom = max(18, width // 11), int(height * 0.80)
    left, right = cx - half_width, cx + half_width
    thickness = max(2, width // 160)

    # Dark outline keeps geometry stable while only the liquid RGB changes.
    image[top : top + thickness, left:right] = (235, 235, 235)
    image[top:bottom, left : left + thickness] = (235, 235, 235)
    image[top:bottom, right - thickness : right] = (235, 235, 235)
    image[bottom - thickness : bottom, left:right] = (235, 235, 235)
    liquid_top = int(top + (bottom - top) * 0.46)
    image[liquid_top : bottom - thickness, left + thickness : right - thickness] = liquid_rgb

    # The rod moves identically in positive/no-change counterfactual pairs.
    rod_x = cx + int(np.sin(phase) * half_width * 0.45)
    rod_half = max(1, thickness // 2)
    image[top - 8 : liquid_top + 12, rod_x - rod_half : rod_x + rod_half + 1] = (
        245,
        245,
        245,
    )
    return image


def _replace_instruction(observation: dict[str, Any]) -> dict[str, Any]:
    result = dict(observation)
    annotation_keys = [key for key in result if key.startswith("annotation.")]
    if not annotation_keys:
        raise KeyError("LIBERO observation has no annotation.* key")
    for key in annotation_keys:
        result[key] = INSTRUCTION
    return result


def _add_history_axis(observation: dict[str, Any]) -> dict[str, Any]:
    """Convert a bare LIBERO observation to the T=1 policy history contract."""

    result: dict[str, Any] = {}
    for key, value in observation.items():
        if key.startswith(("video.", "state.")):
            result[key] = np.expand_dims(np.asarray(value), axis=0)
        else:
            result[key] = value
    return result


def _save_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    import cv2

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _save_episode(
    episode_dir: Path,
    arrays: dict[str, list[np.ndarray]],
    stages: list[int],
    boundaries: list[int],
    metadata: dict[str, Any],
    frames: list[np.ndarray],
    fps: int,
) -> None:
    episode_dir.mkdir(parents=True, exist_ok=True)
    payload = {name: np.stack(values).astype(np.float32) for name, values in arrays.items()}
    payload.update(
        {
            "stage": np.asarray(stages, dtype=np.int64),
            "terminal": np.zeros(len(stages), dtype=np.int64),
            "boundary": np.asarray(boundaries, dtype=np.int64),
            "env_step": np.arange(len(stages), dtype=np.int64),
        }
    )
    np.savez_compressed(episode_dir / "features.npz", **payload)
    (episode_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    _save_video(episode_dir / "rollout.mp4", frames, fps)


def _reset_policy(policy: Any) -> None:
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()


def collect_counterfactual_episode(
    *,
    base_observation: dict[str, Any],
    sim_policy: Any,
    capture: Any,
    output_dir: Path,
    pair_id: int,
    changed: bool,
    initial_color: tuple[int, int, int],
    changed_color: tuple[int, int, int],
    change_step: int,
    steps: int,
    fps: int,
) -> dict[str, Any]:
    import torch

    from .collect import _batch_observation, _decoded_action_chunk, _raw_state

    _reset_policy(sim_policy)
    arrays: dict[str, list[np.ndarray]] = {}
    stages: list[int] = []
    boundaries: list[int] = []
    frames: list[np.ndarray] = []
    inference_ms: list[float] = []
    counts: dict[str, int] = {}

    observation = _add_history_axis(_replace_instruction(base_observation))
    for step in range(steps):
        color = changed_color if changed and step >= change_step else initial_color
        phase = 2 * np.pi * step / max(1, fps)
        frame = render_beaker(base_observation["video.image"], color, phase)
        wrist = render_beaker(base_observation["video.wrist_image"], color, phase)
        step_observation = dict(observation)
        step_observation["video.image"] = np.expand_dims(frame, axis=0)
        step_observation["video.wrist_image"] = np.expand_dims(wrist, axis=0)

        capture.begin_step()
        started = perf_counter()
        actions, _ = sim_policy.get_action(_batch_observation(step_observation))
        torch.cuda.synchronize()
        inference_ms.append((perf_counter() - started) * 1000)
        snapshot = capture.snapshot(
            raw_state=_raw_state(step_observation),
            decoded_action_chunk=_decoded_action_chunk(actions),
        )
        counts = snapshot.token_counts
        for name, value in snapshot.arrays.items():
            arrays.setdefault(name, []).append(value)
        stages.append(int(changed and step >= change_step))
        boundaries.append(int(changed and step == change_step))
        frames.append(frame)

    name = (
        f"episode_{pair_id * 2 + int(changed):04d}_pair_{pair_id:03d}_"
        f"{'change' if changed else 'no_change'}"
    )
    episode_dir = output_dir / "color_stop" / name
    metadata = {
        "task": "color_stop",
        "episode_index": pair_id * 2 + int(changed),
        "pair_id": pair_id,
        "condition": "change" if changed else "no_change",
        "instruction": INSTRUCTION,
        "success": bool(changed),
        "exact_boundary_env_step": change_step if changed else None,
        "n_policy_samples": steps,
        "n_terminal_samples": 0,
        "initial_color_rgb": list(initial_color),
        "changed_color_rgb": list(changed_color),
        "backbone_token_counts": counts,
        "inference_latency_ms": inference_ms,
        "paired_controls": "text, state, timing, geometry, and stirring motion are identical",
    }
    _save_episode(episode_dir, arrays, stages, boundaries, metadata, frames, fps)
    return metadata


def run(args: argparse.Namespace) -> None:
    import gymnasium as gym
    import torch

    from .features import GrootFeatureCapture

    isaac_root = Path(os.environ.get("ISAAC_GROOT_ROOT", "/root/projects/Isaac-GR00T"))
    if str(isaac_root) not in sys.path:
        sys.path.insert(0, str(isaac_root))
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    register_libero_envs()
    env = gym.make(ENV_NAME)
    try:
        base_observation, _ = env.reset(seed=args.seed)
    finally:
        env.close()

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = isaac_root / model_path
    gr00t_policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.LIBERO_PANDA,
        model_path=str(model_path),
        device="cuda",
    )
    sim_policy = Gr00tSimPolicyWrapper(gr00t_policy)
    capture = GrootFeatureCapture(gr00t_policy)
    summaries = []
    try:
        rng = np.random.default_rng(args.seed)
        for pair_id in range(args.pairs):
            color_indices = rng.choice(len(PALETTE), size=2, replace=False)
            initial_color = PALETTE[int(color_indices[0])]
            changed_color = PALETTE[int(color_indices[1])]
            change_step = int(rng.integers(args.min_change_step, args.max_change_step + 1))
            for changed in (False, True):
                summary = collect_counterfactual_episode(
                    base_observation=base_observation,
                    sim_policy=sim_policy,
                    capture=capture,
                    output_dir=Path(args.output_dir),
                    pair_id=pair_id,
                    changed=changed,
                    initial_color=initial_color,
                    changed_color=changed_color,
                    change_step=change_step,
                    steps=args.steps,
                    fps=args.fps,
                )
                summaries.append(summary)
                print(
                    f"pair={pair_id} condition={summary['condition']} "
                    f"boundary={summary['exact_boundary_env_step']}"
                )
    finally:
        capture.close()
        del sim_policy, gr00t_policy
        gc.collect()
        torch.cuda.empty_cache()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "collection_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--min-change-step", type=int, default=8)
    parser.add_argument("--max-change-step", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7300)
    parser.add_argument("--output-dir", default="artifacts_color_stop")
    parser.add_argument("--model-path", default="checkpoints/GR00T-N1.7-LIBERO/libero_10")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
