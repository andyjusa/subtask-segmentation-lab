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
import torch

from .features import GrootFeatureCapture
from .online_pause import (
    BoundaryProbeArtifact,
    GrootOnlinePauseGate,
    OnlineBoundaryDetector,
    OnlinePauseController,
    PauseConfig,
)
from .predicates import PredicateInfoWrapper
from .tasks import TASKS, TaskSpec


def _add_isaac_root() -> Path:
    root = Path(os.environ.get("ISAAC_GROOT_ROOT", "/root/projects/Isaac-GR00T")).resolve()
    if not (root / "gr00t").is_dir():
        raise FileNotFoundError(f"Isaac-GR00T not found at {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _batch_observation(observation: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in observation.items():
        if key.startswith("state."):
            result[key] = np.expand_dims(np.asarray(value, dtype=np.float32), axis=0)
        elif key.startswith("video."):
            result[key] = np.expand_dims(np.asarray(value, dtype=np.uint8), axis=0)
        elif key.startswith("annotation."):
            result[key] = [str(value)]
        else:
            result[key] = np.expand_dims(np.asarray(value), axis=0)
    return result


def _unbatch_actions(actions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value[0]) for key, value in actions.items()}


def _libero_hold_action(
    action_template: dict[str, np.ndarray],
    last_executed_action: dict[str, np.ndarray] | None,
    executed_action_steps: int,
) -> dict[str, np.ndarray]:
    """Build a zero-motion LIBERO chunk while preserving gripper state."""

    hold: dict[str, np.ndarray] = {}
    for key, value in action_template.items():
        template = np.asarray(value)
        held = np.zeros_like(template)
        if key == "action.gripper":
            if last_executed_action is None:
                # LIBERO starts with the gripper open; its policy-space command is 1.
                gripper = np.ones(template.shape[1:], dtype=template.dtype)
            else:
                gripper_values = np.asarray(last_executed_action[key])
                executed_index = min(executed_action_steps, len(gripper_values)) - 1
                gripper = gripper_values[executed_index]
            held[...] = gripper
        hold[key] = held
    return hold


def _raw_state(observation: dict[str, Any]) -> np.ndarray:
    values = []
    for key in sorted(k for k in observation if k.startswith("state.")):
        value = np.asarray(observation[key], dtype=np.float32)
        values.append(value[-1].reshape(-1))
    return np.concatenate(values).astype(np.float32, copy=False)


def _decoded_action_chunk(actions: dict[str, np.ndarray]) -> np.ndarray:
    values = [np.asarray(actions[key][0], dtype=np.float32) for key in sorted(actions)]
    return np.concatenate(values, axis=-1)


def _process_status_bytes(field: str) -> int:
    # Linux /proc avoids an extra runtime dependency in the simulator island.
    fields = Path("/proc/self/status").read_text().splitlines()
    for line in fields:
        if line.startswith(f"{field}:"):
            return int(line.split()[1]) * 1024
    return 0


def _process_rss_bytes() -> int:
    return _process_status_bytes("VmRSS")


def _process_peak_rss_bytes() -> int:
    return _process_status_bytes("VmHWM")


def _save_episode(
    episode_dir: Path,
    arrays: dict[str, list[np.ndarray]],
    labels: dict[str, list[int]],
    metadata: dict[str, Any],
) -> None:
    episode_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        name: np.stack(values).astype(np.float32) for name, values in arrays.items()
    }
    payload.update({name: np.asarray(values, dtype=np.int64) for name, values in labels.items()})
    np.savez_compressed(episode_dir / "features.npz", **payload)
    (episode_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    videos = sorted(episode_dir.glob("*.mp4"))
    if len(videos) == 1 and videos[0].name != "rollout.mp4":
        videos[0].rename(episode_dir / "rollout.mp4")


def _build_environment(task: TaskSpec, contract, episode_dir: Path, max_steps: int):
    import gymnasium as gym
    from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs
    from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
    from gr00t.eval.sim.wrapper.video_recording_wrapper import VideoRecordingWrapper

    if task.env_name not in gym.registry:
        register_libero_envs()
    env = gym.make(task.env_name)
    env = PredicateInfoWrapper(env, task)
    env = VideoRecordingWrapper(
        env,
        video_dir=episode_dir,
        steps_per_render=1,
        max_episode_steps=max_steps,
        fps=20,
        codec="h264",
        overlay_text=False,
        record_video_keys=("video.image", "video.wrist_image"),
    )
    return MultiStepWrapper(
        env,
        contract=contract,
        max_episode_steps=max_steps,
        terminate_on_success=True,
    )


def collect_episode(
    task: TaskSpec,
    episode_index: int,
    seed: int,
    episode_dir: Path,
    sim_policy,
    capture: GrootFeatureCapture,
    contract,
    max_steps: int,
    online_pause_gate: GrootOnlinePauseGate | None = None,
) -> dict[str, Any]:
    _seed_everything(seed)
    episode_dir.mkdir(parents=True, exist_ok=True)
    env = _build_environment(task, contract, episode_dir, max_steps)
    arrays: dict[str, list[np.ndarray]] = {}
    labels = {"stage": [], "terminal": [], "boundary": [], "env_step": []}
    inference_ms: list[float] = []
    hook_ms: list[float] = []
    snapshot_ms: list[float] = []
    extraction_ms: dict[str, list[float]] = {}
    pause_events: list[dict[str, float | int | str]] = []
    discarded_inference_ms: list[float] = []
    exact_boundary_step: int | None = None
    predicate_seen = False
    success = False
    episode_done = False
    env_step = 0
    last_executed_action: dict[str, np.ndarray] | None = None
    shapes: dict[str, list[int]] = {}
    token_counts: dict[str, int] = {}
    baseline_rss = _process_rss_bytes()
    baseline_peak_rss = _process_peak_rss_bytes()
    torch.cuda.reset_peak_memory_stats()
    if online_pause_gate is not None:
        online_pause_gate.reset()

    try:
        observation, _ = env.reset(seed=seed)
        while env_step < max_steps and not success and not episode_done:
            stage_before = 1 if predicate_seen else 0
            capture.begin_step()
            started = perf_counter()
            batched_actions, _ = sim_policy.get_action(_batch_observation(observation))
            torch.cuda.synchronize()
            inference_elapsed_ms = (perf_counter() - started) * 1000

            snapshot = capture.snapshot(
                raw_state=_raw_state(observation),
                decoded_action_chunk=_decoded_action_chunk(batched_actions),
            )
            shapes = snapshot.shapes
            token_counts = snapshot.token_counts
            if online_pause_gate is not None:
                representation = online_pause_gate.representation
                if representation not in snapshot.arrays:
                    raise KeyError(f"Online pause representation is unavailable: {representation}")
                event = online_pause_gate.after_inference(
                    snapshot.arrays[representation],
                    env_step=env_step,
                )
                if event is not None:
                    # The action chunk that triggered the boundary is intentionally discarded.
                    # Physics and cameras continue under a zero-motion hold action. No policy
                    # inference runs until the latest post-pause observation is available.
                    discarded_inference_ms.append(inference_elapsed_ms)
                    action_template = _unbatch_actions(batched_actions)
                    executed_action_steps = int(
                        getattr(env, "n_action_steps", len(next(iter(action_template.values()))))
                    )
                    hold_action = _libero_hold_action(
                        action_template,
                        last_executed_action,
                        executed_action_steps,
                    )
                    pause = online_pause_gate.pause_environment(
                        env,
                        observation=observation,
                        hold_action=hold_action,
                        event=event,
                        reset_after_resume=(sim_policy,),
                    )
                    pause_offset = 0
                    for pause_info in pause.infos:
                        predicate_steps = np.asarray(
                            pause_info.get("probe_predicate_1", []), dtype=bool
                        )
                        success_steps = np.asarray(
                            pause_info.get("probe_task_success", []), dtype=bool
                        )
                        first_true = np.flatnonzero(predicate_steps)
                        if exact_boundary_step is None and first_true.size > 0:
                            exact_boundary_step = env_step + pause_offset + int(first_true[0]) + 1
                        predicate_seen = predicate_seen or bool(predicate_steps.any())
                        success = (
                            success
                            or bool(success_steps.any())
                            or bool(np.asarray(pause_info.get("success", [False])).any())
                        )
                        pause_offset += int(pause_info.get("n_env_steps", len(predicate_steps)))
                    env_step += pause.physics_steps
                    observation = pause.observation
                    episode_done = pause.terminated or pause.truncated
                    pause_events.append(
                        {
                            "env_step": pause.env_step,
                            "probability": pause.probability,
                            "simulated_seconds": pause.simulated_seconds,
                            "wall_seconds": pause.wall_seconds,
                            "physics_steps": pause.physics_steps,
                            "macro_steps": pause.macro_steps,
                            "inference_calls_during_pause": 0,
                            "hold_motion_max_abs": max(
                                float(np.abs(value).max(initial=0.0))
                                for key, value in hold_action.items()
                                if key != "action.gripper"
                            ),
                            "hold_gripper_command": (
                                float(np.asarray(hold_action["action.gripper"])[0, 0])
                                if "action.gripper" in hold_action
                                else None
                            ),
                            "representation": representation,
                        }
                    )
                    continue

            inference_ms.append(inference_elapsed_ms)
            hook_ms.append(snapshot.hook_latency_ms)
            snapshot_ms.append(snapshot.snapshot_latency_ms)
            for name, latency_ms in snapshot.extraction_latency_ms.items():
                extraction_ms.setdefault(name, []).append(latency_ms)
            for name, value in snapshot.arrays.items():
                arrays.setdefault(name, []).append(value)

            action = _unbatch_actions(batched_actions)
            next_observation, _, terminated, truncated, info = env.step(action)
            last_executed_action = action
            predicate_steps = np.asarray(info.get("probe_predicate_1", []), dtype=bool)
            success_steps = np.asarray(info.get("probe_task_success", []), dtype=bool)
            first_true = np.flatnonzero(predicate_steps)
            is_boundary_macro = exact_boundary_step is None and first_true.size > 0
            if is_boundary_macro:
                exact_boundary_step = env_step + int(first_true[0]) + 1
            predicate_seen = predicate_seen or bool(predicate_steps.any())
            success = success or bool(success_steps.any()) or bool(info.get("success", [False])[-1])

            labels["stage"].append(stage_before)
            labels["terminal"].append(0)
            labels["boundary"].append(int(is_boundary_macro))
            labels["env_step"].append(env_step)
            env_step += int(info.get("n_env_steps", len(predicate_steps)))
            observation = next_observation
            if terminated or truncated:
                episode_done = True
                break

        # A post-success observation supplies a terminal sample without executing its action.
        if success:
            capture.begin_step()
            terminal_actions, _ = sim_policy.get_action(_batch_observation(observation))
            terminal_snapshot = capture.snapshot(
                raw_state=_raw_state(observation),
                decoded_action_chunk=_decoded_action_chunk(terminal_actions),
            )
            for name, value in terminal_snapshot.arrays.items():
                arrays.setdefault(name, []).append(value)
            labels["stage"].append(-1)
            labels["terminal"].append(1)
            labels["boundary"].append(0)
            labels["env_step"].append(env_step)
    finally:
        env.close()

    metadata = {
        "task": task.slug,
        "env_name": task.env_name,
        "episode_index": episode_index,
        "seed": seed,
        "success": success,
        "env_steps": env_step,
        "predicate_1": list(task.predicate_1),
        "predicate_1_description": task.predicate_1_description,
        "predicate_2_description": task.predicate_2_description,
        "exact_boundary_env_step": exact_boundary_step,
        "n_policy_samples": len(labels["stage"]),
        "n_terminal_samples": int(sum(labels["terminal"])),
        "tensor_shapes": shapes,
        "backbone_token_counts": token_counts,
        "denoise_iterations": int(capture.head.num_inference_timesteps),
        "inference_latency_ms": inference_ms,
        "hook_latency_ms": hook_ms,
        "snapshot_latency_ms": snapshot_ms,
        "feature_extraction_latency_ms": extraction_ms,
        "online_pause_events": pause_events,
        "discarded_pause_inference_latency_ms": discarded_inference_ms,
        "discarded_action_chunks": len(pause_events),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "host_rss_delta_bytes": max(0, _process_rss_bytes() - baseline_rss),
        "host_peak_rss_delta_bytes": max(0, _process_peak_rss_bytes() - baseline_peak_rss),
    }
    _save_episode(episode_dir, arrays, labels, metadata)
    return metadata


def run(args: argparse.Namespace) -> None:
    isaac_root = _add_isaac_root()
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval._horizon_contract import PolicyHorizonSpec
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = isaac_root / model_path
    gr00t_policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.LIBERO_PANDA,
        model_path=str(model_path),
        device="cuda",
    )
    sim_policy = Gr00tSimPolicyWrapper(gr00t_policy)
    contract = PolicyHorizonSpec.from_policy(sim_policy, n_action_steps=args.action_steps)
    capture = GrootFeatureCapture(gr00t_policy)
    online_pause_gate = None
    if args.online_probe is not None:
        artifact = BoundaryProbeArtifact.load(args.online_probe)
        online_pause_gate = GrootOnlinePauseGate(
            OnlineBoundaryDetector(artifact),
            OnlinePauseController(
                PauseConfig(seconds=args.pause_seconds, control_hz=args.pause_control_hz)
            ),
        )
    output = Path(args.output_dir)
    tasks = list(TASKS) if args.tasks == ["all"] else args.tasks
    summaries = []
    try:
        for task_name in tasks:
            task = TASKS[task_name]
            for episode_index in range(args.episodes):
                seed = args.seed + episode_index
                episode_dir = output / task.slug / f"episode_{episode_index:04d}"
                summary = collect_episode(
                    task=task,
                    episode_index=episode_index,
                    seed=seed,
                    episode_dir=episode_dir,
                    sim_policy=sim_policy,
                    capture=capture,
                    contract=contract,
                    max_steps=args.max_steps,
                    online_pause_gate=online_pause_gate,
                )
                summaries.append(summary)
                print(
                    f"{task.slug} episode={episode_index} success={summary['success']} "
                    f"steps={summary['env_steps']} boundary={summary['exact_boundary_env_step']}"
                )
    finally:
        capture.close()
        del sim_policy, gr00t_policy
        gc.collect()
        torch.cuda.empty_cache()
    (output / "collection_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=["all", *TASKS], default=["all"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--action-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=720)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--model-path", default="checkpoints/GR00T-N1.7-LIBERO/libero_10")
    parser.add_argument(
        "--online-probe",
        type=Path,
        help="Serialized boundary probe. Enables physics-continuous Online Pause/Resume.",
    )
    parser.add_argument("--pause-seconds", type=float, default=10.0)
    parser.add_argument("--pause-control-hz", type=float, default=20.0)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
