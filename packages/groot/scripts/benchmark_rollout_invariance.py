from __future__ import annotations

import argparse
import json
from pathlib import Path

from groot_subtask_phase_probe.collect import (
    _add_isaac_root,
    _batch_observation,
    _build_environment,
    _seed_everything,
    _unbatch_actions,
)
from groot_subtask_phase_probe.tasks import TASKS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--action-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=720)
    parser.add_argument("--output-dir", type=Path, default=Path("baseline_artifacts"))
    parser.add_argument("--model-path", default="checkpoints/GR00T-N1.7-LIBERO/libero_10")
    args = parser.parse_args()

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
    results = []
    for task in TASKS.values():
        for episode_index in range(args.episodes):
            seed = args.seed + episode_index
            _seed_everything(seed)
            episode_dir = args.output_dir / task.slug / f"episode_{episode_index:04d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            env = _build_environment(task, contract, episode_dir, args.max_steps)
            success = False
            env_step = 0
            try:
                observation, _ = env.reset(seed=seed)
                while env_step < args.max_steps and not success:
                    batched_actions, _ = sim_policy.get_action(_batch_observation(observation))
                    observation, _, terminated, truncated, info = env.step(
                        _unbatch_actions(batched_actions)
                    )
                    predicate_steps = info.get("probe_predicate_1", [])
                    env_step += int(info.get("n_env_steps", len(predicate_steps)))
                    success_steps = info.get("probe_task_success", [])
                    success = bool(any(success_steps)) or bool(info.get("success", [False])[-1])
                    if terminated or truncated:
                        break
            finally:
                env.close()
            result = {
                "task": task.slug,
                "episode_index": episode_index,
                "seed": seed,
                "success": success,
                "env_steps": env_step,
            }
            results.append(result)
            print(f"{task.slug} episode={episode_index} success={success} steps={env_step}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "baseline_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
