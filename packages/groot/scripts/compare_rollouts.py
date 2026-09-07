from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("instrumented", type=Path)
    parser.add_argument("baseline_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline_json.read_text())
    instrumented = {}
    for metadata_path in args.instrumented.glob("*/*/metadata.json"):
        metadata = json.loads(metadata_path.read_text())
        key = (str(metadata["task"]), int(metadata["episode_index"]))
        instrumented[key] = metadata

    success_mismatches = []
    step_mismatches = []
    paired = []
    for baseline_row in baseline:
        key = (str(baseline_row["task"]), int(baseline_row["episode_index"]))
        measured = instrumented[key]
        row = {
            "task": key[0],
            "episode_index": key[1],
            "seed": int(baseline_row["seed"]),
            "baseline_success": bool(baseline_row["success"]),
            "instrumented_success": bool(measured["success"]),
            "baseline_steps": int(baseline_row["env_steps"]),
            "instrumented_steps": int(measured["env_steps"]),
        }
        paired.append(row)
        if row["baseline_success"] != row["instrumented_success"]:
            success_mismatches.append(row)
        if row["baseline_steps"] != row["instrumented_steps"]:
            step_mismatches.append(row)

    result = {
        "paired_episodes": len(paired),
        "baseline_successes": sum(row["baseline_success"] for row in paired),
        "instrumented_successes": sum(row["instrumented_success"] for row in paired),
        "success_mismatches": success_mismatches,
        "step_mismatches": step_mismatches,
        "pairs": paired,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "pairs"}, indent=2))
    if success_mismatches or step_mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
