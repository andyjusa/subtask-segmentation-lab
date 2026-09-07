from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .detector import ActionPhaseProbeDetector, ProbeParameters
from .training import train_linear_probe


def _csv(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("at least one comma-separated value is required")
    return items


def _frames(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in _csv(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("frames must be comma-separated integers") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or replay a VLA phase probe")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train")
    train.add_argument("--actions", type=Path, required=True)
    train.add_argument("--transitions", type=_frames, required=True)
    train.add_argument("--phases", type=_csv, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--validation-stride", type=int, default=5)

    replay = commands.add_parser("replay")
    replay.add_argument("--actions", type=Path, required=True)
    replay.add_argument("--probe", type=Path, required=True)
    replay.add_argument("--phases", type=_csv, required=True)
    replay.add_argument("--task-id", required=True)
    replay.add_argument("--subtask-ids", type=_csv, required=True)
    replay.add_argument("--confidence", type=float, default=0.5)
    replay.add_argument("--stable-frames", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    actions = np.load(args.actions).astype(np.float32)
    if args.command == "train":
        if len(args.phases) != len(args.transitions) + 1:
            raise ValueError("phase count must equal transition count + 1")
        result = train_linear_probe(
            actions,
            args.transitions,
            validation_stride=args.validation_stride,
        )
        result.parameters.save(args.output, args.phases)
        payload = {
            "frames": len(actions),
            "action_dim": actions.shape[1],
            "phases": args.phases,
            "ground_truth_transitions": args.transitions,
            "validation_accuracy": result.validation_accuracy,
            "predicted_transitions": [
                {"frame": index, "phase_index": int(result.predictions[index])}
                for index in range(1, len(result.predictions))
                if result.predictions[index] != result.predictions[index - 1]
            ],
        }
        args.output.with_suffix(".json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    else:
        detector = ActionPhaseProbeDetector(
            parameters=ProbeParameters.load(args.probe),
            phases=args.phases,
            task_id=args.task_id,
            subtask_ids=args.subtask_ids,
            confidence_threshold=args.confidence,
            stable_frames=args.stable_frames,
        )
        transitions = [{"frame": 0, **detector.start().__dict__}]
        for frame, action in enumerate(actions, start=1):
            signal = detector.observe(action)
            if signal is not None:
                transitions.append({"frame": frame, **signal.__dict__})
        payload = {"frames": len(actions), "transitions": transitions}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
