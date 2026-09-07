from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def _metadata_map(artifacts: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in artifacts.glob("fixed_color_two_boundary/*/episode_*/metadata.json"):
        episode_id = str(path.parent.relative_to(artifacts))
        result[episode_id] = json.loads(path.read_text())
    return result


def _is_exact(row: dict[str, Any], metadata: dict[str, Any]) -> bool:
    stove_prediction, stove_count = row["stove_predictions"][metadata["episode_id"]]
    color_prediction, color_count = row["color_predictions"][metadata["episode_id"]]
    return bool(
        metadata["color_change_enabled"]
        and stove_prediction == metadata["stove_boundary_sample"]
        and color_prediction == metadata["color_boundary_sample"]
        and stove_count == 1
        and color_count == 1
    )


def _render(
    source: Path,
    output: Path,
    *,
    label: str,
    seed: int,
    stove_gt: int | None,
    stove_prediction: int | None,
    stove_count: int,
    color_gt: int | None,
    color_prediction: int | None,
    color_count: int,
) -> None:
    filters = [
        "drawbox=x=0:y=0:w=iw:h=66:color=black@0.72:t=fill",
        f"drawtext=text='{label}  seed {seed}':x=8:y=7:fontsize=15:fontcolor=white",
        (
            f"drawtext=text='stove GT {stove_gt} pred {stove_prediction} events {stove_count}'"
            ":x=8:y=28:fontsize=13:fontcolor=white"
        ),
        (
            f"drawtext=text='color GT {color_gt} pred {color_prediction} events {color_count}'"
            ":x=8:y=48:fontsize=13:fontcolor=white"
        ),
    ]
    if stove_gt is not None:
        filters.append(
            f"drawbox=x=3:y=3:w=iw-6:h=ih-6:color=green:t=6:enable='eq(n,{stove_gt})'"
        )
    if stove_prediction is not None:
        filters.append(
            f"drawbox=x=10:y=10:w=iw-20:h=ih-20:color=cyan:t=5:enable='eq(n,{stove_prediction})'"
        )
    if color_gt is not None:
        filters.append(
            f"drawbox=x=3:y=3:w=iw-6:h=ih-6:color=yellow:t=6:enable='eq(n,{color_gt})'"
        )
    if color_prediction is not None:
        filters.append(
            f"drawbox=x=10:y=10:w=iw-20:h=ih-20:color=red:t=5:enable='eq(n,{color_prediction})'"
        )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vf",
            ",".join(filters),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def run(args: argparse.Namespace) -> None:
    rows = json.loads(args.results.read_text())
    metadata = _metadata_map(args.artifacts)
    for episode_id, item in metadata.items():
        item["episode_id"] = episode_id

    fixed_rows = sorted(
        (row for row in rows if row["color_mode"] == "fixed"),
        key=lambda row: row["ordered_episode_accuracy_at_1"],
        reverse=True,
    )
    best = next(
        (row, metadata[episode_id])
        for row in fixed_rows
        for episode_id in row["stove_predictions"]
        if _is_exact(row, metadata[episode_id])
    )

    random_rows = sorted(
        (row for row in rows if row["color_mode"] == "random"),
        key=lambda row: row["color_f1_at_1"],
    )
    worst = next(
        (row, metadata[episode_id])
        for row in random_rows
        for episode_id, prediction in row["color_predictions"].items()
        if metadata[episode_id]["color_change_enabled"] and prediction[0] is None
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selections = {}
    for label, (row, item) in (("fixed best", best), ("random missed color", worst)):
        episode_id = item["episode_id"]
        stove_prediction, stove_count = row["stove_predictions"][episode_id]
        color_prediction, color_count = row["color_predictions"][episode_id]
        filename = label.replace(" ", "_") + ".mp4"
        output = args.output_dir / filename
        _render(
            args.artifacts / episode_id / "rollout.mp4",
            output,
            label=label,
            seed=int(row["seed"]),
            stove_gt=item["stove_boundary_sample"],
            stove_prediction=stove_prediction,
            stove_count=stove_count,
            color_gt=item["color_boundary_sample"],
            color_prediction=color_prediction,
            color_count=color_count,
        )
        selections[label] = {
            "episode_id": episode_id,
            "seed": row["seed"],
            "stove_gt": item["stove_boundary_sample"],
            "stove_prediction": stove_prediction,
            "stove_event_count": stove_count,
            "color_gt": item["color_boundary_sample"],
            "color_prediction": color_prediction,
            "color_event_count": color_count,
            "video": str(output),
        }
    (args.output_dir / "selections.json").write_text(
        json.dumps(selections, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(selections, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
