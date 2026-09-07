from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from groot_subtask_phase_probe.hydra_action_flow import (
    HydraFlowConfig,
    boundary_targets,
    build_causal_flow_feature,
    point_memberships,
    select_grouped_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--robot-points", type=int, default=24)
    parser.add_argument("--object-points", type=int, default=24)
    parser.add_argument("--background-points", type=int, default=16)
    return parser.parse_args()


def read_boundaries(path: Path, fps: float) -> dict[int, np.ndarray]:
    output = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            output[int(row["episode"])] = np.rint(
                [float(row[f"boundary_{index}_s"]) * fps for index in range(1, 5)]
            ).astype(np.int32)
    return output


def main() -> None:
    args = parse_args()
    stride = round(args.source_fps / args.sample_fps)
    if not np.isclose(args.source_fps / stride, args.sample_fps):
        raise ValueError("sample_fps must evenly divide source_fps")
    config = HydraFlowConfig(
        window_samples=round(args.window_seconds * args.sample_fps),
        robot_points=args.robot_points,
        object_points=args.object_points,
        background_points=args.background_points,
    )
    split = json.loads(args.split.read_text())
    split_by_episode = {int(episode): "train" for episode in split["train"]}
    split_by_episode.update({int(episode): "validation" for episode in split["validation"]})
    boundaries = read_boundaries(args.boundaries, args.source_fps)
    summaries = []
    for episode in range(args.episodes):
        subset = split_by_episode[episode]
        output_dir = args.output_dir / subset / f"episode_{episode:03d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "hydra_action_flow.npz"
        if output.exists():
            print(json.dumps({"episode": episode, "status": "skipped_existing"}), flush=True)
            continue
        track_path = args.tracks_dir / subset / f"episode_{episode:03d}" / "alltracker_tracks.npz"
        with np.load(track_path) as arrays:
            sample_frames = np.arange(0, len(arrays["tracks"]), stride, dtype=np.int32)
            tracks = arrays["tracks"][sample_frames].astype(np.float32)
            visibility = arrays["visibility"][sample_frames]
            confidence = arrays["confidence"][sample_frames].astype(np.float32)

        mask_paths = sorted(
            (args.masks_dir / subset / f"episode_{episode:03d}").glob("anchor_*.npz")
        )
        if not mask_paths:
            raise FileNotFoundError(f"No SAM3 anchors for episode {episode}")
        anchors = []
        for path in mask_paths:
            with np.load(path) as masks:
                anchor_frame = int(masks["anchor_frame"])
                sample_index = int(np.searchsorted(sample_frames, anchor_frame, side="left"))
                sample_index = min(sample_index, len(sample_frames) - 1)
                memberships = point_memberships(
                    tracks[sample_index], masks["robot_mask"], masks["object_mask"]
                )
            selected, groups, valid = select_grouped_points(
                tracks[sample_index], memberships, config
            )
            anchors.append((sample_index, selected, groups, valid))

        features = []
        anchor_indices = []
        anchor_cursor = 0
        for current_index in range(len(sample_frames)):
            while (
                anchor_cursor + 1 < len(anchors) and anchors[anchor_cursor + 1][0] <= current_index
            ):
                anchor_cursor += 1
            anchor_index, selected, groups, valid = anchors[anchor_cursor]
            features.append(
                build_causal_flow_feature(
                    tracks,
                    visibility,
                    confidence,
                    anchor_index,
                    current_index,
                    selected,
                    groups,
                    valid,
                    config,
                ).reshape(-1)
            )
            anchor_indices.append(anchor_index)
        labels = np.searchsorted(boundaries[episode], sample_frames, side="right").astype(np.int8)
        boundary = boundary_targets(sample_frames, boundaries[episode])
        np.savez_compressed(
            output,
            features=np.asarray(features, dtype=np.float16),
            labels=labels,
            boundary=boundary,
            sample_frames=sample_frames,
            anchor_sample_indices=np.asarray(anchor_indices, dtype=np.int32),
            reference_boundaries=boundaries[episode],
            feature_shape=np.asarray(
                [config.window_samples, config.points, config.feature_channels], dtype=np.int16
            ),
        )
        summary = {
            "episode": episode,
            "samples": len(features),
            "feature_size": config.flattened_size,
            "anchors": len(anchors),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "method": "Hydra-style causal action-flow input for sub-task classification",
                "sample_fps": args.sample_fps,
                "window_seconds": args.window_seconds,
                "feature_shape": [
                    config.window_samples,
                    config.points,
                    config.feature_channels,
                ],
                "episodes": summaries,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
