from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--tracks-filename", default="tapnextpp_tracks.npz")
    parser.add_argument("--points", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = json.loads(args.split.read_text())
    expected_split = {int(episode): "train" for episode in split["train"]}
    expected_split.update({int(episode): "validation" for episode in split["validation"]})
    expected_episodes = set(range(args.episodes))
    if set(expected_split) != expected_episodes:
        raise AssertionError("Split does not contain exactly the expected episodes")

    summaries: list[dict] = []
    total_frames = 0
    for episode in range(args.episodes):
        episode_dir = args.output_dir / expected_split[episode] / f"episode_{episode:03d}"
        summary = json.loads((episode_dir / "summary.json").read_text())
        if summary["episode"] != episode or summary["split"] != expected_split[episode]:
            raise AssertionError(f"Episode metadata mismatch: {episode}")
        with np.load(episode_dir / args.tracks_filename) as arrays:
            frames = int(summary["frames"])
            expected_shapes = {
                "tracks": (frames, args.points, 2),
                "visibility": (frames, args.points),
                "motion_features": (frames, 8),
                "query_points": (args.points, 2),
                "fps": (),
                "grid_size": (),
            }
            allowed_arrays = set(expected_shapes) | {"confidence"}
            if not set(expected_shapes).issubset(arrays.files) or not set(arrays.files).issubset(
                allowed_arrays
            ):
                raise AssertionError(f"Unexpected arrays in episode {episode}: {arrays.files}")
            for name, shape in expected_shapes.items():
                if arrays[name].shape != shape:
                    raise AssertionError(
                        f"Episode {episode} {name}: {arrays[name].shape} != {shape}"
                    )
                if not np.isfinite(arrays[name]).all():
                    raise AssertionError(f"Episode {episode} {name} contains non-finite values")
            if "confidence" in arrays:
                if arrays["confidence"].shape != (frames, args.points):
                    raise AssertionError(f"Episode {episode} confidence shape mismatch")
                if not np.isfinite(arrays["confidence"]).all():
                    raise AssertionError(f"Episode {episode} confidence contains non-finite values")
            if int(arrays["grid_size"]) != 20 or float(arrays["fps"]) != 30.0:
                raise AssertionError(f"Episode {episode} scalar metadata mismatch")
        summaries.append(summary)
        total_frames += int(summary["frames"])

    root_summary = json.loads((args.output_dir / "summary.json").read_text())
    if root_summary["completed_episodes"] != args.episodes:
        raise AssertionError("Root summary is incomplete")
    if len(root_summary["episodes"]) != args.episodes:
        raise AssertionError("Root summary episode list is incomplete")

    report = {
        "episodes": len(summaries),
        "train": sum(summary["split"] == "train" for summary in summaries),
        "validation": sum(summary["split"] == "validation" for summary in summaries),
        "total_frames": total_frames,
        "total_video_seconds": total_frames / 30.0,
        "mean_ms_per_frame": float(np.mean([item["ms_per_frame"] for item in summaries])),
        "min_ms_per_frame": float(np.min([item["ms_per_frame"] for item in summaries])),
        "max_ms_per_frame": float(np.max([item["ms_per_frame"] for item in summaries])),
        "total_inference_seconds": float(np.sum([item["elapsed_s"] for item in summaries])),
        "max_vram_allocated_bytes": max(item["peak_vram_allocated_bytes"] for item in summaries),
        "max_vram_reserved_bytes": max(item["peak_vram_reserved_bytes"] for item in summaries),
        "mean_visibility": float(np.mean([item["mean_visibility"] for item in summaries])),
        "status": "valid",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
