from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

STAGE_NAMES = (
    "place_weight_1",
    "place_weight_2",
    "place_weight_3",
    "place_weight_4",
    "final_pointing",
)
VIDEO_KEY = "observation.images.follower_d455f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--train-episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def select_four_candidates(
    candidates: list[tuple[int, float]], fps: float
) -> list[tuple[int, float]]:
    if len(candidates) < 4:
        return []
    if len(candidates) == 4:
        return candidates
    best = None
    for combination in itertools.combinations(candidates, 4):
        frames = np.asarray([item[0] for item in combination])
        gaps = np.diff(frames) / fps
        if len(gaps) and gaps.min() < 4.0:
            continue
        amplitudes = float(np.mean([item[1] for item in combination]))
        regularity = float(np.std(gaps) / max(np.mean(gaps), 1e-6))
        # Four nearly repeated pick-and-place cycles should have comparable
        # spacing.  This prevents the much stronger final hand reset after
        # pointing from replacing the fourth placement release.
        score = amplitudes - regularity
        if best is None or score > best[0]:
            best = (score, combination)
    return list(best[1]) if best is not None else []


def detect_release_boundaries(gripper: np.ndarray, fps: float) -> tuple[list[int], dict]:
    low = float(np.percentile(gripper, 10))
    high = float(np.percentile(gripper, 90))
    span = high - low
    # The right gripper action becomes more negative when it opens to release a
    # weight.  Use a local before/after drop instead of one global threshold:
    # some releases only open part-way and never cross the episode midpoint.
    sample = max(6, round(0.40 * fps))
    guard = max(2, round(0.07 * fps))
    margin = round(2.0 * fps)
    drop = np.zeros(len(gripper), dtype=np.float32)
    for frame in range(margin, len(gripper) - margin):
        before = np.median(gripper[frame - sample : frame - guard])
        after = np.median(gripper[frame + guard : frame + sample])
        drop[frame] = before - after

    minimum_drop = max(3.0, 0.08 * max(span, 1e-6))
    local_radius = max(3, round(0.35 * fps))
    raw_candidates = []
    for frame in range(margin, len(gripper) - margin):
        if drop[frame] < minimum_drop:
            continue
        neighborhood = drop[frame - local_radius : frame + local_radius + 1]
        if drop[frame] >= float(neighborhood.max()):
            raw_candidates.append((frame, float(drop[frame] / max(span, 1e-6))))

    # Merge samples belonging to the same physical opening motion.
    merged = []
    for candidate in raw_candidates:
        if merged and candidate[0] - merged[-1][0] < round(1.5 * fps):
            if candidate[1] > merged[-1][1]:
                merged[-1] = candidate
        else:
            merged.append(candidate)
    chosen = select_four_candidates(merged, fps)
    return [frame for frame, _ in chosen], {
        "low": low,
        "high": high,
        "span": span,
        "minimum_drop": minimum_drop,
        "raw_candidate_frames": [frame for frame, _ in raw_candidates],
        "candidate_frames": [frame for frame, _ in merged],
        "candidate_scores": [score for _, score in merged],
        "chosen_frames": [frame for frame, _ in chosen],
    }


def load_dataset(dataset: Path) -> tuple[dict[int, dict], dict[int, np.ndarray], float]:
    info = json.loads((dataset / "meta/info.json").read_text())
    fps = float(info["fps"])
    episode_table = pq.read_table(dataset / "meta/episodes/chunk-000/file-000.parquet")
    episode_rows = {int(row["episode_index"]): row for row in episode_table.to_pylist()}
    data_table = pq.read_table(
        dataset / "data/chunk-000/file-000.parquet",
        columns=["action", "episode_index", "frame_index"],
    )
    action_array = data_table.column("action").combine_chunks()
    actions = np.asarray(action_array.values).reshape(-1, 16)
    episode_indices = np.asarray(data_table.column("episode_index"))
    frame_indices = np.asarray(data_table.column("frame_index"))
    grippers = {}
    for episode in sorted(np.unique(episode_indices)):
        selected = episode_indices == episode
        order = np.argsort(frame_indices[selected])
        grippers[int(episode)] = actions[selected][order, 15]
    return episode_rows, grippers, fps


def split_episodes(total: int, train_count: int, seed: int) -> tuple[list[int], list[int]]:
    if not 0 < train_count < total:
        raise ValueError("train episode count must be between zero and total")
    shuffled = np.random.default_rng(seed).permutation(total).tolist()
    return sorted(shuffled[:train_count]), sorted(shuffled[train_count:])


def encode_clip(source: Path, start: float, end: float, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.6f}",
            "-to",
            f"{end:.6f}",
            "-i",
            str(source),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "23",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def main() -> None:
    args = parse_args()
    rows, grippers, fps = load_dataset(args.dataset)
    if args.episodes > len(rows):
        raise ValueError(f"requested {args.episodes} episodes but dataset has {len(rows)}")
    train, validation = split_episodes(args.episodes, args.train_episodes, args.seed)
    split_by_episode = {episode: "train" for episode in train}
    split_by_episode.update({episode: "validation" for episode in validation})
    records = []
    failures = []
    for episode in range(args.episodes):
        row = rows[episode]
        gripper = grippers[episode]
        boundaries, diagnostics = detect_release_boundaries(gripper, fps)
        if len(boundaries) != 4:
            failures.append(
                {
                    "episode": episode,
                    "detected": len(boundaries),
                    "diagnostics": diagnostics,
                }
            )
            continue
        duration = len(gripper) / fps
        times = [0.0, *[frame / fps for frame in boundaries], duration]
        source_file_index = int(row[f"videos/{VIDEO_KEY}/file_index"])
        source_from = float(row[f"videos/{VIDEO_KEY}/from_timestamp"])
        source = (
            args.dataset / "videos" / VIDEO_KEY / "chunk-000" / f"file-{source_file_index:03d}.mp4"
        )
        for stage, name in enumerate(STAGE_NAMES):
            relative_start, relative_end = times[stage], times[stage + 1]
            split = split_by_episode[episode]
            relative_output = Path(split) / f"episode_{episode:03d}" / f"subtask_{stage}_{name}.mp4"
            record = {
                "episode": episode,
                "split": split,
                "stage": stage,
                "stage_name": name,
                "start_frame": round(relative_start * fps),
                "end_frame": round(relative_end * fps),
                "start_s": relative_start,
                "end_s": relative_end,
                "duration_s": relative_end - relative_start,
                "source_video": str(source.relative_to(args.dataset)),
                "source_start_s": source_from + relative_start,
                "source_end_s": source_from + relative_end,
                "output_video": str(relative_output),
            }
            records.append(record)
            if not args.analyze_only:
                encode_clip(
                    source,
                    record["source_start_s"],
                    record["source_end_s"],
                    args.output_dir / relative_output,
                )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "split.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "episodes": args.episodes,
                "train": train,
                "validation": validation,
            },
            indent=2,
        )
        + "\n"
    )
    with (args.output_dir / "manifest.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    with (args.output_dir / "boundaries.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "episode",
                "split",
                "boundary_1_s",
                "boundary_2_s",
                "boundary_3_s",
                "boundary_4_s",
                "duration_s",
            )
        )
        for episode in range(args.episodes):
            episode_records = [record for record in records if record["episode"] == episode]
            if len(episode_records) != 5:
                continue
            writer.writerow(
                (
                    episode,
                    split_by_episode[episode],
                    *[record["end_s"] for record in episode_records[:4]],
                    episode_records[-1]["end_s"],
                )
            )
    report = {
        "episodes_requested": args.episodes,
        "episodes_valid": args.episodes - len(failures),
        "train_episodes": len(train),
        "validation_episodes": len(validation),
        "clips_expected": args.episodes * len(STAGE_NAMES),
        "clips_manifested": len(records),
        "analyze_only": args.analyze_only,
        "failures": failures,
    }
    (args.output_dir / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if failures:
        raise RuntimeError(f"boundary detection failed for {len(failures)} episodes")


if __name__ == "__main__":
    main()
