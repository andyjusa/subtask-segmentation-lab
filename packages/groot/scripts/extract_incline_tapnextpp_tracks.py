from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

VIDEO_KEY = "observation.images.follower_d455f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--vendor-dir", type=Path, default=Path("vendor/tapnet"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--input-resolution", type=int, default=256)
    return parser.parse_args()


def load_episode_rows(dataset: Path) -> dict[int, dict]:
    table = pq.read_table(dataset / "meta/episodes/chunk-000/file-000.parquet")
    return {int(row["episode_index"]): row for row in table.to_pylist()}


def episode_frames(
    dataset: Path,
    row: dict,
    width: int,
    height: int,
) -> Iterator[np.ndarray]:
    source_index = int(row[f"videos/{VIDEO_KEY}/file_index"])
    start_s = float(row[f"videos/{VIDEO_KEY}/from_timestamp"])
    frame_count = int(row["length"])
    source = dataset / "videos" / VIDEO_KEY / "chunk-000" / f"file-{source_index:03d}.mp4"
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_s:.9f}",
            "-i",
            str(source),
            "-frames:v",
            str(frame_count),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    if process.stdout is None:
        raise RuntimeError("Could not open FFmpeg output pipe")
    frame_bytes = width * height * 3
    decoded = 0
    try:
        for _ in range(frame_count):
            payload = process.stdout.read(frame_bytes)
            if len(payload) != frame_bytes:
                break
            decoded += 1
            yield np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
    finally:
        process.stdout.close()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"FFmpeg failed while decoding {source}")
    if decoded != frame_count:
        raise RuntimeError(f"Expected {frame_count} frames, decoded {decoded} from {source}")


def make_grid(height: int, width: int, grid_size: int) -> np.ndarray:
    margin_x = width / (grid_size * 2)
    margin_y = height / (grid_size * 2)
    xs = np.linspace(margin_x, width - margin_x, grid_size, dtype=np.float32)
    ys = np.linspace(margin_y, height - margin_y, grid_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)


def motion_features(tracks: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    deltas = np.diff(tracks, axis=0, prepend=tracks[:1])
    valid = visibility & np.roll(visibility, 1, axis=0)
    valid[0] = False
    output = np.zeros((len(tracks), 8), dtype=np.float32)
    for frame_index in range(len(tracks)):
        selected = deltas[frame_index][valid[frame_index]]
        if not len(selected):
            continue
        magnitude = np.linalg.norm(selected, axis=1)
        output[frame_index] = (
            np.median(selected[:, 0]),
            np.median(selected[:, 1]),
            np.median(magnitude),
            np.quantile(magnitude, 0.75),
            np.quantile(magnitude, 0.90),
            np.mean(magnitude),
            np.std(magnitude),
            valid[frame_index].mean(),
        )
    return output


def write_root_summary(output_dir: Path, summaries: list[dict], grid_size: int) -> None:
    payload = {
        "method": "TAPNext++ online regular grid",
        "grid_size": grid_size,
        "completed_episodes": len(summaries),
        "episodes": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this batch run")
    device = torch.device("cuda")
    sys.path.insert(0, str(args.vendor_dir.resolve()))
    from tapnet.tapnextpp.votsp2026.model import TAPNextPP

    info = json.loads((args.dataset / "meta/info.json").read_text())
    fps = float(info["fps"])
    height, width, _ = info["features"][VIDEO_KEY]["shape"]
    split = json.loads(args.split.read_text())
    split_by_episode = {int(episode): "train" for episode in split["train"]}
    split_by_episode.update({int(episode): "validation" for episode in split["validation"]})
    rows = load_episode_rows(args.dataset)
    queries = make_grid(height, width, args.grid_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = TAPNextPP.from_checkpoint(
        args.checkpoint,
        device=device,
        half_precision=True,
        input_resolution=args.input_resolution,
    )
    model_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    summaries: list[dict] = []
    for episode in range(args.episodes):
        episode_dir = args.output_dir / split_by_episode[episode] / f"episode_{episode:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        summary_path = episode_dir / "summary.json"
        tracks_path = episode_dir / "tapnextpp_tracks.npz"
        if summary_path.exists() and tracks_path.exists():
            summary = json.loads(summary_path.read_text())
            summaries.append(summary)
            print(json.dumps({**summary, "status": "skipped_existing"}), flush=True)
            continue

        frame_count = int(rows[episode]["length"])
        tracks = np.empty((frame_count, len(queries), 2), dtype=np.float16)
        visibility = np.empty((frame_count, len(queries)), dtype=bool)
        state = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for frame_index, frame in enumerate(
            episode_frames(args.dataset, rows[episode], width, height)
        ):
            positions, visible, state = model.track_frame(
                frame,
                query_points_xy=queries if frame_index == 0 else None,
                state=state,
                autocast=True,
            )
            tracks[frame_index] = positions
            visibility[frame_index] = visible
            if (frame_index + 1) % 300 == 0:
                print(
                    json.dumps(
                        {
                            "episode": episode,
                            "progress": frame_index + 1,
                            "frames": frame_count,
                        }
                    ),
                    flush=True,
                )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        features = motion_features(tracks.astype(np.float32), visibility)
        np.savez_compressed(
            tracks_path,
            tracks=tracks,
            visibility=visibility,
            motion_features=features.astype(np.float16),
            query_points=queries.astype(np.float16),
            fps=np.asarray(fps, dtype=np.float32),
            grid_size=np.asarray(args.grid_size, dtype=np.int16),
        )
        summary = {
            "episode": episode,
            "split": split_by_episode[episode],
            "frames": frame_count,
            "points": len(queries),
            "fps": fps,
            "precision": "fp16",
            "model_input_resolution": args.input_resolution,
            "model_parameters": model_parameter_count,
            "elapsed_s": elapsed,
            "ms_per_frame": elapsed * 1000 / frame_count,
            "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "mean_visibility": float(visibility.mean()),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        summaries.append(summary)
        write_root_summary(args.output_dir, summaries, args.grid_size)
        print(json.dumps(summary), flush=True)
        del tracks, visibility, features, state
        torch.cuda.empty_cache()

    write_root_summary(args.output_dir, summaries, args.grid_size)


if __name__ == "__main__":
    main()
