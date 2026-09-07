from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

VIDEO_KEY = "observation.images.follower_d455f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--vendor-dir", type=Path, default=Path("vendor/alltracker"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--window-len", type=int, default=16)
    parser.add_argument("--inference-iters", type=int, default=4)
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    return parser.parse_args()


def load_episode_rows(dataset: Path) -> dict[int, dict]:
    table = pq.read_table(dataset / "meta/episodes/chunk-000/file-000.parquet")
    return {int(row["episode_index"]): row for row in table.to_pylist()}


def model_resolution(source_width: int, source_height: int, image_size: int) -> tuple[int, int]:
    scale = min(image_size / source_width, image_size / source_height)
    width = max(8, round(source_width * scale) // 8 * 8)
    height = max(8, round(source_height * scale) // 8 * 8)
    return width, height


def decode_episode(
    dataset: Path,
    row: dict,
    width: int,
    height: int,
) -> np.ndarray:
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
            "-vf",
            f"scale={width}:{height}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    if process.stdout is None:
        raise RuntimeError("Could not open FFmpeg output pipe")
    expected_bytes = frame_count * width * height * 3
    payload = process.stdout.read(expected_bytes)
    process.stdout.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"FFmpeg failed while decoding {source}")
    if len(payload) != expected_bytes:
        decoded = len(payload) // (width * height * 3)
        raise RuntimeError(f"Expected {frame_count} frames, decoded {decoded} from {source}")
    return np.frombuffer(payload, dtype=np.uint8).reshape(frame_count, height, width, 3)


def grid_indices(height: int, width: int, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.rint(np.linspace(width / (grid_size * 2), width - width / (grid_size * 2), grid_size))
    ys = np.rint(
        np.linspace(height / (grid_size * 2), height - height / (grid_size * 2), grid_size)
    )
    grid_x, grid_y = np.meshgrid(xs, ys)
    return grid_y.astype(np.int64).ravel(), grid_x.astype(np.int64).ravel()


def select_tracks(
    flows: torch.Tensor,
    visconf: torch.Tensor,
    grid_y: np.ndarray,
    grid_x: np.ndarray,
    source_width: int,
    source_height: int,
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model_height, model_width = flows.shape[-2:]
    selected_flow = flows[0, :, :, grid_y, grid_x].permute(0, 2, 1).numpy()
    base = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
    tracks = selected_flow + base[None]
    scale = np.asarray([source_width / model_width, source_height / model_height], dtype=np.float32)
    tracks *= scale
    query_points = base * scale
    confidence = visconf[0, :, 1, grid_y, grid_x].numpy()
    visibility = confidence > confidence_threshold
    return tracks, confidence, visibility, query_points


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


def write_root_summary(output_dir: Path, summaries: list[dict], args: argparse.Namespace) -> None:
    payload = {
        "method": "AllTracker dense flow sampled on a regular grid",
        "grid_size": args.grid_size,
        "model_input_size": args.image_size,
        "window_len": args.window_len,
        "inference_iters": args.inference_iters,
        "confidence_threshold": args.confidence_threshold,
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
    sys.path.insert(0, str(args.vendor_dir.resolve()))
    from nets.alltracker import Net

    info = json.loads((args.dataset / "meta/info.json").read_text())
    fps = float(info["fps"])
    source_height, source_width, _ = info["features"][VIDEO_KEY]["shape"]
    width, height = model_resolution(source_width, source_height, args.image_size)
    split = json.loads(args.split.read_text())
    split_by_episode = {int(episode): "train" for episode in split["train"]}
    split_by_episode.update({int(episode): "validation" for episode in split["validation"]})
    rows = load_episode_rows(args.dataset)
    grid_y, grid_x = grid_indices(height, width, args.grid_size)

    device = torch.device("cuda")
    model = Net(args.window_len)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model_parameter_count = sum(parameter.numel() for parameter in model.parameters())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    for episode in range(args.episodes):
        episode_dir = args.output_dir / split_by_episode[episode] / f"episode_{episode:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        tracks_path = episode_dir / "alltracker_tracks.npz"
        summary_path = episode_dir / "summary.json"
        if tracks_path.exists() and summary_path.exists():
            summary = json.loads(summary_path.read_text())
            summaries.append(summary)
            print(json.dumps({**summary, "status": "skipped_existing"}), flush=True)
            continue

        frames = decode_episode(args.dataset, rows[episode], width, height)
        video = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2)[None].float()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            flows, visconf, _, _ = model.forward_sliding(
                video,
                iters=args.inference_iters,
                sw=None,
                is_training=False,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        tracks, confidence, visibility, query_points = select_tracks(
            flows,
            visconf,
            grid_y,
            grid_x,
            source_width,
            source_height,
            args.confidence_threshold,
        )
        features = motion_features(tracks, visibility)
        np.savez_compressed(
            tracks_path,
            tracks=tracks.astype(np.float16),
            confidence=confidence.astype(np.float16),
            visibility=visibility,
            motion_features=features.astype(np.float16),
            query_points=query_points.astype(np.float16),
            fps=np.asarray(fps, dtype=np.float32),
            grid_size=np.asarray(args.grid_size, dtype=np.int16),
        )
        summary = {
            "episode": episode,
            "split": split_by_episode[episode],
            "frames": len(frames),
            "points": len(query_points),
            "fps": fps,
            "precision": "fp32 inference / fp16 storage",
            "model_input_resolution": [width, height],
            "model_parameters": model_parameter_count,
            "elapsed_s": elapsed,
            "ms_per_frame": elapsed * 1000 / len(frames),
            "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "mean_visibility": float(visibility.mean()),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        summaries.append(summary)
        write_root_summary(args.output_dir, summaries, args)
        print(json.dumps(summary), flush=True)
        del frames, video, flows, visconf, tracks, confidence, visibility, features


if __name__ == "__main__":
    main()
