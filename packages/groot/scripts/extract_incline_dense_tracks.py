from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch

VIDEO_KEY = "observation.images.follower_d455f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--visualize-episodes", default="0")
    return parser.parse_args()


def load_episode_rows(dataset: Path) -> dict[int, dict]:
    table = pq.read_table(dataset / "meta/episodes/chunk-000/file-000.parquet")
    return {int(row["episode_index"]): row for row in table.to_pylist()}


def load_episode_frames(
    dataset: Path, row: dict, fps: float, width: int, height: int
) -> list[np.ndarray]:
    source_index = int(row[f"videos/{VIDEO_KEY}/file_index"])
    start_s = float(row[f"videos/{VIDEO_KEY}/from_timestamp"])
    frame_count = int(row["length"])
    source = dataset / "videos" / VIDEO_KEY / "chunk-000" / f"file-{source_index:03d}.mp4"
    # The LeRobot source is AV1.  FFmpeg's software decoder is reliable on WSL,
    # whereas OpenCV 5 may incorrectly request unavailable AV1 hardware decode.
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
            "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    if process.stdout is None:
        raise RuntimeError("Could not open FFmpeg output pipe")
    frame_bytes = width * height * 3
    frames = []
    try:
        for _ in range(frame_count):
            payload = process.stdout.read(frame_bytes)
            if len(payload) != frame_bytes:
                break
            frames.append(np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3))
    finally:
        process.stdout.close()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"FFmpeg failed while decoding {source}")
    if len(frames) != frame_count:
        raise RuntimeError(f"Expected {frame_count} frames, decoded {len(frames)} from {source}")
    return frames


def track_online(
    model: torch.nn.Module,
    frames: list[np.ndarray],
    grid_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    start = time.perf_counter()
    peak_before = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    is_first_step = True
    pred_tracks = pred_visibility = None
    for index in range(model.step, len(frames) + model.step, model.step):
        window = frames[max(0, index - model.step * 2) : min(index, len(frames))]
        if len(window) < model.step:
            continue
        video_chunk = (
            torch.from_numpy(np.stack(window))
            .to(device=device, dtype=torch.float32)
            .permute(0, 3, 1, 2)[None]
        )
        pred_tracks, pred_visibility = model(
            video_chunk,
            is_first_step=is_first_step,
            grid_size=grid_size,
            grid_query_frame=0,
        )
        is_first_step = False
        del video_chunk
    if pred_tracks is None or pred_visibility is None:
        raise RuntimeError("CoTracker did not return tracks")
    latency = time.perf_counter() - start
    peak_after = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    tracks = pred_tracks[0].detach().cpu().numpy()
    visibility = pred_visibility[0].detach().cpu().numpy().astype(bool)
    return tracks, visibility, latency, max(0, peak_after - peak_before)


def motion_features(tracks: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    deltas = np.diff(tracks, axis=0, prepend=tracks[:1])
    valid = visibility & np.roll(visibility, 1, axis=0)
    valid[0] = False
    output = np.zeros((len(tracks), 8), dtype=np.float32)
    for frame in range(len(tracks)):
        selected = deltas[frame][valid[frame]]
        if not len(selected):
            continue
        magnitude = np.linalg.norm(selected, axis=1)
        output[frame] = (
            np.median(selected[:, 0]),
            np.median(selected[:, 1]),
            np.median(magnitude),
            np.quantile(magnitude, 0.75),
            np.quantile(magnitude, 0.90),
            np.mean(magnitude),
            np.std(magnitude),
            valid[frame].mean(),
        )
    return output


def render_tracks(
    frames: list[np.ndarray],
    tracks: np.ndarray,
    visibility: np.ndarray,
    fps: float,
    output: Path,
) -> None:
    count = min(len(frames), len(tracks))
    height, width = frames[0].shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tracking.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    colors = np.random.default_rng(42).integers(50, 256, size=(tracks.shape[1], 3))
    for frame_index in range(count):
        canvas = cv2.cvtColor(frames[frame_index], cv2.COLOR_RGB2BGR)
        for point_index in np.flatnonzero(visibility[frame_index]):
            start = max(0, frame_index - round(fps))
            valid = visibility[start : frame_index + 1, point_index]
            trail = tracks[start : frame_index + 1, point_index][valid]
            if len(trail) < 2:
                continue
            color = tuple(int(value) for value in colors[point_index])
            cv2.polylines(
                canvas,
                [np.round(trail).astype(np.int32).reshape(-1, 1, 2)],
                False,
                color,
                1,
                cv2.LINE_AA,
            )
            cv2.circle(canvas, tuple(np.round(trail[-1]).astype(int)), 2, color, -1)
        cv2.putText(
            canvas,
            f"CoTracker3 quasi-dense | visible {int(visibility[frame_index].sum())}",
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(canvas)
    writer.release()
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(temporary),
            "-c:v",
            "libx264",
            "-crf",
            "25",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    temporary.unlink()


def main() -> None:
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        raise RuntimeError("CUDA or Apple MPS is required for the CoTracker run")
    info = json.loads((args.dataset / "meta/info.json").read_text())
    fps = float(info["fps"])
    height, width, _ = info["features"][VIDEO_KEY]["shape"]
    split = json.loads(args.split.read_text())
    split_by_episode = {episode: "train" for episode in split["train"]}
    split_by_episode.update({episode: "validation" for episode in split["validation"]})
    rows = load_episode_rows(args.dataset)
    visualized = {int(value) for value in args.visualize_episodes.split(",") if value.strip()}
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online", trust_repo=True).to(
        device
    )
    model.eval()
    summaries = []
    for episode in range(args.episodes):
        episode_dir = args.output_dir / split_by_episode[episode] / f"episode_{episode:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        summary_path = episode_dir / "summary.json"
        tracks_path = episode_dir / "dense_tracks.npz"
        if summary_path.exists() and tracks_path.exists():
            summary = json.loads(summary_path.read_text())
            summaries.append(summary)
            print(json.dumps({**summary, "status": "skipped_existing"}), flush=True)
            continue
        frames = load_episode_frames(args.dataset, rows[episode], fps, width, height)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        tracks, visibility, elapsed, peak_delta = track_online(
            model, frames, args.grid_size, device
        )
        features = motion_features(tracks, visibility)
        np.savez_compressed(
            tracks_path,
            tracks=tracks.astype(np.float16),
            visibility=visibility,
            motion_features=features.astype(np.float16),
            fps=np.asarray(fps, dtype=np.float32),
            grid_size=np.asarray(args.grid_size, dtype=np.int16),
        )
        if episode in visualized:
            render_tracks(
                frames,
                tracks,
                visibility,
                fps,
                episode_dir / "dense_tracks_preview.mp4",
            )
        summary = {
            "episode": episode,
            "split": split_by_episode[episode],
            "frames": len(frames),
            "track_frames": len(tracks),
            "points": int(tracks.shape[1]),
            "fps": fps,
            "elapsed_s": elapsed,
            "ms_per_frame": elapsed * 1000 / max(len(tracks), 1),
            "peak_vram_delta_bytes": peak_delta,
            "mean_visibility": float(visibility.mean()),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
        del frames, tracks, visibility, features
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "method": "CoTracker3 online quasi-dense regular grid",
                "grid_size": args.grid_size,
                "episodes": summaries,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
