from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

CHECKPOINT_URL = "https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--vendor-dir", type=Path, default=Path("vendor/tapnet"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--input-resolution", type=int, default=256)
    parser.add_argument("--precision", choices=("auto", "fp32"), default="auto")
    return parser.parse_args()


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_video(path: Path, max_frames: int) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    try:
        while len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames, fps


def make_grid(height: int, width: int, grid_size: int) -> np.ndarray:
    margin_x = width / (grid_size * 2)
    margin_y = height / (grid_size * 2)
    xs = np.linspace(margin_x, width - margin_x, grid_size, dtype=np.float32)
    ys = np.linspace(margin_y, height - margin_y, grid_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)


def point_colors(points: np.ndarray, height: int, width: int) -> np.ndarray:
    hue = (points[:, 0] / max(width - 1, 1) * 179).astype(np.uint8)
    saturation = np.full_like(hue, 230)
    value = (80 + points[:, 1] / max(height - 1, 1) * 175).astype(np.uint8)
    hsv = np.stack([hue, saturation, value], axis=-1).reshape(1, -1, 3)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)


def render_preview(
    frames: list[np.ndarray],
    tracks: np.ndarray,
    visibility: np.ndarray,
    fps: float,
    output: Path,
) -> None:
    height, width = frames[0].shape[:2]
    colors = point_colors(tracks[0], height, width)
    temporary = output.with_suffix(".tracking.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    trail_frames = max(1, round(fps * 0.7))
    for frame_index, frame in enumerate(frames[: len(tracks)]):
        canvas = frame.copy()
        visible_indices = np.flatnonzero(visibility[frame_index])
        for point_index in visible_indices:
            start = max(0, frame_index - trail_frames)
            valid = visibility[start : frame_index + 1, point_index]
            trail = tracks[start : frame_index + 1, point_index][valid]
            if not len(trail):
                continue
            color = tuple(int(value) for value in colors[point_index])
            if len(trail) > 1:
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
            f"TAPNext++ online | visible {len(visible_indices)} / {tracks.shape[1]}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
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


def download_checkpoint(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.hub.download_url_to_file(CHECKPOINT_URL, str(temporary), progress=True)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    download_checkpoint(args.checkpoint)
    sys.path.insert(0, str(args.vendor_dir.resolve()))
    from tapnet.tapnextpp.votsp2026.model import TAPNextPP

    frames, fps = load_video(args.input, args.max_frames)
    height, width = frames[0].shape[:2]
    queries = make_grid(height, width, args.grid_size)
    device = choose_device()

    use_half_precision = device.type == "cuda" and args.precision == "auto"
    model = TAPNextPP.from_checkpoint(
        args.checkpoint,
        device=device,
        half_precision=use_half_precision,
        input_resolution=args.input_resolution,
    )
    model_parameter_count = sum(parameter.numel() for parameter in model.parameters())

    positions_by_frame: list[np.ndarray] = []
    visibility_by_frame: list[np.ndarray] = []
    state = None
    if device.type == "mps":
        torch.mps.synchronize()
    start = time.perf_counter()
    for frame_index, frame in enumerate(frames):
        positions, visible, state = model.track_frame(
            frame,
            query_points_xy=queries if frame_index == 0 else None,
            state=state,
            autocast=device.type == "cuda",
        )
        positions_by_frame.append(positions)
        visibility_by_frame.append(visible)
        if (frame_index + 1) % 50 == 0:
            print(f"processed {frame_index + 1}/{len(frames)} frames", flush=True)
    if device.type == "mps":
        torch.mps.synchronize()
    inference_seconds = time.perf_counter() - start

    tracks = np.stack(positions_by_frame).astype(np.float16)
    visibility = np.stack(visibility_by_frame)
    result_path = args.output_dir / "tapnextpp_tracks.npz"
    np.savez_compressed(
        result_path,
        tracks=tracks,
        visibility=visibility,
        query_points=queries.astype(np.float16),
        fps=np.float32(fps),
    )
    preview_path = args.output_dir / "tapnextpp_tracks_preview.mp4"
    render_preview(frames, tracks.astype(np.float32), visibility, fps, preview_path)

    summary = {
        "model": "TAPNext++",
        "device": str(device),
        "precision": "fp16" if use_half_precision else "fp32",
        "input": str(args.input),
        "frames": len(frames),
        "fps": fps,
        "source_resolution": [width, height],
        "model_input_resolution": args.input_resolution,
        "tracked_points": len(queries),
        "model_parameters": model_parameter_count,
        "tracks_shape": list(tracks.shape),
        "mean_visibility": float(visibility.mean()),
        "inference_s": inference_seconds,
        "ms_per_frame": inference_seconds / len(frames) * 1000,
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "output_video": str(preview_path),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
