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

WEIGHTS_URL = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--vendor-dir", type=Path, default=Path("vendor/alltracker"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--window-len", type=int, default=16)
    parser.add_argument("--inference-iters", type=int, default=4)
    parser.add_argument("--rate", type=int, default=8)
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    return parser.parse_args()


def load_video(path: Path, max_frames: int, image_size: int) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    try:
        while len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    source_height, source_width = frames[0].shape[:2]
    scale = min(image_size / source_height, image_size / source_width)
    height = max(8, round(source_height * scale) // 8 * 8)
    width = max(8, round(source_width * scale) // 8 * 8)
    resized = np.stack(
        [cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR) for frame in frames]
    )
    return resized, fps


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def color_grid(height: int, width: int, rate: int) -> np.ndarray:
    ys, xs = np.meshgrid(np.arange(0, height, rate), np.arange(0, width, rate), indexing="ij")
    hue = (xs.astype(np.float32) / max(width - 1, 1) * 179).astype(np.uint8)
    saturation = np.full_like(hue, 230)
    value = (80 + ys.astype(np.float32) / max(height - 1, 1) * 175).astype(np.uint8)
    hsv = np.stack([hue, saturation, value], axis=-1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)


def render_preview(
    frames: np.ndarray,
    tracks: np.ndarray,
    visibility: np.ndarray,
    fps: float,
    rate: int,
    output: Path,
) -> None:
    height, width = frames.shape[1:3]
    colors = color_grid(height, width, rate)
    temporary = output.with_suffix(".tracking.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    trail_frames = max(1, round(fps * 0.7))
    for frame_index, rgb in enumerate(frames[: len(tracks)]):
        canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
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
            cv2.circle(canvas, tuple(np.round(trail[-1]).astype(int)), 1, color, -1)
        cv2.putText(
            canvas,
            f"AllTracker dense | visible {len(visible_indices)} / {tracks.shape[1]}",
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


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.vendor_dir.resolve()))
    from nets.alltracker import Net

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames, fps = load_video(args.input, args.max_frames, args.image_size)
    device = choose_device()
    model = Net(args.window_len)
    state = torch.hub.load_state_dict_from_url(WEIGHTS_URL, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float()
    start = time.perf_counter()
    with torch.inference_mode():
        flows, visconf, _, _ = model.forward_sliding(
            video,
            iters=args.inference_iters,
            sw=None,
            is_training=False,
        )
    elapsed = time.perf_counter() - start
    height, width = frames.shape[1:3]
    ys, xs = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    grid = torch.stack([xs, ys], dim=0)[None, None].float()
    trajectories = flows + grid
    tracks = (
        trajectories[0, :, :, :: args.rate, :: args.rate]
        .reshape(len(trajectories[0]), 2, -1)
        .permute(0, 2, 1)
        .numpy()
    )
    confidence = visconf[0, :, 1, :: args.rate, :: args.rate].reshape(len(visconf[0]), -1).numpy()
    visibility = confidence > args.confidence_threshold
    np.savez_compressed(
        args.output_dir / "alltracker_dense_tracks.npz",
        tracks=tracks.astype(np.float16),
        confidence=confidence.astype(np.float16),
        visibility=visibility,
        fps=np.asarray(fps, dtype=np.float32),
        rate=np.asarray(args.rate, dtype=np.int16),
    )
    output_video = args.output_dir / "alltracker_dense_tracks_preview.mp4"
    render_preview(frames, tracks, visibility, fps, args.rate, output_video)
    report = {
        "model": "AllTracker",
        "device": str(device),
        "input": str(args.input),
        "frames": len(frames),
        "fps": fps,
        "resolution": [width, height],
        "dense_output_shape": list(flows.shape),
        "saved_points": tracks.shape[1],
        "subsample_rate": args.rate,
        "mean_visibility": float(visibility.mean()),
        "inference_s": elapsed,
        "ms_per_frame": elapsed * 1000 / len(frames),
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "output_video": str(output_video),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
