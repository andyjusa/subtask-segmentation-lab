from __future__ import annotations

import argparse
import json
import subprocess
from collections import deque
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=180)
    parser.add_argument("--refresh-frames", type=int, default=15)
    parser.add_argument("--trail-frames", type=int, default=45)
    parser.add_argument("--motion-threshold", type=float, default=1.5)
    return parser.parse_args()


def detect_points(gray: np.ndarray, mask: np.ndarray, limit: int) -> np.ndarray:
    points = cv2.goodFeaturesToTrack(
        gray,
        mask=mask,
        maxCorners=limit,
        qualityLevel=0.015,
        minDistance=8,
        blockSize=7,
    )
    if points is None:
        return np.empty((0, 2), dtype=np.float32)
    return points.reshape(-1, 2).astype(np.float32)


def main() -> None:
    args = parse_args()
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {args.input}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError("Could not read the first frame")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tracking.mp4")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {temporary}")

    previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    points = detect_points(
        previous_gray, np.full_like(previous_gray, 255), args.max_points
    )
    tracks = [deque([tuple(point)], maxlen=args.trail_frames) for point in points]
    ages = np.zeros(len(points), dtype=np.int32)
    frame_index = 0
    stats = []
    lk_params = {
        "winSize": (31, 31),
        "maxLevel": 4,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    }

    try:
        while True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frame_index > 0 and len(points):
                old = points.reshape(-1, 1, 2)
                new, status, _ = cv2.calcOpticalFlowPyrLK(
                    previous_gray, gray, old, None, **lk_params
                )
                backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                    gray, previous_gray, new, None, **lk_params
                )
                candidate = new.reshape(-1, 2)
                forward_backward = np.linalg.norm(old - backward, axis=2).reshape(-1)
                valid = (
                    status.reshape(-1).astype(bool)
                    & backward_status.reshape(-1).astype(bool)
                    & (forward_backward < 1.5)
                    & (candidate[:, 0] >= 0)
                    & (candidate[:, 0] < width)
                    & (candidate[:, 1] >= 0)
                    & (candidate[:, 1] < height)
                )
                points = candidate[valid]
                tracks = [track for track, keep in zip(tracks, valid, strict=True) if keep]
                ages = ages[valid] + 1
                for track, point in zip(tracks, points, strict=True):
                    track.append(tuple(point))

            if frame_index % args.refresh_frames == 0 and len(points) < args.max_points:
                feature_mask = np.full_like(gray, 255)
                for point in points:
                    cv2.circle(
                        feature_mask, tuple(np.round(point).astype(int)), 10, 0, -1
                    )
                fresh = detect_points(gray, feature_mask, args.max_points - len(points))
                if len(fresh):
                    points = np.concatenate([points, fresh])
                    tracks.extend(
                        deque([tuple(point)], maxlen=args.trail_frames) for point in fresh
                    )
                    ages = np.concatenate([ages, np.zeros(len(fresh), dtype=np.int32)])

            canvas = frame.copy()
            moving_count = 0
            for track, point, age in zip(tracks, points, ages, strict=True):
                if len(track) < 3:
                    continue
                displacement = np.linalg.norm(
                    np.asarray(track[-1]) - np.asarray(track[0])
                )
                if displacement < args.motion_threshold:
                    continue
                moving_count += 1
                intensity = min(1.0, displacement / 25.0)
                color = (
                    int(40 + 30 * (1 - intensity)),
                    int(170 + 85 * intensity),
                    int(255 - 120 * intensity),
                )
                trail = np.asarray(track, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [trail], False, color, 1, cv2.LINE_AA)
                radius = 3 if age > args.refresh_frames else 2
                cv2.circle(canvas, tuple(np.round(point).astype(int)), radius, color, -1)

            cv2.putText(
                canvas,
                f"Point tracking | moving {moving_count} / active {len(points)}",
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(canvas)
            if frame_index % max(1, round(fps)) == 0:
                stats.append(
                    {
                        "frame": frame_index,
                        "timestamp_s": frame_index / fps,
                        "active_points": len(points),
                        "moving_points": moving_count,
                    }
                )

            previous_gray = gray
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
    finally:
        capture.release()
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
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(args.output),
        ],
        check=True,
    )
    temporary.unlink()
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "method": "Shi-Tomasi points + pyramidal Lucas-Kanade optical flow",
        "source_fps": fps,
        "output_fps": fps,
        "source_frames": source_frames,
        "output_frames": frame_index + 1,
        "max_points": args.max_points,
        "refresh_frames": args.refresh_frames,
        "trail_frames": args.trail_frames,
        "motion_threshold": args.motion_threshold,
        "stats": stats,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "stats"}))


if __name__ == "__main__":
    main()
