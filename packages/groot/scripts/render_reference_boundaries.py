from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

STAGE_NAMES = (
    "Place weight 1",
    "Place weight 2",
    "Place weight 3",
    "Place weight 4",
    "Final pointing",
)
COLORS = (
    (70, 190, 255),
    (70, 230, 100),
    (230, 190, 60),
    (230, 90, 160),
    (180, 100, 240),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boundaries", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    boundary_seconds = np.asarray(
        [float(value) for value in args.boundaries.split(",")], dtype=np.float32
    )
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {args.input}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    boundary_frames = np.round(boundary_seconds * fps).astype(np.int32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tracking.mp4")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            stage = int(np.searchsorted(boundary_frames, frame_index, side="right"))
            color = COLORS[stage]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 44), color, -1)
            frame = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)
            cv2.putText(
                frame,
                f"REFERENCE SUBTASK {stage + 1}/5 | {STAGE_NAMES[stage]}",
                (12, 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (15, 15, 15),
                2,
                cv2.LINE_AA,
            )
            timeline_y = height - 20
            cv2.line(
                frame,
                (14, timeline_y),
                (width - 14, timeline_y),
                (255, 255, 255),
                4,
            )
            for index, boundary in enumerate(boundary_frames, start=1):
                x = 14 + round((width - 28) * boundary / frame_count)
                cv2.line(
                    frame,
                    (x, timeline_y - 9),
                    (x, timeline_y + 9),
                    COLORS[index],
                    3,
                )
            cursor = 14 + round((width - 28) * frame_index / frame_count)
            cv2.circle(frame, (cursor, timeline_y), 6, color, -1)
            if len(boundary_frames):
                closest = int(np.argmin(np.abs(boundary_frames - frame_index)))
                if abs(int(boundary_frames[closest]) - frame_index) <= round(0.4 * fps):
                    cv2.putText(
                        frame,
                        f"BOUNDARY {closest + 1} | {boundary_seconds[closest]:.2f}s",
                        (width // 2 - 115, 78),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.66,
                        COLORS[closest + 1],
                        2,
                        cv2.LINE_AA,
                    )
            writer.write(frame)
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
    print(
        f"wrote {args.output} ({frame_index} frames, {fps:.3f} FPS, "
        f"boundaries={boundary_seconds.tolist()})"
    )


if __name__ == "__main__":
    main()
