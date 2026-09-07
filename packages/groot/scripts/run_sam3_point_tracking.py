from __future__ import annotations

import argparse
import json
import subprocess
from collections import deque
from pathlib import Path

import cv2
import numpy as np

COLORS = [
    (255, 170, 40),
    (60, 230, 80),
    (80, 210, 255),
    (255, 100, 100),
    (220, 100, 255),
    (40, 130, 255),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--car-seed", type=Path, required=True)
    parser.add_argument("--puck-seed", type=Path, required=True)
    parser.add_argument("--cylinder-seed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points-per-object", type=int, default=20)
    parser.add_argument("--trail-frames", type=int, default=60)
    return parser.parse_args()


def load_seed(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return data["scores"], data["boxes"], data["masks"]


def choose_masks(
    car_seed: Path, puck_seed: Path, cylinder_seed: Path
) -> list[tuple[str, np.ndarray]]:
    car_scores, _, car_masks = load_seed(car_seed)
    _, puck_boxes, puck_masks = load_seed(puck_seed)
    chosen = [("cart", car_masks[int(np.argmax(car_scores))].astype(np.uint8))]
    puck_indices = []
    for index, box in enumerate(puck_boxes):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        area = (x2 - x1) * (y2 - y1)
        if 300 <= cx <= 460 and 225 <= cy <= 430 and 40 <= area <= 2500:
            puck_indices.append(index)
    for number, index in enumerate(puck_indices[:4], start=1):
        chosen.append((f"weight {number}", puck_masks[index].astype(np.uint8)))
    cylinder_scores, cylinder_boxes, cylinder_masks = load_seed(cylinder_seed)
    candidates = []
    for index, (score, box) in enumerate(zip(cylinder_scores, cylinder_boxes, strict=True)):
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if (
            150 <= cx <= 320
            and 190 <= cy <= 280
            and width >= 70
            and width / max(height, 1) >= 2
        ):
            candidates.append((float(score), index))
    if not candidates:
        raise RuntimeError("SAM3 did not produce a usable cylinder mask")
    _, cylinder_index = max(candidates)
    chosen.append(("cylinder", cylinder_masks[cylinder_index].astype(np.uint8)))
    return chosen


def seed_points(
    gray: np.ndarray, masks: list[tuple[str, np.ndarray]], points_per_object: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    all_points = []
    all_ids = []
    names = []
    for object_id, (name, mask) in enumerate(masks):
        points = cv2.goodFeaturesToTrack(
            gray,
            mask=(mask * 255).astype(np.uint8),
            maxCorners=points_per_object,
            qualityLevel=0.005,
            minDistance=3,
            blockSize=5,
        )
        if points is None:
            ys, xs = np.where(mask.astype(bool))
            if len(xs) == 0:
                continue
            points = np.asarray([[[xs.mean(), ys.mean()]]], dtype=np.float32)
        all_points.append(points.reshape(-1, 2))
        all_ids.extend([object_id] * len(points))
        names.append(name)
    return (
        np.concatenate(all_points).astype(np.float32),
        np.asarray(all_ids, dtype=np.int32),
        names,
    )


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
    previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    masks = choose_masks(args.car_seed, args.puck_seed, args.cylinder_seed)
    points, object_ids, names = seed_points(
        previous_gray, masks, args.points_per_object
    )
    histories = [deque([tuple(point)], maxlen=args.trail_frames) for point in points]
    initial_counts = {
        name: int((object_ids == object_id).sum())
        for object_id, name in enumerate(names)
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tracking.mp4")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {temporary}")
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
                    & (forward_backward < 3.0)
                    & (candidate[:, 0] >= 0)
                    & (candidate[:, 0] < width)
                    & (candidate[:, 1] >= 0)
                    & (candidate[:, 1] < height)
                )
                points = candidate[valid]
                object_ids = object_ids[valid]
                histories = [
                    history
                    for history, keep in zip(histories, valid, strict=True)
                    if keep
                ]
                for history, point in zip(histories, points, strict=True):
                    history.append(tuple(point))

            canvas = frame.copy()
            for history, point, object_id in zip(
                histories, points, object_ids, strict=True
            ):
                color = COLORS[object_id % len(COLORS)]
                trail = np.asarray(history, dtype=np.int32).reshape(-1, 1, 2)
                if len(trail) > 1:
                    cv2.polylines(canvas, [trail], False, color, 1, cv2.LINE_AA)
                cv2.circle(canvas, tuple(np.round(point).astype(int)), 4, color, -1)

            counts = {}
            for object_id, name in enumerate(names):
                selected = object_ids == object_id
                counts[name] = int(selected.sum())
                if selected.any():
                    center = np.median(points[selected], axis=0).astype(int)
                    color = COLORS[object_id % len(COLORS)]
                    cv2.putText(
                        canvas,
                        f"{name}: {counts[name]} pts",
                        (int(center[0]) + 6, int(center[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
            cv2.putText(
                canvas,
                "SAM3 object masks (frame 0) -> point tracking | 30 FPS",
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
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
                        "alive_points": counts,
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
    final_counts = {
        name: int((object_ids == object_id).sum())
        for object_id, name in enumerate(names)
    }
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "method": "SAM3 frame-0 object masks + pyramidal Lucas-Kanade points",
        "source_fps": fps,
        "output_fps": fps,
        "source_frames": source_frames,
        "output_frames": frame_index + 1,
        "objects": names,
        "initial_points": initial_counts,
        "final_alive_points": final_counts,
        "stats": stats,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "stats"}))


if __name__ == "__main__":
    main()
