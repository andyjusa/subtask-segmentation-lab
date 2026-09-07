from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROMPTS = ("car", "metal puck", "metal cylinder")
NAMES = ("cart", "weights", "cylinder")
COLORS = ((255, 170, 40), (60, 230, 80), (40, 130, 255))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--sam3-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reseed-seconds", type=float, default=2.0)
    parser.add_argument("--points-per-mask", type=int, default=12)
    parser.add_argument("--trail-frames", type=int, default=60)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    return parser.parse_args()


def to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def select_masks(prompt: str, scores, boxes, masks) -> list[np.ndarray]:
    selected = []
    candidates = []
    for score, box, mask in zip(scores, boxes, masks, strict=True):
        x1, y1, x2, y2 = box.tolist()
        width, height = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        area = max(0.0, width) * max(0.0, height)
        if prompt == "car" and score >= 0.2 or (
            prompt == "metal puck"
            and score >= 0.28
            and 300 <= cx <= 460
            and 225 <= cy <= 430
            and 40 <= area <= 2500
        ) or (
            prompt == "metal cylinder"
            and score >= 0.12
            and 145 <= cx <= 330
            and 185 <= cy <= 285
            and width >= 65
            and width / max(height, 1) >= 2
        ):
            candidates.append((float(score), mask))
    candidates.sort(key=lambda item: item[0], reverse=True)
    limit = 4 if prompt == "metal puck" else 1
    selected.extend(mask.astype(np.uint8) for _, mask in candidates[:limit])
    return selected


def detect_masks(processor, frame: np.ndarray) -> dict[int, list[np.ndarray]]:
    state = processor.set_image(
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    )
    detected = {}
    for object_id, prompt in enumerate(PROMPTS):
        result = processor.set_text_prompt(prompt, state)
        scores = to_numpy(result["scores"]).astype(float)
        boxes = to_numpy(result["boxes"]).astype(float)
        masks = to_numpy(result["masks"]).astype(bool)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        order = np.argsort(-scores)
        detected[object_id] = select_masks(
            prompt, scores[order], boxes[order], masks[order]
        )
    return detected


def points_from_masks(
    gray: np.ndarray, masks: list[np.ndarray], points_per_mask: int
) -> np.ndarray:
    groups = []
    for mask in masks:
        points = cv2.goodFeaturesToTrack(
            gray,
            mask=(mask * 255).astype(np.uint8),
            maxCorners=points_per_mask,
            qualityLevel=0.005,
            minDistance=3,
            blockSize=5,
        )
        if points is None:
            ys, xs = np.where(mask.astype(bool))
            if len(xs):
                groups.append(
                    np.asarray([[xs.mean(), ys.mean()]], dtype=np.float32)
                )
        else:
            groups.append(points.reshape(-1, 2).astype(np.float32))
    if not groups:
        return np.empty((0, 2), dtype=np.float32)
    return np.concatenate(groups)


def track_points(
    previous_gray: np.ndarray,
    gray: np.ndarray,
    points: np.ndarray,
    object_ids: np.ndarray,
    histories: list[deque],
) -> tuple[np.ndarray, np.ndarray, list[deque]]:
    if len(points) == 0:
        return points, object_ids, histories
    lk_params = {
        "winSize": (31, 31),
        "maxLevel": 4,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    }
    old = points.reshape(-1, 1, 2)
    new, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, gray, old, None, **lk_params
    )
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        gray, previous_gray, new, None, **lk_params
    )
    candidate = new.reshape(-1, 2)
    error = np.linalg.norm(old - backward, axis=2).reshape(-1)
    height, width = gray.shape
    valid = (
        status.reshape(-1).astype(bool)
        & backward_status.reshape(-1).astype(bool)
        & (error < 3.0)
        & (candidate[:, 0] >= 0)
        & (candidate[:, 0] < width)
        & (candidate[:, 1] >= 0)
        & (candidate[:, 1] < height)
    )
    points = candidate[valid]
    object_ids = object_ids[valid]
    histories = [
        history for history, keep in zip(histories, valid, strict=True) if keep
    ]
    for history, point in zip(histories, points, strict=True):
        history.append(tuple(point))
    return points, object_ids, histories


def render(
    frame: np.ndarray,
    points: np.ndarray,
    object_ids: np.ndarray,
    histories: list[deque],
    reseed_seconds: float,
) -> np.ndarray:
    canvas = frame.copy()
    for history, point, object_id in zip(histories, points, object_ids, strict=True):
        color = COLORS[int(object_id)]
        trail = np.asarray(history, dtype=np.int32).reshape(-1, 1, 2)
        if len(trail) > 1:
            cv2.polylines(canvas, [trail], False, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(np.round(point).astype(int)), 4, color, -1)
    for object_id, name in enumerate(NAMES):
        selected = object_ids == object_id
        if not selected.any():
            continue
        center = np.median(points[selected], axis=0).astype(int)
        cv2.putText(
            canvas,
            f"{name}: {int(selected.sum())} pts",
            (int(center[0]) + 6, int(center[1]) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLORS[object_id],
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        f"SAM3 re-detect {reseed_seconds:g}s + point tracking | 30 FPS",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def summarize_points(
    points: np.ndarray, object_ids: np.ndarray, width: int, height: int
) -> np.ndarray:
    values = []
    scale = np.asarray([width, height], dtype=np.float32)
    for object_id in range(len(NAMES)):
        selected = points[object_ids == object_id]
        if len(selected) == 0:
            values.extend((0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        normalized = selected / scale
        values.extend(
            (
                float(min(len(selected), 48) / 48.0),
                float(normalized[:, 0].mean()),
                float(normalized[:, 1].mean()),
                float(normalized[:, 0].std()),
                float(normalized[:, 1].std()),
            )
        )
    return np.asarray(values, dtype=np.float32)


def main() -> None:
    args = parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    sys.path.insert(0, str(args.sam3_root))
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {args.input}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    reseed_interval = max(1, round(args.reseed_seconds * fps))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tracking.mp4")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    model_started = time.perf_counter()
    model = build_sam3_image_model(device=args.device).eval()
    processor = Sam3Processor(model, confidence_threshold=0.1, device=args.device)
    model_load_s = time.perf_counter() - model_started
    points = np.empty((0, 2), dtype=np.float32)
    object_ids = np.empty(0, dtype=np.int32)
    histories: list[deque] = []
    previous_gray = None
    frame_index = 0
    reseeds = []
    point_features = []
    reseed_flags = []
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if previous_gray is not None:
                points, object_ids, histories = track_points(
                    previous_gray, gray, points, object_ids, histories
                )
            if frame_index % reseed_interval == 0:
                inference_started = time.perf_counter()
                masks_by_object = detect_masks(processor, frame)
                refreshed_points = []
                refreshed_ids = []
                for object_id in range(len(NAMES)):
                    detected_points = points_from_masks(
                        gray,
                        masks_by_object[object_id],
                        args.points_per_mask,
                    )
                    if len(detected_points):
                        refreshed_points.append(detected_points)
                        refreshed_ids.extend([object_id] * len(detected_points))
                    else:
                        retained = points[object_ids == object_id]
                        if len(retained):
                            refreshed_points.append(retained)
                            refreshed_ids.extend([object_id] * len(retained))
                if refreshed_points:
                    points = np.concatenate(refreshed_points).astype(np.float32)
                    object_ids = np.asarray(refreshed_ids, dtype=np.int32)
                    histories = [
                        deque([tuple(point)], maxlen=args.trail_frames)
                        for point in points
                    ]
                counts = {
                    name: int((object_ids == object_id).sum())
                    for object_id, name in enumerate(NAMES)
                }
                latency = time.perf_counter() - inference_started
                reseeds.append(
                    {
                        "frame": frame_index,
                        "timestamp_s": frame_index / fps,
                        "latency_s": latency,
                        "points": counts,
                    }
                )
                print(
                    f"reseed {frame_index}/{source_frames}: {counts}, {latency:.2f}s",
                    flush=True,
                )
            point_features.append(summarize_points(points, object_ids, width, height))
            reseed_flags.append(frame_index % reseed_interval == 0)
            writer.write(
                render(frame, points, object_ids, histories, args.reseed_seconds)
            )
            previous_gray = gray
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
        "method": "periodic SAM3 image re-detection + KLT point tracking",
        "device": args.device,
        "source_fps": fps,
        "output_fps": fps,
        "source_frames": source_frames,
        "output_frames": frame_index,
        "reseed_seconds": args.reseed_seconds,
        "model_load_s": model_load_s,
        "processing_s": time.perf_counter() - started,
        "reseeds": reseeds,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output.with_suffix(".features.npz"),
        features=np.stack(point_features),
        reseed_flags=np.asarray(reseed_flags, dtype=bool),
        fps=np.asarray(fps),
        names=np.asarray(NAMES),
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "reseeds"}))


if __name__ == "__main__":
    main()
