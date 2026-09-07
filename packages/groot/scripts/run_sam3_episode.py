from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

COLORS = np.asarray(
    [
        (32, 180, 255),
        (0, 220, 120),
        (255, 90, 80),
        (210, 80, 220),
        (60, 240, 240),
        (230, 170, 50),
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--sam3-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--times", default="5,18,29,40,50,54")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    return parser.parse_args()


def to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def read_frame(capture: cv2.VideoCapture, timestamp: float) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Could not read frame at {timestamp:.3f}s")
    return frame


def render(frame: np.ndarray, prompt: str, scores, boxes, masks) -> np.ndarray:
    overlay = frame.copy()
    labels = frame.copy()
    for index, (score, box, mask) in enumerate(zip(scores, boxes, masks, strict=True)):
        color = tuple(int(value) for value in COLORS[index % len(COLORS)])
        overlay[mask] = color
        x1, y1, x2, y2 = [round(value) for value in box]
        cv2.rectangle(labels, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            labels,
            f"{prompt} {score:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return cv2.addWeighted(overlay, 0.42, labels, 0.58, 0)


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    sys.path.insert(0, str(args.sam3_root))
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_started = time.perf_counter()
    model = build_sam3_image_model(device=args.device).eval()
    processor = Sam3Processor(
        model, confidence_threshold=args.threshold, device=args.device
    )
    model_load_seconds = time.perf_counter() - model_started
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {args.input}")

    records = []
    times = [float(value) for value in args.times.split(",")]
    try:
        for timestamp in times:
            frame = read_frame(capture, timestamp)
            for prompt in args.prompt:
                started = time.perf_counter()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                state = processor.set_image(Image.fromarray(rgb))
                result = processor.set_text_prompt(prompt, state)
                latency_seconds = time.perf_counter() - started
                scores = to_numpy(result["scores"]).astype(float)
                boxes = to_numpy(result["boxes"]).astype(float)
                masks = to_numpy(result["masks"]).astype(bool)
                if masks.ndim == 4 and masks.shape[1] == 1:
                    masks = masks[:, 0]
                order = np.argsort(-scores)
                scores, boxes, masks = scores[order], boxes[order], masks[order]
                slug = prompt.lower().replace(" ", "_")
                stem = f"t{timestamp:06.2f}_{slug}"
                output_image = args.output_dir / f"{stem}.jpg"
                cv2.imwrite(str(output_image), render(frame, prompt, scores, boxes, masks))
                np.savez_compressed(
                    args.output_dir / f"{stem}.npz",
                    scores=scores.astype(np.float32),
                    boxes=boxes.astype(np.float32),
                    masks=masks,
                )
                records.append(
                    {
                        "timestamp_s": timestamp,
                        "prompt": prompt,
                        "detections": len(scores),
                        "scores": scores.tolist(),
                        "boxes": boxes.tolist(),
                        "latency_s": latency_seconds,
                        "image": output_image.name,
                    }
                )
    finally:
        capture.release()
    summary = {
        "input": str(args.input),
        "device": args.device,
        "threshold": args.threshold,
        "model_load_s": model_load_seconds,
        "sam3_cache": os.getenv("SAM3_HF_CACHE"),
        "records": records,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
