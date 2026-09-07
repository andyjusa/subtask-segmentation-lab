from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROMPTS = ("car", "metal puck")
COLORS = {
    "car": (255, 170, 40),
    "metal puck": (70, 230, 80),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--sam3-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    return parser.parse_args()


def to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def select_instances(prompt: str, scores, boxes, masks) -> list[dict]:
    instances = []
    for score, box, mask in zip(scores, boxes, masks, strict=True):
        x1, y1, x2, y2 = box.tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if prompt == "metal puck" and not (
            300 <= cx <= 460 and 225 <= cy <= 430 and 40 <= area <= 2500
        ):
            continue
        instances.append(
            {
                "prompt": prompt,
                "score": float(score),
                "box": [float(value) for value in box],
                "mask": mask.astype(np.uint8),
            }
        )
    if prompt == "car":
        return instances[:1]
    if prompt == "metal puck":
        return instances[:4]
    return instances


def detect(processor, frame: np.ndarray, prompts: tuple[str, ...]) -> dict[str, list[dict]]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    state = processor.set_image(Image.fromarray(rgb))
    detected = {}
    for prompt in prompts:
        result = processor.set_text_prompt(prompt, state)
        scores = to_numpy(result["scores"]).astype(float)
        boxes = to_numpy(result["boxes"]).astype(float)
        masks = to_numpy(result["masks"]).astype(bool)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        order = np.argsort(-scores)
        detected[prompt] = select_instances(
            prompt, scores[order], boxes[order], masks[order]
        )
    return detected


def warp_instances(instances: dict[str, list[dict]], flow: np.ndarray) -> None:
    height, width = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    map_x = grid_x - flow[..., 0]
    map_y = grid_y - flow[..., 1]
    for prompt_instances in instances.values():
        for instance in prompt_instances:
            instance["mask"] = cv2.remap(
                instance["mask"],
                map_x,
                map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )


def render(frame: np.ndarray, instances: dict[str, list[dict]]) -> np.ndarray:
    overlay = frame.copy()
    labels = frame.copy()
    for prompt, prompt_instances in instances.items():
        color = COLORS[prompt]
        for instance in prompt_instances:
            mask = instance["mask"].astype(bool)
            if not mask.any():
                continue
            overlay[mask] = color
            ys, xs = np.where(mask)
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            cv2.rectangle(labels, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                labels,
                f"{prompt} {instance['score']:.2f}",
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    output = cv2.addWeighted(overlay, 0.42, labels, 0.58, 0)
    cv2.putText(
        output,
        "SAM3 keyframes + optical-flow tracking | output: original FPS",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


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
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, round(fps / args.sample_fps))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tracking.mp4")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {temporary}")

    model_started = time.perf_counter()
    model = build_sam3_image_model(device=args.device).eval()
    processor = Sam3Processor(
        model, confidence_threshold=args.threshold, device=args.device
    )
    model_load_s = time.perf_counter() - model_started

    previous_gray = None
    instances: dict[str, list[dict]] = {prompt: [] for prompt in PROMPTS}
    keyframes = []
    frame_index = 0
    processing_started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frame_index % interval == 0:
                started = time.perf_counter()
                detected = detect(processor, frame, PROMPTS)
                latency_s = time.perf_counter() - started
                for prompt in PROMPTS:
                    if detected[prompt]:
                        instances[prompt] = detected[prompt]
                keyframes.append(
                    {
                        "frame": frame_index,
                        "timestamp_s": frame_index / fps,
                        "latency_s": latency_s,
                        "detections": {
                            prompt: [
                                {
                                    "score": item["score"],
                                    "box": item["box"],
                                }
                                for item in detected[prompt]
                            ]
                            for prompt in PROMPTS
                        },
                    }
                )
                print(
                    f"keyframe {frame_index}/{frame_count}: "
                    f"car={len(detected['car'])}, "
                    f"puck={len(detected['metal puck'])}, {latency_s:.2f}s",
                    flush=True,
                )
            elif previous_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    previous_gray,
                    gray,
                    None,
                    0.5,
                    3,
                    21,
                    3,
                    5,
                    1.2,
                    0,
                )
                warp_instances(instances, flow)
            writer.write(render(frame, instances))
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
        "device": args.device,
        "source_fps": fps,
        "output_fps": fps,
        "source_frames": frame_count,
        "output_frames": frame_index,
        "sam3_sample_fps": args.sample_fps,
        "sam3_interval_frames": interval,
        "threshold": args.threshold,
        "prompts": list(PROMPTS),
        "model_load_s": model_load_s,
        "processing_s": time.perf_counter() - processing_started,
        "keyframes": keyframes,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "keyframes"}))


if __name__ == "__main__":
    main()
