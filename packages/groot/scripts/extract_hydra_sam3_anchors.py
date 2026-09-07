from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from groot_subtask_phase_probe.hydra_action_flow import select_mask_union

VIDEO_KEY = "observation.images.follower_d455f"
PROMPT_SPECS = (
    ("robot", "robot arm", 2),
    ("weights", "metal puck", 4),
    ("cart", "blue cart", 1),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--sam3-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--anchor-seconds", type=float, default=4.0)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    return parser.parse_args()


def to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def select_union(result: dict, limit: int, threshold: float) -> np.ndarray:
    scores = to_numpy(result["scores"]).astype(np.float32)
    masks = to_numpy(result["masks"]).astype(bool)
    return select_mask_union(scores, masks, limit, threshold)


def read_rgb_frame(source: Path, timestamp_s: float, width: int, height: int) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp_s:.9f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    expected = width * height * 3
    if len(result.stdout) != expected:
        raise RuntimeError(f"Expected {expected} frame bytes, got {len(result.stdout)}")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, width, 3)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.sam3_root.resolve()))
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    info = json.loads((args.dataset / "meta/info.json").read_text())
    fps = float(info["fps"])
    height, width, _ = info["features"][VIDEO_KEY]["shape"]
    rows = {
        int(row["episode_index"]): row
        for row in pq.read_table(
            args.dataset / "meta/episodes/chunk-000/file-000.parquet"
        ).to_pylist()
    }
    split = json.loads(args.split.read_text())
    split_by_episode = {int(episode): "train" for episode in split["train"]}
    split_by_episode.update({int(episode): "validation" for episode in split["validation"]})

    model = build_sam3_image_model(device=args.device).eval()
    processor = Sam3Processor(model, confidence_threshold=args.threshold, device=args.device)
    anchor_stride = max(1, round(args.anchor_seconds * fps))
    summaries = []
    for episode in range(args.episodes):
        row = rows[episode]
        episode_dir = args.output_dir / split_by_episode[episode] / f"episode_{episode:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        source_index = int(row[f"videos/{VIDEO_KEY}/file_index"])
        source_start = float(row[f"videos/{VIDEO_KEY}/from_timestamp"])
        source = args.dataset / "videos" / VIDEO_KEY / "chunk-000" / f"file-{source_index:03d}.mp4"
        records = []
        for anchor_frame in range(0, int(row["length"]), anchor_stride):
            output = episode_dir / f"anchor_{anchor_frame:06d}.npz"
            if output.exists():
                records.append({"anchor_frame": anchor_frame, "status": "skipped_existing"})
                continue
            frame = read_rgb_frame(source, source_start + anchor_frame / fps, width, height)
            started = time.perf_counter()
            state = processor.set_image(Image.fromarray(frame))
            masks = {}
            for name, prompt, limit in PROMPT_SPECS:
                masks[name] = select_union(
                    processor.set_text_prompt(prompt, state), limit, args.threshold
                )
            robot_mask = masks["robot"]
            object_mask = masks["weights"] | masks["cart"]
            np.savez_compressed(
                output,
                robot_mask=robot_mask,
                object_mask=object_mask,
                weights_mask=masks["weights"],
                cart_mask=masks["cart"],
                anchor_frame=np.asarray(anchor_frame, dtype=np.int32),
            )
            records.append(
                {
                    "anchor_frame": anchor_frame,
                    "latency_s": time.perf_counter() - started,
                    "robot_pixels": int(robot_mask.sum()),
                    "object_pixels": int(object_mask.sum()),
                }
            )
        summary = {"episode": episode, "anchors": records}
        (episode_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        summaries.append(summary)
        print(json.dumps({"episode": episode, "anchors": len(records)}), flush=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "method": "SAM3 at each Hydra action-flow anchor",
                "anchor_seconds": args.anchor_seconds,
                "prompts": [spec[1] for spec in PROMPT_SPECS],
                "episodes": summaries,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
