from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

VIDEO_KEYS = {
    "external": "observation.images.follower_d455f",
    "left_wrist": "observation.images.left_wrist",
    "right_wrist": "observation.images.right_wrist",
}


def sample_frame_indices(length: int, source_fps: float, sample_fps: float) -> np.ndarray:
    if source_fps <= 0 or sample_fps <= 0:
        raise ValueError("FPS values must be positive")
    stride = source_fps / sample_fps
    if not np.isclose(stride, round(stride)):
        raise ValueError("sample_fps must evenly divide source_fps")
    return np.arange(0, length, round(stride), dtype=np.int32)


def align_decoded_frames(frames: np.ndarray, expected: int) -> tuple[np.ndarray, int]:
    if len(frames) >= expected:
        return frames[:expected], 0
    missing = expected - len(frames)
    if missing > 1:
        raise ValueError(f"Decoded video is {missing} frames shorter than metadata")
    padding = np.repeat(frames[-1:], missing, axis=0)
    return np.concatenate([frames, padding]), missing


def _decode_video_segment(
    video: Path,
    *,
    start_s: float,
    duration_s: float,
    sample_fps: float,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_s:.9f}",
        "-t",
        f"{duration_s:.9f}",
        "-i",
        str(video),
        "-vf",
        f"fps={sample_fps}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    if process.stdout is None:
        raise RuntimeError("ffmpeg stdout pipe was not created")
    frame_bytes = width * height * 3
    frames = []
    while payload := process.stdout.read(frame_bytes):
        if len(payload) != frame_bytes:
            process.kill()
            raise RuntimeError(f"Short decoded frame: {len(payload)}/{frame_bytes}")
        frames.append(np.frombuffer(payload, np.uint8).reshape(height, width, 3).copy())
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {video}")
    if not frames:
        raise RuntimeError(f"No frames decoded from {video} at {start_s:.3f}s")
    return np.stack(frames)


def _to_device(value: Any, device, dtype):
    import torch

    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            return value.to(device=device, dtype=dtype)
        return value.to(device)
    if isinstance(value, dict) or hasattr(value, "items"):
        return {key: _to_device(item, device, dtype) for key, item in value.items()}
    return value


def pool_backbone_batch(output: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    hidden = output["backbone_features"]
    attention = output["backbone_attention_mask"].bool()
    image = output["image_mask"].bool() & attention
    attention_weights = attention.to(hidden.dtype).unsqueeze(-1)
    image_weights = image.to(hidden.dtype).unsqueeze(-1)
    hidden_mean = (hidden * attention_weights).sum(1) / attention_weights.sum(1).clamp_min(1)
    vision_mean = (hidden * image_weights).sum(1) / image_weights.sum(1).clamp_min(1)
    return (
        hidden_mean.float().cpu().numpy(),
        vision_mean.float().cpu().numpy(),
        {
            "valid_min": int(attention.sum(1).min()),
            "valid_max": int(attention.sum(1).max()),
            "vision_min": int(image.sum(1).min()),
            "vision_max": int(image.sum(1).max()),
        },
    )


def _episode_metadata(dataset: Path) -> dict[int, dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(dataset / "meta/episodes/chunk-000/file-000.parquet")
    return {int(row["episode_index"]): row for row in table.to_pylist()}


def _robot_states(dataset: Path) -> dict[int, np.ndarray]:
    import pyarrow.parquet as pq

    table = pq.read_table(
        dataset / "data/chunk-000/file-000.parquet",
        columns=["episode_index", "frame_index", "observation.state"],
    )
    episode_indices = table["episode_index"].to_numpy()
    frame_indices = table["frame_index"].to_numpy()
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    output = {}
    for episode in np.unique(episode_indices):
        selected = episode_indices == episode
        order = np.argsort(frame_indices[selected])
        output[int(episode)] = states[selected][order]
    return output


def _video_path(dataset: Path, row: dict[str, Any], key: str) -> Path:
    chunk = int(row[f"videos/{key}/chunk_index"])
    file_index = int(row[f"videos/{key}/file_index"])
    return dataset / "videos" / key / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"


def extract_hidden_states(
    *,
    dataset: Path,
    checkpoint: Path,
    output_dir: Path,
    split_path: Path,
    isaac_root: Path,
    sample_fps: float,
    batch_size: int,
    episode_limit: int | None,
) -> dict[str, Any]:
    import torch
    from peft.tuners.tuners_utils import BaseTunerLayer

    sys.path.insert(0, str(isaac_root))
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    split = json.loads(split_path.read_text())
    split_by_episode = {int(value): "train" for value in split["train"]}
    split_by_episode.update({int(value): "validation" for value in split["validation"]})
    episode_ids = sorted(split_by_episode)
    if episode_limit is not None:
        episode_ids = episode_ids[:episode_limit]
    metadata = _episode_metadata(dataset)
    robot_states = _robot_states(dataset)

    policy = Gr00tPolicy("NEW_EMBODIMENT", str(checkpoint), device="cuda:0")
    model = policy.model
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer in model.modules():
        if isinstance(layer, BaseTunerLayer):
            layer.enable_adapters(False)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    torch.cuda.reset_peak_memory_stats(device)

    summaries = []
    total_started = time.perf_counter()
    for episode in episode_ids:
        episode_started = time.perf_counter()
        row = metadata[episode]
        length = int(row["length"])
        sample_frames = sample_frame_indices(length, 30.0, sample_fps)
        states = robot_states[episode][sample_frames]
        cameras = {}
        for camera, key in VIDEO_KEYS.items():
            start_s = float(row[f"videos/{key}/from_timestamp"])
            end_s = float(row[f"videos/{key}/to_timestamp"])
            cameras[camera] = _decode_video_segment(
                _video_path(dataset, row, key),
                start_s=start_s,
                duration_s=end_s - start_s,
                sample_fps=sample_fps,
            )
        frame_count = len(sample_frames)
        tail_padding = {}
        for name, frames in cameras.items():
            cameras[name], tail_padding[name] = align_decoded_frames(frames, frame_count)

        hidden_chunks = []
        vision_chunks = []
        latencies = []
        token_counts: dict[str, int] | None = None
        task = str(row["tasks"][0])
        for start in range(0, frame_count, batch_size):
            stop = min(frame_count, start + batch_size)
            chunk_states = states[start:stop]
            observation = {
                "video.external": cameras["external"][start:stop, None],
                "video.left_wrist": cameras["left_wrist"][start:stop, None],
                "video.right_wrist": cameras["right_wrist"][start:stop, None],
                "state.left_arm": chunk_states[:, None, :7],
                "state.left_gripper": chunk_states[:, None, 7:8],
                "state.right_arm": chunk_states[:, None, 8:15],
                "state.right_gripper": chunk_states[:, None, 15:16],
                "annotation.human.task_description": [task] * (stop - start),
            }
            processed = policy.processor.process_observation(observation, policy.embodiment_tag)
            backbone_input = _to_device(model.backbone.prepare_input(processed), device, dtype)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode():
                backbone_output = model.backbone(backbone_input)
            torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            hidden, vision, counts = pool_backbone_batch(backbone_output)
            hidden_chunks.append(hidden)
            vision_chunks.append(vision)
            latencies.extend([elapsed_ms / (stop - start)] * (stop - start))
            token_counts = counts

        split_name = split_by_episode[episode]
        episode_dir = output_dir / split_name / f"episode_{episode:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        output_path = episode_dir / "groot_backbone_hidden.npz"
        np.savez_compressed(
            output_path,
            hidden_mean=np.concatenate(hidden_chunks).astype(np.float16),
            vision_mean=np.concatenate(vision_chunks).astype(np.float16),
            sample_frames=sample_frames,
            latency_ms=np.asarray(latencies, dtype=np.float32),
        )
        summary = {
            "episode": episode,
            "split": split_name,
            "frames": frame_count,
            "hidden_dim": int(hidden_chunks[0].shape[1]),
            "sample_fps": sample_fps,
            "batch_size": batch_size,
            "mean_backbone_latency_ms_per_frame": float(np.mean(latencies)),
            "runtime_s": time.perf_counter() - episode_started,
            "token_counts": token_counts,
            "video_tail_padding_frames": tail_padding,
            "output": str(output_path),
        }
        summaries.append(summary)
        print(json.dumps(summary), flush=True)

    result = {
        "episodes": len(summaries),
        "samples": int(sum(item["frames"] for item in summaries)),
        "sample_fps": sample_fps,
        "batch_size": batch_size,
        "hidden_contract": "R3 attention-mask mean of GR00T backbone_features",
        "vision_contract": "R3b image-mask mean of GR00T backbone_features",
        "dynamic_cameras": list(VIDEO_KEYS),
        "peft_adapters_enabled": False,
        "action_head_used": False,
        "mean_backbone_latency_ms_per_frame": float(
            np.average(
                [item["mean_backbone_latency_ms_per_frame"] for item in summaries],
                weights=[item["frames"] for item in summaries],
            )
        ),
        "peak_vram_mib": float(torch.cuda.max_memory_allocated(device) / (1024**2)),
        "runtime_s": time.perf_counter() - total_started,
        "episode_summaries": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--isaac-root", type=Path, default=Path("/root/projects/Isaac-GR00T"))
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--episode-limit", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = extract_hidden_states(
        dataset=args.dataset,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        split_path=args.split,
        isaac_root=args.isaac_root,
        sample_fps=args.sample_fps,
        batch_size=args.batch_size,
        episode_limit=args.episode_limit,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
