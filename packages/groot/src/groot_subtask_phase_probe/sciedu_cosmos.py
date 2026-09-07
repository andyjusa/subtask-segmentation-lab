from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TASK = (
    "Put the cart on the ramp, add weights one by one, remove all the weights, "
    "then move the cart back to the desk."
)
STAGE_NAMES = (
    "put_cart_on_ramp",
    "add_weights",
    "remove_weights",
    "return_cart_to_desk",
)
BOUNDARY_SECONDS = np.asarray([22.0, 47.5, 63.25], dtype=np.float64)
TERMINAL_SECONDS = 73.0
VIDEO_KEYS = {
    "external": "observation.images.follower_d455f",
    "left_wrist": "observation.images.left_wrist",
    "right_wrist": "observation.images.right_wrist",
}


@dataclass(frozen=True)
class ClipConfig:
    seconds: float
    fps: float

    @property
    def frame_count(self) -> int:
        return max(1, round(self.seconds * self.fps))

    @property
    def slug(self) -> str:
        seconds = str(self.seconds).replace(".", "p")
        fps = str(self.fps).replace(".", "p")
        return f"{seconds}s_{fps}fps"


def stage_labels(timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    stage = np.searchsorted(BOUNDARY_SECONDS, timestamps, side="right").astype(np.int64)
    terminal = timestamps >= TERMINAL_SECONDS
    return stage, terminal


def trailing_clip_times(end_seconds: float, config: ClipConfig) -> np.ndarray:
    offsets = np.arange(config.frame_count - 1, -1, -1, dtype=np.float64) / config.fps
    return np.maximum(0.0, end_seconds - offsets)


def blocked_stage_folds(
    labels: np.ndarray, *, folds: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = np.asarray(labels, dtype=np.int64)
    fold_assignments = np.full(len(labels), -1, dtype=np.int64)
    for stage in sorted(np.unique(labels)):
        indices = np.flatnonzero(labels == stage)
        if len(indices) < folds:
            raise ValueError(f"Stage {stage} has only {len(indices)} samples for {folds} folds")
        for fold, block in enumerate(np.array_split(indices, folds)):
            fold_assignments[block] = fold
    result = []
    for fold in range(folds):
        test = np.flatnonzero(fold_assignments == fold)
        train = np.flatnonzero(fold_assignments != fold)
        result.append((train, test))
    return result


def _fit_transform(
    train_x: np.ndarray, test_x: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train_x)
    train_scaled = scaler.transform(train_x)
    test_scaled = scaler.transform(test_x)
    components = min(32, train_scaled.shape[0] - 4, train_scaled.shape[1])
    if components < 2:
        return train_scaled, test_scaled
    pca = PCA(n_components=components, random_state=seed).fit(train_scaled)
    return pca.transform(train_scaled), pca.transform(test_scaled)


def _causal_monotonic_decode(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    smoothed = np.empty_like(probabilities)
    smoothed[0] = probabilities[0]
    for index in range(1, len(probabilities)):
        smoothed[index] = 0.5 * probabilities[index] + 0.5 * smoothed[index - 1]
    result = np.zeros(len(smoothed), dtype=np.int64)
    current = 0
    consecutive = 0
    for index, row in enumerate(smoothed):
        if current < row.shape[0] - 1 and row[current + 1] > row[current]:
            consecutive += 1
            if consecutive >= 2:
                current += 1
                consecutive = 0
        else:
            consecutive = 0
        result[index] = current
    return result


def _predicted_boundaries(timestamps: np.ndarray, stages: np.ndarray) -> list[float]:
    changes = np.flatnonzero(np.diff(stages) > 0) + 1
    return [float(timestamps[index]) for index in changes]


def _event_metrics(predicted: list[float], tolerance: float) -> dict[str, float]:
    unmatched = list(predicted)
    errors = []
    for target in BOUNDARY_SECONDS:
        candidates = [(abs(value - target), index) for index, value in enumerate(unmatched)]
        if not candidates:
            continue
        error, index = min(candidates)
        if error <= tolerance:
            errors.append(float(error))
            unmatched.pop(index)
    true_positive = len(errors)
    false_positive = len(predicted) - true_positive
    false_negative = len(BOUNDARY_SECONDS) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        f"boundary_f1_at_{tolerance:g}s": (
            2 * true_positive / denominator if denominator else 0.0
        ),
        f"boundary_median_error_at_{tolerance:g}s": (
            float(np.median(errors)) if errors else float("nan")
        ),
        f"boundary_false_positive_at_{tolerance:g}s": float(false_positive),
    }


def evaluate_features(features_path: Path, *, seed: int = 17) -> dict[str, Any]:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    with np.load(features_path) as archive:
        timestamps = np.asarray(archive["timestamp"], dtype=np.float64)
        features = np.asarray(archive["vision_mean"], dtype=np.float32)
        labels = np.asarray(archive["stage"], dtype=np.int64)
        terminal = np.asarray(archive["terminal"], dtype=bool)
        latency = np.asarray(archive["backbone_latency_ms"], dtype=np.float64)
        peak_vram = float(np.asarray(archive["peak_vram_mib"]).reshape(-1)[0])
    boundary_distance = np.min(np.abs(timestamps[:, None] - BOUNDARY_SECONDS[None]), axis=1)
    valid = (~terminal) & (boundary_distance > 0.5)
    timestamps = timestamps[valid]
    features = features[valid]
    labels = labels[valid]
    probabilities = np.zeros((len(labels), len(STAGE_NAMES)), dtype=np.float64)
    out_of_fold_prediction = np.full(len(labels), -1, dtype=np.int64)
    fold_f1 = []
    fold_balanced = []
    for fold, (train, test) in enumerate(blocked_stage_folds(labels)):
        train_x, test_x = _fit_transform(features[train], features[test], seed=seed + fold)
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            solver="lbfgs",
            random_state=seed + fold,
        ).fit(train_x, labels[train])
        prediction = model.predict(test_x)
        out_of_fold_prediction[test] = prediction
        probabilities[test] = model.predict_proba(test_x)
        fold_f1.append(f1_score(labels[test], prediction, average="macro"))
        fold_balanced.append(balanced_accuracy_score(labels[test], prediction))
    decoded = _causal_monotonic_decode(probabilities)
    predicted_boundaries = _predicted_boundaries(timestamps, decoded)
    scaled = StandardScaler().fit_transform(features)
    components = min(32, len(features) - 1, features.shape[1])
    embedded = PCA(n_components=components, random_state=seed).fit_transform(scaled)
    clusters = KMeans(n_clusters=len(STAGE_NAMES), n_init=20, random_state=seed).fit_predict(
        embedded
    )
    result: dict[str, Any] = {
        "config": features_path.stem,
        "samples": len(labels),
        "stage_macro_f1": float(np.mean(fold_f1)),
        "stage_macro_f1_ci95": float(1.96 * np.std(fold_f1, ddof=1) / np.sqrt(len(fold_f1))),
        "stage_balanced_accuracy": float(np.mean(fold_balanced)),
        "unsupervised_adjusted_rand": float(adjusted_rand_score(labels, clusters)),
        "predicted_boundaries_s": predicted_boundaries,
        "backbone_latency_ms": float(np.mean(latency)),
        "backbone_latency_p95_ms": float(np.percentile(latency, 95)),
        "peak_vram_mib": peak_vram,
        "confusion_matrix": [
            [
                int(np.sum((labels == actual) & (out_of_fold_prediction == predicted)))
                for predicted in range(len(STAGE_NAMES))
            ]
            for actual in range(len(STAGE_NAMES))
        ],
        "misclassified_samples": [
            {
                "timestamp_s": float(timestamps[index]),
                "actual_stage": STAGE_NAMES[int(labels[index])],
                "predicted_stage": STAGE_NAMES[int(out_of_fold_prediction[index])],
            }
            for index in np.flatnonzero(out_of_fold_prediction != labels)
        ],
    }
    for stage_index, stage_name in enumerate(STAGE_NAMES):
        stage_indices = np.flatnonzero(labels == stage_index)
        result[f"accuracy_{stage_name}"] = float(
            np.mean(out_of_fold_prediction[stage_indices] == stage_index)
        )
        for section_name, section_indices in zip(
            ("start", "middle", "end"), np.array_split(stage_indices, 3), strict=True
        ):
            result[f"accuracy_{stage_name}_{section_name}"] = float(
                np.mean(out_of_fold_prediction[section_indices] == stage_index)
            )
    result.update(_event_metrics(predicted_boundaries, 0.5))
    result.update(_event_metrics(predicted_boundaries, 1.0))
    return result


def decode_video(video: Path, *, fps: float = 10.0) -> np.ndarray:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"fps={fps}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    if process.stdout is None:
        raise RuntimeError("ffmpeg stdout pipe was not created")
    frame_bytes = 640 * 480 * 3
    frames = []
    while payload := process.stdout.read(frame_bytes):
        if len(payload) != frame_bytes:
            process.kill()
            raise RuntimeError(f"Short decoded frame: {len(payload)}/{frame_bytes}")
        frames.append(np.frombuffer(payload, np.uint8).reshape(480, 640, 3).copy())
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {video}")
    return np.stack(frames)


def _to_device(value: Any, device, dtype):
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device)
    if isinstance(value, dict) or hasattr(value, "items"):
        return {key: _to_device(item, device, dtype) for key, item in value.items()}
    return value


def _pool_backbone(output: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    hidden = output["backbone_features"]
    attention = output["backbone_attention_mask"].bool()
    image = output["image_mask"].bool() & attention
    attention_weights = attention.to(hidden.dtype).unsqueeze(-1)
    image_weights = image.to(hidden.dtype).unsqueeze(-1)
    hidden_mean = (hidden * attention_weights).sum(1) / attention_weights.sum(1).clamp_min(1)
    vision_mean = (hidden * image_weights).sum(1) / image_weights.sum(1).clamp_min(1)
    return (
        hidden_mean[0].float().cpu().numpy(),
        vision_mean[0].float().cpu().numpy(),
        {"valid": int(attention.sum()), "vision": int(image.sum())},
    )


@contextmanager
def disabled_peft_adapters(model):
    from peft.tuners.tuners_utils import BaseTunerLayer

    layers = [module for module in model.modules() if isinstance(module, BaseTunerLayer)]
    if not layers:
        raise TypeError("No PEFT tuner layers were found to disable")
    for layer in layers:
        layer.enable_adapters(False)
    try:
        yield
    finally:
        for layer in layers:
            layer.enable_adapters(True)


def extract_config(
    *,
    dataset: Path,
    checkpoint: Path,
    output_path: Path,
    config: ClipConfig,
    evaluation_fps: float,
    isaac_root: Path,
) -> dict[str, Any]:
    import pyarrow.parquet as pq
    import torch

    sys.path.insert(0, str(isaac_root))
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    table = pq.read_table(dataset / "data/chunk-000/file-000.parquet", columns=["observation.state"])
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    decoded = {
        name: decode_video(
            dataset / "videos" / key / "chunk-000/file-000.mp4",
            fps=10.0 if name == "external" else 1.0 / 76.7,
        )
        for name, key in VIDEO_KEYS.items()
    }
    policy = Gr00tPolicy("NEW_EMBODIMENT", str(checkpoint), device="cuda:0")
    model = policy.model
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    sample_times = np.arange(0.0, TERMINAL_SECONDS + 1e-9, 1.0 / evaluation_fps)
    hidden_means = []
    vision_means = []
    latencies = []
    token_counts = []
    torch.cuda.reset_peak_memory_stats(device)
    for end_seconds in sample_times:
        clip_times = trailing_clip_times(float(end_seconds), config)
        frame_indices = np.clip(np.rint(clip_times * 10.0).astype(np.int64), 0, len(decoded["external"]) - 1)
        state_indices = np.clip(np.rint(clip_times * 30.0).astype(np.int64), 0, len(states) - 1)
        external = decoded["external"][frame_indices]
        left = np.repeat(decoded["left_wrist"][:1], len(clip_times), axis=0)
        right = np.repeat(decoded["right_wrist"][:1], len(clip_times), axis=0)
        clip_states = states[state_indices]
        observation = {
            "video.external": external[None],
            "video.left_wrist": left[None],
            "video.right_wrist": right[None],
            "state.left_arm": clip_states[None, :, :7],
            "state.left_gripper": clip_states[None, :, 7:8],
            "state.right_arm": clip_states[None, :, 8:15],
            "state.right_gripper": clip_states[None, :, 15:16],
            "annotation.human.task_description": [TASK],
        }
        processed = policy.processor.process_observation(observation, policy.embodiment_tag)
        backbone_input = _to_device(model.backbone.prepare_input(processed), device, dtype)
        started = time.perf_counter()
        with torch.inference_mode(), disabled_peft_adapters(model):
            output = model.backbone(backbone_input)
        torch.cuda.synchronize(device)
        latencies.append((time.perf_counter() - started) * 1000)
        hidden, vision, counts = _pool_backbone(output)
        hidden_means.append(hidden)
        vision_means.append(vision)
        token_counts.append(counts)
    stage, terminal = stage_labels(sample_times)
    peak_vram_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        hidden_mean=np.asarray(hidden_means, dtype=np.float32),
        vision_mean=np.asarray(vision_means, dtype=np.float32),
        timestamp=sample_times.astype(np.float32),
        stage=stage,
        terminal=terminal,
        backbone_latency_ms=np.asarray(latencies, dtype=np.float32),
        peak_vram_mib=np.asarray([peak_vram_mib], dtype=np.float32),
    )
    metadata = {
        "config": config.slug,
        "seconds": config.seconds,
        "fps": config.fps,
        "frames": config.frame_count,
        "evaluation_fps": evaluation_fps,
        "adapter_enabled": False,
        "action_head_used": False,
        "external_camera_dynamic": True,
        "wrist_cameras_dynamic": False,
        "mean_latency_ms": float(np.mean(latencies)),
        "peak_vram_mib": float(peak_vram_mib),
        "token_counts": token_counts[0],
        "output": str(output_path),
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def extract_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--evaluation-fps", type=float, default=1.0)
    parser.add_argument("--isaac-root", type=Path, default=Path("/root/projects/Isaac-GR00T"))
    args = parser.parse_args()
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    config = ClipConfig(args.seconds, args.fps)
    result = extract_config(
        dataset=args.dataset,
        checkpoint=args.checkpoint,
        output_path=args.output_dir / f"{config.slug}.npz",
        config=config,
        evaluation_fps=args.evaluation_fps,
        isaac_root=args.isaac_root,
    )
    print(json.dumps(result, ensure_ascii=False))


def evaluate_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("features_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [evaluate_features(path) for path in sorted(args.features_dir.glob("*.npz"))]
    if not rows:
        raise FileNotFoundError(f"No feature archives under {args.features_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_only_fields = {"predicted_boundaries_s", "misclassified_samples"}
    fields = sorted({key for row in rows for key in row if key not in json_only_fields})
    with (args.output_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if key in fields} for row in rows)
    (args.output_dir / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=True) + "\n"
    )
    print(json.dumps(rows, ensure_ascii=False, allow_nan=True))


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"extract", "evaluate"}:
        raise SystemExit("usage: sciedu_cosmos.py {extract|evaluate} [options]")
    command = sys.argv.pop(1)
    extract_main() if command == "extract" else evaluate_main()


if __name__ == "__main__":
    main()
