"""Read-only inference on saved GR00T features; no robot, API, or ordered decoding."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages/groot/scripts"))
from train_tapnextpp_subtask_probe import make_model, predict_probabilities

STAGES = [
    "place_weight_1",
    "place_weight_2",
    "place_weight_3",
    "place_weight_4",
    "final_pointing",
]


def prepare_features(path: Path, key: str, mean: np.ndarray, scale: np.ndarray):
    """Validate the input contract before passing data to PyTorch."""
    if key not in {"vision_mean", "hidden_mean"}:
        raise ValueError(f"unsupported hidden key: {key}")
    with np.load(path, allow_pickle=False) as data:
        missing = {key, "sample_frames"} - set(data.files)
        if missing:
            raise ValueError(f"feature archive missing keys: {sorted(missing)}")
        x = data[key]
        frames = data["sample_frames"]
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != 2048:
        raise ValueError(f"expected nonempty [N, 2048] features, got {x.shape}")
    if not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x).all():
        raise ValueError("features must be finite floating-point values")
    if frames.shape != (len(x),) or not np.issubdtype(frames.dtype, np.integer):
        raise ValueError("sample_frames must be a length-N integer vector")
    if np.any(frames < 0) or np.any(frames[1:] <= frames[:-1]):
        raise ValueError("sample_frames must be nonnegative and strictly increasing")
    if mean.shape != (2048,) or scale.shape != (2048,):
        raise ValueError("scaler mean and scale must each have shape [2048]")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("scaler must be finite with positive scale")
    normalized = (x.astype(np.float32) - mean) / scale
    if not np.isfinite(normalized).all():
        raise ValueError("normalization overflow; check features and scaler")
    info = {
        "input_shape": list(x.shape),
        "input_dtype": str(x.dtype),
        "clip_fraction": float(np.mean(np.abs(normalized) > 8)),
    }
    return np.clip(normalized, -8, 8).astype(np.float32), frames, info


def run(model_dir: Path, features: Path, model_name: str, batch_size: int = 256):
    if model_name not in {"linear", "mlp"} or batch_size < 1:
        raise ValueError("choose linear/mlp and a positive batch size")
    metadata = json.loads((model_dir / "results.json").read_text())
    key = metadata.get("hidden_key")
    if key not in {"vision_mean", "hidden_mean"}:
        raise ValueError("model directory is not an incline50 training output: hidden_key missing")
    with np.load(model_dir / "scaler.npz", allow_pickle=False) as scaler:
        x, frames, info = prepare_features(features, key, scaler["mean"], scaler["scale"])
    model = make_model(model_name, x.shape[1])
    checkpoint = model_dir / f"{model_name}.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    probabilities = predict_probabilities(model, x, torch.device("cpu"), batch_size)
    if not np.isfinite(probabilities).all():
        raise ValueError("model produced nonfinite probabilities; check checkpoint")
    labels = probabilities.argmax(axis=1)
    records = [
        {
            "sample_index": i,
            "source_frame": int(frame),
            "stage_id": int(label),
            "stage": STAGES[int(label)],
            "probabilities": row.tolist(),
        }
        for i, (frame, label, row) in enumerate(zip(frames, labels, probabilities))
    ]
    info.update(
        model=model_name,
        hidden_key=key,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        feature_sha256=hashlib.sha256(features.read_bytes()).hexdigest(),
        scope="saved-feature raw inference; not live GR00T or ordered decoding",
        transitions=[
            {"sample_index": int(i), "source_frame": int(frames[i]), "stage": STAGES[labels[i]]}
            for i in np.flatnonzero(np.diff(labels)) + 1
        ],
    )
    return records, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", choices=("linear", "mlp"), default="mlp")
    parser.add_argument(
        "--output", type=Path, required=True, help="new JSONL path; never overwrite"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--traceback", action="store_true", help="keep full exception for pdb/IDE")
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch-size and threads must be positive")
    if args.output.exists():
        parser.error("output already exists; choose a new JSONL path")
    try:
        torch.set_num_threads(args.threads)
        records, info = run(args.model_dir, args.features, args.model, args.batch_size)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as target:
            for record in records:
                target.write(json.dumps(record) + "\n")
        print(json.dumps(info, indent=2))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        if args.traceback:
            raise
        parser.exit(2, f"{type(exc).__name__}: {exc}\nUse --traceback for debugging.\n")


if __name__ == "__main__":
    main()
