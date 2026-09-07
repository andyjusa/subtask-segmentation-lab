from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--boundaries", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--train-stride", type=int, default=3)
    return parser.parse_args()


def stage_block_folds(labels: np.ndarray, folds: int) -> np.ndarray:
    assignments = np.full(len(labels), -1, dtype=np.int32)
    for stage in np.unique(labels):
        indices = np.flatnonzero(labels == stage)
        for fold, block in enumerate(np.array_split(indices, folds)):
            assignments[block] = fold
    if (assignments < 0).any():
        raise RuntimeError("Some frames were not assigned to a fold")
    return assignments


def make_model(name: str):
    if name == "linear":
        return LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-3,
            batch_size=128,
            learning_rate_init=1e-3,
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
        )
    raise ValueError(name)


def cross_validated_probabilities(
    features: np.ndarray,
    labels: np.ndarray,
    assignments: np.ndarray,
    model_name: str,
    train_stride: int,
) -> np.ndarray:
    classes = np.unique(labels)
    probabilities = np.zeros((len(labels), len(classes)), dtype=np.float32)
    for fold in np.unique(assignments):
        test = np.flatnonzero(assignments == fold)
        train = np.flatnonzero(assignments != fold)[::train_stride]
        scaler = StandardScaler().fit(features[train])
        train_features = np.clip(scaler.transform(features[train]), -8.0, 8.0)
        test_features = np.clip(scaler.transform(features[test]), -8.0, 8.0)
        model = make_model(model_name)
        if model_name == "mlp":
            weights = compute_sample_weight(class_weight="balanced", y=labels[train])
            model.fit(train_features, labels[train], sample_weight=weights)
        else:
            model.fit(train_features, labels[train])
        fold_probabilities = model.predict_proba(test_features)
        for source_column, stage in enumerate(model.classes_):
            probabilities[test, int(stage)] = fold_probabilities[:, source_column]
    return probabilities


def ordered_decode(
    probabilities: np.ndarray, fps: float, min_segment_s: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    stage_count = probabilities.shape[1]
    sample_step = max(1, round(fps / 5.0))
    sampled = probabilities[::sample_step]
    sampled = gaussian_filter1d(sampled, sigma=2.0, axis=0)
    sampled = np.clip(sampled, 1e-7, 1.0)
    log_probabilities = np.log(sampled)
    cumulative = np.vstack(
        [np.zeros((1, stage_count)), log_probabilities.cumsum(axis=0)]
    )
    count = len(sampled)
    min_length = max(1, round(min_segment_s * fps / sample_step))
    negative_infinity = -1e30
    dp = np.full((stage_count, count + 1), negative_infinity, dtype=np.float64)
    previous = np.full((stage_count, count + 1), -1, dtype=np.int32)
    for end in range(min_length, count + 1):
        dp[0, end] = cumulative[end, 0]
    for stage in range(1, stage_count):
        earliest_end = (stage + 1) * min_length
        for end in range(earliest_end, count + 1):
            starts = np.arange(stage * min_length, end - min_length + 1)
            scores = dp[stage - 1, starts] + (
                cumulative[end, stage] - cumulative[starts, stage]
            )
            best_offset = int(np.argmax(scores))
            dp[stage, end] = scores[best_offset]
            previous[stage, end] = int(starts[best_offset])
    end = count
    boundaries_sampled = []
    for stage in range(stage_count - 1, 0, -1):
        start = int(previous[stage, end])
        if start < 0:
            raise RuntimeError("Ordered stage decoding failed")
        boundaries_sampled.append(start)
        end = start
    boundaries = np.asarray(boundaries_sampled[::-1], dtype=np.int32) * sample_step
    decoded = np.searchsorted(boundaries, np.arange(len(probabilities)), side="right")
    return decoded.astype(np.int32), boundaries


def summarize(
    labels: np.ndarray,
    probabilities: np.ndarray,
    decoded: np.ndarray,
    boundaries: np.ndarray,
    reference_frames: np.ndarray,
    fps: float,
) -> dict:
    frame_predictions = probabilities.argmax(axis=1)
    errors = np.abs(boundaries - reference_frames) / fps
    return {
        "frame_argmax_macro_f1": float(
            f1_score(labels, frame_predictions, average="macro")
        ),
        "frame_argmax_balanced_accuracy": float(
            balanced_accuracy_score(labels, frame_predictions)
        ),
        "ordered_macro_f1": float(f1_score(labels, decoded, average="macro")),
        "ordered_balanced_accuracy": float(
            balanced_accuracy_score(labels, decoded)
        ),
        "ordered_confusion_matrix": confusion_matrix(labels, decoded).tolist(),
        "predicted_boundaries_s": (boundaries / fps).tolist(),
        "boundary_absolute_errors_s": errors.tolist(),
        "boundary_mean_absolute_error_s": float(errors.mean()),
        "boundary_median_absolute_error_s": float(np.median(errors)),
        "boundary_within_1s": int((errors <= 1.0).sum()),
        "boundary_within_2s": int((errors <= 2.0).sum()),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.features) as data:
        features = data["features"].astype(np.float32)
        fps = float(data["fps"])
    reference_s = np.asarray(
        [float(value) for value in args.boundaries.split(",")], dtype=np.float32
    )
    reference_frames = np.round(reference_s * fps).astype(np.int32)
    labels = np.searchsorted(reference_frames, np.arange(len(features)), side="right")
    assignments = stage_block_folds(labels, args.folds)
    all_probabilities = {}
    all_decoded = {}
    all_boundaries = {}
    results = {}
    for model_name in ("linear", "mlp"):
        probabilities = cross_validated_probabilities(
            features,
            labels,
            assignments,
            model_name,
            args.train_stride,
        )
        decoded, boundaries = ordered_decode(probabilities, fps)
        all_probabilities[model_name] = probabilities
        all_decoded[model_name] = decoded
        all_boundaries[model_name] = boundaries
        results[model_name] = summarize(
            labels,
            probabilities,
            decoded,
            boundaries,
            reference_frames,
            fps,
        )
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        labels=labels,
        folds=assignments,
        linear_probabilities=all_probabilities["linear"],
        linear_decoded=all_decoded["linear"],
        linear_boundaries=all_boundaries["linear"],
        mlp_probabilities=all_probabilities["mlp"],
        mlp_decoded=all_decoded["mlp"],
        mlp_boundaries=all_boundaries["mlp"],
        fps=np.asarray(fps),
    )
    report = {
        "input": str(args.features),
        "feature_contract": "68D point-flow only; no RGB embedding, robot state, SAM3, or time index",
        "evaluation": (
            f"{args.folds}-fold within-stage contiguous-block out-of-fold predictions"
        ),
        "reference_boundaries_s": reference_s.tolist(),
        "models": results,
        "limitations": [
            "One episode only; this is not an episode-level generalization result.",
            "Labels are used to train the probes; only their inputs are point-only.",
            "Ordered decoding knows that five stages occur in a fixed order.",
        ],
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
