from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .evaluate import SEEDS, Episode


@dataclass(frozen=True)
class TwoBoundarySource:
    episode_id: str
    pair_id: int
    color_mode: str
    color_change_enabled: bool
    features: dict[str, np.ndarray]
    sample_index: np.ndarray
    stove_boundary_sample: int | None
    color_boundary_sample: int | None


def load_sources(root: Path) -> list[TwoBoundarySource]:
    sources: list[TwoBoundarySource] = []
    pattern = "fixed_color_two_boundary/*/episode_*/metadata.json"
    for metadata_path in sorted(root.glob(pattern)):
        metadata = json.loads(metadata_path.read_text())
        with np.load(metadata_path.with_name("features.npz")) as archive:
            sources.append(
                TwoBoundarySource(
                    episode_id=str(metadata_path.parent.relative_to(root)),
                    pair_id=int(metadata["pair_id"]),
                    color_mode=str(metadata["color_mode"]),
                    color_change_enabled=bool(metadata["color_change_enabled"]),
                    features={
                        name: archive[name].astype(np.float32)
                        for name in archive.files
                        if name.startswith("R")
                    },
                    sample_index=archive["sample_index"].astype(np.int64),
                    stove_boundary_sample=metadata["stove_boundary_sample"],
                    color_boundary_sample=metadata["color_boundary_sample"],
                )
            )
    if not sources:
        raise FileNotFoundError(f"No two-boundary artifacts found under {root}")
    return sources


def paired_stratified_split(
    sources: Sequence[TwoBoundarySource], seed: int
) -> tuple[set[int], set[int], set[int]]:
    by_pair: dict[int, bool] = {}
    for source in sources:
        prior = by_pair.setdefault(source.pair_id, source.color_change_enabled)
        if prior != source.color_change_enabled:
            raise ValueError(f"Pair {source.pair_id} has inconsistent change labels")
    rng = np.random.default_rng(seed)
    split: list[set[int]] = [set(), set(), set()]
    for condition in (False, True):
        pair_ids = np.asarray([pair_id for pair_id, changed in by_pair.items() if changed == condition])
        if len(pair_ids) < 3:
            raise ValueError("Each change stratum needs at least three pairs")
        rng.shuffle(pair_ids)
        n_train = max(1, math.floor(0.6 * len(pair_ids)))
        n_validation = max(1, math.floor(0.2 * len(pair_ids)))
        if n_train + n_validation >= len(pair_ids):
            n_train = len(pair_ids) - 2
            n_validation = 1
        split[0].update(int(value) for value in pair_ids[:n_train])
        split[1].update(int(value) for value in pair_ids[n_train : n_train + n_validation])
        split[2].update(int(value) for value in pair_ids[n_train + n_validation :])
    return split[0], split[1], split[2]


def as_target_episode(source: TwoBoundarySource, target: str) -> Episode:
    if target not in {"stove", "color"}:
        raise ValueError(f"Unknown target: {target}")
    boundary = (
        source.stove_boundary_sample if target == "stove" else source.color_boundary_sample
    )
    stage = np.zeros(len(source.sample_index), dtype=np.int64)
    if boundary is not None:
        stage[source.sample_index >= boundary] = 1
    success = boundary is not None
    return Episode(
        episode_id=source.episode_id,
        task=f"{source.color_mode}_{target}",
        features=source.features,
        stage=stage,
        terminal=np.zeros(len(stage), dtype=np.int64),
        env_step=source.sample_index,
        boundary_step=boundary,
        success=success,
    )


def first_event(probabilities: np.ndarray, threshold: float) -> tuple[int | None, int]:
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    above = values >= threshold
    rising = np.flatnonzero(above & np.concatenate(([True], ~above[:-1])))
    return (int(rising[0]) if len(rising) else None, len(rising))


def event_metrics(
    ground_truth: Mapping[str, int | None],
    predictions: Mapping[str, tuple[int | None, int]],
    *,
    tolerance: int,
) -> dict[str, float | int]:
    tp = fp = fn = 0
    errors: list[int] = []
    negative_count = negative_fp = 0
    false_events = 0
    for episode_id, target in ground_truth.items():
        predicted, event_count = predictions[episode_id]
        if target is None:
            negative_count += 1
            negative_fp += int(predicted is not None)
            fp += event_count
            false_events += event_count
        elif predicted is None:
            fn += 1
        else:
            error = abs(predicted - target)
            errors.append(error)
            if error <= tolerance:
                tp += 1
                fp += max(0, event_count - 1)
                false_events += max(0, event_count - 1)
            else:
                fp += event_count
                fn += 1
                false_events += event_count
    denominator = 2 * tp + fp + fn
    return {
        "f1": 0.0 if denominator == 0 else 2 * tp / denominator,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "median_abs_error_samples": float(np.median(errors)) if errors else float("nan"),
        "negative_fpr": negative_fp / negative_count if negative_count else float("nan"),
        "false_events_per_episode": false_events / max(1, len(ground_truth)),
    }


def ordered_accuracy(
    sources: Sequence[TwoBoundarySource],
    stove_predictions: Mapping[str, tuple[int | None, int]],
    color_predictions: Mapping[str, tuple[int | None, int]],
    *,
    tolerance: int,
) -> dict[str, float | int]:
    correct = positive_correct = 0
    positives = 0
    for source in sources:
        stove_prediction, stove_events = stove_predictions[source.episode_id]
        color_prediction, color_events = color_predictions[source.episode_id]
        stove_ok = (
            source.stove_boundary_sample is not None
            and stove_prediction is not None
            and abs(stove_prediction - source.stove_boundary_sample) <= tolerance
            and stove_events == 1
        )
        if source.color_boundary_sample is None:
            color_ok = color_prediction is None and color_events == 0
        else:
            positives += 1
            color_ok = (
                color_prediction is not None
                and abs(color_prediction - source.color_boundary_sample) <= tolerance
                and color_events == 1
                and stove_prediction is not None
                and stove_prediction < color_prediction
            )
            positive_correct += int(stove_ok and color_ok)
        correct += int(stove_ok and color_ok)
    return {
        "ordered_episode_accuracy": correct / max(1, len(sources)),
        "ordered_positive_accuracy": positive_correct / max(1, positives),
        "ordered_correct": correct,
        "episodes": len(sources),
        "positive_episodes": positives,
    }


def _mean_ci95(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.nanmean(array))
    if len(array) < 2:
        return mean, 0.0
    return mean, float(1.96 * np.nanstd(array, ddof=1) / np.sqrt(len(array)))


class _CheckpointTransform:
    def __init__(self, checkpoint: Mapping[str, Any]):
        self.mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
        self.scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
        self.components = (
            np.asarray(checkpoint["pca_components"], dtype=np.float32)
            if "pca_components" in checkpoint
            else None
        )
        self.pca_mean = (
            np.asarray(checkpoint["pca_mean"], dtype=np.float32)
            if "pca_mean" in checkpoint
            else None
        )
        self.output_dim = int(checkpoint.get("pca_output_dim", len(self.mean)))

    def apply(self, values: np.ndarray) -> np.ndarray:
        result = (np.asarray(values, dtype=np.float32) - self.mean) / self.scale
        if self.components is not None and self.pca_mean is not None:
            result = (result - self.pca_mean) @ self.components.T
            if result.shape[1] < self.output_dim:
                result = np.pad(result, ((0, 0), (0, self.output_dim - result.shape[1])))
        return result.astype(np.float32, copy=False)


def _predict_checkpoint(
    checkpoint_path: Path,
    episodes: Sequence[Episode],
) -> tuple[dict[str, tuple[int | None, int]], float]:
    import torch

    from .streaming_transformer import (
        CausalSlidingWindowTransformer,
        predict_streaming_sequence,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = CausalSlidingWindowTransformer(
        input_dim=int(checkpoint["input_dim"]), **checkpoint["model_config"]
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    transform = _CheckpointTransform(checkpoint)
    threshold = float(checkpoint["boundary_threshold"])
    predictions: dict[str, tuple[int | None, int]] = {}
    latencies = []
    for episode in episodes:
        valid = (episode.terminal == 0) & (episode.stage >= 0)
        values = transform.apply(episode.features[str(checkpoint["representation"])][valid])
        _, probabilities, latency = predict_streaming_sequence(
            model, values, device=device, measure_latency=True
        )
        predictions[episode.episode_id] = first_event(probabilities, threshold)
        latencies.append(latency)
    return predictions, float(np.mean(latencies))


def _train_target(
    episodes: list[Episode],
    *,
    split_ids: tuple[set[str], set[str], set[str]],
    target: str,
    color_mode: str,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    import torch

    from .streaming_transformer import TransformerConfig, train_one

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return train_one(
        episodes,
        representation="R3b_vision_tokens",
        common_256=True,
        seed=seed,
        config=TransformerConfig(),
        device=device,
        output_dir=output_dir / color_mode / target / f"seed_{seed}",
        epochs=30,
        batch_size=64,
        learning_rate=3e-4,
        weight_decay=1e-4,
        boundary_loss_weight=1.0,
        patience=6,
        backbone_select_layer=-1,
        split_ids=split_ids,
        stage_loss_weight=0.0,
        boundary_radius=0,
        include_input_delta=False,
        label_source=f"two-boundary {target} exact sample",
    )


def run(args: argparse.Namespace) -> None:
    sources = load_sources(Path(args.artifacts))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for color_mode in ("fixed", "random"):
        mode_sources = [source for source in sources if source.color_mode == color_mode]
        for seed in args.seeds:
            train_pairs, validation_pairs, test_pairs = paired_stratified_split(mode_sources, seed)
            split_pairs = (train_pairs, validation_pairs, test_pairs)
            split_ids = tuple(
                {
                    source.episode_id
                    for source in mode_sources
                    if source.pair_id in pair_ids
                }
                for pair_ids in split_pairs
            )
            target_episodes = {
                target: [as_target_episode(source, target) for source in mode_sources]
                for target in ("stove", "color")
            }
            trained = {
                target: _train_target(
                    target_episodes[target],
                    split_ids=split_ids,
                    target=target,
                    color_mode=color_mode,
                    seed=seed,
                    output_dir=output_dir,
                )
                for target in ("stove", "color")
            }
            test_sources = [source for source in mode_sources if source.pair_id in test_pairs]
            predictions = {}
            latencies = {}
            for target in ("stove", "color"):
                checkpoint_path = Path(str(trained[target]["checkpoint"]))
                target_test = [as_target_episode(source, target) for source in test_sources]
                predictions[target], latencies[target] = _predict_checkpoint(
                    checkpoint_path, target_test
                )
            metrics = {}
            for target in ("stove", "color"):
                ground_truth = {
                    source.episode_id: (
                        source.stove_boundary_sample
                        if target == "stove"
                        else source.color_boundary_sample
                    )
                    for source in test_sources
                }
                for tolerance in (1, 2):
                    for key, value in event_metrics(
                        ground_truth, predictions[target], tolerance=tolerance
                    ).items():
                        metrics[f"{target}_{key}_at_{tolerance}"] = value
            for tolerance in (1, 2):
                for key, value in ordered_accuracy(
                    test_sources,
                    predictions["stove"],
                    predictions["color"],
                    tolerance=tolerance,
                ).items():
                    metrics[f"{key}_at_{tolerance}"] = value
            row = {
                "color_mode": color_mode,
                "seed": seed,
                "train_pair_ids": sorted(train_pairs),
                "validation_pair_ids": sorted(validation_pairs),
                "test_pair_ids": sorted(test_pairs),
                "stove_checkpoint": trained["stove"]["checkpoint"],
                "color_checkpoint": trained["color"]["checkpoint"],
                "probe_latency_ms_per_sample": latencies["stove"] + latencies["color"],
                "parameter_count": int(trained["stove"]["parameter_count"])
                + int(trained["color"]["parameter_count"]),
                "stove_predictions": predictions["stove"],
                "color_predictions": predictions["color"],
                **metrics,
            }
            rows.append(row)
            print(
                f"mode={color_mode} seed={seed} "
                f"ordered={row['ordered_episode_accuracy_at_1']:.3f} "
                f"stove_f1={row['stove_f1_at_1']:.3f} color_f1={row['color_f1_at_1']:.3f}"
            )

    summary = []
    metric_names = [
        "ordered_episode_accuracy_at_1",
        "ordered_positive_accuracy_at_1",
        "stove_f1_at_1",
        "color_f1_at_1",
        "color_negative_fpr_at_1",
        "stove_median_abs_error_samples_at_1",
        "color_median_abs_error_samples_at_1",
        "probe_latency_ms_per_sample",
    ]
    for color_mode in ("fixed", "random"):
        selected = [row for row in rows if row["color_mode"] == color_mode]
        item: dict[str, object] = {"color_mode": color_mode, "seeds": len(selected)}
        for metric in metric_names:
            mean, ci95 = _mean_ci95([float(row[metric]) for row in selected])
            item[f"{metric}_mean"] = mean
            item[f"{metric}_ci95"] = ci95
        summary.append(item)

    (output_dir / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    report = [
        "# GR00T 고정색 vs 무작위색 2경계 C2 실험",
        "",
        "| 조건 | 순서 포함 정확도@±1 | Stove F1@±1 | Color F1@±1 | Color no-change FPR |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summary:
        report.append(
            f"| {item['color_mode']} | "
            f"{item['ordered_episode_accuracy_at_1_mean']:.3f} ± "
            f"{item['ordered_episode_accuracy_at_1_ci95']:.3f} | "
            f"{item['stove_f1_at_1_mean']:.3f} ± {item['stove_f1_at_1_ci95']:.3f} | "
            f"{item['color_f1_at_1_mean']:.3f} ± {item['color_f1_at_1_ci95']:.3f} | "
            f"{item['color_negative_fpr_at_1_mean']:.3f} |"
        )
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts")
    parser.add_argument("--output-dir", default="reports/fixed_color_two_boundary")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
