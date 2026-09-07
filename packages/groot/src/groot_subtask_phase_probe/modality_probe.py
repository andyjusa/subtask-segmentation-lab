from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

from .evaluate import (
    BOUNDARY_THRESHOLDS,
    C_GRID,
    SEEDS,
    Episode,
    Transform,
    _boundary_labels,
    _event_metrics,
    load_episodes,
    split_episode_ids,
)

MODALITY_REPRESENTATIONS = (
    "R1_action_chunk",
    "R3_vlm_hidden",
    "R3b_vision_tokens",
    "R3c_instruction_tokens",
    "R5_final_dit_hidden",
    "F1_vision_instruction",
    "F2_vision_action_output",
    "F3_instruction_action_output",
    "F4_vision_instruction_action_output",
    "F5_vision_action_hidden",
    "F6_instruction_action_hidden",
    "F7_vision_instruction_action_hidden",
)


def derive_instruction_mean(
    joint_mean: np.ndarray,
    vision_mean: np.ndarray,
    *,
    valid_tokens: int,
    vision_tokens: int,
) -> np.ndarray:
    """Recover a close non-image mean approximation from pooled BF16 summaries.

    Directly captured instruction tokens are preferred for final measurements;
    independently rounded pooled means can introduce small reconstruction error.
    """

    instruction_tokens = valid_tokens - vision_tokens
    if valid_tokens <= 0 or vision_tokens <= 0 or instruction_tokens <= 0:
        raise ValueError(
            "Token counts must contain positive vision and instruction subsets: "
            f"valid={valid_tokens}, vision={vision_tokens}"
        )
    joint = np.asarray(joint_mean, dtype=np.float64)
    vision = np.asarray(vision_mean, dtype=np.float64)
    if joint.shape != vision.shape:
        raise ValueError(f"Joint and vision shapes differ: {joint.shape} != {vision.shape}")
    recovered = (joint * valid_tokens - vision * vision_tokens) / instruction_tokens
    return recovered.astype(np.float32)


def load_token_calibration(path: Path) -> dict[str, dict[str, int]]:
    payload = json.loads(path.read_text())
    tasks = payload.get("tasks", payload)
    calibration: dict[str, dict[str, int]] = {}
    for task, counts in tasks.items():
        normalized = {name: int(counts[name]) for name in ("valid", "vision", "instruction")}
        if normalized["valid"] != normalized["vision"] + normalized["instruction"]:
            raise ValueError(f"Invalid token partition for {task}: {normalized}")
        calibration[str(task)] = normalized
    return calibration


def calibration_from_artifacts(root: Path) -> dict[str, dict[str, int]]:
    calibration: dict[str, dict[str, int]] = {}
    for metadata_path in sorted(root.glob("*/episode_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        counts = metadata.get("backbone_token_counts")
        if not counts:
            continue
        task = str(metadata["task"])
        normalized = {name: int(counts[name]) for name in ("valid", "vision", "instruction")}
        previous = calibration.setdefault(task, normalized)
        if previous != normalized:
            raise ValueError(
                f"Backbone token counts changed within task {task}: {previous} != {normalized}"
            )
    if not calibration:
        raise ValueError(f"No backbone_token_counts found under {root}")
    return calibration


def add_modality_representations(
    episode: Episode,
    token_calibration: dict[str, dict[str, int]],
) -> None:
    features = episode.features
    if "R3c_instruction_tokens" not in features:
        try:
            counts = token_calibration[episode.task]
        except KeyError as exc:
            raise KeyError(f"Missing token calibration for task {episode.task}") from exc
        features["R3c_instruction_tokens"] = derive_instruction_mean(
            features["R3_vlm_hidden"],
            features["R3b_vision_tokens"],
            valid_tokens=counts["valid"],
            vision_tokens=counts["vision"],
        )

    vision = features["R3b_vision_tokens"]
    instruction = features["R3c_instruction_tokens"]
    action_output = features["R1_action_chunk"]
    action_hidden = features["R5_final_dit_hidden"]
    features["F1_vision_instruction"] = np.concatenate([vision, instruction], axis=1)
    features["F2_vision_action_output"] = np.concatenate([vision, action_output], axis=1)
    features["F3_instruction_action_output"] = np.concatenate(
        [instruction, action_output], axis=1
    )
    features["F4_vision_instruction_action_output"] = np.concatenate(
        [vision, instruction, action_output], axis=1
    )
    features["F5_vision_action_hidden"] = np.concatenate([vision, action_hidden], axis=1)
    features["F6_instruction_action_hidden"] = np.concatenate(
        [instruction, action_hidden], axis=1
    )
    features["F7_vision_instruction_action_hidden"] = np.concatenate(
        [vision, instruction, action_hidden], axis=1
    )


def _episode_map(episodes: Iterable[Episode]) -> dict[str, Episode]:
    return {episode.episode_id: episode for episode in episodes}


def _stage_xy(
    episodes: dict[str, Episode], ids: set[str], representation: str
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for episode_id in sorted(ids):
        episode = episodes[episode_id]
        valid = (episode.terminal == 0) & (episode.stage >= 0)
        xs.append(episode.features[representation][valid])
        ys.append(episode.stage[valid])
    return np.concatenate(xs), np.concatenate(ys)


def _boundary_xy(
    episodes: dict[str, Episode],
    ids: set[str],
    representation: str,
    transform: Transform,
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for episode_id in sorted(ids):
        episode = episodes[episode_id]
        valid = episode.terminal == 0
        z = transform.apply(episode.features[representation][valid])
        delta = np.vstack([np.zeros((1, z.shape[1]), dtype=z.dtype), np.diff(z, axis=0)])
        xs.append(np.concatenate([z, delta], axis=1))
        ys.append(_boundary_labels(episode)[valid])
    return np.concatenate(xs), np.concatenate(ys)


def _fit_classifier(kind: str, x: np.ndarray, y: np.ndarray, value: float, seed: int):
    if len(np.unique(y)) < 2:
        raise ValueError("Probe train split contains only one class")
    if kind == "linear":
        return LogisticRegression(
            C=value,
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
            random_state=seed,
        ).fit(x, y)
    if kind == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=value,
            batch_size=min(128, len(x)),
            learning_rate_init=1e-3,
            max_iter=10,
            tol=1e-3,
            early_stopping=False,
            random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            return model.fit(x, y, sample_weight=compute_sample_weight("balanced", y))
    raise ValueError(f"Unknown classifier kind: {kind}")


def _hyperparameters(kind: str) -> tuple[float, ...]:
    # A fixed MLP regularizer keeps representation comparisons capacity-matched
    # without multiplying every stage/boundary run by a tuning grid.
    return C_GRID if kind == "linear" else (1e-3,)


def _boundary_probabilities(
    model,
    transform: Transform,
    episodes: dict[str, Episode],
    ids: set[str],
    representation: str,
) -> dict[str, np.ndarray]:
    result = {}
    for episode_id in ids:
        episode = episodes[episode_id]
        valid = episode.terminal == 0
        z = transform.apply(episode.features[representation][valid])
        delta = np.vstack([np.zeros((1, z.shape[1]), dtype=z.dtype), np.diff(z, axis=0)])
        result[episode_id] = model.predict_proba(np.concatenate([z, delta], axis=1))[:, 1]
    return result


def evaluate_one(
    episodes_list: list[Episode],
    representation: str,
    classifier: str,
    common_256: bool,
    seed: int,
    split_ids: tuple[set[str], set[str], set[str]] | None = None,
) -> dict[str, float | int | str]:
    episodes = _episode_map(episodes_list)
    train_ids, validation_ids, test_ids = (
        split_episode_ids(episodes_list, seed) if split_ids is None else split_ids
    )
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise RuntimeError("Episode split leakage detected")
    train_x, train_y = _stage_xy(episodes, train_ids, representation)
    val_x, val_y = _stage_xy(episodes, validation_ids, representation)
    test_x, test_y = _stage_xy(episodes, test_ids, representation)
    transform = Transform(common_256, seed).fit(train_x)

    best_stage = None
    for value in _hyperparameters(classifier):
        model = _fit_classifier(classifier, transform.apply(train_x), train_y, value, seed)
        score = f1_score(val_y, model.predict(transform.apply(val_x)), average="macro")
        if best_stage is None or score > best_stage[0]:
            best_stage = (score, value, model)
    assert best_stage is not None
    stage_prediction = best_stage[2].predict(transform.apply(test_x))

    boundary_train_x, boundary_train_y = _boundary_xy(
        episodes, train_ids, representation, transform
    )
    best_boundary = None
    for value in _hyperparameters(classifier):
        model = _fit_classifier(classifier, boundary_train_x, boundary_train_y, value, seed)
        probabilities = _boundary_probabilities(
            model, transform, episodes, validation_ids, representation
        )
        for threshold in BOUNDARY_THRESHOLDS:
            metrics = _event_metrics(episodes, validation_ids, probabilities, float(threshold))
            score = (
                metrics["boundary_event_f1_at_5"],
                -metrics["false_boundaries_per_episode"],
                -abs(float(threshold) - 0.5),
            )
            if best_boundary is None or score > best_boundary[0]:
                best_boundary = (score, value, float(threshold), model)
    assert best_boundary is not None
    test_probabilities = _boundary_probabilities(
        best_boundary[3], transform, episodes, test_ids, representation
    )
    boundary_metrics = _event_metrics(
        episodes, test_ids, test_probabilities, best_boundary[2]
    )

    boundary_test_x, _ = _boundary_xy(episodes, test_ids, representation, transform)
    repeats = max(10, min(100, 10_000 // max(1, len(boundary_test_x))))
    started = perf_counter()
    for _ in range(repeats):
        best_boundary[3].predict_proba(boundary_test_x)
    latency = (perf_counter() - started) * 1000 / (repeats * len(boundary_test_x))
    return {
        "representation": representation,
        "classifier": classifier,
        "projection": "common_256" if common_256 else "native",
        "seed": seed,
        "native_dim": int(train_x.shape[1]),
        "probe_dim": int(transform.apply(train_x[:1]).shape[1]),
        "stage_hyperparameter": float(best_stage[1]),
        "boundary_hyperparameter": float(best_boundary[1]),
        "boundary_threshold": float(best_boundary[2]),
        "stage_macro_f1": float(f1_score(test_y, stage_prediction, average="macro")),
        "stage_balanced_accuracy": float(balanced_accuracy_score(test_y, stage_prediction)),
        "boundary_event_f1_at_5": float(boundary_metrics["boundary_event_f1_at_5"]),
        "boundary_event_f1_at_10": float(boundary_metrics["boundary_event_f1_at_10"]),
        "boundary_median_abs_error": float(boundary_metrics["boundary_median_abs_error"]),
        "false_boundaries_per_episode": float(
            boundary_metrics["false_boundaries_per_episode"]
        ),
        "no_boundary_failure_fpr": float(boundary_metrics["no_boundary_failure_fpr"]),
        "probe_latency_ms_per_sample": float(latency),
        "episode_split_leakage": 0,
    }


def _mean_ci95(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    valid = array[~np.isnan(array)]
    mean = float(valid.mean()) if valid.size else float("nan")
    if valid.size < 2:
        return mean, float("nan")
    return mean, float(1.96 * valid.std(ddof=1) / np.sqrt(valid.size))


def write_outputs(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault(
            (row["representation"], row["classifier"], row["projection"]), []
        ).append(row)
    summary = []
    for (representation, classifier, projection), values in grouped.items():
        stage, stage_ci = _mean_ci95(value["stage_macro_f1"] for value in values)
        boundary, boundary_ci = _mean_ci95(
            value["boundary_event_f1_at_5"] for value in values
        )
        summary.append(
            {
                "representation": representation,
                "classifier": classifier,
                "projection": projection,
                "stage_macro_f1": stage,
                "stage_macro_f1_ci95": stage_ci,
                "boundary_event_f1_at_5": boundary,
                "boundary_event_f1_at_5_ci95": boundary_ci,
                "native_dim": values[0]["native_dim"],
                "probe_dim": values[0]["probe_dim"],
                "probe_latency_ms_per_sample": float(
                    np.mean([value["probe_latency_ms_per_sample"] for value in values])
                ),
            }
        )
    summary.sort(
        key=lambda row: (-row["boundary_event_f1_at_5"], -row["stage_macro_f1"])
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    lines = [
        "# GR00T modality separation and fusion probe",
        "",
        "동일 episode split과 seed에서 vision, instruction, action 단독 및 concat fusion을 비교했다.",
        "",
        "| 표현 | 분류기 | 투영 | Stage F1 | Boundary F1@±5 | 차원 | Probe ms |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['representation']} | {row['classifier']} | {row['projection']} | "
            f"{row['stage_macro_f1']:.3f} ± {row['stage_macro_f1_ci95']:.3f} | "
            f"{row['boundary_event_f1_at_5']:.3f} ± "
            f"{row['boundary_event_f1_at_5_ci95']:.3f} | {row['probe_dim']} | "
            f"{row['probe_latency_ms_per_sample']:.6f} |"
        )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    if args.write_calibration_only:
        calibration = calibration_from_artifacts(Path(args.artifacts))
        Path(args.calibration_output).write_text(
            json.dumps({"tasks": calibration}, ensure_ascii=False, indent=2) + "\n"
        )
        return

    calibration = load_token_calibration(Path(args.calibration)) if args.calibration else {}
    episodes = load_episodes(Path(args.artifacts))
    for episode in episodes:
        add_modality_representations(episode, calibration)
    requested = tuple(args.representations or MODALITY_REPRESENTATIONS)
    unavailable = sorted(set(requested) - set.intersection(*(set(ep.features) for ep in episodes)))
    if unavailable:
        raise KeyError(f"Representations unavailable: {unavailable}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    rows = []
    if args.resume and progress_path.exists():
        rows = [json.loads(line) for line in progress_path.read_text().splitlines() if line]
    completed = {
        (row["representation"], row["classifier"], row["projection"], int(row["seed"]))
        for row in rows
    }
    jobs = [
        (representation, classifier, projection, seed)
        for representation in requested
        for classifier in args.classifiers
        for projection in args.projections
        for seed in args.seeds
        if (representation, classifier, projection, seed) not in completed
    ]

    def evaluate(job):
        representation, classifier, projection, seed = job
        return evaluate_one(
            episodes,
            representation,
            classifier,
            projection == "common_256",
            seed,
        )

    mode = "a" if args.resume else "w"
    with progress_path.open(mode, buffering=1) as progress, ThreadPoolExecutor(
        max_workers=args.jobs
    ) as executor:
        futures = {executor.submit(evaluate, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            progress.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                row["representation"],
                row["classifier"],
                row["projection"],
                row["seed"],
                f"stage={row['stage_macro_f1']:.3f}",
                f"boundary={row['boundary_event_f1_at_5']:.3f}",
                flush=True,
            )
    write_outputs(rows, Path(args.output_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts")
    parser.add_argument("--calibration")
    parser.add_argument("--output-dir", default="reports/modality_probe")
    parser.add_argument("--representations", nargs="*")
    parser.add_argument("--classifiers", nargs="+", choices=("linear", "mlp"), default=("linear", "mlp"))
    parser.add_argument("--projections", nargs="+", choices=("native", "common_256"), default=("native", "common_256"))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--write-calibration-only", action="store_true")
    parser.add_argument("--calibration-output", default="reports/modality_token_counts.json")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
