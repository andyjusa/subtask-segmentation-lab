from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
SEEDS = (17, 29, 41, 53, 67)
BOUNDARY_THRESHOLDS = tuple(np.linspace(0.05, 0.95, 19))


@dataclass
class Episode:
    episode_id: str
    task: str
    features: dict[str, np.ndarray]
    stage: np.ndarray
    terminal: np.ndarray
    env_step: np.ndarray
    boundary_step: int | None
    success: bool
    mean_hook_latency_ms: float = float("nan")
    mean_inference_latency_ms: float = float("nan")
    peak_vram_bytes: int = 0
    host_rss_delta_bytes: int = 0


def load_episodes(root: Path) -> list[Episode]:
    episodes: list[Episode] = []
    for metadata_path in sorted(root.glob("*/episode_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        with np.load(metadata_path.with_name("features.npz")) as archive:
            feature_names = [name for name in archive.files if name.startswith("R")]
            episodes.append(
                Episode(
                    episode_id=str(metadata_path.parent.relative_to(root)),
                    task=str(metadata["task"]),
                    features={name: archive[name].astype(np.float32) for name in feature_names},
                    stage=archive["stage"].astype(np.int64),
                    terminal=archive["terminal"].astype(np.int64),
                    env_step=archive["env_step"].astype(np.int64),
                    boundary_step=metadata["exact_boundary_env_step"],
                    success=bool(metadata["success"]),
                    mean_hook_latency_ms=float(
                        np.mean(metadata.get("hook_latency_ms", [float("nan")]))
                    ),
                    mean_inference_latency_ms=float(
                        np.mean(metadata.get("inference_latency_ms", [float("nan")]))
                    ),
                    peak_vram_bytes=int(metadata.get("peak_vram_bytes", 0)),
                    host_rss_delta_bytes=int(metadata.get("host_rss_delta_bytes", 0)),
                )
            )
    if not episodes:
        raise FileNotFoundError(f"No episode artifacts found under {root}")
    return episodes


def split_episode_ids(episodes: list[Episode], seed: int) -> tuple[set[str], set[str], set[str]]:
    rng = np.random.default_rng(seed)
    train: set[str] = set()
    validation: set[str] = set()
    test: set[str] = set()
    for task in sorted({episode.task for episode in episodes}):
        ids = np.array([episode.episode_id for episode in episodes if episode.task == task])
        rng.shuffle(ids)
        n = len(ids)
        n_train = max(1, int(np.floor(0.6 * n)))
        n_validation = max(1, int(np.floor(0.2 * n)))
        if n_train + n_validation >= n:
            n_train = max(1, n - 2)
            n_validation = 1
        train.update(ids[:n_train])
        validation.update(ids[n_train : n_train + n_validation])
        test.update(ids[n_train + n_validation :])
    if not test:
        raise ValueError("At least 5 episodes per task are required for a 60/20/20 split")
    return train, validation, test


class Transform:
    def __init__(self, common_256: bool, seed: int):
        self.common_256 = common_256
        self.seed = seed
        self.scaler = StandardScaler()
        self.pca: PCA | None = None

    def fit(self, x: np.ndarray) -> Transform:
        scaled = self.scaler.fit_transform(x)
        if self.common_256:
            n_components = min(256, scaled.shape[0] - 1, scaled.shape[1])
            if n_components < 1:
                raise ValueError("Not enough train samples for PCA")
            self.pca = PCA(n_components=n_components, random_state=self.seed)
            self.pca.fit(scaled)
        return self

    def apply(self, x: np.ndarray) -> np.ndarray:
        result = self.scaler.transform(x)
        if self.pca is not None:
            result = self.pca.transform(result)
            if result.shape[1] < 256:
                result = np.pad(result, ((0, 0), (0, 256 - result.shape[1])))
        return result.astype(np.float32, copy=False)


def _episode_map(episodes: Iterable[Episode]) -> dict[str, Episode]:
    return {episode.episode_id: episode for episode in episodes}


def _stage_xy(
    episodes: dict[str, Episode], ids: set[str], representation: str
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for episode_id in sorted(ids):
        episode = episodes[episode_id]
        mask = (episode.terminal == 0) & (episode.stage >= 0)
        xs.append(episode.features[representation][mask])
        ys.append(episode.stage[mask])
    return np.concatenate(xs), np.concatenate(ys)


def _boundary_labels(episode: Episode) -> np.ndarray:
    labels = np.zeros(len(episode.env_step), dtype=np.int64)
    valid = episode.terminal == 0
    if episode.boundary_step is not None and valid.any():
        indices = np.flatnonzero(valid)
        nearest = indices[np.argmin(np.abs(episode.env_step[indices] - episode.boundary_step))]
        lo = max(indices[0], nearest - 2)
        hi = min(indices[-1], nearest + 2)
        labels[lo : hi + 1] = 1
    return labels


def _boundary_xy(
    episodes: dict[str, Episode], ids: set[str], representation: str
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for episode_id in sorted(ids):
        episode = episodes[episode_id]
        valid = episode.terminal == 0
        z = episode.features[representation][valid]
        delta = np.vstack([np.zeros((1, z.shape[1]), dtype=z.dtype), np.diff(z, axis=0)])
        xs.append(np.concatenate([z, delta], axis=1))
        ys.append(_boundary_labels(episode)[valid])
    return np.concatenate(xs), np.concatenate(ys)


def _fit_logistic(x: np.ndarray, y: np.ndarray, c: float) -> LogisticRegression:
    if len(np.unique(y)) < 2:
        raise ValueError("Probe train split contains only one class")
    return LogisticRegression(
        C=c,
        class_weight="balanced",
        max_iter=3000,
        solver="liblinear",
        random_state=0,
    ).fit(x, y)


def _event_metrics(
    episodes: dict[str, Episode],
    ids: set[str],
    probabilities: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, float]:
    errors: list[float] = []
    counts = {5: [0, 0, 0], 10: [0, 0, 0]}  # tp, fp, fn
    false_events = 0
    no_boundary_failures = 0
    no_boundary_failure_fp = 0
    for episode_id in sorted(ids):
        episode = episodes[episode_id]
        probs = probabilities[episode_id]
        above = probs >= threshold
        rising = np.flatnonzero(above & np.concatenate([[True], ~above[:-1]]))
        predicted_steps = episode.env_step[episode.terminal == 0][rising]
        false_events += len(predicted_steps)
        prediction = int(predicted_steps[0]) if len(predicted_steps) else None
        gt = episode.boundary_step
        if gt is None and not episode.success:
            no_boundary_failures += 1
            no_boundary_failure_fp += int(prediction is not None)
        if gt is not None and prediction is not None:
            error = abs(prediction - gt)
            errors.append(float(error))
        for tolerance in (5, 10):
            tp, fp, fn = counts[tolerance]
            if gt is None:
                fp += int(prediction is not None)
            elif prediction is None:
                fn += 1
            elif abs(prediction - gt) <= tolerance:
                tp += 1
            else:
                fp += 1
                fn += 1
            counts[tolerance] = [tp, fp, fn]
    result: dict[str, float] = {}
    for tolerance, (tp, fp, fn) in counts.items():
        denominator = 2 * tp + fp + fn
        result[f"boundary_event_f1_at_{tolerance}"] = (
            0.0 if denominator == 0 else 2 * tp / denominator
        )
    result["boundary_median_abs_error"] = float(np.median(errors)) if errors else float("nan")
    result["false_boundaries_per_episode"] = false_events / max(1, len(ids))
    result["no_boundary_failure_fpr"] = (
        no_boundary_failure_fp / no_boundary_failures if no_boundary_failures else float("nan")
    )
    return result


def _boundary_probabilities(
    model: LogisticRegression,
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
    episodes_list: list[Episode], representation: str, common_256: bool, seed: int
) -> dict[str, float | int | str]:
    episodes = _episode_map(episodes_list)
    train_ids, validation_ids, test_ids = split_episode_ids(episodes_list, seed)
    stage_train_x, stage_train_y = _stage_xy(episodes, train_ids, representation)
    stage_val_x, stage_val_y = _stage_xy(episodes, validation_ids, representation)
    stage_test_x, stage_test_y = _stage_xy(episodes, test_ids, representation)
    transform = Transform(common_256, seed).fit(stage_train_x)

    best_stage: tuple[float, float, LogisticRegression] | None = None
    for c in C_GRID:
        model = _fit_logistic(transform.apply(stage_train_x), stage_train_y, c)
        score = f1_score(stage_val_y, model.predict(transform.apply(stage_val_x)), average="macro")
        if best_stage is None or score > best_stage[0]:
            best_stage = (score, c, model)
    assert best_stage is not None
    stage_model = best_stage[2]
    stage_pred = stage_model.predict(transform.apply(stage_test_x))
    precision, recall, _, _ = precision_recall_fscore_support(
        stage_test_y, stage_pred, labels=[0, 1], zero_division=0
    )

    # Boundary inputs must be constructed after transforming z; rebuild them here.
    def transformed_boundary(ids: set[str]) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for episode_id in sorted(ids):
            episode = episodes[episode_id]
            valid = episode.terminal == 0
            z = transform.apply(episode.features[representation][valid])
            delta = np.vstack([np.zeros((1, z.shape[1]), dtype=z.dtype), np.diff(z, axis=0)])
            xs.append(np.concatenate([z, delta], axis=1))
            ys.append(_boundary_labels(episode)[valid])
        return np.concatenate(xs), np.concatenate(ys)

    boundary_train_x, boundary_train_y = transformed_boundary(train_ids)
    best_boundary: tuple[tuple[float, float, float], float, float, LogisticRegression] | None = None
    for c in C_GRID:
        model = _fit_logistic(boundary_train_x, boundary_train_y, c)
        val_probabilities = _boundary_probabilities(
            model, transform, episodes, validation_ids, representation
        )
        for threshold in BOUNDARY_THRESHOLDS:
            metrics = _event_metrics(episodes, validation_ids, val_probabilities, float(threshold))
            score = (
                metrics["boundary_event_f1_at_5"],
                -metrics["false_boundaries_per_episode"],
                -abs(float(threshold) - 0.5),
            )
            if best_boundary is None or score > best_boundary[0]:
                best_boundary = (score, c, float(threshold), model)
    assert best_boundary is not None
    test_probabilities = _boundary_probabilities(
        best_boundary[3], transform, episodes, test_ids, representation
    )
    boundary_metrics = _event_metrics(
        episodes,
        test_ids,
        test_probabilities,
        best_boundary[2],
    )

    boundary_test_x, _ = transformed_boundary(test_ids)
    repeats = max(10, min(100, 10_000 // max(1, len(boundary_test_x))))
    started = perf_counter()
    for _ in range(repeats):
        best_boundary[3].predict_proba(boundary_test_x)
    probe_latency_ms = (perf_counter() - started) * 1000 / (repeats * max(1, len(boundary_test_x)))

    test_episodes = [episodes[episode_id] for episode_id in test_ids]
    return {
        "representation": representation,
        "projection": "common_256" if common_256 else "native",
        "seed": seed,
        "native_dim": int(stage_train_x.shape[1]),
        "probe_dim": int(transform.apply(stage_train_x[:1]).shape[1]),
        "stage_c": best_stage[1],
        "boundary_c": best_boundary[1],
        "boundary_threshold": best_boundary[2],
        "stage_macro_f1": f1_score(stage_test_y, stage_pred, average="macro"),
        "stage_balanced_accuracy": balanced_accuracy_score(stage_test_y, stage_pred),
        "stage_0_precision": precision[0],
        "stage_0_recall": recall[0],
        "stage_1_precision": precision[1],
        "stage_1_recall": recall[1],
        "terminal_test_samples": int(
            sum(int(episodes[episode_id].terminal.sum()) for episode_id in test_ids)
        ),
        "probe_latency_ms_per_sample": probe_latency_ms,
        "feature_hook_latency_ms_per_step": float(
            np.nanmean([episode.mean_hook_latency_ms for episode in test_episodes])
        ),
        "model_inference_latency_ms_per_step": float(
            np.nanmean([episode.mean_inference_latency_ms for episode in test_episodes])
        ),
        "peak_vram_bytes": max(episode.peak_vram_bytes for episode in test_episodes),
        "host_rss_delta_bytes": max(episode.host_rss_delta_bytes for episode in test_episodes),
        **boundary_metrics,
    }


def _nanmean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    valid = array[~np.isnan(array)]
    return float(valid.mean()) if valid.size else float("nan")


def _mean_ci95(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    valid = array[~np.isnan(array)]
    mean = float(valid.mean()) if valid.size else float("nan")
    if valid.size < 2:
        return mean, float("nan")
    return mean, float(1.96 * valid.std(ddof=1) / np.sqrt(valid.size))


def _paired_tie(best: dict, candidate: dict, metric: str) -> bool:
    best_by_seed = {int(row["seed"]): float(row[metric]) for row in best["rows"]}
    candidate_by_seed = {int(row["seed"]): float(row[metric]) for row in candidate["rows"]}
    shared = sorted(best_by_seed.keys() & candidate_by_seed.keys())
    differences = np.asarray(
        [best_by_seed[seed] - candidate_by_seed[seed] for seed in shared],
        dtype=np.float64,
    )
    differences = differences[~np.isnan(differences)]
    if differences.size < 2:
        return bool(differences.size and np.isclose(differences[0], 0.0))
    margin = 1.96 * differences.std(ddof=1) / np.sqrt(differences.size)
    return float(differences.mean() - margin) <= 0.0


def _write_report(rows: list[dict], episodes: list[Episode], output_dir: Path) -> None:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["representation"], row["projection"]), []).append(row)
    summary = []
    for (representation, projection), values in grouped.items():
        boundary, boundary_ci95 = _mean_ci95(v["boundary_event_f1_at_5"] for v in values)
        stage, stage_ci95 = _mean_ci95(v["stage_macro_f1"] for v in values)
        summary.append(
            {
                "representation": representation,
                "projection": projection,
                "boundary": boundary,
                "boundary_ci95": boundary_ci95,
                "stage": stage,
                "stage_ci95": stage_ci95,
                "failure_fpr": _nanmean(v["no_boundary_failure_fpr"] for v in values),
                "dim": values[0]["probe_dim"],
                "probe_ms": _nanmean(v["probe_latency_ms_per_sample"] for v in values),
                "hook_ms": _nanmean(v["feature_hook_latency_ms_per_step"] for v in values),
                "boundary_10": _nanmean(v["boundary_event_f1_at_10"] for v in values),
                "median_error": _nanmean(v["boundary_median_abs_error"] for v in values),
                "false_events": _nanmean(v["false_boundaries_per_episode"] for v in values),
                "rows": values,
            }
        )

    best_boundary = max(summary, key=lambda item: item["boundary"])
    boundary_ties = [
        item for item in summary if _paired_tie(best_boundary, item, "boundary_event_f1_at_5")
    ]
    best_stage = max(boundary_ties, key=lambda item: item["stage"])
    stage_ties = [item for item in boundary_ties if _paired_tie(best_stage, item, "stage_macro_f1")]
    winner = min(
        stage_ties,
        key=lambda item: (
            item["failure_fpr"] if not np.isnan(item["failure_fpr"]) else float("inf"),
            item["dim"],
            item["probe_ms"],
        ),
    )
    summary.sort(
        key=lambda item: (
            -item["boundary"],
            -item["stage"],
            item["failure_fpr"] if not np.isnan(item["failure_fpr"]) else float("inf"),
            item["dim"],
        )
    )
    successes = sum(episode.success for episode in episodes)
    lines = [
        "# GR00T N1.7 서브태스크 분리 실험 보고서",
        "",
        f"- 수집 episode: {len(episodes)}개",
        f"- rollout 성공: {successes}/{len(episodes)}",
        "- split: episode 단위 60/20/20, 공통 seed 5개",
        "- 모델: frozen GR00T N1.7 + L2 logistic linear probe",
        "",
        "## 표현별 평균 결과",
        "",
        "| 표현 | 투영 | Boundary F1 ±5 (95% CI) | Stage macro F1 (95% CI) | 실패 FPR | 차원 | Probe ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| {item['representation']} | {item['projection']} | "
            f"{item['boundary']:.3f} ± {item['boundary_ci95']:.3f} | "
            f"{item['stage']:.3f} ± {item['stage_ci95']:.3f} | "
            f"{item['failure_fpr']:.3f} | {item['dim']} | {item['probe_ms']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 경계 보조 지표",
            "",
            "| 표현 | 투영 | Boundary F1 ±10 | 경계 오차 중앙값(step) | 예측 경계 수/episode |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in summary:
        lines.append(
            f"| {item['representation']} | {item['projection']} | "
            f"{item['boundary_10']:.3f} | {item['median_error']:.1f} | "
            f"{item['false_events']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 선정",
            "",
            (
                f"규칙상 1위는 **{winner['representation']} ({winner['projection']})**다. "
                "공통 seed별 paired 차이의 95% CI로 Boundary F1 ±5와 Stage macro F1의 "
                "통계적 동률을 판단한 뒤, 실패 episode FPR, 차원, probe 지연 순으로 "
                "더 단순한 표현을 선택했다."
            ),
            "",
            (
                f"전체 R0~R6 동시 계측 hook 지연은 step당 평균 "
                f"{_nanmean(episode.mean_hook_latency_ms for episode in episodes):.4f} ms다."
            ),
        ]
    )
    invariance_path = output_dir / "rollout_invariance.json"
    if invariance_path.is_file():
        invariance = json.loads(invariance_path.read_text())
        lines.extend(
            [
                "",
                "## 계측 전후 rollout 불변성",
                "",
                (
                    f"동일 seed paired {invariance['paired_episodes']}개에서 baseline과 계측 "
                    f"성공은 각각 {invariance['baseline_successes']}/"
                    f"{invariance['paired_episodes']}, {invariance['instrumented_successes']}/"
                    f"{invariance['paired_episodes']}였다. 성공 불일치 "
                    f"{len(invariance['success_mismatches'])}개, 종료 step 불일치 "
                    f"{len(invariance['step_mismatches'])}개였다."
                ),
            ]
        )
    cost_path = output_dir / "cost_summary.json"
    if cost_path.is_file():
        cost = json.loads(cost_path.read_text())
        lines.extend(
            [
                "",
                "## 계측 비용",
                "",
                f"- 표본: policy step {cost['samples']}개(첫 호출 포함)",
                f"- 모델 추론: 평균 {cost['model_inference']['mean_ms']:.3f} ms/step",
                f"- 전체 feature snapshot: 평균 {cost['snapshot']['mean_ms']:.3f} ms/step",
                f"- forward hook 자체: 평균 {cost['hook']['mean_ms']:.4f} ms/step",
                f"- peak VRAM: {cost['peak_vram_bytes'] / 2**30:.3f} GiB",
                f"- episode 내 peak host RSS 증가: {cost['host_peak_rss_delta_bytes'] / 2**20:.3f} MiB",
                "",
                "| 표현 | 평균 추출 ms | P95 ms |",
                "|---|---:|---:|",
            ]
        )
        for name, values in cost["representations"].items():
            lines.append(f"| {name} | {values['mean_ms']:.4f} | {values['p95_ms']:.4f} |")
    lines.extend(
        [
            "",
            (
                "R5는 discrete action token이 아니라 마지막 denoise iteration의 "
                "action-token-like continuous latent로 해석해야 한다."
            ),
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    episodes = load_episodes(Path(args.artifacts))
    representations = sorted(set.intersection(*(set(ep.features) for ep in episodes)))
    rows = []
    for representation in representations:
        for common_256 in (False, True):
            for seed in SEEDS:
                rows.append(evaluate_one(episodes, representation, common_256, seed))
                print(representation, "common_256" if common_256 else "native", seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (output_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_report(rows, episodes, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts")
    parser.add_argument("--output-dir", default="reports")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
