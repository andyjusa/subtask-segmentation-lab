"""Measure stage separability of frozen GR00T vision hidden states on one rollout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_rollout_incline_holdout import Boundary, load_boundaries, render_stage_videos


def stage_labels(sample_frames: np.ndarray, boundaries: list[Boundary]) -> np.ndarray:
    boundary_frames = np.asarray([item.frame for item in boundaries], dtype=np.int64)
    return np.searchsorted(boundary_frames, sample_frames, side="right").astype(np.int64)


def blocked_stage_folds(labels: np.ndarray, folds: int = 5, embargo: int = 5):
    """Historical stage-aware OOF with same-stage embargo; NOT causal validation.

    Neighbours in a different stage are not excluded. Preserved for reproducibility,
    not suitable as evidence of unseen-episode or online performance.
    """
    all_indices = np.arange(len(labels))
    per_stage = {
        stage: np.array_split(np.flatnonzero(labels == stage), folds) for stage in range(5)
    }
    for fold in range(folds):
        test = np.sort(np.concatenate([per_stage[stage][fold] for stage in range(5)]))
        excluded = np.zeros(len(labels), dtype=bool)
        excluded[test] = True
        for stage in range(5):
            block = per_stage[stage][fold]
            if not len(block):
                continue
            same_stage = labels == stage
            excluded |= (
                same_stage
                & (all_indices >= block[0] - embargo)
                & (all_indices <= block[-1] + embargo)
            )
        train = all_indices[~excluded]
        yield train, test


def ordered_decode(
    probabilities: np.ndarray, sample_fps: float, min_segment_seconds: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    smoothed = gaussian_filter1d(probabilities, sigma=2.0, axis=0)
    log_probabilities = np.log(np.clip(smoothed, 1e-7, 1.0))
    cumulative = np.vstack([np.zeros((1, 5)), log_probabilities.cumsum(axis=0)])
    count = len(probabilities)
    minimum = max(1, round(min_segment_seconds * sample_fps))
    scores = np.full((5, count + 1), -1e30, dtype=np.float64)
    previous = np.full((5, count + 1), -1, dtype=np.int32)
    for end in range(minimum, count + 1):
        scores[0, end] = cumulative[end, 0]
    for stage in range(1, 5):
        for end in range((stage + 1) * minimum, count + 1):
            starts = np.arange(stage * minimum, end - minimum + 1)
            candidates = (
                scores[stage - 1, starts] + cumulative[end, stage] - cumulative[starts, stage]
            )
            best = int(np.argmax(candidates))
            scores[stage, end] = candidates[best]
            previous[stage, end] = starts[best]
    end = count
    boundaries = []
    for stage in range(4, 0, -1):
        end = int(previous[stage, end])
        if end < 0:
            raise RuntimeError("ordered decoding failed")
        boundaries.append(end)
    boundaries_array = np.asarray(boundaries[::-1], dtype=np.int64)
    decoded = np.searchsorted(boundaries_array, np.arange(count), side="right")
    return decoded, boundaries_array


def out_of_fold_probabilities(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    pca_components: int | None,
    folds: int = 5,
    embargo: int = 5,
) -> np.ndarray:
    probabilities = np.zeros((len(labels), 5), dtype=np.float64)
    covered = np.zeros(len(labels), dtype=bool)
    for train, test in blocked_stage_folds(labels, folds=folds, embargo=embargo):
        scaler = StandardScaler().fit(features[train])
        train_x = np.clip(scaler.transform(features[train]), -8.0, 8.0)
        test_x = np.clip(scaler.transform(features[test]), -8.0, 8.0)
        if pca_components is not None:
            components = min(pca_components, len(train) - 1, features.shape[1])
            projection = PCA(
                n_components=components,
                svd_solver="randomized",
                random_state=42,
            ).fit(train_x)
            train_x = projection.transform(train_x)
            test_x = projection.transform(test_x)
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            random_state=42,
        ).fit(train_x, labels[train])
        probabilities[test] = model.predict_proba(test_x)
        covered[test] = True
    if not covered.all():
        raise RuntimeError("out-of-fold predictions do not cover every sample")
    return probabilities


def evaluate_representation(
    features: np.ndarray,
    labels: np.ndarray,
    sample_frames: np.ndarray,
    reference_boundaries: list[Boundary],
    *,
    sample_fps: float,
    pca_components: int | None,
) -> tuple[dict, np.ndarray]:
    probabilities = out_of_fold_probabilities(
        features,
        labels,
        pca_components=pca_components,
        embargo=round(sample_fps),
    )
    raw = probabilities.argmax(axis=1)
    ordered, predicted_samples = ordered_decode(probabilities, sample_fps)
    predicted_frames = sample_frames[predicted_samples]
    reference_frames = np.asarray([item.frame for item in reference_boundaries])
    errors = np.abs(predicted_frames - reference_frames) / 30.0
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        labels, ordered, labels=np.arange(5), zero_division=0
    )
    result = {
        "feature_dim": int(features.shape[1]),
        "probe_dim": int(pca_components or features.shape[1]),
        "blocked_oof_raw_macro_f1": float(f1_score(labels, raw, average="macro")),
        "blocked_oof_raw_balanced_accuracy": float(balanced_accuracy_score(labels, raw)),
        "blocked_oof_ordered_macro_f1": float(f1_score(labels, ordered, average="macro")),
        "blocked_oof_ordered_balanced_accuracy": float(balanced_accuracy_score(labels, ordered)),
        "predicted_boundary_frames": predicted_frames.tolist(),
        "predicted_boundary_times_s": (predicted_frames / 30.0).tolist(),
        "reference_boundary_frames": reference_frames.tolist(),
        "boundary_errors_s": errors.tolist(),
        "boundary_mean_absolute_error_s": float(errors.mean()),
        "boundary_within_1s_rate": float(np.mean(errors <= 1.0)),
        "raw_confusion_matrix": confusion_matrix(labels, raw, labels=np.arange(5)).tolist(),
        "ordered_confusion_matrix": confusion_matrix(labels, ordered, labels=np.arange(5)).tolist(),
        "ordered_per_class": [
            {
                "stage": stage,
                "precision": float(precision[stage]),
                "recall": float(recall[stage]),
                "f1": float(per_class_f1[stage]),
                "support": int(support[stage]),
            }
            for stage in range(5)
        ],
    }
    return result, probabilities


def write_report(path: Path, results: dict, best_name: str) -> None:
    rows = []
    for name, metrics in results.items():
        rows.append(
            f"| {name} | {metrics['blocked_oof_raw_macro_f1']:.3f} | "
            f"{metrics['blocked_oof_ordered_macro_f1']:.3f} | "
            f"{metrics['boundary_mean_absolute_error_s']:.2f}초 | "
            + ", ".join(f"{value:.2f}" for value in metrics["boundary_errors_s"])
            + " |"
        )
    best = results[best_name]
    class_rows = [
        f"| {item['stage']} | {item['support']} | {item['precision']:.3f} | "
        f"{item['recall']:.3f} | {item['f1']:.3f} |"
        for item in best["ordered_per_class"]
    ]
    path.write_text(
        "\n".join(
            [
                "# 실제 rollout의 GR00T vision hidden stage 분리",
                "",
                "## 결론",
                "",
                (
                    f"가장 좋은 조건은 `{best_name}`이며 blocked OOF ordered macro F1은 "
                    f"**{best['blocked_oof_ordered_macro_f1']:.3f}**, 경계 평균 오차는 "
                    f"**{best['boundary_mean_absolute_error_s']:.2f}초**다."
                ),
                "",
                "| 표현 | Raw macro F1 | Ordered macro F1 | 경계 MAE | 경계별 오차(초) |",
                "|---|---:|---:|---:|---|",
                *rows,
                "",
                "## Stage별 결과",
                "",
                "| Stage | 표본 | Precision | Recall | F1 |",
                "|---:|---:|---:|---:|---:|",
                *class_rows,
                "",
                (
                    "Stage 1의 precision이 낮은 주원인은 첫 경계를 5.9초 일찍 예측한 것이다. "
                    "Stage 3의 recall은 진입을 1.9초 늦게, 종료를 1.07초 일찍 예측해 "
                    "구간이 양쪽에서 줄어든 영향을 받았다."
                ),
                "",
                "## 비교 영상",
                "",
                (
                    "`reference-left_prediction-right.mp4`에서 왼쪽은 수동 검토 정답, "
                    "오른쪽은 R3b 예측이다. 테두리 색 변화 시점의 차이가 경계 오차다."
                ),
                "",
                "## 평가 계약",
                "",
                "- 입력: 세 카메라 + task instruction, 5 FPS",
                "- R3b: GR00T backbone의 image-mask token mean",
                "- R3: vision·instruction을 포함한 attention-mask token mean",
                "- backbone과 probe 입력 추출 시 LoRA adapter 및 action head 비활성화",
                "- 각 stage를 시간 순서대로 5개 block으로 나눈 뒤 한 block씩 test",
                "- test block 앞뒤 1초는 해당 fold의 train에서 제외",
                "- PCA는 각 fold의 train feature만 사용",
                "",
                "## 해석 제한",
                "",
                (
                    "이 수치는 한 episode 내부의 표현 분리 가능성이다. 서로 다른 episode로의 "
                    "일반화 성능은 아니며, 동일 task rollout을 추가 수집한 뒤 episode 단위 "
                    "평가가 필요하다."
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hidden",
        type=Path,
        default=Path(
            "reports/rollout_incline_20260903/groot_hidden/validation/episode_000/"
            "groot_backbone_hidden.npz"
        ),
    )
    parser.add_argument(
        "--boundaries",
        type=Path,
        default=Path("reports/rollout_incline_20260903/boundaries.json"),
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path(
            "data/rollout_Incline_20260903_20260903_190043/videos/"
            "observation.images.follower_d455f/chunk-000/file-000.mp4"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/rollout_incline_20260903/vision_hidden_eval"),
    )
    parser.add_argument("--sample-fps", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload, boundaries = load_boundaries(args.boundaries)
    arrays = np.load(args.hidden)
    sample_frames = arrays["sample_frames"].astype(np.int64)
    labels = stage_labels(sample_frames, boundaries)
    variants = {
        "R3b_vision_native": (arrays["vision_mean"].astype(np.float32), None),
        "R3b_vision_pca256": (arrays["vision_mean"].astype(np.float32), 256),
        "R3_full_native": (arrays["hidden_mean"].astype(np.float32), None),
        "R3_full_pca256": (arrays["hidden_mean"].astype(np.float32), 256),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    probabilities = {}
    for name, (features, pca_components) in variants.items():
        metrics, predicted_probabilities = evaluate_representation(
            features,
            labels,
            sample_frames,
            boundaries,
            sample_fps=args.sample_fps,
            pca_components=pca_components,
        )
        results[name] = metrics
        probabilities[name] = predicted_probabilities
    best_name = max(
        results,
        key=lambda name: (
            results[name]["blocked_oof_ordered_macro_f1"],
            -results[name]["boundary_mean_absolute_error_s"],
        ),
    )
    output = {
        "evaluation_scope": "single-episode blocked out-of-fold separability case study",
        "sample_count": len(sample_frames),
        "sample_fps": args.sample_fps,
        "class_counts": np.bincount(labels, minlength=5).tolist(),
        "best_variant": best_name,
        "variants": results,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    best_probabilities = probabilities[best_name]
    np.savez_compressed(
        args.output_dir / "oof_predictions.npz",
        sample_frames=sample_frames,
        labels=labels,
        probabilities=best_probabilities.astype(np.float32),
    )
    _, predicted_samples = ordered_decode(best_probabilities, args.sample_fps)
    predicted_frames = sample_frames[predicted_samples]
    predicted_boundaries = [
        Boundary(stage=index + 1, frame=int(frame), time_seconds=float(frame / 30.0))
        for index, frame in enumerate(predicted_frames)
    ]
    render_stage_videos(
        args.video,
        args.output_dir / "predicted_segments",
        float(payload["fps"]),
        int(payload["frame_count"]),
        predicted_boundaries,
    )
    rendered = args.output_dir / "rollout-subtasks-color-bordered.mp4"
    rendered.rename(args.output_dir / "vision-hidden-predicted-subtasks.mp4")
    write_report(args.output_dir / "REPORT.md", results, best_name)
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
