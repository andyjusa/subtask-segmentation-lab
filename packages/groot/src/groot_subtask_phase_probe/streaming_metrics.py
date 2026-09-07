from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from typing import Any

import numpy as np


def rising_event_steps(
    probabilities: np.ndarray,
    env_steps: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    """Return online rising-edge events without smoothing or future look-ahead."""

    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    steps = np.asarray(env_steps, dtype=np.int64).reshape(-1)
    if values.shape != steps.shape:
        raise ValueError("probabilities and env_steps must have the same one-dimensional shape")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between zero and one")
    above = values >= threshold
    rising = above & np.concatenate([np.asarray([True]), ~above[:-1]])
    return steps[np.flatnonzero(rising)]


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2 * tp / denominator


def boundary_event_metrics(
    episodes: Mapping[str, Any],
    episode_ids: AbstractSet[str],
    probabilities: Mapping[str, np.ndarray],
    *,
    threshold: float,
    tolerances: tuple[int, ...] = (5, 10),
) -> dict[str, float | int]:
    """Evaluate first online boundary predictions as real events.

    Each episode has at most one ground-truth boundary. The first rising prediction
    is the only prediction eligible to match it; later predictions cannot repair an
    early mistake and are always false positives. This matches deployment, where the
    first event triggers the transition. Unlike the older benchmark, missing and
    extra events contribute FN and FP instead of being forced by Viterbi decoding.
    """

    if not tolerances or any(tolerance < 0 for tolerance in tolerances):
        raise ValueError("tolerances must contain non-negative values")
    primary_tolerance = tolerances[0]
    counts = {tolerance: [0, 0, 0] for tolerance in tolerances}
    first_errors: list[float] = []
    false_boundaries_primary = 0
    predicted_boundaries = 0
    no_boundary_failures = 0
    no_boundary_failure_fp = 0

    for episode_id in sorted(episode_ids):
        if episode_id not in episodes or episode_id not in probabilities:
            raise KeyError(f"Missing episode or probabilities for {episode_id}")
        episode = episodes[episode_id]
        valid = np.asarray(episode.terminal, dtype=np.int64) == 0
        if hasattr(episode, "stage"):
            valid &= np.asarray(episode.stage, dtype=np.int64) >= 0
        steps = np.asarray(episode.env_step, dtype=np.int64)[valid]
        values = np.asarray(probabilities[episode_id], dtype=np.float64)
        predicted_steps = rising_event_steps(values, steps, threshold=threshold)
        predicted_boundaries += len(predicted_steps)
        first_prediction = int(predicted_steps[0]) if len(predicted_steps) else None
        ground_truth = None if episode.boundary_step is None else int(episode.boundary_step)

        if ground_truth is None and not bool(episode.success):
            no_boundary_failures += 1
            no_boundary_failure_fp += int(first_prediction is not None)
        if ground_truth is not None and first_prediction is not None:
            first_errors.append(float(abs(first_prediction - ground_truth)))

        for tolerance in tolerances:
            tp, fp, fn = counts[tolerance]
            if ground_truth is None:
                fp += len(predicted_steps)
            elif first_prediction is None:
                fn += 1
            elif abs(first_prediction - ground_truth) <= tolerance:
                tp += 1
                fp += max(0, len(predicted_steps) - 1)
            else:
                fp += len(predicted_steps)
                fn += 1
            counts[tolerance] = [tp, fp, fn]

        if ground_truth is None:
            false_boundaries_primary += len(predicted_steps)
        elif first_prediction is None:
            pass
        elif abs(first_prediction - ground_truth) <= primary_tolerance:
            false_boundaries_primary += max(0, len(predicted_steps) - 1)
        else:
            false_boundaries_primary += len(predicted_steps)

    result: dict[str, float | int] = {
        "boundary_median_abs_error": (
            float(np.median(first_errors)) if first_errors else float("nan")
        ),
        "false_boundaries_per_episode": false_boundaries_primary / max(1, len(episode_ids)),
        "predicted_boundaries_per_episode": predicted_boundaries / max(1, len(episode_ids)),
        "no_boundary_failure_fpr": (
            no_boundary_failure_fp / no_boundary_failures if no_boundary_failures else float("nan")
        ),
        "no_boundary_failure_episodes": no_boundary_failures,
    }
    for tolerance, (tp, fp, fn) in counts.items():
        result[f"boundary_event_f1_at_{tolerance}"] = _f1(tp, fp, fn)
        result[f"boundary_tp_at_{tolerance}"] = tp
        result[f"boundary_fp_at_{tolerance}"] = fp
        result[f"boundary_fn_at_{tolerance}"] = fn
    return result
