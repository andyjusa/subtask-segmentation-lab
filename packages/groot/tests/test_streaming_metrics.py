from types import SimpleNamespace

import numpy as np

from groot_subtask_phase_probe.streaming_metrics import boundary_event_metrics


def _episode(
    *,
    steps: list[int],
    boundary_step: int | None,
    success: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        env_step=np.asarray(steps, dtype=np.int64),
        terminal=np.zeros(len(steps), dtype=np.int64),
        boundary_step=boundary_step,
        success=success,
    )


def test_boundary_event_metrics_count_unmatched_rising_events_as_false_positives() -> None:
    episodes = {
        "positive": _episode(steps=[0, 5, 10, 15, 20], boundary_step=10, success=True),
        "negative": _episode(steps=[0, 5, 10, 15, 20], boundary_step=None, success=False),
    }
    probabilities = {
        # Rising events at 10 (matched) and 20 (false positive).
        "positive": np.asarray([0.1, 0.1, 0.9, 0.1, 0.9]),
        # One false event on a no-boundary failure episode.
        "negative": np.asarray([0.1, 0.8, 0.8, 0.1, 0.1]),
    }

    metrics = boundary_event_metrics(
        episodes,
        set(episodes),
        probabilities,
        threshold=0.5,
    )

    assert metrics["boundary_event_f1_at_5"] == 0.5
    assert metrics["boundary_event_f1_at_10"] == 0.5
    assert metrics["boundary_median_abs_error"] == 0.0
    assert metrics["false_boundaries_per_episode"] == 1.0
    assert metrics["no_boundary_failure_fpr"] == 1.0


def test_boundary_event_metrics_evaluate_the_first_prediction_online() -> None:
    episodes = {
        "episode": _episode(steps=[0, 5, 10, 15], boundary_step=10, success=True),
    }
    probabilities = {
        # The first event is too early. A later event at the GT must not repair it.
        "episode": np.asarray([0.9, 0.1, 0.9, 0.1]),
    }

    metrics = boundary_event_metrics(
        episodes,
        {"episode"},
        probabilities,
        threshold=0.5,
    )

    assert metrics["boundary_event_f1_at_5"] == 0.0
    assert metrics["boundary_event_f1_at_10"] == 2 / 3
    assert metrics["boundary_median_abs_error"] == 10.0
    assert metrics["false_boundaries_per_episode"] == 2.0
