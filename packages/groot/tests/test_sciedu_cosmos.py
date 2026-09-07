import numpy as np

from groot_subtask_phase_probe.sciedu_cosmos import (
    ClipConfig,
    _event_metrics,
    blocked_stage_folds,
    stage_labels,
    trailing_clip_times,
)


def test_stage_labels_and_terminal() -> None:
    timestamps = np.asarray([0.0, 21.99, 22.0, 47.5, 63.25, 72.99, 73.0])
    stage, terminal = stage_labels(timestamps)
    assert stage.tolist() == [0, 0, 1, 2, 3, 3, 3]
    assert terminal.tolist() == [False, False, False, False, False, False, True]


def test_trailing_clip_is_causal_and_has_fixed_count() -> None:
    times = trailing_clip_times(10.0, ClipConfig(4.0, 5.0))
    assert len(times) == 20
    assert times[-1] == 10.0
    assert np.all(times <= 10.0)
    assert np.allclose(np.diff(times), 0.2)


def test_early_trailing_clip_pads_with_first_frame() -> None:
    times = trailing_clip_times(0.5, ClipConfig(2.0, 2.0))
    assert times.tolist() == [0.0, 0.0, 0.0, 0.5]


def test_blocked_stage_folds_cover_every_sample_once() -> None:
    labels = np.repeat(np.arange(4), 10)
    folds = blocked_stage_folds(labels, folds=5)
    tests = np.concatenate([test for _, test in folds])
    assert sorted(tests.tolist()) == list(range(len(labels)))
    for train, test in folds:
        assert not set(train) & set(test)
        assert set(labels[test]) == {0, 1, 2, 3}


def test_event_metrics_penalize_extra_boundaries() -> None:
    exact = _event_metrics([22.0, 47.5, 63.25], 1.0)
    extra = _event_metrics([22.0, 30.0, 47.5, 63.25], 1.0)
    assert exact["boundary_f1_at_1s"] == 1.0
    assert extra["boundary_f1_at_1s"] < 1.0

