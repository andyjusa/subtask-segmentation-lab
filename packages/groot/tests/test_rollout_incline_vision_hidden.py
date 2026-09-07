import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_rollout_incline_holdout import Boundary
from scripts.evaluate_rollout_incline_vision_hidden import (
    blocked_stage_folds,
    ordered_decode,
    stage_labels,
)


def test_stage_labels_advance_at_reviewed_boundaries() -> None:
    boundaries = [
        Boundary(stage=1, frame=10, time_seconds=1.0),
        Boundary(stage=2, frame=20, time_seconds=2.0),
        Boundary(stage=3, frame=30, time_seconds=3.0),
        Boundary(stage=4, frame=40, time_seconds=4.0),
    ]
    frames = np.asarray([0, 9, 10, 19, 20, 40])
    assert stage_labels(frames, boundaries).tolist() == [0, 0, 1, 1, 2, 4]


def test_blocked_folds_cover_every_sample_and_apply_embargo() -> None:
    labels = np.repeat(np.arange(5), 20)
    tests = []
    for train, test in blocked_stage_folds(labels, folds=5, embargo=2):
        assert not np.intersect1d(train, test).size
        tests.extend(test.tolist())
        for index in test:
            nearby_same_stage = train[
                (labels[train] == labels[index]) & (np.abs(train - index) <= 2)
            ]
            assert not len(nearby_same_stage)
    assert sorted(tests) == list(range(len(labels)))


def test_ordered_decode_recovers_clean_five_stage_sequence() -> None:
    probabilities = np.full((100, 5), 0.01)
    for stage in range(5):
        probabilities[stage * 20 : (stage + 1) * 20, stage] = 0.96
    decoded, boundaries = ordered_decode(probabilities, sample_fps=5.0)
    assert boundaries.tolist() == [20, 40, 60, 80]
    assert decoded.tolist() == np.repeat(np.arange(5), 20).tolist()
