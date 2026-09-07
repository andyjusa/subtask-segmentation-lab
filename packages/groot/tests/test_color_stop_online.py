from __future__ import annotations

import numpy as np

from groot_subtask_phase_probe.color_stop_online import (
    ColorStopArtifact,
    ColorStopDetector,
    _first_confirmed,
)


def _artifact(confirm_steps: int = 2) -> ColorStopArtifact:
    coefficient = np.zeros((512, 1), dtype=np.float32)
    coefficient[256, 0] = 10.0
    return ColorStopArtifact(
        representation="R3b_vision_tokens",
        seed=17,
        threshold=0.5,
        confirm_steps=confirm_steps,
        scaler_mean=np.zeros(1, dtype=np.float32),
        scaler_scale=np.ones(1, dtype=np.float32),
        pca_mean=np.zeros(1, dtype=np.float32),
        pca_components=np.ones((1, 1), dtype=np.float32),
        coefficients=(coefficient,),
        intercepts=(np.array([-5.0], dtype=np.float32),),
    )


def test_first_confirmed_requires_consecutive_frames() -> None:
    probabilities = np.array([0.1, 0.9, 0.1, 0.8, 0.7], dtype=np.float32)
    assert _first_confirmed(probabilities, 0.5, 1) == 1
    assert _first_confirmed(probabilities, 0.5, 2) == 4
    assert _first_confirmed(probabilities, 0.5, 3) is None


def test_detector_emits_one_stop_after_confirmation() -> None:
    detector = ColorStopDetector(_artifact(confirm_steps=2))
    assert detector.observe(np.array([0.0]), 0) is None
    assert detector.observe(np.array([1.0]), 1) is None
    event = detector.observe(np.array([2.0]), 2)
    assert event is not None and event[0] == 2
    assert detector.observe(np.array([3.0]), 3) is None


def test_artifact_round_trip(tmp_path) -> None:
    expected = _artifact(confirm_steps=3)
    path = tmp_path / "artifact.npz"
    expected.save(path)
    actual = ColorStopArtifact.load(path)
    assert actual.representation == expected.representation
    assert actual.confirm_steps == 3
    assert actual.threshold == expected.threshold
    np.testing.assert_array_equal(actual.coefficients[0], expected.coefficients[0])
