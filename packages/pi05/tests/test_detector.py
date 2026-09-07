from __future__ import annotations

import numpy as np
import pytest

from vla_subtask_phase_probe.detector import ActionPhaseProbeDetector, ProbeParameters


def detector(stable_frames: int = 2) -> ActionPhaseProbeDetector:
    return ActionPhaseProbeDetector(
        parameters=ProbeParameters(
            weight=np.asarray([[-2.0], [0.0], [2.0]]),
            bias=np.asarray([-1.0, 0.5, -1.0]),
            mean=np.asarray([0.0]),
            scale=np.asarray([1.0]),
        ),
        phases=("picking", "placing", "complete"),
        task_id="task",
        subtask_ids=("pick", "place", "task"),
        stable_frames=stable_frames,
    )


def test_emits_only_ordered_stable_transitions() -> None:
    probe = detector()
    assert probe.start().phase == "picking"
    assert probe.observe(np.asarray([1.0])) is None
    assert probe.observe(np.asarray([0.0])) is None
    placing = probe.observe(np.asarray([0.0]))
    assert placing is not None and placing.phase == "placing"
    assert probe.observe(np.asarray([1.0])) is None
    complete = probe.observe(np.asarray([1.0]))
    assert complete is not None and complete.phase == "complete"
    assert probe.observe(np.asarray([0.0])) is None


def test_transient_prediction_resets_stability_counter() -> None:
    probe = detector()
    probe.start()
    assert probe.observe(np.asarray([0.0])) is None
    assert probe.observe(np.asarray([-1.0])) is None
    assert probe.observe(np.asarray([0.0])) is None
    assert probe.phase == "picking"


def test_rejects_wrong_action_dimension() -> None:
    probe = detector(stable_frames=1)
    probe.start()
    with pytest.raises(ValueError, match="expected 1 action values"):
        probe.observe(np.asarray([0.0, 1.0]))
