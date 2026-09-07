import pytest

from vla_subtask_phase_probe.training import phase_labels


def test_phase_labels_follow_ordered_boundaries() -> None:
    assert phase_labels(7, (2, 5)).tolist() == [0, 0, 1, 1, 1, 2, 2]


@pytest.mark.parametrize("frames", [(0,), (3, 2), (2, 7)])
def test_phase_labels_reject_invalid_boundaries(frames: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        phase_labels(7, frames)
