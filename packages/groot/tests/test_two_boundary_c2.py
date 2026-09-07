import numpy as np

from groot_subtask_phase_probe.two_boundary_c2 import (
    TwoBoundarySource,
    event_metrics,
    ordered_accuracy,
    paired_stratified_split,
)


def _source(pair_id: int, mode: str, changed: bool = True) -> TwoBoundarySource:
    return TwoBoundarySource(
        episode_id=f"fixed_color_two_boundary/{mode}/episode_{pair_id:04d}",
        pair_id=pair_id,
        color_mode=mode,
        color_change_enabled=changed,
        features={"R3b_vision_tokens": np.zeros((8, 4), dtype=np.float32)},
        sample_index=np.arange(8),
        stove_boundary_sample=2,
        color_boundary_sample=5 if changed else None,
    )


def test_paired_split_keeps_fixed_and_random_together() -> None:
    sources = [
        _source(pair_id, mode, changed=pair_id % 4 != 0)
        for pair_id in range(12)
        for mode in ("fixed", "random")
    ]
    splits = paired_stratified_split(sources, seed=17)
    assert not (splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2])
    assert set.union(*splits) == set(range(12))


def test_event_metrics_count_missed_and_false_events() -> None:
    metrics = event_metrics(
        {"a": 4, "b": None, "c": 5},
        {"a": (4, 1), "b": (3, 1), "c": (None, 0)},
        tolerance=1,
    )
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["negative_fpr"] == 1.0


def test_ordered_accuracy_requires_both_events_in_order() -> None:
    changed = _source(1, "fixed", changed=True)
    control = _source(4, "fixed", changed=False)
    metrics = ordered_accuracy(
        [changed, control],
        {
            changed.episode_id: (2, 1),
            control.episode_id: (2, 1),
        },
        {
            changed.episode_id: (5, 1),
            control.episode_id: (None, 0),
        },
        tolerance=1,
    )
    assert metrics["ordered_episode_accuracy"] == 1.0
    assert metrics["ordered_positive_accuracy"] == 1.0
