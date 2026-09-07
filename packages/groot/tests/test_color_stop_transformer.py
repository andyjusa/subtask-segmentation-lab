from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from groot_subtask_phase_probe.color_stop_transformer import BASELINE, CANDIDATES, changed_fields
from groot_subtask_phase_probe.evaluate import Episode, Transform
from groot_subtask_phase_probe.streaming_transformer import _prepare_episodes


def _episode() -> Episode:
    return Episode(
        episode_id="color_stop/example_change",
        task="color_stop",
        features={"R3b_vision_tokens": np.arange(12, dtype=np.float32).reshape(6, 2)},
        stage=np.array([0, 0, 0, 1, 1, 1]),
        terminal=np.zeros(6, dtype=np.int64),
        env_step=np.arange(6),
        boundary_step=3,
        success=True,
    )


def test_exact_boundary_radius_marks_only_change_step() -> None:
    episode = _episode()
    transform = Transform(common_256=False, seed=17).fit(
        episode.features["R3b_vision_tokens"]
    )
    prepared = _prepare_episodes(
        {episode.episode_id: episode},
        {episode.episode_id},
        "R3b_vision_tokens",
        transform,
        boundary_radius=0,
    )[0]
    assert prepared.boundary.tolist() == [0, 0, 0, 1, 0, 0]


def test_explicit_delta_doubles_dimension_and_is_causal() -> None:
    episode = _episode()
    transform = Transform(common_256=False, seed=17).fit(
        episode.features["R3b_vision_tokens"]
    )
    prepared = _prepare_episodes(
        {episode.episode_id: episode},
        {episode.episode_id},
        "R3b_vision_tokens",
        transform,
        include_input_delta=True,
    )[0]
    assert prepared.values.shape == (6, 4)
    np.testing.assert_array_equal(prepared.values[0, 2:], np.zeros(2))
    np.testing.assert_allclose(
        prepared.values[3, 2:], prepared.values[3, :2] - prepared.values[2, :2]
    )


def test_screen_candidates_change_exactly_one_baseline_axis() -> None:
    assert changed_fields(BASELINE) == ()
    for name, candidate in CANDIDATES.items():
        if name == BASELINE.name:
            continue
        assert len(changed_fields(candidate)) == 1, name
