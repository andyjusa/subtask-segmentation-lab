import numpy as np

from groot_subtask_phase_probe.fixed_color_two_boundary import (
    render_history_observation,
    schedule_color_change,
)


def test_color_changes_only_after_stove_and_delay() -> None:
    assert not schedule_color_change(
        predicate_seen=False,
        post_stove_samples=10,
        delay_samples=2,
        change_enabled=True,
    )
    assert not schedule_color_change(
        predicate_seen=True,
        post_stove_samples=1,
        delay_samples=2,
        change_enabled=True,
    )
    assert schedule_color_change(
        predicate_seen=True,
        post_stove_samples=2,
        delay_samples=2,
        change_enabled=True,
    )


def test_no_change_control_never_changes() -> None:
    assert not schedule_color_change(
        predicate_seen=True,
        post_stove_samples=100,
        delay_samples=2,
        change_enabled=False,
    )


def test_render_history_observation_preserves_history_shape() -> None:
    observation = np.zeros((1, 32, 32, 3), dtype=np.uint8)
    rendered, video_frame = render_history_observation(observation, (42, 111, 219), 0.0)
    assert rendered.shape == observation.shape
    assert video_frame.shape == observation.shape[1:]
