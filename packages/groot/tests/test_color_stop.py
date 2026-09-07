import numpy as np

from groot_subtask_phase_probe.color_stop import _add_history_axis, render_beaker


def test_render_beaker_changes_only_liquid_and_rod_pixels() -> None:
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    blue = render_beaker(frame, (10, 20, 200), phase=0.0)
    red = render_beaker(frame, (200, 20, 10), phase=0.0)
    changed = np.any(blue != red, axis=-1)
    assert changed.any()
    blue_colors = blue[changed]
    red_colors = red[changed]
    assert np.all(blue_colors == np.array([10, 20, 200]))
    assert np.all(red_colors == np.array([200, 20, 10]))


def test_stirring_motion_changes_geometry_without_changing_background() -> None:
    frame = np.full((160, 240, 3), 17, dtype=np.uint8)
    first = render_beaker(frame, (10, 20, 200), phase=0.0)
    second = render_beaker(frame, (10, 20, 200), phase=np.pi / 2)
    assert np.any(first != second)
    assert np.array_equal(first[:20, :20], frame[:20, :20])


def test_add_history_axis_matches_policy_contract_before_batching() -> None:
    observation = {
        "video.image": np.zeros((16, 16, 3), dtype=np.uint8),
        "state.joint": np.zeros(7, dtype=np.float32),
        "annotation.task": "stop on change",
    }
    history = _add_history_axis(observation)
    assert history["video.image"].shape == (1, 16, 16, 3)
    assert history["state.joint"].shape == (1, 7)
    assert history["annotation.task"] == "stop on change"
