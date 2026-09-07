import numpy as np

from groot_subtask_phase_probe.hydra_action_flow import (
    HydraFlowConfig,
    boundary_targets,
    build_causal_flow_feature,
    point_memberships,
    select_grouped_points,
    select_mask_union,
)


def test_sam_masks_assign_object_before_robot_and_keep_background() -> None:
    robot = np.zeros((8, 8), dtype=bool)
    objects = np.zeros((8, 8), dtype=bool)
    robot[1, 1] = True
    robot[2, 2] = True
    objects[2, 2] = True
    points = np.asarray([[1, 1], [2, 2], [7, 7]], dtype=np.float32)

    memberships = point_memberships(points, robot, objects)

    assert memberships.tolist() == [
        [True, False, False],
        [False, True, False],
        [False, False, True],
    ]


def test_group_sampling_has_fixed_slots_and_explicit_padding() -> None:
    config = HydraFlowConfig(robot_points=2, object_points=2, background_points=1)
    points = np.asarray([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.float32)
    memberships = np.asarray(
        [
            [True, False, False],
            [False, True, False],
            [False, False, True],
            [False, False, True],
        ]
    )

    indices, groups, valid = select_grouped_points(points, memberships, config)

    assert indices.shape == groups.shape == valid.shape == (5,)
    assert groups.tolist() == [0, 0, 1, 1, 2]
    assert valid.tolist() == [True, False, True, False, True]
    assert indices[~valid].tolist() == [-1, -1]


def test_causal_feature_uses_only_frames_since_latest_sam_anchor() -> None:
    config = HydraFlowConfig(
        window_samples=4,
        robot_points=1,
        object_points=1,
        background_points=0,
        width=10,
        height=10,
    )
    tracks = np.zeros((6, 2, 2), dtype=np.float32)
    tracks[:, 0, 0] = np.arange(6)
    tracks[:, 1, 1] = np.arange(6) * 2
    visibility = np.ones((6, 2), dtype=bool)
    confidence = np.ones((6, 2), dtype=np.float32)

    feature = build_causal_flow_feature(
        tracks,
        visibility,
        confidence,
        anchor_index=3,
        current_index=4,
        selected_indices=np.asarray([0, 1]),
        slot_groups=np.asarray([0, 1]),
        slot_valid=np.asarray([True, True]),
        config=config,
    )

    assert feature.shape == (4, 2, 12)
    assert feature[:2].sum() == 0.0
    np.testing.assert_allclose(feature[-1, 0, 2:4], [0.1, 0.0])
    np.testing.assert_allclose(feature[-1, 1, 2:4], [0.0, 0.2])
    assert feature[-1, 0, 8:11].tolist() == [1.0, 0.0, 0.0]
    assert feature[-1, 1, 8:11].tolist() == [0.0, 1.0, 0.0]
    assert feature[:, :, 11].tolist() == [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]


def test_boundary_target_uses_first_available_sample_after_event() -> None:
    samples = np.asarray([0, 6, 12, 18, 24])
    targets = boundary_targets(samples, np.asarray([7, 18]))
    assert targets.tolist() == [0, 0, 1, 1, 0]


def test_sam3_union_applies_threshold_and_instance_limit() -> None:
    masks = np.zeros((3, 4, 4), dtype=bool)
    masks[0, 0, 0] = True
    masks[1, 1, 1] = True
    masks[2, 2, 2] = True
    selected = select_mask_union(np.asarray([0.8, 0.7, 0.05]), masks, limit=1, threshold=0.1)
    assert selected.sum() == 1
    assert selected[0, 0]
