from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GROUP_NAMES = ("robot", "object", "background")


@dataclass(frozen=True)
class HydraFlowConfig:
    window_samples: int = 20
    robot_points: int = 24
    object_points: int = 24
    background_points: int = 16
    width: int = 640
    height: int = 480

    @property
    def points(self) -> int:
        return self.robot_points + self.object_points + self.background_points

    @property
    def feature_channels(self) -> int:
        # xy, anchor-relative dxy, frame velocity, visibility, confidence,
        # three semantic memberships, and temporal validity.
        return 12

    @property
    def flattened_size(self) -> int:
        return self.window_samples * self.points * self.feature_channels


def select_mask_union(
    scores: np.ndarray,
    masks: np.ndarray,
    limit: int,
    threshold: float,
) -> np.ndarray:
    """Combine the highest-confidence SAM masks under an instance budget."""
    score_values = np.asarray(scores, dtype=np.float32)
    mask_values = np.asarray(masks, dtype=bool)
    if mask_values.ndim == 4 and mask_values.shape[1] == 1:
        mask_values = mask_values[:, 0]
    if mask_values.ndim != 3 or len(mask_values) != len(score_values):
        raise ValueError("masks must have shape [instances, height, width]")
    selected = np.flatnonzero(score_values >= threshold)
    selected = selected[np.argsort(-score_values[selected])[:limit]]
    if not len(selected):
        return np.zeros(mask_values.shape[-2:], dtype=bool)
    return np.any(mask_values[selected], axis=0)


def point_memberships(
    points_xy: np.ndarray,
    robot_mask: np.ndarray,
    object_mask: np.ndarray,
) -> np.ndarray:
    """Assign tracked points to SAM masks at one action-flow anchor frame.

    Object membership takes precedence in overlap pixels so a grasped object is
    not silently converted into part of the robot embodiment.
    """
    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape [points, 2]")
    if robot_mask.shape != object_mask.shape or robot_mask.ndim != 2:
        raise ValueError("robot_mask and object_mask must be equally sized 2D arrays")

    height, width = robot_mask.shape
    x = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, width - 1)
    y = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, height - 1)
    is_object = np.asarray(object_mask, dtype=bool)[y, x]
    is_robot = np.asarray(robot_mask, dtype=bool)[y, x] & ~is_object
    is_background = ~(is_robot | is_object)
    return np.stack([is_robot, is_object, is_background], axis=-1)


def _spread_indices(points_xy: np.ndarray, candidates: np.ndarray, count: int) -> np.ndarray:
    """Deterministically choose spatially spread candidates with farthest sampling."""
    if count < 0:
        raise ValueError("count must be non-negative")
    candidates = np.asarray(candidates, dtype=np.int64)
    if count == 0 or not len(candidates):
        return np.empty(0, dtype=np.int64)
    if len(candidates) <= count:
        return candidates

    points = np.asarray(points_xy, dtype=np.float32)[candidates]
    centroid = points.mean(axis=0)
    chosen_local = [int(np.argmin(np.linalg.norm(points - centroid, axis=1)))]
    minimum_distance = np.linalg.norm(points - points[chosen_local[0]], axis=1)
    for _ in range(1, count):
        next_local = int(np.argmax(minimum_distance))
        chosen_local.append(next_local)
        minimum_distance = np.minimum(
            minimum_distance,
            np.linalg.norm(points - points[next_local], axis=1),
        )
    return candidates[np.asarray(chosen_local)]


def select_grouped_points(
    points_xy: np.ndarray,
    memberships: np.ndarray,
    config: HydraFlowConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create fixed robot/object/background slots without duplicating points."""
    points = np.asarray(points_xy, dtype=np.float32)
    groups = np.asarray(memberships, dtype=bool)
    if groups.shape != (len(points), len(GROUP_NAMES)):
        raise ValueError("memberships must have shape [points, 3]")

    selected_parts = []
    slot_groups = []
    slot_valid = []
    counts = (config.robot_points, config.object_points, config.background_points)
    used: set[int] = set()
    for group_id, count in enumerate(counts):
        candidates = np.flatnonzero(groups[:, group_id])
        if used:
            candidates = np.asarray([index for index in candidates if int(index) not in used])
        selected = _spread_indices(points, candidates, count)
        used.update(int(index) for index in selected)
        padding = count - len(selected)
        selected_parts.append(np.pad(selected, (0, padding), constant_values=-1))
        slot_groups.append(np.full(count, group_id, dtype=np.int8))
        slot_valid.append(np.r_[np.ones(len(selected), dtype=bool), np.zeros(padding, dtype=bool)])
    return (
        np.concatenate(selected_parts),
        np.concatenate(slot_groups),
        np.concatenate(slot_valid),
    )


def build_causal_flow_feature(
    tracks: np.ndarray,
    visibility: np.ndarray,
    confidence: np.ndarray,
    anchor_index: int,
    current_index: int,
    selected_indices: np.ndarray,
    slot_groups: np.ndarray,
    slot_valid: np.ndarray,
    config: HydraFlowConfig,
) -> np.ndarray:
    """Build one fixed Hydra-style action-flow input ending at current_index.

    The history never crosses the latest SAM anchor. Missing prefix timesteps and
    point slots are represented explicitly by validity channels instead of copied
    observations, preventing padding from looking like stationary motion.
    """
    positions = np.asarray(tracks, dtype=np.float32)
    visible = np.asarray(visibility, dtype=bool)
    conf = np.asarray(confidence, dtype=np.float32)
    indices = np.asarray(selected_indices, dtype=np.int64)
    group_ids = np.asarray(slot_groups, dtype=np.int64)
    valid_slots = np.asarray(slot_valid, dtype=bool)
    if not (0 <= anchor_index <= current_index < len(positions)):
        raise ValueError("expected 0 <= anchor_index <= current_index < frame count")
    if positions.shape[:2] != visible.shape or visible.shape != conf.shape:
        raise ValueError("tracks, visibility, and confidence shapes are inconsistent")
    if not (len(indices) == len(group_ids) == len(valid_slots) == config.points):
        raise ValueError("selected point arrays do not match configured point count")

    feature = np.zeros(
        (config.window_samples, config.points, config.feature_channels), dtype=np.float32
    )
    start = max(anchor_index, current_index - config.window_samples + 1)
    history = np.arange(start, current_index + 1, dtype=np.int64)
    output_start = config.window_samples - len(history)
    scale = np.asarray([config.width, config.height], dtype=np.float32)
    valid_point_indices = np.flatnonzero(valid_slots)
    source_indices = indices[valid_point_indices]
    if len(source_indices):
        xy = positions[history[:, None], source_indices[None]] / scale
        anchor_xy = positions[anchor_index, source_indices] / scale
        displacement = xy - anchor_xy[None]
        velocity = np.diff(xy, axis=0, prepend=xy[:1])
        feature[output_start:, valid_point_indices, 0:2] = xy
        feature[output_start:, valid_point_indices, 2:4] = displacement
        feature[output_start:, valid_point_indices, 4:6] = velocity
        feature[output_start:, valid_point_indices, 6] = visible[
            history[:, None], source_indices[None]
        ]
        feature[output_start:, valid_point_indices, 7] = conf[
            history[:, None], source_indices[None]
        ]
        feature[output_start:, valid_point_indices, 11] = 1.0
    for slot, group_id in enumerate(group_ids):
        if valid_slots[slot]:
            feature[output_start:, slot, 8 + group_id] = 1.0
    return feature


def boundary_targets(sample_frames: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Mark the first sampled observation at or after each reference boundary."""
    samples = np.asarray(sample_frames, dtype=np.int64)
    output = np.zeros(len(samples), dtype=np.int8)
    for boundary in np.asarray(boundaries, dtype=np.int64):
        index = int(np.searchsorted(samples, boundary, side="left"))
        if index < len(output):
            output[index] = 1
    return output
