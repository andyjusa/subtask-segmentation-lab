import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_tapnextpp_subtask_probe import (
    load_episodes,
    load_robot_features,
    ordered_decode,
    point_features,
    stage_labels,
)


def test_stage_labels_advance_at_each_boundary() -> None:
    frames = np.asarray([0, 9, 10, 19, 20, 29, 30, 39, 40])
    boundaries = np.asarray([10, 20, 30, 40])
    assert stage_labels(frames, boundaries).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4]


def test_point_features_have_expected_contract() -> None:
    tracks = np.zeros((3, 4, 2), dtype=np.float32)
    visibility = np.ones((3, 4), dtype=bool)
    motion = np.zeros((3, 8), dtype=np.float32)
    queries = np.zeros((4, 2), dtype=np.float32)
    features = point_features(tracks, visibility, motion, queries, np.asarray([0, 2]))
    assert features.shape == (2, 20)
    assert np.isfinite(features).all()


def test_ordered_decode_recovers_five_clean_segments() -> None:
    probabilities = np.full((100, 5), 0.01, dtype=np.float32)
    for stage in range(5):
        probabilities[stage * 20 : (stage + 1) * 20, stage] = 0.96
    decoded, boundaries = ordered_decode(probabilities, sample_fps=5.0)
    assert boundaries.tolist() == [20, 40, 60, 80]
    assert np.array_equal(decoded, np.repeat(np.arange(5), 20))


def test_load_robot_features_orders_frames_and_combines_state_then_action(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "episode_index": [0, 0],
            "frame_index": [1, 0],
            "observation.state": [[11.0, 12.0], [1.0, 2.0]],
            "action": [[13.0, 14.0], [3.0, 4.0]],
        }
    )
    path = tmp_path / "robot.parquet"
    pq.write_table(table, path)
    features = load_robot_features(path, include_action=True, include_state=True)
    np.testing.assert_array_equal(
        features[0],
        np.asarray([[1.0, 2.0, 3.0, 4.0], [11.0, 12.0, 13.0, 14.0]]),
    )


def test_load_robot_features_can_remove_both_action_grippers(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "episode_index": [0],
            "frame_index": [0],
            "action": [list(range(16))],
        }
    )
    path = tmp_path / "robot.parquet"
    pq.write_table(table, path)
    features = load_robot_features(
        path,
        include_action=True,
        include_state=False,
        exclude_action_grippers=True,
    )
    assert features[0].shape == (1, 14)
    assert 7 not in features[0][0]
    assert 15 not in features[0][0]


def test_load_episodes_appends_aligned_hidden_features(tmp_path) -> None:
    tracks_dir = tmp_path / "tracks"
    hidden_dir = tmp_path / "hidden"
    for root in (tracks_dir, hidden_dir):
        episode_dir = root / "train" / "episode_000"
        episode_dir.mkdir(parents=True)
    np.savez_compressed(
        tracks_dir / "train" / "episode_000" / "tracks.npz",
        tracks=np.zeros((12, 2, 2), dtype=np.float32),
        visibility=np.ones((12, 2), dtype=bool),
        motion_features=np.zeros((12, 8), dtype=np.float32),
        query_points=np.zeros((2, 2), dtype=np.float32),
    )
    np.savez_compressed(
        hidden_dir / "train" / "episode_000" / "hidden.npz",
        sample_frames=np.asarray([0, 6]),
        hidden_mean=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16),
    )
    episodes = load_episodes(
        tracks_dir,
        {"train": [0], "validation": []},
        {0: np.asarray([2, 4, 8, 10])},
        sample_fps=5.0,
        tracks_filename="tracks.npz",
        hidden_dir=hidden_dir,
        hidden_filename="hidden.npz",
    )
    assert episodes[0].features.shape == (2, 16)
    np.testing.assert_array_equal(episodes[0].features[:, -2:], [[1.0, 2.0], [3.0, 4.0]])

    hidden_only = load_episodes(
        tracks_dir,
        {"train": [0], "validation": []},
        {0: np.asarray([2, 4, 8, 10])},
        sample_fps=5.0,
        tracks_filename="tracks.npz",
        hidden_dir=hidden_dir,
        hidden_filename="hidden.npz",
        exclude_tracker=True,
    )
    assert hidden_only[0].features.shape == (2, 2)
