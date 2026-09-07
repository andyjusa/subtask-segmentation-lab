import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.extract_incline_alltracker_tracks import (
    grid_indices,
    model_resolution,
    motion_features,
    select_tracks,
)


def test_model_resolution_preserves_aspect_ratio_and_stride() -> None:
    assert model_resolution(640, 480, 256) == (256, 192)


def test_grid_and_track_selection_produce_400_source_space_points() -> None:
    grid_y, grid_x = grid_indices(192, 256, 20)
    flows = torch.zeros((1, 3, 2, 192, 256), dtype=torch.float32)
    flows[:, :, 0] = 2.0
    flows[:, :, 1] = 1.0
    visconf = torch.ones((1, 3, 2, 192, 256), dtype=torch.float32)
    tracks, confidence, visibility, queries = select_tracks(
        flows, visconf, grid_y, grid_x, 640, 480, 0.1
    )
    assert tracks.shape == (3, 400, 2)
    assert confidence.shape == visibility.shape == (3, 400)
    expected_offset = np.tile(np.asarray([5.0, 2.5]), (400, 1))
    np.testing.assert_allclose(tracks[0] - queries, expected_offset, atol=1e-6)
    assert visibility.all()


def test_motion_features_match_probe_contract() -> None:
    tracks = np.zeros((3, 4, 2), dtype=np.float32)
    tracks[1:, :, 0] = np.asarray([1.0, 2.0])[:, None]
    visibility = np.ones((3, 4), dtype=bool)
    features = motion_features(tracks, visibility)
    assert features.shape == (3, 8)
    assert np.isfinite(features).all()
    assert features[1, 2] == 1.0
