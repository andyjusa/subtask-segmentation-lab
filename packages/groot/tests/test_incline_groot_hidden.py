import numpy as np
import torch

from groot_subtask_phase_probe.incline_groot_hidden import (
    align_decoded_frames,
    pool_backbone_batch,
    sample_frame_indices,
)


def test_sample_frame_indices_selects_five_fps_from_thirty() -> None:
    assert sample_frame_indices(14, 30.0, 5.0).tolist() == [0, 6, 12]


def test_pool_backbone_batch_respects_attention_and_image_masks() -> None:
    output = {
        "backbone_features": torch.tensor(
            [[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]], [[2.0, 4.0], [6.0, 8.0], [0.0, 0.0]]]
        ),
        "backbone_attention_mask": torch.tensor([[1, 1, 0], [1, 1, 0]]),
        "image_mask": torch.tensor([[1, 0, 0], [0, 1, 0]]),
    }
    hidden, vision, counts = pool_backbone_batch(output)
    np.testing.assert_array_equal(hidden, [[2.0, 2.0], [4.0, 6.0]])
    np.testing.assert_array_equal(vision, [[1.0, 1.0], [6.0, 8.0]])
    assert counts == {"valid_min": 2, "valid_max": 2, "vision_min": 1, "vision_max": 1}


def test_align_decoded_frames_causally_pads_one_missing_tail_frame() -> None:
    frames = np.asarray([[[[1]]], [[[2]]]], dtype=np.uint8)
    aligned, padding = align_decoded_frames(frames, 3)
    assert padding == 1
    assert aligned[:, 0, 0, 0].tolist() == [1, 2, 2]
