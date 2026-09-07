import numpy as np
import pytest

torch = pytest.importorskip("torch")

from groot_subtask_phase_probe.features import GrootFeatureCapture


class FakeBackbone(torch.nn.Module):
    def forward(self, _value):
        return {
            "backbone_features": torch.arange(24, dtype=torch.float32).reshape(1, 3, 8),
            "backbone_attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.bool),
            "image_mask": torch.tensor([[1, 0, 0]], dtype=torch.bool),
        }


class FakeHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_inference_timesteps = 4
        self.action_horizon = 2
        self.state_encoder = torch.nn.Identity()
        self.action_encoder = torch.nn.Identity()
        self.model = torch.nn.Identity()


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FakeBackbone()
        self.action_head = FakeHead()


class FakePolicy:
    def __init__(self):
        self.model = FakeModel()


def test_capture_uses_final_denoise_and_expected_pooling():
    policy = FakePolicy()
    capture = GrootFeatureCapture(policy)
    capture.begin_step()
    policy.model.backbone(None)
    policy.model.action_head.state_encoder(torch.ones(1, 1, 4))
    for iteration in range(4):
        policy.model.action_head.action_encoder(torch.full((1, 2, 6), float(iteration)))
        policy.model.action_head.model(torch.full((1, 3, 5), float(iteration + 10)))
    snapshot = capture.snapshot(np.arange(4), np.arange(6).reshape(2, 3))
    np.testing.assert_allclose(snapshot.arrays["R4_action_encoder"], 3)
    np.testing.assert_allclose(snapshot.arrays["R5_final_dit_hidden"], 13)
    assert snapshot.arrays["R3_vlm_hidden"].shape == (8,)
    assert snapshot.arrays["R3b_vision_tokens"].shape == (8,)
    np.testing.assert_allclose(snapshot.arrays["R3c_instruction_tokens"], np.arange(8, 16))
    assert snapshot.arrays["F1_vision_instruction"].shape == (16,)
    assert snapshot.token_counts == {"valid": 2, "vision": 1, "instruction": 1}
    assert set(snapshot.extraction_latency_ms) == set(snapshot.arrays)
    assert snapshot.snapshot_latency_ms >= 0
    assert snapshot.denoise_calls == 4
    capture.close()


def test_capture_rejects_missing_denoise_iteration():
    policy = FakePolicy()
    capture = GrootFeatureCapture(policy)
    capture.begin_step()
    policy.model.backbone(None)
    policy.model.action_head.state_encoder(torch.ones(1, 1, 4))
    policy.model.action_head.action_encoder(torch.ones(1, 2, 6))
    policy.model.action_head.model(torch.ones(1, 3, 5))
    with pytest.raises(RuntimeError, match="Expected 4 denoise calls"):
        capture.snapshot(np.arange(4), np.arange(6).reshape(2, 3))
    capture.close()
