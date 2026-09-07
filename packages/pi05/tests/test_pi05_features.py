import torch

from vla_subtask_phase_probe.pi05_features import Pi05RepresentationCollector


def test_vlm_hook_accepts_pi_gemma_norm_tuple() -> None:
    collector = object.__new__(Pi05RepresentationCollector)
    hidden = torch.randn(1, 4, 8)
    gate = torch.zeros(1, 4, 8)

    collector._capture_vlm_hidden(None, (), (hidden, gate))

    assert torch.equal(collector._latest_vlm_hidden, hidden)
