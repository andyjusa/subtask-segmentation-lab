import numpy as np
import torch

from groot_subtask_phase_probe.incline_vision_fusion import (
    ProjectedFusionProbe,
    ResidualFusionProbe,
    TriModalResidualProbe,
    VisionOnlyProbe,
    append_causal_deltas,
    model_factories,
)


def test_fusion_probes_return_five_stage_logits() -> None:
    features = torch.randn(7, 32)

    for factory in model_factories(vision_dim=20, auxiliary_dim=12).values():
        assert factory()(features).shape == (7, 5)


def test_vision_only_probe_ignores_auxiliary_columns() -> None:
    model = VisionOnlyProbe(vision_dim=20).eval()
    vision = torch.randn(7, 20)
    left = torch.cat([vision, torch.zeros(7, 12)], dim=1)
    right = torch.cat([vision, torch.ones(7, 12)], dim=1)

    torch.testing.assert_close(model(left), model(right))


def test_residual_probe_starts_with_a_bounded_auxiliary_gate() -> None:
    model = ResidualFusionProbe(vision_dim=20, auxiliary_dim=12)

    gate = torch.sigmoid(model.auxiliary_gate_logit)

    assert torch.all(gate > 0)
    assert torch.all(gate < 0.15)


def test_projected_probe_keeps_modality_boundary_explicit() -> None:
    model = ProjectedFusionProbe(vision_dim=20, auxiliary_dim=12)

    assert model.vision_dim == 20
    assert model.vision[0].in_features == 20
    assert model.auxiliary[0].in_features == 12


def test_causal_delta_features_use_only_current_and_past_rows() -> None:
    features = np.asarray([[1.0], [3.0], [8.0], [10.0]], dtype=np.float32)

    output = append_causal_deltas(features, (1, 2))

    np.testing.assert_array_equal(
        output,
        [[1.0, 0.0, 0.0], [3.0, 2.0, 2.0], [8.0, 5.0, 7.0], [10.0, 2.0, 7.0]],
    )


def test_tri_modal_probe_keeps_hidden_and_motion_corrections_separate() -> None:
    model = TriModalResidualProbe(vision_dim=20, context_dim=16, auxiliary_dim=12)
    features = torch.randn(7, 48)

    assert model(features).shape == (7, 5)
    assert model.context[0].in_features == 16
    assert model.auxiliary[0].in_features == 12
