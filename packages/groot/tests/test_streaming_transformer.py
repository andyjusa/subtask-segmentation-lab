import numpy as np
import pytest

torch = pytest.importorskip("torch")

from groot_subtask_phase_probe.streaming_transformer import (
    CausalSlidingWindowTransformer,
    _paired_difference,
    checkpoint_selection_score,
    pack_causal_window,
    predict_causal_batched_sequence,
    predict_streaming_sequence,
)


def test_pack_causal_window_never_contains_future_values() -> None:
    values = np.arange(30, dtype=np.float32).reshape(10, 3)

    window, valid = pack_causal_window(values, end=4, window_size=3)

    np.testing.assert_array_equal(window[:3], values[2:5])
    assert valid.tolist() == [True, True, True]
    assert 15 not in window


def test_streaming_predictions_are_invariant_to_appended_future() -> None:
    torch.manual_seed(3)
    model = CausalSlidingWindowTransformer(
        input_dim=4,
        window_size=4,
        model_dim=16,
        num_heads=4,
        num_layers=2,
        feedforward_dim=32,
        dropout=0.0,
    ).eval()
    prefix = np.arange(20, dtype=np.float32).reshape(5, 4) / 10
    future = np.full((3, 4), 999.0, dtype=np.float32)

    prefix_stage, prefix_boundary, _ = predict_streaming_sequence(
        model,
        prefix,
        device=torch.device("cpu"),
        measure_latency=False,
    )
    full_stage, full_boundary, _ = predict_streaming_sequence(
        model,
        np.concatenate([prefix, future]),
        device=torch.device("cpu"),
        measure_latency=False,
    )

    np.testing.assert_allclose(full_stage[: len(prefix)], prefix_stage, atol=1e-7)
    np.testing.assert_allclose(full_boundary[: len(prefix)], prefix_boundary, atol=1e-7)


def test_checkpoint_selection_prioritizes_event_f1_over_validation_loss() -> None:
    better_event = checkpoint_selection_score(
        {
            "boundary_event_f1_at_5": 0.5,
            "boundary_event_f1_at_10": 0.6,
            "false_boundaries_per_episode": 1.0,
        },
        {"stage_macro_f1": 0.7},
        validation_loss=2.0,
    )
    better_loss_only = checkpoint_selection_score(
        {
            "boundary_event_f1_at_5": 0.4,
            "boundary_event_f1_at_10": 0.9,
            "false_boundaries_per_episode": 0.0,
        },
        {"stage_macro_f1": 0.99},
        validation_loss=0.1,
    )

    assert better_event > better_loss_only


def test_causal_validation_batch_matches_serial_streaming() -> None:
    torch.manual_seed(7)
    model = CausalSlidingWindowTransformer(
        input_dim=3,
        window_size=5,
        model_dim=12,
        num_heads=3,
        num_layers=1,
        feedforward_dim=24,
        dropout=0.0,
    ).eval()
    values = np.arange(33, dtype=np.float32).reshape(11, 3) / 100

    serial_stage, serial_boundary, _ = predict_streaming_sequence(
        model,
        values,
        device=torch.device("cpu"),
        measure_latency=False,
    )
    batch_stage, batch_boundary = predict_causal_batched_sequence(
        model,
        values,
        device=torch.device("cpu"),
        batch_size=4,
    )

    np.testing.assert_allclose(batch_stage, serial_stage, atol=1e-7)
    np.testing.assert_allclose(batch_boundary, serial_boundary, atol=1e-7)


def test_paired_difference_uses_shared_seed_pairing() -> None:
    rows = [
        {"representation": "A", "projection": "native", "seed": 1, "score": 0.8},
        {"representation": "A", "projection": "native", "seed": 2, "score": 0.4},
        {"representation": "B", "projection": "native", "seed": 1, "score": 0.6},
        {"representation": "B", "projection": "native", "seed": 2, "score": 0.5},
        {"representation": "B", "projection": "native", "seed": 3, "score": 1.0},
    ]

    mean, ci = _paired_difference(rows, ("A", "native"), ("B", "native"), "score")

    assert mean == pytest.approx(0.05)
    assert ci > 0
