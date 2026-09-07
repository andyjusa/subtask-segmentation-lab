from __future__ import annotations

import numpy as np
import pytest

from groot_subtask_phase_probe.online_pause import (
    BoundaryProbeArtifact,
    GrootOnlinePauseGate,
    OnlineBoundaryDetector,
    OnlinePauseController,
    PauseConfig,
)


def _artifact(*, threshold: float = 0.5) -> BoundaryProbeArtifact:
    # Probability depends only on the current one-dimensional feature.
    return BoundaryProbeArtifact(
        representation="R0_raw_state",
        mean=np.array([0.0], dtype=np.float32),
        scale=np.array([1.0], dtype=np.float32),
        coefficient=np.array([10.0, 0.0], dtype=np.float32),
        intercept=-5.0,
        threshold=threshold,
    )


def test_detector_emits_only_the_first_rising_boundary() -> None:
    detector = OnlineBoundaryDetector(_artifact())

    assert detector.observe(np.array([0.0]), env_step=0) is None
    event = detector.observe(np.array([1.0]), env_step=8)
    assert event is not None
    assert event.env_step == 8
    assert event.representation == "R0_raw_state"

    assert detector.observe(np.array([1.0]), env_step=16) is None
    assert detector.observe(np.array([0.0]), env_step=24) is None
    assert detector.observe(np.array([1.0]), env_step=32) is None

    detector.reset()
    assert detector.observe(np.array([1.0]), env_step=0) is not None


def test_probe_artifact_round_trip(tmp_path) -> None:
    path = tmp_path / "probe.npz"
    expected = _artifact(threshold=0.75)
    expected.save(path)
    actual = BoundaryProbeArtifact.load(path)

    assert actual.representation == expected.representation
    assert actual.intercept == expected.intercept
    assert actual.threshold == expected.threshold
    np.testing.assert_array_equal(actual.mean, expected.mean)
    np.testing.assert_array_equal(actual.scale, expected.scale)
    np.testing.assert_array_equal(actual.coefficient, expected.coefficient)


def test_pause_advances_environment_without_running_inference() -> None:
    calls = {"inference": 1, "environment": 0, "reset": 0}

    class Resettable:
        def reset(self) -> None:
            calls["reset"] += 1

    class Environment:
        n_action_steps = 2

        def step(self, action):
            calls["environment"] += 1
            np.testing.assert_array_equal(action["action.x"], np.zeros((4, 1)))
            np.testing.assert_array_equal(action["action.gripper"], np.ones((4, 1)))
            return (
                {"frame": calls["environment"]},
                0.0,
                False,
                False,
                {
                    "success": [False, False],
                    "n_env_steps": 2,
                },
            )

    controller = OnlinePauseController(PauseConfig(seconds=2.0, control_hz=2.0))
    result = controller.pause_environment(
        Environment(),
        observation={"frame": 0},
        hold_action={
            "action.x": np.zeros((4, 1), dtype=np.float32),
            "action.gripper": np.ones((4, 1), dtype=np.float32),
        },
        env_step=8,
        probability=0.9,
        reset_after_resume=(Resettable(),),
    )

    assert result.simulated_seconds == 2.0
    assert result.physics_steps == 4
    assert result.macro_steps == 2
    assert result.observation == {"frame": 2}
    assert calls == {"inference": 1, "environment": 2, "reset": 1}


def test_libero_hold_uses_last_actually_executed_gripper_command() -> None:
    pytest.importorskip("torch")
    from groot_subtask_phase_probe.collect import _libero_hold_action

    template = {
        "action.x": np.full((16, 1), 0.5, dtype=np.float32),
        "action.gripper": np.zeros((16, 1), dtype=np.float32),
    }
    previous = {
        "action.x": np.full((16, 1), 0.25, dtype=np.float32),
        "action.gripper": np.arange(16, dtype=np.float32).reshape(16, 1),
    }

    hold = _libero_hold_action(template, previous, executed_action_steps=8)

    np.testing.assert_array_equal(hold["action.x"], np.zeros((16, 1), dtype=np.float32))
    np.testing.assert_array_equal(hold["action.gripper"], np.full((16, 1), 7.0, dtype=np.float32))


def test_collect_discards_trigger_action_and_reinfers(monkeypatch, tmp_path) -> None:
    pytest.importorskip("torch")
    from groot_subtask_phase_probe import collect
    from groot_subtask_phase_probe.features import FeatureSnapshot

    observation = {
        "state.joints": np.array([[1.0]], dtype=np.float32),
        "video.image": np.zeros((1, 2, 2, 3), dtype=np.uint8),
        "annotation.human.coarse_action": "task",
    }

    class FakeEnvironment:
        def __init__(self) -> None:
            self.step_calls = 0
            self.actions: list[dict[str, np.ndarray]] = []

        def reset(self, seed: int):
            return observation, {}

        def step(self, action):
            self.step_calls += 1
            self.actions.append(action)
            is_hold = bool(np.all(np.asarray(action["action.joints"]) == 0))
            return (
                observation,
                0.0,
                False,
                False,
                {
                    "probe_predicate_1": [False],
                    "probe_task_success": [not is_hold],
                    "success": [not is_hold],
                    "n_env_steps": 1,
                },
            )

        def close(self) -> None:
            pass

    class FakePolicy:
        def __init__(self) -> None:
            self.calls = 0
            self.reset_calls = 0

        def get_action(self, _observation):
            self.calls += 1
            value = np.array([[[float(self.calls)]]], dtype=np.float32)
            return {"action.joints": value}, {}

        def reset(self) -> None:
            self.reset_calls += 1

    class FakeCapture:
        class Head:
            num_inference_timesteps = 4

        head = Head()

        def begin_step(self) -> None:
            pass

        def snapshot(self, raw_state, decoded_action_chunk) -> FeatureSnapshot:
            return FeatureSnapshot(
                arrays={
                    "R0_raw_state": np.array([1.0], dtype=np.float32),
                    "R1_action_chunk": decoded_action_chunk,
                },
                hook_latency_ms=0.0,
                extraction_latency_ms={"R0_raw_state": 0.0, "R1_action_chunk": 0.0},
                snapshot_latency_ms=0.0,
                shapes={},
                token_counts={},
                denoise_calls=4,
            )

    environment = FakeEnvironment()
    policy = FakePolicy()
    gate = GrootOnlinePauseGate(
        OnlineBoundaryDetector(_artifact()),
        OnlinePauseController(PauseConfig(seconds=2, control_hz=2)),
    )
    monkeypatch.setattr(collect, "_build_environment", lambda *_args: environment)
    monkeypatch.setattr(collect, "_process_rss_bytes", lambda: 0)
    monkeypatch.setattr(collect, "_process_peak_rss_bytes", lambda: 0)
    monkeypatch.setattr(collect.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(collect.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(collect.torch.cuda, "max_memory_allocated", lambda: 0)

    task = type(
        "Task",
        (),
        {
            "slug": "fake",
            "env_name": "fake-v0",
            "predicate_1": ("first",),
            "predicate_1_description": "first",
            "predicate_2_description": "done",
        },
    )()
    metadata = collect.collect_episode(
        task=task,
        episode_index=0,
        seed=1,
        episode_dir=tmp_path / "episode_0000",
        sim_policy=policy,
        capture=FakeCapture(),
        contract=object(),
        max_steps=8,
        online_pause_gate=gate,
    )

    assert policy.calls == 3  # discarded trigger, fresh action, terminal observation
    assert policy.reset_calls == 1
    assert environment.step_calls == 5  # four hold chunks plus one resumed action
    for hold_action in environment.actions[:4]:
        np.testing.assert_array_equal(
            hold_action["action.joints"], np.array([[0.0]], dtype=np.float32)
        )
    np.testing.assert_array_equal(
        environment.actions[-1]["action.joints"], np.array([[2.0]], dtype=np.float32)
    )
    assert metadata["discarded_action_chunks"] == 1
    event = metadata["online_pause_events"][0]
    assert event["simulated_seconds"] == 2.0
    assert event["physics_steps"] == 4
    assert event["inference_calls_during_pause"] == 0
