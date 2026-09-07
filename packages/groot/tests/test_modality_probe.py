from __future__ import annotations

import numpy as np

from groot_subtask_phase_probe.evaluate import Episode
from groot_subtask_phase_probe.modality_probe import (
    add_modality_representations,
    derive_instruction_mean,
    evaluate_one,
)


def test_derive_instruction_mean_recovers_non_image_pool_without_rounding():
    vision = np.array([[1.0, 3.0]], dtype=np.float32)
    instruction = np.array([[5.0, 7.0]], dtype=np.float32)
    joint = (vision * 4 + instruction * 2) / 6
    recovered = derive_instruction_mean(
        joint,
        vision,
        valid_tokens=6,
        vision_tokens=4,
    )
    np.testing.assert_allclose(recovered, instruction, atol=1e-6)


def test_add_modality_representations_builds_two_and_three_way_fusions():
    steps = 3
    vision = np.full((steps, 2), 1.0, dtype=np.float32)
    instruction = np.full((steps, 2), 3.0, dtype=np.float32)
    joint = (vision * 4 + instruction * 2) / 6
    episode = Episode(
        episode_id="task/episode_0000",
        task="task",
        features={
            "R1_action_chunk": np.full((steps, 4), 5.0, dtype=np.float32),
            "R3_vlm_hidden": joint,
            "R3b_vision_tokens": vision,
            "R5_final_dit_hidden": np.full((steps, 3), 7.0, dtype=np.float32),
        },
        stage=np.array([0, 0, 1]),
        terminal=np.zeros(steps, dtype=np.int64),
        env_step=np.arange(steps),
        boundary_step=2,
        success=True,
    )
    add_modality_representations(
        episode,
        {"task": {"valid": 6, "vision": 4, "instruction": 2}},
    )
    np.testing.assert_allclose(episode.features["R3c_instruction_tokens"], instruction)
    assert episode.features["F1_vision_instruction"].shape == (steps, 4)
    assert episode.features["F4_vision_instruction_action_output"].shape == (steps, 8)
    assert episode.features["F7_vision_instruction_action_hidden"].shape == (steps, 7)


def test_linear_and_mlp_share_episode_split_and_learn_separable_stage():
    episodes = []
    for episode_index in range(10):
        stage = np.repeat(np.array([0, 1], dtype=np.int64), 6)
        signal = np.column_stack(
            [stage.astype(np.float32), np.arange(12, dtype=np.float32) / 12]
        )
        episodes.append(
            Episode(
                episode_id=f"task/episode_{episode_index:04d}",
                task="task",
                features={"signal": signal},
                stage=stage,
                terminal=np.zeros(12, dtype=np.int64),
                env_step=np.arange(12, dtype=np.int64),
                boundary_step=6,
                success=True,
            )
        )

    for classifier in ("linear", "mlp"):
        result = evaluate_one(episodes, "signal", classifier, False, 17)
        assert result["episode_split_leakage"] == 0
        assert result["stage_macro_f1"] >= 0.95
