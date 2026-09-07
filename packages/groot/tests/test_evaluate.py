import numpy as np

from groot_subtask_phase_probe.evaluate import Episode, _paired_tie, split_episode_ids


def _episode(task: str, index: int) -> Episode:
    return Episode(
        episode_id=f"{task}/episode_{index:04d}",
        task=task,
        features={"R0": np.zeros((2, 3), dtype=np.float32)},
        stage=np.array([0, 1]),
        terminal=np.array([0, 0]),
        env_step=np.array([0, 8]),
        boundary_step=8,
        success=True,
    )


def test_episode_split_has_no_leakage_and_is_shared_by_task():
    episodes = [_episode(task, index) for task in ("a", "b") for index in range(10)]
    train, validation, test = split_episode_ids(episodes, 17)
    assert not (train & validation or train & test or validation & test)
    assert train | validation | test == {episode.episode_id for episode in episodes}
    assert len(train) == 12
    assert len(validation) == 4
    assert len(test) == 4


def test_paired_tie_uses_shared_seed_differences():
    best = {
        "rows": [
            {"seed": seed, "score": value} for seed, value in enumerate((0.5, 0.7, 0.5, 0.7, 0.5))
        ]
    }
    tied = {
        "rows": [
            {"seed": seed, "score": value} for seed, value in enumerate((0.4, 0.8, 0.4, 0.8, 0.5))
        ]
    }
    worse = {
        "rows": [
            {"seed": seed, "score": value} for seed, value in enumerate((0.2, 0.4, 0.2, 0.4, 0.2))
        ]
    }
    assert _paired_tie(best, tied, "score")
    assert not _paired_tie(best, worse, "score")
