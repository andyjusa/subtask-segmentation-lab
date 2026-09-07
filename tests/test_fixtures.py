import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reproduce", ROOT / "scripts/reproduce.py")
reproduce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduce)


def test_checksums():
    assert reproduce.verify()["verified_files"] == 55


def test_episode_split_and_feature_contract():
    fixture = ROOT / "fixtures/incline50"
    split = json.loads((fixture / "split.json").read_text())
    assert len(split["train"]) == 40 and len(split["validation"]) == 10
    assert not set(split["train"]) & set(split["validation"])
    assert set(split["train"] + split["validation"]) == set(range(50))
    count = 0
    for partition in ("train", "validation"):
        for episode in split[partition]:
            with np.load(
                fixture / partition / f"episode_{episode:03d}/groot_backbone_hidden.npz"
            ) as data:
                n = len(data["sample_frames"])
                assert data["vision_mean"].shape == data["hidden_mean"].shape == (n, 2048)
                assert np.all(np.diff(data["sample_frames"]) > 0)
                assert np.isfinite(data["vision_mean"]).all()
                count += n
    assert count == 9496


def test_rollout_label_and_oof_contract():
    from evaluate_rollout_incline_vision_hidden import (
        blocked_stage_folds,
        load_boundaries,
        stage_labels,
    )

    _, boundaries = load_boundaries(ROOT / "fixtures/rollout/boundaries.json")
    with np.load(ROOT / "fixtures/rollout/groot_backbone_hidden.npz") as data:
        labels = stage_labels(data["sample_frames"], boundaries)
        assert data["vision_mean"].shape == (549, 2048)
        covered = []
        for train, test in blocked_stage_folds(labels):
            assert not set(train) & set(test)
            covered.extend(test)
        assert sorted(covered) == list(range(549))


def test_pi05_actions():
    assert np.load(ROOT / "fixtures/pi05/actions.npy").shape == (410, 6)
