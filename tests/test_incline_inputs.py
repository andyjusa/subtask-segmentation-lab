import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "packages/groot/scripts"))
from incline_inputs import load_inputs


@pytest.mark.parametrize(
    "bad",
    [
        "overlap",
        "duplicate",
        "missing_label",
        "unordered",
        "nan",
        "spacing",
        "fps",
        "short",
        "missing_stage",
    ],
)
def test_reject_bad_training_inputs(tmp_path, bad):
    split = {"train": [1, 2], "validation": [0]}
    if bad == "overlap":
        split["validation"] = [1]
    if bad == "duplicate":
        split["train"] = [1, 1]
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    boundary_path = tmp_path / "boundaries.csv"
    ids = [0, 1] if bad == "missing_label" else [0, 1, 2]
    values = "3,9,6,12" if bad == "unordered" else "3,6,9,12"
    if bad == "missing_stage":
        values = "3,6,9,99"
    boundary_path.write_text(
        "episode,boundary_1_s,boundary_2_s,boundary_3_s,boundary_4_s\n"
        + "".join(f"{i},{values}\n" for i in ids)
    )
    for part, episodes in split.items():
        for episode in episodes:
            path = tmp_path / part / f"episode_{episode:03d}" / "groot_backbone_hidden.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            count = 10 if bad == "short" else 100
            x = np.ones((count, 2048), dtype=np.float32)
            frames = np.arange(count) * (5 if bad == "spacing" else 6)
            if bad == "nan":
                x[0, 0] = np.nan
            np.savez(path, vision_mean=x, sample_frames=frames)
    with pytest.raises(ValueError):
        load_inputs(
            tmp_path, boundary_path, split_path, "vision_mean", 0 if bad == "fps" else 30, 5
        )
