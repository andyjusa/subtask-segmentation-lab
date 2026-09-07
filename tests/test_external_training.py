"""Regression: extracted hidden files must train without any tracker artifacts."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_external_hidden_train_infer(tmp_path):
    source = ROOT / "fixtures/incline50"
    hidden = tmp_path / "external-hidden"
    for part, ids in (("train", [3, 4]), ("validation", [0])):
        for episode in ids:
            path = hidden / part / f"episode_{episode:03d}" / "groot_backbone_hidden.npz"
            path.parent.mkdir(parents=True)
            shutil.copy2(source / part / f"episode_{episode:03d}" / path.name, path)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": [3, 4], "validation": [0]}))
    output = tmp_path / "model"
    command = [
        sys.executable,
        str(ROOT / "scripts/reproduce.py"),
        "incline50",
        "--epochs",
        "2",
        "--features-dir",
        str(hidden),
        "--boundaries",
        str(source / "boundaries.csv"),
        "--split",
        str(split),
        "--selection-episodes",
        "1",
        "--label-version",
        "smoke-only",
        "--output",
        str(output),
    ]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stderr
    result = json.loads((output / "results.json").read_text())
    assert result["validation"] == [0] and len(result["train"]) == 1
    assert result["inputs"]["label_version"] == "smoke-only"
    assert len(result["inputs"]["feature_files"]) == 3
    assert result["inputs"]["split_sha256"]
    predictions = tmp_path / "prediction.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/infer.py"),
            "--model-dir",
            str(output),
            "--features",
            str(hidden / "validation/episode_000/groot_backbone_hidden.npz"),
            "--output",
            str(predictions),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    assert len(rows) == 290 and np.isfinite([r["probabilities"] for r in rows]).all()


def test_legacy_hidden_only_without_tracks(tmp_path):
    sys.path.insert(0, str(ROOT / "packages/groot/scripts"))
    from train_tapnextpp_subtask_probe import load_episodes

    hidden = tmp_path / "hidden/train/episode_000"
    hidden.mkdir(parents=True)
    np.savez(
        hidden / "hidden.npz",
        sample_frames=np.array([0, 6]),
        hidden_mean=np.ones((2, 2048), dtype=np.float32),
    )
    result = load_episodes(
        tmp_path / "nonexistent-tracker",
        {"train": [0], "validation": []},
        {0: np.array([1, 2, 3, 4])},
        5,
        hidden_dir=tmp_path / "hidden",
        hidden_filename="hidden.npz",
        exclude_tracker=True,
    )
    assert result[0].features.shape == (2, 2048)
    fused = load_episodes(
        tmp_path / "nonexistent-tracker",
        {"train": [0], "validation": []},
        {0: np.array([1, 2, 3, 4])},
        5,
        hidden_dir=tmp_path / "hidden",
        hidden_filename="hidden.npz",
        exclude_tracker=True,
        robot_features={0: np.full((12, 2), 2.0)},
    )
    np.testing.assert_array_equal(fused[0].features[:, :2], np.full((2, 2), 2.0))
    np.testing.assert_array_equal(fused[0].features[:, 2:], result[0].features)
