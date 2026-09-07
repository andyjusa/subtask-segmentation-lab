"""CPU contracts and actual train -> reload -> inference checks."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("infer", ROOT / "scripts/infer.py")
infer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infer)


def test_feature_contract(tmp_path):
    path = tmp_path / "features.npz"
    np.savez(path, vision_mean=np.ones((3, 2048)), sample_frames=np.array([0, 6, 12]))
    x, frames, info = infer.prepare_features(path, "vision_mean", np.zeros(2048), np.ones(2048))
    assert x.dtype == np.float32 and x.shape == (3, 2048)
    assert frames.tolist() == [0, 6, 12]
    assert info["clip_fraction"] == 0


@pytest.mark.parametrize("problem", ["missing", "nan", "dimension", "frames", "scale", "empty"])
def test_reject_invalid_features(tmp_path, problem):
    path = tmp_path / "features.npz"
    x = np.ones((3, 2048))
    frames = np.array([0, 6, 12])
    scale = np.ones(2048)
    if problem == "nan":
        x[0, 0] = np.nan
    elif problem == "dimension":
        x = np.ones((3, 12))
    elif problem == "frames":
        frames = np.array([0, 0, 12])
    elif problem == "scale":
        scale[0] = 0
    elif problem == "empty":
        x, frames = np.ones((0, 2048)), np.array([], dtype=int)
    np.savez(path, **({} if problem == "missing" else {"vision_mean": x, "sample_frames": frames}))
    with pytest.raises(ValueError):
        infer.prepare_features(path, "vision_mean", np.zeros(2048), scale)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    output = tmp_path_factory.mktemp("training")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/reproduce.py"),
            "incline50",
            "--epochs",
            "2",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return output


@pytest.mark.parametrize("name", ["linear", "mlp"])
def test_train_save_load_predict(trained, name):
    episode = json.loads((ROOT / "fixtures/incline50/split.json").read_text())["validation"][0]
    features = (
        ROOT / f"fixtures/incline50/validation/episode_{episode:03d}/groot_backbone_hidden.npz"
    )
    records, info = infer.run(trained, features, name)
    assert info["hidden_key"] == "vision_mean"
    assert info["input_shape"] == [len(records), 2048]
    p = np.array([row["probabilities"] for row in records])
    assert p.shape[1] == 5 and np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-6)
    records2, _ = infer.run(trained, features, name)
    np.testing.assert_allclose([r["probabilities"] for r in records2], p)


def test_cli_missing_model_fails_without_output(tmp_path):
    output = tmp_path / "prediction.jsonl"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/infer.py"),
            "--model-dir",
            str(tmp_path / "absent"),
            "--features",
            "absent.npz",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2 and "--traceback" in proc.stderr
    assert not output.exists()


def test_cli_never_overwrites(tmp_path):
    output = tmp_path / "prediction.jsonl"
    output.write_text("keep me")
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/infer.py"),
            "--model-dir",
            "absent",
            "--features",
            "absent",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2 and output.read_text() == "keep me"
