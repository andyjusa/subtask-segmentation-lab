"""Validated external inputs for the existing five-stage incline probe."""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(features_dir, boundaries_path, split_path, hidden_key, source_fps, sample_fps):
    from train_tapnextpp_subtask_probe import EpisodeData, stage_labels

    if not all(math.isfinite(v) and v > 0 for v in (source_fps, sample_fps)):
        raise ValueError("FPS values must be positive and finite")
    stride = source_fps / sample_fps
    if stride < 1 or not np.isclose(stride, round(stride)):
        raise ValueError("sample_fps must evenly divide source_fps")
    split = json.loads(split_path.read_text())
    ids = []
    for partition in ("train", "validation"):
        values = split.get(partition)
        if not isinstance(values, list) or not values:
            raise ValueError(f"split requires nonempty {partition} episode list")
        if any(type(v) is not int or v < 0 for v in values):
            raise ValueError("episode IDs must be nonnegative integers")
        ids.extend(values)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate/overlapping episodes in split")
    boundaries = {}
    with boundaries_path.open(newline="") as source:
        for row in csv.DictReader(source):
            episode = int(row["episode"])
            if episode in boundaries:
                raise ValueError(f"duplicate boundary row: episode {episode}")
            seconds = np.array([float(row[f"boundary_{i}_s"]) for i in range(1, 5)])
            if not np.isfinite(seconds).all() or seconds[0] <= 0 or np.any(np.diff(seconds) <= 0):
                raise ValueError(
                    f"episode {episode}: boundaries must be finite, positive, increasing"
                )
            frames = np.rint(seconds * source_fps).astype(np.int64)
            if np.any(np.diff(frames) <= 0):
                raise ValueError(f"episode {episode}: boundaries collapse onto the same frame")
            boundaries[episode] = frames
    episodes, fingerprints = {}, []
    for partition in ("train", "validation"):
        for episode in split[partition]:
            if episode not in boundaries:
                raise ValueError(f"episode {episode}: missing boundary row")
            relative = Path(partition) / f"episode_{episode:03d}" / "groot_backbone_hidden.npz"
            path = features_dir / relative
            with np.load(path, allow_pickle=False) as data:
                x, frames = data[hidden_key], data["sample_frames"]
            if x.ndim != 2 or x.shape[1] != 2048 or len(x) < 5 * max(1, round(3 * sample_fps)):
                raise ValueError(
                    f"episode {episode}: need [N,2048] and at least five 3-second segments"
                )
            if not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x).all():
                raise ValueError(f"episode {episode}: hidden values must be finite floats")
            if frames.shape != (len(x),) or not np.issubdtype(frames.dtype, np.integer):
                raise ValueError(f"episode {episode}: invalid sample_frames")
            if frames[0] != 0 or np.any(frames[1:] <= frames[:-1]):
                raise ValueError(f"episode {episode}: frames must start at zero and increase")
            if not np.all(np.diff(frames.astype(np.int64)) == round(stride)):
                raise ValueError(f"episode {episode}: frame spacing disagrees with supplied FPS")
            labels = stage_labels(frames, boundaries[episode])
            if set(labels.tolist()) != set(range(5)):
                raise ValueError(f"episode {episode}: all five stages must be represented")
            episodes[episode] = EpisodeData(
                episode, x.astype(np.float32), labels, frames.astype(np.int64), boundaries[episode]
            )
            fingerprints.append({"path": str(relative), "sha256": sha256(path)})
    provenance = {
        "features_dir": str(features_dir.resolve()),
        "boundaries_sha256": sha256(boundaries_path),
        "split_sha256": sha256(split_path),
        "feature_files": fingerprints,
        "source_fps": source_fps,
        "sample_fps": sample_fps,
        "input_dimension": 2048,
        "stage_schema": "incline-next-action-5",
        "dataset_revision": None,
        "backbone_revision": None,
    }
    return split, episodes, provenance
