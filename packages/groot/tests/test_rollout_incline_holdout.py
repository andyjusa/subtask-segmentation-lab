import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_rollout_incline_holdout import (
    Boundary,
    load_boundaries,
    stage_at_frame,
    stage_ranges,
    write_frame_labels,
)

BOUNDARIES = [
    Boundary(stage=1, frame=10, time_seconds=1.0),
    Boundary(stage=2, frame=20, time_seconds=2.0),
    Boundary(stage=3, frame=30, time_seconds=3.0),
    Boundary(stage=4, frame=40, time_seconds=4.0),
]


def test_stage_at_frame_is_causal_at_boundary() -> None:
    assert stage_at_frame(9, BOUNDARIES) == 0
    assert stage_at_frame(10, BOUNDARIES) == 1
    assert stage_at_frame(39, BOUNDARIES) == 3
    assert stage_at_frame(40, BOUNDARIES) == 4


def test_stage_ranges_cover_episode_without_overlap() -> None:
    assert stage_ranges(50, BOUNDARIES) == [
        (0, 10),
        (10, 20),
        (20, 30),
        (30, 40),
        (40, 50),
    ]


def test_write_frame_labels_samples_without_future_frames(tmp_path: Path) -> None:
    output = tmp_path / "labels.csv"
    count = write_frame_labels(output, 50, 10.0, 5.0, BOUNDARIES)
    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert count == 25
    assert rows[4]["source_frame"] == "8"
    assert rows[5]["source_frame"] == "10"
    assert rows[5]["stage"] == "1"


def test_load_boundaries_rejects_missing_stage(tmp_path: Path) -> None:
    path = tmp_path / "boundaries.json"
    path.write_text(
        '{"successful_boundaries": ['
        '{"stage": 1, "frame": 10, "time_seconds": 1},'
        '{"stage": 3, "frame": 20, "time_seconds": 2}'
        "]}",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stages 1, 2, 3, 4"):
        load_boundaries(path)
