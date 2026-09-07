import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "visualize", Path(__file__).resolve().parents[1] / "scripts/visualize.py"
)
visualize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visualize)


def test_render():
    records = [
        {"source_frame": 0, "stage_id": 0, "probabilities": [1, 0, 0, 0, 0]},
        {"source_frame": 6, "stage_id": 1, "probabilities": [0, 1, 0, 0, 0]},
    ]
    page = visualize.render(records, [2, 3, 4, 5])
    assert "<svg" in page and "참조 라벨" in page and "2개 샘플" in page
    assert "<script" not in page


@pytest.mark.parametrize(
    "records",
    [
        [],
        [{"source_frame": 0, "stage_id": 9, "probabilities": [1] * 5}],
        [{"source_frame": 0, "stage_id": 0, "probabilities": [float("nan")] * 5}],
    ],
)
def test_reject_bad_prediction(records):
    with pytest.raises(ValueError):
        visualize.render(records)
