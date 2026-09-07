import json

import numpy as np

from groot_subtask_phase_probe.color_stop_probe import paired_split_ids
from groot_subtask_phase_probe.evaluate import Episode


def test_paired_split_keeps_counterfactual_twins_together(tmp_path) -> None:
    episodes = []
    for pair_id in range(10):
        for condition in ("change", "no_change"):
            episode_id = f"color_stop/pair_{pair_id:03d}_{condition}"
            episode_dir = tmp_path / episode_id
            episode_dir.mkdir(parents=True)
            (episode_dir / "metadata.json").write_text(json.dumps({"pair_id": pair_id}))
            episodes.append(
                Episode(
                    episode_id=episode_id,
                    task="color_stop",
                    features={"x": np.zeros((2, 1), dtype=np.float32)},
                    stage=np.zeros(2, dtype=np.int64),
                    terminal=np.zeros(2, dtype=np.int64),
                    boundary_step=None,
                    env_step=np.arange(2),
                    success=False,
                    mean_inference_latency_ms=0.0,
                    peak_vram_bytes=0,
                    host_rss_delta_bytes=0,
                )
            )
    splits = paired_split_ids(episodes, tmp_path, seed=17)
    assert not (splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2])
    for pair_id in range(10):
        twins = {
            f"color_stop/pair_{pair_id:03d}_change",
            f"color_stop/pair_{pair_id:03d}_no_change",
        }
        assert sum(twins <= split for split in splits) == 1
