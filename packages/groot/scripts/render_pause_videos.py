from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from groot_subtask_phase_probe.evaluate import (
    BOUNDARY_THRESHOLDS,
    C_GRID,
    Transform,
    _boundary_labels,
    _boundary_probabilities,
    _episode_map,
    _event_metrics,
    _fit_logistic,
    _stage_xy,
    load_episodes,
    split_episode_ids,
)


@dataclass
class BoundaryPrediction:
    threshold: float
    probabilities: dict[str, np.ndarray]


def _fit_boundary(episodes_list, representation: str, seed: int) -> BoundaryPrediction:
    episodes = _episode_map(episodes_list)
    train_ids, validation_ids, test_ids = split_episode_ids(episodes_list, seed)
    stage_train_x, _ = _stage_xy(episodes, train_ids, representation)
    transform = Transform(common_256=False, seed=seed).fit(stage_train_x)

    xs, ys = [], []
    for episode_id in sorted(train_ids):
        episode = episodes[episode_id]
        valid = episode.terminal == 0
        z = transform.apply(episode.features[representation][valid])
        delta = np.vstack([np.zeros((1, z.shape[1]), dtype=z.dtype), np.diff(z, axis=0)])
        xs.append(np.concatenate([z, delta], axis=1))
        ys.append(_boundary_labels(episode)[valid])
    train_x = np.concatenate(xs)
    train_y = np.concatenate(ys)

    best = None
    for c in C_GRID:
        model = _fit_logistic(train_x, train_y, c)
        validation_probabilities = _boundary_probabilities(
            model, transform, episodes, validation_ids, representation
        )
        for threshold in BOUNDARY_THRESHOLDS:
            metrics = _event_metrics(
                episodes,
                validation_ids,
                validation_probabilities,
                float(threshold),
            )
            score = (
                metrics["boundary_event_f1_at_5"],
                -metrics["false_boundaries_per_episode"],
                -abs(float(threshold) - 0.5),
            )
            if best is None or score > best[0]:
                best = (score, float(threshold), model)
    if best is None:
        raise RuntimeError(f"Could not fit boundary probe for {representation}")
    return BoundaryPrediction(
        threshold=best[1],
        probabilities=_boundary_probabilities(
            best[2], transform, episodes, test_ids, representation
        ),
    )


def _first_event_step(episode, probabilities: np.ndarray, threshold: float) -> int | None:
    above = probabilities >= threshold
    rising = np.flatnonzero(above & np.concatenate([[True], ~above[:-1]]))
    if not rising.size:
        return None
    valid_steps = episode.env_step[episode.terminal == 0]
    return int(valid_steps[int(rising[0])])


def _video_fps(video: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(Fraction(result.stdout.strip()))


def _render_video(
    source: Path,
    output: Path,
    representation: str,
    predicted_step: int | None,
    ground_truth_step: int,
    pause_seconds: float,
) -> None:
    fps = _video_fps(source)
    filters = []
    if predicted_step is not None:
        filters.extend(
            [
                f"setpts=if(gte(N\\,{predicted_step})\\,PTS+{pause_seconds}/TB\\,PTS)",
                f"fps={fps}",
            ]
        )
    prediction_text = "none" if predicted_step is None else str(predicted_step)
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    filters.append(
        "drawtext="
        f"fontfile={font}:text='{representation}  PRED {prediction_text}  GT {ground_truth_step}':"
        "x=20:y=20:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.65"
    )
    if predicted_step is not None:
        pause_start = predicted_step / fps
        pause_end = pause_start + pause_seconds
        filters.append(
            "drawtext="
            f"fontfile={font}:text='PAUSED {pause_seconds:.1f}s':"
            "x=(w-text_w)/2:y=(h-text_h)/2:fontsize=48:fontcolor=yellow:"
            "box=1:boxcolor=black@0.75:"
            f"enable='between(t,{pause_start:.6f},{pause_end:.6f})'"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-vf",
            ",".join(filters),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("pause_videos"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--pause-seconds", type=float, default=10.0)
    args = parser.parse_args()

    episodes_list = load_episodes(args.artifacts)
    episodes = _episode_map(episodes_list)
    representations = sorted(set.intersection(*(set(ep.features) for ep in episodes_list)))
    _, _, test_ids = split_episode_ids(episodes_list, args.seed)
    fitted = {
        representation: _fit_boundary(episodes_list, representation, args.seed)
        for representation in representations
    }

    candidates = [
        episode_id
        for episode_id in sorted(test_ids)
        if episodes[episode_id].success and episodes[episode_id].boundary_step is not None
    ]
    predictions = {
        representation: {
            episode_id: _first_event_step(
                episodes[episode_id], result.probabilities[episode_id], result.threshold
            )
            for episode_id in candidates
        }
        for representation, result in fitted.items()
    }
    selected = max(
        candidates,
        key=lambda episode_id: sum(
            predictions[representation][episode_id] is not None
            for representation in representations
        ),
    )
    episode = episodes[selected]
    source = args.artifacts / selected / "rollout.mp4"
    rows = []
    for representation in representations:
        predicted_step = predictions[representation][selected]
        output = args.output_dir / f"{representation}.mp4"
        _render_video(
            source,
            output,
            representation,
            predicted_step,
            int(episode.boundary_step),
            args.pause_seconds,
        )
        rows.append(
            {
                "representation": representation,
                "predicted_env_step": predicted_step,
                "ground_truth_env_step": int(episode.boundary_step),
                "absolute_error_steps": None
                if predicted_step is None
                else abs(predicted_step - int(episode.boundary_step)),
                "threshold": fitted[representation].threshold,
                "video": str(output),
            }
        )
    summary = {
        "episode": selected,
        "source_video": str(source),
        "pause_seconds": args.pause_seconds,
        "predictions": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
