from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from groot_subtask_phase_probe.evaluate import Episode, load_episodes
from groot_subtask_phase_probe.streaming_metrics import rising_event_steps
from groot_subtask_phase_probe.streaming_transformer import (
    CausalSlidingWindowTransformer,
    predict_streaming_sequence,
)


def _project(values: np.ndarray, checkpoint: dict[str, object]) -> np.ndarray:
    mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
    projected = (np.asarray(values, dtype=np.float32) - mean) / scale
    if "pca_components" in checkpoint:
        components = np.asarray(checkpoint["pca_components"], dtype=np.float32)
        pca_mean = np.asarray(checkpoint["pca_mean"], dtype=np.float32)
        projected = (projected - pca_mean) @ components.T
        output_dim = int(checkpoint.get("pca_output_dim", projected.shape[1]))
        if projected.shape[1] < output_dim:
            projected = np.pad(projected, ((0, 0), (0, output_dim - projected.shape[1])))
    return projected.astype(np.float32, copy=False)


def _load_model(path: Path, device: torch.device) -> tuple[dict[str, object], object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = CausalSlidingWindowTransformer(
        input_dim=int(checkpoint["input_dim"]),
        **checkpoint["model_config"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return checkpoint, model


def _episode_prediction(
    episode: Episode,
    checkpoint: dict[str, object],
    model: CausalSlidingWindowTransformer,
    device: torch.device,
) -> dict[str, object]:
    valid = (episode.terminal == 0) & (episode.stage >= 0)
    values = _project(episode.features[str(checkpoint["representation"])][valid], checkpoint)
    _, probabilities, _ = predict_streaming_sequence(
        model,
        values,
        device=device,
        measure_latency=False,
    )
    env_steps = episode.env_step[valid]
    threshold = float(checkpoint["boundary_threshold"])
    predicted = rising_event_steps(probabilities, env_steps, threshold=threshold)
    first_prediction = int(predicted[0]) if len(predicted) else None
    ground_truth = episode.boundary_step

    if ground_truth is None:
        category = 3 if first_prediction is not None else 0
        error = 0 if first_prediction is None else 1_000 - first_prediction
    elif first_prediction is None:
        category = 4
        error = 10_000
    else:
        error = abs(first_prediction - int(ground_truth))
        category = 2 if error > 5 else (1 if len(predicted) > 1 else 0)

    return {
        "episode_id": episode.episode_id,
        "ground_truth_step": ground_truth,
        "predicted_steps": [int(step) for step in predicted],
        "first_prediction_step": first_prediction,
        "absolute_error_steps": None if ground_truth is None or first_prediction is None else error,
        "threshold": threshold,
        "probabilities": probabilities.tolist(),
        "env_steps": env_steps.tolist(),
        "rank": [category, error, len(predicted)],
    }


def _annotate_video(
    source: Path,
    output: Path,
    *,
    seed: int,
    ground_truth: int | None,
    predicted: list[int],
    probabilities: list[float],
    threshold: float,
    label: str,
) -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to annotate the selected video")
    prediction_text = "missed" if not predicted else "-".join(map(str, predicted))
    max_probability = max(probabilities, default=0.0)
    filters = [
        "drawbox=x=0:y=0:w=iw:h=62:color=black@0.72:t=fill",
        (
            f"drawtext=text='C2 {label} case  seed "
            f"{seed}':x=10:y=8:fontsize=16:fontcolor=white"
        ),
        (
            f"drawtext=text='GT {ground_truth}  predicted {prediction_text}'"
            ":x=10:y=30:fontsize=16:fontcolor=white"
        ),
        (
            f"drawtext=text='threshold {threshold:.2f}  max probability {max_probability:.3f}'"
            ":x=10:y=50:fontsize=13:fontcolor=white"
        ),
    ]
    if ground_truth is not None:
        filters.append(
            f"drawbox=x=3:y=3:w=iw-6:h=ih-6:color=green:t=6:enable='eq(n,{ground_truth})'"
        )
    for step in predicted:
        filters.append(
            f"drawbox=x=10:y=10:w=iw-20:h=ih-20:color=red:t=5:enable='eq(n,{step})'"
        )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vf",
            ",".join(filters),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts_color_stop"))
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection", choices=("best", "worst"), default="worst")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    episodes = {episode.episode_id: episode for episode in load_episodes(args.artifacts)}
    candidates: list[dict[str, object]] = []
    for checkpoint_path in sorted(args.checkpoints.glob("*.pt")):
        checkpoint, model = _load_model(checkpoint_path, device)
        seed = int(checkpoint["seed"])
        for episode_id in checkpoint["test_episode_ids"]:
            result = _episode_prediction(episodes[episode_id], checkpoint, model, device)
            result["seed"] = seed
            result["checkpoint"] = str(checkpoint_path)
            candidates.append(result)

    if args.selection == "worst":
        selected = max(candidates, key=lambda item: tuple(item["rank"]))
    else:
        successful = [
            item
            for item in candidates
            if item["ground_truth_step"] is not None and item["first_prediction_step"] is not None
        ]
        if not successful:
            raise RuntimeError("C2 has no detected change episode")

        def best_rank(item: dict[str, object]) -> tuple[float, float, float]:
            first_step = int(item["first_prediction_step"])
            error = abs(first_step - int(item["ground_truth_step"]))
            extra_events = len(item["predicted_steps"]) - 1
            confidence = float(item["probabilities"][first_step])
            return (-float(error), -float(extra_events), confidence)

        selected = max(successful, key=best_rank)
    episode = episodes[str(selected["episode_id"])]
    source = args.artifacts / episode.episode_id / "rollout.mp4"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"c2_{args.selection}_case_annotated.mp4"
    _annotate_video(
        source,
        output,
        seed=int(selected["seed"]),
        ground_truth=selected["ground_truth_step"],
        predicted=selected["predicted_steps"],
        probabilities=selected["probabilities"],
        threshold=float(selected["threshold"]),
        label=args.selection,
    )
    report = {key: value for key, value in selected.items() if key != "probabilities"}
    report["source_video"] = str(source)
    report["annotated_video"] = str(output)
    (args.output_dir / f"c2_{args.selection}_case.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
