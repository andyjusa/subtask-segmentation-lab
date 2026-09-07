from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class EpisodeData:
    episode: int
    features: np.ndarray
    labels: np.ndarray
    sample_frames: np.ndarray
    reference_boundaries: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracks-dir",
        type=Path,
        default=Path("reports/incline_new/tapnextpp_tracks_50_cuda"),
    )
    parser.add_argument(
        "--boundaries",
        type=Path,
        default=Path("reports/incline_new/segmented_50/boundaries.csv"),
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("reports/incline_new/segmented_50/split.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/incline_new/tapnextpp_subtask_probe"),
    )
    parser.add_argument("--tracks-filename", default="tapnextpp_tracks.npz")
    parser.add_argument("--tracker-name", default="TAPNext++")
    parser.add_argument("--robot-data", type=Path)
    parser.add_argument("--include-action", action="store_true")
    parser.add_argument("--include-state", action="store_true")
    parser.add_argument("--exclude-action-grippers", action="store_true")
    parser.add_argument("--hidden-dir", type=Path)
    parser.add_argument("--hidden-key", default="hidden_mean")
    parser.add_argument("--hidden-filename", default="groot_backbone_hidden.npz")
    parser.add_argument("--exclude-tracker", action="store_true")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser.parse_args()


def stage_labels(sample_frames: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    return np.searchsorted(boundaries, sample_frames, side="right").astype(np.int64)


def point_features(
    tracks: np.ndarray,
    visibility: np.ndarray,
    motion_features: np.ndarray,
    query_points: np.ndarray,
    sample_frames: np.ndarray,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    scale = np.asarray([width, height], dtype=np.float32)
    displacement = tracks[sample_frames].astype(np.float32) - query_points.astype(np.float32)
    displacement = np.clip(displacement / scale, -2.0, 2.0).reshape(len(sample_frames), -1)
    visible = visibility[sample_frames].astype(np.float32)
    motion = motion_features[sample_frames].astype(np.float32)
    return np.concatenate([displacement, visible, motion], axis=1)


def load_boundaries(path: Path) -> dict[int, np.ndarray]:
    output = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            output[int(row["episode"])] = np.rint(
                np.asarray([float(row[f"boundary_{index}_s"]) for index in range(1, 5)]) * 30.0
            ).astype(np.int32)
    return output


def load_robot_features(
    path: Path,
    include_action: bool,
    include_state: bool,
    exclude_action_grippers: bool = False,
) -> dict[int, np.ndarray]:
    if not include_action and not include_state:
        return {}
    import pyarrow.parquet as pq

    columns = ["episode_index", "frame_index"]
    if include_state:
        columns.append("observation.state")
    if include_action:
        columns.append("action")
    table = pq.read_table(path, columns=columns)
    episode_indices = table["episode_index"].to_numpy()
    frame_indices = table["frame_index"].to_numpy()
    arrays = {
        name: np.asarray(table[name].to_pylist(), dtype=np.float32)
        for name in columns[2:]
    }
    if exclude_action_grippers and "action" in arrays:
        arrays["action"] = np.delete(arrays["action"], [7, 15], axis=1)
    output = {}
    for episode in sorted(np.unique(episode_indices)):
        selected = episode_indices == episode
        order = np.argsort(frame_indices[selected])
        output[int(episode)] = np.concatenate(
            [arrays[name][selected][order] for name in columns[2:]],
            axis=1,
        )
    return output


def load_episodes(
    tracks_dir: Path,
    split: dict,
    boundaries: dict[int, np.ndarray],
    sample_fps: float,
    tracks_filename: str = "tapnextpp_tracks.npz",
    robot_features: dict[int, np.ndarray] | None = None,
    hidden_dir: Path | None = None,
    hidden_key: str = "hidden_mean",
    hidden_filename: str = "groot_backbone_hidden.npz",
    exclude_tracker: bool = False,
) -> dict[int, EpisodeData]:
    stride = round(30.0 / sample_fps)
    if not np.isclose(30.0 / stride, sample_fps):
        raise ValueError("sample_fps must evenly divide 30 FPS")
    split_by_episode = {int(episode): "train" for episode in split["train"]}
    split_by_episode.update({int(episode): "validation" for episode in split["validation"]})
    episodes = {}
    for episode, split_name in sorted(split_by_episode.items()):
        path = tracks_dir / split_name / f"episode_{episode:03d}" / tracks_filename
        with np.load(path) as arrays:
            sample_frames = np.arange(0, len(arrays["tracks"]), stride, dtype=np.int32)
            features = (
                np.empty((len(sample_frames), 0), dtype=np.float32)
                if exclude_tracker
                else point_features(
                    arrays["tracks"],
                    arrays["visibility"],
                    arrays["motion_features"],
                    arrays["query_points"],
                    sample_frames,
                )
            )
            if robot_features:
                episode_robot_features = robot_features[episode]
                if len(episode_robot_features) != len(arrays["tracks"]):
                    raise ValueError(
                        f"Episode {episode} frame mismatch: tracker={len(arrays['tracks'])}, "
                        f"robot={len(episode_robot_features)}"
                    )
                features = np.concatenate(
                    [features, episode_robot_features[sample_frames]],
                    axis=1,
                )
            if hidden_dir is not None:
                hidden_path = (
                    hidden_dir
                    / split_name
                    / f"episode_{episode:03d}"
                    / hidden_filename
                )
                with np.load(hidden_path) as hidden_arrays:
                    hidden_frames = np.asarray(hidden_arrays["sample_frames"], dtype=np.int32)
                    hidden = np.asarray(hidden_arrays[hidden_key], dtype=np.float32)
                if not np.array_equal(hidden_frames, sample_frames):
                    raise ValueError(
                        f"Episode {episode} hidden frame mismatch: "
                        f"hidden={len(hidden_frames)}, tracker={len(sample_frames)}"
                    )
                features = np.concatenate([features, hidden], axis=1)
        episodes[episode] = EpisodeData(
            episode=episode,
            features=features,
            labels=stage_labels(sample_frames, boundaries[episode]),
            sample_frames=sample_frames,
            reference_boundaries=boundaries[episode],
        )
    return episodes


def concatenate(episodes: dict[int, EpisodeData], ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([episodes[episode].features for episode in ids]),
        np.concatenate([episodes[episode].labels for episode in ids]),
    )


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_model(name: str, input_dim: int) -> nn.Module:
    if name == "linear":
        return nn.Linear(input_dim, 5)
    if name == "mlp":
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 5),
        )
    raise ValueError(name)


def predict_probabilities(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            output.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(output)


def train_model(
    name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    selection_x: np.ndarray,
    selection_y: np.ndarray,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
) -> tuple[nn.Module, list[dict], int]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = make_model(name, train_x.shape[1]).to(device)
    counts = np.bincount(train_y, minlength=5).astype(np.float32)
    class_weights = counts.sum() / (len(counts) * counts)
    loss_function = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    best_f1 = -1.0
    best_epoch = -1
    best_state = None
    stale = 0
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        probabilities = predict_probabilities(model, selection_x, device, batch_size)
        prediction = probabilities.argmax(axis=1)
        score = float(f1_score(selection_y, prediction, average="macro"))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "selection_f1": score})
        if score > best_f1 + 1e-5:
            best_f1 = score
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, history, best_epoch


def ordered_decode(
    probabilities: np.ndarray,
    sample_fps: float,
    min_segment_s: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    smoothed = gaussian_filter1d(probabilities, sigma=2.0, axis=0)
    log_probabilities = np.log(np.clip(smoothed, 1e-7, 1.0))
    cumulative = np.vstack([np.zeros((1, 5)), log_probabilities.cumsum(axis=0)])
    count = len(probabilities)
    min_length = max(1, round(min_segment_s * sample_fps))
    dp = np.full((5, count + 1), -1e30, dtype=np.float64)
    previous = np.full((5, count + 1), -1, dtype=np.int32)
    for end in range(min_length, count + 1):
        dp[0, end] = cumulative[end, 0]
    for stage in range(1, 5):
        for end in range((stage + 1) * min_length, count + 1):
            starts = np.arange(stage * min_length, end - min_length + 1)
            scores = dp[stage - 1, starts] + cumulative[end, stage] - cumulative[starts, stage]
            best = int(np.argmax(scores))
            dp[stage, end] = scores[best]
            previous[stage, end] = starts[best]
    end = count
    boundary_samples = []
    for stage in range(4, 0, -1):
        end = int(previous[stage, end])
        if end < 0:
            raise RuntimeError("Ordered decoding failed")
        boundary_samples.append(end)
    boundary_samples = np.asarray(boundary_samples[::-1], dtype=np.int32)
    decoded = np.searchsorted(boundary_samples, np.arange(count), side="right")
    return decoded.astype(np.int64), boundary_samples


def evaluate_model(
    model: nn.Module,
    episodes: dict[int, EpisodeData],
    ids: list[int],
    scaler: StandardScaler,
    device: torch.device,
    batch_size: int,
    sample_fps: float,
    output_dir: Path,
    name: str,
) -> dict:
    all_labels = []
    all_raw = []
    all_ordered = []
    boundary_errors = []
    episode_results = []
    prediction_dir = output_dir / "predictions" / name
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for episode_id in ids:
        episode = episodes[episode_id]
        features = np.clip(scaler.transform(episode.features), -8.0, 8.0).astype(np.float32)
        probabilities = predict_probabilities(model, features, device, batch_size)
        raw = probabilities.argmax(axis=1)
        ordered, boundary_samples = ordered_decode(probabilities, sample_fps)
        predicted_frames = episode.sample_frames[
            np.minimum(boundary_samples, len(episode.sample_frames) - 1)
        ]
        errors_s = np.abs(predicted_frames - episode.reference_boundaries) / 30.0
        ordered_f1 = float(f1_score(episode.labels, ordered, average="macro"))
        episode_results.append(
            {
                "episode": episode_id,
                "raw_macro_f1": float(f1_score(episode.labels, raw, average="macro")),
                "ordered_macro_f1": ordered_f1,
                "boundary_errors_s": errors_s.tolist(),
                "boundary_mean_error_s": float(errors_s.mean()),
                "predicted_boundaries_s": (predicted_frames / 30.0).tolist(),
                "reference_boundaries_s": (episode.reference_boundaries / 30.0).tolist(),
            }
        )
        np.savez_compressed(
            prediction_dir / f"episode_{episode_id:03d}.npz",
            probabilities=probabilities.astype(np.float16),
            raw=raw.astype(np.int8),
            ordered=ordered.astype(np.int8),
            labels=episode.labels.astype(np.int8),
            sample_frames=episode.sample_frames,
            predicted_boundaries=predicted_frames,
            reference_boundaries=episode.reference_boundaries,
        )
        all_labels.append(episode.labels)
        all_raw.append(raw)
        all_ordered.append(ordered)
        boundary_errors.extend(errors_s.tolist())
    labels = np.concatenate(all_labels)
    raw = np.concatenate(all_raw)
    ordered = np.concatenate(all_ordered)
    return {
        "frame_argmax_macro_f1": float(f1_score(labels, raw, average="macro")),
        "frame_argmax_balanced_accuracy": float(balanced_accuracy_score(labels, raw)),
        "ordered_macro_f1": float(f1_score(labels, ordered, average="macro")),
        "ordered_balanced_accuracy": float(balanced_accuracy_score(labels, ordered)),
        "ordered_confusion_matrix": confusion_matrix(labels, ordered).tolist(),
        "ordered_per_class": classification_report(
            labels, ordered, output_dict=True, zero_division=0
        ),
        "boundary_mean_absolute_error_s": float(np.mean(boundary_errors)),
        "boundary_median_absolute_error_s": float(np.median(boundary_errors)),
        "boundary_within_1s_rate": float(np.mean(np.asarray(boundary_errors) <= 1.0)),
        "boundary_within_2s_rate": float(np.mean(np.asarray(boundary_errors) <= 2.0)),
        "episodes": episode_results,
    }


def main() -> None:
    args = parse_args()
    if (args.include_action or args.include_state) and args.robot_data is None:
        raise ValueError("--robot-data is required when action or state features are enabled")
    if args.exclude_action_grippers and not args.include_action:
        raise ValueError("--exclude-action-grippers requires --include-action")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads(args.split.read_text())
    boundaries = load_boundaries(args.boundaries)
    robot_features = (
        load_robot_features(
            args.robot_data,
            args.include_action,
            args.include_state,
            args.exclude_action_grippers,
        )
        if args.robot_data is not None
        else {}
    )
    episodes = load_episodes(
        args.tracks_dir,
        split,
        boundaries,
        args.sample_fps,
        args.tracks_filename,
        robot_features,
        args.hidden_dir,
        args.hidden_key,
        args.hidden_filename,
        args.exclude_tracker,
    )
    outer_train_ids = [int(value) for value in split["train"]]
    test_ids = [int(value) for value in split["validation"]]
    shuffled = np.random.default_rng(args.seed).permutation(outer_train_ids)
    selection_ids = sorted(int(value) for value in shuffled[:8])
    train_ids = sorted(int(value) for value in shuffled[8:])
    train_x, train_y = concatenate(episodes, train_ids)
    selection_x, selection_y = concatenate(episodes, selection_ids)
    scaler = StandardScaler().fit(train_x)
    train_x = np.clip(scaler.transform(train_x), -8.0, 8.0).astype(np.float32)
    selection_x = np.clip(scaler.transform(selection_x), -8.0, 8.0).astype(np.float32)
    device = choose_device(args.device)

    results = {}
    started = time.perf_counter()
    for name in ("linear", "mlp"):
        model, history, best_epoch = train_model(
            name,
            train_x,
            train_y,
            selection_x,
            selection_y,
            device,
            args.epochs,
            args.patience,
            args.batch_size,
            args.seed,
        )
        metrics = evaluate_model(
            model,
            episodes,
            test_ids,
            scaler,
            device,
            args.batch_size,
            args.sample_fps,
            args.output_dir,
            name,
        )
        metrics["best_epoch"] = best_epoch
        metrics["selection_macro_f1"] = max(item["selection_f1"] for item in history)
        metrics["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        results[name] = metrics
        torch.save(model.state_dict(), args.output_dir / f"{name}.pt")
        (args.output_dir / f"{name}_history.json").write_text(json.dumps(history, indent=2) + "\n")
    np.savez_compressed(
        args.output_dir / "scaler.npz",
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
    )
    robot_modalities = []
    if args.include_state:
        robot_modalities.append("16D observation.state")
    if args.include_action:
        action_dims = 14 if args.exclude_action_grippers else 16
        suffix = " without grippers" if args.exclude_action_grippers else ""
        robot_modalities.append(f"{action_dims}D action{suffix}")
    robot_contract = f" + {' + '.join(robot_modalities)}" if robot_modalities else ""
    hidden_contract = f" + 2048D GR00T {args.hidden_key}" if args.hidden_dir else ""
    tracker_contract = (
        ""
        if args.exclude_tracker
        else f"{args.tracker_name} features: 400x2 normalized displacement + "
        "400 visibility + 8 motion statistics"
    )
    parts = [part for part in (tracker_contract, robot_contract.removeprefix(" + "), hidden_contract.removeprefix(" + ")) if part]
    report = {
        "feature_contract": f"{train_x.shape[1]}D {' + '.join(parts)}; no time index",
        "sample_fps": args.sample_fps,
        "device": str(device),
        "seed": args.seed,
        "train_episodes": train_ids,
        "selection_episodes": selection_ids,
        "test_episodes": test_ids,
        "runtime_s": time.perf_counter() - started,
        "models": results,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
