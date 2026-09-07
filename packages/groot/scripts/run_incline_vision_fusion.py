from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import train_tapnextpp_subtask_probe as baseline

from groot_subtask_phase_probe.incline_vision_fusion import (
    append_causal_deltas,
    model_factories,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracks-dir",
        type=Path,
        default=Path("reports/incline_new/alltracker_tracks_50_cuda"),
    )
    parser.add_argument(
        "--hidden-dir",
        type=Path,
        default=Path("reports/incline_new/groot_hidden_50"),
    )
    parser.add_argument(
        "--robot-data",
        type=Path,
        default=Path("data/Incline_new_20260902_203009/data/chunk-000/file-000.parquet"),
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
        default=Path("reports/incline_new/vision_fusion/seed_042"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--auxiliary-lags",
        type=int,
        nargs="*",
        default=[],
        help="causal lag lengths in 5 FPS samples",
    )
    parser.add_argument(
        "--include-full-hidden",
        action="store_true",
        help="add GR00T R3 attention-mask mean as a separately gated branch",
    )
    parser.add_argument(
        "--include-state",
        action="store_true",
        help="add the previously tested 16D robot observation state",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser.parse_args()


def train_candidate(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    selection_x: np.ndarray,
    selection_y: np.ndarray,
    *,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
) -> tuple[nn.Module, list[dict[str, float | int]], int]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = model.to(device)
    counts = np.bincount(train_y, minlength=5).astype(np.float32)
    weights = counts.sum() / (len(counts) * counts)
    loss_function = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_score = -1.0
    best_epoch = -1
    best_state = None
    stale = 0
    history: list[dict[str, float | int]] = []
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
        probabilities = baseline.predict_probabilities(
            model,
            selection_x,
            device,
            batch_size,
        )
        score = float(f1_score(selection_y, probabilities.argmax(axis=1), average="macro"))
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "selection_f1": score,
            }
        )
        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
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


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    split = json.loads(args.split.read_text())
    boundaries = baseline.load_boundaries(args.boundaries)
    robot_features = baseline.load_robot_features(
        args.robot_data,
        include_action=True,
        include_state=args.include_state,
        exclude_action_grippers=True,
    )
    vision = baseline.load_episodes(
        args.tracks_dir,
        split,
        boundaries,
        5.0,
        "alltracker_tracks.npz",
        robot_features={},
        hidden_dir=args.hidden_dir,
        hidden_key="vision_mean",
        exclude_tracker=True,
    )
    context = None
    if args.include_full_hidden:
        context = baseline.load_episodes(
            args.tracks_dir,
            split,
            boundaries,
            5.0,
            "alltracker_tracks.npz",
            robot_features={},
            hidden_dir=args.hidden_dir,
            hidden_key="hidden_mean",
            exclude_tracker=True,
        )
    auxiliary = baseline.load_episodes(
        args.tracks_dir,
        split,
        boundaries,
        5.0,
        "alltracker_tracks.npz",
        robot_features=robot_features,
    )
    if args.auxiliary_lags:
        lags = tuple(args.auxiliary_lags)
        for episode in auxiliary.values():
            episode.features = append_causal_deltas(episode.features, lags)
    episodes = {}
    for episode_id in vision:
        if not np.array_equal(
            vision[episode_id].sample_frames,
            auxiliary[episode_id].sample_frames,
        ):
            raise ValueError(f"Feature alignment failed for episode {episode_id}")
        parts = [vision[episode_id].features]
        if context is not None:
            if not np.array_equal(
                vision[episode_id].sample_frames,
                context[episode_id].sample_frames,
            ):
                raise ValueError(f"Context alignment failed for episode {episode_id}")
            parts.append(context[episode_id].features)
        parts.append(auxiliary[episode_id].features)
        episodes[episode_id] = baseline.EpisodeData(
            episode=episode_id,
            features=np.concatenate(parts, axis=1),
            labels=vision[episode_id].labels,
            sample_frames=vision[episode_id].sample_frames,
            reference_boundaries=vision[episode_id].reference_boundaries,
        )

    outer_train = [int(value) for value in split["train"]]
    test_ids = [int(value) for value in split["validation"]]
    shuffled = np.random.default_rng(args.seed).permutation(outer_train)
    selection_ids = sorted(int(value) for value in shuffled[:8])
    train_ids = sorted(int(value) for value in shuffled[8:])
    train_x, train_y = baseline.concatenate(episodes, train_ids)
    selection_x, selection_y = baseline.concatenate(episodes, selection_ids)
    scaler = StandardScaler().fit(train_x)
    train_x = np.clip(scaler.transform(train_x), -8.0, 8.0).astype(np.float32)
    selection_x = np.clip(scaler.transform(selection_x), -8.0, 8.0).astype(np.float32)
    device = baseline.choose_device(args.device)
    vision_dim = next(iter(vision.values())).features.shape[1]
    context_dim = (
        next(iter(context.values())).features.shape[1] if context is not None else 0
    )
    auxiliary_dim = next(iter(auxiliary.values())).features.shape[1]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, factory in model_factories(
        vision_dim, auxiliary_dim, context_dim
    ).items():
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        model, history, best_epoch = train_candidate(
            factory(),
            train_x,
            train_y,
            selection_x,
            selection_y,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        metrics = baseline.evaluate_model(
            model,
            episodes,
            test_ids,
            scaler,
            device,
            args.batch_size,
            5.0,
            args.output_dir,
            name,
        )
        metrics["best_epoch"] = best_epoch
        metrics["selection_macro_f1"] = max(
            float(item["selection_f1"]) for item in history
        )
        metrics["parameter_count"] = sum(
            parameter.numel() for parameter in model.parameters()
        )
        if hasattr(model, "auxiliary_gate_logit"):
            metrics["auxiliary_gate"] = (
                torch.sigmoid(model.auxiliary_gate_logit).detach().cpu().tolist()
            )
        if hasattr(model, "context_gate_logit"):
            metrics["context_gate"] = (
                torch.sigmoid(model.context_gate_logit).detach().cpu().tolist()
            )
        results[name] = metrics
        torch.save(model.state_dict(), args.output_dir / f"{name}.pt")
        (args.output_dir / f"{name}_history.json").write_text(
            json.dumps(history, indent=2) + "\n"
        )

    report = {
        "feature_contract": (
            f"{vision_dim}D GR00T R3b vision_mean"
            + (f" + {context_dim}D GR00T R3 hidden_mean" if context_dim else "")
            + f" + {auxiliary_dim}D AllTracker, 14D action without grippers"
            + (", and 16D robot state" if args.include_state else "")
            + (
                f" with causal deltas at {args.auxiliary_lags} samples"
                if args.auxiliary_lags
                else ""
            )
        ),
        "seed": args.seed,
        "device": str(device),
        "train_episodes": train_ids,
        "selection_episodes": selection_ids,
        "validation_episodes": test_ids,
        "runtime_s": time.perf_counter() - started,
        "models": results,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
