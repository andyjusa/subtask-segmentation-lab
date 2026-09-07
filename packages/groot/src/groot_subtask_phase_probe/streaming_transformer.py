from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .evaluate import (
    BOUNDARY_THRESHOLDS,
    SEEDS,
    Episode,
    Transform,
    _episode_map,
    _stage_xy,
    load_episodes,
    split_episode_ids,
)
from .streaming_metrics import boundary_event_metrics

STAGE_NAMES = ("before_predicate_1", "after_predicate_1")


@dataclass(frozen=True)
class TransformerConfig:
    window_size: int = 16
    model_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    feedforward_dim: int = 512
    dropout: float = 0.10

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size must be at least two")
        if self.model_dim < 1 or self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be positive and divisible by num_heads")
        if self.num_layers < 1 or self.feedforward_dim < 1:
            raise ValueError("num_layers and feedforward_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class PreparedEpisode:
    episode_id: str
    values: np.ndarray
    stage: np.ndarray
    boundary: np.ndarray


def pack_causal_window(
    values: np.ndarray,
    *,
    end: int,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack only observations at or before ``end`` into a fixed causal window."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("values must have shape [time, feature]")
    if not 0 <= end < len(array):
        raise IndexError(f"end={end} is outside a sequence of length {len(array)}")
    if window_size < 1:
        raise ValueError("window_size must be positive")
    start = max(0, end - window_size + 1)
    source = array[start : end + 1]
    window = np.zeros((window_size, array.shape[1]), dtype=np.float32)
    valid = np.zeros(window_size, dtype=bool)
    window[: len(source)] = source
    valid[: len(source)] = True
    return window, valid


class EpisodeWindowDataset(Dataset):
    def __init__(self, episodes: Sequence[PreparedEpisode], window_size: int):
        self.episodes = tuple(episodes)
        self.window_size = window_size
        self.indices = [
            (episode_index, step)
            for episode_index, episode in enumerate(self.episodes)
            for step in range(len(episode.stage))
        ]
        if not self.indices:
            raise ValueError("The dataset has no non-terminal samples")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_index, step = self.indices[index]
        episode = self.episodes[episode_index]
        values, valid = pack_causal_window(
            episode.values,
            end=step,
            window_size=self.window_size,
        )
        return {
            "values": torch.from_numpy(values),
            "valid": torch.from_numpy(valid),
            "stage": torch.tensor(int(episode.stage[step]), dtype=torch.long),
            "boundary": torch.tensor(float(episode.boundary[step])),
        }


class CausalSlidingWindowTransformer(nn.Module):
    """Two-head temporal probe whose current output can only see a past window."""

    def __init__(
        self,
        input_dim: int,
        window_size: int = 16,
        model_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 512,
        dropout: float = 0.10,
    ):
        super().__init__()
        config = TransformerConfig(
            window_size=window_size,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        self.input_dim = input_dim
        self.config = config
        self.input_projection = nn.Linear(input_dim, config.model_dim)
        self.position = nn.Parameter(torch.zeros(1, config.window_size, config.model_dim))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.model_dim),
            enable_nested_tensor=False,
        )
        self.stage_head = nn.Linear(config.model_dim, len(STAGE_NAMES))
        self.boundary_head = nn.Sequential(
            nn.LayerNorm(config.model_dim * 2),
            nn.Linear(config.model_dim * 2, 1),
        )
        causal_mask = torch.triu(
            torch.ones(config.window_size, config.window_size, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(
        self,
        values: torch.Tensor,
        valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if (
            values.ndim != 3
            or values.shape[1] != self.config.window_size
            or values.shape[2] != self.input_dim
        ):
            raise ValueError(
                f"values must have shape [batch, {self.config.window_size}, {self.input_dim}]"
            )
        if valid.shape != values.shape[:2]:
            raise ValueError("valid must match the batch/time dimensions of values")
        lengths = valid.sum(dim=1)
        if bool((lengths < 1).any()):
            raise ValueError("every window must contain at least one valid timestep")
        hidden = self.input_projection(values) + self.position
        encoded = self.encoder(
            hidden,
            mask=self.causal_mask,
            src_key_padding_mask=~valid,
        )
        rows = torch.arange(len(encoded), device=encoded.device)
        current_index = lengths - 1
        previous_index = (lengths - 2).clamp_min(0)
        current = encoded[rows, current_index]
        previous = encoded[rows, previous_index]
        delta = torch.where(
            (lengths > 1).unsqueeze(-1),
            current - previous,
            torch.zeros_like(current),
        )
        return {
            "stage_logits": self.stage_head(current),
            "boundary_logits": self.boundary_head(torch.cat([current, delta], dim=-1)).squeeze(-1),
        }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def predict_streaming_sequence(
    model: CausalSlidingWindowTransformer,
    values: np.ndarray,
    *,
    device: torch.device,
    measure_latency: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run one step at a time; no call receives a future feature."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != model.input_dim:
        raise ValueError(f"Expected values [time, {model.input_dim}], got {array.shape}")
    stage_probabilities: list[np.ndarray] = []
    boundary_probabilities: list[float] = []
    elapsed_s = 0.0
    model.eval()
    with torch.inference_mode():
        for end in range(len(array)):
            if measure_latency:
                _sync(device)
                started = time.perf_counter()
            window, valid = pack_causal_window(
                array,
                end=end,
                window_size=model.config.window_size,
            )
            output = model(
                torch.from_numpy(window).unsqueeze(0).to(device),
                torch.from_numpy(valid).unsqueeze(0).to(device),
            )
            if measure_latency:
                _sync(device)
                elapsed_s += time.perf_counter() - started
            stage_probabilities.append(
                torch.softmax(output["stage_logits"], dim=-1)[0].float().cpu().numpy()
            )
            boundary_probabilities.append(float(torch.sigmoid(output["boundary_logits"])[0].cpu()))
    return (
        np.asarray(stage_probabilities, dtype=np.float32),
        np.asarray(boundary_probabilities, dtype=np.float32),
        elapsed_s * 1000 / max(1, len(array)),
    )


def predict_causal_batched_sequence(
    model: CausalSlidingWindowTransformer,
    values: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch independent causal windows for faster validation only."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != model.input_dim:
        raise ValueError(f"Expected values [time, {model.input_dim}], got {array.shape}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    windows, masks = zip(
        *(
            pack_causal_window(array, end=end, window_size=model.config.window_size)
            for end in range(len(array))
        ),
        strict=True,
    )
    windows_tensor = torch.from_numpy(np.stack(windows))
    masks_tensor = torch.from_numpy(np.stack(masks))
    stage: list[np.ndarray] = []
    boundary: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(array), batch_size):
            stop = min(len(array), start + batch_size)
            output = model(
                windows_tensor[start:stop].to(device),
                masks_tensor[start:stop].to(device),
            )
            stage.append(torch.softmax(output["stage_logits"], dim=-1).float().cpu().numpy())
            boundary.append(torch.sigmoid(output["boundary_logits"]).float().cpu().numpy())
    return np.concatenate(stage), np.concatenate(boundary)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _prepare_episodes(
    episodes: Mapping[str, Episode],
    episode_ids: set[str],
    representation: str,
    transform: Transform,
    *,
    boundary_radius: int = 2,
    include_input_delta: bool = False,
) -> list[PreparedEpisode]:
    if boundary_radius < 0:
        raise ValueError("boundary_radius must be non-negative")
    prepared = []
    for episode_id in sorted(episode_ids):
        episode = episodes[episode_id]
        valid = (episode.terminal == 0) & (episode.stage >= 0)
        values = transform.apply(episode.features[representation][valid])
        if include_input_delta:
            delta = np.vstack(
                [np.zeros((1, values.shape[1]), dtype=values.dtype), np.diff(values, axis=0)]
            )
            values = np.concatenate([values, delta], axis=1)
        boundary = np.zeros(len(episode.env_step), dtype=np.float32)
        valid_indices = np.flatnonzero(valid)
        if episode.boundary_step is not None and len(valid_indices):
            nearest_offset = int(
                np.argmin(np.abs(episode.env_step[valid_indices] - episode.boundary_step))
            )
            lo = max(0, nearest_offset - boundary_radius)
            hi = min(len(valid_indices) - 1, nearest_offset + boundary_radius)
            boundary[valid_indices[lo : hi + 1]] = 1.0
        prepared.append(
            PreparedEpisode(
                episode_id=episode_id,
                values=values,
                stage=episode.stage[valid].astype(np.int64, copy=False),
                boundary=boundary[valid],
            )
        )
    return prepared


def _loss_weights(
    episodes: Sequence[PreparedEpisode],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    stages = np.concatenate([episode.stage for episode in episodes])
    counts = np.bincount(stages, minlength=len(STAGE_NAMES)).astype(np.float64)
    stage_weights = counts.sum() / (len(STAGE_NAMES) * np.maximum(counts, 1))
    boundaries = np.concatenate([episode.boundary for episode in episodes])
    positives = max(1.0, float(boundaries.sum()))
    negatives = max(1.0, float(len(boundaries) - boundaries.sum()))
    return (
        torch.tensor(stage_weights, dtype=torch.float32, device=device),
        torch.tensor(negatives / positives, dtype=torch.float32, device=device),
    )


def _batch_loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    stage_criterion: nn.Module,
    boundary_criterion: nn.Module,
    stage_loss_weight: float,
    boundary_loss_weight: float,
) -> torch.Tensor:
    stage_loss = stage_criterion(output["stage_logits"], batch["stage"])
    boundary_loss = boundary_criterion(output["boundary_logits"], batch["boundary"])
    return stage_loss_weight * stage_loss + boundary_loss_weight * boundary_loss


def _validation_loss(
    model: CausalSlidingWindowTransformer,
    loader: DataLoader,
    device: torch.device,
    stage_criterion: nn.Module,
    boundary_criterion: nn.Module,
    stage_loss_weight: float,
    boundary_loss_weight: float,
) -> float:
    model.eval()
    total = 0.0
    samples = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = {key: value.to(device) for key, value in raw_batch.items()}
            output = model(batch["values"], batch["valid"])
            loss = _batch_loss(
                output,
                batch,
                stage_criterion,
                boundary_criterion,
                stage_loss_weight,
                boundary_loss_weight,
            )
            total += float(loss) * len(batch["stage"])
            samples += len(batch["stage"])
    return total / max(1, samples)


def _predict_episodes(
    model: CausalSlidingWindowTransformer,
    prepared: Sequence[PreparedEpisode],
    *,
    device: torch.device,
    measure_latency: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    stage: dict[str, np.ndarray] = {}
    boundary: dict[str, np.ndarray] = {}
    weighted_latency = 0.0
    samples = 0
    for episode in prepared:
        if measure_latency:
            stage_values, boundary_values, latency = predict_streaming_sequence(
                model,
                episode.values,
                device=device,
                measure_latency=True,
            )
        else:
            stage_values, boundary_values = predict_causal_batched_sequence(
                model,
                episode.values,
                device=device,
            )
            latency = 0.0
        stage[episode.episode_id] = stage_values
        boundary[episode.episode_id] = boundary_values
        weighted_latency += latency * len(episode.values)
        samples += len(episode.values)
    return stage, boundary, weighted_latency / max(1, samples)


def _stage_metrics(
    prepared: Sequence[PreparedEpisode],
    probabilities: Mapping[str, np.ndarray],
) -> dict[str, float]:
    reference = np.concatenate([episode.stage for episode in prepared])
    prediction = np.concatenate(
        [probabilities[episode.episode_id].argmax(axis=-1) for episode in prepared]
    )
    precision, recall, _, _ = precision_recall_fscore_support(
        reference,
        prediction,
        labels=[0, 1],
        zero_division=0,
    )
    return {
        "stage_macro_f1": float(f1_score(reference, prediction, average="macro")),
        "stage_balanced_accuracy": float(balanced_accuracy_score(reference, prediction)),
        "stage_0_precision": float(precision[0]),
        "stage_0_recall": float(recall[0]),
        "stage_1_precision": float(precision[1]),
        "stage_1_recall": float(recall[1]),
    }


def _select_boundary_threshold(
    episodes: Mapping[str, Episode],
    validation_ids: set[str],
    probabilities: Mapping[str, np.ndarray],
) -> tuple[float, dict[str, float | int]]:
    best: tuple[tuple[float, float, float, float], float, dict[str, float | int]] | None = None
    for threshold in BOUNDARY_THRESHOLDS:
        metrics = boundary_event_metrics(
            episodes,
            validation_ids,
            probabilities,
            threshold=float(threshold),
        )
        score = (
            float(metrics["boundary_event_f1_at_5"]),
            float(metrics["boundary_event_f1_at_10"]),
            -float(metrics["false_boundaries_per_episode"]),
            -abs(float(threshold) - 0.5),
        )
        if best is None or score > best[0]:
            best = (score, float(threshold), metrics)
    if best is None:
        raise RuntimeError("No boundary threshold was selected")
    return best[1], best[2]


def checkpoint_selection_score(
    boundary_metrics: Mapping[str, float | int],
    stage_metrics: Mapping[str, float],
    validation_loss: float,
) -> tuple[float, float, float, float, float]:
    """Apply the deployment selection rule before considering surrogate loss."""

    return (
        float(boundary_metrics["boundary_event_f1_at_5"]),
        float(boundary_metrics["boundary_event_f1_at_10"]),
        float(stage_metrics["stage_macro_f1"]),
        -float(boundary_metrics["false_boundaries_per_episode"]),
        -validation_loss,
    )


def _transform_artifact(transform: Transform) -> dict[str, object]:
    result: dict[str, object] = {
        "scaler_mean": np.asarray(transform.scaler.mean_, dtype=np.float32),
        "scaler_scale": np.asarray(transform.scaler.scale_, dtype=np.float32),
    }
    if transform.pca is not None:
        result.update(
            {
                "pca_components": np.asarray(transform.pca.components_, dtype=np.float32),
                "pca_mean": np.asarray(transform.pca.mean_, dtype=np.float32),
                "pca_output_dim": 256,
            }
        )
    return result


def train_one(
    episodes_list: list[Episode],
    *,
    representation: str,
    common_256: bool,
    seed: int,
    config: TransformerConfig,
    device: torch.device,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    boundary_loss_weight: float,
    patience: int,
    backbone_select_layer: int,
    split_ids: tuple[set[str], set[str], set[str]] | None = None,
    stage_loss_weight: float = 1.0,
    boundary_radius: int = 2,
    include_input_delta: bool = False,
    label_source: str = "LIBERO environment predicate",
) -> dict[str, object]:
    if stage_loss_weight < 0:
        raise ValueError("stage_loss_weight must be non-negative")
    _seed_everything(seed)
    episodes = _episode_map(episodes_list)
    train_ids, validation_ids, test_ids = (
        split_episode_ids(episodes_list, seed) if split_ids is None else split_ids
    )
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise RuntimeError("Episode split leakage detected")
    stage_train_x, _ = _stage_xy(episodes, train_ids, representation)
    transform = Transform(common_256, seed).fit(stage_train_x)
    prepare_options = {
        "boundary_radius": boundary_radius,
        "include_input_delta": include_input_delta,
    }
    train = _prepare_episodes(episodes, train_ids, representation, transform, **prepare_options)
    validation = _prepare_episodes(
        episodes, validation_ids, representation, transform, **prepare_options
    )
    test = _prepare_episodes(episodes, test_ids, representation, transform, **prepare_options)
    train_loader = DataLoader(
        EpisodeWindowDataset(train, config.window_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        EpisodeWindowDataset(validation, config.window_size),
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
    )
    input_dim = int(train[0].values.shape[1])
    model = CausalSlidingWindowTransformer(
        input_dim=input_dim,
        **asdict(config),
    ).to(device)
    stage_weight, boundary_pos_weight = _loss_weights(train, device)
    stage_criterion = nn.CrossEntropyLoss(weight=stage_weight)
    boundary_criterion = nn.BCEWithLogitsLoss(pos_weight=boundary_pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_selection_score: tuple[float, float, float, float, float] | None = None
    history: list[dict[str, float | int]] = []
    stale = 0
    for epoch in range(epochs):
        model.train()
        for raw_batch in train_loader:
            batch = {key: value.to(device) for key, value in raw_batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["values"], batch["valid"])
            loss = _batch_loss(
                output,
                batch,
                stage_criterion,
                boundary_criterion,
                stage_loss_weight,
                boundary_loss_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        validation_loss = _validation_loss(
            model,
            validation_loader,
            device,
            stage_criterion,
            boundary_criterion,
            stage_loss_weight,
            boundary_loss_weight,
        )
        validation_stage, validation_boundary, _ = _predict_episodes(
            model,
            validation,
            device=device,
            measure_latency=False,
        )
        validation_threshold, validation_event = _select_boundary_threshold(
            episodes,
            validation_ids,
            validation_boundary,
        )
        validation_stage_metrics = _stage_metrics(validation, validation_stage)
        selection_score = checkpoint_selection_score(
            validation_event,
            validation_stage_metrics,
            validation_loss,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "validation_loss": validation_loss,
                "boundary_threshold": validation_threshold,
                "boundary_event_f1_at_5": float(validation_event["boundary_event_f1_at_5"]),
                "boundary_event_f1_at_10": float(validation_event["boundary_event_f1_at_10"]),
                "false_boundaries_per_episode": float(
                    validation_event["false_boundaries_per_episode"]
                ),
                "stage_macro_f1": float(validation_stage_metrics["stage_macro_f1"]),
            }
        )
        if best_selection_score is None or selection_score > best_selection_score:
            best_selection_score = selection_score
            best_loss = validation_loss
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
        raise RuntimeError("No Transformer checkpoint was selected")
    model.load_state_dict(best_state)

    _, validation_boundary, _ = _predict_episodes(
        model,
        validation,
        device=device,
        measure_latency=False,
    )
    threshold, validation_boundary_metrics = _select_boundary_threshold(
        episodes,
        validation_ids,
        validation_boundary,
    )
    test_stage, test_boundary, probe_latency = _predict_episodes(
        model,
        test,
        device=device,
        measure_latency=True,
    )
    result: dict[str, object] = {
        "representation": representation,
        "projection": "common_256" if common_256 else "native",
        "seed": seed,
        "native_dim": int(stage_train_x.shape[1]),
        "probe_dim": input_dim,
        "backbone_select_layer": backbone_select_layer,
        "window_size": config.window_size,
        "stage_loss_weight": stage_loss_weight,
        "boundary_radius": boundary_radius,
        "include_input_delta": include_input_delta,
        "best_epoch": best_epoch,
        "validation_loss": best_loss,
        "boundary_threshold": threshold,
        "validation_boundary_event_f1_at_5": validation_boundary_metrics["boundary_event_f1_at_5"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "streaming_probe_latency_ms_per_sample": probe_latency,
        "train_episode_ids": sorted(train_ids),
        "validation_episode_ids": sorted(validation_ids),
        "test_episode_ids": sorted(test_ids),
        "decoder": "strict_streaming_first_rising_event",
        "uses_offline_viterbi": False,
        "checkpoint_selection": "boundary_f1_at_5_then_10_then_stage_then_fp_then_loss",
        **_stage_metrics(test, test_stage),
        **boundary_event_metrics(
            episodes,
            test_ids,
            test_boundary,
            threshold=threshold,
        ),
    }

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / (f"{representation}__{result['projection']}__seed_{seed}.pt")
    torch.save(
        {
            "model_name": "causal_sliding_window_transformer",
            "state_dict": best_state,
            "model_config": asdict(config),
            "input_dim": input_dim,
            "stage_names": STAGE_NAMES,
            "representation": representation,
            "projection": result["projection"],
            "seed": seed,
            "backbone_select_layer": backbone_select_layer,
            "label_source": label_source,
            "boundary_training_window": (
                f"nearest GT policy sample +/- {boundary_radius} samples"
            ),
            "stage_loss_weight": stage_loss_weight,
            "include_input_delta": include_input_delta,
            "boundary_threshold": threshold,
            "decoder": result["decoder"],
            "uses_offline_viterbi": False,
            "checkpoint_selection": result["checkpoint_selection"],
            "train_episode_ids": result["train_episode_ids"],
            "validation_episode_ids": result["validation_episode_ids"],
            "test_episode_ids": result["test_episode_ids"],
            **_transform_artifact(transform),
        },
        checkpoint_path,
    )
    history_dir = output_dir / "histories"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / (f"{representation}__{result['projection']}__seed_{seed}.json")
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n")
    result["checkpoint"] = str(checkpoint_path)
    result["history"] = str(history_path)
    return result


def _mean_ci95(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    valid = array[np.isfinite(array)]
    if not len(valid):
        return float("nan"), float("nan")
    mean = float(valid.mean())
    ci = float(1.96 * valid.std(ddof=1) / math.sqrt(len(valid))) if len(valid) > 1 else float("nan")
    return mean, ci


SUMMARY_METRICS = (
    "stage_macro_f1",
    "stage_balanced_accuracy",
    "boundary_event_f1_at_5",
    "boundary_event_f1_at_10",
    "boundary_median_abs_error",
    "false_boundaries_per_episode",
    "predicted_boundaries_per_episode",
    "no_boundary_failure_fpr",
    "streaming_probe_latency_ms_per_sample",
)


def summarize(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    summary = []
    keys = sorted({(str(row["representation"]), str(row["projection"])) for row in rows})
    for representation, projection in keys:
        selected = [
            row
            for row in rows
            if row["representation"] == representation and row["projection"] == projection
        ]
        item: dict[str, object] = {
            "representation": representation,
            "projection": projection,
            "native_dim": selected[0]["native_dim"],
            "probe_dim": selected[0]["probe_dim"],
            "parameter_count": selected[0]["parameter_count"],
            "window_size": selected[0]["window_size"],
            "seeds": len(selected),
        }
        for metric in SUMMARY_METRICS:
            mean, ci = _mean_ci95(float(row[metric]) for row in selected)
            item[f"{metric}_mean"] = mean
            item[f"{metric}_ci95"] = ci
        summary.append(item)
    return sorted(
        summary,
        key=lambda row: (
            -float(row["boundary_event_f1_at_5_mean"]),
            -float(row["stage_macro_f1_mean"]),
            float(row["false_boundaries_per_episode_mean"]),
            float(row["streaming_probe_latency_ms_per_sample_mean"]),
        ),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _linear_baseline(path: Path | None) -> dict[tuple[str, str], dict[str, float]]:
    if path is None or not path.is_file():
        return {}
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    output: dict[tuple[str, str], dict[str, float]] = {}
    for key in sorted({(row["representation"], row["projection"]) for row in rows}):
        selected = [row for row in rows if (row["representation"], row["projection"]) == key]
        output[key] = {
            "stage": float(np.mean([float(row["stage_macro_f1"]) for row in selected])),
            "boundary": float(np.mean([float(row["boundary_event_f1_at_5"]) for row in selected])),
        }
    return output


def _format_value(value: object, digits: int = 3) -> str:
    number = float(value)
    return "N/A" if not math.isfinite(number) else f"{number:.{digits}f}"


def _paired_difference(
    rows: Sequence[Mapping[str, object]],
    first: tuple[str, str],
    second: tuple[str, str],
    metric: str,
) -> tuple[float, float]:
    def by_seed(key: tuple[str, str]) -> dict[int, float]:
        return {
            int(row["seed"]): float(row[metric])
            for row in rows
            if (str(row["representation"]), str(row["projection"])) == key
        }

    first_values = by_seed(first)
    second_values = by_seed(second)
    shared = sorted(first_values.keys() & second_values.keys())
    differences = np.asarray(
        [first_values[seed] - second_values[seed] for seed in shared],
        dtype=np.float64,
    )
    return _mean_ci95(differences)


def _feature_costs(linear_results: Path | None) -> dict[str, float]:
    if linear_results is None:
        return {}
    path = linear_results.parent / "cost_summary.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    direct = {
        name: float(values["mean_ms"])
        for name, values in payload.get("representations", {}).items()
    }
    if {"R3_vlm_hidden", "R5_final_dit_hidden", "R6_vlm_dit_concat"} <= direct.keys():
        direct["R6_vlm_dit_concat"] = sum(
            direct[name] for name in ("R3_vlm_hidden", "R5_final_dit_hidden", "R6_vlm_dit_concat")
        )
    return direct


def write_report(
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
    summary: Sequence[Mapping[str, object]],
    episodes: Sequence[Episode],
    config: TransformerConfig,
    linear_results: Path | None,
) -> None:
    baseline = _linear_baseline(linear_results)
    feature_costs = _feature_costs(linear_results)
    success_count = sum(int(episode.success) for episode in episodes)
    no_boundary_count = sum(
        int(episode.boundary_step is None and not episode.success) for episode in episodes
    )
    seeds = sorted({int(row["seed"]) for row in rows})
    representations = sorted({str(row["representation"]) for row in rows})
    projections = sorted({str(row["projection"]) for row in rows})
    lines = [
        "# GR00T N1.7 causal sliding-window Transformer 실험",
        "",
        "## 실험 계약",
        "",
        f"- 데이터: LIBERO 3 tasks, {len(episodes)} episodes, 성공 {success_count}/{len(episodes)}",
        "- 정답: 환경 내부 predicate 최초 만족 시점; terminal 표본은 stage 학습에서 제외",
        f"- split: task별 episode 60/20/20, 기존 linear와 동일한 seed {seeds}",
        f"- 실행 표현: {', '.join(representations)}",
        f"- 실행 투영: {', '.join(projections)}; train-only StandardScaler/PCA",
        f"- 모델: 과거 {config.window_size}개 policy sample만 보는 2-layer causal Transformer",
        "- 출력: stage 0/1 head + first-rising boundary head",
        "- 평가: 순차 streaming forward만 사용하며 offline Viterbi/backtracking은 사용하지 않음",
        "- 경계: validation threshold 선택 후 test event TP/FP/FN으로 F1@±5/±10 계산",
        "- R3/R3b는 GR00T N1.7의 실제 select_layer=16 backbone 출력",
        "",
        "## 결과",
        "",
        "| 표현 | 투영 | Transformer Boundary F1@±5 | 기존 Linear† F1@±5 | Transformer Stage F1 | 기존 Linear Stage F1 | FP/episode | 실패 FPR | feature ms | probe ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        key = (str(item["representation"]), str(item["projection"]))
        linear = baseline.get(key, {})
        lines.append(
            f"| {key[0]} | {key[1]} | "
            f"{_format_value(item['boundary_event_f1_at_5_mean'])} ± "
            f"{_format_value(item['boundary_event_f1_at_5_ci95'])} | "
            f"{_format_value(linear.get('boundary', float('nan')))} | "
            f"{_format_value(item['stage_macro_f1_mean'])} ± "
            f"{_format_value(item['stage_macro_f1_ci95'])} | "
            f"{_format_value(linear.get('stage', float('nan')))} | "
            f"{_format_value(item['false_boundaries_per_episode_mean'])} | "
            f"{_format_value(item['no_boundary_failure_fpr_mean'])} | "
            f"{_format_value(feature_costs.get(key[0], float('nan')))} | "
            f"{_format_value(item['streaming_probe_latency_ms_per_sample_mean'])} |"
        )
    winner = summary[0]
    runner_up = summary[1] if len(summary) > 1 else summary[0]
    winner_key = (str(winner["representation"]), str(winner["projection"]))
    runner_key = (str(runner_up["representation"]), str(runner_up["projection"]))
    boundary_difference, boundary_difference_ci = _paired_difference(
        rows,
        winner_key,
        runner_key,
        "boundary_event_f1_at_5",
    )
    stage_difference, stage_difference_ci = _paired_difference(
        rows,
        winner_key,
        runner_key,
        "stage_macro_f1",
    )
    lines.extend(
        [
            "",
            "## 해석",
            "",
            (
                f"현재 최고 Boundary F1@±5는 **{winner['representation']} "
                f"({winner['projection']})**의 "
                f"{_format_value(winner['boundary_event_f1_at_5_mean'])}다. "
                "이 순위는 강제 monotonic 경로가 아닌 실제 첫 streaming 이벤트를 기준으로 한다."
            ),
            (
                f"상위 두 조합의 paired 차이는 Boundary F1@±5가 "
                f"{_format_value(boundary_difference)} ± "
                f"{_format_value(boundary_difference_ci)}, Stage F1이 "
                f"{_format_value(stage_difference)} ± "
                f"{_format_value(stage_difference_ci)}다. 두 구간 모두 0을 포함하면 "
                "통계적으로 단일 승자를 확정하지 않는다."
            ),
            (
                "실용 기본안은 단일 source이면서 feature/probe 지연이 더 작은 "
                "R4 action encoder common-256이다. R6는 R3+R5 fusion 보조 결과로 유지한다."
            ),
            "",
            "## 제한",
            "",
            (
                f"- 무경계 실패 episode가 {no_boundary_count}개뿐이므로 실패 FPR의 분산이 크다. "
                "고의 실패 rollout을 task별로 추가해야 한다."
            ),
            (
                "- †기존 Linear 열은 과거 evaluator 결과로, 첫 예측 뒤의 추가 rising event를 "
                "FP로 벌점 주지 않았다. Transformer의 strict event F1과 직접 우열 비교하지 않는다."
            ),
            "- policy sample은 8 physics step 간격이어서 ±5 env-step 평가는 양자화 영향을 받는다.",
            "- 이번 결과는 동일한 3개 LIBERO task 내 일반화이며 leave-one-task-out 결과가 아니다.",
            "",
            "## 산출물",
            "",
            f"- per-seed runs: {len(rows)}",
            "- `results.csv`: seed별 원본 수치",
            f"- `summary.csv` / `summary.json`: {len(seeds)}-seed 평균과 95% CI",
            "- `checkpoints/`: scaler/PCA, split IDs, select_layer, decoder 계약을 포함한 checkpoint",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines))


def _parse_csv_values(value: str, available: Sequence[str]) -> tuple[str, ...]:
    if value == "all":
        return tuple(available)
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"Unknown values: {sorted(unknown)}")
    return selected


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def run(args: argparse.Namespace) -> None:
    episodes = load_episodes(args.artifacts)
    available = sorted(set.intersection(*(set(episode.features) for episode in episodes)))
    representations = _parse_csv_values(args.representations, available)
    projections = _parse_csv_values(
        args.projections,
        ("native", "common_256"),
    )
    seeds = SEEDS if args.seeds == "all" else tuple(int(value) for value in args.seeds.split(","))
    device = _resolve_device(args.device)
    config = TransformerConfig(
        window_size=args.window_size,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for representation in representations:
        for projection in projections:
            for seed in seeds:
                run_path = runs_dir / f"{representation}__{projection}__seed_{seed}.json"
                if run_path.is_file() and not args.force:
                    result = json.loads(run_path.read_text())
                else:
                    result = train_one(
                        episodes,
                        representation=representation,
                        common_256=projection == "common_256",
                        seed=seed,
                        config=config,
                        device=device,
                        output_dir=output_dir,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        weight_decay=args.weight_decay,
                        boundary_loss_weight=args.boundary_loss_weight,
                        patience=args.patience,
                        backbone_select_layer=args.backbone_select_layer,
                    )
                    run_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
                rows.append(result)
                print(
                    json.dumps(
                        {
                            key: result[key]
                            for key in (
                                "representation",
                                "projection",
                                "seed",
                                "stage_macro_f1",
                                "boundary_event_f1_at_5",
                                "false_boundaries_per_episode",
                            )
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    summary = summarize(rows)
    (output_dir / "per_seed_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    _write_csv(output_dir / "results.csv", rows)
    _write_csv(output_dir / "summary.csv", summary)
    experiment_config = {
        "artifacts": str(args.artifacts),
        "representations": representations,
        "projections": projections,
        "seeds": seeds,
        "device": str(device),
        "backbone_select_layer": args.backbone_select_layer,
        "transformer": asdict(config),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "boundary_loss_weight": args.boundary_loss_weight,
        "patience": args.patience,
        "decoder": "strict_streaming_first_rising_event",
        "uses_offline_viterbi": False,
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(experiment_config, ensure_ascii=False, indent=2) + "\n"
    )
    write_report(
        output_dir,
        rows,
        summary,
        episodes,
        config,
        args.linear_results,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/transformer_streaming"))
    parser.add_argument("--linear-results", type=Path, default=Path("reports_full/results.csv"))
    parser.add_argument("--representations", default="all")
    parser.add_argument("--projections", default="native,common_256")
    parser.add_argument("--seeds", default="all")
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--feedforward-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--boundary-loss-weight", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backbone-select-layer", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
