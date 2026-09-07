"""Run the seed-17 GR00T hidden-state Transformer as a template QA method.

The checkpoint was trained from 1 FPS, 2048-dimensional ``hidden_mean`` features
with an eight-frame causal history.  Its stage targets are algorithmic right-hand
kinematic weak labels, not human annotations or environment-predicate truth.

The source ``features_1fps`` arrays are valid inference inputs because their
``hidden_mean`` and ``timestamp`` arrays are the same arrays copied into the
right-hand relabelled training directory.  This runner deliberately ignores the
``stage`` array stored in either feature directory.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

METHOD_ID = "hidden_transformer_template"
STAGE_NAMES = (
    "open_backpack",
    "put_object_in_backpack",
    "close_backpack",
)
WEAK_LABEL_NOTICE = (
    "right-hand kinematic algorithmic weak labels; not human annotations or "
    "environment-predicate ground truth"
)


class CausalTransformerStageProbe(nn.Module):
    """Exact architecture saved by the remote ``train_probes.py`` trainer."""

    def __init__(self, input_dim: int, window: int, classes: int = 3) -> None:
        super().__init__()
        self.window = window
        self.input_projection = nn.Linear(input_dim, 256)
        self.position = nn.Parameter(torch.zeros(1, window, 256))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=2, norm=nn.LayerNorm(256)
        )
        self.classifier = nn.Linear(256, classes)

    def forward(self, values: Tensor, padding_mask: Tensor) -> Tensor:
        hidden = self.input_projection(values) + self.position
        causal_mask = torch.triu(
            torch.ones(
                self.window,
                self.window,
                dtype=torch.bool,
                device=values.device,
            ),
            diagonal=1,
        )
        hidden = self.encoder(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        last_valid = (~padding_mask).sum(dim=1).sub(1).clamp_min(0)
        selected = hidden[torch.arange(len(hidden), device=values.device), last_valid]
        return self.classifier(selected)


@dataclass(frozen=True)
class LoadedProbe:
    model: CausalTransformerStageProbe
    mean: Tensor
    scale: Tensor
    window: int
    input_dim: int
    seed: int
    stage_names: tuple[str, ...]
    trainable_params: int
    checkpoint_path: Path


@dataclass(frozen=True)
class EpisodeFeatures:
    hidden: np.ndarray
    timestamp: np.ndarray
    path: Path


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _checkpoint_array(value: Any, name: str, input_dim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (input_dim,):
        raise ValueError(f"checkpoint {name} must have shape ({input_dim},)")
    if not np.isfinite(array).all():
        raise ValueError(f"checkpoint {name} must contain only finite values")
    return array


def load_probe(
    checkpoint_path: Path,
    *,
    device: torch.device,
    expected_seed: int = 17,
) -> LoadedProbe:
    """Load one trusted local checkpoint and validate its training contract."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    artifact = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict):
        raise ValueError("checkpoint must contain a mapping")
    if artifact.get("model_name") != "transformer":
        raise ValueError("checkpoint model_name must be 'transformer'")
    seed = int(artifact.get("seed", -1))
    if seed != expected_seed:
        raise ValueError(f"checkpoint seed must be {expected_seed}, got {seed}")
    window = int(artifact.get("window", -1))
    if window != 8:
        raise ValueError(f"checkpoint window must be 8, got {window}")
    input_dim = int(artifact.get("input_dim", -1))
    if input_dim <= 0:
        raise ValueError("checkpoint input_dim must be positive")
    stage_names = tuple(artifact.get("stage_names", ()))
    if stage_names != STAGE_NAMES:
        raise ValueError(
            f"checkpoint stage_names must be {STAGE_NAMES!r}, got {stage_names!r}"
        )
    mean_array = _checkpoint_array(
        artifact.get("normalizer_mean"), "normalizer_mean", input_dim
    )
    scale_array = _checkpoint_array(
        artifact.get("normalizer_scale"), "normalizer_scale", input_dim
    )
    if np.any(scale_array <= 0):
        raise ValueError("checkpoint normalizer_scale must be positive")
    state_dict = artifact.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint state_dict must be a mapping")

    model = CausalTransformerStageProbe(input_dim, window, len(stage_names))
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    mean = torch.from_numpy(mean_array).to(device)
    scale = torch.from_numpy(scale_array).to(device)
    return LoadedProbe(
        model=model,
        mean=mean,
        scale=scale,
        window=window,
        input_dim=input_dim,
        seed=seed,
        stage_names=stage_names,
        trainable_params=sum(parameter.numel() for parameter in model.parameters()),
        checkpoint_path=checkpoint_path.resolve(),
    )


def load_inputs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        required = {
            "benchmark_id",
            "case_id",
            "episode_index",
            "anchor_seconds",
            "qa_mode",
        }
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(
                f"{path}:{line_number}: missing fields: {', '.join(missing)}"
            )
        if value["qa_mode"] not in {"qa", "intermediate_explanation"}:
            raise ValueError(f"{path}:{line_number}: unsupported qa_mode")
        rows.append(value)
    if not rows:
        raise ValueError(f"no campaign inputs in {path}")
    return rows


def load_episode_features(
    feature_dir: Path,
    episode_index: int,
    *,
    feature_key: str,
    input_dim: int,
) -> EpisodeFeatures:
    path = feature_dir / f"episode_{episode_index:06d}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        if feature_key not in archive or "timestamp" not in archive:
            raise ValueError(f"{path}: needs {feature_key!r} and 'timestamp' arrays")
        hidden = np.asarray(archive[feature_key], dtype=np.float32)
        timestamp = np.asarray(archive["timestamp"], dtype=np.float64)
    if hidden.ndim != 2 or hidden.shape[1] != input_dim:
        raise ValueError(f"{path}: {feature_key} must have shape [T, {input_dim}]")
    if timestamp.shape != (len(hidden),):
        raise ValueError(f"{path}: timestamp must have shape [T]")
    if len(hidden) == 0 or not np.isfinite(hidden).all():
        raise ValueError(f"{path}: features must be non-empty and finite")
    if not np.isfinite(timestamp).all() or np.any(np.diff(timestamp) <= 0):
        raise ValueError(f"{path}: timestamps must be finite and strictly increasing")
    return EpisodeFeatures(hidden=hidden, timestamp=timestamp, path=path.resolve())


def build_causal_window(
    features: EpisodeFeatures,
    anchor_seconds: float,
    probe: LoadedProbe,
) -> tuple[Tensor, Tensor, int]:
    if not math.isfinite(anchor_seconds):
        raise ValueError("anchor_seconds must be finite")
    anchor_index = (
        int(np.searchsorted(features.timestamp, anchor_seconds, side="right")) - 1
    )
    if anchor_index < 0:
        raise ValueError("anchor_seconds precedes the first feature timestamp")
    start = max(0, anchor_index - probe.window + 1)
    history = torch.from_numpy(features.hidden[start : anchor_index + 1]).to(
        probe.mean.device
    )
    history = (history - probe.mean) / probe.scale
    valid = len(history)
    values = torch.zeros(
        (1, probe.window, probe.input_dim),
        dtype=torch.float32,
        device=probe.mean.device,
    )
    padding_mask = torch.ones(
        (1, probe.window), dtype=torch.bool, device=probe.mean.device
    )
    values[0, :valid] = history
    padding_mask[0, :valid] = False
    return values, padding_mask, anchor_index


def predict_stage(
    probe: LoadedProbe,
    features: EpisodeFeatures,
    anchor_seconds: float,
) -> tuple[int, list[float], int]:
    values, padding_mask, anchor_index = build_causal_window(
        features, anchor_seconds, probe
    )
    with torch.inference_mode():
        logits = probe.model(values, padding_mask)
        probabilities = torch.softmax(logits, dim=-1)[0]
    return (
        int(torch.argmax(probabilities).item()),
        [float(value) for value in probabilities.cpu().tolist()],
        anchor_index,
    )


def render_answer(stage: int, qa_mode: str) -> str:
    qa_templates = (
        "지금은 가방을 여는 단계입니다.",
        "지금은 물체를 가방 안에 넣는 단계입니다.",
        "지금은 가방을 닫는 단계입니다.",
    )
    narration_templates = (
        "이제 가방을 여는 단계를 시작합니다.",
        "가방을 열었고, 이제 물체를 가방 안에 넣습니다.",
        "물체를 가방에 넣었고, 이제 가방을 닫습니다.",
    )
    if stage not in range(len(STAGE_NAMES)):
        raise ValueError(f"unsupported stage: {stage}")
    if qa_mode == "qa":
        return qa_templates[stage]
    if qa_mode == "intermediate_explanation":
        return narration_templates[stage]
    raise ValueError(f"unsupported qa_mode: {qa_mode!r}")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_vram_mib(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))


def run_campaign(
    inputs: list[dict[str, Any]],
    *,
    probe: LoadedProbe,
    feature_dir: Path,
    feature_key: str = "hidden_mean",
    repetitions: int = 1,
    anchor_tolerance_seconds: float = 1e-6,
    warmup_steps: int = 5,
) -> list[dict[str, Any]]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if not math.isfinite(anchor_tolerance_seconds) or anchor_tolerance_seconds < 0:
        raise ValueError("anchor_tolerance_seconds must be finite and non-negative")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    cache = {
        episode_index: load_episode_features(
            feature_dir,
            episode_index,
            feature_key=feature_key,
            input_dim=probe.input_dim,
        )
        for episode_index in {int(row["episode_index"]) for row in inputs}
    }
    device = probe.mean.device
    warmup_item = inputs[0]
    warmup_episode = cache[int(warmup_item["episode_index"])]
    for _ in range(warmup_steps):
        predict_stage(
            probe,
            warmup_episode,
            float(warmup_item["anchor_seconds"]),
        )
    _synchronize(device)
    output: list[dict[str, Any]] = []
    for item in inputs:
        episode_index = int(item["episode_index"])
        episode = cache[episode_index]
        input_anchor = float(item["anchor_seconds"])
        for repetition in range(repetitions):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            _synchronize(device)
            started = time.perf_counter()
            prefill_started = time.perf_counter()
            stage, probabilities, anchor_index = predict_stage(
                probe, episode, input_anchor
            )
            feature_anchor = float(episode.timestamp[anchor_index])
            anchor_delta = abs(feature_anchor - input_anchor)
            if anchor_delta > anchor_tolerance_seconds:
                raise ValueError(
                    f"{item['case_id']}: input anchor {input_anchor} does not match "
                    f"causal feature timestamp {feature_anchor}; delta={anchor_delta}"
                )
            _synchronize(device)
            prefill_ms = (time.perf_counter() - prefill_started) * 1000
            decode_started = time.perf_counter()
            answer = render_answer(stage, str(item["qa_mode"]))
            decode_ms = (time.perf_counter() - decode_started) * 1000
            end_to_end_ms = (time.perf_counter() - started) * 1000
            output.append(
                {
                    "schema_version": 1,
                    "benchmark_id": str(item["benchmark_id"]),
                    "method_id": METHOD_ID,
                    "method_family": "groot_cosmos",
                    "case_id": str(item["case_id"]),
                    "repetition": repetition,
                    "answer_text": answer,
                    "timing_ms": {
                        "end_to_end": end_to_end_ms,
                        "prefill": prefill_ms,
                        "decode": decode_ms,
                    },
                    "peak_vram_mib": _peak_vram_mib(device),
                    "generated_tokens": 1,
                    "image_encodes": 0,
                    "trainable_params": probe.trainable_params,
                    "adapter_type": "transformer_encoder",
                    "uses_question_conditioning": False,
                    "backbone_shared": True,
                    "metadata": {
                        "checkpoint": str(probe.checkpoint_path),
                        "checkpoint_seed": probe.seed,
                        "feature_source": str(episode.path),
                        "feature_key": feature_key,
                        "feature_extraction_excluded_from_timing": True,
                        "feature_stage_array_ignored": True,
                        "causal_window_frames": probe.window,
                        "feature_rate_fps": 1,
                        "anchor_feature_index": anchor_index,
                        "input_anchor_seconds": input_anchor,
                        "anchor_feature_seconds": feature_anchor,
                        "anchor_delta_seconds": anchor_delta,
                        "predicted_stage": stage,
                        "predicted_stage_name": probe.stage_names[stage],
                        "stage_probabilities": probabilities,
                        "label_status": WEAK_LABEL_NOTICE,
                        "classifier_question_conditioning": False,
                        "response_mode_conditioning": "qa_mode",
                        "generated_tokens_semantics": (
                            "one selected template; no language-model decoding"
                        ),
                        "peak_vram_scope": "probe inference only",
                        "unmeasured_warmup_steps": warmup_steps,
                    },
                }
            )
    return output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Trusted local transformer_seed_17.pt checkpoint.",
    )
    parser.add_argument("--feature-dir", required=True, type=Path)
    parser.add_argument("--feature-key", default="hidden_mean")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--anchor-tolerance-seconds", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    probe = load_probe(
        args.checkpoint,
        device=resolve_device(args.device),
    )
    rows = run_campaign(
        load_inputs(args.inputs),
        probe=probe,
        feature_dir=args.feature_dir,
        feature_key=args.feature_key,
        repetitions=args.repetitions,
        anchor_tolerance_seconds=args.anchor_tolerance_seconds,
        warmup_steps=args.warmup_steps,
    )
    write_jsonl(args.output, rows)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(rows),
                "episodes": len({row["case_id"].split("_")[1] for row in rows}),
                "method_id": METHOD_ID,
                "label_status": WEAK_LABEL_NOTICE,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
