from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .color_stop_probe import paired_split_ids
from .evaluate import SEEDS, load_episodes
from .modality_probe import add_modality_representations, calibration_from_artifacts
from .streaming_transformer import TransformerConfig, train_one


@dataclass(frozen=True)
class Candidate:
    name: str
    representation: str = "R3b_vision_tokens"
    window_size: int = 16
    model_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    feedforward_dim: int = 512
    dropout: float = 0.10
    stage_loss_weight: float = 1.0
    boundary_radius: int = 2
    include_input_delta: bool = False

    def transformer_config(self) -> TransformerConfig:
        return TransformerConfig(
            window_size=self.window_size,
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            feedforward_dim=self.feedforward_dim,
            dropout=self.dropout,
        )


BASELINE = Candidate(name="T0_baseline")
CANDIDATES = {
    candidate.name: candidate
    for candidate in (
        BASELINE,
        Candidate(name="T1_window4", window_size=4),
        Candidate(name="T2_window8", window_size=8),
        Candidate(
            name="T3_tiny64",
            model_dim=64,
            num_layers=1,
            feedforward_dim=128,
        ),
        Candidate(
            name="T4_small128",
            model_dim=128,
            num_layers=1,
            feedforward_dim=256,
        ),
        Candidate(name="T5_boundary_only", stage_loss_weight=0.0),
        Candidate(name="T6_exact_label", boundary_radius=0),
        Candidate(name="T7_radius1", boundary_radius=1),
        Candidate(name="T8_explicit_delta", include_input_delta=True),
        Candidate(name="T9_vision_instruction", representation="F1_vision_instruction"),
        Candidate(name="T10_vision_action", representation="F2_vision_action_output"),
        Candidate(
            name="T11_vision_instruction_action",
            representation="F4_vision_instruction_action_output",
        ),
        Candidate(name="T12_vision_dit", representation="F5_vision_action_hidden"),
        Candidate(
            name="T13_vision_instruction_dit",
            representation="F7_vision_instruction_action_hidden",
        ),
    )
}
COMBINATION_CANDIDATES = {
    candidate.name: candidate
    for candidate in (
        Candidate(
            name="C1_boundary_only_delta",
            stage_loss_weight=0.0,
            include_input_delta=True,
        ),
        Candidate(
            name="C2_boundary_only_exact",
            stage_loss_weight=0.0,
            boundary_radius=0,
        ),
        Candidate(
            name="C3_boundary_only_small128",
            model_dim=128,
            num_layers=1,
            feedforward_dim=256,
            stage_loss_weight=0.0,
        ),
        Candidate(
            name="C4_three_way_dit_boundary_only",
            representation="F7_vision_instruction_action_hidden",
            stage_loss_weight=0.0,
        ),
    )
}
ALL_CANDIDATES = {**CANDIDATES, **COMBINATION_CANDIDATES}
PARENTS = {
    "C1_boundary_only_delta": "T5_boundary_only",
    "C2_boundary_only_exact": "T5_boundary_only",
    "C3_boundary_only_small128": "T5_boundary_only",
    "C4_three_way_dit_boundary_only": "T13_vision_instruction_dit",
}


def changed_fields(candidate: Candidate, baseline: Candidate = BASELINE) -> tuple[str, ...]:
    ignored = {"name"}
    field_to_axis = {
        "model_dim": "capacity",
        "num_layers": "capacity",
        "feedforward_dim": "capacity",
    }
    return tuple(dict.fromkeys(
        field_to_axis.get(field, field)
        for field, value in asdict(candidate).items()
        if field not in ignored and value != asdict(baseline)[field]
    ))


def _parse_candidates(value: str) -> tuple[Candidate, ...]:
    names = tuple(CANDIDATES) if value == "all" else tuple(
        item.strip() for item in value.split(",") if item.strip()
    )
    unknown = set(names) - set(ALL_CANDIDATES)
    if unknown:
        raise ValueError(f"Unknown candidates: {sorted(unknown)}")
    return tuple(ALL_CANDIDATES[name] for name in names)


def _parse_seeds(value: str) -> tuple[int, ...]:
    return SEEDS if value == "all" else tuple(int(item) for item in value.split(","))


def _mean_ci(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    ci = float(1.96 * array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else float("nan")
    return mean, ci


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "boundary_event_f1_at_5",
        "boundary_event_f1_at_10",
        "false_boundaries_per_episode",
        "no_boundary_failure_fpr",
        "stage_macro_f1",
        "streaming_probe_latency_ms_per_sample",
    )
    output = []
    for name in sorted({str(row["candidate"]) for row in rows}):
        selected = [row for row in rows if row["candidate"] == name]
        item: dict[str, Any] = {
            "candidate": name,
            "representation": selected[0]["representation"],
            "seeds": len(selected),
            "parameter_count": selected[0]["parameter_count"],
            "window_size": selected[0]["window_size"],
            "changed_fields": selected[0]["changed_fields"],
        }
        for metric in metrics:
            mean, ci = _mean_ci([float(row[metric]) for row in selected])
            item[f"{metric}_mean"] = mean
            item[f"{metric}_ci95"] = ci
        output.append(item)
    return sorted(
        output,
        key=lambda row: (
            float(row["no_boundary_failure_fpr_mean"]) > 0.25,
            -float(row["boundary_event_f1_at_5_mean"]),
            float(row["false_boundaries_per_episode_mean"]),
            int(row["parameter_count"]),
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write empty campaign results")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    root = Path(args.artifacts)
    episodes = load_episodes(root)
    calibration = calibration_from_artifacts(root)
    for episode in episodes:
        add_modality_representations(episode, calibration)
    candidates = _parse_candidates(args.candidates)
    seeds = _parse_seeds(args.seeds)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_dir = output_dir / candidate.name
        run_dir = candidate_dir / "runs"
        run_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            run_path = run_dir / f"seed_{seed}.json"
            if run_path.is_file() and not args.force:
                result = json.loads(run_path.read_text())
            else:
                result = train_one(
                    episodes,
                    representation=candidate.representation,
                    common_256=True,
                    seed=seed,
                    config=candidate.transformer_config(),
                    device=device,
                    output_dir=candidate_dir,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    boundary_loss_weight=args.boundary_loss_weight,
                    patience=args.patience,
                    backbone_select_layer=16,
                    split_ids=paired_split_ids(episodes, root, seed),
                    stage_loss_weight=candidate.stage_loss_weight,
                    boundary_radius=candidate.boundary_radius,
                    include_input_delta=candidate.include_input_delta,
                    label_source="synthetic beaker liquid color-change step",
                )
                result.update(
                    {
                        "candidate": candidate.name,
                        "candidate_config": asdict(candidate),
                        "changed_fields": changed_fields(candidate),
                        "parent_candidate": PARENTS.get(candidate.name, "T0_baseline"),
                        "paired_counterfactual_split": True,
                    }
                )
                run_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            rows.append(result)
            print(
                json.dumps(
                    {
                        "candidate": candidate.name,
                        "seed": seed,
                        "boundary_f1_at_5": result["boundary_event_f1_at_5"],
                        "no_change_fpr": result["no_boundary_failure_fpr"],
                        "false_boundaries_per_episode": result[
                            "false_boundaries_per_episode"
                        ],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    summary = summarize(rows)
    (output_dir / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    _write_csv(output_dir / "results.csv", rows)
    _write_csv(output_dir / "summary.csv", summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts")
    parser.add_argument("--output-dir", default="reports/color_stop_transformer")
    parser.add_argument("--candidates", default="all")
    parser.add_argument("--seeds", default="17")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--boundary-loss-weight", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
