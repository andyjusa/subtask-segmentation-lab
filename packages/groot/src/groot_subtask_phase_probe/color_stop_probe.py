from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .evaluate import SEEDS, Episode, load_episodes
from .modality_probe import (
    MODALITY_REPRESENTATIONS,
    add_modality_representations,
    calibration_from_artifacts,
    evaluate_one,
    write_outputs,
)


def paired_split_ids(
    episodes: list[Episode], root: Path, seed: int
) -> tuple[set[str], set[str], set[str]]:
    """Split counterfactual pair IDs so change/no-change twins never leak."""

    pair_to_episodes: dict[int, set[str]] = {}
    for episode in episodes:
        metadata = json.loads((root / episode.episode_id / "metadata.json").read_text())
        pair_to_episodes.setdefault(int(metadata["pair_id"]), set()).add(episode.episode_id)
    if any(len(ids) != 2 for ids in pair_to_episodes.values()):
        raise ValueError("Every pair_id must contain exactly change and no-change episodes")

    pairs = np.asarray(sorted(pair_to_episodes))
    np.random.default_rng(seed).shuffle(pairs)
    n = len(pairs)
    if n < 5:
        raise ValueError("At least five counterfactual pairs are required")
    n_train = max(1, int(np.floor(0.6 * n)))
    n_validation = max(1, int(np.floor(0.2 * n)))
    if n_train + n_validation >= n:
        n_train, n_validation = n - 2, 1

    def expand(pair_ids: np.ndarray) -> set[str]:
        return set().union(*(pair_to_episodes[int(pair_id)] for pair_id in pair_ids))

    return (
        expand(pairs[:n_train]),
        expand(pairs[n_train : n_train + n_validation]),
        expand(pairs[n_train + n_validation :]),
    )


def run(args: argparse.Namespace) -> None:
    root = Path(args.artifacts)
    episodes = load_episodes(root)
    calibration = calibration_from_artifacts(root)
    for episode in episodes:
        add_modality_representations(episode, calibration)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    rows = []
    if args.resume and progress_path.exists():
        rows = [json.loads(line) for line in progress_path.read_text().splitlines() if line]
    completed = {
        (row["representation"], row["classifier"], row["projection"], int(row["seed"]))
        for row in rows
    }
    split_by_seed = {seed: paired_split_ids(episodes, root, seed) for seed in args.seeds}
    jobs = [
        (representation, classifier, common_256, seed)
        for seed in args.seeds
        for representation in args.representations
        for classifier in args.classifiers
        for common_256 in args.projections
        if (
            representation,
            classifier,
            "common_256" if common_256 else "native",
            seed,
        )
        not in completed
    ]

    def evaluate(job):
        representation, classifier, common_256, seed = job
        return evaluate_one(
            episodes,
            representation,
            classifier,
            common_256,
            seed,
            split_ids=split_by_seed[seed],
        )

    mode = "a" if args.resume else "w"
    with progress_path.open(mode, buffering=1) as progress, ThreadPoolExecutor(
        max_workers=args.jobs
    ) as executor:
        futures = {executor.submit(evaluate, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            progress.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"{row['representation']} {row['classifier']} {row['projection']} "
                f"seed={row['seed']} stage={row['stage_macro_f1']:.3f} "
                f"boundary={row['boundary_event_f1_at_5']:.3f}",
                flush=True,
            )
    write_outputs(rows, Path(args.output_dir))
    split_audit = {
        str(seed): [sorted(ids) for ids in paired_split_ids(episodes, root, seed)]
        for seed in args.seeds
    }
    Path(args.output_dir, "paired_splits.json").write_text(
        json.dumps(split_audit, ensure_ascii=False, indent=2) + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts_color_stop")
    parser.add_argument("--output-dir", default="reports/color_stop")
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=MODALITY_REPRESENTATIONS,
        default=list(MODALITY_REPRESENTATIONS),
    )
    parser.add_argument("--classifiers", nargs="+", choices=("linear", "mlp"), default=["linear", "mlp"])
    parser.add_argument(
        "--projections",
        nargs="+",
        choices=("native", "common_256"),
        default=["common_256"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.projections = [projection == "common_256" for projection in args.projections]
    run(args)


if __name__ == "__main__":
    main()
