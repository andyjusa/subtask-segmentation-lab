"""CPU reproduction using real saved features; never controls a robot or calls an API."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages/groot/scripts"))


def rollout(output: Path) -> dict:
    from evaluate_rollout_incline_vision_hidden import (
        evaluate_representation,
        load_boundaries,
        stage_labels,
    )

    _, boundaries = load_boundaries(ROOT / "fixtures/rollout/boundaries.json")
    variants = {}
    with np.load(ROOT / "fixtures/rollout/groot_backbone_hidden.npz") as data:
        frames = data["sample_frames"]
        labels = stage_labels(frames, boundaries)
        for key in ("vision_mean", "hidden_mean"):
            for components in (None, 256):
                name = f"{key}_{components or 'native'}"
                metrics, probabilities = evaluate_representation(
                    data[key].astype(np.float32),
                    labels,
                    frames,
                    boundaries,
                    sample_fps=5.0,
                    pca_components=components,
                )
                variants[name] = metrics
                np.savez_compressed(
                    output / f"{name}.npz",
                    probabilities=probabilities,
                    sample_frames=frames,
                    labels=labels,
                )
    return {
        "scope": "legacy single-episode stage-aware OOF; NONCAUSAL ordered decoder",
        "samples": len(frames),
        "variants": variants,
    }


def incline(output: Path, args: argparse.Namespace) -> dict:
    import torch
    import train_tapnextpp_subtask_probe as baseline
    from sklearn.preprocessing import StandardScaler

    torch.set_num_threads(args.threads)
    fixture = ROOT / "fixtures/incline50"
    split = json.loads((fixture / "split.json").read_text())
    boundaries = baseline.load_boundaries(fixture / "boundaries.csv")
    episodes = {}
    for partition in ("train", "validation"):
        for episode in split[partition]:
            path = fixture / partition / f"episode_{episode:03d}" / "groot_backbone_hidden.npz"
            with np.load(path) as data:
                frames = data["sample_frames"].astype(np.int64)
                episodes[episode] = baseline.EpisodeData(
                    episode,
                    data[args.hidden_key].astype(np.float32),
                    baseline.stage_labels(frames, boundaries[episode]),
                    frames,
                    boundaries[episode],
                )
    shuffled = np.random.default_rng(args.seed).permutation(split["train"])
    selection_ids, train_ids = sorted(shuffled[:8].tolist()), sorted(shuffled[8:].tolist())
    train_x, train_y = baseline.concatenate(episodes, train_ids)
    selection_x, selection_y = baseline.concatenate(episodes, selection_ids)
    scaler = StandardScaler().fit(train_x)
    train_x = np.clip(scaler.transform(train_x), -8, 8).astype(np.float32)
    selection_x = np.clip(scaler.transform(selection_x), -8, 8).astype(np.float32)
    results = {}
    device = torch.device("cpu")
    for name in ("linear", "mlp"):
        model, history, epoch = baseline.train_model(
            name,
            train_x,
            train_y,
            selection_x,
            selection_y,
            device,
            args.epochs,
            12,
            256,
            args.seed,
        )
        results[name] = baseline.evaluate_model(
            model,
            episodes,
            split["validation"],
            scaler,
            device,
            256,
            5.0,
            output,
            name,
        )
        results[name]["best_epoch"] = epoch
        (output / f"{name}_history.json").write_text(json.dumps(history, indent=2))
        torch.save(model.state_dict(), output / f"{name}.pt")
    np.savez_compressed(output / "scaler.npz", mean=scaler.mean_, scale=scaler.scale_)
    return {
        "scope": "episode-held-out; NONCAUSAL ordered decoder; weak labels",
        "train": train_ids,
        "selection": selection_ids,
        "validation": split["validation"],
        "hidden_key": args.hidden_key,
        "seed": args.seed,
        "variants": results,
    }


def pi05(output: Path, threads: int) -> dict:
    import torch
    from vla_subtask_phase_probe.training import train_linear_probe

    torch.set_num_threads(threads)
    actions = np.load(ROOT / "fixtures/pi05/actions.npy")
    result = train_linear_probe(actions, (170, 354))
    result.parameters.save(output / "probe.npz", ("picking", "placing", "complete"))
    np.save(output / "predictions.npy", result.predictions)
    return {
        "scope": "same-episode stride validation; accuracy is NOT F1",
        "frames": len(actions),
        "action_dim": actions.shape[1],
        "validation_accuracy": result.validation_accuracy,
        "transitions": (np.flatnonzero(np.diff(result.predictions)) + 1).tolist(),
    }


def verify() -> dict:
    manifest = json.loads((ROOT / "fixtures/manifest.json").read_text())
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"fixture checksum mismatch: {entry['path']}")
    return {"verified_files": len(manifest["files"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("verify", "rollout", "incline50", "pi05"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--hidden-key", choices=("vision_mean", "hidden_mean"), default="vision_mean"
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error("epochs and threads must be positive")
    if args.experiment == "verify":
        print(json.dumps(verify()))
        return
    output = args.output or ROOT / "outputs" / args.experiment
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        parser.error("output already contains results.json; choose a new --output")
    if args.experiment == "rollout":
        result = rollout(output)
    elif args.experiment == "incline50":
        result = incline(output, args)
    else:
        result = pi05(output, args.threads)
    (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {output / 'results.json'}")


if __name__ == "__main__":
    main()
