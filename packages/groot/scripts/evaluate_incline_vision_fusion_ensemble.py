from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import train_tapnextpp_subtask_probe as baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-dirs",
        type=Path,
        nargs="+",
        default=[
            Path(f"reports/incline_new/vision_fusion/seed_{seed}/predictions/residual")
            for seed in (17, 29, 41, 53, 67)
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/incline_new/vision_fusion/ensemble"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    boundary_errors: list[float] = []
    episodes = []
    reference_files = sorted(args.prediction_dirs[0].glob("episode_*.npz"))
    for reference_file in reference_files:
        arrays = [np.load(path / reference_file.name) for path in args.prediction_dirs]
        probabilities = np.mean(
            [array["probabilities"].astype(np.float32) for array in arrays], axis=0
        )
        reference = arrays[0]
        prediction, boundary_samples = baseline.ordered_decode(probabilities, 5.0)
        predicted_boundaries = reference["sample_frames"][boundary_samples]
        errors = (
            np.abs(predicted_boundaries - reference["reference_boundaries"]) / 30.0
        )
        episode_id = int(reference_file.stem.rsplit("_", maxsplit=1)[1])
        episodes.append(
            {
                "episode": episode_id,
                "ordered_macro_f1": float(
                    f1_score(reference["labels"], prediction, average="macro")
                ),
                "boundary_mean_error_s": float(errors.mean()),
            }
        )
        np.savez_compressed(
            args.output_dir / reference_file.name,
            probabilities=probabilities.astype(np.float16),
            ordered=prediction.astype(np.int8),
            labels=reference["labels"],
            sample_frames=reference["sample_frames"],
            predicted_boundaries=predicted_boundaries,
            reference_boundaries=reference["reference_boundaries"],
        )
        labels_all.append(reference["labels"])
        predictions_all.append(prediction)
        boundary_errors.extend(errors.tolist())

    labels = np.concatenate(labels_all)
    predictions = np.concatenate(predictions_all)
    errors = np.asarray(boundary_errors)
    result = {
        "members": [str(path) for path in args.prediction_dirs],
        "ordered_macro_f1": float(f1_score(labels, predictions, average="macro")),
        "ordered_balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "boundary_mean_absolute_error_s": float(errors.mean()),
        "boundary_median_absolute_error_s": float(np.median(errors)),
        "boundary_within_1s_rate": float(np.mean(errors <= 1.0)),
        "boundary_within_2s_rate": float(np.mean(errors <= 2.0)),
        "episodes": episodes,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
