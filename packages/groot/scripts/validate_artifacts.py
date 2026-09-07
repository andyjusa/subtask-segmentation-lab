from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    common_arrays: set[str] | None = None
    episodes = 0
    for feature_path in sorted(args.artifacts.glob("*/*/features.npz")):
        episodes += 1
        metadata_path = feature_path.with_name("metadata.json")
        video_path = feature_path.with_name("rollout.mp4")
        if not metadata_path.is_file():
            errors.append(f"missing metadata: {metadata_path}")
        if not video_path.is_file() or video_path.stat().st_size == 0:
            errors.append(f"missing or empty video: {video_path}")
        else:
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if probe.returncode != 0 or float(probe.stdout.strip() or 0) <= 0:
                errors.append(f"invalid video: {video_path}: {probe.stderr.strip()}")
        with np.load(feature_path) as archive:
            names = set(archive.files)
            common_arrays = names if common_arrays is None else common_arrays & names
            lengths = {name: archive[name].shape[0] for name in archive.files}
            expected = lengths["stage"]
            for name, length in lengths.items():
                if length != expected:
                    errors.append(
                        f"length mismatch: {feature_path} {name}={length} stage={expected}"
                    )
                if name.startswith("R") and not np.isfinite(archive[name]).all():
                    errors.append(f"non-finite feature: {feature_path} {name}")

    print(
        json.dumps(
            {
                "episodes": episodes,
                "errors": errors,
                "common_arrays": sorted(common_arrays or []),
            },
            indent=2,
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
