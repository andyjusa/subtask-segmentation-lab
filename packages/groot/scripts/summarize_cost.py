from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.quantile(array, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    result = {
        "samples": len(metadata["snapshot_latency_ms"]),
        "snapshot": stats(metadata["snapshot_latency_ms"]),
        "hook": stats(metadata["hook_latency_ms"]),
        "model_inference": stats(metadata["inference_latency_ms"]),
        "representations": {
            name: stats(values)
            for name, values in metadata["feature_extraction_latency_ms"].items()
        },
        "peak_vram_bytes": int(metadata["peak_vram_bytes"]),
        "host_rss_delta_bytes": int(metadata["host_rss_delta_bytes"]),
        "host_peak_rss_delta_bytes": int(metadata["host_peak_rss_delta_bytes"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
