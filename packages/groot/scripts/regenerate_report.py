from __future__ import annotations

import argparse
import csv
from pathlib import Path

from groot_subtask_phase_probe.evaluate import _write_report, load_episodes

TEXT_FIELDS = {"representation", "projection"}
INTEGER_FIELDS = {
    "seed",
    "native_dim",
    "probe_dim",
    "terminal_test_samples",
    "peak_vram_bytes",
    "host_rss_delta_bytes",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.results_csv.open(newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = []
    for raw in raw_rows:
        row = {}
        for name, value in raw.items():
            if name in TEXT_FIELDS:
                row[name] = value
            elif name in INTEGER_FIELDS:
                row[name] = int(float(value))
            else:
                row[name] = float(value)
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_report(rows, load_episodes(args.artifacts), args.output_dir)


if __name__ == "__main__":
    main()
