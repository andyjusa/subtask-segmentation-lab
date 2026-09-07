"""Record fixture provenance/checksums, or inventory source files (no network)."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "incline50": "leapshared/Incline_new_20260902_203009",
    "rollout": "leapshared/rollout_Incline_20260903_20260903_190043",
    "pi05": "local pi05-output-boundary-s0.actions.npy (SO101 simulation)",
}


def main():
    files = []
    for directory, source in SOURCES.items():
        for path in sorted((ROOT / "fixtures" / directory).rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "path": path.relative_to(ROOT).as_posix(),
                        "source": source,
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
    payload = {
        "notes": "Derived observations/actions, NOT model weights. Source HF revision was not recorded in the historical run; checksums pin these fixtures.",
        "files": files,
    }
    (ROOT / "fixtures/manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
