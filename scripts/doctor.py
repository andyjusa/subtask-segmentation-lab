"""Read-only environment and fixture diagnostics; never print environment secrets."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys

from reproduce import verify


def main():
    report = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "packages": {},
        "errors": [],
    }
    if sys.version_info[:2] != (3, 12):
        report["errors"].append("Python 3.12 is required; use uv sync --frozen --extra train")
    for name in ("numpy", "scipy", "scikit-learn", "torch", "pytest"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["errors"].append(f"missing package: {name}")
    try:
        report.update(verify())
    except (OSError, ValueError, KeyError) as exc:
        report["errors"].append(f"fixtures: {exc}")
    report["ok"] = not report["errors"]
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
