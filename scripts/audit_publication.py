"""Fail on likely credentials or unapproved binary artifacts; never print matched values."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [
    rb"hf_[A-Za-z0-9]{20,}",
    rb"gh[pousr]_[A-Za-z0-9]{20,}",
    rb"sk-[A-Za-z0-9_-]{24,}",
    rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
    rb"(?i)(?:password|api_key|access_token)\s*=\s*[\"'][^\"'\s]{8,}[\"']",
]


def main():
    paths = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    failures = []
    for relative in filter(None, paths):
        path = ROOT / relative
        data = path.read_bytes()
        if path.suffix in (".npz", ".npy"):
            if not relative.startswith("fixtures/"):
                failures.append((relative, "binary outside fixtures"))
            continue
        if path.name.startswith(".env") or path.suffix in (".pt", ".pth", ".safetensors", ".mp4"):
            failures.append((relative, "excluded artifact"))
        if relative != "scripts/audit_publication.py" and any(re.search(p, data) for p in PATTERNS):
            failures.append((relative, "possible credential; inspect locally"))
    for relative, reason in failures:
        print(relative, reason)
    if failures:
        raise SystemExit(1)
    print(
        f"Publication scan passed: {len(list(filter(None, paths)))} files (heuristic, not a guarantee)"
    )


if __name__ == "__main__":
    main()
