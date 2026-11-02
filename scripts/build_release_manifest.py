"""Freeze distributable files after an intentional packaging change."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "reproducibility/release_manifest.json"
EXCLUDED_PARTS = {".git", "__pycache__", ".ruff_cache", ".pytest_cache", "outputs", "resources"}


def main():
    files = {}
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if not path.is_file() or path == TARGET or any(p in EXCLUDED_PARTS for p in relative.parts):
            continue
        if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
            raise RuntimeError("Remove local credentials before creating a release manifest")
        if path.suffix in {".pyc", ".sqlite3", ".pem", ".key"}:
            raise RuntimeError(f"Unexpected private/generated file: {relative}")
        files[relative.as_posix()] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    TARGET.write_text(
        json.dumps({"kind": "harmony-source-release", "files": files}, indent=2) + "\n"
    )
    print(f"Recorded {len(files)} files in {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
