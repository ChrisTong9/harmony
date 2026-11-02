"""Check source identity and all sixteen profiles without invoking a model."""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = json.loads((ROOT / "reproducibility/SOURCES.json").read_text())
    compiled = 0
    for entry in sources.values():
        source = ROOT / entry["directory"]
        manifest = json.loads((source / "SOURCE.json").read_text())
        for name, expected in manifest["files"].items():
            path = source / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Source changed: {entry['directory']}/{name}")
            if path.suffix == ".py":
                compile(path.read_bytes(), str(path), "exec")
                compiled += 1
    subprocess.run([sys.executable, "-m", "harmony", "verify"], cwd=ROOT, check=True)
    with tempfile.TemporaryDirectory() as directory:
        for cell in ["A0B0", "A1B0", "A0B1", "A1B1"]:
            for task in ["add", "remove", "replace", "volume"]:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "harmony",
                        "run",
                        "--task",
                        task,
                        "--cell",
                        cell,
                        "--input",
                        "examples/input.wav",
                        "--instruction",
                        "Check configuration",
                        "--output-dir",
                        directory,
                        "--dry-run",
                    ],
                    cwd=ROOT,
                    check=True,
                    text=True,
                    capture_output=True,
                )
                plan = json.loads(result.stdout)
                if not plan["no_model_calls"] or not Path(plan["source"]).is_dir():
                    raise ValueError("Invalid dry-run profile")
    print(
        json.dumps(
            {
                "source_snapshots": len(sources),
                "python_files_compiled": compiled,
                "dry_run_profiles": 16,
                "model_calls": 0,
                "passed": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
