from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise ValueError("Usage: python run_mmaudio.py '<json_request>'")

    request = json.loads(sys.argv[1])

    prompt = request["prompt"]
    duration = request.get("duration", 8)
    video_path = Path(request["video_path"]).expanduser().resolve()
    output_dir = Path(request["output_path"]).expanduser().resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    repo_dir = Path(os.environ["MMAUDIO_REPO_DIR"]).expanduser().resolve()
    demo_script = repo_dir / "demo.py"

    if not repo_dir.exists():
        raise FileNotFoundError(f"MMAudio repo not found: {repo_dir}")

    if not demo_script.exists():
        raise FileNotFoundError(f"MMAudio demo.py not found: {demo_script}")

    cmd = [
        sys.executable,
        str(demo_script),
        f"--duration={duration}",
        f"--video={str(video_path)}",
        "--prompt",
        prompt,
        "--output",
        str(output_dir),
    ]

    result = subprocess.run(
        cmd,
        cwd=str(repo_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.stdout:
        pass

    if result.stderr:
        pass

    if result.returncode != 0:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "MMAudio demo.py failed",
                    "returncode": result.returncode,
                    "stderr": result.stderr,
                }
            )
        )
        sys.exit(result.returncode)

    audio_candidates = [
        p
        for p in output_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    ]

    if not audio_candidates:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"No audio file generated in {output_dir}",
                }
            )
        )
        sys.exit(1)

    audio_path = max(audio_candidates, key=lambda p: p.stat().st_mtime)

    print(
        json.dumps(
            {
                "ok": True,
                "audio_path": str(audio_path),
            }
        )
    )


if __name__ == "__main__":
    main()
