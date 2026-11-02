from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


class MMAudioGenerator:
    """
    Agent-side wrapper for MMAudio.

    This file runs in your main agent environment.
    It does NOT import MMAudio directly.
    It calls audio_edit/tools/run_mmaudio.py through subprocess.
    """

    def __init__(self):
        self.python_bin = os.environ.get("MMAUDIO_PYTHON", sys.executable)

        project_root = Path(__file__).resolve().parents[2]
        self.script_path = project_root / "audio_edit" / "tools" / "run_mmaudio.py"

        if not self.script_path.exists():
            raise FileNotFoundError(f"run_mmaudio.py not found: {self.script_path}")

        self.output_dir = project_root / "audio_edit" / "outputs" / "mmaudio"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        prompt: str,
        video_path: str,
        negative_prompt: str | None = None,
        duration: float | int | None = 8,
    ) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("MMAudio prompt must be a non-empty string.")

        if not video_path:
            raise ValueError("MMAudio requires video_path.")

        output_path = self.output_dir

        request = {
            "prompt": prompt.strip(),
            "negative_prompt": negative_prompt,
            "duration": duration,
            "video_path": video_path,
            "output_path": str(self.output_dir),
        }

        cmd = [
            self.python_bin,
            str(self.script_path),
            json.dumps(request),
        ]

        result = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if result.stderr:
            pass

        if result.returncode != 0:
            raise RuntimeError(f"MMAudio subprocess failed with return code {result.returncode}")

        try:
            response = json.loads(result.stdout.strip().splitlines()[-1])
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse MMAudio JSON response. stdout:\n{result.stdout}"
            ) from e

        if not response.get("ok"):
            raise RuntimeError(f"MMAudio generation failed: {response}")

        audio_path = response["audio_path"]

        return audio_path
