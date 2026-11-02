from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


class MixerRunner:
    def __init__(self, output_dir: str = "./audio_edit/outputs"):
        self.project_root = Path(__file__).resolve().parents[2]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.last_warnings = []
        self.last_audit = []

        self.python_path = os.getenv(
            "MIXER_PYTHON",
            str(self.project_root / ".venvs" / "mixer" / "bin" / "python"),
        )

        self.script_path = os.getenv(
            "MIXER_SCRIPT",
            str(self.project_root / "audio_edit" / "tools" / "run_mix.py"),
        )

    def change_volume(
        self,
        original_path: str,
        gain_db: float,
        placements: list[dict] | None = None,
        gain_profile: str = "static",
    ) -> str:
        request_path = self.output_dir / "volume_request.json"
        response_path = self.output_dir / "volume_response.json"
        payload = {
            "operation": "volume",
            "gain_profile": gain_profile,
            "original_path": original_path,
            "output_dir": str(self.output_dir),
            "gain_db": gain_db,
            "placements": placements,
        }
        with open(request_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        cmd = [
            self.python_path,
            self.script_path,
            "--request-json",
            str(request_path.resolve()),
            "--response-json",
            str(response_path.resolve()),
        ]
        env = os.environ.copy()
        project_root = str(self.project_root)
        old_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            project_root if not old_pythonpath else project_root + os.pathsep + old_pythonpath
        )
        subprocess.run(cmd, check=True, env=env)
        with open(response_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        self.last_warnings = result.get("warnings", [])
        self.last_audit = result.get("audit", [])
        return result["output_audio"]

    def mix_background(
        self,
        original_path: str,
        background_path: str,
        start_time: float,
        end_time: float | None,
        acoustic_feature: dict | None,
        placements: list[dict] | None = None,
    ) -> str:
        request_path = self.output_dir / "mixer_request.json"
        response_path = self.output_dir / "mixer_response.json"

        features = dict(acoustic_feature or {})
        features["minimum_added_rms_dbfs"] = -30.0

        payload = {
            "original_path": original_path,
            "background_path": background_path,
            "output_dir": str(self.output_dir),
            "start_time": start_time,
            "end_time": end_time,
            "placements": placements,
            "acoustic_features": features,
        }

        with open(request_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        cmd = [
            self.python_path,
            self.script_path,
            "--request-json",
            str(request_path.resolve()),
            "--response-json",
            str(response_path.resolve()),
        ]

        env = os.environ.copy()
        project_root = str(self.project_root)
        old_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            project_root if not old_pythonpath else project_root + os.pathsep + old_pythonpath
        )

        subprocess.run(cmd, check=True, env=env)

        with open(response_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        self.last_warnings = result.get("warnings", [])
        self.last_audit = result.get("audit", [])
        return result["output_audio"]
