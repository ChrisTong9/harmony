from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


class ReCoRefGenerator:
    """
    Main-environment adapter for ReCo_ref.

    This module never imports ReCo. Instead it starts audio_edit/tools/run_reco_ref.py
    using the Python interpreter configured by RECOREF_PYTHON. That process can live
    in a completely separate Conda/venv environment containing ReCo dependencies.
    """

    def __init__(
        self,
        python_executable: str | None = None,
        tool_path: str | None = None,
    ) -> None:
        self.python_executable = (
            python_executable
            or os.environ.get("RECOREF_PYTHON")
            or sys.executable
        )
        self.tool_path = Path(tool_path) if tool_path else (
            Path(__file__).resolve().parents[1] / "tools" / "run_reco_ref.py"
        )

    def generate(
        self,
        *,
        video_path: str,
        prompt: str,
        reference_image: str | None = None,
        first_frame_image: str | None = None,
        output_dir: str | None = None,
    ) -> str:
        video = Path(video_path)
        if not video.is_file():
            raise FileNotFoundError(f"Input video does not exist: {video}")

        prompt = prompt.strip()
        if not prompt:
            raise ValueError("ReCo_ref prompt must not be empty.")

        self._validate_optional_file(reference_image, "reference_image")
        self._validate_optional_file(first_frame_image, "first_frame_image")

        request: dict[str, Any] = {
            "video_path": str(video.resolve()),
            "prompt": prompt,
            "reference_image": self._resolve_optional(reference_image),
            "first_frame_image": self._resolve_optional(first_frame_image),
            "output_dir": str(Path(output_dir).resolve()) if output_dir else None,
        }

        with tempfile.TemporaryDirectory(prefix="reco_ref_adapter_") as tmp:
            tmp_dir = Path(tmp)
            request_path = tmp_dir / "request.json"
            response_path = tmp_dir / "response.json"

            request_path.write_text(
                json.dumps(request, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            command = [
                self.python_executable,
                str(self.tool_path),
                "--request",
                str(request_path),
                "--response",
                str(response_path),
            ]

            proc = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
            )

            if proc.returncode != 0:
                stderr = (proc.stderr or "").strip()
                stdout = (proc.stdout or "").strip()
                details = stderr or stdout or "No subprocess output."
                raise RuntimeError(
                    f"ReCo_ref adapter failed with exit code "
                    f"{proc.returncode}: {details}"
                )

            if not response_path.is_file():
                raise RuntimeError(
                    "ReCo_ref adapter succeeded but did not write a response JSON."
                )

            try:
                response = json.loads(response_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid ReCo_ref response JSON: {exc}"
                ) from exc

            if not response.get("ok"):
                raise RuntimeError(
                    f"ReCo_ref inference failed: "
                    f"{response.get('error', 'unknown error')}"
                )

            output_video = response.get("output_video")
            if not output_video:
                raise RuntimeError("ReCo_ref response is missing output_video.")

            output_path = Path(output_video)
            if not output_path.is_file():
                raise RuntimeError(
                    f"ReCo_ref reported an output that does not exist: {output_path}"
                )

            return str(output_path.resolve())

    @staticmethod
    def _validate_optional_file(path: str | None, field_name: str) -> None:
        if path and not Path(path).is_file():
            raise FileNotFoundError(f"{field_name} does not exist: {path}")

    @staticmethod
    def _resolve_optional(path: str | None) -> str | None:
        return str(Path(path).resolve()) if path else None
