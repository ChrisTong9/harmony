from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


class QwenImageGenerator:
    """Run Qwen-Image-2512 in its own Python environment."""

    def __init__(
        self,
        python_executable: str | None = None,
        tool_path: str | None = None,
    ) -> None:
        self.python_executable = (
            python_executable or os.environ.get("QWEN_IMAGE_PYTHON") or sys.executable
        )
        self.tool_path = (
            Path(tool_path)
            if tool_path
            else (Path(__file__).resolve().parents[1] / "tools" / "run_qwen_image.py")
        )

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        output_dir: str | None = None,
        width: int = 1328,
        height: int = 1328,
        num_inference_steps: int = 50,
        true_cfg_scale: float = 4.0,
        seed: int = 42,
    ) -> str:
        if not prompt.strip():
            raise ValueError("Qwen image prompt must not be empty.")

        python_path = Path(self.python_executable)
        if not python_path.is_file():
            raise FileNotFoundError(
                "Qwen image Python environment does not exist: "
                f"{self.python_executable}. Set QWEN_IMAGE_PYTHON to the "
                "Qwen-Image-2512 environment's Python executable."
            )
        if not self.tool_path.is_file():
            raise FileNotFoundError(f"Qwen image tool does not exist: {self.tool_path}")

        repo_value = os.environ.get("QWEN_IMAGE_REPO_DIR", "").strip()
        if not repo_value:
            raise RuntimeError("Missing required environment variable: QWEN_IMAGE_REPO_DIR")
        repo_dir = Path(repo_value).expanduser().resolve()
        if not (repo_dir / "README.md").is_file():
            raise FileNotFoundError(f"Qwen-Image repository does not exist: {repo_dir}")

        request: dict[str, Any] = {
            "prompt": prompt.strip(),
            "negative_prompt": negative_prompt.strip(),
            "output_dir": (str(Path(output_dir).expanduser().resolve()) if output_dir else None),
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "true_cfg_scale": true_cfg_scale,
            "seed": seed,
        }

        with tempfile.TemporaryDirectory(prefix="qwen_image_adapter_") as tmp:
            tmp_dir = Path(tmp)
            request_path = tmp_dir / "request.json"
            response_path = tmp_dir / "response.json"
            request_path.write_text(
                json.dumps(request, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    self.python_executable,
                    str(self.tool_path),
                    "--request",
                    str(request_path),
                    "--response",
                    str(response_path),
                ],
                cwd=str(repo_dir),
                text=True,
                capture_output=True,
                check=False,
            )

            if proc.returncode != 0:
                details = (proc.stderr or proc.stdout or "No subprocess output.").strip()
                raise RuntimeError(
                    f"Qwen-Image-2512 generation failed with exit code {proc.returncode}: {details}"
                )
            if not response_path.is_file():
                raise RuntimeError("Qwen image process did not write a response JSON.")

            response = json.loads(response_path.read_text(encoding="utf-8"))
            if not response.get("ok"):
                raise RuntimeError(
                    f"Qwen-Image-2512 generation failed: {response.get('error', 'unknown error')}"
                )

            output_image = response.get("output_image")
            if not output_image or not Path(output_image).is_file():
                raise RuntimeError(f"Qwen image output does not exist: {output_image}")
            return str(Path(output_image).resolve())
