from __future__ import annotations

import os
import subprocess
from pathlib import Path


class StableAudioGenerator:
    def __init__(self, output_dir: str = "./audio_edit/outputs/generated"):
        project_root = Path(__file__).resolve().parents[2]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.python_path = os.getenv(
            "STABLE_AUDIO_PYTHON",
            str(project_root / "external" / "stable-audio-3" / ".venv" / "bin" / "python"),
        )

        self.script_path = os.getenv(
            "STABLE_AUDIO_SCRIPT",
            str(project_root / "audio_edit" / "tools" / "run_stable_audio.py"),
        )

    def generate(
        self,
        prompt: str,
        duration: float,
        negative_prompt: str | None = None,
        reference_path: str | None = None,
        reference_start: float = 0.0,
        reference_end: float | None = None,
        reference_mode: str | None = None,
        init_noise_level: float | None = None,
        complete_event: bool = False,
        seed: int | None = None,
    ) -> str:
        output_path = self.output_dir / "stable_audio_generated.wav"

        cmd = [
            self.python_path,
            self.script_path,
            "--prompt",
            prompt,
            "--duration",
            str(duration),
            "--output",
            str(output_path.resolve()),
        ]

        if negative_prompt:
            cmd.extend(["--negative-prompt", negative_prompt])
        if complete_event:
            if reference_path is not None:
                raise ValueError("Independent event decoding cannot use an audio reference")
            cmd.append("--complete-event")
        if seed is not None:
            if type(seed) is not int or not 0 <= seed < 2**31:
                raise ValueError("Generation seed must be an integer in [0, 2**31)")
            cmd.extend(["--seed", str(seed)])

        if reference_path is not None:
            if reference_mode not in ("structure", "context"):
                raise ValueError("Reference audio requires a declared conditioning mode")
            cmd.extend(
                [
                    "--reference-audio",
                    str(Path(reference_path).resolve()),
                    "--reference-start",
                    str(reference_start),
                    "--reference-mode",
                    reference_mode,
                ]
            )
            if reference_end is not None:
                cmd.extend(["--reference-end", str(reference_end)])
        elif reference_mode is not None:
            raise ValueError("Reference conditioning requires actual audio")

        if init_noise_level is not None:
            import math

            if (
                reference_path is None
                or isinstance(init_noise_level, bool)
                or not math.isfinite(init_noise_level)
                or not 0 <= init_noise_level <= 1
            ):
                raise ValueError(
                    "init_noise_level requires reference audio and a finite value in [0,1]"
                )
            cmd.extend(["--init-noise-level", str(init_noise_level)])

        subprocess.run(cmd, check=True)

        return str(output_path)
