from __future__ import annotations

import os
from pathlib import Path
import subprocess
import uuid

from audio_edit.generators.gemini_omni_video_generator import (
    GeminiOmniVideoGenerator,
)
from audio_edit.models import AudioEditContext
from audio_edit.plans.replace_video_plan import ReplaceVideoPlan


class GeminiOmniVideoRunner:
    """Run Gemini Omni video replacement and remove its generated audio."""

    def __init__(
        self,
        generator: GeminiOmniVideoGenerator | None = None,
    ) -> None:
        self.generator = generator or GeminiOmniVideoGenerator()

    def run(
        self,
        ctx: AudioEditContext,
        plan: ReplaceVideoPlan,
    ) -> AudioEditContext:
        input_video = getattr(ctx, "input_video", None)
        if not input_video:
            raise ValueError("GeminiOmniVideoRunner requires ctx.input_video.")

        input_path = Path(input_video).expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"Input video does not exist: {input_path}")

        output_dir = self._resolve_output_dir(plan)
        output_dir.mkdir(parents=True, exist_ok=True)

        raw_video = self.generator.generate(
            video_path=str(input_path),
            prompt=plan.edit_prompt,
            output_dir=str(output_dir),
        )

        silent_video = (output_dir / f"replace_video_{uuid.uuid4().hex}_silent.mp4").resolve()
        ffmpeg = os.environ.get("FFMPEG_BIN", "ffmpeg")
        command = [
            ffmpeg,
            "-y",
            "-i",
            raw_video,
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-an",
            "-movflags",
            "+faststart",
            str(silent_video),
        ]
        proc = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            details = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"Could not mute Gemini Omni video: {details}")
        if not silent_video.is_file():
            raise RuntimeError(f"Muted Gemini Omni video does not exist: {silent_video}")

        ctx.output_video = str(silent_video)
        return ctx

    @staticmethod
    def _resolve_output_dir(plan: ReplaceVideoPlan) -> Path:
        if plan.output_dir:
            return Path(plan.output_dir).expanduser().resolve()

        env_dir = os.environ.get("GEMINI_OMNI_OUTPUT_DIR")
        if env_dir:
            return Path(env_dir).expanduser().resolve()

        return (Path(__file__).resolve().parents[1] / "outputs" / "gemini_omni").resolve()
