from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import uuid

from audio_edit.generators.reco_ref_generator import ReCoRefGenerator
from audio_edit.models import AudioEditContext
from audio_edit.plans.replace_video_plan import ReplaceVideoPlan


class ReplaceVideoRunner:
    def __init__(
        self,
        generator: ReCoRefGenerator | None = None,
    ) -> None:
        self.generator = generator or ReCoRefGenerator()

    def run(
        self,
        ctx: AudioEditContext,
        plan: ReplaceVideoPlan,
    ) -> AudioEditContext:
        input_video = getattr(ctx, "input_video", None)
        if not input_video:
            raise ValueError("ReplaceVideoRunner requires ctx.input_video.")

        input_path = Path(input_video)
        if not input_path.is_file():
            raise FileNotFoundError(f"Input video does not exist: {input_path}")

        output_dir = self._resolve_output_dir(plan)
        output_dir.mkdir(parents=True, exist_ok=True)

        ffmpeg = os.environ.get("FFMPEG_BIN", "ffmpeg")
        with tempfile.TemporaryDirectory(
            prefix="reco_preprocess_",
            dir=str(output_dir),
        ) as tmp_dir:
            preprocessed_video = Path(tmp_dir) / "input.mp4"
            preprocess_command = [
                ffmpeg,
                "-y",
                "-i",
                str(input_path),
                "-vf",
                (
                    "fps=16,"
                    "scale=832:480:force_original_aspect_ratio=decrease,"
                    "pad=832:480:(ow-iw)/2:(oh-ih)/2,"
                    "tpad=stop_mode=clone:stop_duration=5.1"
                ),
                "-frames:v",
                "81",
                "-an",
                "-pix_fmt",
                "yuv420p",
                str(preprocessed_video),
            ]
            proc = subprocess.run(
                preprocess_command,
                text=True,
                capture_output=True,
                check=False,
            )
            if proc.returncode != 0:
                details = (proc.stderr or proc.stdout or "").strip()
                raise RuntimeError(f"ReCo input preprocessing failed: {details}")

            raw_edited_video = self.generator.generate(
                video_path=str(preprocessed_video),
                prompt=plan.edit_prompt,
                reference_image=plan.reference_image,
                first_frame_image=plan.first_frame_image,
                output_dir=str(output_dir),
            )

        final_path = (output_dir / f"replace_video_{uuid.uuid4().hex}.mp4").resolve()

        command = [
            ffmpeg,
            "-y",
            "-i",
            raw_edited_video,
            "-vf",
            "crop=iw/4:ih/2:iw/4:ih/2",
            "-an",
            str(final_path),
        ]
        proc = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            details = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"ReCo target-half extraction failed: {details}")

        final_video = str(final_path)

        if not Path(final_video).is_file():
            raise RuntimeError(f"ReplaceVideoRunner output does not exist: {final_video}")

        ctx.output_video = str(Path(final_video).resolve())
        return ctx

    @staticmethod
    def _resolve_output_dir(plan: ReplaceVideoPlan) -> Path:
        if plan.output_dir:
            return Path(plan.output_dir).expanduser().resolve()

        env_dir = os.environ.get("RECOREF_OUTPUT_DIR")
        if env_dir:
            return Path(env_dir).expanduser().resolve()

        return (Path(__file__).resolve().parents[1] / "outputs" / "reco_ref").resolve()
