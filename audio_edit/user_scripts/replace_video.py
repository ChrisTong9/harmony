from __future__ import annotations

from pathlib import Path
from typing import Any

from audio_edit.models import AudioEditContext
from audio_edit.runners.replace_video_runner import ReplaceVideoRunner
from audio_edit.specialists.replace_video_specialist import ReplaceVideoSpecialist


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}


def _resolve_input_video(ctx: AudioEditContext) -> str:
    """
    Prefer the explicit video field.

    input_audio is accepted as a compatibility fallback because the current project
    historically stores the original media path there before video-aware fields were
    added to AudioEditContext.
    """
    candidates = [
        getattr(ctx, "input_video", None),
        getattr(ctx, "input_audio", None),
    ]

    for candidate in candidates:
        if not candidate:
            continue

        path = Path(candidate)
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        if not path.is_file():
            raise FileNotFoundError(f"Input video does not exist: {path}")

        return str(path.resolve())

    if getattr(ctx, "input_audio", None):
        suffix = Path(ctx.input_audio).suffix.lower()
        if suffix and suffix not in VIDEO_EXTENSIONS:
            raise ValueError(
                "replace_video only accepts video input "
                f"({', '.join(sorted(VIDEO_EXTENSIONS))}); got {suffix or 'unknown'}."
            )

    raise ValueError("replace_video requires ctx.input_video or a video path in ctx.input_audio.")


def run(
    ctx: AudioEditContext,
    params: dict[str, Any],
) -> AudioEditContext:
    input_video = _resolve_input_video(ctx)
    ctx.input_video = input_video

    specialist = ReplaceVideoSpecialist()
    plan = specialist.plan(ctx, params)

    runner = ReplaceVideoRunner()
    return runner.run(ctx, plan)
