"""Retain an explicitly requested continuous pitch performance in a violin render."""

import json
import os
from pathlib import Path
import re
import subprocess
from tempfile import mkdtemp


def render_request(ctx, plan, reference):
    removal = plan.remove_plan
    edits = (plan.add_plan.intent or {}).get("edits", [])
    if (
        len(edits) != 1
        or removal.mode != "text"
        or removal.result_mode != "residual"
        or removal.use_video_prompt
        or removal.temporal_status != "resolved"
        or removal.temporal_schema_error
        or plan.add_plan.is_speech
        or plan.add_plan.is_on_screen
        or (removal.temporal is not None and removal.temporal.kind != "unspecified")
    ):
        return None
    edit = edits[0]
    window = edit.get("shared_window", {})
    if (
        edit.get("generator") != "audio"
        or edit.get("audio_reference") != "structure"
        or edit.get("timeline_op") != "overlay"
        or edit.get("placement", {}).get("kind") != "cover"
        or window.get("start_time") != 0
        or len(removal.placements) != 1
        or any(
            window.get(k) is not None
            for k in ("reference_event", "event_id", "occurrence_index", "reference_source")
        )
    ):
        return None
    p = removal.placements[0]
    duration = window.get("end_time")
    if (
        p.start_time != 0
        or p.end_time != duration
        or not isinstance(duration, (int, float))
        or duration <= 0
    ):
        return None
    text = ctx.user_prompt.lower()

    if (
        not re.search(r"\b(?:preserv\w*|retain\w*|keep\w*)\b", text)
        or not re.search(r"\b(?:slides?|glissand\w*|portament\w*)\b", text)
        or not re.search(r"\bvibrato\b", text)
        or not re.fullmatch(
            r"(?:a |an |solo |acoustic )*violin(?: (?:tone|timbre|performance|playing))?",
            str(plan.target_sound).strip().lower(),
        )
        or not re.search(r"\b(?:to|with|into) (?:a |an |solo |acoustic )*violin\b", text)
    ):
        return None
    if not reference or not Path(reference).is_file():
        return None
    import soundfile as sf

    info = sf.info(reference)
    if abs(info.duration - duration) > 1e-4:
        return None
    return dict(
        input_audio=str(Path(reference).resolve()),
        duration=duration,
        sample_rate=44100,
        channels=2,
        frames=int(duration * 44100),
        program=40,
    )


def render_continuous_replacement(ctx, plan, reference):
    request = render_request(ctx, plan, reference)
    if request is None:
        return dict(status="not_applicable")
    project = Path(__file__).resolve().parents[2]
    python = os.environ.get(
        "HARMONY_MUSIC_RENDER_PYTHON", str(project / ".venvs/music-render311/bin/python")
    )
    soundfont = os.environ.get(
        "HARMONY_MUSIC_SOUNDFONT", str(project / "externals/music-render/GeneralUser-GS.sf2")
    )
    if not Path(python).is_file() or not Path(soundfont).is_file():
        return dict(status="tool_unconfigured")
    from audio_edit.analyzers.removal_focus import file_sha256

    root = Path("audio_edit/generated")
    root.mkdir(parents=True, exist_ok=True)
    folder = Path(mkdtemp(prefix="continuous-pitch-", dir=root)).resolve()
    request.update(
        input_sha256=file_sha256(reference),
        soundfont=soundfont,
        output_audio=str(folder / "music.wav"),
        report=str(folder / "render.json"),
    )
    path = folder / "request.json"
    path.write_text(json.dumps(request, indent=2) + "\n")
    try:
        with (folder / "render.log").open("x") as log:
            subprocess.run(
                [python, str(Path(__file__).with_name("run_continuous_pitch.py")), str(path)],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=300,
            )
        result = json.loads(Path(request["report"]).read_text())
        if result["status"] != "rendered" or result["input_sha256"] != request["input_sha256"]:
            raise ValueError("Invalid continuous-pitch render")
        return result
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return dict(
            status="tool_failed_existing_generator_retained",
            error=f"{type(exc).__name__}: {exc}",
            log=str(folder / "render.log"),
        )
