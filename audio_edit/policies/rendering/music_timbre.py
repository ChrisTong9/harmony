"""Optional note-preserving renderer for confirmed isolated instrumental replacement."""

from pathlib import Path
from tempfile import mkdtemp
import json
import os
import subprocess
from audio_edit.plans.replacement_semantics import get_replacement_semantics


PROGRAMS = {"piano": 0, "harp": 46, "violin": 40}


def target_program(plan, observation):
    if (
        not observation.get("eligible")
        or observation.get("decision", {}).get("target_is_single_instrument") is not True
    ):
        return None
    if plan.remove_plan.mode != "text" or plan.remove_plan.result_mode != "residual":
        return None
    edits = (plan.add_plan.intent or {}).get("edits", [])
    if (
        len(edits) != 1
        or edits[0].get("generator") != "audio"
        or edits[0].get("audio_reference") != "structure"
        or edits[0].get("placement", {}).get("kind") != "cover"
        or edits[0].get("timeline_op") != "overlay"
        or plan.add_plan.is_speech
    ):
        return None

    semantics = get_replacement_semantics(plan)
    if semantics is None or semantics["preserve_continuous_pitch"]:
        return None
    target = semantics["target"]
    if target["kind"] == "wordless_voice" and target["name"] == "humming":
        return 53
    if target["kind"] != "instrument":
        return None
    return PROGRAMS.get(target["name"])


def render_replacement(input_audio, plan, observation):
    program = target_program(plan, observation)
    if program is None:
        return dict(status="not_applicable")
    project = Path(__file__).resolve().parents[3]
    python = os.environ.get(
        "HARMONY_MUSIC_RENDER_PYTHON", str(project / ".venvs/music-render311/bin/python")
    )
    soundfont = os.environ.get(
        "HARMONY_MUSIC_SOUNDFONT", str(project / "externals/music-render/GeneralUser-GS.sf2")
    )
    if not python or not soundfont or not Path(python).is_file() or not Path(soundfont).is_file():
        return dict(status="tool_unconfigured")
    folder = Path("audio_edit/generated")
    folder.mkdir(parents=True, exist_ok=True)
    folder = Path(mkdtemp(prefix="music-timbre-", dir=folder)).resolve()
    request = dict(
        input_audio=str(Path(input_audio).resolve()),
        output_audio=str(folder / "music.wav"),
        program=program,
        soundfont=soundfont,
        report=str(folder / "render.json"),
    )
    request_path = folder / "request.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    try:
        with (folder / "render.log").open("x") as log:
            subprocess.run(
                [
                    python,
                    str(Path(__file__).resolve().parents[2] / "tools/run_music_timbre.py"),
                    str(request_path),
                ],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=300,
            )
        result = json.loads(Path(request["report"]).read_text())
        if result["status"] != "rendered" or not Path(result["output_audio"]).is_file():
            raise ValueError("Music renderer returned no valid audio")
        return result
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return dict(
            status="tool_failed_existing_generator_retained",
            error=f"{type(exc).__name__}: {exc}",
            log=str(folder / "render.log"),
        )
