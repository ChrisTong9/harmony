"""Give an independent piano replacement a piano voice using its first composition."""

from contextlib import contextmanager
from pathlib import Path
from tempfile import mkdtemp
import json
import os
import re
import subprocess


def requested_program(ctx, plan):
    """Only explicit independent acoustic-piano music on an existing replacement."""
    if ctx.selected_script != "replace":
        return None
    removal = plan.remove_plan
    edits = (plan.add_plan.intent or {}).get("edits", [])
    if (
        removal.mode != "text"
        or removal.result_mode != "residual"
        or removal.use_video_prompt
        or removal.temporal_status != "resolved"
        or removal.temporal_schema_error
        or (removal.temporal and removal.temporal.kind != "unspecified")
        or len(removal.placements) != 1
        or len(edits) != 1
        or plan.add_plan.is_speech
        or plan.add_plan.is_on_screen
    ):
        return None
    edit = edits[0]
    shared = edit.get("shared_window", {})
    independent = (
        edit.get("audio_reference") is None
        and (edit.get("replacement_material") or {}).get("mode") == "independent_texture"
    )

    all_instruments = bool(
        re.match(
            r"^all (?:the )?(?:musical )?instruments\b",
            " ".join(str(plan.source_sound).lower().split()),
        )
    )
    structured_all = (
        all_instruments
        and edit.get("audio_reference") == "structure"
        and (edit.get("replacement_material") or {}).get("mode") == "structured_audio"
    )
    if (
        edit.get("generator") != "audio"
        or not (independent or structured_all)
        or edit.get("placement", {}).get("kind") != "cover"
        or edit.get("timeline_op") != "overlay"
        or shared.get("start_time") != 0
        or any(
            shared.get(k) is not None
            for k in ["reference_event", "reference_source", "event_id", "occurrence_index"]
        )
    ):
        return None
    placement = removal.placements[0]
    if placement.start_time != 0 or placement.end_time != shared.get("end_time"):
        return None
    from audio_edit.utils.media_util import get_media_duration

    if abs(placement.end_time - get_media_duration(ctx.input_audio)) > 1e-6:
        return None
    label = " ".join(str(plan.target_sound).lower().replace(",", " ").split())
    modifiers = r"(?:(?:soft|soothing|gentle|quiet|calm|peaceful|slow|solo|acoustic|grand|concert|natural) )*"
    if not re.fullmatch(
        modifiers + r"piano(?: (?:music|bed|background|track|performance|accompaniment))?", label
    ):
        return None
    return 0


def render_first(asset, program):
    project = Path(__file__).resolve().parents[2]
    python = os.environ.get(
        "HARMONY_MUSIC_RENDER_PYTHON", str(project / ".venvs/music-render311/bin/python")
    )
    font = os.environ.get(
        "HARMONY_MUSIC_SOUNDFONT", str(project / "externals/music-render/GeneralUser-GS.sf2")
    )
    if not Path(python).is_file() or not Path(font).is_file():
        return dict(status="tool_unconfigured_first_material_retained")
    root = Path("audio_edit/generated")
    root.mkdir(parents=True, exist_ok=True)
    folder = Path(mkdtemp(prefix="independent-piano-", dir=root)).resolve()
    request = dict(
        input_audio=str(Path(asset).resolve()),
        output_audio=str(folder / "music.wav"),
        program=program,
        soundfont=font,
        report=str(folder / "render.json"),
    )
    path = folder / "request.json"
    path.write_text(json.dumps(request, indent=2) + "\n")
    try:
        with (folder / "render.log").open("x") as log:
            subprocess.run(
                [python, str(Path(__file__).with_name("run_music_timbre.py")), str(path)],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=300,
            )
        result = json.loads(Path(request["report"]).read_text())
        if result["status"] != "rendered" or not Path(result["output_audio"]).is_file():
            raise ValueError("No valid first piano render")
        result.update(
            input_role="first_generated_replacement_music",
            new_diffusion_generations=0,
            new_source_separations=0,
            original_generator_asset=str(asset),
        )
        return result
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return dict(
            status="tool_failed_first_material_retained",
            error=f"{type(exc).__name__}: {exc}",
            log=str(folder / "render.log"),
        )


@contextmanager
def material_renderer(runner, ctx, plan):
    program = requested_program(ctx, plan)
    if program is None:
        yield
        return
    original_generate = runner.generator_offscreen.generate
    used = False

    def first(**kwargs):
        nonlocal used
        if used:
            raise RuntimeError("Independent piano rendering expects one first music asset")
        used = True
        asset = original_generate(**kwargs)
        result = render_first(asset, program)
        ctx.params["replacement_independent_piano"] = result
        return result["output_audio"] if result.get("status") == "rendered" else asset

    runner.generator_offscreen.generate = first
    try:
        yield
    finally:
        runner.generator_offscreen.generate = original_generate
