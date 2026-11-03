"""Give an independent piano replacement a piano voice using its first composition."""

from pathlib import Path
from tempfile import mkdtemp
import json
import os
import subprocess
from audio_edit.plans.replacement_semantics import get_replacement_semantics


def requested_program(ctx, plan):
    """Select the existing piano renderer from validated replacement semantics."""
    if ctx.selected_script != "replace":
        return None
    removal = plan.remove_plan
    edits = (plan.add_plan.intent or {}).get("edits", [])
    if (
        removal.mode != "text"
        or removal.result_mode != "residual"
        or removal.temporal_status != "resolved"
        or removal.temporal_schema_error
        or (removal.temporal and removal.temporal.kind != "unspecified")
        or len(removal.placements) != 1
        or len(edits) != 1
        or plan.add_plan.is_speech
    ):
        return None
    edit = edits[0]
    shared = edit.get("shared_window", {})
    independent = (
        edit.get("audio_reference") is None
        and (edit.get("replacement_material") or {}).get("mode") == "independent_texture"
    )

    semantics = get_replacement_semantics(plan)
    if semantics is None or semantics["preserve_continuous_pitch"]:
        return None
    target = semantics["target"]
    if target["kind"] != "instrument" or target["name"] != "piano":
        return None
    structured_all = (
        semantics["source_scope"] == "all_instruments"
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
    return 0


def render_first(asset, program):
    project = Path(__file__).resolve().parents[3]
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
                [
                    python,
                    str(Path(__file__).resolve().parents[2] / "tools/run_music_timbre.py"),
                    str(path),
                ],
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
