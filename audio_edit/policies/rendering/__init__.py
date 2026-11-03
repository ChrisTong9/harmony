"""Choose a bounded material renderer without regenerating failed candidates.

Continuous pitch, isolated-instrument notes and retained-source notes are tried
in that order. Only an inapplicable policy advances to the next choice; missing
resources or a failed first render retain the existing generation route.
"""

from contextlib import contextmanager

from . import continuous_pitch, independent_piano, music_stem_timbre, music_timbre
from ..core import PolicySpec, record_policy

__all__ = ["prepare_material", "material_renderer"]

POLICIES = (
    PolicySpec(
        "rendering.continuous_violin",
        "rendering",
        "A whole-file violin substitution has validated continuous-pitch preservation intent and a structural audio reference.",
        "The existing first removed component and its matching duration; no additional source separation.",
        "Render one continuous pYIN pitch contour with a violin SoundFont voice.",
        "One pitch analysis/render; a failed or unconfigured renderer does not advance to another renderer.",
    ),
    PolicySpec(
        "rendering.isolated_instrument",
        "rendering",
        "A confirmed sole instrument is replaced structurally with a supported discrete-note target: piano, harp, violin, or wordless humming.",
        "The existing sole-target observation and original audio, with one shared cover window.",
        "Transcribe the source notes and render the requested supported SoundFont program.",
        "One transcription/render; no new source separation or diffusion generation.",
    ),
    PolicySpec(
        "rendering.removed_stem_timbre",
        "rendering",
        "Validated replacement semantics select piano, harp, violin, guitar, or oboe without continuous-pitch requirements, with an eligible backing decision and structural reference.",
        "The existing first removed component on the original sample grid.",
        "Render its transcribed notes with the requested supported SoundFont program.",
        "One transcription/render on the first stem; missing resources or failed rendering retain generation.",
    ),
    PolicySpec(
        "rendering.first_generated_piano",
        "rendering",
        "Validated semantics select piano without continuous-pitch requirements for one whole-file independent replacement or a structured all-instruments replacement.",
        "The primary generator's first music asset, never a replacement sample chosen after listening.",
        "Render the first composition with an acoustic-piano SoundFont voice.",
        "One existing generator call and one transcription/render; second generation is rejected and the generator is restored.",
    ),
)


def _record(ctx, index, result, subject):
    record_policy(
        ctx,
        POLICIES[index],
        result.get("status", "unknown"),
        subject=str(subject) if subject else None,
        audit_key="replacement_music_render",
    )


def prepare_material(ctx, plan, add_ctx, residual_observation):
    """Select material before the add runner, preserving the original priority."""
    reference = add_ctx.params.get("replacement_reference_asset")
    result = continuous_pitch.render_continuous_replacement(ctx, plan, reference)
    _record(add_ctx, 0, result, reference)
    if result.get("status") == "not_applicable":
        result = music_timbre.render_replacement(ctx.input_audio, plan, residual_observation)
        _record(add_ctx, 1, result, ctx.input_audio)
    if result.get("status") == "not_applicable":
        result = music_stem_timbre.render_stem_replacement(ctx, plan, reference)
        _record(add_ctx, 2, result, reference)
    add_ctx.params["replacement_music_render"] = result
    return result


@contextmanager
def material_renderer(runner, ctx, plan):
    """Wrap only the first primary generation for the explicit piano policy."""
    program = independent_piano.requested_program(ctx, plan)
    record_policy(
        ctx, POLICIES[3], "not_applicable" if program is None else "selected", program=program
    )
    if program is None:
        yield
        return
    original_generate = runner.generator_audio.generate
    used = False

    def first(**kwargs):
        nonlocal used
        if used:
            raise RuntimeError("Independent piano rendering expects one first music asset")
        used = True
        asset = original_generate(**kwargs)
        result = independent_piano.render_first(asset, program)
        ctx.params["replacement_independent_piano"] = result
        record_policy(
            ctx,
            POLICIES[3],
            result.get("status", "unknown"),
            subject=str(asset),
            audit_key="replacement_independent_piano",
        )
        return result["output_audio"] if result.get("status") == "rendered" else asset

    runner.generator_audio.generate = first
    try:
        yield
    finally:
        runner.generator_audio.generate = original_generate
