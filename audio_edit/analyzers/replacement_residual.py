"""Clear a confirmed complete target residual without changing generator inputs."""

from copy import deepcopy
from pathlib import Path
from tempfile import mkdtemp
import re

from audio_edit.analyzers.removal_focus import (
    file_sha256,
    observation_context,
    sole_source_decision,
    render_silence,
)
from audio_edit.plans.remove_sound_plan import shared_removal_windows


def bind_instrument_category(evidence, input_audio, duration, windows, requested):
    """Reuse a sole-instrument fact across generic instrument-category labels.

    This derives a separate binding receipt; the first observation is retained
    verbatim. Named sources, partial selections and mixed sources never qualify.
    """
    if not isinstance(evidence, dict) or evidence.get("target_sound") == requested:
        return evidence

    def category(label):
        return (
            isinstance(label, str)
            and re.fullmatch(
                r"(?:all |non[- ]vocal )?(?:melodic |musical )?(?:instruments|instrumental sounds|instrumentation)",
                label.strip().lower(),
            )
            is not None
        )

    previous = evidence.get("target_sound")

    generic_music = (
        isinstance(requested, str)
        and re.fullmatch(r"(?:original )?music(?: timbre)?", requested.strip().lower()) is not None
    )
    if not ((category(previous) and category(requested)) or generic_music):
        return evidence
    original = sole_source_decision(
        evidence, input_audio, duration, windows, previous, source="qwen_directed_removal"
    )
    if not original.get("eligible") or original.get("target_is_single_instrument") is not True:
        return evidence
    bound = deepcopy(evidence)
    bound["target_sound"] = requested
    bound["target_binding"] = dict(
        rule=(
            "sole_observed_instrument_in_generic_music"
            if generic_music
            else "sole_observed_instrument_in_generic_instrument_category"
        ),
        original_target=previous,
        requested_target=requested,
        original_evidence=deepcopy(evidence),
    )
    return bound


def inspect_sole_instrument(ctx, plan, duration, *, analyzer=None):
    removal = plan.remove_plan
    audit = dict(status="not_applicable", eligible=False, new_observations=0)
    if (
        removal.mode != "text"
        or removal.result_mode != "residual"
        or removal.use_video_prompt
        or removal.temporal_schema_error
        or removal.temporal_status != "resolved"
        or (removal.temporal is not None and removal.temporal.kind != "unspecified")
    ):
        return audit
    edits = (plan.add_plan.intent or {}).get("edits", [])
    if (
        len(edits) != 1
        or edits[0].get("generator") != "audio"
        or edits[0].get("audio_reference") != "structure"
        or edits[0].get("timeline_op") != "overlay"
        or edits[0].get("placement", {}).get("kind") != "cover"
    ):
        return audit
    windows = shared_removal_windows(removal, duration)
    if len(windows) != 1 or windows[0][0] != 0.0 or abs(windows[0][1] - duration) > 1e-6:
        return audit
    shared = edits[0].get("shared_window", {})
    if (shared.get("start_time"), shared.get("end_time")) != windows[0]:
        return dict(audit, status="shared_scope_mismatch")
    evidence = bind_instrument_category(
        removal.removal_evidence, ctx.input_audio, duration, windows, removal.text_prompt
    )
    valid_existing = (
        isinstance(evidence, dict)
        and evidence.get("source") == "qwen_directed_removal"
        and evidence.get("target_sound") == removal.text_prompt
        and evidence.get("audio", {}).get("sha256") == file_sha256(ctx.input_audio)
        and evidence.get("windows")
    )
    if valid_existing:
        status = "existing_observation_retained"
    else:
        if analyzer is None:
            from audio_edit.analyzers.qwen_audio_analyzer import QwenAudioAnalyzer

            analyzer = QwenAudioAnalyzer()
        context = observation_context(removal, duration)
        try:
            analysis = analyzer.analyze_removal(ctx.input_audio, ctx.user_prompt, context)
            evidence = analysis.get("removal_evidence")
        except Exception as exc:
            return dict(
                audit,
                status="optional_observation_failed_existing_path_retained",
                new_observations=1,
                error=f"{type(exc).__name__}: {exc}",
            )
        audit.update(new_observations=1, context=context, analysis=deepcopy(analysis))
        status = "first_observation"
    evidence = bind_instrument_category(
        evidence, ctx.input_audio, duration, windows, removal.text_prompt
    )
    decision = sole_source_decision(
        evidence,
        ctx.input_audio,
        duration,
        windows,
        removal.text_prompt,
        source="qwen_directed_removal",
    )

    return dict(audit, status=status, decision=decision, eligible=decision["eligible"])


def correct_residual(input_audio, cleaned_audio, observation):
    if not observation.get("eligible"):
        return cleaned_audio, None
    root = Path("audio_edit/generated")
    root.mkdir(parents=True, exist_ok=True)
    destination = Path(mkdtemp(prefix="sole-residual-", dir=root)) / "residual.wav"
    result = render_silence(input_audio, destination, observation["decision"])
    return result["output_audio"], dict(
        status="confirmed_complete_target_cleared",
        previous_residual=str(cleaned_audio),
        corrected_residual=result["output_audio"],
        existing_generator_reference_retained=True,
        generator_noise_unchanged=True,
        execution=result["execution"],
    )
