"""Targeted removal observations and the narrowly scoped direct-silence decision."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from audio_edit.plans.remove_sound_plan import shared_removal_windows


REMOVAL_OBSERVATION_PROMPT = """
REMOVAL OBSERVATION (also applies to the OLD source of a replacement):
The target_sound below is the source being removed, not the future added sound.
Use the original instruction only as context. Listen for this target and the
requested timing references; never assume a requested sound is actually present.
Report the target's audible occurrences and requested reference boundaries in
"events". For nth/last/each, distinguish occurrences using the heard sequence.
Also listen for OTHER sounds throughout each assessment window, including quiet
speech, accompaniment, other instruments, ambience and noise that would remain.
A dominant instrument, one instrument class, one event label, or a failed source
separation does NOT establish that the target is the only sound. Silence between
notes and the target's own reverberation are compatible with a solo instrument.
Return no mixing advice and do not change the requested edit scope.
In the SAME JSON response, additionally return:
"removal_assessment": {"windows": [{"window_index": integer,
"target_present": true/false/null, "only_target": true/false/null,
"other_sounds": ["concrete other audible sources"] or null,
"target_is_single_instrument": true/false/null,
"evidence": "brief concrete description of what you actually hear"}]}.
Use the zero-based window_index supplied below. Assess the ENTIRE window, not
just the loudest part or one selected occurrence. only_target=true requires
hearing the target with no other audible source to preserve anywhere in that
window. An absent target is NOT a solo target. If uncertain or masked, use null;
other_sounds=[] means none heard with sufficient certainty, not unchecked.
target_is_single_instrument=true means the target is a solo musical instrument,
not vocals, sound effects, a band, or multiple instruments. Do not infer it just
from the user's desired replacement. Window scope is a question, not evidence.
"""


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def observation_context(plan, duration, analysis=None):
    try:
        windows = shared_removal_windows(plan, duration)
    except ValueError:
        windows = [(0.0, duration)]
    from dataclasses import asdict

    return dict(
        operation="remove",
        target_sound=plan.text_prompt,
        result_mode=plan.result_mode,
        references=[asdict(plan.temporal)] if plan.temporal else [],
        assessment_windows=[
            dict(window_index=i, start_time=a, end_time=b) for i, (a, b) in enumerate(windows)
        ],
        candidates=deepcopy((analysis or {}).get("events", [])),
    )


def attach_evidence(
    data,
    context,
    audio_evidence,
    *,
    assessment_key="removal_assessment",
    evidence_key="removal_evidence",
    source="qwen_directed_removal",
):
    """Bind observations to caller-owned scope/target and the actual input bytes."""
    assessment = data.get(assessment_key)
    rows = assessment.get("windows", []) if isinstance(assessment, dict) else []
    rows = rows if isinstance(rows, list) else []
    windows = []
    for window in context["assessment_windows"]:
        matches = [
            r
            for r in rows
            if isinstance(r, dict)
            and type(r.get("window_index")) is int
            and r["window_index"] == window["window_index"]
        ]
        if len(matches) == 1:
            windows.append(dict(window, observation=deepcopy(matches[0])))
    data[evidence_key] = dict(
        source=source,
        audio=deepcopy(audio_evidence),
        target_sound=context["target_sound"],
        windows=windows,
    )
    return data


def _confirmed(row):
    return (
        isinstance(row, dict)
        and row.get("target_present") is True
        and row.get("only_target") is True
        and row.get("other_sounds") == []
        and isinstance(row.get("evidence"), str)
        and bool(row["evidence"].strip())
    )


def _covered(start, end, windows):
    cursor = start
    for a, b, _ in sorted(windows, key=lambda w: (w[0], w[1])):
        if b <= cursor:
            continue
        if a > cursor + 1e-9:
            return False
        cursor = max(cursor, b)
        if cursor >= end - 1e-9:
            return True
    return False


def silence_decision(plan, input_audio, duration):
    """Model evidence never widens an edit or substitutes for source identity."""
    decision = dict(eligible=False, reason="No confirmed sole-source observation")
    if (
        plan.result_mode != "residual"
        or plan.use_video_prompt
        or plan.temporal_schema_error
        or plan.temporal_status not in ("not_requested", "resolved")
    ):
        return dict(decision, reason="Requires a resolved audio removal scope")
    try:
        scopes = shared_removal_windows(plan, duration)
    except ValueError as exc:
        return dict(decision, reason=str(exc))
    return sole_source_decision(
        plan.removal_evidence,
        input_audio,
        duration,
        scopes,
        plan.text_prompt,
        source="qwen_directed_removal",
    )


def sole_source_decision(evidence, input_audio, duration, scopes, target_sound, *, source):
    """Shared positive-source evidence check for direct deletion or direct gain."""
    decision = dict(eligible=False, reason="No confirmed sole-source observation")
    if (
        not isinstance(evidence, dict)
        or evidence.get("source") != source
        or evidence.get("target_sound") != target_sound
    ):
        return decision
    audio = evidence.get("audio")
    if (
        not isinstance(audio, dict)
        or not audio.get("sha256")
        or audio["sha256"] != file_sha256(input_audio)
    ):
        return dict(decision, reason="Observation does not match the current input audio")
    confirmed = []
    evidence_windows = evidence.get("windows")
    if not isinstance(evidence_windows, list):
        return decision
    for window in evidence_windows:
        if not isinstance(window, dict):
            continue
        a, b = window.get("start_time"), window.get("end_time")
        row = window.get("observation")
        if (
            all(type(v) in (int, float) and math.isfinite(v) for v in (a, b))
            and 0 <= a < b <= duration + 1e-9
            and _confirmed(row)
        ):
            confirmed.append((a, b, row))
    if not all(_covered(a, b, confirmed) for a, b in scopes):
        return decision
    instrument_windows = [w for w in confirmed if w[2].get("target_is_single_instrument") is True]
    return dict(
        eligible=True,
        reason="Directed listening confirms only the requested target throughout the edit scope",
        windows=[dict(start_time=a, end_time=b) for a, b in scopes],
        target_is_single_instrument=all(_covered(a, b, instrument_windows) for a, b in scopes),
        evidence=deepcopy(evidence),
    )


def render_silence(input_audio, output_path, decision):
    """Clear exactly the selected sample ranges, preserving the original grid."""
    import numpy as np
    import soundfile as sf
    from audio_edit.utils.execution_outcome import open_decoded_audio

    if decision.get("eligible") is not True:
        raise ValueError("Direct silence requires a confirmed decision")
    with open_decoded_audio(input_audio) as source:
        sr, channels = source.samplerate, source.channels
        audio = source.read(dtype="float32", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Cannot mute empty or nonfinite input audio")
    ranges = []
    for window in decision["windows"]:
        a, b = round(window["start_time"] * sr), round(window["end_time"] * sr)
        if not 0 <= a < b <= len(audio):
            raise ValueError("Direct silence scope is outside the original sample grid")
        audio[a:b] = 0.0
        ranges.append(dict(start_sample=a, end_sample=b))
    sf.write(output_path, audio, sr, subtype="FLOAT")
    audit = dict(
        code="direct_silence",
        backend="direct_silence",
        sample_rate=sr,
        channels=channels,
        original_samples=len(audio),
        output_samples=len(audio),
        sample_ranges=ranges,
        normalization_applied=False,
        target_is_single_instrument=decision["target_is_single_instrument"],
        reason=decision["reason"],
        evidence=decision["evidence"],
    )
    Path(str(output_path) + ".audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2)
    )
    return dict(
        output_audio=str(Path(output_path).resolve()),
        backend="direct_silence",
        warnings=[],
        execution=audit,
    )
