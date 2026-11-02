"""Compile an explicit first non-overlap statement for an already bound event.

This is a separate, narrow REMOVE execution policy. It neither changes the
model's source_assessments nor upgrades approximate timing to measured timing.
"""
from copy import deepcopy
from dataclasses import asdict
import re

from audio_edit.analyzers.removal_focus import file_sha256
from audio_edit.plans.remove_sound_plan import shared_removal_windows


def decide(ctx, plan):
    no = dict(eligible=False, reason="No bound explicit local non-overlap statement")
    dialogue = ctx.params.get("edit_listening_dialogue") or {}
    if (dialogue.get("operation") != "remove_sound"
            or dialogue.get("original_instruction") != ctx.user_prompt
            or dialogue.get("status") != "ready"):
        return no
    temporal = plan.temporal
    if (plan.mode != "span" or plan.result_mode != "residual"
            or plan.use_video_prompt or plan.temporal_schema_error
            or plan.temporal_status not in ("resolved", "approximate")
            or not temporal or temporal.kind != "event"
            or temporal.reference_source != "audio"
            or temporal.relation or temporal.offset_seconds
            or temporal.count not in (None, 1)
            or temporal.end_reference_event
            or len(plan.placements) != 1):
        return no
    analysis = (ctx.params.get("removal_perception") or {}).get("analysis") or {}
    contract = analysis.get("edit_listening") or {}
    if contract.get("status") != "ready" or contract.get("missing_evidence"):
        return no
    duration = analysis.get("media_duration")
    try:
        windows = shared_removal_windows(plan, duration)
    except (ValueError, TypeError):
        return no
    if len(windows) != 1 or windows[0] == (0., duration):
        return no
    start, end = windows[0]
    placement = plan.placements[0]
    if (placement.start_time, placement.end_time) != (start, end):
        return no
    refs = temporal.reference_event
    refs = refs if isinstance(refs, list) else [refs]
    if refs != [placement.event_id] or not placement.event_id:
        return no
    first_turn = next((t for t in dialogue.get("turns", [])
                       if t.get("action") == "qwen" and t.get("audio_source") == "original"), None)
    if not first_turn:
        return no
    first = first_turn.get("analysis") or {}
    if first.get("evidence_source") != "qwen_observation":
        return no
    digest = file_sha256(ctx.input_audio)
    if any((a.get("audio_evidence") or {}).get("sha256") != digest
           or a.get("media_duration") != duration for a in (first, analysis)):
        return no
    timings = [q for q in contract.get("requirements", [])
               if q.get("focus") == "timing" and q.get("required_fields") == ["interval"]]
    identities = [q for q in contract.get("requirements", [])
                  if q.get("focus") == "target" and q.get("required_fields") == ["identity"]]
    if len(timings) != 1 or len(identities) != 1:
        return no
    timing_id, identity_id = timings[0]["id"], identities[0]["id"]
    events = [e for e in first.get("events", [])
              if timing_id in e.get("question_ids", [])]
    if len(events) != 1:
        return no
    event = events[0]
    if (event.get("event_id") != placement.event_id
            or identity_id not in event.get("question_ids", [])
            or event.get("timing_source") != "qwen_observation"
            or (event.get("start_time"), event.get("end_time")) != (start, end)
            or event.get("partial_start") or event.get("partial_end")):
        return no
    matching = [e for e in analysis.get("events", [])
                if e.get("event_id") == placement.event_id]
    if len(matching) != 1:
        return no
    current = matching[0]
    if any(current.get(k) != event.get(k) for k in
           ("start_time", "end_time", "event", "question_ids", "timing_source")):
        return no
    answers = [a for a in first.get("answers", [])
               if a.get("question_id") == timing_id and a.get("status") == "answered"]
    if len(answers) != 1:
        return no
    answer = answers[0]["answer"]


    match = re.search(r"\bno other sounds overlapping it\b", answer, re.I)
    if not match or re.search(r"\b(maybe|possibly|uncertain|unclear|seems|might|except)\b", answer, re.I):
        return no
    if not any(a == answers[0] for a in analysis.get("answers", [])):
        return no


    saved = [t.get("analysis") or {} for t in dialogue.get("turns", [])] + [analysis]
    for observed in saved:
        for a in observed.get("answers", []):
            if a.get("question_id") == timing_id and a != answers[0]:
                return dict(no, reason="Later saved target answer differs from first non-overlap evidence")
        if observed.get("uncertainties") or observed.get("evidence_needs"):
            return no
        for e in observed.get("events", []):
            if e.get("event_id") == placement.event_id:
                if any(e.get(k) != event.get(k) for k in ("start_time", "end_time", "event", "question_ids")):
                    return no
                if any(a.get("status") == "conflict" for a in e.get("boundary_activity_audit", [])):
                    return no
            elif (type(e.get("start_time")) in (int, float)
                  and type(e.get("end_time")) in (int, float)
                  and max(start, e["start_time"]) < min(end, e["end_time"])):
                return dict(no, reason="Another saved event overlaps the proposed local mute")
        for assessment in observed.get("source_assessments", []):
            a, b = assessment.get("start_time"), assessment.get("end_time")
            if type(a) in (int, float) and type(b) in (int, float) and max(a, start) < min(b, end):
                if (assessment.get("target_present") is not True
                        or assessment.get("only_target") is not True
                        or assessment.get("other_sounds") != []):
                    return no
    participant = contract.get("participant_selection")
    if participant:
        from audio_edit.analyzers.participant_selection import participant_placements
        try:
            selected = participant_placements(analysis, duration)
        except (ValueError, RuntimeError):
            return no
        if [(p.start_time, p.end_time, p.event_id) for p in selected] != [
                (start, end, placement.event_id)]:
            return no
    return dict(eligible=True, reason="First original-audio answer explicitly excludes other sounds in this bound event",
        source="first_explicit_local_non_overlap", audio_sha256=digest,
        target_sound=plan.text_prompt, event_id=placement.event_id,
        identity_question_id=identity_id, timing_question_id=timing_id,
        quote=match.group(), complete_first_answer=answer,
        windows=[dict(start_time=start, end_time=end)],
        temporal_status=plan.temporal_status, approximate=placement.approximate,
        boundary_audit=deepcopy(current.get("boundary_activity_audit", [])),
        raw_model_assessments_modified=False, original_plan=asdict(plan),
        fade_seconds=0.005, limitation="Fallible original perception; approximate native event boundaries remain unchanged")


def render(ctx, decision):
    """Mute the accepted interval, using only five milliseconds inside its edges."""
    from pathlib import Path
    from tempfile import mkdtemp
    import json
    import numpy as np
    import soundfile as sf
    from audio_edit.utils.execution_outcome import open_decoded_audio
    if not decision.get("eligible") or decision["audio_sha256"] != file_sha256(ctx.input_audio):
        raise ValueError("Local mute requires matching bound first evidence")
    with open_decoded_audio(ctx.input_audio) as source:
        sr, channels = source.samplerate, source.channels
        audio = source.read(dtype="float32", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Invalid original audio")
    ranges = []
    for w in decision["windows"]:
        a, b = round(w["start_time"] * sr), round(w["end_time"] * sr)
        if not 0 <= a < b <= len(audio):
            raise ValueError("Existing event window is outside the original sample grid")
        n = min(round(decision["fade_seconds"] * sr), (b-a)//2)
        envelope = np.zeros(b-a, dtype=np.float32)
        if a and n:
            envelope[:n] = np.linspace(1., 0., n)
        if b < len(audio) and n:
            envelope[-n:] = np.linspace(0., 1., n)
        audio[a:b] *= envelope[:, None]
        ranges.append(dict(start_sample=a, end_sample=b, inner_fade_samples=n))
    folder = Path("audio_edit/outputs")
    folder.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix="local-single-source-", dir=folder)).resolve() / "residual.wav"
    sf.write(output, audio, sr, subtype="FLOAT")
    audit = dict(backend="local_single_source", sample_rate=sr, channels=channels,
        original_samples=len(audio), output_samples=len(audio), sample_ranges=ranges,
        normalization_applied=False, decision=decision)
    output.with_suffix(".audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    ctx.output_audio = str(output)
    ctx.params["local_single_source_execution"] = audit
    return ctx
