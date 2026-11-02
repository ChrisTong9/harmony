"""Remember unresolved speech timing until independent evidence can resolve it.

This applies to bounded gaps between speech, observed contradictions, and
requested timing checks. A confident timestamp does not supply missing evidence.
"""

from copy import deepcopy

from audio_edit.analyzers.listening_evidence import check_evidence, check_placement_windows
from audio_edit.analyzers.speech_references import speech_reference_gaps


def update_speech_dependencies(previous, questions, analysis, *, windows, turns, duration):
    pending = deepcopy(previous)
    speech = {
        q["id"]: q
        for q in questions
        if q.get("reference_kind") == "speech"
        and q.get("focus") == "timing"
        and set(q["required_fields"]) & {"interval", "onset", "sound_end", "utterance_end"}
    }
    if not speech:
        return pending
    missing = check_evidence(questions, analysis, duration) if analysis else {}
    paired = check_placement_windows(windows, questions, analysis)
    reasons = {
        key: "The current speech observation lacks required timing evidence."
        for key in speech
        if key in missing
    }
    for key, q in speech.items():
        if q.get("occurrence") != "each" or "interval" not in q["required_fields"]:
            continue
        spans = [e for e in (analysis or {}).get("events", []) if key in e.get("question_ids", [])]
        if (
            len(spans) == 1
            and spans[0].get("start_time") == 0
            and spans[0].get("end_time") == duration
        ):
            reasons[key] = (
                "An each-speech-segment condition was collapsed to the entire recording. "
                "Use lexical evidence to verify coverage and preserve actual pauses."
            )
    for window in windows:
        ids = (window["start_question_id"], window["end_question_id"])
        if (
            all(key in speech for key in ids)
            and set(speech[ids[0]]["required_fields"]) & {"sound_end", "utterance_end"}
            and "onset" in speech[ids[1]]["required_fields"]
        ):
            reasons.update(
                {
                    key: "An insertion between spoken turns needs lexical and acoustic support for both flanking edges; confident free timestamps are insufficient."
                    for key in ids
                }
            )
        elif any(key in paired or key in reasons for key in ids):
            reasons.update(
                {
                    key: "The flanking speech boundaries do not establish a supported positive window."
                    for key in ids
                    if key in speech
                }
            )

    for turn in turns:
        if turn.get("action") not in ("activity_regions", "boundary_activity"):
            continue
        checked = {q["id"] for q in turn.get("questions", [])} & speech.keys()
        for window in windows:
            ids = {window["start_question_id"], window["end_question_id"]}
            if checked & ids:
                checked |= ids & speech.keys()
        for key in checked:
            reasons[key] = (
                "A requested waveform check cannot establish speech content or completed utterance identity."
            )
    for key, reason in reasons.items():
        pending.setdefault(key, dict(reason=reason, first_observed_turn=len(turns)))
    return pending


def speech_dependency_gaps(dependencies, questions, analysis):
    needed = [q for q in questions if q["id"] in dependencies]
    gaps = speech_reference_gaps(needed, analysis)
    for q in needed:
        if not q.get("speech_references"):
            gaps.setdefault(q["id"], []).append(
                "The bounded speech window or prior timing conflict/check still requires lexical evidence. "
                "Acquire speech_transcription, then interpret current word/gap IDs from original audio; "
                "new free timestamps or a quiet subinterval cannot close this dependency."
            )
    return gaps


def speech_prerequisites(dependencies, questions, turns):
    """Capabilities needed before spending the last observation, in either order."""
    if not dependencies:
        return []
    succeeded = {
        turn["action"]
        for turn in turns
        if not turn.get("error")
        and isinstance(turn.get("tool_evidence"), dict)
        and turn["tool_evidence"].get("tool") == turn["action"]
    }
    needs = []
    if "speech_transcription" not in succeeded:
        needs.append("speech_transcription")

    if (
        any(
            q["id"] in dependencies and set(q["required_fields"]) & {"sound_end", "utterance_end"}
            for q in questions
        )
        and "activity_regions" not in succeeded
    ):
        needs.append("activity_regions")
    return needs
