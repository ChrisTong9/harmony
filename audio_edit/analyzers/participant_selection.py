"""Preserve a requested participant selection without claiming speaker recognition."""

from copy import deepcopy
import re

from audio_edit.plans.errors import UnresolvedEditError
from audio_edit.plans.temporal import ResolvedPlacement
from audio_edit.plans.temporal_boundaries import finite, select_event


def routed_selection(params):
    """Expose only edit-selection hints, never execution history or credentials."""
    return {
        key: value
        for key in ("target_speaker", "source_sound", "target_sound", "occurrence")
        if isinstance((value := params.get(key)), (str, int)) and not isinstance(value, bool)
    }


def routed_participant(payload):
    """A literal source-participant extraction may constrain, never extend, the request."""
    if payload.get("operation") not in ("remove_sound", "replace", "volume_change"):
        return None
    reference = payload.get("routed_selection", {}).get("target_speaker")
    instruction = payload.get("original_instruction", "")
    if (
        isinstance(reference, str)
        and reference.strip()
        and reference.strip().casefold() in instruction.casefold()
    ):
        return reference.strip()
    return None


def participant_selection_schema(payload):
    reference = dict(type="string")
    if selected := routed_participant(payload):
        reference["enum"] = [selected]
    props = dict(
        reference_text=reference,
        instruction_quote=dict(type="string"),
        identity_question_id=dict(type="string", enum=[f"q{i}" for i in range(1, 7)]),
        timing_question_id=dict(type="string", enum=[f"q{i}" for i in range(1, 7)]),
    )
    shape = dict(type="object", additionalProperties=False, properties=props, required=list(props))
    return shape if selected else dict(anyOf=[dict(type="null"), shape])


def participant_from_brief(brief, payload):
    selected = brief.get("participant_selection")
    routed = routed_participant(payload)
    if selected is None:
        if routed:
            raise UnresolvedEditError(
                "A requested participant cannot skip identity and event-window grounding"
            )
        return None
    keys = {"reference_text", "instruction_quote", "identity_question_id", "timing_question_id"}
    if (
        not isinstance(selected, dict)
        or set(selected) != keys
        or any(not isinstance(selected[key], str) or not selected[key].strip() for key in keys)
    ):
        raise ValueError(
            "Participant selection needs its original reference and identity/timing question IDs"
        )
    reference, quote = selected["reference_text"], selected["instruction_quote"]
    instruction = payload["original_instruction"]
    if (
        payload.get("operation") == "add_sound"
        and quote in instruction
        and reference.casefold() not in quote.casefold()
    ):
        people = list(
            re.finditer(
                r"(?<!\w)" + re.escape(reference) + r"(?!\w)", instruction, flags=re.IGNORECASE
            )
        )
        quotes = list(re.finditer(re.escape(quote), instruction))
        if len(people) == len(quotes) == 1:
            person, utterance = people[0], quotes[0]
            quote = instruction[
                min(person.start(), utterance.start()) : max(person.end(), utterance.end())
            ]
    if quote not in payload["original_instruction"] or reference.casefold() not in quote.casefold():
        raise ValueError("Participant selection must quote the original requested participant")
    if routed and reference.casefold() != routed.casefold():
        raise UnresolvedEditError("The listening brief changed the routed participant selection")
    return dict(deepcopy(selected), instruction_quote=quote)


def bind_participant_questions(selection, questions, operation):
    if selection is None:
        return
    identity = next((q for q in questions if q["id"] == selection["identity_question_id"]), None)
    timing = next((q for q in questions if q["id"] == selection["timing_question_id"]), None)
    if (
        identity is None
        or identity.get("focus") != "target"
        or identity["required_fields"] != ["identity"]
    ):
        raise UnresolvedEditError(
            "Selected participant needs an audible identity question for its target events"
        )
    if (
        timing is None
        or timing.get("focus") != "timing"
        or (operation != "add_sound" and timing["required_fields"] != ["interval"])
    ):
        raise UnresolvedEditError(
            "Selected participant needs the target event windows, not only a voice description"
        )
    timing["participant_identity_question_id"] = identity["id"]
    identity["reference_kind"] = "participant"

    if timing.get("reference_kind") != "speech":
        timing["reference_kind"] = "participant"
    for q in (identity, timing):
        q["participant_reference"] = selection["reference_text"]
        q["answer_contract"] += (
            f" This selects participant {selection['reference_text']!r}, not an event number. "
            "Distinguish this person from other voices using audible evidence; a later speech segment "
            "does not establish a different speaker. The identity question identifies this person's "
            "requested target events, and the timing question locates THOSE SAME events. "
            "Link their shared observations through observation_ref when applicable. "
            "A sound such as a cough or laugh needs evidence that it belongs to this person; "
            "otherwise report uncertain. SAM text or anonymous activity cannot establish that attribution."
        )


def participant_placements(analysis, duration, *, literal_windows=None):
    """Bind selected event IDs to execution; participant names never supply times."""
    contract = (analysis or {}).get("edit_listening", {})
    selection = contract.get("participant_selection")
    if not selection:
        return None
    if contract.get("status") != "ready":
        raise UnresolvedEditError("Participant selection has no completed listening evidence")
    timing_id, identity_id = selection["timing_question_id"], selection["identity_question_id"]
    question = next((q for q in contract.get("requirements", []) if q["id"] == timing_id), None)
    answers = {a["question_id"]: a for a in analysis.get("answers", [])}
    if (
        question is None
        or question["required_fields"] != ["interval"]
        or any(answers.get(key, {}).get("status") != "answered" for key in (timing_id, identity_id))
    ):
        raise UnresolvedEditError(
            "Participant execution requires answered identity and interval dependencies"
        )
    events = {
        e["event_id"]: e
        for e in analysis.get("events", [])
        if timing_id in e.get("question_ids", [])
    }
    if not events or any(identity_id not in e.get("question_ids", []) for e in events.values()):
        raise UnresolvedEditError("Participant event windows lack the selected identity link")
    if question["occurrence"] == "each":
        selected = list(events.values())
    else:
        try:
            selected = [
                select_event(
                    dict(
                        event_id=list(events),
                        occurrence=None
                        if question["occurrence"] == "one"
                        else question["occurrence"],
                    ),
                    events,
                )[0]
            ]
        except (ValueError, TypeError, KeyError) as exc:
            raise UnresolvedEditError(
                "Participant event occurrence is unresolved: " + str(exc)
            ) from exc
    spans = []
    for event in selected:
        a, b = event.get("start_time"), event.get("end_time")
        if (
            not finite(a)
            or not finite(b)
            or not 0 <= a < b <= duration
            or event.get("resolution_status") == "unresolved"
            or event.get("partial_start")
            or event.get("partial_end")
        ):
            raise UnresolvedEditError("Participant event lacks a complete original-audio interval")
        windows = [(a, b)]
        if literal_windows is not None:
            if not literal_windows or any(
                not finite(x) or not finite(y) or not 0 <= x < y <= duration
                for x, y in literal_windows
            ):
                raise UnresolvedEditError("The explicit participant edit scope has invalid windows")
            windows = [(max(a, x), min(b, y)) for x, y in literal_windows if max(a, x) < min(b, y)]
        for start, end in windows:
            spans.append(
                ResolvedPlacement(
                    start,
                    end,
                    reference_event=event.get("event"),
                    reference_source="audio",
                    occurrence_index=event.get("occurrence_index"),
                    event_id=event["event_id"],
                    approximate=True,
                    resolution_status="approximate",
                    evidence=f"Participant selection linked through {identity_id}/{timing_id}; Qwen attribution, not speaker verification.",
                )
            )
    if not spans:
        raise UnresolvedEditError(
            "Selected participant events do not intersect the explicit edit scope"
        )
    return spans


def participant_binding_audit(analysis, before, placements):
    from dataclasses import asdict

    return dict(
        selection=deepcopy(analysis["edit_listening"]["participant_selection"]),
        before=deepcopy(before),
        placements=[asdict(p) for p in placements],
        scope_source="linked participant event IDs; intersected only with a separately selected literal scope",
        limitation="Qwen attribution is approximate; no diarization or speaker verification was run",
    )
