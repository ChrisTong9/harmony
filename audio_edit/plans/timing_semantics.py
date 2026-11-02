"""Understand requested time semantics BEFORE asking perception where they occur."""

import json
from copy import deepcopy
from audio_edit.analyzers.qwen_audio_analyzer import extract_json_object
from audio_edit.plans.temporal_boundaries import BOUNDARY_ROLES, finite


def preserve_timing_semantics(edits, semantics):
    """Bind exact reference wording to parsed meaning before compilation.

    Evidence and IDs can change, not the meaning of the same reference. No fuzzy
    matching or sound-name inference; absent legacy metadata stays compatible.
    """
    result, audits = deepcopy(edits), []
    references = semantics.get("references") if isinstance(semantics, dict) else None
    references = references if isinstance(references, list) else []

    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            if "event_id" not in value:
                for item in value.values():
                    visit(item)
                return
            matches = [
                r
                for r in references
                if isinstance(r, dict)
                and r.get("reference_text")
                and r.get("reference_text") == value.get("reference_text")
            ]
            if len(matches) != 1:
                audits.append(
                    dict(
                        status="compatibility",
                        anchor=deepcopy(value),
                        reason="No unique exact parsed reference; no semantic guess applied",
                    )
                )
                return
            reference = matches[0]
            geometry = {
                k: reference[k]
                for k in ("media_fraction", "media_edge")
                if reference.get(k) is not None
            }
            if reference.get("reference_kind") == "file" and len(geometry) == 1:
                fraction = geometry.get("media_fraction")
                if (
                    "media_fraction" in geometry and finite(fraction) and 0 <= fraction <= 1
                ) or geometry.get("media_edge") in ("start", "end"):
                    before = deepcopy(value)
                    for key in (
                        "event_id",
                        "seconds",
                        "media_fraction",
                        "media_edge",
                        "edge",
                        "boundary_role",
                        "occurrence",
                    ):
                        value.pop(key, None)
                    value.update(
                        geometry,
                        reference_kind="file",
                        offset_seconds=reference.get("offset_seconds")
                        if finite(reference.get("offset_seconds"))
                        else 0.0,
                    )
                    audits.append(
                        dict(
                            status="restored_parsed_file_geometry",
                            before=before,
                            after=deepcopy(value),
                            reason="A parsed proportional file position/edge is geometry, not a missing acoustic event",
                        )
                    )
                    return
            if reference.get("boundary_role") not in BOUNDARY_ROLES or reference.get(
                "relation"
            ) not in ("before", "at", "after"):
                return
            before = deepcopy(value)
            for key in ("boundary_role", "reference_kind", "relation", "occurrence"):
                if key in reference:
                    if (
                        key == "occurrence"
                        and reference[key] == "each"
                        and isinstance(value.get("event_id"), str)
                        and isinstance(value.get(key), int)
                    ):
                        continue
                    value[key] = reference[key]
            if finite(reference.get("offset_seconds")):
                value["offset_seconds"] = reference["offset_seconds"]
            value["edge"] = "start" if value["boundary_role"] == "onset" else "end"
            if value != before:
                audits.append(
                    dict(
                        status="restored_parsed_semantics",
                        before=before,
                        after=deepcopy(value),
                        reason="Preserve meaning parsed before perception; evidence may change, boundary meaning may not",
                    )
                )

    visit(result)
    return result, audits


def parse_timing_semantics(llm, instruction):
    system = """Extract temporal meaning only. You have NO audio observations, so
do NOT invent event IDs, observed timestamps or an executable edit position.
Return JSON {"references":[{"reference_text":"verbatim reference wording",
"reference_kind":"action|audible_event|utterance|end_region|file",
"occurrence":"first|last|each or one-based integer or null",
"relation":"before|at|after|during|between",
"boundary_role":"onset|action_end|sound_end|utterance_end or null",
"offset_seconds":number,"media_fraction":number or null,"media_edge":"start|end or null"}], "perception_query":"neutral questions about EXISTING
audio only", "reason":"semantic basis"}.
Keep literal seconds in an optional explicit_times field unchanged. An unspecified
ordinal is null, not last. Action completion, complete audible decay, utterance
completion, end-region and actual file end are different. Boundary definitions:
For a completed/perfective action predicate, after refers to its completion, not
to when it begins. Choose onset for an explicit beginning or initiation reference;
do not replace completion with onset merely because the action is brief.
onset = principal activity begins; action_end = the principal driving activity
ceases, possibly leaving audible resonance; sound_end = the whole audible event
INCLUDING its decay disappears; utterance_end = linguistic content finishes,
excluding subsequent room tone or reverberation. These roles concern the same
reference event's phases, not interchangeable event envelopes. Choose from the complete
instruction, not a sound-name lookup. Literal timing positions and durations are
different: explicit_times describes positions, never the requested new asset duration.
Offsets are signed timeline offsets, applied
after the boundary. No offset means zero. File end is not the end of its last sound.
Proportional positions within the recording are reference_kind=file with a numeric
media_fraction in [0,1]; they are NOT the onset of an existing audible event.
Use media_edge only for an actual recording boundary, and null geometry fields
for acoustic references. A vague ending region is reference_kind=end_region,
not automatically the exact file end. Do not ask perception to find a geometric
file position or an existing instance of the NEW sound. Distinguish the location
reference from the sound that will be placed there.
perception_query must ask only about the ORIGINAL reference events and the relevant
phases/occurrences, never mention the desired NEW sound, the generation request,
mixing instructions, or hypothetical output. If there are no existing event timing
references, ask only for the original acoustic scene. Keep references=[] for pure
literal-seconds timing. Preserve the user's wording, including absence of ordinals.
This is semantic parsing, not acoustic evidence or a guessed answer."""
    response = llm.client.chat.completions.create(
        model=llm.model,
        messages=[dict(role="system", content=system), dict(role="user", content=instruction)],
    )
    result = extract_json_object(response.choices[0].message.content or "")
    if not isinstance(result.get("references"), list) or not isinstance(
        result.get("perception_query"), str
    ):
        raise ValueError("Temporal interpretation needs references and a neutral perception query")
    for reference in result["references"]:
        if not isinstance(reference, dict) or not isinstance(reference.get("reference_text"), str):
            raise ValueError("Invalid temporal reference interpretation")
        if reference["reference_text"] not in instruction:
            reference["raw_instruction"] = instruction
            reference["wording_note"] = (
                "Parser paraphrased reference; raw instruction retained verbatim."
            )
    return result
