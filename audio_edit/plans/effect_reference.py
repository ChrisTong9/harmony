"""Check an inferred structure reference on a whole-scope single effect."""

from copy import deepcopy
import json
import re

PROMPT = """Check the content dependency of this proposed replacement asset from
the ORIGINAL instruction. The supplied full-file source window does not prove
that one event occupies it. The proposal's own reason is not a user requirement.
Choose independent_sound for an ordinary non-musical sound substitution whose
new identity does not need the old waveform. Natural pitch variation, duration,
an animal's vocal phrasing, and a general wish for old/new events to align do not
by themselves authorize copying the old sound's acoustic structure. Placement
and content coverage are separate decisions handled by the existing planner.
Choose musical_performance for an instrument, sung or hummed melody transformation;
its musical content must be retained even if the user did not repeat that constraint.
Choose explicit_internal_pattern when the original instruction requires the same
rhythm or another internal sequence not individually supplied as placement windows;
quote the exact requirement. Choose uncertain when those distinctions cannot be made.
For independent_sound, return a short positive target-only noun phrase (at most 24
words) naming the requested new sound and essential acoustic attributes. Do not
include old sound identity, replacement instructions, counts, times or mixing prose.
Return only JSON: dependency=independent_sound|musical_performance|explicit_internal_pattern|uncertain,
target (string only for independent_sound, otherwise null), instruction_quote
(string only for explicit_internal_pattern, otherwise null), reason. Do not alter
the original source, placement, duration, gain, repetition or other plan fields.
"""

STYLE_MUSIC_PROMPT = (
    PROMPT
    + """
This proposed cover is background music. Distinguish changing an existing
performance's instrument/timbre from replacing that background with a new mood,
style, or sustained drone. A new background composition or drone is independent_sound
unless the ORIGINAL instruction requires retaining the old musical performance,
melody, harmony, rhythm, or phrasing. Keeping the speech, vocals, or dialogue
perfectly timed is not a requirement to copy the backing music's structure.
Do not inherit preservation requirements from proposed_prompt or proposed_reason.
For a mood/style replacement, target names only the requested new musical content
and acoustic attributes. A literal instrument timbre transformation still uses
musical_performance; an explicit musical-pattern constraint must remain intact.
"""
)


def style_reference_candidate(data, payload):
    edits = data.get("edits", [])
    if len(edits) != 1 or not isinstance(edits[0], dict):
        return None
    edit = edits[0]
    original = str(payload.get("user_prompt", ""))

    if not re.search(r"\b(?:with|to)\b[^.!?;]*\bdrone\b", original, re.I):
        return None
    if (
        edit.get("kind") != "cover"
        or edit.get("generator") != "audio"
        or edit.get("audio_reference") != "structure"
        or (edit.get("replacement_material") or {}).get("mode") != "structured_audio"
        or edit.get("timeline_op", "overlay") != "overlay"
        or not re.search(
            r"\b(?:background (?:music|track)|accompaniment|backing (?:music|track))\b",
            original,
            re.I,
        )
        or re.search(
            r"\b(?:melody|melodic|rhythm|rhythmic|tempo|chord|harmony|harmonic|timbre)\b",
            original,
            re.I,
        )
    ):
        return None
    generation = edit.get("generation") or {}
    duration = payload.get("media_duration")

    if generation.get("target_seconds") != duration or generation.get("adjustable") is not False:
        return None
    from audio_edit.plans.replacement_coverage import coverage_candidate

    shape = deepcopy(data)
    candidate = shape["edits"][0]
    candidate.update(
        kind="single",
        audio_reference=None,
        replacement_material=dict(mode="independent_event", target="shape check only"),
    )
    candidate["generation"]["source"] = "planner_choice"
    whole = coverage_candidate(shape, payload)
    if whole is None:
        return None
    return dict(
        original_instruction=original,
        source_window=whole["source_window"],
        proposed_prompt=edit.get("audio_prompt"),
        proposed_reason=edit.get("reason"),
        proposed_generation=deepcopy(generation),
        style_music_replacement=True,
    )


def reference_candidate(data, payload):
    style = style_reference_candidate(data, payload)
    if style is not None:
        return style
    edits = data.get("edits", [])
    if len(edits) != 1 or not isinstance(edits[0], dict):
        return None
    edit = edits[0]
    if (
        edit.get("replacement_material", {}).get("mode") != "structured_audio"
        or edit.get("audio_reference") != "structure"
        or edit.get("kind") not in ("single", "cover")
    ):
        return None

    from audio_edit.plans.replacement_coverage import coverage_candidate

    shape = deepcopy(data)
    shape["edits"][0]["replacement_material"] = dict(
        mode="independent_event", target="shape check only"
    )
    shape["edits"][0]["audio_reference"] = None
    shape["edits"][0]["kind"] = "single"

    generation = edit.get("generation") or {}
    if (
        edit.get("kind") == "cover"
        and generation.get("target_seconds") == payload.get("media_duration")
        and generation.get("adjustable") is False
    ):
        shape["edits"][0]["generation"]["source"] = "planner_choice"
    whole = coverage_candidate(shape, payload)
    if whole is None:
        return None
    return dict(
        original_instruction=payload["user_prompt"],
        source_window=whole["source_window"],
        proposed_prompt=edit.get("audio_prompt"),
        proposed_reason=edit.get("reason"),
        proposed_generation=deepcopy(edit.get("generation")),
    )


def review_effect_reference(ctx, data, payload, llm):
    request = reference_candidate(data, payload)
    if request is None:
        return data
    audit = dict(
        stage="effect_reference_requirement",
        request=deepcopy(request),
        status="pending",
        first_decision=True,
    )
    ctx.params.setdefault("effect_reference_checks", []).append(audit)
    ctx.params.setdefault("add_planning_responses", []).append(audit)
    props = dict(
        dependency=dict(
            type="string",
            enum=[
                "independent_sound",
                "musical_performance",
                "explicit_internal_pattern",
                "uncertain",
            ],
        ),
        target=dict(type=["string", "null"]),
        instruction_quote=dict(type=["string", "null"]),
        reason=dict(type="string"),
    )
    kwargs = dict(
        model=llm.model,
        messages=[
            dict(
                role="system",
                content=STYLE_MUSIC_PROMPT if request.get("style_music_replacement") else PROMPT,
            ),
            dict(role="user", content=json.dumps(request, ensure_ascii=False)),
        ],
    )
    if getattr(llm, "backend", None) == "openai":
        kwargs["response_format"] = dict(
            type="json_schema",
            json_schema=dict(
                name="effect_reference_requirement",
                strict=True,
                schema=dict(
                    type="object",
                    additionalProperties=False,
                    properties=props,
                    required=list(props),
                ),
            ),
        )
    response = llm.client.chat.completions.create(**kwargs)
    from audio_edit.analyzers.qwen_audio_analyzer import extract_json_object

    raw = response.choices[0].message.content
    answer = extract_json_object(raw)
    audit.update(raw_response=raw, response=deepcopy(answer), status="original_material_retained")
    if (
        not isinstance(answer, dict)
        or set(answer) != set(props)
        or not isinstance(answer.get("reason"), str)
    ):
        return data
    target = answer.get("target")
    if (
        answer.get("dependency") != "independent_sound"
        or answer.get("instruction_quote") is not None
        or not isinstance(target, str)
        or not 0 < len(target.split()) <= 24
    ):
        return data
    result = deepcopy(data)
    edit = result["edits"][0]
    edit.update(
        replacement_material=dict(
            mode="independent_texture" if edit.get("kind") == "cover" else "independent_event",
            target=target.strip(),
        ),
        audio_reference=None,
        audio_prompt=target.strip(),
    )
    audit.update(
        status="independent_content_dependency",
        changed_fields=["replacement_material", "audio_reference", "audio_prompt"],
    )
    return result
