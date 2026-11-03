"""Separate a whole-source window from a single observed event, before rendering."""

from copy import deepcopy
import json
import math


COVERAGE_PROMPT = """Check the content unit of one proposed REPLACE edit.
The supplied window is the entire unselected source scope, not evidence that
exactly one old event occurred. The source separator handles the named class.
Decide from original_instruction, not from the planner's proposed event duration.
Use source_window when the request replaces that sound class throughout its
scope. A singular source label or "a cat meow" naming the new sound does not by
itself request just one short event at the beginning. The new sound sequence
should cover the source window, with natural activity and pauses.
Use retain_user_limited if the user explicitly limits the NEW content by a count,
single occurrence, duration, or placement. Quote the exact limiting words from
original_instruction; keep the existing proposal without broadening it. This
decision does not certify an otherwise incorrect count or placement.
Use unresolved if the request's content scope cannot be decided. Never invent
audible events, counts, onsets, source identity, or new sound descriptions.
Return only JSON with reason, coverage=source_window|retain_user_limited|unresolved,
and instruction_quote. source_window and unresolved use a null quote. No other
plan fields may change; existing removal scope and original timing stay fixed.
"""


def coverage_candidate(data, payload):
    """Narrow gate; ordinary ADD, measured events and fixed durations stay intact."""
    windows = payload.get("shared_replacement_windows")
    edits = data.get("edits")
    if (
        payload.get("source_type") != "audio"
        or data.get("is_on_screen") is not False
        or not isinstance(windows, list)
        or len(windows) != 1
        or not isinstance(edits, list)
        or len(edits) != 1
    ):
        return None
    window, edit = windows[0], edits[0]
    if not isinstance(window, dict) or not isinstance(edit, dict):
        return None
    if any(
        window.get(k) is not None
        for k in ("reference_event", "event_id", "reference_source", "occurrence_index")
    ):
        return None
    start, end, duration = (
        window.get("start_time"),
        window.get("end_time"),
        payload.get("media_duration"),
    )
    if (
        not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in (start, end, duration)
        )
        or start != 0
        or duration <= 0
        or abs(end - duration) > 1e-6
    ):
        return None
    listening = (payload.get("audio_analysis") or {}).get("edit_listening") or {}
    if listening.get("status") != "not_required" or listening.get("requirements"):
        return None
    material, generation = edit.get("replacement_material") or {}, edit.get("generation") or {}
    if (
        type(edit.get("shared_window_index")) is not int
        or edit["shared_window_index"] != 0
        or edit.get("kind") != "single"
        or edit.get("generator") != "audio"
        or edit.get("audio_reference") is not None
        or edit.get("repeat")
        or material.get("mode") != "independent_event"
        or generation.get("source") != "planner_choice"
    ):
        return None
    return dict(
        original_instruction=payload.get("user_prompt"),
        source_window=deepcopy(window),
        new_sound=material.get("target"),
        proposed_generation=deepcopy(generation),
        proposed_kind="single",
        scope_fact="Unselected whole-source window; no event inventory is asserted.",
    )


def complete_replacement_coverage(ctx, data, payload, llm):
    request = coverage_candidate(data, payload)
    if request is None:
        return data
    audit = dict(
        stage="complete_replacement_coverage",
        request=deepcopy(request),
        status="pending",
        first_decision=True,
    )
    ctx.params.setdefault("replacement_coverage_checks", []).append(audit)
    ctx.params.setdefault("add_planning_responses", []).append(audit)
    properties = dict(
        reason=dict(type="string"),
        coverage=dict(type="string", enum=["source_window", "retain_user_limited", "unresolved"]),
        instruction_quote=dict(type=["string", "null"]),
    )
    options = {}
    if getattr(llm, "backend", None) == "openai":
        options["response_format"] = dict(
            type="json_schema",
            json_schema=dict(
                name="replacement_content_coverage",
                strict=True,
                schema=dict(
                    type="object",
                    additionalProperties=False,
                    properties=properties,
                    required=list(properties),
                ),
            ),
        )
    messages = [
        dict(role="system", content=COVERAGE_PROMPT),
        dict(role="user", content=json.dumps(request, ensure_ascii=False)),
    ]
    audit["messages"] = deepcopy(messages)

    response = llm.client.chat.completions.create(model=llm.model, messages=messages, **options)
    raw = response.choices[0].message.content
    audit["raw_response"] = raw
    from audio_edit.analyzers.qwen_audio_analyzer import extract_json_object

    answer = extract_json_object(raw)
    audit["response"] = deepcopy(answer)
    if (
        not isinstance(answer, dict)
        or set(answer) != set(properties)
        or not isinstance(answer.get("reason"), str)
    ):
        audit["status"] = "invalid_decision_original_plan_retained"
        return data
    choice, quote = answer.get("coverage"), answer.get("instruction_quote")
    if choice == "retain_user_limited":
        if (
            not isinstance(quote, str)
            or not quote.strip()
            or quote not in request["original_instruction"]
        ):
            audit["status"] = "unsupported_quote_original_plan_retained"
            return data
        audit["status"] = "explicit_user_limit_retained"
        return data
    if choice != "source_window" or quote is not None:
        audit["status"] = "unresolved_original_plan_retained"
        return data
    result = deepcopy(data)
    edit = result["edits"][0]
    edit["kind"] = "cover"
    edit["replacement_material"]["mode"] = "independent_texture"
    edit["generation"] = dict(
        target_seconds=request["source_window"]["end_time"],
        source="planner_choice",
        adjustable=False,
    )
    audit["status"] = "whole_source_coverage"
    audit["changed_fields"] = ["kind", "replacement_material.mode", "generation"]
    return result
