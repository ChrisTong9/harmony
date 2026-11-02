"""Keep identity observations from silently narrowing an unselected source edit."""

from copy import deepcopy
from dataclasses import asdict
import math

from audio_edit.plans.temporal import TemporalConstraint


def bind(ctx, plan, analysis, media_duration):
    contract = (analysis or {}).get("edit_listening", {})
    dialogue = ctx.params.get("edit_listening_dialogue", {})
    selection = contract.get("selection", {})
    temporal = plan.temporal
    if (
        plan.backend != "samaudio"
        or plan.mode != "span"
        or plan.result_mode != "residual"
        or plan.use_video_prompt
        or plan.visual_prompt
        or plan.temporal_schema_error
        or plan.temporal_status not in ("resolved", "approximate")
        or temporal is None
        or temporal.kind != "event"
        or temporal.reference_source != "audio"
        or temporal.occurrence not in (None, "each", "all")
        or temporal.relation
        or temporal.end_reference_event
        or temporal.boundary_role
        or temporal.start_time is not None
        or temporal.end_time is not None
        or temporal.offset_seconds != 0
        or temporal.spacing_seconds is not None
        or temporal.window_seconds is not None
        or contract.get("status") != "ready"
        or contract.get("missing_evidence")
        or contract.get("instruction_source") != "original_user_instruction"
        or contract.get("participant_selection")
        or contract.get("placement_windows")
        or selection.get("scope") != "whole_file"
        or selection.get("occurrence") != "unselected"
        or dialogue.get("status") != "ready"
        or dialogue.get("operation") != "remove_sound"
        or dialogue.get("original_instruction") != ctx.user_prompt
        or dialogue.get("selection") != selection
        or dialogue.get("participant_selection")
    ):
        return plan
    questions = contract.get("requirements", [])
    if not questions or any(set(q.get("required_fields", [])) != {"identity"} for q in questions):
        return plan
    identity_ids = {q.get("id") for q in questions}
    if None in identity_ids:
        return plan
    windows = [(p.start, p.end) for p in (plan.time_span or [])]
    if windows != [(p.start_time, p.end_time) for p in plan.placements] or not windows:
        return plan
    if (
        not isinstance(media_duration, (int, float))
        or not math.isfinite(media_duration)
        or media_duration <= 0
    ):
        return plan
    if any(
        not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in (a, b)
        )
        or not 0 <= a < b <= media_duration
        for a, b in windows
    ):
        return plan

    covered_until = 0.0
    gap = False
    for start, end in sorted(windows):
        gap = gap or start > covered_until + 1e-6
        covered_until = max(covered_until, end)
    if not gap and covered_until >= media_duration - 1e-6:
        return plan
    events = {e.get("event_id"): e for e in (analysis or {}).get("events", [])}
    refs = temporal.reference_event
    if not isinstance(refs, list) or refs != [p.event_id for p in plan.placements]:
        return plan
    for placement in plan.placements:
        event = events.get(placement.event_id, {})
        qids = set(event.get("question_ids", []))
        if (
            not qids
            or not qids <= identity_ids
            or event.get("timing_source") != "qwen_observation"
            or event.get("start_time") != placement.start_time
            or event.get("end_time") != placement.end_time
        ):
            return plan
    revised = deepcopy(plan)
    revised.mode = "text"
    revised.time_span = None
    revised.placements = []
    revised.temporal = TemporalConstraint(kind="unspecified", reference_source="audio")
    revised.temporal_status = "not_requested"
    revised.temporal_reason = None
    ctx.params["identity_only_scope_binding"] = dict(
        policy="identity_observation_does_not_select_edit_windows",
        original_plan=asdict(plan),
        execution_scope="whole_file",
        identity_question_ids=sorted(identity_ids),
        new_model_calls=0,
        reason="The original listening contract selects the whole source, with identity evidence only.",
    )
    return revised
