from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

from audio_edit.analyzers.qwen_audio_analyzer import QwenAudioAnalyzer
from audio_edit.llm.client import LLMClient
from audio_edit.models import AudioEditContext
from audio_edit.runners.add_sound_runner import AddSoundRunner
from audio_edit.specialists.add_sound_specialist import AddSoundSpecialist
from audio_edit.utils.audio_input import validate_audio_input
from audio_edit.utils.media_util import get_media_duration
from audio_edit.plans.temporal_boundaries import uncertain_anchors
from audio_edit.plans.add_timing import compile_plan
from audio_edit.plans.add_intent import references, bound_requests, bind_references
from audio_edit.plans.event_ids import assign_event_ids
from audio_edit.plans.errors import UnresolvedEditError
from audio_edit.utils.bounded_tools import bounded_call
from audio_edit.analyzers.edit_listening import listen_for_edit
from audio_edit.plans.add_recovery import (
    enabled as recovery_enabled,
    recover_plan,
    unresolved_analysis,
)


def has_event_anchor(value):
    if isinstance(value, dict):
        return "event_id" in value or any(has_event_anchor(v) for v in value.values())
    return isinstance(value, list) and any(has_event_anchor(v) for v in value)


def has_uncertain_event_anchor(value, audio_analysis):
    events = {e.get("event_id"): e for e in (audio_analysis or {}).get("events", [])}

    def uncertain(item):
        if isinstance(item, dict):
            if "event_id" in item:
                if item.get("boundary_role") or isinstance(item["event_id"], list):
                    return uncertain_anchors(item, events)
                event = events.get(item["event_id"])
                return event is None or event.get("resolution_status") != "resolved"
            return any(uncertain(v) for v in item.values())
        return isinstance(item, list) and any(uncertain(v) for v in item)

    return uncertain(value)


def run(ctx: AudioEditContext, params: dict) -> AudioEditContext:
    validate_audio_input(ctx.input_audio)
    llm = LLMClient()
    try:
        audio_analysis = listen_for_edit(ctx, llm, QwenAudioAnalyzer(), operation="add_sound")
    except UnresolvedEditError as exc:
        if not recovery_enabled(ctx):
            raise
        duration = get_media_duration(ctx.input_audio)
        plan = recover_plan(
            ctx,
            llm,
            duration=duration,
            analysis=unresolved_analysis(ctx),
            reason=str(exc),
        )
        return AddSoundRunner().run(ctx, plan)
    specialist = AddSoundSpecialist(llm)
    plan = bounded_call(
        ctx,
        "initial_add_intent",
        lambda: specialist.plan(ctx, audio_analysis=audio_analysis),
    )
    duration = get_media_duration(ctx.input_audio)
    assign_event_ids(audio_analysis, "audio", duration)
    refs = references(plan.intent) if plan.intent["edits"] else {}
    try:
        bindings = (
            bounded_call(
                ctx,
                "bind_add_references",
                lambda: bind_references(
                    llm,
                    plan.intent,
                    audio_analysis,
                    audit=ctx.params.setdefault("add_binding_calls", []),
                ),
            )
            if refs
            else {}
        )
    except ValueError as exc:
        if not recovery_enabled(ctx):
            raise
        plan = recover_plan(
            ctx,
            llm,
            duration=duration,
            analysis=audio_analysis,
            reason=str(exc),
            prior_plan=plan,
        )
        return AddSoundRunner().run(ctx, plan)
    ctx.params["add_reference_bindings"] = deepcopy(bindings)
    if plan.intent["edits"]:
        bound_requests(plan.intent, bindings)

    activity = [
        result
        for call in ctx.params.get("edit_listening_dialogue", {}).get("tool_calls", [])
        if call.get("name") == "boundary_activity" and isinstance(call.get("results"), list)
        for result in call["results"]
    ]
    ctx.params["boundary_activity_checks"] = activity
    plan = compile_plan(
        plan,
        duration=duration,
        audio_analysis=audio_analysis,
        bindings=bindings,
        adjust_choices=True,
    )
    ctx.params["add_intent"] = deepcopy(plan.intent)
    ctx.params["intent_revisions"] = deepcopy(plan.intent.get("revisions", []))
    ctx.params["add_reference_bindings"] = deepcopy(bindings)
    ctx.params["temporal_resolution_status"] = plan.temporal_status
    ctx.params["anchor_audit"] = plan.edit_windows
    ctx.params["add_plan"] = asdict(plan)
    if plan.temporal_status == "unresolved":
        if not recovery_enabled(ctx):
            raise UnresolvedEditError(plan.temporal_reason)
        plan = recover_plan(
            ctx,
            llm,
            duration=duration,
            analysis=audio_analysis,
            reason=plan.temporal_reason,
            prior_plan=plan,
        )
    return AddSoundRunner().run(ctx, plan)
