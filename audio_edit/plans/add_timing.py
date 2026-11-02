"""Recompute an owned add intent; review changes evidence, not semantic meaning."""

from copy import deepcopy
from audio_edit.plans.add_intent import compile_add_intent, DurationChoiceConflict
from audio_edit.plans.temporal_boundaries import BoundaryEvidenceError
from audio_edit.plans.temporal import ResolvedPlacement


def compile_plan(
    plan,
    *,
    duration,
    audio_analysis=None,
    visual_analysis=None,
    bindings=None,
    asset_durations=None,
    adjust_choices=False,
    replacement_extent_limits=None,
):
    plan = deepcopy(plan)
    from audio_edit.plans.add_sound_plan import MediaAnalysis

    plan.media_analysis = plan.media_analysis or MediaAnalysis()
    if audio_analysis is not None:
        plan.media_analysis.audio_analysis = deepcopy(audio_analysis)
    if visual_analysis is not None:
        plan.media_analysis.visual_analysis = deepcopy(visual_analysis)
    if bindings is not None:
        plan.reference_bindings = deepcopy(bindings)
    plan.execution_placements = []
    plan.placements = []
    plan.edit_windows = []
    if plan.intent.get("entry_error"):
        plan.temporal_status = "unresolved"
        plan.temporal_reason = plan.intent["entry_error"]
        plan.compile_error_kind = plan.intent.get("error_kind", "constraint")
        return plan
    while True:
        try:
            windows, placements = compile_add_intent(
                plan.intent,
                duration=duration,
                audio_analysis=plan.media_analysis.audio_analysis,
                visual_analysis=plan.media_analysis.visual_analysis,
                bindings=plan.reference_bindings,
                asset_durations=asset_durations,
                replacement_extent_limits=replacement_extent_limits,
            )
            break
        except DurationChoiceConflict as exc:
            e = plan.intent["edits"][exc.index]
            g = e["generation"]
            if (
                adjust_choices
                and asset_durations is None
                and g.get("source") == "planner_choice"
                and g.get("adjustable")
                and g.get("min_seconds", float("inf")) <= exc.available + 1e-8
                and exc.available < g["target_seconds"]
            ):
                before = plan.intent["revision"]
                old = g["target_seconds"]
                plan.intent = deepcopy(plan.intent)
                plan.intent["revision"] = before + 1
                adjusted = max(g["min_seconds"], exc.available)
                plan.intent["edits"][exc.index]["generation"]["target_seconds"] = adjusted
                revision = dict(
                    from_revision=before,
                    to_revision=before + 1,
                    edit_index=exc.index,
                    field="generation.target_seconds",
                    before=old,
                    after=adjusted,
                    source="planner_choice",
                    reason=str(exc),
                )
                plan.intent.setdefault("revisions", []).append(revision)
                continue
            plan.temporal_status = "unresolved"
            plan.temporal_reason = str(exc)
            plan.compile_error_kind = "constraint"
            return plan
        except (ValueError, TypeError, KeyError) as exc:
            plan.temporal_status = "unresolved"
            plan.temporal_reason = str(exc)
            plan.compile_error_kind = (
                "evidence" if isinstance(exc, BoundaryEvidenceError) else "constraint"
            )
            return plan
    plan.requested_edits = deepcopy(plan.intent["edits"])
    plan.edit_windows = windows
    plan.execution_placements = placements
    targets = [
        p["generation_target_seconds"]
        for p in placements
        if p["generation_target_seconds"] is not None
    ]
    plan.duration = max(targets) if targets else None
    plan.repeat_count = getattr(plan.temporal, "count", None)
    plan.placements = [
        ResolvedPlacement(
            p["start_time"],
            p["end_time"],
            approximate=p["status"] != "resolved",
            resolution_status=p["status"],
            evidence=p["evidence"],
        )
        for p in placements
    ]
    plan.temporal_status = (
        "estimated"
        if any(p["status"] == "estimated" for p in placements)
        else "approximate"
        if any(p["status"] == "approximate" for p in placements)
        else "resolved"
    )
    plan.temporal_reason = None
    plan.compile_error_kind = None
    plan.start_time = placements[0]["start_time"]
    plan.end_time = placements[0]["end_time"]
    plan.material_pending = asset_durations is None
    recovery = plan.intent.get("timing_recovery")
    if recovery:
        if (
            recovery.get("policy") != "overlap_preserve"
            or recovery.get("authority") != "user_execution_policy"
        ):
            raise ValueError("Unknown ADD timing recovery policy")
        plan.temporal_status = "estimated"
        for item in plan.execution_placements + plan.edit_windows:
            item["status"] = "estimated"
            item.setdefault("evidence", []).append(
                dict(
                    basis="user_authorized_estimated_overlay",
                    original_failure=recovery.get("original_failure"),
                    measured_timing=False,
                )
            )
        for item in plan.placements:
            item.resolution_status = "estimated"
            item.approximate = True
    return plan


def estimated_plan(plan, *, audio_analysis, visual_analysis, duration, input_audio):
    """Legacy public name: evidence can be approximate, absent evidence cannot execute.

    No RMS phase creation, implicit scene anchor or restoration of a failed plan.
    The active add workflow calls compile_plan directly.
    """
    if not plan.intent:
        from audio_edit.plans.add_compat import adapt_edits, adapt_runner_plan

        plan = deepcopy(plan)
        if plan.requested_edits:
            plan.intent = adapt_edits(
                dict(
                    edits=plan.requested_edits,
                    duration=plan.duration,
                    audio_prompt=plan.audio_prompt,
                ),
                audio_analysis=audio_analysis,
                visual_analysis=visual_analysis,
                duration=duration,
            )
        else:
            plan = adapt_runner_plan(plan, duration)
    result = compile_plan(
        plan,
        duration=duration,
        audio_analysis=audio_analysis,
        visual_analysis=visual_analysis,
    )
    if result.temporal_status == "unresolved":
        raise ValueError(result.temporal_reason)
    return result
