"""Ordered removal strategies with explicit eligibility and existing call limits.

These policies handle distinct source classes and evidence conditions. Selection
does not imply a general solution for every source-removal request.
"""

from audio_edit.policies.core import PolicySpec, record_policy
from . import (
    music_source_binding,
    residual_completion,
    residual_query,
    retained_complement,
    single_source_region,
    vocal_activity_query,
    vocal_activity_retention,
)


LOCAL_REGION = PolicySpec(
    policy_id="removal.single_source_region",
    stage="removal",
    condition="One bound event region with explicit first-observation evidence of no overlapping source",
    evidence="Original audio identity, selected event boundaries, and first listening evidence",
    action="Mute only that region with internal edge fades",
    budget="No new model calls; one deterministic render",
)
COMPLEMENT = PolicySpec(
    policy_id="removal.retained_complement",
    stage="removal",
    condition="Whole-file music removal with exactly one supported nonmusical class to preserve",
    evidence="First original-source facts and quoted original instruction",
    action="Extract the retained class with the existing separator",
    budget="At most one original-source observation, one binding decision, and one separation",
)
MUSIC_QUERY = PolicySpec(
    policy_id="removal.music_source_binding",
    stage="removal",
    condition="Eligible whole-music request whose first facts provide a more specific source phrase",
    evidence="Existing first source observation; exact instruction and fact quotations",
    action="Change only the removal query before separation",
    budget="One persistent first binding; no new audio observation or retry decision",
)
VOCAL_RETENTION = PolicySpec(
    policy_id="removal.vocal_activity_retention",
    stage="removal",
    condition="Explicit whole-file extraction of two distinct supported vocal activities",
    evidence="Original extraction instruction and matching two-activity source query",
    action="Separate each activity from the same original and sum unscaled estimates",
    budget="Two separation calls, one per requested activity; no gain search",
)
SOURCE_SEPARATION = PolicySpec(
    policy_id="removal.source_separation",
    stage="removal",
    condition="No earlier direct-render or retained-source strategy was selected",
    evidence="The compiled removal plan and its original-source evidence",
    action="Execute the existing removal runner",
    budget="One runner invocation; its existing direct-silence decision remains available",
)
PARTICIPANT_INTERVALS = PolicySpec(
    policy_id="removal.participant_intervals",
    stage="removal",
    condition="An explicitly selected participant extraction has validated identity-linked intervals",
    evidence="Existing participant-extraction plan and first separated output",
    action="Keep only the selected intervals and skip residual-removal policies",
    budget="No additional separation or listening calls",
)
RESIDUAL_COMPLETION = PolicySpec(
    policy_id="removal.residual_completion",
    stage="removal",
    condition="A selected music event remains in an eligible generic residual",
    evidence="Bound original event and one persistent first residual-source inventory",
    action="Remove the supported remaining event within its existing window",
    budget="At most one residual observation and one additional separation",
)
VOCAL_QUERY = PolicySpec(
    policy_id="removal.vocal_activity_query",
    stage="removal",
    condition="Whole-file laughter removal still has explicit evidence of the same laughter",
    evidence="Persistent first original and residual descriptions and one source binding",
    action="Refine the laughter query only and separate once from the original",
    budget="At most two first source observations, one binding, and one additional separation",
)
RESIDUAL_QUERY = PolicySpec(
    policy_id="removal.residual_query",
    stage="removal",
    condition="Whole-file knock removal, or noise removal preserving voice, with supported residual identity",
    evidence="Persistent first original/residual descriptions with exact supporting quotations",
    action="Refine only the source query and reuse or execute its first separation",
    budget="At most two first source observations, one binding, and one persistent separation execution",
)

POLICIES = (
    LOCAL_REGION,
    COMPLEMENT,
    MUSIC_QUERY,
    VOCAL_RETENTION,
    SOURCE_SEPARATION,
    PARTICIPANT_INTERVALS,
    RESIDUAL_COMPLETION,
    VOCAL_QUERY,
    RESIDUAL_QUERY,
)


def _record_correction(ctx, policy, audit_key):
    """Summarize the existing receipt without repeating its eligibility checks."""
    audit = ctx.params.get(audit_key)
    if not audit:
        record_policy(ctx, policy, "not_applicable")
        return
    status = "retained"
    if (
        audit.get("new_SAM_calls", 0)
        or audit.get("status") == "reused_first_acoustic_query_execution"
    ):
        status = "applied"
    record_policy(ctx, policy, status, audit_key=audit_key, result=audit.get("status"))


def run_removal(ctx, plan, runner):
    """Apply the original strategy order once, retaining all branch-specific gates."""
    current = LOCAL_REGION
    try:
        local = single_source_region.decide(ctx, plan)
        if local.get("eligible"):
            record_policy(ctx, current, "selected", reason=local.get("reason"))
            result = single_source_region.render(ctx, local)
            record_policy(result, current, "applied")
            return result
        record_policy(ctx, current, "not_applicable", reason=local.get("reason"))

        current = COMPLEMENT
        decision = retained_complement.decide(ctx, plan)
        ctx.params["removal_complement"] = decision
        if decision.get("eligible"):
            from audio_edit.tools.complementary_source import retain_music_vocals

            record_policy(ctx, current, "selected", result=decision.get("status"))
            path, audit = retain_music_vocals(ctx.input_audio, None, decision)
            ctx.params["removal_complement_execution"] = audit
            ctx.output_audio = path
            record_policy(ctx, current, "applied" if path else "retained")
            return ctx
        record_policy(ctx, current, "not_applicable", result=decision.get("status"))

        current = MUSIC_QUERY
        plan = music_source_binding.apply(ctx, plan)
        binding = ctx.params.get("single_music_query") or {}
        record_policy(
            ctx,
            current,
            "selected" if binding.get("eligible") else "retained" if binding else "not_applicable",
            result=binding.get("status"),
        )

        current = VOCAL_RETENTION
        retained = vocal_activity_retention.decide(ctx, plan)
        if retained.get("eligible"):
            record_policy(ctx, current, "selected", reason=retained.get("reason"))
            result = vocal_activity_retention.render(ctx, plan, retained, runner)
            record_policy(result, current, "applied")
            return result
        record_policy(ctx, current, "not_applicable", reason=retained.get("reason"))

        current = SOURCE_SEPARATION
        ctx = runner.run_remove(ctx, plan)
        record_policy(ctx, current, "applied")

        current = PARTICIPANT_INTERVALS
        if ctx.params.get("participant_extraction", {}).get("eligible"):
            from audio_edit.analyzers.participant_extraction import retain_selected_intervals

            result = retain_selected_intervals(ctx)
            record_policy(result, current, "applied")
            return result
        record_policy(ctx, current, "not_applicable")

        current = RESIDUAL_COMPLETION
        ctx = residual_completion.apply(ctx, plan)
        _record_correction(ctx, current, "typed_residual_completion")
        current = VOCAL_QUERY
        ctx = vocal_activity_query.apply(ctx, plan)
        _record_correction(ctx, current, "laughter_residual_query_execution")
        current = RESIDUAL_QUERY
        ctx = residual_query.apply(ctx, plan)
        _record_correction(ctx, current, "residual_source_query_execution")
        return ctx
    except Exception as exc:
        record_policy(ctx, current, "error", error=f"{type(exc).__name__}: {exc}")
        raise
