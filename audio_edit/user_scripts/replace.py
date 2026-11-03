from __future__ import annotations

from copy import deepcopy

from audio_edit.analyzers.qwen_audio_analyzer import QwenAudioAnalyzer
from audio_edit.analyzers.edit_listening import listen_for_edit
from audio_edit.llm.client import LLMClient
from audio_edit.models import AudioEditContext

from audio_edit.specialists.replace_sound_specialist import ReplaceSoundSpecialist
from audio_edit.user_scripts.remove_sound import prepare_plan as prepare_remove_plan
from audio_edit.specialists.add_sound_specialist import AddSoundSpecialist

from audio_edit.plans.replace_sound_plan import ReplaceSoundPlan
from audio_edit.runners.replace_runner import ReplaceSoundRunner
from audio_edit.utils.audio_input import validate_audio_input


def run(ctx: AudioEditContext, params: dict) -> AudioEditContext:
    full_user_prompt = ctx.user_prompt or params.get("user_prompt") or params.get("prompt")

    if not isinstance(full_user_prompt, str) or not full_user_prompt.strip():
        raise ValueError(
            "replace_sound requires ctx.user_prompt, "
            "params['user_prompt'], or params['prompt']. "
            f"Got params={params}"
        )

    full_user_prompt = full_user_prompt.strip()

    ctx.user_prompt = full_user_prompt

    validate_audio_input(ctx.input_audio)

    llm = LLMClient()
    audio_analysis = listen_for_edit(ctx, llm, QwenAudioAnalyzer(), operation="replace")

    replace_specialist = ReplaceSoundSpecialist(llm=llm)

    replace_prompt_plan = replace_specialist.plan(
        ctx=ctx,
        user_prompt=full_user_prompt,
        audio_analysis=audio_analysis,
    )

    remove_ctx = deepcopy(ctx)
    remove_ctx.user_prompt = full_user_prompt
    remove_ctx.params["planner_subgoal"] = dict(
        source="planner_derived",
        operation="remove",
        description=replace_prompt_plan.remove_user_prompt,
    )

    remove_plan, audio_analysis = prepare_remove_plan(
        remove_ctx,
        llm=llm,
        audio_analysis=audio_analysis,
        require_placements=True,
    )
    shared_placements = deepcopy(remove_plan.placements)
    ctx.params["replacement_window_owner"] = "remove_plan.time_span / placements"
    ctx.params["remove_plan"] = remove_ctx.params.get("remove_plan")
    ctx.params["removal_perception"] = deepcopy(remove_ctx.params.get("removal_perception"))
    ctx.warnings.extend(remove_ctx.warnings[len(ctx.warnings) :])
    if "grounding_resolution" in remove_ctx.params:
        ctx.params["grounding_resolution"] = remove_ctx.params["grounding_resolution"]

    add_ctx = deepcopy(ctx)
    add_ctx.user_prompt = full_user_prompt
    add_ctx.params["planner_subgoal"] = dict(
        source="planner_derived", operation="add", description=replace_prompt_plan.add_user_prompt
    )

    add_specialist = AddSoundSpecialist(llm=llm)

    try:
        add_plan = add_specialist.plan(
            ctx=add_ctx,
            audio_analysis=audio_analysis,
            resolved_placements=shared_placements,
        )
    finally:
        if "add_plan_response" in add_ctx.params:
            ctx.params["replacement_add_plan_response"] = deepcopy(
                add_ctx.params["add_plan_response"]
            )
        for key in ("add_planning_request", "add_planning_responses", "generation_decisions"):
            if key in add_ctx.params:
                ctx.params["replacement_" + key] = deepcopy(add_ctx.params[key])

    ctx.params["replacement_add_intent_revisions"] = deepcopy(add_plan.intent.get("revisions", []))

    replace_plan = ReplaceSoundPlan(
        user_prompt=ctx.user_prompt,
        source_sound=replace_prompt_plan.source_sound,
        target_sound=replace_prompt_plan.target_sound,
        remove_user_prompt=replace_prompt_plan.remove_user_prompt,
        add_user_prompt=replace_prompt_plan.add_user_prompt,
        remove_plan=remove_plan,
        add_plan=add_plan,
        notes=replace_prompt_plan.notes,
        temporal=remove_plan.temporal,
        temporal_status=remove_plan.temporal_status,
        temporal_reason=remove_plan.temporal_reason,
        placements=shared_placements,
    )

    runner = ReplaceSoundRunner()
    result_ctx = runner.run(ctx, replace_plan)
    from audio_edit.analyzers.replace_music_residual import apply as complete_music_residual

    result_ctx = complete_music_residual(
        result_ctx, replace_plan, ctx.input_audio, full_user_prompt
    )

    return result_ctx
