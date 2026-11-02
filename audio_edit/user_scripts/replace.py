from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from audio_edit.analyzers.gemini_visual_analyzer import GeminiVideoAnalyzer
from audio_edit.analyzers.qwen_audio_analyzer import QwenAudioAnalyzer
from audio_edit.analyzers.edit_listening import listen_for_edit
from audio_edit.llm.client import LLMClient
from audio_edit.models import AudioEditContext

from audio_edit.specialists.replace_sound_specialist import ReplaceSoundSpecialist
from audio_edit.user_scripts.remove_sound import prepare_plan as prepare_remove_plan
from audio_edit.specialists.add_sound_specialist import AddSoundSpecialist

from audio_edit.plans.replace_sound_plan import ReplaceSoundPlan
from audio_edit.runners.replace_runner import ReplaceSoundRunner
from audio_edit.runners.gemini_omni_video_runner import GeminiOmniVideoRunner
from audio_edit.specialists.gemini_omni_video_prompt_specialist import (
    GeminiOmniVideoPromptSpecialist,
)
from audio_edit.specialists.replace_video_specialist import ReplaceVideoSpecialist
from audio_edit.utils.video_mux import mux_video_with_audio
from audio_edit.utils.extract_audio import extract_audio_for_analysis


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def run(ctx: AudioEditContext, params: dict) -> AudioEditContext:
    source_sound = params.get("source_sound")
    target_sound = params.get("target_sound")

    full_user_prompt = ctx.user_prompt or params.get("user_prompt") or params.get("prompt")

    if not isinstance(full_user_prompt, str) or not full_user_prompt.strip():
        raise ValueError(
            "replace_sound requires ctx.user_prompt, "
            "params['user_prompt'], or params['prompt']. "
            f"Got params={params}"
        )

    full_user_prompt = full_user_prompt.strip()

    audio_user_prompt = full_user_prompt
    ctx.user_prompt = full_user_prompt

    if not ctx.input_audio:
        raise ValueError("ctx.input_audio is required for replace_sound.")

    input_path = Path(ctx.input_audio)
    suffix = input_path.suffix.lower()

    visual_analysis = None
    audio_analysis = None
    original_video = None

    if suffix in VIDEO_SUFFIXES:
        original_video = ctx.input_audio

        visual_analysis = GeminiVideoAnalyzer().analyze(
            input_video=original_video,
            user_prompt=ctx.user_prompt,
        )

        audio_for_analysis = extract_audio_for_analysis(original_video)

        ctx.input_video = original_video
        ctx.input_audio = audio_for_analysis

    elif suffix in AUDIO_SUFFIXES:
        visual_analysis = None

    else:
        raise ValueError(f"Unsupported input type for replace_sound: {ctx.input_audio}")

    llm = LLMClient()
    audio_analysis = listen_for_edit(ctx, llm, QwenAudioAnalyzer(), operation="replace")

    replace_specialist = ReplaceSoundSpecialist(llm=llm)

    replace_prompt_plan = replace_specialist.plan(
        ctx=ctx,
        user_prompt=full_user_prompt,
        visual_analysis=visual_analysis,
        audio_analysis=audio_analysis,
    )

    video_replacement = replace_prompt_plan.video_replacement
    source_object = replace_prompt_plan.source_object
    target_object = replace_prompt_plan.target_object

    if video_replacement and original_video is None:
        raise ValueError("Video replacement was requested, but the input media has no video.")

    remove_ctx = deepcopy(ctx)
    remove_ctx.user_prompt = full_user_prompt
    remove_ctx.params["planner_subgoal"] = dict(
        source="planner_derived",
        operation="remove",
        description=replace_prompt_plan.remove_user_prompt,
    )

    remove_plan, audio_analysis, visual_analysis = prepare_remove_plan(
        remove_ctx,
        llm=llm,
        audio_analysis=audio_analysis,
        visual_analysis=visual_analysis,
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
            visual_analysis=visual_analysis,
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

    if video_replacement:
        add_plan.source_type = "video"
        add_plan.is_on_screen = True
        if isinstance(add_plan.composer, dict):
            add_plan.composer["is_onscreen"] = True

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

    edited_video = None
    if video_replacement:
        video_ctx = deepcopy(ctx)
        video_ctx.input_video = original_video

        video_edit_prompt = GeminiOmniVideoPromptSpecialist(
            llm=llm,
        ).build_prompt(
            user_prompt=full_user_prompt,
            source_object=source_object,
            target_object=target_object,
            source_sound=replace_prompt_plan.source_sound,
            target_sound=replace_prompt_plan.target_sound,
            visual_analysis=visual_analysis,
            audio_analysis=audio_analysis,
        )
        video_params = {
            "source_object": source_object,
            "target_object": target_object,
            "edit_prompt": video_edit_prompt,
        }

        video_specialist = ReplaceVideoSpecialist()
        video_plan = video_specialist.plan(
            ctx=video_ctx,
            params=video_params,
        )

        video_runner = GeminiOmniVideoRunner()
        video_result_ctx = video_runner.run(video_ctx, video_plan)
        edited_video = video_result_ctx.output_video

        if not edited_video:
            raise RuntimeError("Video replacement completed without producing output_video.")

        ctx.input_video = edited_video

    runner = ReplaceSoundRunner()
    result_ctx = runner.run(ctx, replace_plan)
    from audio_edit.analyzers.replace_music_residual import apply as complete_music_residual

    result_ctx = complete_music_residual(
        result_ctx, replace_plan, ctx.input_audio, full_user_prompt
    )

    if not video_replacement:
        if original_video:
            if not result_ctx.output_audio:
                raise RuntimeError("Audio replacement completed without producing output_audio.")

            original_video_path = Path(original_video)
            final_output = (
                Path(result_ctx.output_audio).resolve().parent
                / f"{original_video_path.stem}_with_replaced_audio.mp4"
            )

            result_ctx.output_video = mux_video_with_audio(
                video_path=original_video,
                audio_path=result_ctx.output_audio,
                output_path=str(final_output),
            )

        return result_ctx

    if not result_ctx.output_audio:
        raise RuntimeError("Audio replacement completed without producing output_audio.")

    edited_video_path = Path(edited_video)

    final_output = edited_video_path.parent / f"{edited_video_path.stem}_with_replaced_audio.mp4"

    final_video = mux_video_with_audio(
        video_path=edited_video,
        audio_path=result_ctx.output_audio,
        output_path=str(final_output),
    )

    result_ctx.output_video = final_video

    return result_ctx
