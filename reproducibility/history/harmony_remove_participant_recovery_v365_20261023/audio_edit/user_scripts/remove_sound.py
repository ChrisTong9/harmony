from pathlib import Path
from copy import deepcopy
from dataclasses import asdict

from audio_edit.analyzers.gemini_visual_analyzer import GeminiVideoAnalyzer
from audio_edit.analyzers.qwen_audio_analyzer import QwenAudioAnalyzer
from audio_edit.models import AudioEditContext
from audio_edit.llm.client import LLMClient
from audio_edit.specialists.remove_sound_specialist import RemoveSoundSpecialist
from audio_edit.runners.samaudio_runner import SAMAudioRunner
from audio_edit.plans.temporal import requires_event_grounding, ResolvedPlacement
from audio_edit.utils.extract_audio import extract_audio_for_analysis
from audio_edit.utils.media_util import get_media_duration
from audio_edit.plans.semantic_grounding import ground_from_observations
from audio_edit.plans.remove_sound_plan import TimeSpan, shared_removal_windows
from audio_edit.plans.errors import UnresolvedEditError
from audio_edit.analyzers.edit_listening import listen_for_edit, source_evidence
from audio_edit.analyzers.participant_selection import participant_placements, participant_binding_audit


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def prepare_plan(ctx: AudioEditContext, *, llm=None, audio_analysis=None,
                 visual_analysis=None, require_placements=False):
    """Listen once per request, then plan removal; replacement shares its answers."""
    if not ctx.input_audio:
        raise ValueError("ctx.input_audio is required for remove_sound.")

    input_path = Path(ctx.input_audio)
    suffix = input_path.suffix.lower()
    if suffix not in VIDEO_SUFFIXES | AUDIO_SUFFIXES:
        raise ValueError(f"Unsupported input type for remove_sound: {ctx.input_audio}")

    llm = llm or LLMClient()
    specialist = RemoveSoundSpecialist(llm=llm)
    media_duration = get_media_duration(ctx.input_audio)

    analysis_audio = ctx.input_audio
    if suffix in VIDEO_SUFFIXES:
        ctx.input_video = ctx.input_audio
        analysis_audio = extract_audio_for_analysis(ctx.input_video)
    if audio_analysis is None:
        audio_analysis = listen_for_edit(ctx, llm, QwenAudioAnalyzer(),
            operation='remove_sound', input_audio=analysis_audio)

    from audio_edit.analyzers.participant_extraction import compile_participant_extraction
    ctx.params.pop('participant_extraction',None)
    plan = compile_participant_extraction(ctx,audio_analysis,media_duration) if not require_placements else None
    participant_output = plan is not None
    if plan is None:
        plan = specialist.plan(ctx.user_prompt, media_duration=media_duration,
            audio_analysis=audio_analysis, visual_analysis=visual_analysis,
            planner_subgoal=ctx.params.get('planner_subgoal'),
            require_placements=require_placements)
    if ((audio_analysis or {}).get('edit_listening', {}).get('participant_selection')
            and not participant_output and not plan.temporal_schema_error
            and plan.mode != 'video' and plan.result_mode == 'result'):

        feedback = dict(error='Selected participant removal requires bounded residual source removal',
            previous_plan=asdict(plan))
        ctx.params['participant_mode_recovery'] = dict(status='attempted', feedback=deepcopy(feedback), maximum_replans=1)
        plan = specialist.plan(ctx.user_prompt, media_duration=media_duration,
            audio_analysis=audio_analysis, visual_analysis=visual_analysis,
            planner_subgoal=ctx.params.get('planner_subgoal'),
            require_placements=require_placements, execution_feedback=feedback)
        ctx.params['participant_mode_recovery'].update(
            status='corrected' if plan.result_mode == 'residual' else 'incompatible_no_retry',
            corrected_plan=asdict(plan))
    if (audio_analysis or {}).get('edit_listening', {}).get('participant_selection') and plan.temporal_schema_error:
        raise UnresolvedEditError('Participant window binding cannot repair an invalid temporal request: '+plan.temporal_schema_error)
    plan.removal_evidence = source_evidence(audio_analysis, plan.text_prompt, 'qwen_directed_removal')
    ctx.params['removal_perception'] = dict(status='dialogue', analysis=deepcopy(audio_analysis))
    event_timing = requires_event_grounding(plan.temporal)
    reference_source = plan.temporal.reference_source if event_timing else "audio"
    if suffix in VIDEO_SUFFIXES and event_timing and reference_source in {'visual', 'either'} and visual_analysis is None:
        visual_analysis = GeminiVideoAnalyzer().analyze(
            input_video=ctx.input_video, user_prompt=ctx.user_prompt)

    if event_timing:
        plan = specialist.ground_plan(
            plan, audio_analysis=audio_analysis, visual_analysis=visual_analysis,
            media_duration=media_duration,
        )

    if (require_placements and plan.mode == "text" and plan.temporal_status == "not_requested"
            and (plan.temporal is None or plan.temporal.kind == "unspecified")):
        windows = shared_removal_windows(plan, media_duration)
        plan.placements = [ResolvedPlacement(a, b, evidence="Whole-file edit scope; source identity remains SAM's removal target") for a, b in windows]
        plan.temporal_status = "resolved"
    needs_review = plan.temporal_status == 'unresolved' or (require_placements and not plan.placements)
    known_interval = plan.temporal is not None and plan.temporal.kind == 'interval'

    if needs_review and not known_interval:
        resolution = ground_from_observations(llm, user_prompt=ctx.user_prompt,
            target_sound=plan.text_prompt, media_duration=media_duration,
            audio_analysis=audio_analysis, operation='remove',
            temporal_constraint=asdict(plan.temporal) if plan.temporal else None)
        ctx.params['grounding_resolution'] = dict(status=resolution.status, reason=resolution.reason)
        plan.temporal_status, plan.temporal_reason = resolution.status, resolution.reason
        plan.placements = resolution.placements
        if resolution.status in ('resolved', 'approximate'):
            if resolution.target_description:
                plan.text_prompt = resolution.target_description
            plan.time_span = [TimeSpan(p.start_time, p.end_time) for p in resolution.placements]
            plan.mode, plan.use_video_prompt = 'span', False
    if suffix in VIDEO_SUFFIXES and plan.mode != "video":
        ctx.input_audio = analysis_audio
    if (audio_analysis or {}).get('edit_listening', {}).get('participant_selection'):
        contract = audio_analysis['edit_listening']
        literal = (contract.get('selection') or {}).get('scope')=='literal_time'
        current_windows = [(p.start_time,p.end_time) for p in plan.placements]
        if not current_windows and plan.time_span:
            current_windows = [(p.start,p.end) for p in plan.time_span]
        placements = participant_placements(audio_analysis, media_duration,
            literal_windows=current_windows if literal else None)
        if (plan.mode=='video' or (plan.result_mode!='residual'
                and not (participant_output and plan.result_mode=='result'))):
            raise UnresolvedEditError('Selected participant removal requires bounded residual source removal')
        ctx.params['participant_scope_binding'] = participant_binding_audit(audio_analysis,
            dict(mode=plan.mode, windows=current_windows, temporal_status=plan.temporal_status), placements)
        plan.mode, plan.use_video_prompt = 'span', False
        plan.placements = placements
        plan.time_span = [TimeSpan(p.start_time,p.end_time) for p in placements]
        plan.temporal_status = 'approximate'
        plan.temporal_reason = 'Bound to the selected participant event windows from the listening contract'
    if not require_placements:
        from audio_edit.analyzers.source_query_binding import bind
        plan = bind(ctx, plan, audio_analysis)
        from audio_edit.analyzers.identity_only_removal_scope import bind as bind_identity_scope
        plan = bind_identity_scope(ctx, plan, audio_analysis, media_duration)


    ctx.params['remove_plan'] = asdict(plan)
    if require_placements:
        if plan.temporal_status == 'unresolved' or not plan.placements:
            raise UnresolvedEditError(plan.temporal_reason or 'Removal has no resolved windows for addition')
        try:
            windows = shared_removal_windows(plan, media_duration)
        except ValueError as exc:
            raise UnresolvedEditError(str(exc)) from exc
        if [(p.start_time, p.end_time) for p in plan.placements] != windows:
            raise ValueError("Removal placements and execution scope differ")
    return plan, audio_analysis, visual_analysis


def run(ctx: AudioEditContext, params: dict) -> AudioEditContext:
    plan, _, _ = prepare_plan(ctx)
    from audio_edit.analyzers.local_single_source import decide as local_decide, render as local_render
    local = local_decide(ctx, plan)
    if local.get('eligible'):
        return local_render(ctx, local)
    from audio_edit.analyzers.removal_complement import decide
    decision=decide(ctx,plan)
    ctx.params['removal_complement']=decision
    if decision.get('eligible'):
        from audio_edit.tools.complementary_source import retain_music_vocals
        path,audit=retain_music_vocals(ctx.input_audio,None,decision)
        ctx.params['removal_complement_execution']=audit
        ctx.output_audio=path
        return ctx
    from audio_edit.tools.retained_vocal_activities import decide as retained_decide, render as retained_render
    retained = retained_decide(ctx, plan)
    if retained.get('eligible'):
        return retained_render(ctx, plan, retained, SAMAudioRunner())
    result = SAMAudioRunner().run_remove(ctx, plan)
    if result.params.get('participant_extraction',{}).get('eligible'):
        from audio_edit.analyzers.participant_extraction import retain_selected_intervals
        return retain_selected_intervals(result)
    from audio_edit.analyzers.typed_residual_removal import apply as complete_typed_residual
    result = complete_typed_residual(result, plan)
    from audio_edit.analyzers.laughter_residual_query import apply as refine_laughter
    result = refine_laughter(result, plan)
    from audio_edit.analyzers.residual_source_query import apply as refine_source_query
    return refine_source_query(result, plan)
