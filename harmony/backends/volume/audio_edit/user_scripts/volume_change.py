from __future__ import annotations

from audio_edit.utils.ablation_policy import execution_corrections_enabled

from pathlib import Path
from dataclasses import asdict
import json

from audio_edit.analyzers.gemini_visual_analyzer import GeminiVideoAnalyzer
from audio_edit.analyzers.qwen_audio_analyzer import QwenAudioAnalyzer
from audio_edit.llm.client import LLMClient
from audio_edit.models import AudioEditContext
from audio_edit.runners.volume_change_runner import VolumeChangeRunner
from audio_edit.specialists.volume_change_specialist import VolumeChangeSpecialist
from audio_edit.plans.temporal import requires_event_grounding
from audio_edit.utils.extract_audio import extract_audio_for_analysis
from audio_edit.utils.media_util import get_media_duration
from audio_edit.plans.semantic_grounding import ground_from_observations
from audio_edit.analyzers.edit_listening import listen_for_edit, source_evidence
from audio_edit.analyzers.participant_selection import participant_placements, participant_binding_audit
from audio_edit.plans.errors import UnresolvedEditError


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def run(ctx: AudioEditContext, params: dict) -> AudioEditContext:
    if not ctx.input_audio:
        raise ValueError("ctx.input_audio is required for volume_change.")

    suffix = Path(ctx.input_audio).suffix.lower()
    is_video = suffix in VIDEO_SUFFIXES
    visual_analysis = None
    if is_video:
        ctx.input_video = ctx.input_audio
        ctx.input_audio = extract_audio_for_analysis(ctx.input_video)
    elif suffix not in AUDIO_SUFFIXES:
        raise ValueError(f"Unsupported input type for volume_change: {ctx.input_audio}")

    specialist = VolumeChangeSpecialist(LLMClient())
    media_duration = get_media_duration(ctx.input_audio)
    audio_analysis = listen_for_edit(ctx, specialist.llm, QwenAudioAnalyzer(), operation='volume_change')
    plan = specialist.plan(ctx.user_prompt, media_duration=media_duration,
                           audio_analysis=audio_analysis)
    actions = plan.changes or [plan]
    if (audio_analysis or {}).get('edit_listening', {}).get('participant_selection') and any(
            action.temporal_schema_error for action in actions):
        raise UnresolvedEditError('Participant window binding cannot repair an invalid temporal request')
    event_actions = [p for p in actions if requires_event_grounding(p.temporal)]
    sources = {p.temporal.reference_source for p in event_actions}
    for action in actions:
        targets = action.effective_target_sounds()
        if action.apply_to == 'target_source' and len(targets) == 1:
            action.source_evidence = source_evidence(audio_analysis, targets[0], 'qwen_directed_volume')
    ctx.params['volume_perception'] = dict(status='dialogue', analysis=audio_analysis)
    if is_video and sources & {'visual', 'either'}:
        visual_analysis = GeminiVideoAnalyzer().analyze(input_video=ctx.input_video, user_prompt=ctx.user_prompt)
    if event_actions:
        specialist.ground_plan(plan, audio_analysis=audio_analysis, visual_analysis=visual_analysis,
                               media_duration=media_duration)
    for index, action in enumerate(actions):
        if action.temporal_status != 'unresolved' or not execution_corrections_enabled('volume_plan_grounding'):
            continue


        action_prompt = ("Resolve ONLY the current volume action. Other actions in the "
            "original instruction are context, not additional regions for this action. "
            + json.dumps(dict(original_instruction=ctx.user_prompt, action_index=index,
                target_sounds=action.effective_target_sounds(), gain_profile=action.gain_profile,
                gain_db=action.gain_db, temporal=asdict(action.temporal) if action.temporal else action.temporal_input,
                notes=action.notes), ensure_ascii=False))
        resolution = ground_from_observations(specialist.llm, user_prompt=action_prompt,
            target_sound=action.target_sound, media_duration=media_duration,
            audio_analysis=audio_analysis, operation='volume',
            temporal_constraint=asdict(action.temporal) if action.temporal else None)
        ctx.params.setdefault('volume_grounding_resolutions', []).append(dict(
            action_index=index, status=resolution.status, reason=resolution.reason))
        action.temporal_status, action.temporal_reason = resolution.status, resolution.reason
        action.placements = resolution.placements
        if resolution.target_description and resolution.status != 'unresolved':
            action.target_sound = resolution.target_description
            if len(action.target_sounds) == 1:
                action.target_sounds = [resolution.target_description]
    contract = (audio_analysis or {}).get('edit_listening', {})
    if contract.get('participant_selection'):
        if len(actions)!=1:
            raise UnresolvedEditError('Multiple volume actions need explicit participant ownership before binding windows')
        action = actions[0]
        if action.apply_to!='target_source' or not action.effective_target_sounds():
            raise UnresolvedEditError('A selected participant cannot authorize whole-mix volume changes')
        literal = (contract.get('selection') or {}).get('scope')=='literal_time'
        placements = participant_placements(audio_analysis, media_duration,
            literal_windows=[(p.start_time,p.end_time) for p in action.placements] if literal else None)
        ctx.params['participant_scope_binding'] = participant_binding_audit(audio_analysis,
            dict(windows=[(p.start_time,p.end_time) for p in action.placements],
                 temporal_status=action.temporal_status), placements)
        action.placements = placements
        action.temporal_status = 'approximate'
        action.temporal_reason = 'Bound to the selected participant event windows from the listening contract'
    plan.refresh_timing_status()

    return VolumeChangeRunner().run(ctx, plan)
