from __future__ import annotations

from audio_edit.utils.ablation_policy import execution_corrections_enabled

from copy import deepcopy
from dataclasses import asdict

from audio_edit.models import AudioEditContext
from audio_edit.plans.replace_sound_plan import ReplaceSoundPlan
from audio_edit.plans.errors import UnresolvedEditError
from audio_edit.plans.remove_sound_plan import shared_removal_windows
from audio_edit.runners.samaudio_runner import SAMAudioRunner
from audio_edit.runners.add_sound_runner import AddSoundRunner
from audio_edit.utils.media_util import get_media_duration
from audio_edit.utils.execution_outcome import compare_waveforms


class ReplaceSoundRunner:
    """
    Execute replace by composing existing remove + add runners.

    Flow:
        1. Remove source sound.
        2. Use cleaned audio as new input audio.
        3. Add target sound.
    """

    def __init__(self):
        self.remove_runner = SAMAudioRunner()
        self.add_runner = AddSoundRunner()

    def run(self, ctx: AudioEditContext, plan: ReplaceSoundPlan) -> AudioEditContext:
        ctx.params['replacement_plan'] = asdict(plan)


        if not ctx.input_audio:
            raise ValueError("ctx.input_audio is required for replace.")
        if plan.temporal_status == "unresolved":
            raise UnresolvedEditError(
                "Cannot execute temporal replacement request: "
                f"{plan.temporal_reason or 'temporal grounding is unavailable'}"
            )


        if plan.add_plan.temporal_status == 'unresolved':
            raise UnresolvedEditError('Cannot execute temporal add request: '+str(plan.add_plan.temporal_reason))


        removal = plan.remove_plan
        if removal.temporal_status == 'unresolved':
            raise UnresolvedEditError(removal.temporal_reason or 'Removal is unresolved')
        duration = get_media_duration(ctx.input_audio)
        try:
            windows = shared_removal_windows(removal, duration)
        except ValueError as exc:
            raise UnresolvedEditError(str(exc)) from exc
        if [(p.start_time, p.end_time) for p in removal.placements] != windows:
            raise ValueError("Removal placements and execution scope differ")
        if plan.add_plan.intent:
            shared = [e.get('shared_window') for e in plan.add_plan.intent['edits']]
            if any(not isinstance(w,dict) for w in shared) or [
                    (w.get('start_time'),w.get('end_time')) for w in shared] != windows:
                raise ValueError('Add shared windows differ from removal time_span')
        else:

            shared = [(p.start_time,p.end_time) for p in plan.add_plan.placements] or [
                (plan.add_plan.start_time,plan.add_plan.end_time)]
            if shared != windows:
                raise ValueError('Legacy add scopes differ from removal time_span')
        ctx.params['replacement_shared_windows'] = [dict(start=a,end=b) for a,b in windows]

        from audio_edit.analyzers.replacement_residual import inspect_sole_instrument, correct_residual
        from audio_edit.plans.music_backing import decide_backing


        if ctx.params.get('replacement_source_facts') is not None:
            residual_observation = dict(eligible=False, status='existing_first_source_facts_retained', new_observations=0)
        else:
            residual_observation = inspect_sole_instrument(ctx, plan, duration)
        ctx.params['replacement_residual_inspection'] = residual_observation
        analysis = residual_observation.get('analysis')
        if residual_observation.get('status') == 'existing_observation_retained':
            analysis = dict(removal_assessment=dict(windows=[
                deepcopy(window.get('observation', {}))
                for window in plan.remove_plan.removal_evidence['windows']]))
        if isinstance(analysis, dict) and ctx.params.get('replacement_source_facts') is None:
            import json
            from audio_edit.analyzers.removal_focus import file_sha256
            ctx.params['replacement_source_facts'] = dict(audio_sha256=file_sha256(ctx.input_audio),
                text=json.dumps(analysis.get('removal_assessment', {})), first_observation=True,
                origin='first_directed_source_observation')
        if residual_observation.get('eligible'):
            backing_decision = dict(eligible=False, status='confirmed_sole_target_needs_no_complement', new_source_queries=0)
        elif residual_observation.get('status') == 'optional_observation_failed_existing_path_retained':
            backing_decision = dict(eligible=False, status='first_source_observation_failed_original_path_retained', new_source_queries=0)
        else:
            backing_decision = decide_backing(ctx, plan, duration)
        ctx.params['replacement_music_backing_decision'] = backing_decision


        if not residual_observation.get('eligible') and not backing_decision.get('eligible'):
            from audio_edit.analyzers.first_sole_residual import inspect_first_sole
            sole_first = inspect_first_sole(ctx, plan, duration)
            ctx.params['replacement_first_facts_sole_decision'] = sole_first
            if sole_first.get('eligible'):
                residual_observation = sole_first
                ctx.params['replacement_residual_inspection'] = sole_first


        remove_ctx = deepcopy(ctx)
        remove_ctx.user_prompt = plan.remove_user_prompt

        remove_ctx = self.remove_runner.run_remove(remove_ctx, plan.remove_plan)

        if not remove_ctx.output_audio:
            raise RuntimeError("Replace failed: remove step did not produce output_audio.")

        cleaned_audio = remove_ctx.output_audio

        removal_outcome = compare_waveforms(ctx.input_audio, cleaned_audio)
        ctx.params['replacement_removal_outcome'] = removal_outcome
        remove_ctx.params['replacement_removal_outcome'] = removal_outcome
        if removal_outcome['no_op']:
            raise UnresolvedEditError('Removal output is unchanged; replacement cannot be claimed before adding')


        add_ctx = deepcopy(remove_ctx)
        add_ctx.user_prompt = plan.add_user_prompt
        add_ctx.params.pop('replacement_init_noise_level', None)
        removal_execution = remove_ctx.params.get('removal_execution', {})
        if (removal_execution.get('backend') == 'direct_silence'
                and removal_execution.get('target_is_single_instrument') is True):

            add_ctx.params['replacement_init_noise_level'] = 0.4


        if abs(get_media_duration(cleaned_audio)-duration) > 1e-3:
            raise ValueError('Removal changed the timeline; shared add windows cannot be reused')
        if any(e.get("audio_reference") for e in plan.add_plan.intent.get("edits", [])):
            from pathlib import Path
            from tempfile import mkdtemp
            from audio_edit.utils.execution_outcome import removed_component_reference
            root = Path("audio_edit/generated")
            root.mkdir(parents=True, exist_ok=True)
            path = Path(mkdtemp(prefix="replacement-reference-", dir=root)) / "removed.wav"
            audit = removed_component_reference(ctx.input_audio, cleaned_audio, path)
            add_ctx.params["replacement_reference_asset"] = audit["path"]
            add_ctx.params["replacement_reference_audit"] = audit
        residual_correction = None
        if execution_corrections_enabled("replace_residual_revision"):
            cleaned_audio, residual_correction = correct_residual(ctx.input_audio, cleaned_audio, residual_observation)
            if residual_correction is None and not backing_decision.get('eligible'):
                from audio_edit.analyzers.residual_target_subset import inspect_residual_subset
                subset = inspect_residual_subset(ctx, plan, cleaned_audio, duration)
                add_ctx.params['replacement_residual_subset_decision'] = subset
                if ctx.params.get('replacement_residual_first_facts') is not None:
                    add_ctx.params['replacement_residual_first_facts'] = deepcopy(ctx.params['replacement_residual_first_facts'])
                if subset.get('eligible'):
                    cleaned_audio, residual_correction = correct_residual(cleaned_audio, cleaned_audio, subset)
            if residual_correction is not None:
                add_ctx.params['replacement_residual_correction'] = residual_correction
        from audio_edit.tools.complementary_source import retain_music_vocals
        cleaned_audio, backing_execution = retain_music_vocals(ctx.input_audio, cleaned_audio, backing_decision)
        if backing_execution is not None:
            add_ctx.params['replacement_music_backing_execution'] = backing_execution
        add_ctx.input_audio = cleaned_audio
        add_ctx.params['replacement_reference_path'] = ctx.input_audio
        add_ctx.params['replacement_plan'] = __import__('dataclasses').asdict(plan)
        add_ctx.output_audio = None

        from audio_edit.tools.music_timbre import render_replacement
        from audio_edit.tools.continuous_pitch import render_continuous_replacement
        add_ctx.params['replacement_music_render'] = render_continuous_replacement(ctx, plan, add_ctx.params.get('replacement_reference_asset'))
        if add_ctx.params['replacement_music_render'].get('status') == 'not_applicable':
            add_ctx.params['replacement_music_render'] = render_replacement(ctx.input_audio, plan, residual_observation)
        if add_ctx.params['replacement_music_render'].get('status') == 'not_applicable':
            from audio_edit.tools.music_stem_timbre import render_stem_replacement
            add_ctx.params['replacement_music_render'] = render_stem_replacement(ctx, plan, add_ctx.params.get('replacement_reference_asset'))

        from audio_edit.tools.independent_piano import material_renderer
        with material_renderer(self.add_runner, add_ctx, plan):
            final_ctx = self.add_runner.run(add_ctx, plan.add_plan)

        if not final_ctx.output_audio:
            raise RuntimeError("Replace failed: add step did not produce output_audio.")

        if residual_correction is not None:
            from audio_edit.analyzers.replacement_material_guard import retain_residual_if_contradicted
            final_ctx = retain_residual_if_contradicted(final_ctx, plan, self.add_runner.mixer,
                residual_correction['previous_residual'], original_instruction=ctx.user_prompt)


        return final_ctx
