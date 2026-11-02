from __future__ import annotations

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
        add_ctx.input_audio = cleaned_audio
        add_ctx.params['replacement_reference_path'] = ctx.input_audio
        add_ctx.params['replacement_plan'] = __import__('dataclasses').asdict(plan)
        add_ctx.output_audio = None

        final_ctx = self.add_runner.run(add_ctx, plan.add_plan)

        if not final_ctx.output_audio:
            raise RuntimeError("Replace failed: add step did not produce output_audio.")


        return final_ctx
