from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy

from audio_edit.models import AudioEditContext
from audio_edit.plans.volume_change_plan import VolumeChangePlan
from audio_edit.plans.errors import UnresolvedEditError
from audio_edit.runners.mixer_runner import MixerRunner
from audio_edit.runners.samaudio_runner import SAMAudioRunner


class VolumeChangeRunner:
    def __init__(self):
        self.mixer = MixerRunner()
        self.samaudio = SAMAudioRunner()

    def run(self, ctx: AudioEditContext, plan: VolumeChangePlan) -> AudioEditContext:
        plan.refresh_timing_status()
        ctx.params["volume_plan"] = asdict(plan)
        ctx.params["temporal_resolution_status"] = plan.temporal_status
        if plan.changes:
            if any(p.changes or p.temporal_status == "unresolved" for p in plan.changes):
                raise UnresolvedEditError("All volume actions must resolve before execution")
            working = deepcopy(ctx)
            steps = []
            for index, action in enumerate(plan.changes):
                working = self.run(working, action)
                if not working.output_audio:
                    raise RuntimeError("Volume action produced no output")
                steps.append(
                    dict(
                        action_index=index,
                        plan=asdict(action),
                        output_audio=working.output_audio,
                        audit=deepcopy(working.params.get("volume_audit", [])),
                        backend=working.params.get("volume_backend"),
                        source_decision=deepcopy(working.params.get("volume_source_decision")),
                    )
                )
                working.input_audio = working.output_audio
            working.input_audio = ctx.input_audio
            working.params["volume_plan"] = asdict(plan)
            working.params["temporal_resolution_status"] = plan.temporal_status
            working.params["volume_steps"] = steps
            return working
        if not ctx.input_audio:
            raise ValueError("ctx.input_audio is required for volume change.")
        if plan.temporal_status == "unresolved":
            raise UnresolvedEditError(
                "Cannot execute temporal volume request: "
                f"{plan.temporal_reason or 'temporal grounding is unavailable'}"
            )

        placements = [asdict(item) for item in plan.placements] or None
        profile_options = (
            {"gain_profile": plan.gain_profile} if plan.gain_profile != "static" else {}
        )
        ctx.params["volume_audit"] = []
        ctx.params.pop("volume_source_decision", None)
        direct_source_gain = False
        if plan.apply_to == "target_source":
            decision = dict(eligible=False, reason="No bound sole-source evidence")
            if plan.source_evidence:
                from audio_edit.analyzers.volume_focus import direct_gain_decision
                from audio_edit.utils.media_util import get_media_duration

                decision = direct_gain_decision(
                    plan, ctx.input_audio, get_media_duration(ctx.input_audio)
                )
            ctx.params["volume_source_decision"] = decision
            direct_source_gain = decision["eligible"]
        if plan.apply_to == "target_source" and not direct_source_gain:
            ctx.params["volume_backend"] = "sam_source_gain"
            targets = plan.effective_target_sounds()
            if not targets:
                raise ValueError("target_source volume change requires target_sounds")
            ctx.output_audio = self.samaudio.run_source_volume(
                original_path=ctx.input_audio,
                target_sounds=targets,
                gain_db=plan.gain_db,
                placements=placements,
                **profile_options,
            )
            ctx.warnings.extend(self.samaudio.last_warnings)
            ctx.params["volume_audit"] = deepcopy(self.samaudio.last_warnings)
            call = getattr(self.samaudio, "last_call", None)
            if isinstance(call, dict):
                ctx.params.setdefault("separation_calls", []).append(deepcopy(call))
        else:
            ctx.params["volume_backend"] = (
                "direct_source_gain" if direct_source_gain else "direct_mix_gain"
            )
            ctx.output_audio = self.mixer.change_volume(
                original_path=ctx.input_audio,
                gain_db=plan.gain_db,
                placements=placements,
                **profile_options,
            )
            warnings = getattr(self.mixer, "last_warnings", [])
            ctx.warnings.extend(warnings if isinstance(warnings, list) else [])
            ctx.params["volume_audit"] = getattr(self.mixer, "last_audit", [])
        return ctx
