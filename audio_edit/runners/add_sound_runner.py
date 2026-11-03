from copy import deepcopy
import hashlib
import os
from dataclasses import asdict
from pathlib import Path
import shutil
import tempfile
from audio_edit.models import AudioEditContext
from audio_edit.generators.stable_audio_generator import StableAudioGenerator
from audio_edit.generators.openai_speech_generator import OpenAISpeechGenerator
from audio_edit.runners.mixer_runner import MixerRunner
from audio_edit.plans.add_sound_plan import AddSoundPlan
from audio_edit.plans.errors import UnresolvedEditError
from audio_edit.utils.media_util import get_media_duration
from audio_edit.utils.bounded_tools import bounded_call, valid_audio
from audio_edit.plans.add_compat import adapt_runner_plan
from audio_edit.plans.add_timing import compile_plan


class AddSoundRunner:
    def __init__(self):
        self.generator_audio = StableAudioGenerator()
        self.generator_speech = OpenAISpeechGenerator()
        self.mixer = MixerRunner()

    def run(self, ctx: AudioEditContext, plan: AddSoundPlan) -> AudioEditContext:
        ctx.params["add_plan"] = asdict(plan)
        if plan.temporal_status == "unresolved":
            raise UnresolvedEditError(
                "Cannot execute temporal add request: " + str(plan.temporal_reason)
            )
        duration = get_media_duration(ctx.input_audio)
        if not plan.intent:
            plan = adapt_runner_plan(plan, duration)
            ctx.warnings.append(
                dict(code="legacy_add_contract", reason="One-way legacy entry adaptation")
            )
        if plan.temporal_status == "unresolved":
            raise UnresolvedEditError(plan.temporal_reason)
        if plan.composer is not None and not isinstance(plan.composer, dict):
            raise ValueError(
                "Executor requires normalized composer settings from the entry adapter"
            )
        placements = deepcopy(plan.execution_placements)
        for p in placements:
            declared = plan.intent["edits"][p["edit_index"]]
            for field in ("generator", "audio_reference"):
                if field in declared and p.get(field) != declared[field]:
                    raise ValueError(f"Compiled placement lost its declared {field}")
        ctx.params["temporal_resolution_status"] = plan.temporal_status
        if plan.temporal_status in ("approximate", "estimated"):
            ctx.warnings.append(
                dict(
                    code=plan.temporal_status + "_timing",
                    reason="Uncertain boundary/default; see anchor evidence and bounded timing review",
                )
            )
        root = Path("./audio_edit/generated")
        root.mkdir(parents=True, exist_ok=True)
        assets = Path(tempfile.mkdtemp(prefix="placements-", dir=root))
        cache = {}
        actual_lengths = {}
        assets_by_key = {}
        for index, p in enumerate(placements):
            declared = plan.intent["edits"][p["edit_index"]]
            material = (
                declared.get("replacement_material")
                if p.get("shared_window") and declared.get("generator") == "audio"
                else None
            )
            independent = material and material.get("mode") in (
                "independent_event",
                "independent_texture",
            )
            complete_event = bool(independent and material["mode"] == "independent_event")
            generation_options = {}
            if complete_event:
                generation_options["complete_event"] = True
            if independent and os.environ.get("REPLACEMENT_AUDIO_SEED") is not None:
                generation_options["seed"] = int(os.environ["REPLACEMENT_AUDIO_SEED"])
            prompt = p.get("audio_prompt") or plan.audio_prompt
            d = p["generation_target_seconds"]
            speech = p.get("generator", "speech" if plan.is_speech else "audio") == "speech"
            speech_text = p.get("speech_text", plan.speech_text)
            voice_prompt = p.get("voice_prompt", plan.voice_prompt)
            reference = {}
            mode = p.get("audio_reference")
            if independent and (mode is not None or prompt != material["target"]):
                raise ValueError("Independent material lost its target-only generation contract")
            if mode:
                if speech:
                    raise ValueError(
                        "Selected generator cannot consume the requested audio reference"
                    )
                reference_path = ctx.params.get("replacement_reference_asset") or ctx.input_audio
                a, b = 0.0, get_media_duration(reference_path)
                if mode == "structure" and p["timeline_op"] == "overlay":
                    a = p.get("window_start", p["start_time"])
                    b = p.get("window_end", p["end_time"])
                if b is None or not 0 <= a < b <= get_media_duration(reference_path) + 1e-4:
                    raise ValueError(
                        "Requested structural reference needs a complete in-file scope"
                    )
                reference = dict(
                    reference_path=reference_path,
                    reference_start=a,
                    reference_end=b,
                    reference_mode=mode,
                )
                if (
                    mode == "structure"
                    and ctx.params.get("replacement_init_noise_level") is not None
                ):
                    reference["init_noise_level"] = ctx.params["replacement_init_noise_level"]
                ctx.params.setdefault("generation_references", []).append(
                    dict(
                        key=p["key"],
                        **dict(
                            reference,
                            init_noise_level=reference.get(
                                "init_noise_level", 0.55 if mode == "structure" else 0.85
                            ),
                        ),
                        source="removed_component"
                        if ctx.params.get("replacement_reference_asset")
                        else "original_input",
                        quality="unreviewed",
                        timing_authority="unchanged add intent",
                    )
                )
            generation_input = dict(
                key=p["key"],
                generator="speech" if speech else "audio",
                prompt=prompt,
                duration=d,
                speech_text=speech_text if speech else None,
                audio_reference=mode,
                reference=deepcopy(reference),
                **generation_options,
            )
            if reference:
                generation_input["reference_sha256"] = hashlib.sha256(
                    Path(reference["reference_path"]).read_bytes()
                ).hexdigest()
            music_render = ctx.params.get("replacement_music_render", {})
            use_music_render = (
                music_render.get("status") == "rendered"
                and len(plan.intent["edits"]) == 1
                and mode == "structure"
                and not speech
            )
            if use_music_render:
                generation_input.update(
                    backend=music_render.get("backend", "basic_pitch_soundfont"),
                    tool_input=music_render["input_audio"],
                    tool_input_sha256=music_render["input_sha256"],
                    program=music_render["program"],
                )
            ctx.params.setdefault("generation_inputs", []).append(generation_input)

            if d is None and not speech:
                raise ValueError("The selected generator requires an explicit generation target")
            key = (
                prompt,
                d,
                speech,
                speech_text,
                voice_prompt,
                tuple(sorted(reference.items())),
                tuple(sorted(generation_options.items())),
            )
            if key not in cache:

                def generate():
                    if use_music_render:
                        return music_render["output_audio"]
                    if speech:
                        return self.generator_speech.generate(
                            text=speech_text, voice_prompt=voice_prompt
                        )
                    return self.generator_audio.generate(
                        prompt=prompt,
                        negative_prompt=plan.negative_audio_prompt,
                        duration=d,
                        **reference,
                        **generation_options,
                    )

                generated = bounded_call(
                    ctx,
                    "generate_placement_" + str(index),
                    generate,
                    validate=valid_audio,
                )
                if ctx.params.get("replacement_plan") and not use_music_render:
                    from audio_edit.policies.material import clean_generated_material

                    generated = clean_generated_material(ctx, plan, p, generated)
                asset = assets / (str(index) + ".wav")
                if complete_event:
                    from audio_edit.tools.event_material import first_event_asset

                    raw_asset = assets / (str(index) + ".decoded.wav")
                    shutil.copy2(generated, raw_asset)
                    preparation = first_event_asset(raw_asset, asset, d)
                    ctx.params.setdefault("material_preparations", []).append(
                        dict(key=p["key"], **preparation)
                    )
                else:
                    shutil.copy2(generated, asset)
                cache[key] = str(asset.resolve())
            assets_by_key[p["key"]] = cache[key]
            actual_lengths[p["key"]] = get_media_duration(cache[key])
        ctx.params["generation_targets"] = [
            dict(key=p["key"], target_seconds=p["generation_target_seconds"]) for p in placements
        ]
        ctx.params["actual_asset_lengths"] = actual_lengths
        from audio_edit.plans.replacement_extent import replacement_extent_limits

        extent_limits = replacement_extent_limits(plan, ctx.params.get("remove_plan"), duration)
        if extent_limits:
            ctx.params["replacement_extent_limits"] = deepcopy(extent_limits)
        from audio_edit.tools.event_fit import fit_replacement_assets

        assets_by_key, actual_lengths, fit_audit = fit_replacement_assets(
            plan, placements, assets_by_key, actual_lengths, extent_limits, assets
        )
        if fit_audit:
            ctx.params["replacement_event_fit"] = fit_audit
            ctx.params["actual_asset_lengths"] = deepcopy(actual_lengths)
        plan = compile_plan(
            plan,
            duration=duration,
            asset_durations=actual_lengths,
            replacement_extent_limits=extent_limits,
        )
        ctx.params["add_plan"] = asdict(plan)
        ctx.params["add_intent"] = deepcopy(plan.intent)
        ctx.params["temporal_resolution_status"] = plan.temporal_status
        ctx.params["anchor_audit"] = plan.edit_windows
        if plan.temporal_status == "unresolved":
            from audio_edit.plans.add_recovery import enabled, fit_existing_assets

            if not enabled(ctx) or any(p.get("shared_window") for p in placements):
                raise UnresolvedEditError(
                    "Actual material is infeasible: " + str(plan.temporal_reason)
                )
            plan, assets_by_key, actual_lengths = fit_existing_assets(
                ctx, plan, placements, assets_by_key, actual_lengths, duration, plan.temporal_reason
            )
            ctx.params.update(
                add_plan=asdict(plan),
                add_intent=deepcopy(plan.intent),
                temporal_resolution_status=plan.temporal_status,
                anchor_audit=plan.edit_windows,
                actual_asset_lengths=actual_lengths,
            )
        placements = deepcopy(plan.execution_placements)
        for p in placements:
            if p["compiled_phase"] != "final":
                raise ValueError("Executor requires final material-derived timing")
            p["background_path"] = assets_by_key[p["key"]]
        features = dict(plan.composer or {})
        if plan.layer_role:
            features["layer_role"] = plan.layer_role
        if ctx.params.get("replacement_reference_path"):
            features["replacement_reference_path"] = ctx.params["replacement_reference_path"]
        ctx.params["executed_placements"] = placements
        ctx.output_audio = bounded_call(
            ctx,
            "render_add_placements",
            lambda: self.mixer.mix_background(
                original_path=ctx.input_audio,
                background_path=placements[0]["background_path"],
                start_time=plan.start_time,
                end_time=plan.end_time,
                acoustic_feature=features,
                placements=placements,
            ),
            validate=valid_audio,
        )
        ctx.params["mix_audit"] = getattr(self.mixer, "last_audit", [])
        ctx.warnings.extend(getattr(self.mixer, "last_warnings", []))
        return ctx
