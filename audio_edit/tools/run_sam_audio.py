from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from sam_audio import SAMAudio, SAMAudioProcessor

from audio_edit.plans.remove_sound_plan import RemoveSoundPlan, TimeSpan, validate_result_mode


DEFAULT_MODEL_ID = "facebook/sam-audio-large"
RERANKING_CANDIDATES = 8
MIN_TARGET_TO_MIX_DB = -30.0


class RealSAMAudioRemoveRunner:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        output_dir: str = "./audio_edit/outputs",
    ):
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

        self.model = None
        self.processor = None
        self.diagnostics: list[dict] = []
        self.inference_calls: list[dict] = []

    def _lazy_load(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        self.model = SAMAudio.from_pretrained(self.model_id).eval()
        self.model = self.model.to(device=self.device, dtype=self.dtype).eval()

        self.processor = SAMAudioProcessor.from_pretrained(self.model_id)

    def _separate(self, inputs):
        """Use SAM's candidate ranking and predict missing text-prompt anchors."""
        visual = inputs.masked_video is not None
        provided_anchors = inputs.anchors is not None
        predict_spans = not visual and not provided_anchors
        ranker_name = "visual_ranker" if visual else "text_ranker"
        ranker = getattr(self.model, ranker_name, None)
        if ranker is None:
            raise RuntimeError(
                f"SAM Audio {RERANKING_CANDIDATES}-candidate separation requires "
                f"{ranker_name}; refusing to silently select the first candidate"
            )
        if predict_spans and getattr(self.model, "span_predictor", None) is None:
            raise RuntimeError("SAM Audio automatic span prediction requires span_predictor")
        settings = {
            "model_id": self.model_id,
            "reranking_candidates": RERANKING_CANDIDATES,
            "predict_spans": predict_spans,
            "provided_anchors": provided_anchors,
            "ranker": type(ranker).__name__,
        }
        self.inference_calls.append(settings)

        with torch.inference_mode():
            return self.model.separate(
                inputs,
                predict_spans=predict_spans,
                reranking_candidates=RERANKING_CANDIDATES,
            )

    def run(self, input_audio: str, plan: RemoveSoundPlan) -> str:
        validate_result_mode(plan.result_mode)
        if plan.mode == "text":
            return self._run_text_mode(input_audio, plan)

        if plan.mode == "span":
            return self._run_span_mode(input_audio, plan)

        if plan.mode == "video":
            return self._run_video_mode(input_audio, plan)

        raise ValueError(f"Unsupported mode: {plan.mode}")

    def _run_text_mode(self, input_audio: str, plan: RemoveSoundPlan) -> str:
        input_path = Path(input_audio)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")

        text_prompt = plan.text_prompt.strip().lower()
        if not text_prompt:
            raise ValueError("text_prompt is empty.")

        self._lazy_load()

        inputs = self.processor(
            audios=[str(input_path)],
            descriptions=[text_prompt],
        ).to(self.device)

        result = self._separate(inputs)

        return self._save_result(
            input_path=input_path,
            result=result,
            result_mode=plan.result_mode,
            name_suffix=self._safe_name(text_prompt),
        )

    def _run_span_mode(self, input_audio: str, plan: RemoveSoundPlan) -> str:
        input_path = Path(input_audio)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")

        if not plan.time_span:
            raise ValueError("span mode requires plan.time_span")

        text_prompt = plan.text_prompt.strip().lower()
        if not text_prompt:
            raise ValueError("text_prompt is empty.")

        self._lazy_load()

        anchors = [[["+", span.start, span.end] for span in plan.time_span]]

        inputs = self.processor(
            audios=[str(input_path)],
            descriptions=[text_prompt],
            anchors=anchors,
        ).to(self.device)

        result = self._separate(inputs)

        if len(plan.time_span) == 1:
            span_name = f"{plan.time_span[0].start:g}_{plan.time_span[0].end:g}"
        else:
            span_name = f"{len(plan.time_span)}spans"

        return self._save_result(
            input_path=input_path,
            result=result,
            result_mode=plan.result_mode,
            name_suffix=f"{self._safe_name(text_prompt)}_{span_name}",
            spans=plan.time_span,
        )

    def _run_video_mode(self, input_audio: str, plan: RemoveSoundPlan) -> str:
        input_media = Path(input_audio)

        if not input_media.exists():
            raise FileNotFoundError(f"Input media does not exist: {input_media}")

        if input_media.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm"}:
            raise ValueError(f"SAMAudio video mode requires video input, got: {input_media}")

        if not plan.visual_prompt:
            raise ValueError("video mode requires plan.visual_prompt")

        self._lazy_load()

        from audio_edit.vision.sam3_mask_provider import SAM3MaskProvider

        mask_provider = SAM3MaskProvider()

        mask_result = mask_provider.generate_mask(
            video_path=str(input_media),
            visual_prompt=plan.visual_prompt,
        )

        masked_videos = self.processor.mask_videos(
            [mask_result.frames],
            [mask_result.mask],
        )

        inputs = self.processor(
            audios=[str(input_media)],
            descriptions=[""],
            masked_videos=masked_videos,
        ).to(self.device)

        result = self._separate(inputs)

        safe_audio_prompt = self._safe_name(plan.text_prompt)
        safe_visual_prompt = self._safe_name(plan.visual_prompt)

        return self._save_result(
            input_path=input_media,
            result=result,
            result_mode=plan.result_mode,
            name_suffix=f"{safe_audio_prompt}_{safe_visual_prompt}_video",
        )

    def run_source_volume(
        self,
        input_audio: str,
        target_sound: str | None = None,
        gain_db: float = 0.0,
        placements: list[dict] | None = None,
        target_sounds: list[str] | None = None,
        gain_profile: str = "static",
    ) -> str:
        """Apply gain to independently separated sources, preserving the input."""
        self.diagnostics = []

        input_path = Path(input_audio)
        if not input_path.is_file():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")
        targets = [item.strip().lower() for item in (target_sounds or []) if item.strip()]
        if not targets and target_sound and target_sound.strip():
            targets = [target_sound.strip().lower()]
        targets = list(dict.fromkeys(targets))
        if not targets:
            raise ValueError("target_sounds is empty")
        gain_db = float(gain_db)
        if not -24.0 <= gain_db <= 24.0:
            raise ValueError("gain_db must be between -24 and 24 dB")

        anchors = None
        if placements:
            anchors = [[]]
            for placement in placements:
                start = float(placement["start_time"])
                raw_end = placement.get("end_time")
                if raw_end is None:
                    raise ValueError(f"Target-source volume placement needs end_time: {placement}")
                end = float(raw_end)
                if start < 0 or end <= start:
                    raise ValueError(f"Invalid target-source volume placement: {placement}")
                anchors[0].append(["+", start, end])

        self._lazy_load()

        original, original_sample_rate = torchaudio.load(str(input_path))
        separated_sample_rate = int(self.processor.audio_sampling_rate)
        factor = math.pow(10.0, gain_db / 20.0)

        delta = torch.zeros_like(original)
        mask = None
        if placements:
            mask = torch.zeros(original.shape[1], dtype=original.dtype)
            for placement in placements:
                start = max(0, int(round(float(placement["start_time"]) * original_sample_rate)))
                end = min(
                    original.shape[1],
                    int(round(float(placement["end_time"]) * original_sample_rate)),
                )
                mask[start:end] = 1.0

        scope = mask.bool() if mask is not None else torch.ones(original.shape[1], dtype=torch.bool)
        if not torch.any(scope):
            raise RuntimeError("Volume edit has no audio samples in its target scope")
        original_scope_rms = torch.sqrt(torch.mean(original[:, scope] ** 2)).item()
        if not math.isfinite(original_scope_rms) or original_scope_rms <= 1e-8:
            raise RuntimeError("Volume edit target scope is silent or invalid")
        for target_name in targets:
            processor_args = {
                "audios": [str(input_path)],
                "descriptions": [target_name],
            }
            if anchors is not None:
                processor_args["anchors"] = anchors
            inputs = self.processor(**processor_args).to(self.device)
            result = self._separate(inputs)

            target = result.target[0].detach().cpu().float()
            if target.ndim == 1:
                target = target.unsqueeze(0)
            if separated_sample_rate != original_sample_rate:
                target = torchaudio.functional.resample(
                    target, separated_sample_rate, original_sample_rate
                )
            if target.shape[0] != original.shape[0]:
                if target.shape[0] == 1:
                    target = target.repeat(original.shape[0], 1)
                elif original.shape[0] == 1:
                    target = target.mean(dim=0, keepdim=True)
                else:
                    raise RuntimeError(
                        "Cannot align SAMAudio target channels to original: "
                        f"{target.shape[0]} vs {original.shape[0]}"
                    )
            if target.shape[1] < original.shape[1]:
                target = torch.nn.functional.pad(target, (0, original.shape[1] - target.shape[1]))
            else:
                target = target[:, : original.shape[1]]

            initial_rms = torch.sqrt(torch.mean(target[:, scope] ** 2)).item()
            if gain_profile == "level" and initial_rms < original_scope_rms * 10 ** (
                MIN_TARGET_TO_MIX_DB / 20
            ):
                words = target_name.split()
                singular = (
                    " ".join(words[:-1] + [words[-1][:-1]])
                    if words[-1].endswith("s") and not words[-1].endswith("ss")
                    else target_name
                )
                alternatives = list(dict.fromkeys([singular, "The sound of " + singular]))
                best_rms = initial_rms
                for description in alternatives:
                    if description == target_name:
                        continue
                    retry_args = {**processor_args, "descriptions": [description]}
                    retry_inputs = self.processor(**retry_args).to(self.device)
                    retry_result = self._separate(retry_inputs)
                    candidate = retry_result.target[0].detach().cpu().float()
                    if candidate.ndim == 1:
                        candidate = candidate.unsqueeze(0)
                    if separated_sample_rate != original_sample_rate:
                        candidate = torchaudio.functional.resample(
                            candidate, separated_sample_rate, original_sample_rate
                        )
                    if candidate.shape[0] == 1 and original.shape[0] > 1:
                        candidate = candidate.repeat(original.shape[0], 1)
                    elif original.shape[0] == 1 and candidate.shape[0] > 1:
                        candidate = candidate.mean(dim=0, keepdim=True)
                    candidate = torch.nn.functional.pad(
                        candidate, (0, max(0, original.shape[1] - candidate.shape[1]))
                    )[:, : original.shape[1]]
                    if not torch.isfinite(candidate).all():
                        continue
                    rms = torch.sqrt(torch.mean(candidate[:, scope] ** 2)).item()

                    if rms > best_rms:
                        target, best_rms = candidate, rms
                self.diagnostics.append(
                    {
                        "code": "separation_retry",
                        "target_sound": target_name,
                        "descriptions": alternatives,
                        "initial_target_to_mix_db": 20
                        * math.log10(max(initial_rms / original_scope_rms, 1e-12)),
                        "selected_target_to_mix_db": 20
                        * math.log10(max(best_rms / original_scope_rms, 1e-12)),
                    }
                )

            target_scope_rms = torch.sqrt(torch.mean(target[:, scope] ** 2)).item()
            if not torch.isfinite(target).all():
                raise RuntimeError(f"SAMAudio returned non-finite samples for {target_name!r}")
            ratio = target_scope_rms / max(original_scope_rms, 1e-12)
            ratio_db = 20.0 * math.log10(max(ratio, 1e-12))

            if ratio_db < MIN_TARGET_TO_MIX_DB:
                warning = {
                    "code": "weak_separation",
                    "target_sound": target_name,
                    "target_to_mix_db": ratio_db,
                    "threshold_db": MIN_TARGET_TO_MIX_DB,
                }
                self.diagnostics.append(warning)
                print("[run_samaudio_volume] WARNING: weak target; returning actual edit:", warning)

            if gain_profile == "static":
                target_delta = (factor - 1.0) * target
            else:
                from audio_edit.tools.gain_envelope import gain_envelope

                envelope = gain_envelope(
                    target,
                    original_sample_rate,
                    gain_db=gain_db,
                    profile=gain_profile,
                    placements=placements,
                )
                target_delta = (envelope - 1.0) * target

            if mask is not None:
                target_delta = target_delta * mask.unsqueeze(0)
            delta = delta + target_delta

        from audio_edit.tools.gain_envelope import apply_volume_delta, volume_scope_envelope

        if placements:
            transition = volume_scope_envelope(
                original.shape[-1],
                original_sample_rate,
                placements,
                gain_profile,
                device=original.device,
                dtype=original.dtype,
            )
            delta *= transition
        output, actual_delta_scale = apply_volume_delta(
            original, delta, self.diagnostics, "source_volume"
        )
        self.diagnostics.append(
            {
                "code": "source_volume_execution",
                "requested_gain_db": gain_db,
                "actual_delta_scale": actual_delta_scale,
                "gain_profile": gain_profile,
                "delta_scale_kind": "energy_ratio",
                "static_effective_gain_db": gain_db if gain_profile == "static" else None,
                "static_core_gain_db_range": [gain_db, gain_db]
                if gain_profile == "static"
                else None,
                "normalization_applied": False,
                "output_policy": "preserve_requested_gain_float_wav",
                "allowed_ranges": placements or "whole_input",
                "transition_seconds": 0.005
                if placements and gain_profile in {"static", "level"}
                else 0.0,
                "edge_policy": "preserve_requested_fade_and_mute_endpoints",
            }
        )
        suffix = (
            f"gain_{gain_db:+.1f}dB".replace("+", "plus").replace("-", "minus").replace(".", "p")
        )
        target_label = "_and_".join(targets)
        if gain_profile != "static":
            suffix += "_" + gain_profile
        output_path = self.output_dir / (
            f"{input_path.stem}_{self._safe_name(target_label)}_{suffix}.wav"
        )
        torchaudio.save(
            str(output_path), output, original_sample_rate, encoding="PCM_F", bits_per_sample=32
        )

        return str(output_path)

    def _save_result(
        self, input_path: Path, result, result_mode: str, name_suffix: str, spans=None
    ) -> str:
        sample_rate = int(self.processor.audio_sampling_rate)

        if result_mode == "residual":
            output_path = self.output_dir / f"{input_path.stem}_{name_suffix}_residual.wav"
            waveform = result.residual[0].unsqueeze(0).detach().cpu().float()

        elif result_mode == "result":
            output_path = self.output_dir / f"{input_path.stem}_{name_suffix}_result.wav"
            waveform = result.target[0].unsqueeze(0).detach().cpu().float()

        else:
            raise ValueError(f"Unsupported result_mode: {result_mode}")

        if result_mode == "result" and not spans:
            original, original_sr = torchaudio.load(str(input_path))
            from audio_edit.tools.separation_alignment import target_alignment

            alignment = target_alignment(
                int(original.shape[-1]),
                int(original_sr),
                int(waveform.shape[-1]),
                sample_rate,
                int(getattr(self.processor, "audio_hop_length", 1)),
            )
            waveform = waveform[..., : alignment["output_samples"]]
            self.diagnostics.append(alignment)

        if spans or result_mode == "residual":
            original, original_sr = torchaudio.load(str(input_path))
            if waveform.ndim == 3:
                waveform = waveform.squeeze(0)
            if sample_rate != original_sr:
                waveform = torchaudio.functional.resample(waveform, sample_rate, original_sr)
            if waveform.shape[0] == 1 and original.shape[0] > 1:
                waveform = waveform.repeat(original.shape[0], 1)
            elif original.shape[0] == 1 and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if waveform.shape[-1] < original.shape[-1]:
                waveform = torch.nn.functional.pad(
                    waveform, (0, original.shape[-1] - waveform.shape[-1])
                )
            waveform = waveform[:, : original.shape[-1]]
            from audio_edit.tools.scene_mix import bounded_delta, scope_envelope

            mask = (
                scope_envelope(
                    original.shape[-1],
                    original_sr,
                    [{"start_time": span.start, "end_time": span.end} for span in spans],
                    device=original.device,
                    dtype=original.dtype,
                )
                if spans
                else torch.ones(original.shape[-1], dtype=original.dtype)
            )
            waveform, scale = bounded_delta(
                original,
                (waveform - original) * mask,
                self.diagnostics,
                "local_removal",
                sample_rate=original_sr,
            )
            self.diagnostics.append(
                {
                    "code": "removal_alignment",
                    "seconds": 0.005 if spans else 0,
                    "policy": "original sample count/rate; crossfade only inside local intervals",
                    "delta_scale": scale,
                    "original_samples": original.shape[-1],
                    "output_samples": waveform.shape[-1],
                }
            )
            sample_rate = original_sr
        torchaudio.save(str(output_path), waveform, sample_rate)

        return str(output_path)

    def _safe_name(self, text: str) -> str:
        safe = "".join(c if c.isalnum() else "_" for c in text.lower())
        safe = "_".join(part for part in safe.split("_") if part)
        return safe[:60] or "prompt"


def load_plan(path: str) -> tuple[str, str, RemoveSoundPlan]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    input_audio = payload["input_audio"]
    output_dir = payload["output_dir"]
    plan_data = payload["plan"]

    time_span = None
    if plan_data.get("time_span") is not None:
        time_span = [
            TimeSpan(
                start=float(span["start"]),
                end=float(span["end"]),
            )
            for span in plan_data["time_span"]
        ]

    plan = RemoveSoundPlan(
        backend=plan_data["backend"],
        mode=plan_data["mode"],
        result_mode=plan_data["result_mode"],
        text_prompt=plan_data["text_prompt"],
        time_span=time_span,
        use_video_prompt=bool(plan_data.get("use_video_prompt", False)),
        visual_prompt=plan_data.get("visual_prompt"),
        notes=plan_data.get("notes"),
    )

    return input_audio, output_dir, plan


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--plan-json")
    source.add_argument("--volume-json")
    parser.add_argument("--result-json", required=True)
    parser.add_argument(
        "--model-id",
        default=os.getenv("SAMAUDIO_MODEL_ID", DEFAULT_MODEL_ID),
    )
    args = parser.parse_args()

    if args.plan_json:
        input_audio, output_dir, plan = load_plan(args.plan_json)
        runner = RealSAMAudioRemoveRunner(
            model_id=args.model_id,
            output_dir=output_dir,
        )
        output_audio = runner.run(input_audio=input_audio, plan=plan)
    else:
        with open(args.volume_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("operation") != "source_volume":
            raise ValueError("volume JSON operation must be source_volume")
        runner = RealSAMAudioRemoveRunner(
            model_id=args.model_id,
            output_dir=payload["output_dir"],
        )
        output_audio = runner.run_source_volume(
            input_audio=payload["input_audio"],
            target_sound=payload.get("target_sound"),
            target_sounds=payload.get("target_sounds"),
            gain_db=payload["gain_db"],
            placements=payload.get("placements"),
            gain_profile=payload.get("gain_profile", "static"),
        )

    result_data = {
        "output_audio": output_audio,
        "warnings": runner.diagnostics,
        "sam_audio_inference": runner.inference_calls,
    }

    with open(args.result_json, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)


if __name__ == "__main__":
    main()
