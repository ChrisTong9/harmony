from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import torch
import torchaudio
import torchaudio.functional as F
from audio_edit.tools.scene_mix import (
    active_rms,
    bounded_delta,
    edge_envelope,
    finite_audio,
    prepare_snippet,
    render_timeline,
    scene_gain,
    scope_envelope,
)
from audio_edit.tools.sample_placement import placement_sample_bounds


def pad_to_length(audio, target_len):
    return torch.nn.functional.pad(audio[:, :target_len], (0, max(0, target_len - audio.shape[-1])))


def match_channels(audio, target_channels):
    if audio.shape[0] == target_channels:
        return audio
    if target_channels == 1:
        return audio.mean(0, keepdim=True)
    return audio.repeat(math.ceil(target_channels / audio.shape[0]), 1)[:target_channels]


def apply_fade(audio, sample_rate, fade_in=0.02, fade_out=0.02):
    audio = audio.clone()
    for side, seconds in [("in", fade_in), ("out", fade_out)]:
        n = min(round(seconds * sample_rate), audio.shape[-1] // 2)
        if n:
            ramp = torch.linspace(0, 1, n, device=audio.device, dtype=audio.dtype)
            if side == "in":
                audio[:, :n] *= ramp
            else:
                audio[:, -n:] *= ramp.flip(0)
    return audio


def apply_offscreen_muffle(audio, sample_rate):
    return F.lowpass_biquad(audio, sample_rate, cutoff_freq=min(1000.0, sample_rate * 0.4))


def normalize_generated_audio(audio, *, target_rms_dbfs, max_gain_db=30.0, peak_ceiling=0.95):
    """Legacy utility; the scene mixer separately applies an active-sound floor."""
    finite_audio(audio)
    if not math.isfinite(target_rms_dbfs) or not -60 <= target_rms_dbfs <= -6:
        raise ValueError("target_rms_dbfs outside [-60,-6]")
    if not math.isfinite(max_gain_db) or not 0 <= max_gain_db <= 40 or not 0 < peak_ceiling <= 1:
        raise ValueError("Invalid normalization bounds")
    rms, peak = float(audio.square().mean().sqrt()), float(audio.abs().max())
    if min(rms, peak) < 1e-6:
        return audio
    return audio * min(
        10 ** (target_rms_dbfs / 20) / rms, 10 ** (max_gain_db / 20), peak_ceiling / peak
    )


def apply_synthetic_reverb(audio, sample_rate, room_type):
    if room_type not in ("room", "hall"):
        return audio
    decay = 0.8 if room_type == "room" else 2.5
    n = round(sample_rate * decay)
    t = torch.linspace(0, decay, n, device=audio.device)
    rng = torch.Generator(device=audio.device).manual_seed(0)
    ir = torch.randn(1, n, generator=rng, device=audio.device) * torch.exp(-t * (6 / decay))
    ir = F.lowpass_biquad(ir, sample_rate, cutoff_freq=min(3000.0, sample_rate * 0.4))
    ir /= ir.square().sum().sqrt().clamp_min(1e-12)
    wet = F.fftconvolve(audio, ir.expand(audio.shape[0], -1))[:, : audio.shape[-1]]
    ratio = 0.2 if room_type == "room" else 0.4
    return audio * (1 - ratio) + wet * ratio


def apply_ducking(original, start_sample, end_sample, ducking_ratio, sample_rate):
    """Compatibility explicit opt-in only, transitions INSIDE the permitted span."""
    output = original.clone()
    envelope = edge_envelope(
        end_sample - start_sample,
        sample_rate,
        seconds=0.05,
        device=original.device,
        dtype=original.dtype,
    )
    output[:, start_sample:end_sample] *= 1 + (ducking_ratio - 1) * envelope
    return output


def save_with_audit(path, audio, sr, audit):
    finite_audio(audio)
    torchaudio.save(str(path), audio, sr, encoding="PCM_F", bits_per_sample=32)
    Path(str(path) + ".audit.json").write_text(json.dumps(audit, indent=2))
    return str(path)


def apply_gain_change(original_path, output_dir, gain_db, placements=None, gain_profile="static"):
    source = Path(original_path).resolve()
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    audio, sr = torchaudio.load(str(source))
    if not math.isfinite(gain_db) or not -24 <= gain_db <= 24:
        raise ValueError("Gain must be finite and within [-24,24] dB")
    from audio_edit.tools.gain_envelope import (
        gain_envelope,
        apply_volume_delta,
        volume_scope_envelope,
    )

    mask = volume_scope_envelope(
        audio.shape[-1], sr, placements, gain_profile, device=audio.device, dtype=audio.dtype
    )
    if gain_profile == "static":
        envelope = torch.full_like(mask, 10 ** (gain_db / 20))
    else:
        envelope = gain_envelope(
            audio, sr, gain_db=gain_db, profile=gain_profile, placements=placements
        )
    delta = audio * (envelope - 1) * mask
    audit = []
    output, scale = apply_volume_delta(audio, delta, audit, "volume_delta")
    nonzero = audio.abs() > 1e-8
    effective = output[nonzero] / audio[nonzero] if nonzero.any() else torch.ones(1)
    audit.append(
        dict(
            code="volume_execution",
            requested_gain_db=gain_db,
            profile=gain_profile,
            actual_gain_min_db=20 * math.log10(max(float(effective.min()), 1e-12)),
            actual_gain_max_db=20 * math.log10(max(float(effective.max()), 1e-12)),
            delta_scale=scale,
            delta_scale_kind="energy_ratio",
            normalization_applied=False,
            output_policy="preserve_requested_gain_float_wav",
            allowed_ranges=placements or "whole_input",
            transition_seconds=0.005 if placements and gain_profile in {"static", "level"} else 0.0,
            edge_policy="preserve_requested_fade_and_mute_endpoints",
        )
    )
    return save_with_audit(root / (source.stem + "_gain.wav"), output, sr, audit)


def mix_background(
    original_path,
    background_path,
    output_dir,
    start_time=0.0,
    end_time=None,
    placements=None,
    acoustic_features=None,
):
    source = Path(original_path).resolve()
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    original, sr = torchaudio.load(str(source))
    features = dict(acoustic_features or {})
    audit, prepared = [], []
    raw = placements or [dict(start_time=start_time, end_time=end_time)]
    reference_original = None
    if features.get("replacement_reference_path"):
        reference_original, rsr = torchaudio.load(features["replacement_reference_path"])
        if rsr != sr:
            reference_original = F.resample(reference_original, rsr, sr)
        reference_original = match_channels(reference_original, original.shape[0])
        reference_original = pad_to_length(reference_original, original.shape[-1])
    for index, item in enumerate(raw):
        p = dict(item)
        if p.get("contract_version") == 2:
            if p.get("compiled_phase") != "final" or any(
                p.get(k) is None
                for k in ("start_time", "end_time", "timeline_op", "overflow_policy")
            ):
                raise ValueError("Add v2 renderer requires final explicit execution parameters")
        asset = p.get("background_path") or background_path
        audio, gsr = torchaudio.load(str(asset))
        native_frames = audio.shape[-1]
        if gsr != sr:
            audio = F.resample(audio, gsr, sr)
        audio = match_channels(audio, original.shape[0])
        local_audit = []
        start, end = placement_sample_bounds(
            p,
            native_frames=native_frames,
            native_rate=gsr,
            rendered_frames=audio.shape[-1],
            sample_rate=sr,
            diagnostics=local_audit,
        )
        p["start_time"] = start / sr
        p["end_time"] = end / sr
        if "timeline_op" not in p:
            p["timeline_op"] = "overlay"
            p["overflow_policy"] = "extend" if end > original.shape[-1] else "error"
            audit.append(
                dict(
                    code="legacy_timeline_default",
                    index=index,
                    policy=p["overflow_policy"],
                    reason="Legacy placement API; no automatic duration truncation",
                )
            )
        per = dict(features)
        if p.get("layer_role") is not None:
            per["layer_role"] = p["layer_role"]

        if "relative_level_db" in p:
            per["relative_level_db"] = p["relative_level_db"]
        if "audio_reference" in p:
            per["audio_reference"] = p["audio_reference"]
        per["default_replacement_background"] = bool(
            reference_original is not None
            and "relative_level_db" in per
            and per["relative_level_db"] is None
            and per.get("layer_role") == "background"
            and p.get("coverage") in ("continuous", "single")
            and p["timeline_op"] == "overlay"
        )
        if per.get("minimum_added_rms_dbfs") is not None:
            p["minimum_added_rms_dbfs"] = per["minimum_added_rms_dbfs"]
        audio = prepare_snippet(audio, sr, end - start, p.get("coverage", "single"), local_audit)
        if per.get("apply_offscreen_muffle", False):
            audio = apply_offscreen_muffle(audio, sr)
        if per.get("reverb_type") in ("room", "hall"):
            audio = apply_synthetic_reverb(audio, sr, per["reverb_type"])
            local_audit.append(
                dict(
                    code="effect_tail_scope",
                    policy="tail contained and faded inside placement",
                    allowed_start=start / sr,
                    allowed_end=end / sr,
                )
            )
        audio = apply_fade(audio, sr)
        target = None
        if reference_original is not None:
            target = (reference_original - original)[:, start : min(end, original.shape[-1])]
        audio = scene_gain(audio, original, sr, start, end, per, local_audit, reference=target)

        p["minimum_added_rms_dbfs"] = local_audit[-1]["minimum_added_rms_dbfs"]
        gain_audit = local_audit[-1]
        if gain_audit.get("replacement_background_default"):
            p["headroom_policy"] = "default_replacement_background_local_delta"
        p["level_target"] = dict(
            active_rms=gain_audit["requested_active_rms"],
            reference_active_rms=gain_audit["reference_active_rms"],
            relative_level_db=gain_audit["relative_level_db"],
            basis=gain_audit["basis"],
        )
        if (
            reference_original is None
            and "relative_level_db" in per
            and per["relative_level_db"] is None
            and per.get("audio_reference") == "context"
            and p.get("coverage") == "continuous"
            and p["timeline_op"] == "overlay"
        ):
            p["headroom_policy"] = "default_context_local_delta"
        if per.get("apply_ducking"):
            local_audit.append(
                dict(
                    code="original_ducking_not_applied",
                    reason="Explicit ducking is ignored; original gain is preserved",
                )
            )
        audit.append(
            dict(
                code="prepared_placement",
                index=index,
                asset=str(asset),
                placement=p,
                measurements=local_audit,
            )
        )
        prepared.append((p, audio))
    output = render_timeline(original, sr, prepared, audit)
    return save_with_audit(root / (source.stem + "_mixed.wav"), output, sr, audit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--response-json", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.request_json).read_text())
    if payload.get("operation") == "volume":
        output = apply_gain_change(
            payload["original_path"],
            payload.get("output_dir", "./output"),
            float(payload["gain_db"]),
            payload.get("placements"),
            payload.get("gain_profile", "static"),
        )
    else:
        output = mix_background(
            payload["original_path"],
            payload["background_path"],
            payload.get("output_dir", "./output"),
            float(payload.get("start_time", 0)),
            payload.get("end_time"),
            payload.get("placements"),
            payload.get("acoustic_features"),
        )
    audit = json.loads(Path(output + ".audit.json").read_text())

    warnings = []
    for item in audit:
        for entry in [item] + item.get("constraints", []) + item.get("measurements", []):
            if entry.get("code") in (
                "headroom_limited",
                "audibility_floor_limited",
                "added_layer_balance_limited",
                "mix_peak_over_full_scale",
                "volume_peak_over_full_scale",
                "original_ducking_not_applied",
                "legacy_mix_level_ignored",
            ):
                warnings.append(entry)
    Path(args.response_json).write_text(
        json.dumps(dict(output_audio=output, audit=audit, warnings=warnings), indent=2)
    )


if __name__ == "__main__":
    main()
