"""Time-varying source gain, also usable for a whole waveform."""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def gain_envelope(audio, sample_rate, *, gain_db=0.0, profile="static", placements=None):
    if profile not in {"static", "fade_in", "fade_out", "level", "mute"}:
        raise ValueError(f"Unknown gain profile: {profile}")
    if not math.isfinite(gain_db) or not -24 <= gain_db <= 24:
        raise ValueError("Gain must be finite and between -24 and 24 dB")
    n = audio.shape[-1]
    result = torch.ones(n, dtype=audio.dtype, device=audio.device)
    regions = placements or [{"start_time": 0.0, "end_time": n / sample_rate}]
    factor = 10 ** (gain_db / 20)
    for region in regions:
        start = max(0, round(region["start_time"] * sample_rate))
        end = min(n, round(region["end_time"] * sample_rate))
        if end <= start:
            raise ValueError("Volume region has no samples")
        length = end - start
        if profile == "mute":
            gain = torch.zeros(length, dtype=audio.dtype, device=audio.device)
        elif profile == "static":
            gain = torch.full((length,), factor, dtype=audio.dtype, device=audio.device)
        elif profile in {"fade_in", "fade_out"}:
            a, b = (0.0, factor) if profile == "fade_in" else (factor, 0.0)
            gain = torch.linspace(a, b, length, dtype=audio.dtype, device=audio.device)
        else:
            chunk = audio[:, start:end]
            frame = max(1, round(0.05 * sample_rate))
            power = chunk.square().mean(dim=0)
            pad = (-length) % frame
            padded = F.pad(power.reshape(1, 1, -1), (0, pad), mode="replicate")
            rms = padded.reshape(-1, frame).mean(dim=1).sqrt()
            active = rms > max(float(rms.max()) * 0.01, 1e-7)
            frame_gain = torch.ones_like(rms)
            if active.any():
                target = torch.quantile(rms[active], 0.9) * factor

                frame_gain[active] = (target / rms[active]).clamp(10 ** (-24 / 20), 10 ** (24 / 20))
            gain = F.interpolate(
                frame_gain.reshape(1, 1, -1), size=length, mode="linear", align_corners=False
            ).reshape(-1)
        result[start:end] = gain
    return result


def volume_scope_envelope(length, sr, placements, profile, *, device=None, dtype=None):
    """Do not fade a requested fade/mute delta back to the unedited waveform."""
    if not placements:
        return torch.ones(length, device=device, dtype=dtype)
    if profile not in {"fade_in", "fade_out", "mute"}:
        from audio_edit.tools.scene_mix import scope_envelope

        return scope_envelope(length, sr, placements, device=device, dtype=dtype)
    mask = torch.zeros(length, device=device, dtype=dtype)
    for region in placements:
        a, b = region["start_time"], region["end_time"]
        if not all(type(v) in (int, float) and math.isfinite(v) for v in (a, b)):
            raise ValueError("Volume scope needs finite endpoints")
        start, end = round(a * sr), round(b * sr)
        if not 0 <= start < end <= length:
            raise ValueError("Volume scope is outside the original sample grid")
        mask[start:end] = 1.0
    return mask


def apply_volume_delta(original, delta, diagnostics, label):
    """Preserve the requested volume change in float audio without master gain."""
    if original.shape != delta.shape:
        raise ValueError("Volume delta must use the original sample grid")
    if not torch.isfinite(original).all() or not torch.isfinite(delta).all():
        raise ValueError("Volume input and delta must be finite")
    output = original + delta
    if not torch.isfinite(output).all():
        raise ValueError("Volume output must be finite")
    peak = float(output.abs().max()) if output.numel() else 0.0
    if peak > 1.0:
        diagnostics.append(
            {
                "code": "volume_peak_over_full_scale",
                "layer": label,
                "peak": peak,
                "peak_dbfs": 20 * math.log10(peak),
                "samples_over_full_scale": int((output.abs() > 1.0).sum()),
                "normalization_applied": False,
                "master_gain": 1.0,
                "reason": "Requested gain retained in float WAV; playback may clip",
            }
        )
    return output, 1.0
