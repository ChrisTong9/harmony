from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio
from stable_audio_3 import StableAudioModel


def reference_conditioning(path, start, end, mode, init_noise_level=None):
    """Existing Stable Audio variation input; no pitch/time transformation here."""
    import math

    waveform, sr = torchaudio.load(str(path))
    duration = waveform.shape[-1] / sr
    end = duration if end is None else end
    if (
        not all(math.isfinite(x) for x in (start, end))
        or not 0 <= start < end <= duration + 0.5 / sr
    ):
        raise ValueError("Reference interval is outside its original audio")
    first, last = round(start * sr), round(end * sr)
    audio = waveform[:, first:last]
    if not audio.numel() or not torch.isfinite(audio).all():
        raise ValueError("Reference audio is empty or nonfinite")


    levels = {"structure": 0.55, "context": 0.85}
    if mode not in levels:
        raise ValueError("Unknown reference conditioning mode")
    noise = levels[mode] if init_noise_level is None else init_noise_level
    if isinstance(noise, bool) or not math.isfinite(noise) or not 0 <= noise <= 1:
        raise ValueError("init_noise_level must be finite in [0,1]")

    return dict(init_audio=(sr, audio), init_noise_level=noise)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--model", default="medium")
    parser.add_argument("--reference-audio")
    parser.add_argument("--reference-start", type=float, default=0.0)
    parser.add_argument("--reference-end", type=float)
    parser.add_argument("--reference-mode", choices=("structure", "context"))
    parser.add_argument("--init-noise-level", type=float)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reference = (
        reference_conditioning(
            args.reference_audio,
            args.reference_start,
            args.reference_end,
            args.reference_mode,
            args.init_noise_level,
        )
        if args.reference_audio
        else {}
    )
    if (args.reference_mode or args.init_noise_level is not None) and not args.reference_audio:
        raise ValueError("Reference mode/noise level requires input samples")


    model = StableAudioModel.from_pretrained(args.model)


    with torch.inference_mode():
        audio = model.generate(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            duration=args.duration,
            **reference,
        )

    waveform = audio.detach().cpu().float()

    if waveform.ndim == 3:
        waveform = waveform[0]

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    waveform = torch.clamp(waveform, -1.0, 1.0)

    torchaudio.save(str(output_path), waveform, 44100)

    print(f"[run_stable_audio] Saved: {output_path}")


if __name__ == "__main__":
    main()
