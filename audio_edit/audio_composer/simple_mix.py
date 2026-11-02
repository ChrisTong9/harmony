from __future__ import annotations

from pathlib import Path

import torch
import torchaudio


class SimpleMixer:
    def __init__(self, output_dir: str = "./audio_edit/outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def mix_background(
        self,
        original_path: str,
        background_path: str,
        mix_level: float = 0.7,
    ) -> str:
        original_file = Path(original_path)
        background_file = Path(background_path)

        if not original_file.exists():
            raise FileNotFoundError(f"Original file does not exist: {original_file}")

        if not background_file.exists():
            raise FileNotFoundError(f"Background file does not exist: {background_file}")

        original, sr = torchaudio.load(str(original_file))
        background, bg_sr = torchaudio.load(str(background_file))

        if bg_sr != sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=bg_sr,
                new_freq=sr,
            )
            background = resampler(background)
            bg_sr = sr

        if original.shape[0] != background.shape[0]:
            background = self._match_channels(background, original.shape[0])

        max_len = max(original.shape[1], background.shape[1])

        original = self._pad_to_length(original, max_len)
        background = self._pad_to_length(background, max_len)

        output = original + background * float(mix_level)

        peak = output.abs().max()
        if peak > 1.0:
            output = output / peak

        output_path = self.output_dir / f"{original_file.stem}_add_background.wav"
        torchaudio.save(str(output_path), output, sr)

        return str(output_path)

    def _pad_to_length(self, audio: torch.Tensor, target_len: int) -> torch.Tensor:
        current_len = audio.shape[1]

        if current_len == target_len:
            return audio

        if current_len > target_len:
            return audio[:, :target_len]

        pad_len = target_len - current_len
        padding = torch.zeros(
            audio.shape[0],
            pad_len,
            dtype=audio.dtype,
            device=audio.device,
        )

        return torch.cat([audio, padding], dim=1)

    def _match_channels(self, audio: torch.Tensor, target_channels: int) -> torch.Tensor:
        current_channels = audio.shape[0]

        if current_channels == target_channels:
            return audio

        if current_channels == 1 and target_channels == 2:
            return audio.repeat(2, 1)

        if current_channels == 2 and target_channels == 1:
            return audio.mean(dim=0, keepdim=True)

        if current_channels > target_channels:
            return audio[:target_channels, :]

        if current_channels < target_channels:
            repeats = target_channels // current_channels
            remainder = target_channels % current_channels

            repeated = audio.repeat(repeats, 1)

            if remainder > 0:
                repeated = torch.cat([repeated, audio[:remainder]], dim=0)

            return repeated

        return audio
