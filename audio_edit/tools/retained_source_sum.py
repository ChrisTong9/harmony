"""Compose retained sources without rescaling their original amplitudes."""

from pathlib import Path
import subprocess


def combine(input_audio, source_paths, folder):
    import numpy as np
    import soundfile as sf

    original, rate = sf.read(input_audio, dtype="float32", always_2d=True)
    output = np.zeros_like(original)
    alignment = []
    for index, path in enumerate(source_paths):
        stem, sr = sf.read(path, dtype="float32", always_2d=True)
        if sr != rate:
            converted = Path(folder) / f"stem-{index}-aligned.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-i",
                    str(path),
                    "-ar",
                    str(rate),
                    "-c:a",
                    "pcm_f32le",
                    str(converted),
                ],
                check=True,
            )
            stem, sr = sf.read(converted, dtype="float32", always_2d=True)
        if stem.shape[1] == 1 and original.shape[1] > 1:
            stem = np.repeat(stem, original.shape[1], axis=1)
        elif original.shape[1] == 1 and stem.shape[1] > 1:
            stem = stem.mean(axis=1, keepdims=True)
        if sr != rate or stem.shape[1] != original.shape[1] or abs(len(stem) - len(original)) > 2:
            raise ValueError("Retained source does not match original sample grid")
        if not np.isfinite(stem).all():
            raise ValueError("Non-finite retained source")
        aligned = np.zeros_like(original)
        length = min(len(aligned), len(stem))
        aligned[:length] = stem[:length]
        output += aligned
        alignment.append(dict(source=str(path), samples=len(stem), sample_rate=sr))
    if not np.isfinite(output).all():
        raise ValueError("Non-finite retained-source composition")
    path = Path(folder) / "retained_sources.wav"
    sf.write(path, output, rate, subtype="FLOAT")
    return str(path), dict(
        alignment=alignment,
        original_samples=len(original),
        sample_rate=rate,
        gain_per_source=1,
        normalization_applied=False,
        peak=float(np.max(np.abs(output))),
        output_format="FLOAT",
    )
