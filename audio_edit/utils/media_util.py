from __future__ import annotations

from pathlib import Path
import subprocess

import soundfile as sf
from audio_edit.utils.audio_input import validate_audio_input


def get_media_duration(path: str | Path) -> float:
    media_path = validate_audio_input(path)

    if media_path.suffix.lower() in {".wav", ".flac"}:
        info = sf.info(str(media_path))
        if info.samplerate > 0 and info.frames > 0:
            return info.frames / info.samplerate

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {media_path}:\n{result.stderr}")

    raw = result.stdout.strip()

    if not raw:
        raise RuntimeError(f"ffprobe returned empty duration for {media_path}")

    return float(raw)


def get_audio_duration(path: str | Path) -> float:
    return get_media_duration(path)
