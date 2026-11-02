from __future__ import annotations

from pathlib import Path
import subprocess

import torchaudio


AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}


def get_media_duration(path: str | Path) -> float:
    media_path = Path(path).expanduser().resolve()

    if not media_path.exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    if media_path.is_dir():
        raise IsADirectoryError(f"Expected media file, got directory: {media_path}")


    if media_path.suffix.lower() in {'.wav', '.flac'}:
        info = torchaudio.info(str(media_path))
        if info.sample_rate > 0 and info.num_frames > 0:
            return info.num_frames / info.sample_rate

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


def get_video_duration(path: str | Path) -> float:
    return get_media_duration(path)

def get_audio_duration(path: str) -> float:
    info = torchaudio.info(path)
    return float(info.num_frames) / float(info.sample_rate)


def get_video_duration_ffprobe(path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    return float(result.stdout.strip())
