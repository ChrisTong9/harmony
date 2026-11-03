from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


def probe_duration(path: str) -> float | None:
    ffprobe = os.environ.get("FFPROBE_BIN", "ffprobe")
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            path,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None

    try:
        payload = json.loads(proc.stdout)
        return float(payload["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def mux_video_with_audio(
    *,
    video_path: str,
    audio_path: str,
    output_path: str,
) -> str:
    """
    Copy one video stream and mux the first audio stream from another media file.

    MVP duration policy:
    - The video stream is authoritative.
    - ffmpeg uses -shortest so the result does not outlive the shorter stream.
    """
    source_video = Path(video_path)
    source_audio = Path(audio_path)
    destination = Path(output_path)

    if not source_video.is_file():
        raise FileNotFoundError(f"Video does not exist: {source_video}")
    if not source_audio.is_file():
        raise FileNotFoundError(f"Audio source does not exist: {source_audio}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = os.environ.get("FFMPEG_BIN", "ffmpeg")
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source_video),
        "-i",
        str(source_audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(destination),
    ]

    proc = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )

    if proc.returncode != 0:
        details = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"ffmpeg mux failed: {details}")

    if not destination.is_file():
        raise RuntimeError(f"ffmpeg reported success but output is missing: {destination}")

    return str(destination.resolve())
