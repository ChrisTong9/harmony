"""Validate supported audio paths before planning or decoding."""

from pathlib import Path

AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"})


def validate_audio_input(path):
    if not path:
        raise ValueError("An input audio file is required.")
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() not in AUDIO_SUFFIXES:
        formats = ", ".join(sorted(AUDIO_SUFFIXES))
        raise ValueError(f"Unsupported audio format {path.suffix or '(none)'}. Use {formats}.")
    if not path.is_file():
        raise FileNotFoundError(f"Input audio file not found: {path}")
    return path
