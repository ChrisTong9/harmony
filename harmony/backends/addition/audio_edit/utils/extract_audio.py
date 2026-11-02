from __future__ import annotations

import subprocess
from pathlib import Path


def extract_audio_for_analysis(
    input_media: str,
    output_dir: str = "./audio_edit/outputs/analysis",
) -> str:
    input_path = Path(input_media)

    if not input_path.exists():
        raise FileNotFoundError(f"Input media does not exist: {input_path}")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    output_path = output_dir_path / f"{input_path.stem}_analysis.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]


    subprocess.run(cmd, check=True)

    return str(output_path)
