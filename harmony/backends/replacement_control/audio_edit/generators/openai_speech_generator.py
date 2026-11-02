from __future__ import annotations

import os
from pathlib import Path
from openai import OpenAI


class OpenAISpeechGenerator:
    def __init__(
            self,
            output_dir:str = "audio_edit/outputs/generated",
            model: str | None = None,
            default_voice: str | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = model or os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
        self.default_voice = default_voice or os.getenv("OPENAI_TTS_VOICE", "alloy")

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(
        self,
        text: str,
        voice_prompt: str | None = None,
        voice: str | None = None,
        output_name: str = "openai_speech_generated.wav",
    ) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("OpenAISpeechGenerator requires non-empty speech text.")

        output_path = self.output_dir / output_name

        selected_voice = voice or self.default_voice


        instructions = voice_prompt or "Speak naturally and clearly."


        response = self.client.audio.speech.create(
            model=self.model,
            voice=selected_voice,
            input=text.strip(),
            instructions=instructions,
            response_format="wav",
        )


        response.write_to_file(str(output_path))

        if not output_path.exists():
            raise FileNotFoundError(f"OpenAI speech output not found: {output_path}")

        return str(output_path)
