from __future__ import annotations

import base64
import os
from pathlib import Path
import time
from typing import Any
import uuid

from dotenv import load_dotenv
from google import genai


class GeminiOmniVideoGenerator:
    """Edit a video through the Gemini Omni Interactions API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        poll_interval_seconds: float = 2.0,
        upload_timeout_seconds: float = 600.0,
        generation_timeout_seconds: float = 1_800.0,
    ) -> None:
        env_path = Path(__file__).resolve().parents[1] / ".env"
        load_dotenv(env_path)

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError("Missing GEMINI_API_KEY in audio_edit/.env.")

        self.client = genai.Client(api_key=resolved_key)
        self.model = (
            model
            or os.environ.get("GEMINI_OMNI_VIDEO_MODEL")
            or "gemini-omni-1.1-flash"
        )
        self.poll_interval_seconds = poll_interval_seconds
        self.upload_timeout_seconds = upload_timeout_seconds
        self.generation_timeout_seconds = generation_timeout_seconds

    def generate(
        self,
        *,
        video_path: str,
        prompt: str,
        output_dir: str | None = None,
    ) -> str:
        input_path = Path(video_path).expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"Input video does not exist: {input_path}")

        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Gemini Omni video edit prompt must not be empty.")

        resolved_output = (
            output_dir
            or os.environ.get("GEMINI_OMNI_OUTPUT_DIR")
            or str(Path(__file__).resolve().parents[1] / "outputs" / "gemini_omni")
        )
        output_path = Path(resolved_output).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        mime_type = {
            ".mp4": "video/mp4",
            ".mov": "video/mov",
            ".webm": "video/webm",
            ".avi": "video/avi",
            ".mpeg": "video/mpeg",
            ".mpg": "video/mpg",
        }.get(input_path.suffix.lower(), "video/mp4")


        interaction = self.client.interactions.create(
            model=self.model,
            input=[
                {
                    "type": "video",
                    "data": input_path,
                    "mime_type": mime_type,
                },
                {"type": "text", "text": prompt},
            ],
            response_format={
                "type": "video",
                "delivery": "uri",
            },
            background=True,
            timeout=120.0,
        )
        interaction = self._wait_for_interaction(interaction)

        video_bytes, mime_type = self._extract_video(interaction)
        suffix = ".webm" if mime_type == "video/webm" else ".mp4"
        final_path = output_path / f"gemini_omni_{uuid.uuid4().hex}{suffix}"
        final_path.write_bytes(video_bytes)

        if not final_path.is_file() or final_path.stat().st_size == 0:
            raise RuntimeError("Gemini Omni returned an empty video file.")


        return str(final_path)

    def _wait_for_upload(self, uploaded: Any) -> Any:
        deadline = time.monotonic() + self.upload_timeout_seconds
        while self._state_name(uploaded) == "PROCESSING":
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for Gemini video upload.")
            time.sleep(self.poll_interval_seconds)
            uploaded = self.client.files.get(name=uploaded.name)

        state = self._state_name(uploaded)
        if state == "FAILED":
            raise RuntimeError("Gemini rejected the uploaded video.")
        return uploaded

    def _wait_for_interaction(self, interaction: Any) -> Any:
        deadline = time.monotonic() + self.generation_timeout_seconds
        status = self._interaction_status(interaction)
        while status in {"queued", "in_progress"}:
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for Gemini Omni video edit.")

            time.sleep(self.poll_interval_seconds)
            interaction = self.client.interactions.get(
                interaction.id,
                include_input=False,
                timeout=120.0,
            )
            status = self._interaction_status(interaction)

        if status != "completed":
            raise RuntimeError(
                f"Gemini Omni interaction ended with status={status}: "
                f"{getattr(interaction, 'errors', None)}"
            )
        return interaction

    def _extract_video(self, interaction: Any) -> tuple[bytes, str]:
        output_video = getattr(interaction, "output_video", None)
        if output_video is not None:
            mime_type = getattr(output_video, "mime_type", None) or "video/mp4"
            uri = getattr(output_video, "uri", None)
            if uri:
                return self.client.files.download(file=uri), mime_type

            encoded_data = getattr(output_video, "data", None)
            if encoded_data:
                return base64.b64decode(str(encoded_data)), mime_type

        raise RuntimeError(
            "Gemini Omni response did not contain a video. "
            f"status={getattr(interaction, 'status', None)}, "
            f"errors={getattr(interaction, 'errors', None)}"
        )

    @staticmethod
    def _state_name(file: Any) -> str:
        state = getattr(file, "state", None)
        return str(getattr(state, "name", state) or "").upper()

    @staticmethod
    def _interaction_status(interaction: Any) -> str:
        return str(getattr(interaction, "status", "") or "").lower()
