from __future__ import annotations

import os
from pathlib import Path
from time import time
from typing import Any

from dotenv import load_dotenv
from google import genai

import json
import re
import time


def extract_json_object(text: str) -> dict[str, Any]:
    """
    Robustly parse JSON object from LLM output.
    Handles:
    - ```json ... ```
    - extra text before/after JSON
    - Python-style None/True/False
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected string, got {type(text)}")

    raw = text.strip()

    if raw.startswith("```json"):
        raw = raw.removeprefix("```json").strip()
    elif raw.startswith("```"):
        raw = raw.removeprefix("```").strip()

    if raw.endswith("```"):
        raw = raw.removesuffix("```").strip()

    raw = raw.replace(": None", ": null")
    raw = raw.replace(": True", ": true")
    raw = raw.replace(": False", ": false")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got: {type(data)}")

    return data


class GeminiVideoAnalyzer:


    def __init__(self, model: str | None = None):
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY in environment or .env")

        self.model = model or os.getenv("GEMINI_VIDEO_MODEL", "gemini-2.5-flash")
        self.client = genai.Client(api_key=api_key)

    def analyze(self, input_video: str, user_prompt: str) -> dict[str, Any]:
        video_path = Path(input_video)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file does not exist: {video_path}")


        uploaded_file = self.client.files.upload(file=str(video_path))

        prompt = self._build_prompt(user_prompt)


        while uploaded_file.state.name == "PROCESSING":

            time.sleep(2)

            uploaded_file = self.client.files.get(name=uploaded_file.name)


        if uploaded_file.state.name == "FAILED":
            raise ValueError("Gemini failed to process the uploaded file.")
        response = self.client.models.generate_content(
            model=self.model,
            contents=[uploaded_file, prompt],
        )

        data = extract_json_object(response.text)
        self._validate(data)
        from audio_edit.plans.event_ids import assign_event_ids
        from audio_edit.utils.media_util import get_media_duration
        assign_event_ids(data, 'visual', get_media_duration(str(video_path)))
        return data

    def _build_prompt(self, user_prompt: str) -> str:
        return f"""
You are the Visual Perception Module for an intelligent audio editing agent. 
Your ONLY job is to analyze the silent video track to provide spatial and visual-timing parameters for the downstream audio mixer.

User's Audio Editing Instruction: 
"{user_prompt}"

Task:
Analyze the video specifically in the context of the user's prompt. 
1. Determine the physical environment to guide reverb settings.
2. Check if the sound source implied by the prompt is physically visible on screen.
3. Locate visual events that the instruction uses as temporal references, plus
   other salient visual events that are clear enough to timestamp.

Timestamp policy:
- Times are seconds from the beginning of the video.
- Emit separate entries for repeated occurrences and order them by start_time.
- Use one-based occurrence_index for repeated instances of the same event.
- Mark approximate boundaries with approximate=true.
- Use JSON null when a boundary or confidence is uncertain. Never invent a
  timestamp merely to satisfy the editing instruction.


Output ONLY valid JSON. No markdown formatting, no explanations.

Required JSON Schema:
{{
  "environmental_context": {{
    "scene_type": "indoor | outdoor | mixed",
    "space_size": "small (e.g., car, bathroom) | medium (e.g., living room) | large (e.g., concert hall, open field) | unknown",
    "visual_description": "Brief 1-sentence description of the physical space."
  }},
  "target_source_analysis": {{
    "is_on_screen": true | false,
    "visibility_reason": "Why it is considered on or off screen (e.g., 'The speaker's face is clearly visible' or 'The camera points at the sky, the ambulance is not visible')."
  }},
  "visual_events": [
    {{
      "event": "concise visible event identity",
      "start_time": 6.4,
      "end_time": 6.8,
      "approximate": true,
      "confidence": 0.9,
      "occurrence_index": 1
    }}
  ]
}}
"""

    def _validate(self, data: dict[str, Any]) -> None:
        required = [
            "environmental_context",
            "target_source_analysis"
        ]

        for key in required:
            if key not in data or not isinstance(data[key], dict):
                raise ValueError(f"Video analysis missing object '{key}': {data}")

        events = data.setdefault("visual_events", [])
        if not isinstance(events, list):
            raise ValueError("visual_events must be a JSON array")
        for event in events:
            if not isinstance(event, dict) or not str(event.get("event", "")).strip():
                raise ValueError(f"Invalid visual event: {event}")
            for name in ("start_time", "end_time", "confidence"):
                value = event.get(name)
                if value is not None and not isinstance(value, (int, float)):
                    raise ValueError(f"Visual event {name} must be numeric or null: {event}")
            start = event.get("start_time")
            end = event.get("end_time")
            if start is not None and start < 0:
                raise ValueError(f"Visual event start_time cannot be negative: {event}")
            if start is not None and end is not None and end <= start:
                raise ValueError(f"Visual event end_time must be after start_time: {event}")
            occurrence = event.get("occurrence_index")
            if occurrence is not None and (not isinstance(occurrence, int) or occurrence < 1):
                raise ValueError(f"occurrence_index must be a positive integer or null: {event}")

        events.sort(
            key=lambda event: (
                event.get("start_time") is None,
                float(event.get("start_time") or 0.0),
            )
        )


