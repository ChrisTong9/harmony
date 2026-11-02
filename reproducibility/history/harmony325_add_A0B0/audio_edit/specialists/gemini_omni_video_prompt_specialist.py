from __future__ import annotations

import json
from typing import Any

from audio_edit.llm.client import LLMClient


class GeminiOmniVideoPromptSpecialist:
    """Create a behavior-aware visual edit prompt for Gemini Omni."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def build_prompt(
        self,
        *,
        user_prompt: str,
        source_object: str,
        target_object: str,
        source_sound: str,
        target_sound: str,
        visual_analysis: dict[str, Any],
        audio_analysis: dict[str, Any],
    ) -> str:
        payload = {
            "user_prompt": user_prompt,
            "source_object": source_object,
            "target_object": target_object,
            "source_sound": source_sound,
            "target_sound": target_sound,
            "gemini_video_analysis": visual_analysis,
            "audio_analysis": audio_analysis,
        }
        response = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, indent=2),
                },
            ],
        )

        content = response.choices[0].message.content
        data = self._parse_json(content)
        edit_prompt = data.get("edit_prompt")
        if not isinstance(edit_prompt, str) or not edit_prompt.strip():
            raise ValueError(
                "Gemini Omni prompt response requires a non-empty edit_prompt."
            )
        return edit_prompt.strip()

    @staticmethod
    def _system_prompt() -> str:
        return """
You write a single English visual video-editing instruction for Gemini Omni.
The edited video will later be used as the visual condition for an audio model,
so visible sound-producing behavior is essential.

Requirements:
- Replace every visible source_object with a realistic target_object.
- Infer the natural visible action that physically produces target_sound.
- Make the target_object clearly perform that action at the moments where the
  source object originally makes sound or performs the corresponding action.
- Describe concrete, anatomically or mechanically plausible motion. For
  example, for a cat meowing, require natural mouth opening and closing, jaw
  and throat movement, a subtle head motion, and an unmistakable meowing pose.
- Do not merely create a silent or motionless target object when target_sound
  implies visible behavior.
- Preserve original duration, timing, camera motion, framing, lighting,
  background, subject position, motion trajectory, and unrelated objects.
- Keep identity and motion temporally consistent across every frame.
- Do not add text, subtitles, borders, split screens, or unrelated changes.
- Focus on the visual edit. The generated audio will be removed downstream.
- Do not mention implementation details, model names, or JSON in edit_prompt.

Return ONLY valid JSON:
{
  "edit_prompt": "complete English video editing instruction"
}
"""

    @staticmethod
    def _parse_json(content: str | None) -> dict[str, Any]:
        if content is None:
            raise ValueError("Gemini Omni prompt LLM response is empty.")

        text = content.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                f"Could not parse Gemini Omni prompt JSON: {content}"
            )

        data = json.loads(text[start : end + 1])
        if not isinstance(data, dict):
            raise ValueError("Gemini Omni prompt response must be a JSON object.")
        return data
