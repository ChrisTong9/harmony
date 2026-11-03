from __future__ import annotations

import json
from typing import Any

from audio_edit.llm.client import LLMClient
from audio_edit.plans.reference_image_plan import ReferenceImagePromptPlan


class ReferenceImagePromptSpecialist:
    """Build a target-object image prompt from the video analysis."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def plan(
        self,
        *,
        user_prompt: str,
        source_object: str,
        target_object: str,
        visual_analysis: dict[str, Any],
    ) -> ReferenceImagePromptPlan:
        payload = {
            "user_prompt": user_prompt,
            "source_object": source_object,
            "target_object": target_object,
            "gemini_video_analysis": visual_analysis,
        }
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, indent=2),
            },
        ]

        response = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
        )
        content = response.choices[0].message.content
        data = self._parse_json(content)

        prompt = self._required_text(data.get("prompt"), "prompt")
        negative_prompt = self._required_text(
            data.get("negative_prompt"),
            "negative_prompt",
        )

        return ReferenceImagePromptPlan(
            prompt=prompt,
            negative_prompt=negative_prompt,
        )

    @staticmethod
    def _system_prompt() -> str:
        return """
You create a text-to-image prompt for a reference image used by a video
object-replacement model.

Use the target object and Gemini's video analysis to create one clear,
photorealistic reference image that is visually compatible with the video's
environment, lighting, camera viewpoint, and scale.

The generated image must:
- contain exactly one instance of the target object;
- show the complete target object, centered and unobstructed;
- use a natural pose and a simple, unobtrusive background;
- make the target object's identity, shape, texture, and color unambiguous;
- contain no source object, people, extra animals or objects, text, logo,
  watermark, collage, split screen, or border.

Do not ask the image model to perform video editing. Describe only the single
reference image to generate. Do not invent a special breed, color, costume, or
accessory unless the user explicitly requested it or the video analysis makes
it necessary for visual compatibility.

Return ONLY valid JSON with this schema:
{
  "prompt": "English image-generation prompt",
  "negative_prompt": "English comma-separated exclusions"
}
"""

    @staticmethod
    def _parse_json(content: str | None) -> dict[str, Any]:
        if content is None:
            raise ValueError("Reference-image prompt LLM response is empty.")

        text = content.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                f"Could not parse reference-image prompt JSON from LLM output: {content}"
            )

        data = json.loads(text[start : end + 1])
        if not isinstance(data, dict):
            raise ValueError("Reference-image prompt response must be an object.")
        return data

    @staticmethod
    def _required_text(value: Any, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Reference-image prompt field '{field_name}' must be non-empty.")
        return value.strip()
