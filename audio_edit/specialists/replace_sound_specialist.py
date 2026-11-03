from __future__ import annotations

import json
from audio_edit.analyzers.edit_listening import EDIT_EVIDENCE_RULES
from typing import Any

from audio_edit.llm.client import LLMClient
from audio_edit.models import AudioEditContext
from audio_edit.plans.replace_sound_plan import ReplacePromptPlan
from audio_edit.plans.replacement_semantics import SEMANTICS_PROMPT, validate_replacement_semantics


class ReplaceSoundSpecialist:
    """
    Replace specialist only decomposes the user request.

    It does NOT create RemoveSoundPlan.
    It does NOT create AddSoundPlan.

    It only rewrites:

        "replace dog barking with cat meowing"

    into:

        remove_user_prompt = "remove dog barking"
        add_user_prompt = "add realistic, clean, isolated cat meowing, natural, suitable for mixing"
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def plan(
        self,
        ctx: AudioEditContext,
        user_prompt: str,
        audio_analysis: dict[str, Any] | None = None,
    ) -> ReplacePromptPlan:
        system_prompt = self._system_prompt() + SEMANTICS_PROMPT + EDIT_EVIDENCE_RULES

        user_payload = {
            "user_prompt": user_prompt,
            "input_media": ctx.input_audio,
            "audio_analysis": audio_analysis,
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(user_payload, indent=2, ensure_ascii=False),
            },
        ]

        response = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
        )

        content = response.choices[0].message.content

        data = self._parse_json(content)
        self._validate_data(data)
        if data.get("temporal") is not None:
            ctx.params["replace_decomposition_legacy_timing_ignored"] = True

        semantics = None
        try:
            semantics = validate_replacement_semantics(data.get("semantics"), user_prompt)
            ctx.params["replacement_semantics"] = {"status": "available"}
        except (TypeError, ValueError) as exc:
            ctx.params["replacement_semantics"] = {"status": "unavailable", "reason": str(exc)}

        return ReplacePromptPlan(
            user_prompt=user_prompt,
            source_sound=data["source_sound"].strip(),
            target_sound=data["target_sound"].strip(),
            remove_user_prompt=data["remove_user_prompt"].strip(),
            add_user_prompt=data["add_user_prompt"].strip(),
            notes=(
                data["notes"].strip()
                if isinstance(data.get("notes"), str) and data["notes"].strip()
                else None
            ),
            semantics=semantics,
        )

    def _system_prompt(self) -> str:
        return """
You are a replace-sound decomposition specialist.

Your primary task is SOUND replacement.

Your job is NOT to create low-level audio editing parameters.

Your job is to decompose a replace request into:
1. a sound removal prompt,
2. a sound addition prompt.

Return ONLY valid JSON.
Do not include markdown, explanations, or comments.

Required JSON fields:

{
  "source_sound": "the old sound to remove",
  "target_sound": "the new sound to add",
  "remove_user_prompt": "remove the old sound",
  "add_user_prompt": "add the new sound",
  "notes": null
}

AUDIO REPLACEMENT RULES:

- source_sound is the original sound that should disappear.

- target_sound is the new replacement sound that should be generated.

- remove_user_prompt should be simple and direct.

Example:

User:
"replace dog barking with cat meowing"

Output concepts:
source_sound = "dog barking"
target_sound = "cat meowing"
remove_user_prompt = "remove dog barking"
add_user_prompt = "add realistic, clean, isolated cat meowing, natural feline meows, suitable for mixing"

- add_user_prompt should be written as a normal add-sound request.

- Describe the new sound itself clearly.

- Use words such as realistic, clean, isolated, natural, and suitable for mixing when appropriate.

- Do not include backend names such as SAMAudio or Stable Audio.

- Do not create low-level remove or add plans.

- Do not include duration unless the user explicitly specifies a duration.

- Do not include a time span unless the user explicitly specifies timing.

- Do not invent multiple replacement regions.

- Let RemoveSoundSpecialist decide how to remove the original sound.

- Let AddSoundSpecialist decide how to add the new sound.


INTERFACE RULES:

Only describe what to remove and what to add. Do not return temporal, placements, event IDs or computed durations.
Preserve useful occurrence, exclusion and timing language in the subgoal text.
The full original request is also passed unchanged to both specialists.
RemoveSoundSpecialist determines removal scopes. AddSoundSpecialist receives
those same scopes and owns the new sound's generation and repetition within them.
Replacement itself does not select, reinterpret or repair a time window.


IMPORTANT:

Sound replacement is always the primary task.

"""

    def _parse_json(self, content: str | None) -> dict[str, Any]:
        if content is None:
            raise ValueError("LLM response content is None.")

        text = content.strip()

        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.startswith("json"):
                text = text[4:].strip()

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Could not parse JSON from LLM output:\n{content}")

        return json.loads(text[start : end + 1])

    def _validate_data(self, data: dict[str, Any]) -> None:
        required_keys = [
            "source_sound",
            "target_sound",
            "remove_user_prompt",
            "add_user_prompt",
        ]

        for key in required_keys:
            if key not in data:
                raise ValueError(f"Replace decomposition missing key '{key}': {data}")

            if not isinstance(data[key], str) or not data[key].strip():
                raise ValueError(f"Replace decomposition key '{key}' must be non-empty: {data}")
