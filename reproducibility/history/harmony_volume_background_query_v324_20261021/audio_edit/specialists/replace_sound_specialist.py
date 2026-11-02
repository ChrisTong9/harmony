from __future__ import annotations

import json
from audio_edit.analyzers.edit_listening import EDIT_EVIDENCE_RULES
from typing import Any

from audio_edit.llm.client import LLMClient
from audio_edit.models import AudioEditContext
from audio_edit.plans.replace_sound_plan import ReplacePromptPlan


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
        visual_analysis: dict[str, Any] | None = None,
        audio_analysis: dict[str, Any] | None = None,
    ) -> ReplacePromptPlan:


        system_prompt = self._system_prompt() + EDIT_EVIDENCE_RULES

        user_payload = {
            "user_prompt": user_prompt,
            "input_media": ctx.input_audio,
            "visual_analysis": visual_analysis,
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


        return ReplacePromptPlan(
            user_prompt=user_prompt,
            source_sound=data["source_sound"].strip(),
            target_sound=data["target_sound"].strip(),
            remove_user_prompt=data["remove_user_prompt"].strip(),
            add_user_prompt=data["add_user_prompt"].strip(),
            video_replacement=data.get("video_replacement", False),
            source_object=data.get("source_object"),
            target_object=data.get("target_object"),
            notes=(
                data["notes"].strip()
                if isinstance(data.get("notes"), str) and data["notes"].strip()
                else None
            ),
        )


    def _system_prompt(self) -> str:
        return """
You are a replace-sound decomposition specialist.

Your primary task is SOUND replacement.

Your job is NOT to create low-level audio or video editing parameters.

Your job is to decompose a replace request into:
1. a sound removal prompt,
2. a sound addition prompt,
3. and, ONLY when the user explicitly requests a visual change, identify the corresponding video object replacement.

Return ONLY valid JSON.
Do not include markdown, explanations, or comments.

Required JSON schema:

{
  "source_sound": "the old sound to remove",
  "target_sound": "the new sound to add",
  "remove_user_prompt": "remove the old sound",
  "add_user_prompt": "add the new sound",
  "video_replacement": false,
  "source_object": null,
  "target_object": null,
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

- If the target sound corresponds to an on-screen animal or object and useful visual information is available, visual information may be used to improve the SOUND description, such as size, age, material, or behavior.

- Using visual information to improve sound generation does NOT mean the video should be edited.

- Do not include backend names such as SAMAudio, Stable Audio, or MMAudio.

- Do not create low-level remove or add plans.

- Do not include duration unless the user explicitly specifies a duration.

- Do not include a time span unless the user explicitly specifies timing.

- Do not invent multiple replacement regions.

- Let RemoveSoundSpecialist decide how to remove the original sound.

- Let AddSoundSpecialist decide how to add the new sound.


VIDEO REPLACEMENT RULES:

Video replacement is OPTIONAL and secondary to sound replacement.

Set "video_replacement" to true ONLY when the user EXPLICITLY requests that the visual content of the video should also change.

Explicit visual requests include language such as:
- "replace the cat in the video with a dog"
- "also change the cat into a dog visually"
- "replace the person on screen with a robot"
- "change the object in the video"
- "replace the cat with a dog in the scene"

Do NOT infer video replacement simply because:
- the input is a video,
- the sound source is visible,
- visual analysis detects the object,
- changing the video might make the result more consistent,
- or the sound replacement implies a different object.

If the user only requests a sound change:
- video_replacement must be false
- source_object must be null
- target_object must be null

Example:

User:
"replace cat meowing with dog barking"

Then:
video_replacement = false
source_object = null
target_object = null

Even if a cat is clearly visible in the video, DO NOT request a video replacement.

If the user explicitly requests both sound and visual replacement:

User:
"replace the cat meowing with dog barking and replace the cat in the video with a dog"

Then:
source_sound = "cat meowing"
target_sound = "dog barking"
video_replacement = true
source_object = "cat"
target_object = "dog"

When video_replacement is true:
- source_object is the visible object that should be visually replaced.
- target_object is the visible object that should replace it.
- Assume explicitly requested visual targets refer to on-screen entities.
- Do not reason about off-screen visual objects.
- Do not infer any additional visual edits beyond what the user explicitly requests.

When video_replacement is false:
- source_object MUST be null.
- target_object MUST be null.


INTERFACE RULES:

Only describe what to remove and what to add, plus explicitly requested visual
changes. Do not return temporal, placements, event IDs or computed durations.
Preserve useful occurrence, exclusion and timing language in the subgoal text.
The full original request is also passed unchanged to both specialists.
RemoveSoundSpecialist determines removal scopes. AddSoundSpecialist receives
those same scopes and owns the new sound's generation and repetition within them.
Replacement itself does not select, reinterpret or repair a time window.


IMPORTANT:

Sound replacement is always the primary task.

Video replacement is only an optional additional action when explicitly requested by the user.

Never automatically propagate an audio edit into a video edit.
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


        video_replacement = data.get("video_replacement", False)
        if not isinstance(video_replacement, bool):
            raise ValueError(
                "Replace decomposition key 'video_replacement' must be a bool: "
                f"{data}"
            )

        if video_replacement:
            for key in ("source_object", "target_object"):
                if not isinstance(data.get(key), str) or not data[key].strip():
                    raise ValueError(
                        f"Replace decomposition key '{key}' must be non-empty "
                        f"when video_replacement is true: {data}"
                    )
