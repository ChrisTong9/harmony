from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReferenceImagePromptPlan:
    """Text prompts used to generate a ReCo target-object reference image."""

    prompt: str
    negative_prompt: str
