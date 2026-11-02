from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReplaceVideoPlan:
    """Structured plan for visual video replacement."""

    user_prompt: str
    edit_prompt: str
    source_object: str | None = None
    target_object: str | None = None
    reference_image: str | None = None
    first_frame_image: str | None = None
    output_dir: str | None = None
    notes: str | None = None
