from __future__ import annotations

from dataclasses import dataclass, field

from audio_edit.plans.remove_sound_plan import RemoveSoundPlan
from audio_edit.plans.add_sound_plan import AddSoundPlan
from audio_edit.plans.temporal import ResolvedPlacement, TemporalConstraint


@dataclass
class ReplacePromptPlan:
    """
    High-level decomposition result.

    ReplaceSoundSpecialist only produces this:
    - what to remove
    - what to add
    - two rewritten prompts
    """

    user_prompt: str

    source_sound: str
    target_sound: str

    remove_user_prompt: str
    add_user_prompt: str

    video_replacement: bool = False
    source_object: str | None = None
    target_object: str | None = None

    notes: str | None = None
    temporal: TemporalConstraint | None = None
    temporal_status: str = "not_requested"
    temporal_reason: str | None = None
    placements: list[ResolvedPlacement] = field(default_factory=list)


@dataclass
class ReplaceSoundPlan:
    """
    Executable replace plan.

    This is created after running:
    - RemoveSoundSpecialist
    - AddSoundSpecialist
    """

    user_prompt: str

    source_sound: str
    target_sound: str

    remove_user_prompt: str
    add_user_prompt: str

    remove_plan: RemoveSoundPlan
    add_plan: AddSoundPlan

    notes: str | None = None
    temporal: TemporalConstraint | None = None
    temporal_status: str = "not_requested"
    temporal_reason: str | None = None
    placements: list[ResolvedPlacement] = field(default_factory=list)
