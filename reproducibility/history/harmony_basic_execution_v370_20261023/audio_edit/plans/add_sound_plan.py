from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from audio_edit.plans.temporal import ResolvedPlacement, TemporalConstraint


SourceType = Literal["audio","video"]

@dataclass
class MediaAnalysis:
    visual_analysis: dict[str, Any] | None = None
    audio_analysis: dict[str, Any] | None = None

@dataclass
class AddSoundPlan:
    source_type: SourceType
    user_prompt:str
    is_on_screen : bool
    is_speech: bool
    audio_prompt: str
    duration: float | None
    start_time: float
    end_time: float | None
    target_sound: str | None = None
    layer_role: Literal["foreground", "background"] | None = None
    negative_audio_prompt: str | None = None
    composer: dict[str, Any] | None = None
    speech_text: str | None = None
    voice_prompt: str | None = None
    visual_edit_prompt : str | None = None
    use_gemini_analysis: bool = False

    media_analysis: MediaAnalysis | None = None
    notes: str | None = None


    temporal: TemporalConstraint | None = None
    temporal_status: str = "not_requested"
    temporal_reason: str | None = None
    placements: list[ResolvedPlacement] = field(default_factory=list)
    repeat_count: int | None = None
    edit_windows: list[dict] = field(default_factory=list)
    execution_placements: list[dict] = field(default_factory=list)
    contract_version: int = 0
    requested_edits: list[dict] = field(default_factory=list)


    intent: dict = field(default_factory=dict)
    reference_bindings: dict = field(default_factory=dict)
    material_pending: bool = True
    compile_error_kind: str | None = None
