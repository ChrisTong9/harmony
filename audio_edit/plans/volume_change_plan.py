from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from audio_edit.plans.temporal import ResolvedPlacement, TemporalConstraint


VolumeTargetScope = Literal["whole_mix", "target_source"]


@dataclass
class VolumeChangePlan:
    user_prompt: str
    target_sound: str | None
    gain_db: float
    apply_to: VolumeTargetScope = "whole_mix"
    target_sounds: list[str] = field(default_factory=list)
    gain_db_explicit: bool = False
    gain_profile: str = "static"
    temporal: TemporalConstraint | None = None
    temporal_status: str = "not_requested"
    temporal_reason: str | None = None
    placements: list[ResolvedPlacement] = field(default_factory=list)
    notes: str | None = None
    changes: list[VolumeChangePlan] = field(default_factory=list)
    temporal_schema_error: str | None = None
    temporal_input: dict | None = None
    source_evidence: dict | None = None

    def refresh_timing_status(self):
        """A composed edit retains the least certain child's timing status."""
        if self.changes:
            statuses = {p.temporal_status for p in self.changes}
            self.temporal_status = next(
                status
                for status in (
                    "unresolved",
                    "estimated",
                    "approximate",
                    "resolved",
                    "not_requested",
                )
                if status in statuses
            )
            self.temporal_reason = (
                "; ".join(p.temporal_reason for p in self.changes if p.temporal_reason) or None
            )
        return self

    def effective_target_sounds(self) -> list[str]:
        """Return plural targets while accepting legacy singular plans."""

        if self.target_sounds:
            return self.target_sounds
        return [self.target_sound] if self.target_sound else []
