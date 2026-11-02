from dataclasses import dataclass, field
from typing import Literal, get_args

from audio_edit.plans.temporal import ResolvedPlacement, TemporalConstraint


RemoveMode = Literal["text", "span", "video"]
Result_mode = Literal["result", "residual"]
RESULT_MODES = get_args(Result_mode)


def validate_result_mode(result_mode: str) -> None:
    if result_mode not in RESULT_MODES:
        raise ValueError(
            f"Unsupported result_mode: {result_mode!r}. SAMAudio returns one audio: "
            "use 'result' to keep text_prompt, or 'residual' to remove text_prompt."
        )


@dataclass
class TimeSpan:
    start: float
    end: float


@dataclass
class RemoveSoundPlan:
    backend: str
    mode: RemoveMode
    result_mode: Result_mode
    text_prompt: str
    time_span: list[TimeSpan] | None = None
    use_video_prompt: bool = False
    visual_prompt: str | None = None
    notes: str | None = None
    temporal: TemporalConstraint | None = None
    temporal_status: str = "not_requested"
    temporal_reason: str | None = None
    placements: list[ResolvedPlacement] = field(default_factory=list)
    temporal_schema_error: str | None = None
    temporal_input: dict | None = None
    removal_evidence: dict | None = None

    def __post_init__(self) -> None:
        validate_result_mode(self.result_mode)


def shared_removal_windows(plan, media_duration):
    """Numeric execution scope for composing remove + add; not source evidence."""
    import math

    if plan.result_mode != "residual":
        raise ValueError("Replacement removal must return residual audio")
    if plan.temporal_status == "unresolved":
        raise ValueError(plan.temporal_reason or "Removal scope is unresolved")
    if (
        plan.mode == "text"
        and not plan.time_span
        and (plan.temporal is None or plan.temporal.kind == "unspecified")
    ):
        windows = [(0.0, media_duration)]
    elif plan.mode == "span":
        windows = [(p.start, p.end) for p in (plan.time_span or [])]
    else:
        raise ValueError("Replacement needs a known text or span removal scope")
    if not windows or any(
        not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in (a, b)
        )
        or not 0 <= a < b <= media_duration
        for a, b in windows
    ):
        raise ValueError("Removal window has missing/invalid endpoints; no clipping")
    return windows
