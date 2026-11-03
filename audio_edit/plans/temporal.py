from __future__ import annotations

from dataclasses import dataclass, field
import re
import math
from typing import Any, Literal


TemporalKind = Literal[
    "unspecified",
    "point",
    "interval",
    "boundary",
    "event",
    "relative",
]
TemporalRelation = Literal["before", "at", "after", "during", "between"]
ReferenceSource = Literal["audio", "visual", "either"]
ResolutionStatus = Literal["not_requested", "resolved", "approximate", "estimated", "unresolved"]


@dataclass
class TemporalConstraint:
    """A planner-level temporal request.

    Every field is optional so old/simple plans remain valid. ``occurrence`` is
    either ``first``, ``last``, ``each`` or a one-based integer.
    """

    kind: TemporalKind = "unspecified"
    start_time: float | None = None
    end_time: float | None = None
    boundary: Literal["beginning", "end", "middle"] | None = None
    placement_mode: Literal["overlay", "insert", "prepend", "append"] = "overlay"
    relation: TemporalRelation | None = None
    reference_event: str | list[str] | None = None
    end_reference_event: str | list[str] | None = None
    reference_source: ReferenceSource = "either"
    occurrence: Literal["first", "last", "each"] | int | None = None
    offset_seconds: float = 0.0
    count: int | None = None
    spacing_seconds: float | None = None
    approximate: bool = False
    reference_text: str | None = None
    reference_kind: str | None = None
    boundary_role: str | None = None
    overflow_policy: str = "error"
    window_seconds: float | None = None


@dataclass
class ResolvedPlacement:
    start_time: float
    end_time: float | None = None
    reference_event: str | None = None
    reference_source: Literal["audio", "visual"] | None = None
    occurrence_index: int | None = None
    approximate: bool = False
    confidence: float | None = None
    event_id: str | None = None
    resolution_status: str = "resolved"
    evidence: str | None = None


@dataclass
class TemporalResolution:
    status: ResolutionStatus
    placements: list[ResolvedPlacement] = field(default_factory=list)
    reason: str | None = None
    target_description: str | None = None


def temporal_constraint_from_dict(
    value: dict[str, Any] | TemporalConstraint | None,
    *,
    allow_timeline_operations: bool = False,
) -> TemporalConstraint | None:
    if value is None or isinstance(value, TemporalConstraint):
        return value
    if not isinstance(value, dict):
        raise ValueError("temporal must be an object or null")

    kind = value.get("kind", "unspecified")
    if kind not in {"unspecified", "point", "interval", "boundary", "event", "relative"}:
        raise ValueError(f"Unsupported temporal kind: {kind}")

    occurrence = value.get("occurrence")
    if isinstance(occurrence, str) and occurrence.isdigit():
        occurrence = int(occurrence)
    if occurrence not in {None, "first", "last", "each"} and not (
        isinstance(occurrence, int) and not isinstance(occurrence, bool) and occurrence >= 1
    ):
        raise ValueError(f"Unsupported occurrence selector: {occurrence}")

    reference_source = value.get("reference_source") or "either"
    if reference_source not in {"audio", "visual", "either"}:
        if kind in {"event", "relative"}:
            raise ValueError(f"Unsupported reference_source: {reference_source}")
        reference_source = "either"
    placement_mode = value.get("placement_mode") or "overlay"
    modes = (
        {"overlay", "insert", "prepend", "append"}
        if allow_timeline_operations
        else {"overlay", "append"}
    )
    if placement_mode not in modes:
        raise ValueError(f"Unsupported placement_mode: {placement_mode}")
    boundary = value.get("boundary")
    if boundary not in {None, "beginning", "end", "middle"}:
        raise ValueError(f"Unsupported boundary: {boundary}")
    relation = value.get("relation")
    if isinstance(relation, str):
        if relation.strip().casefold() in {
            "middle",
            "in the middle",
            "in the middle of the audio",
            "in the middle of the media",
        }:
            kind, boundary, relation = "boundary", "middle", None
    if isinstance(relation, str):
        relation = {
            "when": "during",
            "while": "during",
        }.get(relation.strip().casefold(), relation)
    if relation not in {None, "before", "at", "after", "during", "between"}:
        raise ValueError(f"Unsupported temporal relation: {relation}")

    count = _optional_int(value.get("count"), "count")
    if count is not None and count < 1:
        raise ValueError("count must be at least 1")
    spacing_seconds = _optional_float(value.get("spacing_seconds"), "spacing_seconds")
    if spacing_seconds is not None and spacing_seconds <= 0:
        raise ValueError("spacing_seconds must be positive")

    return TemporalConstraint(
        kind=kind,
        start_time=_optional_float(value.get("start_time"), "start_time"),
        end_time=_optional_float(value.get("end_time"), "end_time"),
        boundary=boundary,
        placement_mode=placement_mode,
        relation=relation,
        reference_event=value.get("reference_event"),
        end_reference_event=value.get("end_reference_event"),
        reference_source=reference_source,
        occurrence=occurrence,
        offset_seconds=float(value.get("offset_seconds") or 0.0),
        count=count,
        spacing_seconds=spacing_seconds,
        approximate=bool(value.get("approximate", False)),
        reference_text=value.get("reference_text"),
        reference_kind=value.get("reference_kind"),
        boundary_role=value.get("boundary_role"),
        overflow_policy=value.get("overflow_policy", "error"),
        window_seconds=_optional_float(value.get("window_seconds"), "window_seconds"),
    )


def resolved_placement_from_dict(
    value: dict[str, Any] | ResolvedPlacement,
) -> ResolvedPlacement:
    if isinstance(value, ResolvedPlacement):
        return value
    return ResolvedPlacement(
        start_time=float(value["start_time"]),
        end_time=_optional_float(value.get("end_time"), "end_time"),
        reference_event=_optional_text(value.get("reference_event")),
        reference_source=value.get("reference_source"),
        occurrence_index=_optional_int(value.get("occurrence_index"), "occurrence_index"),
        approximate=bool(value.get("approximate", False)),
        confidence=_optional_float(value.get("confidence"), "confidence"),
        event_id=value.get("event_id"),
        resolution_status=value.get("resolution_status", "resolved"),
        evidence=value.get("evidence"),
    )


def resolve_temporal_constraint(
    constraint: TemporalConstraint | None,
    *,
    media_duration: float | None = None,
    event_duration: float | None = None,
    audio_analysis: dict[str, Any] | None = None,
    visual_analysis: dict[str, Any] | None = None,
) -> TemporalResolution:
    """Resolve a generic constraint without guessing missing event timestamps."""

    if constraint is None or constraint.kind == "unspecified":
        return TemporalResolution(status="not_requested")

    if constraint.kind == "point":
        if constraint.start_time is None:
            return _unresolved("A point constraint is missing start_time.")
        return _from_start(
            constraint.start_time,
            constraint.end_time,
            event_duration,
            constraint,
            media_duration,
        )

    if constraint.kind == "interval":
        if constraint.boundary is not None:
            if (
                constraint.reference_event
                or constraint.end_reference_event
                or constraint.relation
                or constraint.offset_seconds
            ):
                return _unresolved(
                    "A file-relative window cannot also declare independent endpoints/events."
                )
            if media_duration is None or not math.isfinite(media_duration) or media_duration <= 0:
                return _unresolved("File-relative interval needs media duration.")
            width = constraint.window_seconds
            explicit = constraint.start_time is not None or constraint.end_time is not None
            if explicit:
                a, b = constraint.start_time, constraint.end_time
                if (
                    a is None
                    or b is None
                    or not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b <= media_duration)
                ):
                    return _unresolved(
                        "A redundant file window needs two valid explicit endpoints."
                    )
                width = b - a if width is None else width
            if width is None and constraint.approximate:
                width = min(3.0, media_duration)
            if width is None or not math.isfinite(width) or not 0 < width <= media_duration:
                return _unresolved(
                    "File-relative interval needs a positive in-file width; no clipping."
                )
            starts = {
                "beginning": 0.0,
                "middle": (media_duration - width) / 2,
                "end": media_duration - width,
            }
            if constraint.boundary not in starts:
                return _unresolved("Unknown file interval boundary.")
            if explicit and not (
                math.isclose(constraint.start_time, starts[constraint.boundary], abs_tol=1e-8)
                and math.isclose(
                    constraint.end_time, starts[constraint.boundary] + width, abs_tol=1e-8
                )
            ):
                return _unresolved(
                    "Explicit endpoints contradict the file-relative window; no precedence or clipping."
                )
            result = _from_start(
                starts[constraint.boundary],
                starts[constraint.boundary] + width,
                None,
                constraint,
                media_duration,
            )
            if constraint.approximate and result.status == "resolved":
                result.status = "approximate"
                result.reason = (
                    "File interval uses the declared approximate width"
                    if constraint.window_seconds is not None
                    else "Vague file interval uses the existing up-to-three-second default"
                )
                for placement in result.placements:
                    placement.resolution_status = "approximate"
                    placement.evidence = result.reason
            return result
        if constraint.start_time is None or constraint.end_time is None:
            return _unresolved("An interval constraint needs both start_time and end_time.")
        return _from_start(
            constraint.start_time,
            constraint.end_time,
            None,
            constraint,
            media_duration,
        )

    if constraint.kind == "boundary":
        if constraint.boundary not in {"beginning", "end", "middle"}:
            return _unresolved("A boundary constraint must specify beginning or end.")
        if event_duration is None:
            return _unresolved("Boundary placement needs a generated/event duration.")
        if constraint.boundary == "beginning":
            start = 0.0
        elif media_duration is None:
            return _unresolved("End placement needs the media duration.")
        elif constraint.boundary == "middle":
            start = max(0.0, (media_duration - event_duration) / 2)
        elif constraint.placement_mode == "append":
            start = media_duration
        else:
            start = max(0.0, media_duration - event_duration)
        return _from_start(
            start,
            start + event_duration,
            None,
            constraint,
            None if constraint.placement_mode == "append" else media_duration,
        )

    if not constraint.reference_event:
        return _unresolved("Event-based timing is missing reference_event.")

    if constraint.kind == "relative" and constraint.relation == "between":
        return _resolve_between(
            constraint, media_duration, event_duration, audio_analysis, visual_analysis
        )

    candidates = _matching_events(
        constraint.reference_event,
        constraint.reference_source,
        audio_analysis,
        visual_analysis,
    )
    if not candidates:
        source = constraint.reference_source
        return _unresolved(
            f"No timed {source} event matched reference_event '{constraint.reference_event}'."
        )

    selected, error = _select_occurrences(candidates, constraint.occurrence)
    if error:
        return _unresolved(error)

    placements: list[ResolvedPlacement] = []
    for event in selected:
        event_start = event.get("start_time")
        event_end = event.get("end_time")
        boundary_audit = None
        boundary_status = None
        if constraint.boundary_role:
            from audio_edit.plans.temporal_boundaries import event_boundary

            key = event.get("event_id", "legacy_reference")
            try:
                boundary, boundary_status, boundary_audit = event_boundary(
                    dict(
                        event_id=key,
                        edge="end" if constraint.relation == "after" else "start",
                        boundary_role=constraint.boundary_role,
                        reference_text=constraint.reference_text,
                    ),
                    {key: dict(event, event_id=key)},
                )
            except ValueError as exc:
                return _unresolved(str(exc))
        if (
            event_start is None
            and event_end is None
            and not (
                constraint.kind == "relative"
                and constraint.relation in ("before", "after", "at")
                and constraint.boundary_role
                and boundary_audit is not None
            )
        ):
            return _unresolved("A requested occurrence has no measured boundaries")

        if constraint.kind == "event":
            start = event_start if event_start is not None else event_end
            end = event_end
            if end is None and start is not None and event_duration is not None:
                end = start + event_duration
        elif constraint.relation == "after":
            if event_duration is None:
                return _unresolved("After-event placement needs the duration of the edited event.")
            start = (
                boundary
                if constraint.boundary_role
                else event_end
                if event_end is not None
                else event_start
            )
            start = None if start is None else start + constraint.offset_seconds
            end = None if start is None else start + event_duration
        elif constraint.relation == "before":
            if event_duration is None:
                return _unresolved("Before-event placement needs the duration of the edited event.")
            anchor = (
                boundary
                if constraint.boundary_role
                else event_start
                if event_start is not None
                else event_end
            )
            end = None if anchor is None else anchor - constraint.offset_seconds
            start = None if end is None else end - event_duration
        elif constraint.relation == "at":
            start = (
                boundary if constraint.boundary_role else event_start
            ) + constraint.offset_seconds
            end = start + event_duration if event_duration is not None else None
        elif constraint.relation == "during":
            start = event_start
            end = event_end
            if start is not None and event_duration is not None:
                if end is not None and start + event_duration > end + 1e-8:
                    return _unresolved(
                        "Complete event does not fit the referenced window; no shortening allowed"
                    )
                end = start + event_duration
        else:
            return _unresolved("A relative constraint needs relation before, after, or during.")

        normalized = _normalize_interval(
            start, end, None if constraint.overflow_policy == "extend" else media_duration
        )
        if normalized is None:
            return _unresolved(
                "A requested occurrence lies outside constraints; refusing partial execution"
            )
        start, end = normalized
        placements.append(
            ResolvedPlacement(
                start_time=start,
                end_time=end,
                reference_event=str(event.get("event") or constraint.reference_event),
                reference_source=event.get("_source"),
                occurrence_index=_optional_int(event.get("occurrence_index"), "occurrence_index"),
                approximate=bool(
                    event.get("approximate", False)
                    or event.get("resolution_status") == "approximate"
                    or constraint.approximate
                ),
                confidence=_optional_float(event.get("confidence"), "confidence"),
                event_id=event.get("event_id"),
                resolution_status=boundary_status
                or event.get(
                    "resolution_status", "approximate" if event.get("approximate") else "resolved"
                ),
                evidence=boundary_audit
                or event.get(
                    "timing_evidence",
                    "Legacy name/alias matching; no stable event reference supplied",
                ),
            )
        )

    if not placements:
        return _unresolved(
            f"Matched '{constraint.reference_event}', but it had no usable timestamp."
        )

    status = (
        "approximate"
        if any(p.event_id and p.approximate for p in placements)
        and (
            isinstance(constraint.reference_event, list)
            or str(constraint.reference_event).startswith(("audio_", "visual_"))
        )
        else "resolved"
    )
    if constraint.boundary_role:
        status = (
            "estimated"
            if any(p.resolution_status == "estimated" for p in placements)
            else "approximate"
            if any(p.resolution_status == "approximate" for p in placements)
            else "resolved"
        )
    return TemporalResolution(status=status, placements=placements)


def requires_event_grounding(constraint: TemporalConstraint | None) -> bool:
    return bool(constraint and constraint.kind in {"event", "relative"})


def _resolve_between(
    constraint: TemporalConstraint,
    media_duration: float | None,
    event_duration: float | None,
    audio_analysis: dict[str, Any] | None,
    visual_analysis: dict[str, Any] | None,
) -> TemporalResolution:
    """Resolve observed gaps without guessing missing or ambiguous anchors."""
    left = _matching_events(
        constraint.reference_event, constraint.reference_source, audio_analysis, visual_analysis
    )
    if constraint.end_reference_event:
        right = _matching_events(
            constraint.end_reference_event,
            constraint.reference_source,
            audio_analysis,
            visual_analysis,
        )
        pairs = [
            (a, b)
            for a in left
            for b in right
            if a.get("end_time") is not None
            and b.get("start_time") is not None
            and a["end_time"] < b["start_time"]
        ]
    else:
        pairs = [
            (a, b)
            for a, b in zip(left, left[1:])
            if a.get("end_time") is not None
            and b.get("start_time") is not None
            and a["end_time"] < b["start_time"]
        ]
    gaps = [
        {
            "start_time": a["end_time"],
            "end_time": b["start_time"],
            "approximate": a.get("approximate", False) or b.get("approximate", False),
        }
        for a, b in pairs
    ]
    if not gaps:
        return _unresolved("Between-event timing needs two observed, non-overlapping boundaries.")
    selected, error = _select_occurrences(gaps, constraint.occurrence)
    if error:
        return _unresolved(error)
    placements = []
    for gap in selected:
        start = gap["start_time"] + constraint.offset_seconds
        end = gap["end_time"]
        if start < gap["start_time"]:
            return _unresolved("Between-event offset would overlap the first reference.")
        if event_duration is not None:
            if event_duration <= 0 or start + event_duration > end + 1e-9:
                return _unresolved("Generated event does not fit between the reference events.")
            end = start + event_duration
        normalized = _normalize_interval(start, end, media_duration)
        if normalized is None:
            return _unresolved("Between-event gap lies outside the media.")
        placements.append(
            ResolvedPlacement(
                start_time=normalized[0],
                end_time=normalized[1],
                reference_event=constraint.reference_event,
                approximate=gap["approximate"] or constraint.approximate,
            )
        )
    return TemporalResolution(status="resolved", placements=placements)


def expand_repeated_placements(
    placements: list[ResolvedPlacement],
    constraint: TemporalConstraint | None,
    event_duration: float,
) -> list[ResolvedPlacement]:
    """Turn a numeric count into executable placements of one generated event."""

    if not constraint or not constraint.count or constraint.count <= 1:
        return placements
    if event_duration <= 0:
        raise ValueError("Repeated placement needs a positive event_duration")

    expanded: list[ResolvedPlacement] = []
    for base in placements:
        if constraint.spacing_seconds is not None:
            step = constraint.spacing_seconds
        elif constraint.kind == "interval" and base.end_time is not None:
            available = base.end_time - base.start_time
            if constraint.count * event_duration > available + 1e-9:
                raise ValueError(
                    "Repeated events do not fit in the requested interval; "
                    "use a shorter event duration or explicit spacing."
                )
            step = max(0.0, available - event_duration) / (constraint.count - 1)
        else:
            step = event_duration

        for index in range(constraint.count):
            start = round(base.start_time + index * step, 6)
            end = round(start + event_duration, 6)
            if (
                constraint.kind == "interval"
                and base.end_time is not None
                and end > base.end_time + 1e-9
            ):
                raise ValueError("Repeated placement exceeds the requested interval.")
            expanded.append(
                ResolvedPlacement(
                    start_time=start,
                    end_time=end,
                    reference_event=base.reference_event,
                    reference_source=base.reference_source,
                    occurrence_index=base.occurrence_index,
                    approximate=base.approximate,
                    confidence=base.confidence,
                )
            )
    return expanded


def _from_start(
    start: float,
    end: float | None,
    duration: float | None,
    constraint: TemporalConstraint,
    media_duration: float | None,
) -> TemporalResolution:
    if end is None and duration is not None:
        end = start + duration
    normalized = _normalize_interval(start, end, media_duration)
    if normalized is None:
        return _unresolved("The requested time is outside the media or has an invalid interval.")
    start, end = normalized
    return TemporalResolution(
        status="resolved",
        placements=[
            ResolvedPlacement(
                start_time=start,
                end_time=end,
                approximate=constraint.approximate,
            )
        ],
    )


def _matching_events(
    reference: str,
    source: ReferenceSource,
    audio_analysis: dict[str, Any] | None,
    visual_analysis: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if source in {"audio", "either"} and audio_analysis:
        for event in audio_analysis.get("events") or []:
            if isinstance(event, dict):
                candidates.append({**event, "_source": "audio"})
    if source in {"visual", "either"} and visual_analysis:
        for event in visual_analysis.get("visual_events") or []:
            if isinstance(event, dict):
                candidates.append({**event, "_source": "visual"})

    candidates = [
        e
        for e in candidates
        if not (
            e.get("partial_start")
            or e.get("partial_end")
            or e.get("review_context_only")
            or e.get("invalid_crop_boundary")
        )
    ]
    ids = reference if isinstance(reference, list) else [reference]
    from audio_edit.plans.event_ids import observation_time

    if isinstance(reference, list) or any(str(r).startswith(("audio_", "visual_")) for r in ids):
        if any(not any(e.get("event_id") == r for e in candidates) for r in ids):
            return []
        return sorted(
            [
                e
                for e in candidates
                if e.get("event_id") in ids and e.get("resolution_status") != "unresolved"
            ],
            key=observation_time,
        )
    matched = [
        event
        for event in candidates
        if any(
            _event_matches(reference, name)
            for name in [str(event.get("event", "")), *(event.get("aliases") or [])]
            if isinstance(name, str)
        )
    ]
    return sorted(
        matched,
        key=lambda event: (
            event.get("start_time") is None,
            float(event.get("start_time") or 0.0),
            int(event.get("occurrence_index") or 0),
        ),
    )


def _event_matches(reference: str, candidate: str) -> bool:
    reference_tokens = _tokens(reference)
    candidate_tokens = _tokens(candidate)
    if not reference_tokens or not candidate_tokens:
        return False
    if reference_tokens <= candidate_tokens or candidate_tokens <= reference_tokens:
        return True
    intersection = len(reference_tokens & candidate_tokens)
    return (
        intersection >= 2
        and intersection / min(len(reference_tokens), len(candidate_tokens)) >= 0.6
    )


def _tokens(text: str) -> set[str]:
    return {_stem_token(token) for token in re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)}


def _stem_token(token: str) -> str:
    """Small language-agnostic-ish normalization, not an event keyword list."""
    if len(token) > 5 and token.endswith("ing"):
        stem = token[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] in "bdgmnprt":
            stem = stem[:-1]
        return stem
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _select_occurrences(
    events: list[dict[str, Any]],
    occurrence: Literal["first", "last", "each"] | int | None,
) -> tuple[list[dict[str, Any]], str | None]:
    if occurrence == "first":
        return events[:1], None
    if occurrence == "last":
        return events[-1:], None
    if occurrence == "each":
        return events, None
    if isinstance(occurrence, int):
        if occurrence > len(events):
            return (
                [],
                f"Requested occurrence {occurrence}, but only {len(events)} timed events matched.",
            )
        return [events[occurrence - 1]], None
    if len(events) == 1:
        return events, None
    return [], (
        f"Reference event matched {len(events)} occurrences; specify first, last, each, "
        "or a one-based occurrence index."
    )


def _normalize_interval(
    start: float | None,
    end: float | None,
    media_duration: float | None,
) -> tuple[float, float | None] | None:
    if start is None:
        return None
    start = float(start)
    end = float(end) if end is not None else None
    if not math.isfinite(start) or start < 0 or (end is not None and not math.isfinite(end)):
        return None
    if media_duration is not None and (
        start > media_duration or (end is not None and end > media_duration)
    ):
        return None
    if end is not None and end <= start:
        return None
    return start, end


def _unresolved(reason: str) -> TemporalResolution:
    return TemporalResolution(status="unresolved", reason=reason)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric or null") from exc


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer or null")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer or null") from exc
