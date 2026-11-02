"""Semantic boundary selection, separate from asset duration and timeline policy.

No sound-name rules: the planner supplies the meaning, perception supplies evidence.
Optional phase measurements never invalidate otherwise usable legacy observations.
"""

from copy import deepcopy
import math


BOUNDARY_ROLES = ("onset", "action_end", "sound_end", "utterance_end")


class BoundaryEvidenceError(ValueError):
    pass


def finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def boundary_measurement(event, role):
    phases = event.get("boundaries")
    item = phases.get(role) if isinstance(phases, dict) else None
    if not isinstance(item, dict) or not finite(item.get("time")) or item.get("partial"):
        return None
    start, end = event.get("start_time"), event.get("end_time")
    if event.get("timing_kind") == "boundary" and start is None and end is None:
        duration = event.get("media_duration")
        if item["time"] < 0 or (finite(duration) and item["time"] > duration):
            return None
    elif not finite(start) or not finite(end) or not start <= item["time"] <= end:
        return None
    if not item.get("evidence") or item.get("status") == "unresolved":
        return None
    return item


def select_event(anchor, events, *, best_effort=False):
    ids = anchor.get("event_id")
    ids = ids if isinstance(ids, list) else [ids]
    candidates = [
        events[i]
        for i in ids
        if isinstance(i, str)
        and i in events
        and events[i].get("resolution_status") != "unresolved"
        and not events[i].get("review_context_only")
    ]
    from audio_edit.plans.event_ids import observation_time

    candidates.sort(key=observation_time)
    occurrence = anchor.get("occurrence")
    reason = None
    if not candidates:
        if not best_effort:
            raise BoundaryEvidenceError("Unknown or unlocated event_id: " + str(ids))
        return (
            None,
            "Reference not located; use an explicitly estimated scene position, not a measured event boundary.",
        )
    if occurrence == "each":
        raise BoundaryEvidenceError("each must be expanded into independent edit anchors")
    if occurrence in ("first", "last"):
        selected = candidates[0 if occurrence == "first" else -1]
    elif isinstance(occurrence, int) and not isinstance(occurrence, bool) and occurrence > 0:
        indexed = [e for e in candidates if e.get("occurrence_index") == occurrence]
        if len(indexed) == 1:
            return indexed[0], reason
        if occurrence > len(candidates):
            if not best_effort:
                raise BoundaryEvidenceError("Requested occurrence was not observed")
            reason = "Requested ordinal not observed; nearest available ordinal is estimated, not a confirmed match."
        selected = candidates[min(occurrence, len(candidates)) - 1]
    elif len(candidates) == 1:
        selected = candidates[0]
    elif occurrence is None:
        if not best_effort:
            raise BoundaryEvidenceError("Multiple reference events without an occurrence selector")
        selected = candidates[0]
        reason = "Ordinal unspecified among multiple candidates; earliest candidate chosen as an explicit estimated default."
    else:
        raise ValueError("Invalid occurrence selector")
    return selected, reason


def event_boundary(anchor, events, *, best_effort=False):
    event, estimate_reason = select_event(anchor, events, best_effort=best_effort)
    evidence = dict(anchor)
    evidence["candidate_ids"] = anchor.get("event_id")
    if event is None:
        available = [
            e["start_time"]
            for e in events.values()
            if finite(e.get("start_time"))
            and not e.get("review_context_only")
            and e.get("resolution_status") != "unresolved"
        ]
        value = min(available) if available else 0.0
        evidence.update(
            reason=estimate_reason,
            selected_event_id=None,
            basis="First observed scene onset, or recording start when no timing evidence exists",
        )
        return value, "estimated", evidence
    edge = anchor.get("edge", "end" if anchor.get("relation") == "after" else "start")
    if edge not in ("start", "end"):
        raise ValueError("Event anchor requires start/end edge")
    role = anchor.get("boundary_role")
    if role is not None and role not in BOUNDARY_ROLES:
        raise ValueError("Unknown semantic boundary role: " + str(role))
    measurement = boundary_measurement(event, role) if role else None
    if role and measurement is None:
        if role == "onset" and finite(event.get("start_time")) and not event.get("partial_start"):
            measurement = dict(
                time=event["start_time"],
                evidence="Complete observed event onset",
                status=event.get("resolution_status", "approximate"),
            )
        elif not best_effort:
            raise BoundaryEvidenceError(
                "Insufficient boundary granularity: " + str(event["event_id"]) + "/" + role
            )
        else:
            estimate_reason = estimate_reason or (
                "Required " + role + " not measured after bounded review; "
                "using available event edge as an estimate, not a phase measurement."
            )
    if measurement:
        value = measurement["time"]
        status = measurement.get(
            "status", "approximate" if measurement.get("approximate") else "resolved"
        )
        confidence = measurement.get("confidence")
        if finite(confidence) and confidence < 0.75 and status == "resolved":
            status = "approximate"
        if status == "resolved" and event.get("resolution_status") in ("approximate", "estimated"):
            status = event["resolution_status"]
        evidence["measurement"] = deepcopy(measurement)
    else:
        if event.get("partial_" + edge) and not best_effort:
            raise BoundaryEvidenceError("Cannot use a truncated observation boundary")
        if event.get("partial_" + edge):
            estimate_reason = "Only a partial observation exists; fragment edge is an estimate, not a measured boundary."
        value = event.get(edge + "_time")
        if not finite(value):
            raise BoundaryEvidenceError("Missing usable observed boundary")
        status = event.get("resolution_status", "approximate")
    if estimate_reason:
        status = "estimated"
    evidence.update(
        selected_event_id=event.get("event_id"),
        observation=deepcopy(event),
        selected_boundary=value,
        boundary_role=role or "legacy_" + edge,
        reason=estimate_reason
        or (measurement or {}).get("evidence")
        or event.get("timing_evidence"),
    )
    return value, status, evidence


def uncertain_anchors(value, events):
    if isinstance(value, dict):
        if "event_id" in value:
            if value.get("occurrence") == "each":
                ids = (
                    value["event_id"]
                    if isinstance(value["event_id"], list)
                    else [value["event_id"]]
                )
                return not ids or any(
                    uncertain_anchors(dict(value, event_id=i, occurrence=None), events) for i in ids
                )
            try:
                _, status, _ = event_boundary(value, events)
                return status != "resolved"
            except (ValueError, TypeError, KeyError):
                return True
        return any(uncertain_anchors(v, events) for v in value.values())
    return isinstance(value, list) and any(uncertain_anchors(v, events) for v in value)


def estimate_action_boundaries(analysis, input_audio):
    """Last strong-body frame, not a source classifier. Only an estimated fallback.

    This is applied AFTER the one perceptual review; quiet actions and overlapping
    sources are ambiguous. It never overwrites a measured phase or labels it exact.
    """
    import numpy as np
    from audio_edit.utils.execution_outcome import open_decoded_audio

    result = deepcopy(analysis or {})
    with open_decoded_audio(input_audio) as source:
        for event in result.get("events", []):
            if boundary_measurement(event, "action_end"):
                continue
            a, b = event.get("start_time"), event.get("end_time")
            if not finite(a) or not finite(b) or not 0 <= a < b <= len(source) / source.samplerate:
                continue
            first, last = round(a * source.samplerate), round(b * source.samplerate)
            source.seek(first)
            data = source.read(last - first, dtype="float32", always_2d=True)
            hop = max(1, round(0.02 * source.samplerate))
            energy = np.array(
                [
                    np.sqrt(np.mean(data[i : i + hop].astype("float64") ** 2))
                    for i in range(0, len(data), hop)
                ]
            )
            if not len(energy) or not np.isfinite(energy).all() or energy.max() < 1e-8:
                continue
            if event.get("partial_start") and energy[0] >= 0.35 * energy.max():
                continue
            active = np.flatnonzero(energy >= 0.35 * energy.max())
            value = min(last, first + (int(active[-1]) + 1) * hop) / source.samplerate
            if not isinstance(event.get("boundaries"), dict):
                event["boundaries"] = {}
            event["boundaries"]["action_end"] = dict(
                time=value,
                status="estimated",
                evidence="After bounded perception review: last 20ms frame above 35% of candidate peak RMS; "
                "strong-body proxy only, not source-separated or a confirmed semantic endpoint.",
                quiet_prefix_observed=bool(energy[0] < 0.35 * energy.max()),
                model_reported_partial_start=bool(event.get("partial_start")),
            )
    return result


def expand_each(edits, events):
    """One edit per occurrence; no arithmetic by the language model."""
    result = []
    for edit in edits:
        anchor = edit.get("window", {}).get("start", {})
        if not isinstance(anchor, dict) or anchor.get("occurrence") != "each":
            result.append(edit)
            continue
        ids = anchor.get("event_id", [])
        ids = ids if isinstance(ids, list) else [ids]
        if not ids:
            result.append(edit)
            continue
        from audio_edit.plans.event_ids import observation_time

        ids = sorted(ids, key=lambda i: observation_time(events.get(i, {})))
        individual = edit.get("placement_anchors")
        enumerated = (
            isinstance(individual, list)
            and len(individual) == len(ids)
            and all(isinstance(a, dict) and isinstance(a.get("event_id"), str) for a in individual)
            and set(a["event_id"] for a in individual) == set(ids)
        )
        for event_id in ids:

            def replace(value):
                if isinstance(value, dict):
                    if (
                        value.get("occurrence") == "each"
                        and value.get("event_id") == anchor["event_id"]
                    ):
                        return dict(value, event_id=event_id, occurrence=None)
                    return {k: replace(v) for k, v in value.items()}
                return [replace(v) for v in value] if isinstance(value, list) else value

            expanded = replace(deepcopy(edit))

            if (
                edit.get("window", {}).get("end") is None
                and edit.get("coverage") == "repeated"
                and edit.get("count") == len(ids)
                and edit.get("offsets_seconds") in (None, [])
                and (not individual or enumerated)
            ):
                expanded.update(coverage="single", count=1, offsets_seconds=[0])
                if enumerated:
                    expanded["placement_anchors"] = [
                        deepcopy(next(a for a in individual if a["event_id"] == event_id))
                    ]
                expanded["reason"] = (
                    edit.get("reason", "")
                    + "; Canonicalized total each-reference count into one independent event per reference."
                )
            result.append(expanded)
    return result
