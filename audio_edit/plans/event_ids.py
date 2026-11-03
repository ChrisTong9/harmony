"""Opaque observation IDs. IDs identify observations, not ground-truth boundaries."""

import hashlib
import json
import math


def has_measured_timing(event):
    """Evidence eligibility, independent of a sound's name or any benchmark."""

    def finite_number(value):
        return (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        )

    if any(finite_number(event.get(k)) for k in ("start_time", "end_time")):
        return True
    return any(
        isinstance(m, dict) and finite_number(m.get("time"))
        for m in (event.get("boundaries") or {}).values()
    )


def observation_time(event):
    """Order partial observations by a measured boundary, never by a guessed span."""
    values = [event.get(k) for k in ("start_time", "end_time")]
    phases = event.get("boundaries")
    phases = phases if isinstance(phases, dict) else {}
    values.extend(m.get("time") for m in phases.values() if isinstance(m, dict))
    return min(
        (
            v
            for v in values
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        ),
        default=float("inf"),
    )


def assign_event_ids(analysis, source="audio", media_duration=None):
    if analysis is None:
        return None
    key = "events" if source == "audio" else "visual_events"
    seen = {}
    unique = []
    for event in analysis.get(key, []):
        identity = {
            k: event.get(k) for k in ("event", "start_time", "end_time", "occurrence_index")
        }
        token = (
            source
            + "_"
            + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        )
        event_id = event.get("event_id") or token
        raw_key = json.dumps(
            {
                k: v
                for k, v in event.items()
                if k not in ("event_id", "resolution_status", "timing_evidence")
            },
            sort_keys=True,
        )
        if event_id in seen:
            if seen[event_id] == raw_key:
                analysis.setdefault("identity_audit", []).append(
                    dict(code="exact_duplicate_observation", event_id=event_id)
                )
                continue
            if event.get("event_id"):
                raise ValueError("Conflicting observation event_id: " + str(event_id))
            event_id += "_" + hashlib.sha256(raw_key.encode()).hexdigest()[:12]
            if event_id in seen:
                if seen[event_id] == raw_key:
                    continue
                raise ValueError("Conflicting generated observation ID")
        seen[event_id] = raw_key
        unique.append(event)
        event["event_id"] = event_id
        valid = not event.get("invalid_crop_boundary", False)
        for edge in ("start_time", "end_time"):
            v = event.get(edge)
            if (
                v is None
                or not math.isfinite(float(v))
                or float(v) < 0
                or (media_duration is not None and float(v) > media_duration)
            ):
                valid = False
        if valid and float(event["end_time"]) <= float(event["start_time"]):
            valid = False
        if event.get("timing_kind") == "boundary" and all(
            event.get(k) is None for k in ("start_time", "end_time")
        ):
            phases = event.get("boundaries") or {}
            valid = not event.get("invalid_crop_boundary", False) and any(
                role in ("onset", "action_end", "sound_end", "utterance_end")
                and isinstance(m, dict)
                and isinstance(m.get("time"), (int, float))
                and not isinstance(m["time"], bool)
                and math.isfinite(m["time"])
                and 0 <= m["time"]
                and (media_duration is None or m["time"] <= media_duration)
                and m.get("evidence")
                and not m.get("partial")
                and m.get("status") != "unresolved"
                for role, m in phases.items()
            )

            event["media_duration"] = media_duration
        confidence = event.get("confidence")
        if confidence is not None and (
            not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1
        ):
            valid = False
        event["resolution_status"] = (
            "unresolved"
            if not valid
            else "approximate"
            if event.get("approximate", True) or confidence is None or float(confidence) < 0.8
            else "resolved"
        )
        event.setdefault(
            "timing_evidence",
            "Observed by "
            + source
            + " perception; boundaries are model estimates, not forced alignment.",
        )
    analysis[key] = unique
    return analysis


EVENT_REFERENCE_RULES = """
When analysis is supplied, use its stable event_id as temporal.reference_event
and temporal.end_reference_event, never rephrase an observed event name for matching.
For first/last/each of a semantic category, reference_event may be a JSON array of
the observed IDs in that category; occurrence chooses among them deterministically.
If the target is absent, leave it unresolved. A search window is NOT a detected target.
Do not invent a central-third target interval when a named sound was not found.
"""
