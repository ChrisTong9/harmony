"""Perceptual source association to trusted waveform regions, without new seconds."""

from copy import deepcopy

from audio_edit.analyzers.measurement_tools import PHYSICAL_FIELDS, measurement_question_ids
from audio_edit.analyzers.event_groups import measured_group, GROUP_FIELDS


MEASURED_KINDS = {"measured_region", "measured_group"}


def _region_ids(event):
    return (
        event.get("region_ids", [])
        if event.get("timing_kind") == "measured_group"
        else [event.get("region_id")]
    )


def without_rejected_regions(questions, rejected):
    """Apply rejection to every refreshed question, including coupled facts."""
    result = deepcopy(questions)
    for q in result:
        measurement = q.get("measured_regions")
        if not measurement:
            continue
        excluded = {
            region_id
            for e in rejected
            if e.get("question_id") == q["id"]
            and e.get("audio_sha256") == measurement["audio_sha256"]
            for region_id in e.get("region_ids", [e.get("region_id")])
        }
        measurement["regions"] = [
            r for r in measurement["regions"] if r["region_id"] not in excluded
        ]
        if not measurement["regions"]:
            q.pop("measured_regions")
    return result


def with_measured_regions(questions, measurement, *, rejected=()):
    result = deepcopy(questions)
    eligible = set(measurement_question_ids(questions))
    for q in result:
        q.pop("measured_regions", None)
        if not measurement or q["id"] not in eligible or q["id"] not in measurement["question_ids"]:
            continue
        regions = [deepcopy(r) for r in measurement["regions"] if r["complete"]]
        if regions:
            q["measured_regions"] = dict(
                regions=regions,
                **{
                    k: deepcopy(measurement[k])
                    for k in (
                        "tool",
                        "audio_sha256",
                        "original_audio_sha256",
                        "source_description",
                        "resolution_seconds",
                        "method",
                        "limitation",
                    )
                    if k in measurement
                },
            )
            if q.get("occurrence") == "each" and measurement["tool"] == "source_activity":
                q["measured_regions"]["inventory_required"] = True
    return without_rejected_regions(result, rejected)


def resolve_measured_references(data, questions, *, recover=False, check_inventory=True):
    """Only an explicit positive association may supply a region's timing.

    Free-form evidence is not a boolean match. Old payloads without the explicit
    association decision fail closed, even when they contain a plausible region
    ID. Negative/unknown decisions remain auditable without becoming events.
    """

    for event in data.get("events", []):
        if isinstance(event, dict):
            event.pop("measurement_provenance", None)
    if check_inventory:
        from audio_edit.analyzers.source_inventory import validate_source_inventories

        validate_source_inventories(data, questions)
    if recover:
        return _recover_references(data, questions)
    known = {q["id"]: q for q in questions}
    selected = {}
    for key, fact in data.get("facts", {}).items():
        for event in fact.get("events", []):
            if event.get("timing_kind") not in MEASURED_KINDS:
                event.pop("measurement_provenance", None)
                continue
            q = known.get(key, {})
            measurement = q.get("measured_regions", {})
            grouped = event["timing_kind"] == "measured_group"
            region = (
                measured_group(q, event)
                if grouped
                else next(
                    (
                        r
                        for r in measurement.get("regions", [])
                        if r["region_id"] == event.get("region_id")
                    ),
                    None,
                )
            )
            if (
                not region
                or not region["complete"]
                or q.get("focus") != "timing"
                or not set(q["required_fields"]) <= PHYSICAL_FIELDS
            ):
                raise ValueError(
                    "Measured reference must identify an available physical region for this question"
                )
            expected = (
                GROUP_FIELDS
                if grouped
                else {
                    "timing_kind",
                    "event",
                    "occurrence_index",
                    "region_id",
                    "association_status",
                    "association_evidence",
                }
            )
            if set(event) != expected:
                raise ValueError(
                    "Measured references cannot override times, boundaries or provenance"
                )
            if (
                not isinstance(event["association_evidence"], str)
                or not event["association_evidence"].strip()
            ):
                raise ValueError("Measured reference requires audible source-association evidence")
            if event["association_status"] not in ("match", "no_match", "unknown"):
                raise ValueError(
                    "Measured reference needs an explicit match, no_match or unknown association"
                )
            if event["association_status"] != "match" or fact.get("status") != "answered":
                raise ValueError(
                    "Only an answered fact with association_status=match can bind a measured region"
                )
            identity = (event["event"], event["occurrence_index"])
            for region_id in _region_ids(event):
                token = (measurement["audio_sha256"], region_id)
                if token in selected and selected[token] != identity:
                    raise ValueError(
                        "A merged activity region cannot certify separate named events or occurrences"
                    )
                selected[token] = identity
            raw = deepcopy(event)
            event.update(
                timing_kind="interval",
                start_time=region["start_time"],
                end_time=region["end_time"],
                approximate=True,
                timing_source=measurement["tool"],
                resolution_status="estimated",
                boundaries={
                    role: dict(
                        time=region[edge],
                        approximate=True,
                        timing_source=measurement["tool"],
                        evidence=raw["association_evidence"],
                    )
                    for role, edge in [("onset", "start_time"), ("sound_end", "end_time")]
                },
            )
            event["measurement_provenance"] = dict(
                tool=measurement["tool"],
                audio_sha256=measurement["audio_sha256"],
                original_audio_sha256=measurement.get("original_audio_sha256"),
                method=measurement["method"],
                resolution_seconds=measurement["resolution_seconds"],
                association_status="match",
                identity_source="Qwen explicit association of the entire isolated component; no semantic identity from energy",
                original_observation=raw,
            )
            if grouped:
                event["measurement_provenance"].update(
                    region_ids=region["region_ids"],
                    components=region["components"],
                    group_relation=raw["group_relation"],
                    identity_source="Qwen explicitly identifies consecutive measured components as one selected event sequence; energy alone does not group events",
                )
                event.pop("region_ids")
                event.pop("group_relation")
            else:
                event["measurement_provenance"]["region_id"] = region["region_id"]
                event.pop("region_id")
            event.pop("association_evidence")
            event.pop("association_status")


def _recover_references(data, questions):
    """Reject a failed question's timing, not unrelated valid listening facts.

    Reject every member of a collision, independent of response order. Preserve
    identities only as unlocated observations; they can request separation but
    cannot authorize any edit. Never parse seconds out of association prose.
    """
    if "facts" not in data:
        return
    raw = deepcopy(data["facts"])
    known = {q["id"]: q for q in questions}
    tokens, bad = {}, {}
    for key, fact in raw.items():
        for index, event in enumerate(fact.get("events", [])):
            if event.get("timing_kind") not in MEASURED_KINDS:
                continue
            try:
                trial = dict(facts={key: dict(fact, events=[deepcopy(event)])})
                resolve_measured_references(trial, questions, check_inventory=False)
            except (ValueError, TypeError) as exc:
                bad.setdefault(key, []).append((index, str(exc)))
                continue
            measurement = known[key]["measured_regions"]
            for region_id in _region_ids(event):
                token = (measurement["audio_sha256"], region_id)
                tokens.setdefault(token, []).append(
                    (key, index, event["event"], event["occurrence_index"])
                )
    for entries in tokens.values():
        if len({(e[2], e[3]) for e in entries}) > 1:
            for key, index, _, _ in entries:
                bad.setdefault(key, []).append(
                    (
                        index,
                        "A merged activity region cannot certify separate named events or occurrences",
                    )
                )

    changed = True
    while changed:
        changed = False
        for key, fact in raw.items():
            if key not in bad and any(
                e.get("timing_kind") == "reference"
                and e.get("observation_ref", {}).get("question_id") in bad
                for e in fact.get("events", [])
            ):
                bad[key] = [(None, "The referenced question has unresolved timing")]
                changed = True
    errors, unlocated = [], []
    for key, issues in bad.items():
        measurement = known.get(key, {}).get("measured_regions", {})
        fact = raw[key]
        for index, reason in issues:
            event = fact["events"][index] if index is not None else {}
            errors.append(
                dict(
                    question_id=key,
                    region_id=event.get("region_id"),
                    audio_sha256=measurement.get("audio_sha256"),
                    reason=reason,
                    association_status=event.get("association_status", "missing"),
                    association_evidence=event.get("association_evidence"),
                )
            )
            if event.get("timing_kind") == "measured_group":
                errors[-1]["region_ids"] = deepcopy(event.get("region_ids", []))
        for event in fact.get("events", []):
            if event.get("event") and (
                event.get("timing_kind") not in MEASURED_KINDS
                or event.get("association_status") == "match"
                and fact.get("status") == "answered"
            ):
                unlocated.append(
                    dict(
                        question_id=key,
                        event=event["event"],
                        occurrence_index=event.get("occurrence_index"),
                        evidence=fact.get("answer"),
                        timing_status="unresolved",
                        reason="Measured association rejected; no usable event time",
                    )
                )
        data["facts"][key] = dict(status="uncertain", answer=fact["answer"], events=[])
    resolve_measured_references(data, questions, check_inventory=False)
    if errors:
        data.setdefault("measurement_reference_errors", []).extend(errors)
        data["unlocated_observations"] = unlocated
