"""One bounded perception review: search ranges are never executed edit ranges."""

from copy import deepcopy
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from audio_edit.analyzers.qwen_audio_analyzer import extract_json_object
from audio_edit.plans.event_ids import assign_event_ids
from audio_edit.plans.temporal_boundaries import select_event, finite, boundary_measurement


def reference_anchors(value):
    if isinstance(value, dict):
        if "event_id" in value:
            yield value
        else:
            for child in value.values():
                yield from reference_anchors(child)
    elif isinstance(value, list):
        for child in value:
            yield from reference_anchors(child)


def _referenced_search_one(boundary_requests, analysis, duration):
    """Prefer a complete short reference over a guessed last/cropped tail.

    This selects a SEARCH range only. It never grants an execution window or
    claims the reference has been localized. Long references use the prior path.
    """
    events = {e.get("event_id"): e for e in (analysis or {}).get("events", [])}
    for anchor in reference_anchors(boundary_requests):
        if anchor.get("occurrence") == "each":
            ids = (
                anchor["event_id"] if isinstance(anchor["event_id"], list) else [anchor["event_id"]]
            )
            candidates = [
                events[i]
                for i in ids
                if i in events
                and finite(events[i].get("start_time"))
                and finite(events[i].get("end_time"))
            ]
            if not ids or len(candidates) != len(ids):
                return None
            if candidates:
                a, b = (
                    min(e["start_time"] for e in candidates),
                    max(e["end_time"] for e in candidates),
                )
                if 0 <= a < b <= duration and b - a <= 15:
                    return dict(
                        purpose="search",
                        regions=[
                            dict(
                                start_seconds=a,
                                end_seconds=b,
                                reason="Recheck all referenced occurrences together; search range only",
                            )
                        ],
                        reference_anchor=anchor,
                        reason="One bounded search preserves each, not first-only review",
                    )

            regions = []
            for event_id in ids:
                search = _referenced_search_one(
                    dict(anchor, event_id=event_id, occurrence=None), analysis, duration
                )
                if search is None:
                    return None
                regions.extend(search["regions"])
            return dict(
                purpose="search",
                regions=regions,
                reference_anchor=anchor,
                reason="All distant each occurrences retained; never substitute first",
            )
        try:
            event, reason = select_event(anchor, events, best_effort=True)
        except (ValueError, TypeError, KeyError):
            continue
        if event is None:
            continue
        a, b = event.get("start_time"), event.get("end_time")

        needs_context = anchor.get("measurement") == "beat_onsets" or (
            anchor.get("reference_kind") == "utterance"
            and not boundary_measurement(event, anchor.get("boundary_role"))
        )
        if needs_context and finite(a) and finite(b) and 0 <= a < b <= duration:
            span = min(14.0, duration)
            start = max(0.0, min((a + b - span) / 2, duration - span))
            return dict(
                purpose="search",
                regions=[
                    dict(
                        start_seconds=start,
                        end_seconds=start + span,
                        reason="Retain surrounding utterance/beat sequence; candidate fragment lacks the requested measurement",
                    )
                ],
                reason="Bounded context search only; no inferred semantic endpoint or beat interval",
                reference_anchor=anchor,
            )

        if finite(a) and finite(b):
            for check in event.get("boundary_activity_audit", []):
                end = check.get("search_context", {}).get("end_seconds")
                if check.get("status") in ("conflict", "ambiguous") and finite(end):
                    if 0 <= a < max(b, end) <= duration and max(b, end) - a <= 15:
                        b = max(b, end)
        if finite(a) and finite(b) and 0 <= a < b <= duration and b - a <= 15:
            return dict(
                purpose="search",
                regions=[
                    dict(
                        start_seconds=a,
                        end_seconds=b,
                        reason="Recheck complete short referenced event, including activity and decay; "
                        + (reason or "explicit reference/occurrence retained"),
                    )
                ],
                reason="Deterministic reference search, not an executable edit interval",
                reference_anchor=anchor,
            )
    return None


def referenced_search(boundary_requests, analysis, duration):
    anchors = list(reference_anchors(boundary_requests))
    regions = []
    for anchor in anchors:
        candidate = _referenced_search_one(anchor, analysis, duration)
        if candidate is None:
            return None
        for region in candidate["regions"]:
            if region not in regions:
                regions.append(region)
    if not regions:
        return None
    combined = []
    for region in sorted(regions, key=lambda r: r["start_seconds"]):
        if (
            combined
            and region["start_seconds"] <= combined[-1]["end_seconds"]
            and region["end_seconds"] - combined[-1]["start_seconds"] <= 15
        ):
            combined[-1]["end_seconds"] = max(combined[-1]["end_seconds"], region["end_seconds"])
        else:
            combined.append(dict(region))
    if len(combined) > 3:
        return None
    return dict(
        purpose="search",
        regions=combined,
        reference_anchors=anchors,
        reason="Review all selected endpoints/occurrences within the existing bounded budget; search only",
    )


def validate_search_regions(data, duration):
    if data.get("purpose") != "search":
        raise ValueError("Review requires explicitly labelled search regions")
    regions = data.get("regions", [])
    if not isinstance(regions, list) or len(regions) > 3:
        raise ValueError("At most three focused search regions are allowed")
    result = []
    for region in regions:
        start, end = region["start_seconds"], region["end_seconds"]
        if isinstance(start, bool) or isinstance(end, bool):
            raise ValueError("Search boundaries must be numeric")
        start, end = float(start), float(end)
        if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start < end <= duration:
            raise ValueError("Search region is outside the recording; no clipping")
        if end - start > 15.000001 or not region.get("reason"):
            raise ValueError("Search regions need evidence and must be at most 15 seconds")
        result.append(dict(start_seconds=start, end_seconds=end, reason=region["reason"]))
    return result


def review_audio(
    llm,
    analyzer,
    *,
    input_audio,
    user_prompt,
    target_sound,
    media_duration,
    audio_analysis,
    operation,
    boundary_requests=None,
    max_regions=3,
    perception_prompt=None,
):
    payload = dict(
        instruction=user_prompt,
        target_sound=target_sound,
        operation=operation,
        media_duration=media_duration,
        analysis=audio_analysis,
        boundary_requests=boundary_requests,
    )
    system = """Choose candidate audio fragments to RECHECK an uncertain editing reference.
Return JSON {"purpose":"search", "regions":[{"start_seconds":number,
"end_seconds":number,"reason":"evidence for searching here"}],"reason":"..."}.
Choose at most 3 regions, each <=15 seconds, within the recording. Include context
around the expected target and its timing references. Times here are SEARCH hints,
not measured target boundaries, not executable edits. Use the complete instruction
and observed scene. A middle/end search range may be approximate, explicitly saying
so in reason. Do not equate a search range to an absent target. For add, search for
EXISTING timing references, not the hypothetical new sound. For replace, search for
the OLD sound including non-X/exclusion descriptions. Cover the complete reference
event or the relevant boundary when possible; do not pretend a fragment edge is an
event boundary. If there is no useful bounded search, return regions=[] and a reason."""
    system += f"\nFor this request the hard limit is {max_regions} fragment(s). Prioritize the requested semantic boundary, not the full event envelope."
    search = (
        referenced_search(boundary_requests, audio_analysis, media_duration)
        if operation == "add"
        else None
    )
    if search is None:
        system += "\nPreserve requested occurrence. Unspecified is NOT last. A review does not authorize changing boundary meaning."
        response = llm.client.chat.completions.create(
            model=llm.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        search = extract_json_object(response.choices[0].message.content or "")
    regions = validate_search_regions(search, media_duration)
    combined = []
    for region in sorted(regions, key=lambda r: r["start_seconds"]):
        if (
            combined
            and region["start_seconds"] <= combined[-1]["end_seconds"]
            and max(region["end_seconds"], combined[-1]["end_seconds"])
            - combined[-1]["start_seconds"]
            <= 14
        ):
            combined[-1]["end_seconds"] = max(combined[-1]["end_seconds"], region["end_seconds"])
            combined[-1]["reason"] += "; " + region["reason"]
        else:
            combined.append(dict(region))
    regions = combined
    if len(regions) > max_regions:
        search["budget_note"] = (
            "Only the first prioritized region is reviewed within the call budget."
        )
        regions = regions[:max_regions]
    if not regions:
        return audio_analysis, dict(
            search=search,
            status="unresolved",
            reason=search.get("reason") or "No useful search region",
        )
    reviewed, crops = analyze_regions(
        analyzer,
        input_audio,
        perception_prompt or user_prompt,
        regions,
        audio_analysis,
        target_sound,
        operation,
        boundary_requests,
        timing_only=perception_prompt is not None,
    )
    return reviewed, dict(
        search=search,
        crops=crops,
        status="reviewed",
        method="Qwen on actual cropped audio; Python maps local times to original timeline",
    )


def analyze_regions(
    analyzer,
    input_audio,
    user_prompt,
    regions,
    previous_analysis,
    target_sound="",
    operation="remove",
    boundary_requests=None,
    timing_only=False,
):
    import soundfile as sf

    info = sf.info(input_audio)
    duration = info.frames / info.samplerate
    regions = validate_search_regions({"purpose": "search", "regions": regions}, duration)
    observations, crops = [], []
    with (
        TemporaryDirectory(prefix="audio-grounding-review-") as directory,
        sf.SoundFile(input_audio) as source,
    ):
        for index, region in enumerate(regions):
            first = max(0, math.floor((region["start_seconds"] - 0.5) * info.samplerate))
            last = min(info.frames, math.ceil((region["end_seconds"] + 0.5) * info.samplerate))
            source.seek(first)
            audio = source.read(last - first, dtype="float32", always_2d=True)
            path = Path(directory) / f"review-{index}.wav"
            sf.write(path, audio, info.samplerate, subtype="FLOAT")
            offset, extent = first / info.samplerate, (last - first) / info.samplerate
            prompt = (
                user_prompt
                + f"\nFocused review for {operation}; original target/reference: {target_sound}. "
                "The attached audio is a FRAGMENT. Timestamp relative to this attached fragment, "
                "not the full recording. Locate the existing target and surrounding timing references; "
                "do not hallucinate the requested new sound. Return null boundaries when not located. "
                "A sound reaching a fragment edge may be only a partial event. "
                "Use only the attached fragment timeline. Absolute crop offsets are handled outside perception."
            )
            if (operation == "add" or timing_only) and callable(
                getattr(analyzer, "analyze_timing", None)
            ):
                candidates = [
                    dict(
                        event_id=e.get("event_id"),
                        event=e.get("event"),
                        global_occurrence_index=e.get("occurrence_index"),
                        search_hint_local_start=e["start_time"] - offset,
                        search_hint_local_end=e["end_time"] - offset,
                    )
                    for e in (previous_analysis or {}).get("events", [])
                    if finite(e.get("start_time"))
                    and finite(e.get("end_time"))
                    and e["start_time"] < last / info.samplerate
                    and e["end_time"] > offset
                ]
                analysis = analyzer.analyze_timing(
                    str(path),
                    prompt,
                    dict(
                        references=list(reference_anchors(boundary_requests)),
                        candidates=candidates,
                        candidate_times_are_search_hints=True,
                    ),
                )
            else:
                analysis = analyzer.analyze(str(path), prompt)
            mapped = []
            for raw in analysis.get("events", []):
                event = deepcopy(raw)
                event.pop("event_id", None)
                event.pop("timing_evidence", None)
                local = {edge: event.get(edge + "_time") for edge in ("start", "end")}
                invalid = raw.get("resolution_status") == "unresolved"
                for edge in ("start", "end"):
                    value = local[edge]
                    if value is not None:
                        value = float(value)
                        invalid |= not math.isfinite(value) or not 0 <= value <= extent

                        if math.isfinite(value):
                            sample = first + round(value * info.samplerate)
                            event[edge + "_time"] = sample / info.samplerate
                            event[edge + "_sample"] = sample
                event["invalid_crop_boundary"] = bool(invalid)
                rhythm = event.get("rhythm")
                if isinstance(rhythm, dict):
                    onsets = rhythm.get("onsets_seconds")
                    if isinstance(onsets, list) and all(
                        isinstance(v, (int, float))
                        and not isinstance(v, bool)
                        and math.isfinite(v)
                        and 0 <= v <= extent
                        for v in onsets
                    ):
                        event["rhythm"] = dict(
                            rhythm,
                            onsets_seconds=[
                                (first + round(v * info.samplerate)) / info.samplerate
                                for v in onsets
                            ],
                            crop_offset=offset,
                            local_onsets_seconds=onsets,
                        )
                    else:
                        event.pop("rhythm", None)

                boundaries = event.get("boundaries")
                event["boundaries"] = {}
                for role, measurement in boundaries.items() if isinstance(boundaries, dict) else []:
                    if not isinstance(measurement, dict):
                        continue
                    value = measurement.get("time")
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or not 0 <= value <= extent
                    ):
                        event.setdefault("invalid_optional_boundaries", []).append(role)
                        continue
                    mapped_phase = dict(
                        measurement,
                        time=(first + round(value * info.samplerate)) / info.samplerate,
                        local_time=value,
                        crop_offset=offset,
                    )
                    mapped_phase["partial"] = bool(
                        measurement.get("partial")
                        or (first > 0 and value <= 0.03)
                        or (last < info.frames and value >= extent - 0.03)
                    )
                    event["boundaries"][role] = mapped_phase

                event["partial_start"] = (
                    first > 0 and local["start"] is not None and float(local["start"]) <= 0.03
                )
                event["partial_end"] = (
                    last < info.frames
                    and local["end"] is not None
                    and float(local["end"]) >= extent - 0.03
                )
                event["review_observation"] = dict(
                    crop_start=offset,
                    crop_end=last / info.samplerate,
                    local_start=local["start"],
                    local_end=local["end"],
                )
                if operation == "add":
                    event["local_occurrence_index"] = event.pop("occurrence_index", None)
                    event["occurrence_scope"] = "fragment"
                event["timing_evidence"] = (
                    "Observed in cropped audio; Python added crop offset on the integer sample grid; model-estimated boundaries."
                )
                mapped.append(event)
            assign_event_ids({"events": mapped}, "audio", duration)
            observations.extend(mapped)
            crops.append(
                dict(
                    search=region, crop_start=offset, crop_end=last / info.samplerate, events=mapped
                )
            )
    merged = deepcopy(previous_analysis or {})
    merged["events"] = merge_add_observations(merged.get("events", []), observations)
    assign_event_ids(merged, "audio", duration)
    return merged, crops


def merge_add_observations(previous, observations):
    """Refine explicit identities, never delete original references by crop overlap.

    Only a one-to-one, overlapping, complete observation can replace an envelope.
    Partial/ambiguous review evidence is retained separately, not promoted by name
    guessing. Global occurrence identity belongs to the original observation.
    """
    originals = {e["event_id"]: deepcopy(e) for e in previous if e.get("event_id")}
    observations = deepcopy(observations)
    for event in observations:
        if event.get("refines_event_id") is not None:
            continue

        matches = [
            key
            for key, old in originals.items()
            if all(
                event.get(field) == old.get(field) for field in ("event", "start_time", "end_time")
            )
        ]
        if len(matches) == 1:
            event["refines_event_id"] = matches[0]
            event["identity_compatibility"] = (
                "Exact original observation key repeated; ignore crop-local ordinal and retain original ID"
            )
    links = {}
    for event in observations:
        key = event.get("refines_event_id")
        if isinstance(key, str):
            links.setdefault(key, []).append(event)
    extras = []
    for raw in observations:
        event = deepcopy(raw)
        key = event.get("refines_event_id")
        old = originals.get(key) if isinstance(key, str) else None
        usable = (
            old is not None
            and len(links.get(key, [])) == 1
            and not event.get("invalid_crop_boundary")
        )
        usable = usable and all(
            finite(e.get(edge)) for e in (old, event) for edge in ("start_time", "end_time")
        )
        usable = usable and max(old["start_time"], event["start_time"]) < min(
            old["end_time"], event["end_time"]
        )
        if usable and not event.get("partial_start") and not event.get("partial_end"):
            replacement = dict(
                event,
                event_id=key,
                event=old["event"],
                occurrence_index=old.get("occurrence_index"),
                occurrence_scope="original_recording",
                review_label=event.get("event"),
                identity_evidence="Explicit one-to-one refines_event_id with overlapping complete observation; original ordinal retained",
            )
            replacement["review_history"] = old.get("review_history", []) + [
                dict(
                    original_start=old["start_time"],
                    original_end=old["end_time"],
                    original_occurrence=old.get("occurrence_index"),
                    observation=event,
                )
            ]

            phases = (
                dict(old.get("boundaries") or {}) if isinstance(old.get("boundaries"), dict) else {}
            )
            phases.update(event.get("boundaries") or {})
            replacement["boundaries"] = {
                role: value
                for role, value in phases.items()
                if isinstance(value, dict)
                and finite(value.get("time"))
                and replacement["start_time"] <= value["time"] <= replacement["end_time"]
            }

            def complete_rhythm(item):
                rhythm = item.get("rhythm") or {}
                beats = rhythm.get("onsets_seconds") if isinstance(rhythm, dict) else None
                return (
                    isinstance(beats, list)
                    and len(beats) >= 3
                    and rhythm.get("evidence")
                    and all(finite(t) and t >= 0 for t in beats)
                    and all(a < b for a, b in zip(beats, beats[1:]))
                )

            if complete_rhythm(old) and not complete_rhythm(event):
                replacement["rhythm"] = deepcopy(old["rhythm"])
                replacement.setdefault("review_notes", []).append(
                    "Review supplied no complete replacement beat sequence; retained original measured beat evidence"
                )
            originals[key] = replacement
        else:
            if old is not None:
                old.setdefault("review_notes", []).append(
                    "Review did not provide an unambiguous complete refinement; original evidence/ordinal retained"
                )
            if key is not None:
                event["review_context_only"] = True
                event["identity_warning"] = (
                    "Unverified, partial or ambiguous refinement; not a replacement for the referenced original event"
                )
            extras.append(event)
    unique = dict(originals)

    for event in extras:
        unique.setdefault(event["event_id"], event)
    return sorted(unique.values(), key=lambda e: float(e.get("start_time") or 0))
