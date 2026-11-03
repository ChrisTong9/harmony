"""Semantic edit intent -> deterministic windows -> independently sized placements.

All coordinates are in the ORIGINAL input timeline. No clipping or unlogged
redistribution takes place here. Existing generators consume the compiled durations.
"""

from __future__ import annotations
import math
from dataclasses import asdict
from audio_edit.plans.event_ids import assign_event_ids
from audio_edit.plans.temporal_boundaries import event_boundary, expand_each
from audio_edit.plans.contract_normalization import normalize_edit, REFERENCE_KEYS


CONTRACT_PROMPT = """
Also return contract_version=1 and edits (mandatory for this planner).
First reason about the COMPLETE instruction and observed scene, then choose each
editing need, then its event duration. Never use a single duration for all windows.
edits is an array with this schema:
{
 "window": {"start": ANCHOR, "end": ANCHOR},
 "timeline_op": "overlay|insert|prepend|append",
 "overflow_policy": "error|extend",
 "coverage": "single|repeated|continuous",
 "event_duration": number or null,
 "count": integer,
 "distribution": "explicit|sequential|uniform",
 "offsets_seconds": [numbers relative to window start],
 "placement_anchors": [ANCHOR] or null,
 "audio_prompt": "new sound only, appropriate to this particular edit",
 "layer_role": "foreground|background",
 "relative_level_db": number or null,
 "reason": "semantic/event evidence and placement/coverage choice",
 "default_reason": "explanation of ambiguous timing/operation defaults, or null"
}
ANCHOR is {"seconds": number} for literal seconds, OR
{"event_id": "observed opaque ID or array of candidate IDs", "edge":"start|end",
 "reference_text":"verbatim reference wording from the instruction",
 "reference_kind":"action|audible_event|utterance|end_region|file",
 "occurrence":"first|last|each or one-based integer or null",
 "relation":"before|at|after", "boundary_role":"onset|action_end|sound_end|utterance_end",
 "offset_seconds":number},
OR {"media_edge":"start|end", "offset_seconds":number}.
For a proportional file position use {"media_fraction":number in [0,1]}.
This is file geometry, not an inferred audio event. Keep ALL anchors nested in
window.start/window.end, including literal seconds. Empty optional lists mean absent.
File geometry never needs a detected event_id. Use the supplied semantic fraction
or file edge directly rather than send it through missing-event recovery.
For continuous coverage, placement_anchors=null: its window already defines the
complete interval. For insertion/append, end=null is valid when event duration is
known; Python derives the inserted extent without moving the insertion point.
FIRST parse reference wording, occurrence, relation, required boundary and signed
explicit offset; THEN choose the anchor. Action completion, disappearance of its
audible decay, utterance completion, an end-region and actual file end are distinct.
Choose boundary_role from the COMPLETE instruction, never just a sound category.
action_end is cessation of the principal driving activity, with possible remaining
decay; sound_end includes disappearance of that decay; utterance_end ends the
linguistic content, not subsequent room tone or reverberation. These are phases of
the chosen event, not aliases for its full envelope end.
When timing_semantics is supplied, preserve that interpretation of the instruction
in your anchors. Acoustic analysis locates it; it cannot change the reference to
the hypothetical NEW sound or change the requested boundary, offset or ordinal.
After does NOT automatically mean the coarse event end. Keep missing phase roles
in the plan: Python can request one focused review. Do not bypass missing phase
evidence using seconds or media end. Existing long events or events ending at EOF
are not inherently wrong. Unspecified occurrence is null, never implicitly last.
For a category with several occurrences pass all relevant IDs and the selector;
each on window.start expands into independent edits. Preserve raw reference_text.
For one sound after each reference, use coverage=single, count=1, window.end=null,
offsets_seconds=[0]. The reference list supplies the repetition; do not multiply
count by the number of references or duplicate them in placement_anchors.
Python chooses a boundary and THEN adds offset_seconds (signed timeline offset).
For a single event ending before an anchor use window.start=that anchor,
window.end=null and alignment=before; Python subtracts event_duration only AFTER
resolving the boundary and offset. This subtraction is for overlay only.
Insert/prepend BEFORE an event instead uses its insertion anchor and alignment=start:
the inserted extent shifts the original event later. Do not subtract it again.
Otherwise alignment=start. Do not compute times.
End-region is a vague region choice requiring default_reason, not media_edge=end.
If a reference remains uncertain after review, retain the intended semantic anchor
and candidate IDs, with a reason; do not abandon a feasible edit or invent precision.
Window is where an event may be placed, NOT necessarily its duration.
For a single/repeated sound, event_duration is ONE complete natural event, count
is exact, offsets_seconds specifies each start within the window. Choose natural
positions using the FULL scene/instruction; don't hardcode a sound category to one
instance, the full clip, or uniformly spaced points. Uniform is allowed ONLY when
the instruction actually requests it. Explicit offsets can be approximate scene-
based choices, explained in reason/default_reason. Use separate edits for unequal
windows or different durations. For continuous, count=1 and event_duration=null:
Python calculates exact coverage duration from the window. Never shorten coverage.
For event-related individual placements, supply placement_anchors with event IDs
and edges, not offsets_seconds calculated by you from observed timestamps. Python
computes the offsets. offsets_seconds is for explicit numeric offsets or genuinely
vague position choices, which must be explained as approximate in default_reason.
For uniform placement also return distribution_explicitly_requested=true.
Do not describe continuous ambience in the prompt for a requested isolated transient.
For repeated coverage, audio_prompt describes ONE complete event only. Count,
sequence length and inter-event spacing belong exclusively in the structured fields,
not in the generation prompt; otherwise each occurrence would generate a sequence.
Insert/prepend uses window.start as the insertion point, and shifts subsequent
original samples; window.end may equal window.start. Append starts at media end.
Repeated insert/prepend/append retains nonnegative, ordered offsets_seconds,
including silent gaps. Those offsets describe the NEW inserted sequence; every
occurrence shares the original insertion coordinate. Python inserts the gaps too.
Overlay leaves original time unchanged; overflow_policy=extend explicitly permits
extending past EOF, otherwise out-of-bounds is a conflict.
Plan feasibility before returning: preserve explicit durations and ranges, but
choose unspecified natural event durations that fit their observed windows.
If a required before/between gap cannot contain the complete sound, use an explicit
insert/prepend with a reason only when the instruction allows shifting the original
timeline. Do not keep an impossible overlay. Explicitly fixed timing, duration,
overlay-only or unchanged-length requirements must not be overridden. Extend does
not create space between adjacent original events; insertion does. For audio-only
requests for a prelude or a sound between events with no fixed-timeline requirement,
insertion is a valid ordering-preserving plan. Report this choice in default_reason.
For an open-ended single sound after an event, use window.end=null together with
overlay: Python derives the extent from the anchor and event_duration, then checks
overflow_policy. Material length must never change the selected anchor to EOF.
Never use a zero-width overlay window such as [media end, media end]. A zero-width
window is an insertion POINT only for insert/prepend/append, not overlay coverage.
'Beginning/end' alone does not decide operation. Interpret the entire instruction;
if truly ambiguous, choose overlay and state the default_reason. Explicit insert,
prepend or append MUST use that timeline operation. After speech must stay AFTER.
For no numeric timing, choose relevant observed event boundaries or explain a vague
placement default. Never invent a measured named anchor. If evidence is insufficient,
retain event_id=null or candidate IDs and the original reference wording so the
bounded review can run. After timing_review_complete=true with best_effort=true,
return a feasible estimated edit with default_reason explaining the remaining
uncertainty and why its placement and timeline operation best preserve the request.
Do not return an empty plan merely because a reference, exact boundary or rhythmic
detail is uncertain. Preserve count, coverage and temporal order. An estimated
prelude can explicitly prepend at the recording start; do not emit negative time.
Use this only when the complete instruction calls for before-ordering and allows
timeline change, never as a generic missing-event default. Genuinely incompatible
explicit requirements remain a constraint conflict. A central-third search range
is not a resolved event. Uncertain event-derived anchors must retain their event
reference rather than disguise a guessed timestamp as literal seconds.
An event marked partial_start or partial_end reaches an audio crop edge: that edge
is NOT a measured boundary and cannot define placement or coverage. Its other,
complete edge can still be referenced. Use complete whole-recording references
when a partial review cannot supply the required boundary. Do not use invalid or
explicitly superseded (review_context_only) observations.
For vague additions, missing EXISTING instances of the NEW sound does not justify
full-track coverage. Use the whole scene and instruction to choose reaction points,
repetition or ambience, and explain why that coverage fits the requested sound.
In particular, absence of detected new-sound instances is not evidence that a continuous
bed is appropriate. Conversely, do not force an explicitly continuous request into
isolated bursts. Do not generate speech or other original-scene content in audio_prompt.
The legacy temporal/duration fields remain for API compatibility only; edits are
authoritative. When shared_replacement_windows is supplied, use exactly those
windows in the same order, overlay/error; per-window coverage can differ but no
replacement sound may leak into another interval. For a global replacement choose
observed OLD event windows when available, otherwise explicitly describe its scope.
Do not request automatic original-track ducking. relative_level_db expresses the
NEW layer's intended level relative to local scene. For ADD, null defaults to
+6 dB for every layer role, including background music, accompaniment and harmony.
Background describes purpose, not low volume. Return null unless the user
explicitly requests a loudness change; preserve numeric requests, and use -12 dB
for explicitly quiet/low-volume additions without a number. Calm/gentle style
alone does not mean low volume. Shared replacements keep their existing balance.
This is the ONE semantic loudness control for each placement; do not express it
again in composer.mix_level.
Return composer.mix_level=1.0 for compatibility; it is not an additional attenuation
or boost on structured placements. Python measures the prepared sound and applies
this scene target once. Preserve original gain and the raw final sum without
master normalization.
"""


def number(value, name):
    if isinstance(value, bool):
        raise ValueError(name + " must be finite numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(name + " must be finite numeric") from exc
    if not math.isfinite(result):
        raise ValueError(name + " must be finite numeric")
    return result


def resolve_anchor(anchor, events, duration, *, best_effort=False):
    if not isinstance(anchor, dict):
        raise ValueError("Missing explicit time anchor")
    kinds = [k for k in REFERENCE_KEYS if k in anchor]
    if len(kinds) != 1:
        raise ValueError(
            "Anchor must have exactly one reference type: seconds, event_id, media_edge, or media_fraction. File-relative geometry needs media_fraction; reference_text alone is not a coordinate."
        )
    status, evidence = "resolved", dict(anchor)
    if "seconds" in anchor:
        value = number(anchor["seconds"], "seconds")
    elif "media_fraction" in anchor:
        fraction = number(anchor["media_fraction"], "media_fraction")
        if not 0 <= fraction <= 1:
            raise ValueError("media_fraction must be in [0,1]")
        value = duration * fraction
        evidence["basis"] = "Requested file fraction times measured duration; not an audio event"
    elif "event_id" in anchor:
        value, status, evidence = event_boundary(anchor, events, best_effort=best_effort)
    else:
        if anchor["media_edge"] not in ("start", "end"):
            raise ValueError("Only actual media edges are anchors")
        value = 0.0 if anchor["media_edge"] == "start" else duration
    offset = number(anchor.get("offset_seconds", 0), "offset_seconds")
    evidence.update(selected_boundary=value, applied_offset=offset)
    value += offset
    evidence.update(final_anchor=value, status=status)
    return value, status, evidence


def compile_edits(
    edits,
    *,
    duration,
    audio_analysis=None,
    visual_analysis=None,
    shared_windows=None,
    best_effort=False,
    default_duration=None,
):
    assign_event_ids(audio_analysis, "audio", duration)
    assign_event_ids(visual_analysis, "visual", duration)
    events = {
        e["event_id"]: e
        for a, k in ((audio_analysis, "events"), (visual_analysis, "visual_events"))
        for e in (a or {}).get(k, [])
    }
    if not isinstance(edits, list) or not edits:
        raise ValueError("Unresolved edit: planner returned no grounded edit needs")
    normalized = []
    for edit in edits:
        canonical, audit = normalize_edit(edit)
        canonical["_normalizations"] = audit
        normalized.append(canonical)
    edits = expand_each(normalized, events)
    if shared_windows is not None and len(edits) != len(shared_windows):
        raise ValueError("Replacement edit count differs from authoritative windows")
    placements, windows = [], []
    for i, edit in enumerate(edits):
        start, ss, se = resolve_anchor(
            edit["window"]["start"], events, duration, best_effort=best_effort
        )
        op = edit.get("timeline_op")
        coverage = edit.get("coverage")
        overflow = edit.get("overflow_policy", "error")
        count = edit.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1 or count > 100:
            raise ValueError("count must be an integer in [1,100]")
        if edit["window"].get("end") is None:
            if coverage == "continuous" and op == "overlay" and overflow != "extend":
                end = duration
                es, ee = (
                    "approximate",
                    {
                        "derived": "Open continuous coverage defaults to remaining recording; original start unchanged"
                    },
                )
            else:
                d = number(
                    edit.get("event_duration")
                    if edit.get("event_duration") is not None
                    else default_duration,
                    "event_duration (required to derive open-ended extent)",
                )
                offsets = edit.get("offsets_seconds")
                if edit.get("distribution") == "sequential":
                    offsets = [j * d for j in range(count)]
                if coverage == "continuous":
                    offsets = [0.0]
                if not isinstance(offsets, list) or len(offsets) != count:
                    raise ValueError(
                        "Open repeated window needs one explicit offset per occurrence or sequential distribution"
                    )
                extent = max(number(v, "placement offset") for v in offsets) + d
                if edit.get("alignment", "start") == "before":
                    start -= extent
                    se["extent_alignment"] = (
                        "Entire event sequence ends at resolved anchor; never move to EOF"
                    )
                elif edit.get("alignment", "start") != "start":
                    raise ValueError("Unknown anchor alignment")
                end = start + extent
                es, ee = (
                    ss,
                    {
                        "derived": "Open extent = anchored sequence offsets plus planned event duration",
                        "duration_source": "event_duration"
                        if edit.get("event_duration") is not None
                        else "planner_default_duration",
                    },
                )
                if coverage == "continuous":
                    es = "approximate"
        else:
            end, es, ee = resolve_anchor(
                edit["window"]["end"], events, duration, best_effort=best_effort
            )
        if op not in ("overlay", "insert", "prepend", "append") or coverage not in (
            "single",
            "repeated",
            "continuous",
        ):
            raise ValueError("Explicit timeline_op and coverage are required")
        if overflow not in ("error", "extend"):
            raise ValueError("Unknown overflow policy")
        if not edit.get("reason"):
            raise ValueError("Edit needs semantic evidence/rationale")
        if start < 0 or end < start or (op in ("insert", "prepend") and start > duration):
            raise ValueError(
                f"Invalid window {start:g}..{end:g} for media duration {duration:g}; no clipping allowed. A before-sequence must fit before its anchor. If it precedes recording start, explicitly replan prepend/insert only if the instruction permits a timeline change."
            )
        if op == "prepend" and abs(start) > 1e-8:
            raise ValueError("Prepend must start at zero")
        if op == "append" and abs(start - duration) > 1e-8:
            raise ValueError("Append must reference original media end")
        if op == "overlay" and end > duration + 1e-8 and overflow != "extend":
            raise ValueError(
                f"Overlay window exceeds media without explicit extend policy: anchor={start:g}, end={end:g}, media_end={duration:g}, available={max(0, duration - start):g} seconds. Keep the anchor. If event duration was a planner choice, plan a complete shorter event that fits; otherwise use extend only when the instruction permits it. Never change explicit user duration or coverage."
            )
        if shared_windows is not None:
            sw = shared_windows[i]
            sw = asdict(sw) if not isinstance(sw, dict) else sw
            if (
                op != "overlay"
                or overflow != "error"
                or abs(start - sw["start_time"]) > 1e-6
                or abs(end - sw["end_time"]) > 1e-6
            ):
                raise ValueError("Replacement changed authoritative removal window")
        if coverage in ("single", "continuous") and count != 1:
            raise ValueError("single/continuous requires count=1")
        event_duration = edit.get("event_duration")
        if coverage != "continuous" and event_duration is None and default_duration is not None:
            event_duration = default_duration
            edit["_normalizations"].append(
                dict(
                    code="legacy_event_duration",
                    before=None,
                    after=default_duration,
                    reason="Use the same planner-provided single-event duration; never derive it from a shorter window",
                )
            )
        d = end - start if coverage == "continuous" else number(event_duration, "event_duration")
        if d <= 0:
            raise ValueError("Event/coverage duration must be positive")
        distribution = edit.get("distribution", "explicit")
        offsets = edit.get("offsets_seconds")
        placement_evidence = []
        placement_statuses = []
        if edit.get("placement_anchors") is not None:
            if distribution != "explicit" or coverage == "continuous":
                raise ValueError(
                    "Placement anchors are for explicitly selected discrete occurrences"
                )
            anchors = [
                resolve_anchor(a, events, duration, best_effort=best_effort)
                for a in edit["placement_anchors"]
            ]
            if edit.get("alignment") == "before":
                anchors = [
                    (value - d, status, dict(evidence, extent_alignment="end at anchor"))
                    for value, status, evidence in anchors
                ]
            offsets = [a[0] - start for a in anchors]
            placement_statuses = [a[1] for a in anchors]
            placement_evidence = [a[2] for a in anchors]
        if coverage == "continuous":
            if offsets not in (None, [], [0], [0.0]):
                raise ValueError("Continuous coverage starts at its window, not an offset")
            offsets = [0.0]
        elif distribution == "sequential":
            offsets = [j * d for j in range(count)]
        elif distribution == "uniform":
            if not edit.get("distribution_explicitly_requested", False):
                raise ValueError("Uniform placement requires explicit user intent")
            offsets = (
                [0.0] if count == 1 else [j * (end - start - d) / (count - 1) for j in range(count)]
            )
        elif distribution != "explicit":
            raise ValueError("Unknown distribution")
        if not isinstance(offsets, list) or len(offsets) != count:
            raise ValueError("Each occurrence requires its own placement offset")
        status = (
            "approximate"
            if "approximate" in (ss, es, *placement_statuses) or edit.get("default_reason")
            else "resolved"
        )
        if "estimated" in (ss, es, *placement_statuses) or (
            best_effort and status == "approximate"
        ):
            status = "estimated"
        window = dict(
            start_time=start,
            end_time=end,
            coverage=coverage,
            event_duration=d,
            count=count,
            distribution=distribution,
            timeline_op=op,
            overflow_policy=overflow,
            status=status,
            evidence=[se, ee, *placement_evidence],
            reason=edit["reason"],
            default_reason=edit.get("default_reason"),
        )
        if edit["_normalizations"]:
            window["normalizations"] = edit["_normalizations"]
        windows.append(window)
        previous_insert_end = 0.0
        for j, offset in enumerate(offsets):
            offset = number(offset, "placement offset")
            a, b = start + offset, start + offset + d
            if offset < 0 or (op == "overlay" and b > end + 1e-8):
                raise ValueError(
                    f"Constraint conflict in window {i}: complete event {a:g}-{b:g} does not fit {start:g}-{end:g}. Available range is {end - start:g} seconds. Keep the reference/window and all explicit durations. If a requested between/before gap is empty, explicitly replan insert/prepend if timeline change is allowed; extend only permits media EOF overflow, not overlap with protected reference events."
                )

            anchor = start if op in ("insert", "prepend", "append") else a
            inserted_gap = None
            if op in ("insert", "prepend", "append"):
                inserted_gap = offset - previous_insert_end
                if inserted_gap < -1e-8:
                    raise ValueError(
                        "Inserted occurrence offsets overlap or are out of order; explicitly plan a combined layer instead of silently moving them"
                    )
                previous_insert_end = offset + d
            placements.append(
                dict(
                    start_time=anchor,
                    end_time=anchor + d,
                    event_duration=d,
                    window_index=i,
                    occurrence_index=j + 1,
                    window_start=start,
                    window_end=end,
                    timeline_op=op,
                    overflow_policy=overflow,
                    coverage=coverage,
                    status=status,
                    audio_prompt=edit.get("audio_prompt"),
                    layer_role=edit.get("layer_role"),
                    relative_level_db=edit.get("relative_level_db"),
                    evidence=window["evidence"],
                    reason=edit["reason"],
                    allowed_start=a,
                    allowed_end=b,
                )
            )
            if inserted_gap is not None:
                placements[-1]["insert_gap_before_seconds"] = max(0.0, inserted_gap)
    return windows, placements
