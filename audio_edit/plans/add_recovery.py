"""Explicit user policy: estimated ADD overlays instead of timing-only no-ops.

This is an alternative execution policy, not a claim that missing evidence was
resolved. Strict successful plans and replacement windows never enter this path.
"""

from copy import deepcopy
import math
import os


POLICY = "overlap_preserve"
ENVIRONMENT = "AUDIO_ADD_TIMING_RECOVERY"
RECOVERY_PROMPT = """You plan one bounded best-effort ADD after timing resolution failed.
The user explicitly prefers adding the requested sound, allowing overlap with
existing sounds, and keeping the original recording duration unchanged. This
policy permits estimated placement; it does not make uncertain observations true.
Read the ORIGINAL request to preserve WHAT to add, musical conditioning, event
count and explicit loudness. Use the supplied observations as uncertain context.
Do not remove, replace, duck or retime original audio. Do not return an empty edit.

Return compact JSON: {"intent_version":2,"is_on_screen":false,"edits":[EDIT]}.
Each EDIT has generator="audio"|"speech", audio_reference=null|"context"|"structure",
audio_prompt describing only the new sound, layer_role="foreground"|"background",
relative_level_db=null unless the user specified loudness, timeline_op="overlay",
overflow_policy="error", reason, placement, and provenance sources={
"placement":"planner_choice","timeline_op":"system_default",
"overflow_policy":"system_default"}. Placement and generation belong directly
on EDIT, never inside sources. Sources contains provenance strings only.
Use context for musical accompaniment responding to original audio; otherwise
null for independent effects/ambience. Speech needs exact speech_text.

Use only numerical estimated positions {"seconds":number} or file geometry
{ "media_edge":"start|end" } / {"media_fraction":number}. No event references,
question_id, each selectors, interval_edge, shared_window or listening windows.
Prefer a plausible position from the original instruction and existing evidence.
If position is unlocatable, choose a reasonable default and say it is a default.
Do not turn an uncertain long decay into a tiny, ineffective addition at EOF.
Decide the requested sound's temporal extent BEFORE considering spare space.
An ongoing texture, bed or accompaniment is continuous even when the user says
"add the sound of ... starting after ..." without an explicit ending. Starting
once does not make an ongoing sound a discrete event. A specifically brief burst,
one-shot effect or repeated event remains discrete; preserve that duration/count.
For continuous additions return placement={"kind":"cover","range":{
"start":ANCHOR,"end":ANCHOR}} and continuous_coverage={"minimum_seconds":NUMBER,
"reason":"why this duration preserves the requested ongoing sound"}.
Choose that positive minimum from the requested sound's sustained body, capped
by recording length, independently of the remaining gap. Never derive this
minimum from a tiny EOF remainder. The executor can move the estimated range
earlier to fit this duration while overlapping the original. Do not pretend that
the adjusted position is an observed event boundary. If the request specifies a
duration or fixed window, preserve it; do not widen it with a default minimum.
An unspecified ending extends to the recording end. If the requested start is
uncertain, choose a plausible start and meaningful coverage with overlap allowed.
Untimed continuous additions still cover the whole recording. Omit generation
for cover: its length comes from the complete fitted range, not a short asset.
One discrete sound uses placement={kind:"single",anchor:ANCHOR,alignment:"start|end|center"}
and generation={target_seconds:positive_number,source:"planner_choice",adjustable:true,
min_seconds:positive_number}. Allocate enough time for the whole natural sound,
within the recording duration. Do not generate an arbitrarily short click when
the requested sound has a body/tail. Keep repetition out of audio_prompt.
Several sounds use kind="repeat", anchors:[ANCHOR,...], alignment="start" and
generation for ONE sound. Preserve requested count; for 'each', use the observed
count when available; if unknown, explicitly note that the count is estimated.
Fit complete sounds inside the recording. Overlap is allowed when space is tight.
Never extend duration. Keep all normal non-timing choices if previous edits exist.
Return JSON only. These are estimated execution choices, not measured timestamps."""


def enabled(ctx):
    from audio_edit.utils.ablation_policy import execution_corrections_enabled

    if not execution_corrections_enabled("add_plan_or_asset_recovery"):
        return False
    return (
        os.environ.get(ENVIRONMENT) == POLICY
        and not ctx.params.get("replacement_reference_asset")
        and not ctx.params.get("replacement_reference_path")
    )


def unresolved_analysis(ctx):
    dialogue = ctx.params.get("edit_listening_dialogue", {})
    last = next(
        (
            t.get("analysis")
            for t in reversed(dialogue.get("turns", []))
            if isinstance(t.get("analysis"), dict)
        ),
        {},
    )
    result = deepcopy(last)
    result["edit_listening"] = {
        k: deepcopy(dialogue.get(k))
        for k in ("requirements", "placement_windows", "timeline_policy", "selection")
    }
    result["edit_listening"].update(
        status="unresolved",
        missing_evidence=deepcopy(
            next(
                (
                    t.get("missing_evidence")
                    for t in reversed(dialogue.get("turns", []))
                    if t.get("missing_evidence")
                ),
                {},
            )
        ),
    )
    return result


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Recovery placement needs finite numerical time")
    return float(value)


def _coordinate(anchor, duration):
    if not isinstance(anchor, dict) or len(anchor) != 1:
        raise ValueError("Recovery must use numerical/file positions, not event evidence")
    if "seconds" in anchor:
        return _number(anchor["seconds"])
    if "media_fraction" in anchor:
        return _number(anchor["media_fraction"]) * duration
    if anchor.get("media_edge") in ("start", "end"):
        return 0.0 if anchor["media_edge"] == "start" else duration
    raise ValueError("Recovery cannot invent a measured event reference")


def _literal_repeat(placement, generation, duration):
    """Expand a literal count/spacing sequence using whole-sequence alignment."""
    if placement.get("kind") != "repeat" or "repeat" not in placement:
        return placement
    if set(placement) - {"kind", "anchor", "alignment", "repeat"}:
        raise ValueError("Recovery repeat must have one unambiguous literal origin")
    rule = placement["repeat"]
    if not isinstance(rule, dict) or set(rule) != {"count", "spacing_seconds"}:
        raise ValueError("Recovery repeat needs a count and numerical onset spacing")
    count = rule["count"]
    if type(count) is not int or not 1 <= count <= 100:
        raise ValueError("Recovery event count must be in [1,100]")
    spacing = _number(rule["spacing_seconds"])
    if spacing <= 0:
        raise ValueError("Recovery repetition spacing must be positive")
    duration = _number(duration)
    raw_length = (generation or {}).get("target_seconds")
    length = min(duration, _number(raw_length)) if raw_length is not None else min(1.0, duration)
    if length <= 0:
        raise ValueError("Recovery generation must have positive duration")
    extent = (count - 1) * spacing + length
    if extent > duration:
        raise ValueError("Recovery repeat cannot fit without changing its count or spacing")
    alignment = placement.get("alignment", "start")
    if alignment not in ("start", "end", "center"):
        raise ValueError("Recovery repeat needs a valid extent alignment")
    origin = _coordinate(placement.get("anchor"), duration)
    start = origin - {"start": 0.0, "end": extent, "center": extent / 2}[alignment]

    start = max(0.0, min(duration - extent, start))
    return dict(
        kind="repeat",
        alignment="start",
        anchors=[dict(seconds=start + i * spacing) for i in range(count)],
    )


def _lift_cover_metadata(edit):
    """Move known misplaced cover metadata without resolving ambiguous values."""
    from audio_edit.plans.add_intent import SOURCES

    placement = edit.get("placement")
    if not isinstance(placement, dict) or placement.get("kind") != "cover":
        return
    if "continuous_coverage" in placement:
        coverage = placement["continuous_coverage"]
        if (
            not isinstance(coverage, dict)
            or set(coverage) != {"minimum_seconds", "reason"}
            or _number(coverage["minimum_seconds"]) <= 0
            or not isinstance(coverage["reason"], str)
            or not coverage["reason"].strip()
        ):
            raise ValueError(
                "Nested continuous recovery needs a positive justified coverage budget"
            )
        if "continuous_coverage" in edit and edit["continuous_coverage"] != coverage:
            raise ValueError("Conflicting recovery continuous_coverage representations")
        edit["continuous_coverage"] = deepcopy(coverage)
        del placement["continuous_coverage"]
    if "provenance_sources" in placement:
        sources = placement["provenance_sources"]
        if (
            not isinstance(sources, dict)
            or not sources
            or set(sources) - {"placement", "timeline_op", "overflow_policy"}
            or any(not isinstance(v, str) or v not in SOURCES for v in sources.values())
        ):
            raise ValueError("Nested recovery provenance must contain parameter source labels only")
        if "sources" in edit and edit["sources"] != sources:
            raise ValueError("Conflicting recovery provenance representations")
        edit["sources"] = deepcopy(sources)
        del placement["provenance_sources"]


def fit_data(data, duration):
    """Fit literal fallback geometry before generation; never crop a real asset."""
    from audio_edit.plans.add_intent import validate_intent

    duration = _number(duration)
    if duration <= 0:
        raise ValueError("Recovery requires a nonempty recording")
    data = deepcopy(data)
    edits = data.get("edits")
    if not isinstance(edits, list) or not edits or len(edits) > 100:
        raise ValueError("Recovery requires a bounded nonempty edit list")
    adjustments = []
    for index, edit in enumerate(edits):
        before = deepcopy(edit)
        explicit_window = (
            isinstance(edit.get("sources"), dict)
            and edit["sources"].get("placement") == "user_explicit"
        )

        nested = edit.get("sources")
        if isinstance(nested, list):
            if isinstance(edit.get("placement"), dict) and all(
                isinstance(item, str) for item in nested
            ):
                nested = None
            elif len(nested) == 1 and isinstance(nested[0], dict):
                nested = nested[0]
            else:
                raise ValueError("Ambiguous recovery source wrappers")
        if isinstance(nested, dict) and isinstance(nested.get("placement"), dict):
            for key in ("placement", "generation"):
                if key in nested:
                    if key in edit and edit[key] != nested[key]:
                        raise ValueError("Conflicting recovery " + key + " representations")
                    edit[key] = deepcopy(nested[key])
        _lift_cover_metadata(edit)
        explicit_window = (
            isinstance(edit.get("sources"), dict)
            and edit["sources"].get("placement") == "user_explicit"
        )
        if any(
            k in edit for k in ("shared_window", "shared_window_index", "listening_window_index")
        ):
            raise ValueError("ADD recovery cannot modify shared replacement/listening windows")
        edit.update(timeline_op="overlay", overflow_policy="error")
        edit["sources"] = dict(
            placement="planner_choice",
            timeline_op="system_default",
            overflow_policy="system_default",
        )
        p = _literal_repeat(edit["placement"], edit.get("generation"), duration)
        edit["placement"] = p
        coverage = edit.pop("continuous_coverage", None)
        minimum = None
        if coverage is not None:
            if not isinstance(coverage, dict):
                raise ValueError("Continuous recovery needs a coverage budget object")
            minimum = min(duration, _number(coverage.get("minimum_seconds")))
            if (
                minimum <= 0
                or not isinstance(coverage.get("reason"), str)
                or not coverage["reason"].strip()
            ):
                raise ValueError("Continuous recovery needs a positive justified coverage budget")
            if edit["generator"] != "audio" or p["kind"] not in ("cover", "single"):
                raise ValueError("Continuous coverage cannot replace speech or repeated events")
            if p["kind"] == "single":
                allowed = {"kind", "anchor", "alignment"}
                if set(p) - allowed or explicit_window:
                    raise ValueError(
                        "Continuous recovery cannot replace an explicit discrete constraint"
                    )
                offset = {"start": 0.0, "end": minimum, "center": minimum / 2}[
                    p.get("alignment", "start")
                ]
                a = _coordinate(p["anchor"], duration) - offset
                p = dict(
                    kind="cover", range=dict(start=dict(seconds=a), end=dict(seconds=a + minimum))
                )
        if p["kind"] == "cover":
            if set(p) != {"kind", "range"} or set(p["range"]) != {"start", "end"}:
                raise ValueError("Recovery cover needs exactly one literal range")
            a = max(0.0, min(duration, _coordinate(p["range"]["start"], duration)))
            b = max(0.0, min(duration, _coordinate(p["range"]["end"], duration)))
            if b <= a and minimum is not None and not explicit_window:
                b = a
            elif b <= a:
                a, b = 0.0, duration
            if minimum is not None and not explicit_window:
                length = min(duration, max(b - a, minimum))
                a = max(0.0, min(a, duration - length))
                b = a + length
            edit["placement"] = dict(
                kind="cover", range=dict(start=dict(seconds=a), end=dict(seconds=b))
            )
            edit.pop("generation", None)
        elif p["kind"] in ("single", "repeat"):
            multiple = p["kind"] == "repeat"
            allowed = {"kind", "anchors" if multiple else "anchor", "alignment"}
            if set(p) - allowed:
                raise ValueError(
                    "Recovery uses explicit discrete positions, not another timing contract"
                )
            anchors = p["anchors"] if multiple else [p["anchor"]]
            if not isinstance(anchors, list) or not 1 <= len(anchors) <= 100:
                raise ValueError("Recovery event count must be in [1,100]")
            g = edit.get("generation") or {}
            raw = g.get("target_seconds")
            length = min(duration, _number(raw)) if raw is not None else min(1.0, duration)
            if length <= 0:
                raise ValueError("Recovery generation must have positive duration")

            edit["generation"] = dict(
                target_seconds=None if edit["generator"] == "speech" else length,
                source="planner_choice",
                adjustable=True,
                min_seconds=min(length, 0.1),
            )
            alignment = p.get("alignment", "start")
            offset = {"start": 0.0, "end": length, "center": length / 2}[alignment]
            starts = [
                max(0.0, min(duration - length, _coordinate(a, duration) - offset)) for a in anchors
            ]
            placement = dict(kind=p["kind"], alignment="start")
            placement["anchors" if multiple else "anchor"] = (
                [dict(seconds=s) for s in starts] if multiple else dict(seconds=starts[0])
            )
            edit["placement"] = placement
        else:
            raise ValueError("Unsupported recovery placement")
        if edit != before:
            adjustments.append(dict(edit_index=index, before=before, after=deepcopy(edit)))
    validate_intent(dict(version=2, edits=edits))
    return data, adjustments


def mark_plan(plan, duration, details, asset_durations=None):
    from audio_edit.plans.add_timing import compile_plan

    plan.intent["timing_recovery"] = dict(
        policy=POLICY, authority="user_execution_policy", approximate=True, **deepcopy(details)
    )
    result = compile_plan(plan, duration=duration, asset_durations=asset_durations)
    if result.temporal_status == "unresolved":
        raise ValueError("Recovery plan did not fit: " + str(result.temporal_reason))
    return result


def _expand_grouped_singles(data, previous, duration, adaptations):
    """Keep homogeneous ordered one-shot edits when recovery groups their anchors.

    This is a representation change only: no missing event, distinct sound, or
    unselected 'each' expansion can acquire a guessed counterpart.
    """
    proposed = data.get("edits", [])
    if len(proposed) == len(previous):
        return data
    if not (2 <= len(previous) <= 100 and len(proposed) == 1):
        return data
    new = proposed[0]
    p = new.get("placement", {})
    anchors = p.get("anchors")
    if (
        p.get("kind") != "repeat"
        or set(p) - {"kind", "anchors", "alignment"}
        or p.get("alignment", "start") != "start"
        or not isinstance(anchors, list)
        or len(anchors) != len(previous)
    ):
        return data
    first = previous[0]
    fields = (
        "audio_prompt",
        "generator",
        "audio_reference",
        "speech_text",
        "voice_prompt",
        "layer_role",
        "relative_level_db",
        "generation",
    )
    if first.get("generator") != "audio" or first.get("audio_reference") is not None:
        return data
    original_anchors = []
    for edit in previous:
        old = edit.get("placement", {})
        if (
            any(edit.get(k) != first.get(k) for k in fields)
            or any(
                k in edit
                for k in ("shared_window", "shared_window_index", "listening_window_index")
            )
            or old.get("kind") != "single"
            or set(old) - {"kind", "anchor", "alignment"}
            or old.get("alignment", "start") != "start"
            or not isinstance(old.get("anchor"), dict)
        ):
            return data
        original_anchors.append(old["anchor"])

    if all(set(a) == {"seconds"} for a in original_anchors):
        order = [_number(a["seconds"]) for a in original_anchors]
    else:
        template = {k: v for k, v in original_anchors[0].items() if k != "occurrence"}
        if not template or any(
            {k: v for k, v in a.items() if k != "occurrence"} != template for a in original_anchors
        ):
            return data
        order = [a.get("occurrence") for a in original_anchors]
        if any(type(i) is not int or i <= 0 for i in order):
            return data
    if any(b <= a for a, b in zip(order, order[1:])):
        return data
    starts = [_coordinate(a, duration) for a in anchors]
    length = (first.get("generation") or {}).get("target_seconds")
    if type(length) not in (int, float) or not math.isfinite(length) or length <= 0:
        return data
    if any(b <= a for a, b in zip(starts, starts[1:])) or any(
        a < 0 or a + length > duration + 1e-8 for a in starts
    ):
        return data
    expanded = []
    for anchor in anchors:
        edit = deepcopy(new)
        edit["placement"] = dict(kind="single", anchor=deepcopy(anchor), alignment="start")
        expanded.append(edit)
    result = deepcopy(data)
    result["edits"] = expanded
    adaptations.append(
        dict(
            code="expanded_grouped_single_edits",
            original_edit_count=len(previous),
            response_edit_count=1,
            retained_event_count=len(expanded),
            reason="Ordered explicit recovery anchors map one-to-one to identical prior one-shot edits; retain each original content and duration",
        )
    )
    return result


def recover_plan(ctx, llm, *, duration, analysis, visual_analysis, reason, prior_plan=None):
    """One extra planner call at most; no new listening or rendered candidates."""
    import json
    from audio_edit.specialists.add_sound_specialist import AddSoundSpecialist
    from audio_edit.utils.bounded_tools import bounded_call

    if not enabled(ctx):
        raise ValueError("ADD timing recovery is not enabled")
    if ctx.params.get("add_timing_recovery"):
        raise ValueError("ADD timing recovery already attempted")
    audit = dict(
        policy=POLICY,
        authority="user_execution_policy",
        status="started",
        original_failure=str(reason),
        planner_calls=1,
    )
    ctx.params["add_timing_recovery"] = audit
    prior = deepcopy(prior_plan.intent if prior_plan else None)
    payload = dict(
        user_prompt=ctx.user_prompt,
        media_duration=duration,
        execution_policy=dict(mode=POLICY, allow_overlap=True, preserve_duration=True),
        failure=str(reason),
        uncertain_analysis=analysis,
        previous_intent=prior,
    )
    audit["request"] = deepcopy(payload)

    def call():
        return llm.client.chat.completions.create(
            model=llm.model,
            messages=[
                dict(role="system", content=RECOVERY_PROMPT),
                dict(role="user", content=json.dumps(payload, ensure_ascii=False)),
            ],
        )

    response = bounded_call(ctx, "add_timing_recovery_plan", call, attempts=1)
    raw = response.choices[0].message.content
    audit["raw_response"] = raw
    specialist = AddSoundSpecialist(llm)
    data = specialist._parse_json(raw)
    previous_edits = (prior or {}).get("edits", [])
    placement_adaptations = []

    for index, edit in enumerate(data.get("edits", [])):
        before = deepcopy(edit)
        _lift_cover_metadata(edit)
        if edit != before:
            placement_adaptations.append(
                dict(
                    code="lifted_cover_metadata",
                    edit_index=index,
                    before=before,
                    after=deepcopy(edit),
                )
            )
    data = _expand_grouped_singles(data, previous_edits, duration, placement_adaptations)
    if previous_edits:
        if len(data.get("edits", [])) != len(previous_edits):
            raise ValueError("Timing recovery cannot discard an existing requested edit")
        for edit_index, (old, new) in enumerate(zip(previous_edits, data["edits"])):
            for key in (
                "audio_prompt",
                "generator",
                "audio_reference",
                "speech_text",
                "voice_prompt",
                "layer_role",
                "relative_level_db",
                "generation",
            ):
                if key in old:
                    new[key] = deepcopy(old[key])
                else:
                    new.pop(key, None)
            old_place, new_place = old["placement"], new["placement"]
            if (
                old_place["kind"] == "single"
                and new_place["kind"] == "cover"
                and old_place.get("anchor", {}).get("occurrence") != "each"
                and type((old.get("generation") or {}).get("target_seconds")) in (int, float)
                and old["generation"]["target_seconds"] > 0
            ):
                proposed = deepcopy(new_place)
                alignment = old_place.get("alignment", "start")
                bounds = new_place["range"]
                start = _coordinate(bounds["start"], duration)
                end = _coordinate(bounds["end"], duration)
                if end < start:
                    raise ValueError("Recovery range is reversed")
                anchor = {"start": start, "end": end, "center": (start + end) / 2}[alignment]
                new_place = dict(kind="single", anchor=dict(seconds=anchor), alignment=alignment)
                new["placement"] = new_place
                placement_adaptations.append(
                    dict(
                        edit_index=edit_index,
                        proposed=proposed,
                        retained_kind="single",
                        retained_generation=deepcopy(old["generation"]),
                        reason="Use estimated range position without shrinking the original complete asset",
                    )
                )
            if (old_place["kind"] == "cover") != (new_place["kind"] == "cover"):
                raise ValueError("Timing recovery cannot change continuous/discrete coverage")
            if old_place["kind"] != "cover":
                new.pop("continuous_coverage", None)
            elif old.get("sources", {}).get("placement") == "user_explicit":
                new.pop("continuous_coverage", None)
            if old_place["kind"] == "repeat":
                normalized = _literal_repeat(new_place, new.get("generation"), duration)
                if normalized != new_place:
                    placement_adaptations.append(
                        dict(
                            edit_index=edit_index,
                            proposed=deepcopy(new_place),
                            normalized=deepcopy(normalized),
                            reason="Expand literal repeat count and spacing with whole-sequence alignment",
                        )
                    )
                    new["placement"] = new_place = normalized
                count = (
                    len(old_place["anchors"])
                    if "anchors" in old_place
                    else old_place.get("repeat", {}).get("count")
                )
                if count is not None and len(new_place.get("anchors", [])) != count:
                    raise ValueError("Timing recovery cannot change the requested repetition count")
    data, adjustments = fit_data(data, duration)

    plan = specialist._to_plan(
        data, "video" if ctx.input_video else "audio", ctx.user_prompt, None, None, duration
    )
    plan = mark_plan(
        plan,
        duration,
        dict(
            original_failure=str(reason),
            original_intent=prior,
            adjustments=adjustments,
            placement_adaptations=placement_adaptations,
            uncertainty="Original event timing/constraints unresolved; overlay positions are estimates.",
        ),
    )
    audit.update(status="planned", intent=deepcopy(plan.intent))
    ctx.warnings.append(
        dict(
            code="estimated_add_overlap",
            reason="User requested adding despite timing failure; overlap allowed and duration preserved.",
        )
    )
    return plan


def fit_existing_assets(ctx, plan, placements, assets, lengths, duration, reason):
    """Reposition complete generated material, keeping count and generator output."""
    from audio_edit.specialists.add_sound_specialist import AddSoundSpecialist

    if not enabled(ctx) or any(p.get("shared_window") for p in placements):
        raise ValueError("Material recovery is limited to standalone ADD")
    edits, remapped_assets, remapped_lengths = [], {}, {}
    for index, p in enumerate(placements):
        key = p["key"]
        length = _number(lengths[key])
        if length <= 0 or length > duration + 1e-4:
            raise ValueError("Complete material is longer than the entire fixed recording")
        length = min(length, duration)
        start = max(0.0, min(duration - length, p["start_time"]))
        old = plan.intent["edits"][p["edit_index"]]
        e = {
            k: deepcopy(old[k])
            for k in (
                "audio_prompt",
                "generator",
                "audio_reference",
                "speech_text",
                "voice_prompt",
                "layer_role",
                "relative_level_db",
                "reason",
            )
            if k in old
        }
        e.update(
            placement=dict(kind="single", anchor=dict(seconds=start), alignment="start"),
            generation=dict(target_seconds=length, source="planner_choice", adjustable=False),
            timeline_op="overlay",
            overflow_policy="error",
            sources=dict(
                placement="planner_choice",
                timeline_op="system_default",
                overflow_policy="system_default",
            ),
        )
        edits.append(e)
        new_key = f"{index}:0:0"
        remapped_assets[new_key], remapped_lengths[new_key] = assets[key], lengths[key]
    data = dict(
        intent_version=2,
        is_on_screen=plan.is_on_screen,
        edits=edits,
        audio_prompt=plan.audio_prompt,
        composer=deepcopy(plan.composer),
    )
    recovered = AddSoundSpecialist(None)._to_plan(
        data, plan.source_type, ctx.user_prompt, None, None, duration
    )
    recovered = mark_plan(
        recovered,
        duration,
        dict(
            original_failure=str(reason), original_intent=deepcopy(plan.intent), reused_assets=True
        ),
        asset_durations=remapped_lengths,
    )
    ctx.params.setdefault("add_material_recovery", []).append(
        dict(
            policy=POLICY,
            reason=str(reason),
            original_keys=list(assets),
            recovered_keys=list(remapped_assets),
            reused_assets=True,
            generation_calls=0,
        )
    )
    return recovered, remapped_assets, remapped_lengths
