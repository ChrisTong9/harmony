"""One-way add input adapters. No legacy timing is consulted after this boundary."""

from copy import deepcopy
from dataclasses import asdict, is_dataclass
import math

from audio_edit.plans.contract_normalization import normalize_edit
from audio_edit.plans.temporal_boundaries import expand_each, BoundaryEvidenceError


def normalize_composer(value):
    """An optional mixer object, or a legacy name retained as inert metadata."""
    if value is None or isinstance(value, dict):
        return deepcopy(value), []
    if isinstance(value, str):
        result = {"name": value}
        return result, [
            dict(
                code="composer_label",
                field="composer",
                before=value,
                after=result,
                reason="A composer name is metadata, not executable mix settings; existing mixer defaults apply",
            )
        ]
    raise ValueError("composer must be a mixer settings object, a name string, or null")


def normalize_modern_edit(edit):
    """Unwrap equivalent wire shapes; never infer an event, extend, or move time."""
    e = deepcopy(edit)
    audit = []
    p = e.get("placement")
    if not isinstance(p, dict):
        raise ValueError("Each edit needs a placement object")

    rule = p.get("repeat")
    if (
        p.get("kind") == "repeat"
        and "anchors" not in p
        and isinstance(rule, dict)
        and type(rule.get("count")) is int
        and rule["count"] == 1
        and set(rule) <= {"count", "spacing_seconds"}
    ):
        spacing = rule.get("spacing_seconds")
        if spacing is None or (
            isinstance(spacing, (int, float))
            and not isinstance(spacing, bool)
            and math.isfinite(spacing)
            and spacing >= 0
        ):
            before = deepcopy(rule)
            p["kind"] = "single"
            p.pop("repeat")
            audit.append(
                dict(
                    code="single_count_repetition",
                    before=before,
                    reason="One sound per selected origin; preserve each selector, phase, offset and duration without artificial spacing",
                )
            )
    anchor = p.get("anchor")
    if isinstance(anchor, dict) and set(anchor) == {"within"}:
        if "within" in p and p["within"] != anchor["within"]:
            raise ValueError("Conflicting within ranges")
        p["within"] = p.pop("anchor")["within"]
        audit.append(
            dict(
                code="within_unwrapped",
                reason="Move the same range to placement.within; no coordinate changes",
            )
        )
    for key, default in (("timeline_op", "overlay"), ("overflow_policy", "error")):
        if e.get(key) is None:
            e[key] = default
            audit.append(
                dict(
                    code="entry_default",
                    field=key,
                    after=default,
                    reason="Existing conservative default; no extension authorized",
                )
            )
    if (
        e["timeline_op"] in ("append", "prepend")
        and p.get("kind") in ("single", "repeat")
        and not any(k in p for k in ("anchor", "anchors", "within"))
    ):
        p["anchor"] = {"media_edge": "end" if e["timeline_op"] == "append" else "start"}
        audit.append(
            dict(
                code="operation_coordinate",
                reason="Insertion coordinate follows the declared operation",
            )
        )
    within = p.get("within")
    if isinstance(within, dict) and "selection" in within:
        nested = within["selection"]
        if "selection" in p and p["selection"] != nested:
            raise ValueError("Conflicting random placement selections")
        p["selection"] = within.pop("selection")
        audit.append(
            dict(
                code="within_selection_unwrapped",
                reason="Preserve declared selection at placement level; keep both range anchors unchanged",
            )
        )
    if p.get("selection") in ("start", "end", "center"):
        selected = p["selection"]
        if "alignment" in p and p["alignment"] != selected:
            raise ValueError("Conflicting placement alignment and selection")
        p["alignment"] = p.pop("selection")
        audit.append(
            dict(
                code="selection_alignment_alias",
                after=selected,
                reason="Preserve declared deterministic range alignment",
            )
        )
    if p.get("selection") == "random":
        import secrets

        p["selection"] = dict(mode="random", unit=secrets.randbits(53) / 2**53)
        audit.append(
            dict(
                code="random_choice",
                after=deepcopy(p["selection"]),
                reason="Draw once at entry; compiler reuses this choice before and after generation",
            )
        )
    sources = e.get("sources")
    if sources is None:
        sources = {}
    if not isinstance(sources, dict):
        raise ValueError("sources must be an object or null")
    extra = {
        k: v for k, v in sources.items() if k not in ("placement", "timeline_op", "overflow_policy")
    }
    if extra:
        audit.append(
            dict(
                code="unused_source_metadata",
                before=extra,
                reason="Generation provenance belongs to generation.source; only timing feasibility fields remain here",
            )
        )
    e["sources"] = {k: v for k, v in sources.items() if k not in extra}
    for key in ("placement", "timeline_op", "overflow_policy"):
        if e["sources"].get(key) is None or any(
            a.get("code") == "entry_default" and a.get("field") == key for a in audit
        ):
            e["sources"][key] = "system_default"

    def reference(value):
        if isinstance(value, dict):
            if value.get("boundary_role") == "utterance_start":
                value["boundary_role"] = "onset"
                audit.append(
                    dict(
                        code="utterance_start_alias",
                        after="onset",
                        reason="Same utterance onset; reference and occurrence unchanged",
                    )
                )
            keys = (
                "seconds",
                "event_id",
                "media_edge",
                "media_fraction",
                "reference_text",
            )
            populated = [k for k in keys if value.get(k) is not None]
            if len(populated) == 1:
                for k in keys:
                    if k != populated[0] and k in value and value[k] is None:
                        value.pop(k)
                        audit.append(dict(code="null_reference_field_removed", field=k))
            for v in value.values():
                reference(v)
        elif isinstance(value, list):
            for v in value:
                reference(v)

    reference(p)
    return e, audit


def _generation(data, edit=None):
    value = (edit or {}).get("event_duration")
    if value is None:
        value = data.get("duration")
    return dict(target_seconds=value, source="system_default", adjustable=False)


def adapt_edits(data, *, audio_analysis=None, visual_analysis=None, duration=None):
    """v1 callers supplied redundant windows/anchors. Normalize exactly once."""
    events = {
        e["event_id"]: e
        for a, k in ((audio_analysis, "events"), (visual_analysis, "visual_events"))
        for e in (a or {}).get(k, [])
    }
    converted = []
    audits = []
    for raw in data["edits"]:
        edit, audit = normalize_edit(raw)
        audits.extend(audit)
        for e in expand_each([edit], events):
            window = e["window"]
            start = window["start"]
            end = window.get("end")
            kind = e["coverage"]
            generation = _generation(data, e)
            if kind == "continuous":
                if end is None:
                    if (
                        e.get("timeline_op") == "overlay"
                        and e.get("overflow_policy", "error") != "extend"
                    ):
                        end = {"media_edge": "end"}
                        audits.append(
                            dict(
                                code="legacy_open_coverage_to_eof",
                                source="system_default",
                            )
                        )
                    else:
                        raise ValueError(
                            "Legacy open continuous coverage needs an explicit end; no guessed extent"
                        )
                placement = dict(kind="cover", range=dict(start=start, end=end))
            else:
                alignment = "end" if e.get("alignment") == "before" else "start"
                placement = dict(
                    kind="repeat" if kind == "repeated" else "single",
                    alignment=alignment,
                )
                anchors = e.get("placement_anchors")
                if anchors:
                    if len(anchors) == 1 and kind == "single":
                        placement["anchor"] = anchors[0]
                    else:
                        placement.update(kind="repeat", anchors=anchors)
                else:
                    placement["anchor"] = start
                    if kind == "repeated":
                        repetition = {"count": e["count"]}
                        if e.get("distribution") == "uniform":
                            repetition.update(
                                mode="uniform",
                                explicitly_requested=e.get(
                                    "distribution_explicitly_requested", False
                                ),
                            )
                        elif e.get("distribution") == "sequential":
                            repetition["mode"] = "sequential"
                        else:
                            repetition["offsets_seconds"] = e.get("offsets_seconds")
                        placement["repeat"] = repetition
                if end is not None and e["timeline_op"] == "overlay":
                    placement["within"] = dict(start=start, end=end)
            converted.append(
                dict(
                    placement=placement,
                    generation=generation,
                    timeline_op=e["timeline_op"],
                    overflow_policy=e.get("overflow_policy", "error"),
                    reason=e.get("reason", "Legacy explicit plan"),
                    default_reason=e.get("default_reason"),
                    audio_prompt=e.get("audio_prompt") or data.get("audio_prompt"),
                    layer_role=e.get("layer_role") or data.get("layer_role"),
                    relative_level_db=e.get("relative_level_db"),
                    sources=dict(
                        placement="system_default",
                        timeline_op="system_default",
                        overflow_policy="system_default",
                    ),
                )
            )
    return dict(
        version=2,
        revision=1,
        edits=converted,
        compatibility=[dict(adapter="edits_v1", normalizations=audits)],
    )


def adapt_temporal(
    data,
    *,
    duration,
    audio_analysis=None,
    visual_analysis=None,
    temporal_hint=None,
    shared_windows=None,
):
    from audio_edit.plans.temporal import (
        temporal_constraint_from_dict,
        resolve_temporal_constraint,
        expand_repeated_placements,
        resolved_placement_from_dict,
    )

    temporal = temporal_constraint_from_dict(temporal_hint or data.get("temporal"))
    d = float(data["duration"])
    if shared_windows is not None:
        spans = [resolved_placement_from_dict(p) for p in shared_windows]
        edits = []
        for span in spans:
            if span.end_time is None:
                raise BoundaryEvidenceError("Replacement shared window needs both boundaries")
            count = getattr(temporal, "count", None) or 1
            target = d if count > 1 else span.end_time - span.start_time
            region = dict(start={"seconds": span.start_time}, end={"seconds": span.end_time})
            placement = dict(
                kind="repeat" if count > 1 else "single",
                within=region,
                alignment="start",
            )
            if count > 1:
                placement["repeat"] = dict(
                    count=count,
                    **(
                        {"spacing_seconds": temporal.spacing_seconds}
                        if temporal.spacing_seconds is not None
                        else {"mode": "sequential"}
                    ),
                )
            edits.append(
                dict(
                    placement=placement,
                    generation=dict(
                        target_seconds=target, source="system_default", adjustable=False
                    ),
                    timeline_op="overlay",
                    overflow_policy="error",
                    audio_prompt=data["audio_prompt"],
                    layer_role=data.get("layer_role"),
                    reason="One-way legacy replacement adaptation; authoritative range retained",
                    default_reason="Approximate authoritative replacement boundary"
                    if span.approximate
                    else None,
                )
            )
        return dict(
            version=2,
            revision=1,
            edits=edits,
            compatibility=[
                dict(
                    adapter="temporal_v0_shared",
                    policy="Legacy unconstrained generation length derives from each supplied window; repeat duration remains bounded",
                )
            ],
        ), temporal
    else:
        result = resolve_temporal_constraint(
            temporal,
            media_duration=duration,
            event_duration=d,
            audio_analysis=audio_analysis,
            visual_analysis=visual_analysis,
        )
        if result.status == "unresolved":
            raise BoundaryEvidenceError(result.reason)
        spans = result.placements
        if not spans:
            from audio_edit.plans.temporal import ResolvedPlacement

            a = float(data.get("start_time") or 0.0)
            b = data.get("end_time")
            spans = [ResolvedPlacement(a, float(b) if b is not None else a + d)]
        spans = expand_repeated_placements(spans, temporal, d)
    edits = []
    op = getattr(temporal, "placement_mode", "overlay")
    for p in spans:
        start = duration if op == "append" else p.start_time
        end = p.end_time if p.end_time is not None else start + d
        edits.append(
            dict(
                placement=dict(
                    kind="single",
                    anchor={"seconds": start},
                    within=dict(start={"seconds": start}, end={"seconds": end}),
                    alignment="start",
                ),
                generation=dict(target_seconds=d, source="system_default", adjustable=False),
                timeline_op=op,
                overflow_policy="extend"
                if op == "append"
                else getattr(temporal, "overflow_policy", "error"),
                audio_prompt=data["audio_prompt"],
                layer_role=data.get("layer_role"),
                reason="One-way legacy temporal adaptation",
                default_reason="Legacy approximate evidence" if p.approximate else None,
            )
        )
    return dict(
        version=2, revision=1, edits=edits, compatibility=[dict(adapter="temporal_v0")]
    ), temporal


def shared_intent(data, shared_windows):
    """Add receives scopes from replacement, not permission to relocate them."""
    spans = [asdict(p) if is_dataclass(p) else deepcopy(p) for p in shared_windows]
    raw = data.get("edits", [])
    if not spans or len(raw) != len(spans):
        raise ValueError("Replacement edit count differs from authoritative windows")
    edits, audit = [], []
    for i, (e, sw) in enumerate(zip(raw, spans)):
        if e.get("shared_window_index") is not None and (
            isinstance(e["shared_window_index"], bool) or e["shared_window_index"] != i
        ):
            raise ValueError(
                "Replacement addition must reference its shared window index, not replan coordinates"
            )
        if (e.get("timeline_op") or "overlay") != "overlay" or (
            e.get("overflow_policy") or "error"
        ) != "error":
            raise ValueError("Replacement cannot change timeline operation or overflow policy")
        proposed = e.get("placement")
        if proposed is not None and not isinstance(proposed, dict):
            raise ValueError("Shared placement must be an object or null")
        kind = e.get("kind") or (proposed or {}).get("kind", "single")
        region = dict(start={"seconds": sw["start_time"]}, end={"seconds": sw["end_time"]})
        placement = dict(
            kind=kind,
            **({"range": region} if kind == "cover" else {"within": region, "alignment": "start"}),
        )
        if kind == "repeat":
            placement["repeat"] = deepcopy(e.get("repeat") or (proposed or {}).get("repeat", {}))
        if proposed is not None:
            if proposed != placement:
                raise ValueError("Replacement addition conflicts with authoritative shared window")
            audit.append(
                dict(
                    code="duplicate_shared_placement",
                    edit_index=i,
                    reason="Exact duplicate of supplied scope; only shared scope remains authoritative",
                )
            )
        if e.get("shared_window_index") is None:
            audit.append(
                dict(
                    code="shared_window_order",
                    edit_index=i,
                    reason="One edit per window in declared order; no event selection",
                )
            )
        converted = {
            k: deepcopy(v)
            for k, v in e.items()
            if k not in ("shared_window_index", "kind", "repeat", "placement")
        }
        converted.update(
            placement=placement,
            shared_window=sw,
            timeline_op="overlay",
            overflow_policy="error",
        )
        if sw.get("approximate") or sw.get("resolution_status") in (
            "approximate",
            "estimated",
        ):
            converted["default_reason"] = (
                "Authoritative replacement window retains upstream timing uncertainty"
            )
        edits.append(converted)
    return dict(version=2, revision=1, edits=edits, compatibility=audit)


def attach_shared_windows(
    intent, shared_windows, *, duration, audio_analysis=None, visual_analysis=None
):
    """Compatibility validation for already explicit v1 plans; never overwrite conflicts."""
    spans = [asdict(p) if is_dataclass(p) else deepcopy(p) for p in shared_windows]
    if len(intent["edits"]) != len(spans):
        raise ValueError("Replacement edit count differs from authoritative windows")
    from audio_edit.plans.edit_contract import resolve_anchor

    events = {
        e["event_id"]: e
        for a, k in ((audio_analysis, "events"), (visual_analysis, "visual_events"))
        for e in (a or {}).get(k, [])
    }
    for e, sw in zip(intent["edits"], spans):
        region = e["placement"].get("range", e["placement"].get("within"))
        if not region:
            raise ValueError("Replacement changed authoritative removal window")
        a = resolve_anchor(region["start"], events, duration)[0]
        b = resolve_anchor(region["end"], events, duration)[0]
        if (
            abs(a - sw["start_time"]) > 1e-6
            or abs(b - sw["end_time"]) > 1e-6
            or e["timeline_op"] != "overlay"
            or e["overflow_policy"] != "error"
        ):
            raise ValueError("Replacement changed authoritative removal window")
        e["shared_window"] = sw
    return intent


def adapt_runner_plan(plan, duration):
    """Old dataclass callers enter the same compiler once, before any generation."""
    from audio_edit.plans.add_sound_plan import MediaAnalysis
    from audio_edit.plans.add_timing import compile_plan

    plan = deepcopy(plan)
    plan.media_analysis = plan.media_analysis or MediaAnalysis()
    spans = (
        plan.execution_placements
        or [asdict(p) for p in plan.placements]
        or [dict(start_time=plan.start_time, end_time=plan.end_time)]
    )
    edits = []
    for p in spans:
        op = p.get("timeline_op") or getattr(plan.temporal, "placement_mode", "overlay")
        start = duration if op == "append" else p["start_time"]
        d = p.get("event_duration", plan.duration)
        end = p.get("end_time")
        if op == "append":
            end = None
        if end is not None and d is not None and d > end - start + 1e-6:
            raise ValueError(
                "Legacy event duration exceeds placement; explicit re-planning required"
            )
        edits.append(
            dict(
                window=dict(
                    start={"seconds": start},
                    end={"seconds": end} if end is not None else None,
                ),
                coverage=p.get("coverage", "single"),
                event_duration=d,
                count=1,
                distribution="explicit",
                offsets_seconds=[0.0],
                timeline_op=op,
                overflow_policy=p.get(
                    "overflow_policy",
                    "extend"
                    if op == "append"
                    else getattr(plan.temporal, "overflow_policy", "error"),
                ),
                reason="One-way legacy runner entry",
                audio_prompt=p.get("audio_prompt") or plan.audio_prompt,
                layer_role=p.get("layer_role") or plan.layer_role,
                relative_level_db=p.get("relative_level_db"),
            )
        )
    plan.intent = adapt_edits(
        dict(edits=edits, audio_prompt=plan.audio_prompt, duration=plan.duration),
        duration=duration,
    )
    plan.intent["compatibility"].append(dict(adapter="runner_dataclass"))
    plan.composer, audit = normalize_composer(plan.composer)
    plan.intent["compatibility"].extend(audit)
    plan.contract_version = 2
    return compile_plan(plan, duration=duration)
