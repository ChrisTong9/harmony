"""Preserve question-local evidence and paired scopes across the planning handoff.

No source-name heuristics, new timestamps, or extra model calls. The first brief
owns the dependency pair; the specialist still owns sound and generation choices.
"""

from copy import deepcopy

from audio_edit.plans.event_ids import has_measured_timing


def reference_key(anchor):
    return anchor.get("question_id") or anchor["reference_text"]


def _occurrence(value):
    return None if value == "one" else 1 if value == "first" else value


def _same_occurrence(question, request, analysis):
    if _occurrence(question["occurrence"]) == _occurrence(request.get("occurrence")):
        return True

    selector = request.get("occurrence")
    contract = (analysis or {}).get("edit_listening", {})
    if (
        question["occurrence"] != "one"
        or not (selector == "first" or type(selector) is int and selector == 1)
        or request.get("question_id") != question["id"]
        or request.get("reference_source") != "audio"
        or question.get("participant_identity_question_id")
        or contract.get("status") != "ready"
        or contract.get("missing_evidence")
    ):
        return False
    answers = [
        a for a in (analysis or {}).get("answers", []) if a.get("question_id") == question["id"]
    ]
    if len(answers) != 1 or answers[0].get("status") != "answered":
        return False

    events = [
        e
        for e in (analysis or {}).get("events", [])
        if question["id"] in (e.get("question_ids") or [])
    ]
    if len(events) != 1:
        return False
    event = events[0]
    return (
        type(event.get("occurrence_index")) is int
        and event["occurrence_index"] == 1
        and has_measured_timing(event)
        and event.get("resolution_status") in ("resolved", "approximate", "estimated")
        and not event.get("review_context_only")
        and not event.get("invalid_crop_boundary")
    )


def _interval_coverage(intent, analysis):
    """Cover a selected interval without claiming a separately measured phase.

    This applies only to a paired cover of the same interval question. A discrete
    sound placed after an utterance/action still needs its original phase proof.
    Observations, selectors, confidence, partial flags and timestamps are untouched.
    """
    known = {q["id"]: q for q in (analysis or {}).get("edit_listening", {}).get("requirements", [])}
    result = deepcopy(intent)

    def markers(value):
        if isinstance(value, dict):
            if "interval_edge" in value:
                yield value
            for item in value.values():
                yield from markers(item)
        elif isinstance(value, list):
            for item in value:
                yield from markers(item)

    for index, edit in enumerate(result["edits"]):
        placement = edit["placement"]
        region = placement.get("range") or {}
        pair = [region.get(edge) for edge in ("start", "end")]
        marked = list(markers(placement))
        question = None
        if placement.get("kind") == "cover" and all(isinstance(a, dict) for a in pair):
            start, end = pair
            q = known.get(start.get("question_id"))
            if (
                q
                and end.get("question_id") == q["id"]
                and q["focus"] == "timing"
                and q["required_fields"] == ["interval"]
                and q.get("reference_kind") in ("speech", "sound_event", "participant")
                and (
                    not q.get("participant_identity_question_id")
                    or _participant_interval_window_supported(q, analysis)
                )
                and all(
                    a.get("reference_source") == "audio"
                    and _same_occurrence(q, a, analysis)
                    and a.get("offset_seconds", 0) == 0
                    and not any(
                        k in a for k in ("seconds", "media_edge", "media_fraction", "event_id")
                    )
                    for a in pair
                )
            ):
                question = q
                if _occurrence(q["occurrence"]) is None:
                    candidates = [
                        e
                        for e in (analysis or {}).get("events", [])
                        if q["id"] in (e.get("question_ids") or [])
                        and has_measured_timing(e)
                        and not e.get("review_context_only")
                        and e.get("resolution_status") != "unresolved"
                    ]

                    # license to cover only the first fragment or their envelope.
                    if len(candidates) != 1:
                        question = None
        if marked:
            if (
                question is None
                or len(marked) != 2
                or {id(a) for a in marked} != {id(a) for a in pair}
                or any(
                    a.get("interval_edge") != edge or a.get("edge") != edge or "boundary_role" in a
                    for edge, a in zip(("start", "end"), pair)
                )
            ):
                raise ValueError(
                    "Interval edges require a paired cover of the same interval question"
                )
            continue
        if question is None:
            continue
        end_role = "utterance_end" if question["reference_kind"] == "speech" else "sound_end"
        if pair[0].get("boundary_role") != "onset" or pair[1].get("boundary_role") != end_role:
            continue
        before = deepcopy(region)
        for edge, anchor in zip(("start", "end"), pair):
            anchor.pop("boundary_role")
            anchor.update(edge=edge, interval_edge=edge)
        result.setdefault("listening_contract_adjustments", []).append(
            dict(
                edit_index=index,
                field="placement.range",
                before=before,
                after=deepcopy(region),
                reason="Paired coverage uses the selected interval edges, not a separate utterance/action completion",
            )
        )
    return result


def _participant_interval_window_supported(question, analysis):
    """A same-question cover keeps the existing complete participant checks."""
    from audio_edit.analyzers.participant_selection import participant_placements
    from audio_edit.plans.errors import UnresolvedEditError

    contract = (analysis or {}).get("edit_listening", {})
    selected = contract.get("participant_selection") or {}
    if (
        contract.get("status") != "ready"
        or contract.get("missing_evidence")
        or selected.get("timing_question_id") != question["id"]
        or selected.get("identity_question_id") != question.get("participant_identity_question_id")
        or not any(
            w.get("start_question_id") == w.get("end_question_id") == question["id"]
            for w in contract.get("placement_windows", [])
        )
    ):
        return False
    try:
        return len(participant_placements(analysis, analysis["media_duration"])) == 1
    except (UnresolvedEditError, ValueError, TypeError, KeyError):
        return False


def question_bindings(refs, analysis):
    requirements = (analysis or {}).get("edit_listening", {}).get("requirements", [])
    known = {q["id"]: q for q in requirements}
    bindings = {}
    for key, requests in refs.items():
        interval_pair = (
            len(requests) >= 2
            and {r.get("interval_edge") for r in requests} == {"start", "end"}
            and all(
                r.get("edge") == r.get("interval_edge") and "boundary_role" not in r
                for r in requests
            )
        )
        matches = []
        for request in requests:
            explicit = request.get("question_id")
            if explicit and explicit not in known:
                raise ValueError("Unknown listening question reference")
            if request.get("reference_source") == "visual":
                if explicit:
                    raise ValueError("An audio listening question cannot bind a visual reference")
                break
            candidates = (
                [known[explicit]]
                if explicit in known
                else [
                    q
                    for q in requirements
                    if q["subject"].strip().casefold()
                    == request["reference_text"].strip().casefold()
                ]
            )
            candidates = [
                q
                for q in candidates
                if q["focus"] == "timing"
                and _same_occurrence(q, request, analysis)
                and (
                    request.get("boundary_role") in q["required_fields"]
                    or interval_pair
                    and q["required_fields"] == ["interval"]
                    or request.get("measurement") == "beat_onsets"
                    and "rhythm" in q["required_fields"]
                )
            ]
            if len(candidates) != 1:
                if explicit:
                    raise ValueError(
                        "Listening reference changes or lacks its question, occurrence, or boundary role"
                    )
                break
            matches.append(candidates[0]["id"])
        if len(matches) != len(requests) or len(set(matches)) != 1:
            continue
        bindings[key] = [
            e["event_id"]
            for e in (analysis or {}).get("events", [])
            if matches[0] in (e.get("question_ids") or [])
            and has_measured_timing(e)
            and not e.get("review_context_only")
            and e.get("resolution_status") != "unresolved"
        ]
    return bindings


def _window_anchors(window, requirements):
    known = {q["id"]: q for q in requirements}
    anchors = {}
    same_interval = window["start_question_id"] == window["end_question_id"]
    for edge in ("start", "end"):
        q = known[window[edge + "_question_id"]]
        anchors[edge] = dict(
            question_id=q["id"],
            reference_text=q["subject"],
            reference_source="audio",
            reference_kind="utterance"
            if q["required_fields"][0] == "utterance_end"
            else "audible_event",
            occurrence=_occurrence(q["occurrence"]),
            boundary_role=q["required_fields"][0],
        )
        if same_interval:
            if (
                q["focus"] != "timing"
                or q["required_fields"] != ["interval"]
                or q.get("reference_kind") not in ("sound_event", "speech", "participant")
                or q.get("occurrence") == "each"
            ):
                raise ValueError(
                    "A same-question coverage window needs one selected complete interval"
                )
            anchors[edge].pop("boundary_role")
            anchors[edge].update(edge=edge, interval_edge=edge)
    return anchors


def apply_listening_windows(intent, analysis):
    """Entry adaptation: use both dependencies, never guess the missing endpoint."""
    intent = _interval_coverage(intent, analysis)
    contract = (analysis or {}).get("edit_listening", {})
    windows = contract.get("placement_windows", [])
    policy = contract.get("timeline_policy")
    if not windows and not policy:
        return intent
    result = deepcopy(intent)
    if policy and policy["mode"] == "preserve":
        for index, edit in enumerate(result["edits"]):
            if edit["timeline_op"] in ("append", "prepend"):
                raise ValueError(
                    "Adding timeline space conflicts with the instruction-derived preserve policy"
                )
            for field, value in (("timeline_op", "overlay"), ("overflow_policy", "error")):
                if edit[field] != value:
                    result.setdefault("listening_contract_adjustments", []).append(
                        dict(
                            edit_index=index,
                            field=field,
                            before=edit[field],
                            after=value,
                            reason="Preserve policy was fixed from the original instruction before listening",
                        )
                    )
                    edit[field] = value
                    edit.setdefault("sources", {})[field] = "system_default"
    if not windows:
        return result
    used = set()
    for edit_index, edit in enumerate(result["edits"]):
        index = edit.get("listening_window_index")
        if index is None and len(windows) == len(result["edits"]) == 1:
            index = 0
        if index is None:
            continue
        if type(index) is not int or not 0 <= index < len(windows):
            raise ValueError("Unknown listening placement window")
        if edit["timeline_op"] != "overlay":
            if edit.get("sources", {}).get("timeline_op") != "system_default":
                raise ValueError(
                    "A bounded original-timeline window conflicts with requested retiming"
                )
            edit["timeline_op"] = "overlay"
        if windows[index]["start_question_id"] == windows[index]["end_question_id"]:
            if edit["placement"]["kind"] != "cover" or set(edit["placement"]) != {"kind", "range"}:
                raise ValueError(
                    "A complete interval window requires continuous coverage of that passage"
                )
            required_range = _window_anchors(windows[index], contract["requirements"])
            if edit["placement"]["range"] != required_range:
                result.setdefault("listening_contract_adjustments", []).append(
                    dict(
                        edit_index=edit_index,
                        field="placement.range",
                        before=deepcopy(edit["placement"]["range"]),
                        after=deepcopy(required_range),
                        reason="Cover both edges of the original selected interval question; no invented phase or time",
                    )
                )
                edit["placement"]["range"] = required_range
            edit["listening_window_index"] = index
            used.add(index)
            continue
        if edit["placement"]["kind"] == "cover":
            raise ValueError(
                "A bounded discrete addition cannot silently become continuous coverage"
            )
        edit["listening_window_index"] = index
        edit["placement"]["within"] = _window_anchors(windows[index], contract["requirements"])

        used.add(index)
    if used != set(range(len(windows))):
        raise ValueError("The add plan omitted a required listening placement window")
    if any(w["start_question_id"] == w["end_question_id"] for w in windows):
        return _interval_coverage(result, analysis)
    return result


def validate_listening_windows(intent, analysis):
    """No later recompilation/generation step may discard the entry contract."""
    expected = apply_listening_windows(intent, analysis)
    if expected != intent:
        raise ValueError(
            "The add intent lost a required paired listening window or timeline constraint"
        )
