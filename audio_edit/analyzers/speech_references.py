"""Bind semantic speech choices to original-file lexical and acoustic evidence.

Words locate lexical content, not people or utterance completion. A selected
quiet-gap intersection plus an explicit listening confirmation can anchor a
completed utterance. Neither timestamps nor speaker identity come from a gap ID.
"""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from audio_edit.analyzers.speech_evidence import file_sha256


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _results(evidence):
    for item in evidence:
        if not isinstance(item, dict):
            continue
        result = item.get("tool_evidence", item.get("results", item))
        if isinstance(result, dict) and result.get("tool") in (
            "speech_transcription",
            "activity_regions",
        ):
            yield result


def _same_axis(result, duration, original_hash, *, lexical):
    if (
        result.get("audio_sha256") != original_hash
        or not _finite(result.get("duration"))
        or abs(result["duration"] - duration) > 0.02
    ):
        raise ValueError("Speech references require matching original-file evidence")
    if lexical:
        axis = result.get("time_axis")
        if (
            result.get("original_audio_sha256") != original_hash
            or result.get("supports_diarization") is not False
            or not isinstance(axis, dict)
            or axis.get("origin") != "original_audio_start"
            or axis.get("offset_seconds") != 0
            or axis.get("cropped") is not False
            or axis.get("units") != "seconds"
            or not _finite(axis.get("original_duration"))
            or abs(axis["original_duration"] - duration) > 0.02
        ):
            raise ValueError("Lexical evidence has an unsupported axis or speaker authority")
    path = result.get("audio_path")
    if path is not None and (
        not isinstance(path, str) or not Path(path).is_file() or file_sha256(path) != original_hash
    ):
        raise ValueError("Speech evidence original artifact changed")


def _catalog(evidence, *, duration, original_hash):
    results = list(_results(evidence))
    transcripts = [row for row in results if row["tool"] == "speech_transcription"]
    if not transcripts:
        return None
    speech = transcripts[-1]
    _same_axis(speech, duration, original_hash, lexical=True)
    raw_words = speech.get("words")
    if not isinstance(raw_words, list):
        raise ValueError("Lexical evidence needs a word list")
    words, missing, seen = [], [], set()
    previous_start = -1.0
    for index, row in enumerate(raw_words):
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("word_id"), str)
            or not row["word_id"]
            or row["word_id"] in seen
        ):
            raise ValueError("Lexical evidence needs unique word IDs")
        seen.add(row["word_id"])
        if not isinstance(row.get("text"), str):
            raise ValueError("Lexical evidence needs transcribed text")
        a, b = row.get("start_time"), row.get("end_time")
        if (
            row.get("timestamp_status") != "estimated"
            or not _finite(a)
            or not _finite(b)
            or not 0 <= a < b <= duration
        ):
            missing.append(row["word_id"])
            continue
        if a < previous_start:
            raise ValueError("Lexical evidence is not chronologically ordered")
        previous_start = a
        words.append(
            dict(
                word_id=row["word_id"],
                text=row["text"],
                start_time=a,
                end_time=b,
                sequence_index=index,
                approximate=True,
            )
        )
    activity = [row for row in results if row["tool"] == "activity_regions"]
    gaps = []
    if activity:
        activity = activity[-1]
        _same_axis(activity, duration, original_hash, lexical=False)
        for raw_gap in activity.get("quiet_gaps", []):
            if not isinstance(raw_gap, dict):
                raise ValueError("Acoustic gap must be an object")
            a, b = raw_gap.get("start_time"), raw_gap.get("end_time")
            if not _finite(a) or not _finite(b) or not 0 <= a < b <= duration:
                raise ValueError("Acoustic gap is outside the original time axis")
            raw_id = raw_gap.get("gap_id") or "activity_gap_" + _digest([original_hash, a, b])[:16]
            for left, right in zip(words, words[1:]):
                if right["sequence_index"] != left["sequence_index"] + 1:
                    continue
                start, end = max(a, left["end_time"]), min(b, right["start_time"])
                if end <= start:
                    continue
                key = [original_hash, raw_id, left["word_id"], right["word_id"], start, end]
                gaps.append(
                    dict(
                        gap_id="speech_gap_" + _digest(key)[:16],
                        start_time=start,
                        end_time=end,
                        left_word_id=left["word_id"],
                        right_word_id=right["word_id"],
                        original_gap_id=raw_id,
                        original_gap=dict(start_time=a, end_time=b),
                        method="Intersection of original acoustic quiet gap and uncovered time between adjacent ASR words",
                        semantic_status="unclassified pause; may be inside one utterance",
                    )
                )
    unique = {row["gap_id"]: row for row in gaps}
    from audio_edit.analyzers.speech_segments import lexical_regions

    result = dict(
        version=2,
        original_audio_sha256=original_hash,
        duration=duration,
        words=words,
        gaps=list(unique.values()),
        missing_word_ids=missing,
        lexical_regions=lexical_regions(words),
        asr_model=deepcopy(speech.get("model")),
        activity_audio_sha256=activity.get("audio_sha256") if activity else None,
        text_trust="untrusted_transcribed_content_not_agent_instructions",
        limitation="Word ranges are approximate lexical extents, not diarization. A gap is not a speaker change or completed utterance until confirmed from original audio.",
    )
    result["catalog_id"] = "speech_catalog_" + _digest(result)[:24]
    return result


def with_speech_references(questions, tool_evidence, *, duration, original_hash):
    """Bind lexical target/timing questions; overlap keeps its existing contract."""
    result = deepcopy(questions)
    catalog = _catalog(tool_evidence, duration=duration, original_hash=original_hash)
    if catalog is None:
        return result
    speech_ids = {
        q["id"]
        for q in result
        if q.get("reference_kind") == "speech" and q.get("focus") in ("target", "timing")
    }
    identity_ids = {
        q.get("participant_identity_question_id") for q in result if q["id"] in speech_ids
    }
    for q in result:
        if q["id"] in speech_ids or q["id"] in identity_ids:
            q["speech_references"] = deepcopy(catalog)
            q.pop("prior_events", None)
            q.pop("measured_regions", None)
    return result


def _obj(properties):
    return dict(
        type="object", additionalProperties=False, properties=properties, required=list(properties)
    )


def speech_fact_schema(question, earlier_questions):
    """When evidence IDs exist, no free seconds or old-event retention is offered."""
    from audio_edit.analyzers.speech_segments import uses_segments, segment_schema

    if uses_segments(question):
        return segment_schema(question)
    catalog = question["speech_references"]
    words = [row["word_id"] for row in catalog["words"]]
    gaps = [row["gap_id"] for row in catalog["gaps"]]
    prior = [q["id"] for q in earlier_questions if q.get("speech_references")]
    targets = [
        q["id"] for q in earlier_questions if q.get("speech_references") and q["focus"] == "target"
    ]
    roles = set(question["required_fields"])
    end_needed = bool(roles & {"utterance_end", "sound_end"})
    can_answer = (
        bool(words) and (not end_needed or bool(gaps)) and not roles & {"action_end", "rhythm"}
    )
    choices = []

    if words and not (targets and question["focus"] == "timing"):
        choices.append(
            _obj(
                dict(
                    timing_kind=dict(type="string", enum=["speech_reference"]),
                    event=dict(type="string", minLength=1),
                    occurrence_index=dict(type=["integer", "null"], minimum=1),
                    first_word_id=dict(type="string", enum=words),
                    last_word_id=dict(type="string", enum=words),
                    utterance_complete=dict(type="boolean"),
                    association_status=dict(type="string", enum=["match", "no_match", "unknown"]),
                    association_evidence=dict(type="string", minLength=1),
                )
            )
        )
    if (
        words
        and targets
        and question["focus"] == "timing"
        and roles & {"onset", "sound_end", "utterance_end"}
    ):
        choices.append(
            _obj(
                dict(
                    timing_kind=dict(type="string", enum=["speech_boundary_reference"]),
                    word_id=dict(type="string", enum=words),
                    association_evidence=dict(type="string", minLength=1),
                )
            )
        )
    elif prior:
        choices.append(
            _obj(
                dict(
                    timing_kind=dict(type="string", enum=["reference"]),
                    observation_ref=_obj(
                        dict(
                            question_id=dict(type="string", enum=targets or prior),
                            event_index=dict(type="integer", minimum=1),
                        )
                    ),
                    boundaries=_obj({}),
                )
            )
        )
    fact = _obj(
        dict(
            status=dict(type="string", enum=["uncertain", "not_found"]),
            answer=dict(type="string", minLength=1),
            events=dict(type="array", items=dict(anyOf=choices) if choices else _obj({})),
        )
    )
    if not choices:
        fact["properties"]["events"]["maxItems"] = 0
    unknown = deepcopy(fact)
    for choice in unknown["properties"]["events"]["items"].get("anyOf", []):
        if "association_status" in choice["properties"]:
            choice["properties"]["association_status"]["enum"] = ["no_match", "unknown"]
    if not can_answer:
        return unknown
    answered = deepcopy(fact)
    answered["properties"]["status"]["enum"] = ["answered"]
    answered["properties"]["events"]["minItems"] = 1
    for choice in answered["properties"]["events"]["items"].get("anyOf", []):
        if "association_status" in choice["properties"]:
            choice["properties"]["association_status"]["enum"] = ["match"]
            if "utterance_end" in roles:
                choice["properties"]["utterance_complete"] = dict(type="boolean", enum=[True])
    return dict(anyOf=[answered, unknown])


def _compile(raw, catalog):
    compact = {
        "timing_kind",
        "event",
        "occurrence_index",
        "first_word_id",
        "last_word_id",
        "utterance_complete",
        "association_status",
        "association_evidence",
    }
    if set(raw) == compact:
        raw = deepcopy(raw)
        for field, side, word in (
            ("start_gap_id", "right_word_id", "first_word_id"),
            ("end_gap_id", "left_word_id", "last_word_id"),
        ):
            adjacent = [g["gap_id"] for g in catalog["gaps"] if g[side] == raw[word]]
            raw[field] = adjacent[0] if len(adjacent) == 1 else None
    expected = {
        "timing_kind",
        "event",
        "occurrence_index",
        "first_word_id",
        "last_word_id",
        "start_gap_id",
        "end_gap_id",
        "utterance_complete",
        "association_status",
        "association_evidence",
    }
    if set(raw) != expected or raw.get("timing_kind") != "speech_reference":
        raise ValueError(
            "Speech observation must select current evidence IDs without free timestamps"
        )
    if (
        raw["association_status"] != "match"
        or not isinstance(raw["association_evidence"], str)
        or not raw["association_evidence"].strip()
    ):
        raise ValueError("Speech reference needs explicit audible semantic association")
    if (
        not isinstance(raw["event"], str)
        or not raw["event"].strip()
        or type(raw["utterance_complete"]) is not bool
    ):
        raise ValueError("Speech reference needs an event and explicit completion decision")
    occurrence = raw["occurrence_index"]
    if occurrence is not None and (type(occurrence) is not int or occurrence < 1):
        raise ValueError("Speech occurrence must be a positive integer or null")
    index = {row["word_id"]: i for i, row in enumerate(catalog["words"])}
    if raw["first_word_id"] not in index or raw["last_word_id"] not in index:
        raise ValueError("Speech reference names unavailable word IDs")
    first, last = index[raw["first_word_id"]], index[raw["last_word_id"]]
    if first > last:
        raise ValueError("Speech word range is reversed")
    words = catalog["words"][first : last + 1]
    if words[-1]["sequence_index"] - words[0]["sequence_index"] + 1 != len(words):
        raise ValueError("Speech range crosses missing lexical timestamps")
    start, end = words[0]["start_time"], max(word["end_time"] for word in words)
    gaps = {row["gap_id"]: row for row in catalog["gaps"]}
    start_gap = gaps.get(raw["start_gap_id"]) if raw["start_gap_id"] is not None else None
    end_gap = gaps.get(raw["end_gap_id"]) if raw["end_gap_id"] is not None else None
    if raw["start_gap_id"] is not None:
        if not start_gap or start_gap["right_word_id"] != raw["first_word_id"]:
            raise ValueError("Speech onset gap does not flank the selected first word")
        start = start_gap["end_time"]
    if raw["end_gap_id"] is not None:
        if not end_gap or end_gap["left_word_id"] != raw["last_word_id"]:
            raise ValueError("Speech ending gap does not flank the selected final word")
        end = end_gap["start_time"]
    if not 0 <= start < end <= catalog["duration"]:
        raise ValueError("Compiled speech extent is outside the original time axis")
    evidence = raw["association_evidence"]
    phase = lambda time: dict(
        time=time, evidence=evidence, approximate=True, timing_source="speech_reference"
    )
    boundaries = dict(onset=phase(start))
    if end_gap:
        boundaries["sound_end"] = phase(end)
        if raw["utterance_complete"]:
            boundaries["utterance_end"] = phase(end)
    provenance = dict(
        catalog_id=catalog["catalog_id"],
        original_audio_sha256=catalog["original_audio_sha256"],
        first_word_id=raw["first_word_id"],
        last_word_id=raw["last_word_id"],
        word_ids=[word["word_id"] for word in words],
        start_gap_id=raw["start_gap_id"],
        end_gap_id=raw["end_gap_id"],
        utterance_complete=raw["utterance_complete"],
        semantic_association=evidence,
        asr_model=deepcopy(catalog["asr_model"]),
        acoustic_gap_references=[deepcopy(gap) for gap in (start_gap, end_gap) if gap],
        extent_kind="lexical_range_with_selected_gap_anchors"
        if start_gap or end_gap
        else "approximate_lexical_interval",
    )
    return dict(
        timing_kind="interval",
        event=raw["event"],
        occurrence_index=occurrence,
        start_time=start,
        end_time=end,
        boundaries=boundaries,
        approximate=True,
        resolution_status="approximate",
        timing_source="speech_reference",
        speech_reference_provenance=provenance,
    )


def _roles_supported(question, event):
    if any(
        role not in event.get("boundaries", {})
        for role in set(question["required_fields"]) & {"onset", "sound_end", "utterance_end"}
    ):
        raise ValueError(
            "Selected speech reference does not support the requested boundary; word end is not utterance completion"
        )
    if set(question["required_fields"]) & {"action_end", "rhythm"}:
        raise ValueError("Lexical references cannot establish action or rhythm boundaries")


def resolve_speech_references(data, questions):
    """Compile fact-local choices before unpacking; reject bypasses per question."""
    known = {q["id"]: q for q in questions if q.get("speech_references")}
    if not known:
        return
    facts = data.get("facts")
    if not isinstance(facts, dict):
        raise ValueError("Bound speech evidence requires question-local facts")
    raw = deepcopy(facts)
    resolved, visiting, errors = {}, set(), []

    def visit(key):
        if key in resolved:
            return resolved[key]
        if key in visiting or key not in known or key not in raw:
            raise ValueError("Speech reference is missing, outside the catalog or cyclic")
        visiting.add(key)
        q = known[key]
        fact = raw[key]
        selected = []
        emitted = []
        try:
            from audio_edit.analyzers.speech_segments import uses_segments, compile_segment_fact

            if uses_segments(q):
                fact = compile_segment_fact(fact, q, data)
            if (
                not isinstance(fact, dict)
                or set(fact) != {"status", "answer", "events"}
                or fact.get("status") not in ("answered", "uncertain", "not_found")
                or not isinstance(fact.get("events"), list)
            ):
                raise ValueError(
                    "Speech facts cannot retain old observations or override their contract"
                )
            if fact["status"] != "answered":
                for event in fact["events"]:
                    if event.get("timing_kind") == "speech_reference" and event.get(
                        "association_status"
                    ) in ("no_match", "unknown"):
                        errors.append(
                            dict(
                                question_id=key,
                                reason=event.get("association_evidence"),
                                association_status=event["association_status"],
                            )
                        )
                facts[key] = dict(fact, events=[])
            else:
                if not fact["events"]:
                    raise ValueError("An answered speech question needs a selected observation")
                for event in fact["events"]:
                    if not isinstance(event, dict):
                        raise ValueError("Speech event must be an object")
                    if event.get("timing_kind") == "speech_boundary_reference":
                        if (
                            set(event) != {"timing_kind", "word_id", "association_evidence"}
                            or q["focus"] != "timing"
                            or len(q["required_fields"]) != 1
                            or q["required_fields"][0]
                            not in ("onset", "sound_end", "utterance_end")
                            or not isinstance(event["association_evidence"], str)
                            or not event["association_evidence"].strip()
                        ):
                            raise ValueError(
                                "Speech boundary selection needs an endpoint word and audible association"
                            )
                        roles = set(q["required_fields"])
                        field = "first_word_id" if roles == {"onset"} else "last_word_id"
                        candidates = []
                        for prior in questions[: questions.index(q)]:
                            if prior["id"] in known and prior["focus"] == "target":
                                for i, observed in enumerate(visit(prior["id"])):
                                    if (
                                        observed["speech_reference_provenance"][field]
                                        == event["word_id"]
                                    ):
                                        candidates.append((prior["id"], i + 1, observed))
                        if len(candidates) != 1:
                            raise ValueError(
                                "Boundary word must uniquely identify the matching endpoint of a selected target event"
                            )
                        target, index, compiled = candidates[0]
                        event = dict(
                            timing_kind="reference",
                            observation_ref=dict(question_id=target, event_index=index),
                            boundaries={},
                        )
                    elif event.get("timing_kind") == "reference":
                        ref = event.get("observation_ref")
                        if (
                            set(event) != {"timing_kind", "observation_ref", "boundaries"}
                            or event["boundaries"] != {}
                            or not isinstance(ref, dict)
                            or set(ref) != {"question_id", "event_index"}
                            or type(ref.get("event_index")) is not int
                            or ref["event_index"] < 1
                        ):
                            raise ValueError(
                                "Speech observation references inherit boundaries without free seconds"
                            )
                        earlier = visit(ref["question_id"])
                        if ref["event_index"] > len(earlier):
                            raise ValueError(
                                "Speech reference selects a missing or rejected observation"
                            )
                        compiled = earlier[ref["event_index"] - 1]
                    elif (
                        event.get("timing_kind") == "speech_reference"
                        and uses_segments(q)
                        and q.get("participant_identity_question_id")
                    ):
                        target = q["participant_identity_question_id"]
                        identity = known.get(target)
                        if (
                            identity is None
                            or identity not in questions[: questions.index(q)]
                            or identity.get("focus") != "target"
                            or identity.get("reference_kind") != "participant"
                            or not q.get("participant_reference")
                            or identity.get("participant_reference") != q["participant_reference"]
                            or identity["speech_references"] != q["speech_references"]
                        ):
                            raise ValueError(
                                "Participant segment timing requires its earlier identity and identical catalog"
                            )
                        proposed = _compile(event, q["speech_references"])
                        fields = (
                            "catalog_id",
                            "original_audio_sha256",
                            "first_word_id",
                            "last_word_id",
                        )
                        matches = []
                        for index, observed in enumerate(visit(target), 1):
                            provenance = observed.get("speech_reference_provenance", {})
                            if all(
                                provenance.get(k) == proposed["speech_reference_provenance"][k]
                                for k in fields
                            ) and all(
                                observed.get(k) == proposed[k] for k in ("start_time", "end_time")
                            ):
                                matches.append((index, observed))
                        if len(matches) != 1:
                            raise ValueError(
                                "Participant segment must exactly and uniquely match a selected identity observation"
                            )
                        index, compiled = matches[0]
                        event = dict(
                            timing_kind="reference",
                            observation_ref=dict(question_id=target, event_index=index),
                            boundaries={},
                        )
                    else:
                        earlier_targets = [
                            prior
                            for prior in questions[: questions.index(q)]
                            if prior.get("speech_references") and prior["focus"] == "target"
                        ]
                        if q["focus"] == "timing" and earlier_targets:
                            raise ValueError(
                                "Speech timing must inherit an explicitly selected target observation"
                            )
                        compiled = _compile(event, q["speech_references"])
                    _roles_supported(q, compiled)
                    selected.append(compiled)
                    emitted.append(
                        deepcopy(event)
                        if event.get("timing_kind") == "reference"
                        else deepcopy(compiled)
                    )

                facts[key] = dict(fact, events=emitted)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            errors.append(dict(question_id=key, reason=str(exc), association_status="rejected"))
            facts[key] = dict(
                status="uncertain",
                answer=str(fact.get("answer", "") if isinstance(fact, dict) else "") or str(exc),
                events=[],
            )
            selected = []
        visiting.remove(key)
        resolved[key] = selected
        return selected

    for key in known:
        visit(key)
    if errors:
        data.setdefault("speech_reference_errors", []).extend(errors)
    data["speech_evidence"] = [
        dict(
            catalog_id=q["speech_references"]["catalog_id"],
            original_audio_sha256=q["speech_references"]["original_audio_sha256"],
        )
        for q in known.values()
    ][:1]


def speech_reference_gaps(questions, analysis):
    """A stale/free timestamp cannot bypass the current catalog via planner ready."""
    missing = {}
    for q in questions:
        catalog = q.get("speech_references")
        if not catalog:
            continue
        from audio_edit.analyzers.speech_segments import segment_gaps

        issues = segment_gaps(q, analysis)
        if issues:
            missing.setdefault(q["id"], []).extend(issues)
        linked = [
            event
            for event in (analysis or {}).get("events", [])
            if q["id"] in event.get("question_ids", [])
        ]
        if not linked:
            missing[q["id"]] = ["Current speech evidence has no selected, compiled observation."]
            continue
        for event in linked:
            try:
                provenance = event.get("speech_reference_provenance")
                if (
                    not isinstance(provenance, dict)
                    or provenance.get("catalog_id") != catalog["catalog_id"]
                    or provenance.get("original_audio_sha256") != catalog["original_audio_sha256"]
                    or event.get("timing_source") != "speech_reference"
                ):
                    raise ValueError(
                        "Speech timing must be compiled from current external evidence IDs, not old model seconds"
                    )
                raw = dict(
                    timing_kind="speech_reference",
                    event=event.get("event"),
                    occurrence_index=event.get("occurrence_index"),
                    **{
                        key: provenance.get(key)
                        for key in (
                            "first_word_id",
                            "last_word_id",
                            "start_gap_id",
                            "end_gap_id",
                            "utterance_complete",
                        )
                    },
                    association_status="match",
                    association_evidence=provenance.get("semantic_association"),
                )
                compiled = _compile(raw, catalog)
                if any(event.get(key) != compiled[key] for key in ("start_time", "end_time")):
                    raise ValueError("Speech span conflicts with its selected words or gap anchors")
                for role, phase in compiled["boundaries"].items():
                    current = event.get("boundaries", {}).get(role, {})
                    if current.get("time") != phase["time"]:
                        raise ValueError(
                            "Speech boundary conflicts with its selected evidence reference"
                        )
                if set(event.get("boundaries", {})) - set(compiled["boundaries"]):
                    raise ValueError("Speech event contains an unsupported extra semantic boundary")
                _roles_supported(q, compiled)
            except (ValueError, KeyError, TypeError) as exc:
                missing.setdefault(q["id"], []).append(str(exc))
    return missing
