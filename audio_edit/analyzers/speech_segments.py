"""Approximate lexical coverage for edits during speech, not speaker turns.

Continuous background energy cannot fill a gap in recognized speech. Regions
remain candidates: the original-audio observer must classify every offered one.
"""

from copy import deepcopy

MAX_LEXICAL_BRIDGE_SECONDS = 0.25


def lexical_regions(words):
    groups = []
    for word in words:
        previous = groups[-1][-1] if groups else None
        if (
            previous is None
            or word["sequence_index"] != previous["sequence_index"] + 1
            or word["start_time"] - max(w["end_time"] for w in groups[-1])
            > MAX_LEXICAL_BRIDGE_SECONDS
        ):
            groups.append([])
        groups[-1].append(word)
    return [
        dict(
            segment_id="lexical_" + group[0]["word_id"] + "__" + group[-1]["word_id"],
            first_word_id=group[0]["word_id"],
            last_word_id=group[-1]["word_id"],
            text="".join(w["text"] for w in group),
            start_time=group[0]["start_time"],
            end_time=max(w["end_time"] for w in group),
            approximate=True,
            limitation="ASR coverage with at most 250 ms bridged between adjacent words; not silence or diarization.",
        )
        for group in groups
    ]


def uses_segments(q):
    return (
        q.get("reference_kind") == "speech"
        and q.get("focus") == "timing"
        and q.get("speech_scope") == "all_spoken_activity"
        and q.get("occurrence") == "each"
        and q.get("required_fields") == ["interval"]
        and "lexical_regions" in q.get("speech_references", {})
    )


def segment_schema(q):
    def obj(p):
        return dict(type="object", additionalProperties=False, properties=p, required=list(p))

    verdict = obj(
        dict(
            status=dict(type="string", enum=["match", "no_match", "unknown"]),
            evidence=dict(type="string", minLength=1),
        )
    )
    return obj(
        dict(
            status=dict(type="string", enum=["answered", "uncertain", "not_found"]),
            answer=dict(type="string", minLength=1),
            segment_assessments=obj(
                {
                    r["segment_id"]: deepcopy(verdict)
                    for r in q["speech_references"]["lexical_regions"]
                }
            ),
        )
    )


def compile_segment_fact(fact, q, data):
    catalog = q["speech_references"]
    regions = catalog["lexical_regions"]
    ids = {r["segment_id"] for r in regions}
    if not isinstance(fact, dict) or set(fact) != {"status", "answer", "segment_assessments"}:
        raise ValueError(
            "Each speech interval must classify the supplied lexical segments, without a blanket word range"
        )
    verdicts = fact["segment_assessments"]
    if not isinstance(verdicts, dict) or set(verdicts) != ids:
        raise ValueError("Every lexical segment needs an explicit speech association")
    if any(
        not isinstance(v, dict)
        or set(v) != {"status", "evidence"}
        or v["status"] not in ("match", "no_match", "unknown")
        or not isinstance(v["evidence"], str)
        or not v["evidence"].strip()
        for v in verdicts.values()
    ):
        raise ValueError(
            "Speech segment verdict requires match/no_match/unknown and an audible cue"
        )
    matched = [r for r in regions if verdicts[r["segment_id"]]["status"] == "match"]
    complete = (
        fact["status"] == "answered"
        and bool(matched)
        and all(v["status"] != "unknown" for v in verdicts.values())
    )
    data.setdefault("speech_segment_assessments", []).append(
        dict(
            question_id=q["id"],
            catalog_id=catalog["catalog_id"],
            assessments=deepcopy(verdicts),
            complete=complete,
        )
    )
    if fact["status"] == "answered" and not complete:
        raise ValueError("Uncertain or empty speech coverage cannot authorize an each-segment edit")
    events = (
        [
            dict(
                timing_kind="speech_reference",
                event=q["subject"],
                occurrence_index=i + 1,
                first_word_id=r["first_word_id"],
                last_word_id=r["last_word_id"],
                start_gap_id=None,
                end_gap_id=None,
                utterance_complete=False,
                association_status="match",
                association_evidence=verdicts[r["segment_id"]]["evidence"],
            )
            for i, r in enumerate(matched)
        ]
        if complete
        else []
    )
    return dict(status=fact["status"], answer=fact["answer"], events=events)


def segment_gaps(q, analysis):
    if not uses_segments(q):
        return []
    catalog = q["speech_references"]
    records = [
        r
        for r in (analysis or {}).get("speech_segment_assessments", [])
        if r.get("question_id") == q["id"] and r.get("catalog_id") == catalog["catalog_id"]
    ]
    if not records or not records[-1].get("complete"):
        return ["The each-speech scope needs an audible verdict for every offered lexical segment."]
    verdicts = records[-1].get("assessments", {})
    regions = catalog["lexical_regions"]
    if set(verdicts) != {r["segment_id"] for r in regions}:
        return ["Speech segment coverage belongs to an incomplete or different catalog."]
    expected = {
        (r["first_word_id"], r["last_word_id"])
        for r in regions
        if verdicts[r["segment_id"]].get("status") == "match"
    }
    events = [e for e in (analysis or {}).get("events", []) if q["id"] in e.get("question_ids", [])]
    actual = [
        (
            e.get("speech_reference_provenance", {}).get("first_word_id"),
            e.get("speech_reference_provenance", {}).get("last_word_id"),
        )
        for e in events
    ]
    if (
        not expected
        or len(actual) != len(set(actual))
        or set(actual) != expected
        or any(v.get("status") not in ("match", "no_match") for v in verdicts.values())
    ):
        return [
            "Executable speech intervals must exactly match the confirmed lexical segments and preserve their gaps."
        ]
    return []
