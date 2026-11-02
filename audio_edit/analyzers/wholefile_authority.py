"""One bounded interpretation of explicit whole-file authority; no new audio facts."""

from copy import deepcopy
from dataclasses import asdict
import hashlib, json, math, os, re
from pathlib import Path
from audio_edit.plans.temporal import TemporalConstraint

PROTOCOL = "remove_explicit_wholefile_authority_v1"
PROMPT = """Interpret the ORIGINAL user instruction using the saved first original
audio observations. The existing removal target query is fixed. Do not decide
whether removal worked and do not rewrite any observation or sound description.
Decide only whether the user's explicit throughout-the-entire-recording authority
means separating this named source over the whole recording, while the planner's
extra timing question merely described observed occurrences. An observation of
one interval does not itself authorize limiting an explicitly global source edit.
Use whole-file scope only if the instruction positively requests that scope and
does not select a person, subset, ordinal, count, time window, or event relation.
Identity must already be answered. Do not broaden a source whose identity is
defined by a selected interval or relationship. Do not include other vocalizations
or infer that a target exists outside its observed intervals. Source identity and
all observed evidence stay unchanged; separation over the whole file remains
source-selective. If uncertain, contradictory, or bounded for a semantic reason,
return keep_whole_file=false. Do not invent extra events, times or sources.
Return keep_whole_file, instruction_quote (an exact contiguous quote explicitly
authorizing global scope), source_fact_quote (an exact contiguous quote supporting
the same existing target identity), and reason. When false use null quotes.
No audio output, score, new listening, retries, or alternative decisions exist.
"""


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def request_for(ctx, plan, analysis, duration):
    contract = (analysis or {}).get("edit_listening", {})
    dialogue = ctx.params.get("edit_listening_dialogue", {})
    selection = contract.get("selection", {})
    t = plan.temporal

    instruction = ctx.user_prompt
    if (
        not isinstance(instruction, str)
        or not re.search(r"\bthroughout\b.{0,24}\b(?:entire|whole)\b", instruction, re.I)
        or re.search(
            r"\b(?:first|last|second|third|fourth|nth|each|selected|before|after|during|between|until|when)\b|\d",
            instruction,
            re.I,
        )
    ):
        return None
    if (
        plan.backend != "samaudio"
        or plan.mode != "span"
        or plan.result_mode != "residual"
        or plan.use_video_prompt
        or plan.visual_prompt
        or plan.temporal_schema_error
        or plan.temporal_status not in ("resolved", "approximate")
        or t is None
        or t.kind != "event"
        or t.reference_source != "audio"
        or t.occurrence not in (None, "each", "all")
        or t.relation
        or t.end_reference_event
        or t.boundary
        or t.boundary_role
        or t.reference_text
        or t.start_time is not None
        or t.end_time is not None
        or t.offset_seconds != 0
        or t.count is not None
        or t.spacing_seconds is not None
        or t.window_seconds is not None
        or contract.get("status") != "ready"
        or contract.get("missing_evidence")
        or contract.get("instruction_source") != "original_user_instruction"
        or contract.get("participant_selection")
        or contract.get("placement_windows")
        or selection.get("scope") != "whole_file"
        or selection.get("occurrence") != "unselected"
        or selection.get("target") not in ("named_source", "ambiguous_source")
        or dialogue.get("status") != "ready"
        or dialogue.get("operation") != "remove_sound"
        or dialogue.get("original_instruction") != instruction
        or dialogue.get("selection") != selection
        or dialogue.get("participant_selection")
        or dialogue.get("placement_windows")
        or not _number(duration)
        or duration <= 0
    ):
        return None
    requirements = contract.get("requirements", [])
    if len(requirements) != 2 or requirements != dialogue.get("requirements"):
        return None
    identity = [
        q
        for q in requirements
        if q.get("focus") == "target" and set(q.get("required_fields", [])) == {"identity"}
    ]
    timing = [
        q
        for q in requirements
        if q.get("focus") == "timing" and set(q.get("required_fields", [])) == {"interval"}
    ]
    if len(identity) != 1 or len(timing) != 1:
        return None
    q1, q2 = identity[0], timing[0]
    if any(
        q.get("reference_kind") != "sound_event"
        or q.get("speech_scope")
        or q.get("occurrence") != "each"
        for q in requirements
    ):
        return None
    if (
        not isinstance(q1.get("id"), str)
        or not isinstance(q2.get("id"), str)
        or q1["id"] == q2["id"]
        or q1.get("subject", "").strip().casefold() != plan.text_prompt.strip().casefold()
        or q2.get("subject", "").strip().casefold() != plan.text_prompt.strip().casefold()
    ):
        return None
    usage = dialogue.get("usage", {})
    turns = [x for x in dialogue.get("turns", []) if x.get("action") == "qwen"]
    if (
        usage.get("qwen_listens") != 1
        or usage.get("tool_calls") != 0
        or usage.get("source_extractions") != 0
        or len(turns) != 1
        or turns[0].get("audio_source") != "original"
    ):
        return None
    original = turns[0].get("analysis", {})
    if original.get("events") != (analysis or {}).get("events") or original.get("answers") != (
        analysis or {}
    ).get("answers"):
        return None
    evidence = (analysis or {}).get("audio_evidence", {})
    try:
        if evidence != original.get("audio_evidence") or evidence.get("sha256") != _sha(
            ctx.input_audio
        ):
            return None
    except (OSError, TypeError):
        return None
    answers = {x.get("question_id"): x for x in (analysis or {}).get("answers", [])}
    if set(answers) != {q1["id"], q2["id"]} or any(
        x.get("status") != "answered"
        or not isinstance(x.get("answer"), str)
        or not x["answer"].strip()
        for x in answers.values()
    ):
        return None
    windows = [(p.start, p.end) for p in (plan.time_span or [])]
    if not windows or windows != [(p.start_time, p.end_time) for p in plan.placements]:
        return None
    if any(not _number(a) or not _number(b) or not 0 <= a < b <= duration for a, b in windows):
        return None
    covered = 0.0
    gaps = False
    for a, b in sorted(windows):
        gaps = gaps or a > covered + 1e-6
        covered = max(covered, b)
    if not gaps and covered >= duration - 1e-6:
        return None
    refs = t.reference_event if isinstance(t.reference_event, list) else [t.reference_event]
    if refs != [p.event_id for p in plan.placements]:
        return None
    events = {x.get("event_id"): x for x in (analysis or {}).get("events", [])}
    for p in plan.placements:
        e = events.get(p.event_id, {})
        if (
            set(e.get("question_ids", [])) != {q1["id"], q2["id"]}
            or e.get("timing_source") != "qwen_observation"
            or e.get("start_time") != p.start_time
            or e.get("end_time") != p.end_time
            or e.get("partial")
            or e.get("partial_start")
            or e.get("partial_end")
        ):
            return None
    return dict(
        original_instruction=instruction,
        current_source_query=plan.text_prompt,
        existing_selection=deepcopy(selection),
        first_identity_question=deepcopy(q1),
        first_identity_answer=answers[q1["id"]]["answer"],
        first_timing_question=deepcopy(q2),
        first_timing_answer=answers[q2["id"]]["answer"],
        current_event_windows=[dict(start=a, end=b) for a, b in windows],
        recording_duration=duration,
    )


def transport_for(model, payload, backend="openai"):
    props = dict(
        keep_whole_file=dict(type="boolean"),
        instruction_quote=dict(type=["string", "null"]),
        source_fact_quote=dict(type=["string", "null"]),
        reason=dict(type="string"),
    )
    request = dict(
        model=model,
        messages=[
            dict(role="system", content=PROMPT),
            dict(role="user", content=json.dumps(payload, ensure_ascii=False)),
        ],
    )
    if backend == "openai":
        request["response_format"] = dict(
            type="json_schema",
            json_schema=dict(
                name="remove_wholefile_authority",
                strict=True,
                schema=dict(
                    type="object",
                    additionalProperties=False,
                    properties=props,
                    required=list(props),
                ),
            ),
        )
    return request


def _valid(response, payload):
    if not isinstance(response, dict) or set(response) != {
        "keep_whole_file",
        "instruction_quote",
        "source_fact_quote",
        "reason",
    }:
        return False
    if (
        response.get("keep_whole_file") is not True
        or not isinstance(response.get("reason"), str)
        or not response["reason"].strip()
    ):
        return False
    quote = response.get("instruction_quote")
    fact = response.get("source_fact_quote")
    return (
        isinstance(quote, str)
        and quote.strip()
        and quote in payload["original_instruction"]
        and bool(re.search(r"\bthroughout\b.{0,24}\b(?:entire|whole)\b", quote, re.I))
        and isinstance(fact, str)
        and fact.strip()
        and fact in payload["first_identity_answer"]
    )


def bind(ctx, plan, analysis, duration, *, llm=None):
    payload = request_for(ctx, plan, analysis, duration)
    if payload is None:
        return plan
    if llm is None:
        from audio_edit.llm.client import LLMClient

        llm = LLMClient()
    request = transport_for(llm.model, payload, getattr(llm, "backend", None))
    identity = dict(
        protocol=PROTOCOL, audio_sha256=_sha(ctx.input_audio), request=request, plan=asdict(plan)
    )
    key = hashlib.sha256(_canonical(identity).encode()).hexdigest()
    base = Path(
        os.environ.get(
            "AUDIO_EDIT_EVIDENCE_DIR", str(Path(ctx.input_audio).parent / ".audio_edit_evidence")
        )
    )
    folder = base / "wholefile_authority" / key
    folder.mkdir(parents=True, exist_ok=True)
    response_path = folder / "first_decision.json"
    claim = folder / "first_request_claim.json"
    audit = dict(
        eligible=False,
        policy=PROTOCOL,
        request=payload,
        cache_identity=identity,
        first_decision_path=str(response_path),
        new_text_calls=0,
        status="original_path_retained",
    )
    ctx.params["wholefile_authority_decision"] = audit
    try:
        if response_path.exists():
            saved = json.loads(response_path.read_text())
            if saved.get("identity") != identity or saved.get("first_decision") is not True:
                return plan
            response = saved["response"]
            audit["cache_reused"] = True
        else:
            with claim.open("x") as f:
                json.dump(dict(identity=identity, first_only=True), f, ensure_ascii=False, indent=2)
            audit["new_text_calls"] = 1
            result = llm.client.chat.completions.create(**request)
            raw = result.model_dump() if hasattr(result, "model_dump") else None
            text = result.choices[0].message.content
            with (folder / "first_transport.json").open("x") as f:
                json.dump(
                    dict(
                        identity=identity,
                        raw_response=raw,
                        content=text,
                        finish_reason=result.choices[0].finish_reason,
                    ),
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            if result.choices[0].finish_reason != "stop":
                return plan
            from audio_edit.analyzers.qwen_audio_analyzer import extract_json_object

            response = extract_json_object(text)
            with response_path.open("x") as f:
                json.dump(
                    dict(
                        identity=identity, first_decision=True, response=response, raw_response=raw
                    ),
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        audit["response"] = deepcopy(response)
        if not _valid(response, payload):
            return plan
    except Exception as exc:
        audit.update(
            status="original_path_retained_first_decision_unavailable",
            error=f"{type(exc).__name__}: {exc}",
        )
        return plan
    revised = deepcopy(plan)
    revised.mode = "text"
    revised.time_span = None
    revised.placements = []
    revised.temporal = TemporalConstraint(kind="unspecified", reference_source="audio")
    revised.temporal_status = "not_requested"
    revised.temporal_reason = None
    audit.update(
        eligible=True,
        status="explicit_wholefile_authority_bound",
        original_plan=asdict(plan),
        execution_scope="whole_file",
        changed_fields=[
            "mode",
            "time_span",
            "placements",
            "temporal",
            "temporal_status",
            "temporal_reason",
        ],
        source_query_unchanged=True,
        first_observations_and_validators_unchanged=True,
    )
    return revised
