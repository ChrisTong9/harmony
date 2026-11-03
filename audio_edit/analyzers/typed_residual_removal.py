"""Complete an already selected music subtype only after a bound first residual fact.

The original multi-source plan is kept. A structural gate precedes one neutral
first-residual inventory. Existing facts and durable claims prevent repeat calls.
"""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import re
from pathlib import Path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def music_quote(text):
    for paragraph in str(text).split("\n\n"):
        value = paragraph.strip()
        if re.match(
            r"^(?:A|The) (?:background )?(?:musical score|music) (?:plays|is present|continues)\b",
            value,
            re.I,
        ):
            if not re.search(
                r"\b(?:not|no|uncertain|possibly|perhaps|might|could|if|absent)\b", value, re.I
            ):
                return value
    return None


def structural_candidate(ctx, plan):
    no = {"eligible": False}
    if ctx.params.get("typed_residual_completion"):
        return no
    try:
        dialogue = ctx.params["edit_listening_dialogue"]
        if (
            dialogue.get("status") != "ready"
            or dialogue.get("operation") != "remove_sound"
            or dialogue.get("original_instruction") != ctx.user_prompt
            or dialogue.get("participant_selection")
            or dialogue.get("selection", {}).get("scope") != "whole_file"
            or dialogue.get("selection", {}).get("occurrence") != "unselected"
        ):
            return no
        turns = dialogue.get("turns", [])
        if (
            len(turns) != 1
            or turns[0].get("action") != "qwen"
            or turns[0].get("audio_source") != "original"
        ):
            return no
        turn = turns[0]
        first = turn["analysis"]
        current = ctx.params["removal_perception"]["analysis"]
        if turn.get("review", {}).get("status") != "ready" or first.get("uncertainties"):
            return no
        for key in ("events", "answers", "audio_evidence"):
            if first.get(key) != current.get(key):
                return no
        original_sha = _sha(ctx.input_audio)
        if first["audio_evidence"].get("sha256") != original_sha:
            return no
        if (
            plan.mode != "span"
            or plan.result_mode != "residual"
            or plan.use_video_prompt
            or plan.temporal_status not in ("resolved", "approximate")
            or not plan.temporal
        ):
            return no
        temporal = asdict(plan.temporal)
        if (
            temporal["kind"] != "event"
            or temporal.get("relation")
            or temporal.get("count") is not None
            or temporal.get("offset_seconds", 0) != 0
            or temporal.get("occurrence") != "each"
            or temporal.get("reference_source") != "audio"
            or temporal.get("end_reference_event")
        ):
            return no
        placements = [asdict(x) for x in plan.placements]
        refs = temporal.get("reference_event")
        if (
            not isinstance(refs, list)
            or len(refs) < 2
            or [x.get("event_id") for x in placements] != refs
        ):
            return no
        if [(x.start, x.end) for x in plan.time_span or []] != [
            (x["start_time"], x["end_time"]) for x in placements
        ]:
            return no
        events = {x["event_id"]: x for x in first["events"]}
        if len(events) != len(first["events"]) or any(x not in events for x in refs):
            return no
        labels = [events[x].get("event", "").strip().lower() for x in refs]
        if len(set(labels)) < 2 or plan.text_prompt.strip().lower() in labels:
            return no
        questions = [
            q
            for q in turn["questions"]
            if q.get("focus") == "target"
            and q.get("reference_kind") == "sound_event"
            and q.get("subject", "").strip().lower() == plan.text_prompt.strip().lower()
            and "identity" in q.get("required_fields", [])
            and q.get("occurrence") == "each"
        ]
        if len(questions) != 1:
            return no
        qid = questions[0]["id"]
        assessment = turn["review"]["evidence_assessment"].get(qid, {})
        answers = [
            a
            for a in first["answers"]
            if a.get("question_id") == qid and a.get("status") == "answered"
        ]
        cue = assessment.get("cue_quote")
        if (
            len(answers) != 1
            or assessment.get("problem")
            or not cue
            or cue not in answers[0]["answer"]
        ):
            return no
        for placement in placements:
            event = events[placement["event_id"]]
            if (
                qid not in event.get("question_ids", [])
                or event.get("timing_source") != "qwen_observation"
                or event.get("start_time") != placement["start_time"]
                or event.get("end_time") != placement["end_time"]
                or event.get("event") != placement.get("reference_event")
            ):
                return no

        selected = [
            p
            for p in placements
            if events[p["event_id"]]["event"].strip().lower() in {"music", "background music"}
        ]
        if len(selected) != 1:
            return no
        selected = selected[0]
        return dict(
            eligible=True,
            event_id=selected["event_id"],
            query=events[selected["event_id"]]["event"],
            original_query=plan.text_prompt,
            placement=selected,
            original_audio_sha256=original_sha,
            first_original_identity_answer=answers[0]["answer"],
        )
    except (KeyError, TypeError, ValueError, OSError):
        return no


def decide(ctx, plan):
    no = {"eligible": False}
    binding = ctx.params.get("typed_first_residual_inventory")
    candidate = structural_candidate(ctx, plan)
    if not isinstance(binding, dict) or not candidate.get("eligible"):
        return no
    try:
        if binding.get("original_audio_sha256") != candidate["original_audio_sha256"]:
            return no
        observation_path = binding["observation_path"]
        if _sha(observation_path) != binding["observation_sha256"]:
            return no
        observation = json.loads(Path(observation_path).read_text())
        audio = observation["input"]
        if (
            not observation.get("first_observation")
            or audio.get("role") != "residual"
            or Path(binding["residual_audio_path"]).resolve() != Path(ctx.output_audio).resolve()
        ):
            return no
        residual_sha = _sha(ctx.output_audio)
        if residual_sha != audio["sha256"] or residual_sha != binding["residual_audio_sha256"]:
            return no
        if Path(audio["path"]).is_file() and _sha(audio["path"]) != residual_sha:
            return no
        quote = music_quote(observation.get("response"))
        if not quote:
            return no
        return dict(
            **candidate,
            policy="first_selected_music_remaining_in_generic_residual",
            first_residual_audio=str(Path(ctx.output_audio).resolve()),
            first_residual_audio_sha256=residual_sha,
            observation_path=observation_path,
            observation_sha256=binding["observation_sha256"],
            first_residual_quote=quote,
            complete_environment_removal_claim=False,
        )
    except (KeyError, TypeError, ValueError, OSError):
        return no


def acquire_first_inventory(ctx, plan):
    """Attach one persistent first neutral fact after the original structural gate.

    The cache is keyed by residual bytes, not sample identity or score. A prior
    response (including a negative one) is reused; an unanswered claim is never
    reissued. The configured local Qwen analyzer supplies the identical neutral
    inventory protocol used for other source facts. Failure preserves the first
    generic removal rather than discarding that output.
    """
    if "typed_first_residual_inventory" in ctx.params:
        return
    candidate = structural_candidate(ctx, plan)
    if not candidate.get("eligible"):
        return
    import os
    from urllib.parse import urlparse

    audit = dict(status="not_requested", new_residual_queries=0)
    try:
        residual = Path(ctx.output_audio).resolve()
        digest = _sha(residual)
        root = (
            Path(os.environ.get("AUDIO_EDIT_EVIDENCE_DIR", "audio_edit/evidence"))
            / "first_residual_sources"
        )
        root = root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        path = root / (digest + ".json")
        claim = root / (digest + ".claim.json")
        if path.exists():
            observation = json.loads(path.read_text())
            audit["status"] = "reused_first_residual_fact"
        elif claim.exists():
            audit["status"] = "unanswered_first_claim_no_retry"
            return
        else:
            from audio_edit.analyzers.qwen_audio_analyzer import QwenAudioAnalyzer

            analyzer = QwenAudioAnalyzer()
            endpoint = urlparse(str(analyzer.client.base_url))
            hostname = (endpoint.hostname or "").lower()
            local = hostname in {"localhost", "127.0.0.1", "::1"} or hostname.split(".")[
                0
            ].startswith("gpu-")
            if (
                analyzer.backend != "openai_compatible"
                or endpoint.scheme not in {"http", "https"}
                or not local
            ):
                audit["status"] = "local_residual_analyzer_unavailable"
                return
            analyzer.client = analyzer.client.with_options(max_retries=0)
            with claim.open("x") as stream:
                json.dump(
                    dict(
                        input_sha256=digest,
                        input_path=str(residual),
                        protocol="neutral_sources_v1",
                        model=analyzer.model,
                        endpoint=str(analyzer.client.base_url),
                    ),
                    stream,
                    indent=2,
                )
            audit["new_residual_queries"] = 1
            text = analyzer.describe_sources(str(residual))
            if not isinstance(text, str) or not text.strip():
                raise ValueError("No first residual description")
            observation = dict(
                input=dict(path=str(residual), sha256=digest, role="residual"),
                response=text,
                first_observation=True,
                sampling_temperature=0,
                protocol="neutral_sources_v1",
            )
            with path.open("x") as stream:
                json.dump(observation, stream, indent=2)
            audit["status"] = "saved_first_residual_fact"
        audio = observation.get("input", {})
        if (
            observation.get("first_observation") is not True
            or audio.get("role") != "residual"
            or audio.get("sha256") != digest
            or not isinstance(observation.get("response"), str)
            or not observation["response"].strip()
        ):
            audit["status"] = "invalid_cached_first_fact_no_retry"
            return
        if Path(audio["path"]).is_file() and _sha(audio["path"]) != digest:
            audit["status"] = "first_fact_source_changed_no_retry"
            return
        ctx.params["typed_first_residual_inventory"] = dict(
            observation_path=str(path),
            observation_sha256=_sha(path),
            original_audio_sha256=candidate["original_audio_sha256"],
            residual_audio_path=str(residual),
            residual_audio_sha256=digest,
        )
        audit.update(
            observation_path=str(path), observation_sha256=_sha(path), residual_sha256=digest
        )
    except Exception as exc:
        audit.update(
            status="first_residual_fact_unavailable_no_retry", error=f"{type(exc).__name__}: {exc}"
        )
    finally:
        ctx.params["typed_residual_first_inventory"] = audit


def apply(ctx, plan):
    acquire_first_inventory(ctx, plan)
    decision = decide(ctx, plan)
    if not decision.get("eligible"):
        return ctx
    from audio_edit.runners.samaudio_runner import SAMAudioRunner
    from audio_edit.plans.remove_sound_plan import TimeSpan
    from audio_edit.plans.temporal import ResolvedPlacement

    follow = deepcopy(plan)
    p = decision["placement"]
    follow.text_prompt = decision["query"]
    follow.time_span = [TimeSpan(p["start_time"], p["end_time"])]
    follow.placements = [ResolvedPlacement(**p)]
    follow.temporal.reference_event = [decision["event_id"]]

    follow.removal_evidence = None
    continuation = deepcopy(ctx)
    continuation.input_audio = ctx.output_audio
    continuation.output_audio = None
    continuation.params = {}
    result = SAMAudioRunner().run_remove(continuation, follow)
    ctx.output_audio = result.output_audio
    ctx.warnings.extend(result.warnings)
    ctx.params["typed_residual_completion"] = {
        **decision,
        "derived_plan": asdict(follow),
        "separation_calls": result.params.get("separation_calls", []),
        "output_audio": ctx.output_audio,
        "output_sha256": _sha(ctx.output_audio),
        "new_SAM_calls": 1,
    }
    return ctx
