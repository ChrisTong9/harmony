"""Ground a complementary SAM source query in first original-audio facts."""

from copy import deepcopy
import json
import re


def source_quote_bound(quote, source):
    """Bind a literal excerpt or ordered, complete verbatim source sentences."""
    if not isinstance(quote, str) or not quote.strip():
        return False
    if quote in source:
        return True
    excerpts = re.split(r"(?<=[.!?])\s+", quote.strip())
    sentences = re.split(r"(?<=[.!?])\s+", source.strip())
    if len(excerpts) < 2 or any(not s.endswith((".", "!", "?")) for s in excerpts):
        return False
    cursor = 0
    for excerpt in excerpts:
        try:
            cursor = sentences.index(excerpt, cursor) + 1
        except ValueError:
            return False
    return True


PROMPT = """Plan how the EXISTING SAM Audio separator can retain the correct
residual for a whole-recording music replacement. Use original_instruction and
first_source_facts, which describe the original recording, not an edited output.
Use a complementary query ONLY if the user replaces the complete instrumental
backing and the supplied facts identify a single audible source class that must
remain (for example singing vocals). SAM can extract that retained source directly
instead of subtracting an aggregate of several instruments. This changes only the
separation query and which SAM output is kept; it does not change the requested edit.
The facts must support the retained source. Do not infer that a requested new
sound is already present. Do not invent source identities or timing. A whole-file
time range does not authorize changing all sources. If the request selects only
one instrument, transforms vocals/the whole recording, or additional protected
sources such as crowd or environmental noise are heard/requested, retain the old
separation strategy. Likewise retain it if the preserved-source inventory is unclear.
Return JSON: use_complement (boolean), retained_source (short positive audible noun
phrase, max 24 words; no edit commands), instruction_quote (exact words establishing
complete backing replacement), source_fact_quote (exact words supporting the retained
source), reason. Use null for the three string fields when use_complement=false.
Do not alter generated material, reference audio, windows, duration, levels or seed.
"""

EFFECT_PROMPT = """Use the EXISTING SAM Audio separator to retain the correct
residual of a whole-recording named sound-class replacement. The existing target
material is fixed. The first_source_facts describe the original audio only.
Select a complementary extraction ONLY when the facts positively identify the
requested old source AND exactly one other audible source class that must remain,
with no additional protected sources or uncertainty about that inventory. For
example, a clearly observed speaking voice can remain while a continuous engine
source is replaced. Do not choose the requested new effect as a retained source.
The original instruction must replace the complete named class within this scope;
a whole-file window alone does not authorize removal of other sounds. Do not use
this strategy for selected occurrences, partial changes, transformations of the
preserved voice itself, or an unclear/more complex source inventory. Return false
when those conditions are not met. Do not invent an identity or a time interval.
Return JSON: use_complement (boolean), retained_source (short positive audible
noun phrase, at most 24 words), instruction_quote (exact substring establishing
the named-source replacement), source_fact_quote (exact substring supporting the
retained source), reason. The three string fields are null when false. Change no
target material, reference, duration, windows, level, seed or other plan field.
"""


def backing_request(ctx, plan, duration):
    removal = plan.remove_plan
    if (
        removal.mode != "text"
        or removal.result_mode != "residual"
        or removal.use_video_prompt
        or removal.temporal_status != "resolved"
        or removal.temporal_schema_error
        or (removal.temporal is not None and removal.temporal.kind != "unspecified")
    ):
        return None
    edits = (plan.add_plan.intent or {}).get("edits", [])
    if len(edits) != 1:
        return None
    edit = edits[0]
    independent = (
        edit.get("audio_reference") is None
        and (edit.get("replacement_material") or {}).get("mode") == "independent_texture"
    )
    if (
        edit.get("generator") != "audio"
        or not (edit.get("audio_reference") == "structure" or independent)
        or edit.get("timeline_op") != "overlay"
        or edit.get("placement", {}).get("kind") != "cover"
    ):
        return None
    window = edit.get("shared_window", {})
    if (
        window.get("start_time") != 0
        or abs(window.get("end_time", -1) - duration) > 1e-6
        or any(
            window.get(k) is not None
            for k in ["reference_event", "event_id", "occurrence_index", "reference_source"]
        )
    ):
        return None
    if (
        len(removal.placements) != 1
        or removal.placements[0].start_time != 0
        or abs(removal.placements[0].end_time - duration) > 1e-6
    ):
        return None
    request = dict(
        original_instruction=ctx.user_prompt,
        proposed_source=plan.source_sound,
        new_source=plan.target_sound,
        window=dict(start=0, end=duration),
    )
    if independent:
        request["replacement_material"] = "independent_texture"
    return request


def decide_backing(ctx, plan, duration, *, llm=None):
    request = backing_request(ctx, plan, duration)
    audit = dict(eligible=False, status="not_applicable", request=request, new_source_queries=0)
    if request is None:
        return audit
    from audio_edit.analyzers.removal_focus import file_sha256, observation_context

    facts = ctx.params.get("replacement_source_facts")
    if facts is None:
        from audio_edit.analyzers.qwen_audio_analyzer import QwenAudioAnalyzer

        try:
            analysis = QwenAudioAnalyzer().analyze_removal(
                ctx.input_audio, ctx.user_prompt, observation_context(plan.remove_plan, duration)
            )
            facts = dict(
                audio_sha256=file_sha256(ctx.input_audio),
                text=json.dumps(analysis.get("removal_assessment", {})),
                first_observation=True,
                origin="first_directed_source_observation",
            )
            audit["new_source_queries"] = 1
            ctx.params["replacement_source_facts"] = facts
        except Exception as exc:
            return dict(
                audit,
                status="source_analysis_unavailable_original_path_retained",
                error=f"{type(exc).__name__}: {exc}",
            )
    if (
        not isinstance(facts, dict)
        or facts.get("audio_sha256") != file_sha256(ctx.input_audio)
        or facts.get("first_observation") is not True
        or not isinstance(facts.get("text"), str)
        or not facts["text"].strip()
    ):
        return dict(audit, status="unbound_source_facts_original_path_retained")
    request["first_source_facts"] = facts["text"]
    audit["source_facts"] = deepcopy(facts)
    if llm is None:
        from audio_edit.llm.client import LLMClient

        llm = LLMClient()
    props = dict(
        use_complement=dict(type="boolean"),
        retained_source=dict(type=["string", "null"]),
        instruction_quote=dict(type=["string", "null"]),
        source_fact_quote=dict(type=["string", "null"]),
        reason=dict(type="string"),
    )
    kwargs = dict(
        model=llm.model,
        messages=[
            dict(
                role="system",
                content=EFFECT_PROMPT
                if request.get("replacement_material") == "independent_texture"
                else PROMPT,
            ),
            dict(role="user", content=json.dumps(request, ensure_ascii=False)),
        ],
    )
    if getattr(llm, "backend", None) == "openai":
        kwargs["response_format"] = dict(
            type="json_schema",
            json_schema=dict(
                name="sam_retained_source",
                strict=True,
                schema=dict(
                    type="object",
                    additionalProperties=False,
                    properties=props,
                    required=list(props),
                ),
            ),
        )
    response = llm.client.chat.completions.create(**kwargs)
    from audio_edit.analyzers.qwen_audio_analyzer import extract_json_object

    raw = response.choices[0].message.content
    answer = extract_json_object(raw)
    audit.update(
        status="original_path_retained",
        raw_response=raw,
        response=deepcopy(answer),
        first_decision=True,
    )
    if (
        not isinstance(answer, dict)
        or set(answer) != set(props)
        or not isinstance(answer.get("reason"), str)
    ):
        return audit
    target, quote, evidence = (
        answer.get("retained_source"),
        answer.get("instruction_quote"),
        answer.get("source_fact_quote"),
    )
    if (
        answer.get("use_complement") is True
        and isinstance(target, str)
        and 0 < len(target.split()) <= 24
        and isinstance(quote, str)
        and quote.strip()
        and quote in request["original_instruction"]
        and source_quote_bound(evidence, request["first_source_facts"])
    ):
        audit.update(
            eligible=True, status="observed_complement_selected", retained_source=target.strip()
        )
    return audit
