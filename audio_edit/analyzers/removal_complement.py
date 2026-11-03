"""Whole-music removal may keep a uniquely observed non-musical source via SAM."""

from copy import deepcopy
import json

PROMPT = """Use the EXISTING SAM Audio separator for the original whole-file music
removal. First_source_facts describe ONLY the original recording. A direct
extraction can preserve one clearly observed non-musical class while excluding
the complete music layer (including any singing that belongs to that music).
Use it only when the original request removes the complete music class and the
facts positively identify the music and exactly one remaining class that must
be kept, with no other protected sound or uncertainty about the inventory.
Speaking/shouting voices distinct from a song may be a retained class. Do not
invent a speaker identity. Do not remove nonmusical impacts, ambient sound, or
other sources merely because a voice is prominent. A particular instrument,
selected event, partial scope, or selected participant is ineligible. A whole
time range alone never grants permission to remove all sounds.
Return use_complement boolean, retained_source short positive audible phrase
(at most 24 words), instruction_quote exact words authorizing full music removal,
source_fact_quote exact words supporting that retained source, and reason.
When unsupported return false and null for all three source/quote fields.
No output, score, listening retries or alternate models are available.
"""


def candidate(ctx, plan):
    contract = ctx.params.get("edit_listening_dialogue", {})
    selection = contract.get("selection", {})
    return (
        plan.backend == "samaudio"
        and plan.mode == "text"
        and plan.result_mode == "residual"
        and not plan.use_video_prompt
        and not plan.placements
        and not plan.time_span
        and not plan.temporal_schema_error
        and plan.temporal_status == "not_requested"
        and (plan.temporal is None or plan.temporal.kind == "unspecified")
        and plan.text_prompt.strip().casefold()
        in {
            "music",
            "background music",
            "bgm",
            "instrumental music",
            "background instrumental music",
        }
        and selection.get("target") == "named_source"
        and selection.get("scope") == "whole_file"
        and selection.get("occurrence") == "unselected"
        and not contract.get("participant_selection")
    )


def _request(llm, request):
    props = dict(
        use_complement=dict(type="boolean"),
        retained_source=dict(type=["string", "null"]),
        instruction_quote=dict(type=["string", "null"]),
        source_fact_quote=dict(type=["string", "null"]),
        reason=dict(type="string"),
    )
    options = {}
    if getattr(llm, "backend", None) == "openai":
        options["response_format"] = dict(
            type="json_schema",
            json_schema=dict(
                name="remove_retained_source",
                strict=True,
                schema=dict(
                    type="object",
                    additionalProperties=False,
                    properties=props,
                    required=list(props),
                ),
            ),
        )
    response = llm.client.chat.completions.create(
        model=llm.model,
        messages=[
            dict(role="system", content=PROMPT),
            dict(role="user", content=json.dumps(request, ensure_ascii=False)),
        ],
        **options,
    )
    from audio_edit.analyzers.qwen_audio_analyzer import extract_json_object

    return extract_json_object(response.choices[0].message.content)


def decide(ctx, plan, *, llm=None):
    audit = dict(eligible=False, status="not_applicable", new_source_queries=0)
    if not candidate(ctx, plan):
        return audit
    from audio_edit.analyzers.removal_focus import file_sha256, observation_context
    from audio_edit.utils.media_util import get_media_duration

    facts = ctx.params.get("removal_source_facts")
    if facts is None:
        from audio_edit.analyzers.qwen_audio_analyzer import QwenAudioAnalyzer

        try:
            observation = QwenAudioAnalyzer().analyze_removal(
                ctx.input_audio,
                ctx.user_prompt,
                observation_context(plan, get_media_duration(ctx.input_audio)),
            )
            facts = dict(
                audio_sha256=file_sha256(ctx.input_audio),
                text=json.dumps(observation.get("removal_assessment", {})),
                first_observation=True,
                origin="first_original_source_observation",
            )
            ctx.params["removal_source_facts"] = facts
            audit["new_source_queries"] = 1
        except Exception as exc:
            return dict(
                audit,
                status="original_path_retained_source_analysis_unavailable",
                error=f"{type(exc).__name__}: {exc}",
            )
    if (
        not isinstance(facts, dict)
        or facts.get("audio_sha256") != file_sha256(ctx.input_audio)
        or facts.get("first_observation") is not True
        or not isinstance(facts.get("text"), str)
        or not facts["text"].strip()
    ):
        return dict(audit, status="original_path_retained_unbound_facts")
    request = dict(
        original_instruction=ctx.user_prompt,
        removed_source=plan.text_prompt,
        first_source_facts=facts["text"],
    )
    if llm is None:
        from audio_edit.llm.client import LLMClient

        llm = LLMClient()
    response = _request(llm, request)
    audit.update(
        request=request,
        response=deepcopy(response),
        source_facts=deepcopy(facts),
        first_decision=True,
        status="original_path_retained",
    )
    if not isinstance(response, dict):
        return audit
    target, quote, evidence = (
        response.get("retained_source"),
        response.get("instruction_quote"),
        response.get("source_fact_quote"),
    )
    if (
        response.get("use_complement") is True
        and isinstance(target, str)
        and 0 < len(target.split()) <= 24
        and isinstance(quote, str)
        and quote.strip()
        and quote in request["original_instruction"]
        and isinstance(evidence, str)
        and evidence.strip()
        and evidence in facts["text"]
    ):
        audit.update(
            eligible=True,
            status="observed_nonmusical_complement_selected",
            retained_source=target.strip(),
        )
    return audit
