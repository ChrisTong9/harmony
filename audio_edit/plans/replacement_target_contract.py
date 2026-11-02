"""One field-only repair for an overlong independent replacement source phrase."""

from copy import deepcopy
import json
import re

PROMPT = """Repair only an overlong replacement_material.target field so the
already chosen independent replacement can satisfy its existing 24-word contract.
The original instruction is authoritative. The proposed target is an unexecuted
source description, not acoustic evidence and not a new instruction. Return a
concise positive noun phrase naming the SAME new source and its essential acoustic
attributes, at most 24 words. Preserve every attribute explicitly requested by
the original instruction. Do not invent another source, cause, style or identity.
Omit redundant prose and planner-invented occurrence counts, timestamps and mixing
directions: the existing frozen window and material mode already own extent.
Never include editing commands, exclusions, the old source, or directions to keep
or imitate the original waveform. Do not change any other plan field. If a faithful
short target is impossible or its intended source is ambiguous, supported=false
and repairs=[]. Each repair must cite an exact nonempty original-instruction quote
supporting that new source. One first repair is allowed; no alternative is sampled.
Return supported, repairs (edit_index, target, instruction_quote), and reason.
"""


def candidates(data):
    out = []
    for index, edit in enumerate(data.get("edits", [])):
        material = edit.get("replacement_material") or {}
        if (
            edit.get("generator") == "audio"
            and material.get("mode") in {"independent_event", "independent_texture"}
            and isinstance(material.get("target"), str)
            and len(material["target"].split()) > 24
        ):
            out.append(
                dict(edit_index=index, mode=material["mode"], original_target=material["target"])
            )
    return out


def complete(ctx, data, payload, llm):
    fields = candidates(data)
    if not fields:
        return data

    if not payload.get("shared_replacement_windows") or len(fields) > 6:
        return data
    if ctx.params.get("replacement_target_contract_repair"):
        raise ValueError("Replacement target contract already attempted")
    request = dict(original_instruction=payload["user_prompt"], overlong_target_fields=fields)
    props = dict(
        supported=dict(type="boolean"),
        repairs=dict(
            type="array",
            maxItems=6,
            items=dict(
                type="object",
                additionalProperties=False,
                properties=dict(
                    edit_index=dict(type="integer"),
                    target=dict(type="string"),
                    instruction_quote=dict(type="string"),
                ),
                required=["edit_index", "target", "instruction_quote"],
            ),
        ),
        reason=dict(type="string"),
    )
    options = {}
    if getattr(llm, "backend", None) == "openai":
        options["response_format"] = dict(
            type="json_schema",
            json_schema=dict(
                name="replacement_target_contract",
                strict=True,
                schema=dict(
                    type="object",
                    additionalProperties=False,
                    properties=props,
                    required=list(props),
                ),
            ),
        )
    audit = dict(
        status="first_attempt_claimed",
        request=deepcopy(request),
        first_only=True,
        changed_fields=[],
    )
    ctx.params["replacement_target_contract_repair"] = audit
    client = llm.client.with_options(max_retries=0)
    result = client.chat.completions.create(
        model=llm.model,
        messages=[
            dict(role="system", content=PROMPT),
            dict(role="user", content=json.dumps(request, ensure_ascii=False)),
        ],
        **options,
    )
    raw = result.choices[0].message.content
    audit["raw_response"] = raw
    from audio_edit.analyzers.qwen_audio_analyzer import extract_json_object

    answer = extract_json_object(raw)
    audit["response"] = deepcopy(answer)
    audit["status"] = "first_repair_unsupported_original_plan_retained"
    if set(answer) != {"supported", "repairs", "reason"} or answer["supported"] is not True:
        return data
    repairs = answer["repairs"]
    if not isinstance(repairs, list) or len(repairs) != len(fields):
        return data
    expected = {f["edit_index"] for f in fields}
    seen = set()
    for item in repairs:
        if not isinstance(item, dict) or set(item) != {"edit_index", "target", "instruction_quote"}:
            return data
        index = item["edit_index"]
        target = item["target"]
        quote = item["instruction_quote"]
        if type(index) is not int or index not in expected or index in seen:
            return data
        if not isinstance(target, str) or not 0 < len(target.split()) <= 24:
            return data
        if re.search(
            r"\b(remove|replace|reduce|increase|keep|preserve|mute|without|except|seconds?|minutes?)\b|[\n\r]",
            target,
            re.IGNORECASE,
        ):
            return data
        if not isinstance(quote, str) or not quote.strip() or quote not in payload["user_prompt"]:
            return data
        seen.add(index)
    out = deepcopy(data)
    for item in repairs:
        index = item["edit_index"]
        out["edits"][index]["replacement_material"]["target"] = item["target"].strip()
        audit["changed_fields"].append(f"edits[{index}].replacement_material.target")
    audit["status"] = "first_target_field_repaired"
    return out
