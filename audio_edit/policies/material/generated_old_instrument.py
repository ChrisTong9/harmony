"""Remove a positively observed old instrument from new replacement material."""

from copy import deepcopy
from pathlib import Path
import hashlib
import json
import re
import subprocess
from tempfile import mkdtemp

PROMPT = """Review one FIRST generated replacement asset. This is a new isolated asset, not the original or retained residual. The validated replacement semantics identify one old instrument explicitly replaced by one new instrument from a different instrument family. Decide whether the first material facts positively identify BOTH of those exact instruments in the generated asset. Only then may one existing separator remove the old instrument from that asset. Use the supplied old_identity.name and new_identity.name verbatim; do not substitute a different instrument or infer one from generic music, a synthesized chord, similarity, absence, negation, or uncertainty. Do not reinterpret an ensemble or voice as an isolated instrument. Do not alter preserved original voices or instruments. Return eligible:boolean, old_instrument:string or null, new_instrument:string or null, old_quote:exact continuous substring of first material facts or null, new_quote:exact continuous substring or null, reason:string. Each quote must include the corresponding instrument name and affirmative evidence that it is present, retaining any surrounding qualifiers. When false both instruments and quotes must be null. No generation, level, timing, seed, or original-source query change. Return false when either identity lacks positive evidence. The sole permitted cleanup query is the supplied old_identity.name."""


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


_UNSUPPORTED_EVIDENCE = re.compile(
    r"\b(?:no|not|without|absent|absence|neither|nor|never|unclear|uncertain|unsure|"
    r"possibly|possible|perhaps|maybe|might|may|could|cannot|can't|doesn't|isn't|"
    r"seems?|appears?|suggests?|resembl\w*|similar|imitat\w*)\b|"
    r"\b(?:sounds?\s+like|hard\s+to\s+(?:hear|tell|identify)|rather\s+than)\b|-like\b",
    re.I,
)


def _instrument_pair(ctx, plan, placement):
    from audio_edit.plans.replacement_semantics import get_replacement_semantics

    if not placement.get("shared_window") or placement.get("generator") != "audio":
        return None
    if plan.is_speech:
        return None
    if any(
        ctx.params.get(key, {}).get("status") == "rendered"
        for key in ["replacement_music_render", "replacement_independent_piano"]
    ):
        return None
    semantics = get_replacement_semantics(ctx.params.get("replacement_plan"))
    if semantics is None or semantics["source_scope"] != "selected_source":
        return None
    source, target = semantics["source"], semantics["target"]
    if source["kind"] != "instrument" or target["kind"] != "instrument":
        return None
    if source["family"].casefold() == target["family"].casefold():
        return None
    if source["name"].casefold() == target["name"].casefold():
        return None
    if any(
        _UNSUPPORTED_EVIDENCE.search(identity[field])
        for identity in (source, target)
        for field in ("name", "family", "quote")
    ):
        return None
    return source, target


def applicable(ctx, plan, placement):
    return _instrument_pair(ctx, plan, placement) is not None


def _positive_quote(quote, facts, identity):
    """Bind the instrument name to an exact, non-negated first-fact excerpt."""
    if not isinstance(quote, str) or not quote.strip() or quote not in facts:
        return False
    name = re.escape(identity).replace(r"\ ", r"\s+")
    if not re.search(r"(?<![\w-])" + name + r"(?![\w-])", quote, re.I):
        return False
    if _UNSUPPORTED_EVIDENCE.search(quote):
        return False
    # A short substring cannot discard a qualifier in its containing sentence.
    for match in re.finditer(re.escape(quote), facts):
        start = max(facts.rfind(mark, 0, match.start()) for mark in ".!?;\n") + 1
        ends = [facts.find(mark, match.end()) for mark in ".!?;\n"]
        end = min((index for index in ends if index >= 0), default=len(facts))
        if not _UNSUPPORTED_EVIDENCE.search(facts[start:end]):
            return True
    return False


def decide(ctx, plan, placement, generated, *, llm=None):
    audit = dict(eligible=False, status="not_applicable", new_material_observations=0)
    instruments = _instrument_pair(ctx, plan, placement)
    if instruments is None:
        return audit
    old_identity, new_identity = instruments
    digest = sha(generated)
    facts = ctx.params.get("replacement_material_first_facts", {}).get(digest)
    if facts is None:
        from audio_edit.analyzers.qwen_audio_analyzer import QwenAudioAnalyzer

        try:
            text = QwenAudioAnalyzer().describe_sources(generated)
        except Exception as exc:
            return dict(
                audit,
                status="first_material_facts_unavailable",
                error=f"{type(exc).__name__}: {exc}",
            )
        facts = dict(audio_sha256=digest, text=text, first_observation=True)
        ctx.params.setdefault("replacement_material_first_facts", {})[digest] = deepcopy(facts)
        audit["new_material_observations"] = 1
    if (
        not isinstance(facts, dict)
        or facts.get("audio_sha256") != digest
        or facts.get("first_observation") is not True
        or not isinstance(facts.get("text"), str)
    ):
        return dict(audit, status="unbound_first_material_facts")
    replacement = ctx.params["replacement_plan"]
    payload = dict(
        original_instruction=replacement["user_prompt"],
        old_source=replacement["source_sound"],
        new_source=replacement["target_sound"],
        old_identity=deepcopy(old_identity),
        new_identity=deepcopy(new_identity),
        first_material_facts=facts["text"],
    )
    if llm is None:
        from audio_edit.llm.client import LLMClient

        llm = LLMClient()
    props = dict(
        eligible=dict(type="boolean"),
        old_instrument=dict(type=["string", "null"]),
        new_instrument=dict(type=["string", "null"]),
        old_quote=dict(type=["string", "null"]),
        new_quote=dict(type=["string", "null"]),
        reason=dict(type="string"),
    )
    kwargs = dict(
        model=llm.model,
        messages=[
            dict(role="system", content=PROMPT),
            dict(role="user", content=json.dumps(payload, ensure_ascii=False)),
        ],
    )
    if getattr(llm, "backend", None) == "openai":
        kwargs["response_format"] = dict(
            type="json_schema",
            json_schema=dict(
                name="old_instrument_in_new_material",
                strict=True,
                schema=dict(
                    type="object",
                    additionalProperties=False,
                    properties=props,
                    required=list(props),
                ),
            ),
        )
    from audio_edit.analyzers.qwen_audio_analyzer import extract_json_object

    try:
        response = llm.client.chat.completions.create(**kwargs)
        answer = extract_json_object(response.choices[0].message.content)
    except Exception as exc:
        return dict(
            audit,
            status="first_material_decision_unavailable",
            error=f"{type(exc).__name__}: {exc}",
        )
    audit.update(
        status="first_material_retained",
        material_sha256=digest,
        request=payload,
        response=deepcopy(answer),
        first_decision=True,
    )
    if isinstance(answer, dict) and set(answer) == set(props) and answer.get("eligible") is True:
        if (
            isinstance(answer.get("reason"), str)
            and answer.get("old_instrument") == old_identity["name"]
            and answer.get("new_instrument") == new_identity["name"]
            and _positive_quote(answer.get("old_quote"), facts["text"], old_identity["name"])
            and _positive_quote(answer.get("new_quote"), facts["text"], new_identity["name"])
        ):
            audit.update(
                eligible=True,
                status="replaced_instrument_in_new_material_confirmed",
                sam_query=old_identity["name"],
            )
    return audit


def clean(ctx, plan, placement, generated):
    audit = decide(ctx, plan, placement, generated)
    ctx.params.setdefault("replacement_material_old_instrument_cleanup", []).append(audit)
    if not audit["eligible"]:
        return generated
    from audio_edit.models import AudioEditContext
    from audio_edit.plans.remove_sound_plan import RemoveSoundPlan
    from audio_edit.runners.samaudio_runner import SAMAudioRunner
    import numpy as np
    import soundfile as sf

    root = Path("audio_edit/generated")
    root.mkdir(parents=True, exist_ok=True)
    folder = Path(mkdtemp(prefix="material-old-instrument-", dir=root)).resolve()
    removal = RemoveSoundPlan(
        backend="samaudio",
        mode="text",
        result_mode="residual",
        text_prompt=audit["sam_query"],
        temporal_status="resolved",
        notes="Remove the positively heard old instrument only from the first generated replacement material.",
    )
    result = SAMAudioRunner(output_dir=str(folder)).run_remove(
        AudioEditContext(
            f"Remove replaced {audit['sam_query']} from generated replacement material",
            input_audio=generated,
        ),
        removal,
    )
    original, rate = sf.read(generated, always_2d=True, dtype="float32")
    output, sr = sf.read(result.output_audio, always_2d=True, dtype="float32")
    if sr != rate:
        resampled = folder / "resampled.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                result.output_audio,
                "-ar",
                str(rate),
                "-c:a",
                "pcm_f32le",
                str(resampled),
            ],
            check=True,
        )
        output, sr = sf.read(resampled, always_2d=True, dtype="float32")
    if output.shape[1] == 1 and original.shape[1] > 1:
        output = np.repeat(output, original.shape[1], axis=1)
    elif original.shape[1] == 1 and output.shape[1] > 1:
        output = output.mean(axis=1, keepdims=True)
    assert (
        output.shape[1] == original.shape[1]
        and abs(len(output) - len(original)) <= 2
        and np.isfinite(output).all()
    )
    aligned = np.zeros_like(original)
    aligned[: min(len(output), len(aligned))] = output[: len(aligned)]
    path = folder / "new_instrument.wav"
    sf.write(path, aligned, rate, subtype="FLOAT")
    audit.update(
        executed=True,
        input_audio=str(generated),
        input_sha256=sha(generated),
        output_audio=str(path),
        output_sha256=sha(path),
        separation_calls=result.params.get("separation_calls"),
        first_sample_grid=True,
        original_residual_unchanged=True,
    )
    return str(path)
