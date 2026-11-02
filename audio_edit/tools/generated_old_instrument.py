"""Remove positively observed replaced piano from a generated new-instrument asset."""

from copy import deepcopy
from pathlib import Path
import hashlib, json, re, subprocess
from tempfile import mkdtemp

PROMPT = "Review one FIRST generated replacement asset. This is a new isolated asset, not the original or retained residual. Decide whether it positively contains BOTH the requested new instrument and the old instrument which the original instruction explicitly replaces. Only then may one existing separator remove that old instrument from the generated asset. Do not remove preserved original voices or instruments. Do not infer a piano from a generic sustained synthesized chord, uncertainty, or a score. Do not reinterpret a requested ensemble as an isolated instrument. Return eligible:boolean, old_instrument:string or null, new_instrument:string or null, old_quote:exact continuous substring of first material facts or null, new_quote:exact continuous substring or null, reason:string. When false both instruments and quotes must be null. No generation, level, timing, seed, or source-query change. The sole candidate old-instrument query for this review is piano; if the text/facts do not positively establish piano being replaced and a distinct requested new instrument, return false."


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


INSTRUMENTS = (
    r"\b(piano|guitar|violin|viola|cello|flute|oboe|clarinet|trumpet|saxophone|harp|strings)\b"
)


def applicable(ctx, plan, placement):
    from audio_edit.tools.instrumental_material import applicable as instrumental

    if not instrumental(ctx, plan, placement):
        return False
    replacement = ctx.params["replacement_plan"]
    source = replacement.get("source_sound", "")
    if not isinstance(source, str) or set(re.findall(INSTRUMENTS, source.lower())) != {"piano"}:
        return False
    if not re.search(r"\bpiano\b", replacement.get("user_prompt", ""), re.I):
        return False
    # A generated ensemble or an uncertain retained source never licenses this cleanup.
    target = replacement.get("target_sound", "")
    if re.search(r"\b(ensemble|orchestra|band)\b", target, re.I):
        return False
    new_target = re.split(r"\b(?:replacing|instead of|in place of)\b", target.lower(), maxsplit=1)[
        0
    ]
    named = set(re.findall(INSTRUMENTS, new_target))
    return len(named) == 1 and "piano" not in named


def decide(ctx, plan, placement, generated, *, llm=None):
    audit = dict(eligible=False, status="not_applicable", new_material_observations=0)
    if not applicable(ctx, plan, placement):
        return audit
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
        quotes = [answer.get(k) for k in ["old_quote", "new_quote"]]
        requested = set(re.findall(INSTRUMENTS, replacement["target_sound"].lower())) - {"piano"}
        observed = set(re.findall(INSTRUMENTS, str(answer.get("new_instrument", "")).lower()))
        if (
            answer.get("old_instrument") == "piano"
            and observed == requested
            and all(isinstance(q, str) and q.strip() and q in facts["text"] for q in quotes)
        ):
            audit.update(
                eligible=True, status="replaced_piano_in_new_material_confirmed", sam_query="piano"
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
        text_prompt="piano",
        temporal_status="resolved",
        notes="Remove positively heard replaced piano only from the first generated new-instrument material.",
    )
    result = SAMAudioRunner(output_dir=str(folder)).run_remove(
        AudioEditContext(
            "Remove replaced piano from generated new-instrument material", input_audio=generated
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
