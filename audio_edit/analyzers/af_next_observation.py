"""AF-Next provides an independent note for a bounded Qwen cross-check."""

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import httpx


def observe(audio_b64, evidence, instruction, duration):
    prompt = (
        f"Describe the sounds you hear in this {duration:.2f}-second recording. "
        "Identify the main sound sources and any audible instruments or voices. "
        "Keep the answer brief."
    )
    record = dict(
        at=datetime.now(timezone.utc).isoformat(),
        stage="af_next_observation",
        backend="af_next",
        request=dict(prompt=prompt, audio=evidence),
    )
    try:
        with httpx.Client(timeout=240, trust_env=False) as client:
            response = client.post(
                os.environ["AF_NEXT_BASE_URL"].rstrip("/") + "/analyze",
                headers={"Authorization": "Bearer " + os.environ["AF_NEXT_API_KEY"]},
                json=dict(audio_base64=audio_b64, prompt=prompt, audio_sha256=evidence["sha256"]),
            )
            response.raise_for_status()
            result = response.json()
        record["response"] = result
        if result.get("audio_sha256") != evidence["sha256"]:
            raise ValueError("AF-Next analyzed different audio bytes")
        if result.get("model") != "nvidia/audio-flamingo-next-hf":
            raise ValueError("Unexpected AF-Next service model")
        if not isinstance(result.get("text"), str) or not result["text"].strip():
            raise ValueError("AF-Next returned no observation")
        if len(result["text"]) > 16000:
            raise ValueError("AF-Next observation exceeds the context budget")

        return result
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        log = os.getenv("AUDIO_ANALYSIS_LOG")
        if log:
            with Path(log).open("a") as out:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")


def cross_check_prompt(primary, observation):
    return (
        "You already analyzed this audio independently. Here is your independent draft:\n"
        + json.dumps(primary, ensure_ascii=False)
        + "\nA second model supplied this fallible observation:\n"
        + observation["text"]
        + "\nRe-listen to the original audio. Keep your independent draft as the starting "
        "point. Do not replace it just because the second model says something different. "
        "Only make a correction or add a detail if you can verify it directly by hearing "
        "the audio. When uncertain, retain the independently supported draft and explain "
        "the disagreement. Never merge incompatible sources or average timestamps. "
        "Do not infer instrumentation from familiar lyrics or turn a future edit into "
        "an existing sound. Return the full original JSON schema and cross_check_notes "
        "explaining what, if anything, changed and which claims were rejected."
    )


def audit(observation, qwen_model):
    return dict(
        mode="independent_qwen3_with_af_next_cross_check",
        qwen_model=qwen_model,
        af_next_model=observation["model"],
        af_next_revision=observation.get("revision"),
        af_next_response_sha256=hashlib.sha256(observation["text"].encode()).hexdigest(),
    )
