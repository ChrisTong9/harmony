from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping
import uuid


DEFAULT_MODEL = "Qwen/Qwen-Image-2512"


def run_request(
    request: Mapping[str, Any],
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    env = env or os.environ

    import torch
    from diffusers import QwenImagePipeline

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Qwen-Image-2512 requires a visible CUDA GPU. Run inside an active "
            "Turing Slurm GPU allocation."
        )

    prompt = str(request.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("prompt must not be empty.")

    output_value = request.get("output_dir") or env.get("QWEN_IMAGE_OUTPUT_DIR")
    output_dir = (
        Path(str(output_value)).expanduser()
        if output_value
        else Path.cwd() / "audio_edit" / "outputs" / "qwen_image_refs"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_id = env.get("QWEN_IMAGE_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    cache_value = env.get("QWEN_IMAGE_CACHE_DIR")
    cache_dir = (
        Path(cache_value).expanduser() if cache_value else Path.cwd().parent / "Qwen-Image-models"
    ).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    pipe = QwenImagePipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        cache_dir=str(cache_dir),
    ).to("cuda")

    seed = int(request.get("seed", 42))
    image = pipe(
        prompt=prompt,
        negative_prompt=str(request.get("negative_prompt", "")).strip(),
        width=int(request.get("width", 1328)),
        height=int(request.get("height", 1328)),
        num_inference_steps=int(request.get("num_inference_steps", 50)),
        true_cfg_scale=float(request.get("true_cfg_scale", 4.0)),
        generator=torch.Generator(device="cuda").manual_seed(seed),
    ).images[0]

    output_image = (output_dir / f"reference_{uuid.uuid4().hex}.png").resolve()
    image.save(output_image)

    return {
        "ok": True,
        "output_image": str(output_image),
        "model": model_id,
        "seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()

    request_path = Path(args.request)
    response_path = Path(args.response)

    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        response = run_request(request)
        return_code = 0
    except Exception as exc:
        response = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(response["error"], file=sys.stderr)
        return_code = 1

    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(
        json.dumps(response, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
