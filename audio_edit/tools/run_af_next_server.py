"""Small authenticated AF-Next service; inference is restricted to a GPU allocation."""

import argparse
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import os
import secrets
import subprocess
import threading
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        raise RuntimeError("AF-Next inference requires a Slurm GPU allocation")
    print(subprocess.check_output(["nvidia-smi"], text=True), flush=True)
    import torch
    import librosa
    import soundfile as sf
    from transformers import AutoProcessor, MusicFlamingoForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("No allocated CUDA device")
    key = os.environ["AF_NEXT_API_KEY"]
    processor = AutoProcessor.from_pretrained(args.model)
    model = (
        MusicFlamingoForConditionalGeneration.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to("cuda")
        .eval()
    )
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status, body):
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            return secrets.compare_digest(self.headers.get("Authorization", ""), "Bearer " + key)

        def do_GET(self):
            if not self.authorized():
                return self.reply(401, {"error": "Unauthorized"})
            if self.path != "/health":
                return self.reply(404, {"error": "Unknown endpoint"})
            self.reply(200, dict(model="nvidia/audio-flamingo-next-hf", revision=args.revision))

        def do_POST(self):
            if not self.authorized():
                return self.reply(401, {"error": "Unauthorized"})
            if self.path != "/analyze":
                return self.reply(404, {"error": "Unknown endpoint"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 128 * 1024 * 1024:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(size))
                prompt = payload["prompt"]
                if not isinstance(prompt, str) or not 0 < len(prompt) <= 16000:
                    raise ValueError("Invalid observation prompt")
                audio_bytes = base64.b64decode(payload["audio_base64"], validate=True)
                digest = hashlib.sha256(audio_bytes).hexdigest()
                if digest != payload["audio_sha256"]:
                    raise ValueError("Audio checksum mismatch")
                audio, sr = sf.read(BytesIO(audio_bytes), dtype="float32", always_2d=True)
                duration = len(audio) / sr
                if not 0 < duration <= 1800:
                    raise ValueError("Audio duration outside supported range")
                audio = audio.mean(axis=1)
                if sr != 16000:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            except (ValueError, TypeError, KeyError, RuntimeError) as exc:
                return self.reply(400, dict(error=f"{type(exc).__name__}: {exc}"))
            try:
                with lock:
                    started = time.monotonic()
                    messages = [
                        [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "audio", "audio": audio},
                                ],
                            }
                        ]
                    ]
                    inputs = processor.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True, return_dict=True
                    ).to(model.device)
                    if "input_features" in inputs:
                        inputs["input_features"] = inputs["input_features"].to(model.dtype)
                    with torch.inference_mode():
                        tokens = model.generate(
                            **inputs,
                            max_new_tokens=1024,
                            do_sample=False,
                            repetition_penalty=1.2,
                            use_cache=True,
                        )
                    output = processor.batch_decode(
                        tokens[:, inputs["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0]
                    result = dict(
                        text=output,
                        model="nvidia/audio-flamingo-next-hf",
                        revision=args.revision,
                        audio_sha256=digest,
                        duration=duration,
                        seconds=time.monotonic() - started,
                        generation=dict(
                            max_new_tokens=1024, do_sample=False, repetition_penalty=1.2
                        ),
                        input_shapes={
                            k: list(v.shape) for k, v in inputs.items() if torch.is_tensor(v)
                        },
                    )
                self.reply(200, result)
            except Exception as exc:
                import traceback

                traceback.print_exc()
                self.reply(500, dict(error=f"{type(exc).__name__}: {exc}"))

    print("AF_NEXT_READY", args.port, flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
