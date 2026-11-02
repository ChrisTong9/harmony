"""Check installed resources without loading model weights or editing audio."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .config import load_profile

PROBE = r"""
import contextlib, importlib, importlib.metadata, io, json, sys
modules = json.loads(sys.argv[1])
result = {"python": sys.version.split()[0], "modules": {}}
for module, distribution in modules.items():
    item = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            importlib.import_module(module)
        item["available"] = True
    except Exception as exc:
        item.update(available=False, error=f"{type(exc).__name__}: {exc}"[:600])
    try:
        item["version"] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        item["version"] = None
    result["modules"][module] = item
print(json.dumps(result))
"""


def probe_python(path, modules, pythonpath):
    if not Path(path).is_file() or not os.access(path, os.X_OK):
        return {"available": False, "error": f"Interpreter not executable: {path}"}
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
    env["PYTHONPATH"] = pythonpath
    try:
        completed = subprocess.run(
            [str(path), "-c", PROBE, json.dumps(modules)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if completed.returncode:
            return {"available": False, "error": "Interpreter probe exited with an error"}
        result = json.loads(completed.stdout)
        result["available"] = all(item["available"] for item in result["modules"].values())
        return result
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"[:600]}


def inspect(args, root, *, config=None, check_service=False):
    config = config or load_profile(args.task)
    resource = args.resource_root.resolve()
    checks = []

    def add(name, available, detail, required=True):
        checks.append(
            dict(
                name=name,
                status="ok" if available else "error" if required else "warning",
                detail=detail,
            )
        )

    def relocate(path):
        return Path(str(path).replace("${RESOURCE_ROOT}", str(resource)))

    for executable in ("ffmpeg", "ffprobe"):
        path = shutil.which(executable)
        add(executable, bool(path), path or f"Install {executable} and add it to PATH")
    add(
        "planner_credentials",
        bool(os.getenv("OPENAI_API_KEY")),
        "OPENAI_API_KEY is set" if os.getenv("OPENAI_API_KEY") else "Export OPENAI_API_KEY",
    )
    qwen = args.qwen_url or os.getenv("AUDIO_ANALYSIS_BASE_URL", "")
    parsed = urlparse(qwen)
    valid_url = parsed.scheme in ("http", "https") and bool(parsed.netloc)
    add(
        "qwen_endpoint",
        valid_url,
        f"Expected served model: {config['perception']['model']}"
        if valid_url
        else "Set --qwen-url or AUDIO_ANALYSIS_BASE_URL to an HTTP(S) /v1 endpoint",
    )

    pythonpath = os.pathsep.join(
        [
            str(root),
            str(resource / "externals/sam-audio"),
            str(resource / "externals/stable-audio-3"),
        ]
    )
    generation = args.task in ("add", "replace")
    speech = config["preedit_environment"]["AUDIO_PREEDIT_SPEECH"] == "1"
    environments = [
        (
            "mixer",
            {
                "numpy": "numpy",
                "soundfile": "soundfile",
                "torch": "torch",
                "torchaudio": "torchaudio",
                "openai": "openai",
                "dotenv": "python-dotenv",
                "pydantic": "pydantic",
                "requests": "requests",
                "httpx": "httpx",
                "google.genai": "google-genai",
            },
            True,
        ),
        (
            "sam-audio",
            {
                "sam_audio": "sam_audio",
                "torch": "torch",
                "torchaudio": "torchaudio",
                **({"transformers": "transformers", "soundfile": "soundfile"} if speech else {}),
            },
            True,
        ),
        (
            "stable-audio-3",
            {"stable_audio_3": "stable_audio_3", "torch": "torch", "torchaudio": "torchaudio"},
            generation,
        ),
        (
            "music-render311",
            {
                "basic_pitch": "basic-pitch",
                "onnxruntime": "onnxruntime",
                "tinysoundfont": "tinysoundfont",
                "librosa": "librosa",
                "scipy": "scipy",
                "soundfile": "soundfile",
            },
            False,
        ),
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(probe_python, resource / f".venvs/{name}/bin/python", modules, pythonpath)
            for name, modules, _ in environments
        ]
        for (name, _, required), future in zip(environments, futures):
            result = future.result()
            add(f"runtime:{name}", result["available"], result, required)

    if speech:
        model = relocate(config["speech"]["weights"])
        try:
            manifest = json.loads((model / "speech_model_manifest.json").read_text())
            required_files = {
                "config.json",
                "generation_config.json",
                "model.safetensors",
                "preprocessor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
            }
            files = manifest.get("files", {})
            valid = (
                manifest.get("model_id") == config["speech"]["model"]
                and manifest.get("revision") == config["speech"]["revision"]
                and required_files.issubset(files)
                and all(
                    Path(name).name == name
                    and (model / name).is_file()
                    and (model / name).stat().st_size == record.get("bytes")
                    for name, record in files.items()
                )
            )
            add(
                "speech_model",
                valid,
                "Whisper identity, file list and sizes checked; hashes checked by the speech runtime",
            )
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            add(
                "speech_model",
                False,
                f"Run python -m audio_edit.tools.prepare_speech_model --output {model}: {type(exc).__name__}",
            )
    soundfont = relocate(config["music_soundfont"])
    add("music_soundfont", soundfont.is_file(), str(soundfont), False)

    if check_service and valid_url:
        key = os.getenv("AUDIO_ANALYSIS_API_KEY") or "EMPTY"
        request = urllib.request.Request(
            qwen.rstrip("/") + "/models", headers={"Authorization": f"Bearer {key}"}
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                models = [row["id"] for row in json.load(response)["data"]]
            add(
                "qwen_served_model",
                config["perception"]["model"] in models,
                {"expected": config["perception"]["model"], "available": models},
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            add(
                "qwen_served_model",
                False,
                f"Cannot verify the /models response: {type(exc).__name__}",
            )

    return dict(
        task=args.task,
        resource_root=str(resource),
        checks=checks,
        ready=not any(check["status"] == "error" for check in checks),
        validation_scope="CPU imports and resource checks; no model weights loaded, audio sent, or GPU inference",
        service_checked=check_service and valid_url,
    )


def print_report(report, *, as_json=False):
    if as_json:
        print(json.dumps(report, indent=2))
        return
    for check in report["checks"]:
        detail = check["detail"]
        if isinstance(detail, dict):
            failed = [
                f"{name}: {item.get('error', 'unavailable')}"
                for name, item in detail.get("modules", {}).items()
                if not item["available"]
            ]
            detail = (
                "; ".join(failed)
                or detail.get("error")
                or f"Python {detail.get('python', '?')} imports available"
            )
        print(f"[{check['status'].upper()}] {check['name']}: {detail}")
    print(
        "Resource checks passed."
        if report["ready"]
        else "Resolve the ERROR items before inference."
    )
    print("This check does not validate GPU execution or editing quality.")
