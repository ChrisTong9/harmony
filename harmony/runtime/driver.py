"""Launch the shared editing pipeline in an isolated run directory."""

from pathlib import Path
import json
import os
import shutil
import subprocess

from harmony.config import load_config
from .gpu import require_gpu


def run(args, root):
    config = load_config(qwen_model=args.qwen_model, seed=args.seed)
    if not args.instruction.strip():
        raise ValueError("Provide a nonempty editing instruction")
    source = root / "audio_edit"
    resource = args.resource_root.resolve()
    out = args.output_dir.resolve()
    inp = args.input.resolve()
    if any(out == package or package in out.parents for package in (source, root / "harmony")):
        raise ValueError("Place --output-dir outside the audio_edit/ and harmony/ source packages")
    command = [
        str(resource / ".venvs/mixer/bin/python"),
        "-m",
        "audio_edit.run_edit",
        "--input",
        str(inp),
        "--instruction",
        args.instruction,
        "--report",
        str(out / "report.json"),
    ]
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    action="new_input_inference",
                    source=str(source),
                    operation_selection="planner",
                    model=config["perception"]["model"],
                    generation_seed=config["generation_seed"],
                    command=command,
                    resource_root=str(resource),
                    checkpoint_cache=str(
                        (args.checkpoint_cache or out / "working/.checkpoints").resolve()
                    ),
                    requires="Configured model resources and an existing Qwen audio endpoint",
                    no_model_calls=True,
                ),
                indent=2,
            )
        )
        return 0
    from audio_edit.utils.audio_input import validate_audio_input

    validate_audio_input(inp)
    require_gpu()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in your environment; no credentials are bundled.")
    qwen = args.qwen_url or os.environ.get("AUDIO_ANALYSIS_BASE_URL")
    if not qwen:
        raise RuntimeError(
            "Pass --qwen-url or set AUDIO_ANALYSIS_BASE_URL for your running Qwen service."
        )
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Use a fresh output directory to preserve first outputs.")
    if not Path(command[0]).is_file():
        raise FileNotFoundError(command[0])
    cache = (args.checkpoint_cache or out / "working/.checkpoints").resolve()
    if args.checkpoint_cache and not cache.is_dir():
        raise FileNotFoundError("Legacy tool cache: " + str(cache))
    from harmony.diagnostics import inspect, print_report

    diagnostics = inspect(args, root, config=config, check_service=True)
    if not diagnostics["ready"]:
        print_report(diagnostics)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    work = out / "working"
    work.mkdir()
    local = out / "source"
    ignore = shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".env", ".env.*", "outputs", "generated", "evidence"
    )
    shutil.copytree(source, local / "audio_edit", ignore=ignore)
    shutil.copytree(root / "harmony", local / "harmony", ignore=ignore)
    env = os.environ.copy()
    env.update(config["preedit_environment"])
    env.update(
        LLM_BACKEND="openai",
        OPENAI_MODEL=config["openai_model"],
        QWEN_AUDIO_BACKEND="openai_compatible",
        QWEN_AUDIO_MODEL=config["perception"]["model"],
        AUDIO_ANALYSIS_BASE_URL=qwen,
        AUDIO_ANALYSIS_API_KEY=env.get("AUDIO_ANALYSIS_API_KEY", "EMPTY"),
        AUDIO_ANALYSIS_MAX_TOKENS="4096",
        AUDIO_ANALYSIS_COLLABORATOR="",
        AUDIO_EDIT_EVIDENCE_DIR=str(out / "evidence"),
        AUDIO_ANALYSIS_LOG=str(out / "analysis_calls.jsonl"),
        SAMAUDIO_MODEL_ID=config["samaudio"]["model_id"],
        HARMONY_BASIC_MODEL=config["basic_execution_model"],
        PYTHONDONTWRITEBYTECODE="1",
    )

    def relocate(p):
        return str(p).replace("${RESOURCE_ROOT}", str(resource))

    for key, name in [
        ("SAMAUDIO_PYTHON", "sam-audio"),
        ("MIXER_PYTHON", "mixer"),
        ("STABLE_AUDIO_PYTHON", "stable-audio-3"),
    ]:
        env[key] = str(resource / ".venvs" / name / "bin/python")
    for key, name in [
        ("SAMAUDIO_SCRIPT", "run_sam_audio.py"),
        ("MIXER_SCRIPT", "run_mix.py"),
        ("STABLE_AUDIO_SCRIPT", "run_stable_audio.py"),
    ]:
        env[key] = str(local / "audio_edit/tools" / name)
    env.update(
        AUDIO_SPEECH_PYTHON=relocate(config["speech"]["python"]),
        AUDIO_SPEECH_MODEL=relocate(config["speech"]["weights"]),
        HARMONY_MUSIC_RENDER_PYTHON=relocate(config["music_render_python"]),
        HARMONY_MUSIC_SOUNDFONT=relocate(config["music_soundfont"]),
        HARMONY_RESOURCE_ROOT=str(resource),
    )
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(local),
            str(resource / "externals/stable-audio-3"),
            str(resource / "externals/sam-audio"),
        ]
    )
    if args.checkpoint_cache:
        (work / ".checkpoints").symlink_to(cache, target_is_directory=True)
    else:
        cache.mkdir()
    env["HARMONY_CHECKPOINT_CACHE"] = str(cache)
    from urllib.parse import urlparse

    env["NO_PROXY"] = ",".join(
        filter(None, [env.get("NO_PROXY"), urlparse(qwen).hostname, "127.0.0.1", "localhost"])
    )
    env["no_proxy"] = env["NO_PROXY"]
    seed = config["generation_seed"]
    env.update(
        REPLACEMENT_AUDIO_SEED=str(seed),
        BASIC_EXECUTION_SEED=str(seed),
        HARMONY_STABLE_TARGET=str(local / "audio_edit/tools/run_stable_audio.py"),
        STABLE_AUDIO_SCRIPT=str(local / "harmony/runtime/stable_entry.py"),
    )
    env["HARMONY_COMMON_SOURCE"] = str(local)
    from datetime import datetime, timezone
    import hashlib
    from .provenance import capture

    digest = hashlib.sha256()
    with inp.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    record = dict(
        profile=config,
        source=str(source),
        resource_root=str(resource),
        input_sha256=digest.hexdigest(),
        started_at=datetime.now(timezone.utc).isoformat(),
        diagnostics=diagnostics,
        provenance=capture(root, resource, source=local),
    )
    config_path = out / "run_config.json"
    config_path.write_text(json.dumps(record, indent=2) + "\n")
    with (out / "execution.log").open("w") as log:
        result = subprocess.run(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
    record.update(
        finished_at=datetime.now(timezone.utc).isoformat(),
        exit_code=result.returncode,
        provenance=capture(root, resource, source=local),
    )
    config_path.write_text(json.dumps(record, indent=2) + "\n")
    if result.returncode:
        print("Edit failed; see", out / "execution.log", "exit", result.returncode)
    else:
        print("New edit completed:", out / "report.json")
    return result.returncode
