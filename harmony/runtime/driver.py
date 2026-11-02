"""Relocation adapter for pinned primary code; never reads benchmark IDs or scores."""

from pathlib import Path
import json
import os
import shutil
import subprocess


def require_gpu():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this command inside a Slurm GPU allocation.")
    if not (os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS")):
        raise RuntimeError("No GPU granted by Slurm to this job/step.")
    print(subprocess.check_output(["nvidia-smi"], text=True), flush=True)


def run(args, root):
    config = json.loads((root / "configs/profiles.json").read_text())[args.cell][args.task]
    source = root / config["source_directory"]
    resource = args.resource_root.resolve()
    out = args.output_dir.resolve()
    inp = args.input.resolve()
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
        "--perception-mode",
        config["perception_mode"],
    ]
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    action="new_input_inference",
                    source=str(source),
                    cell=args.cell,
                    task=args.task,
                    command=command,
                    resource_root=str(resource),
                    checkpoint_cache=str(
                        (args.checkpoint_cache or resource / ".checkpoints").resolve()
                    ),
                    requires="Slurm GPU allocation and existing Qwen audio endpoint",
                    primary_profile_only=True,
                    historical_repair_episodes="See reproducibility/SOURCES.json; not automatically selected for new inputs.",
                    no_model_calls=True,
                ),
                indent=2,
            )
        )
        return 0
    require_gpu()
    if not inp.is_file():
        raise FileNotFoundError(inp)
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
    manifest = json.loads((source / "SOURCE.json").read_text())
    import hashlib

    for rel, expected in manifest["files"].items():
        if hashlib.sha256((source / rel).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Pinned source changed: " + rel)
    cache = (args.checkpoint_cache or resource / config["checkpoint_cache"]).resolve()
    if not cache.is_dir():
        raise FileNotFoundError("Model checkpoint cache: " + str(cache))
    out.mkdir(parents=True, exist_ok=True)
    work = out / "working"
    work.mkdir()
    local = out / "source"
    shutil.copytree(source, local)
    env = os.environ.copy()
    env.update(config["preedit_environment"])
    env.update(
        LLM_BACKEND="openai",
        OPENAI_MODEL=config["openai_model"],
        AUDIO_EDIT_PERCEPTION_MODE=config["perception_mode"],
        QWEN_AUDIO_BACKEND="openai_compatible",
        QWEN_AUDIO_MODEL=config["perception"]["model"],
        AUDIO_ANALYSIS_BASE_URL=qwen,
        AUDIO_ANALYSIS_API_KEY=env.get("AUDIO_ANALYSIS_API_KEY", "EMPTY"),
        AUDIO_ANALYSIS_MAX_TOKENS="4096",
        AUDIO_ANALYSIS_COLLABORATOR="",
        AUDIO_EDIT_EVIDENCE_DIR=str(out / "evidence"),
        AUDIO_ANALYSIS_LOG=str(out / "analysis_calls.jsonl"),
        SAMAUDIO_MODEL_ID=config["samaudio"]["model_id"],
        PYTHONDONTWRITEBYTECODE="1",
    )

    def relocate(p):
        return str(p).replace("${RESOURCE_ROOT}", str(resource))

    for key, name in [
        ("SAMAUDIO_PYTHON", "sam-audio"),
        ("MIXER_PYTHON", "mixer"),
        ("STABLE_AUDIO_PYTHON", "stable-audio-3"),
        ("MMAUDIO_PYTHON", "MMAudio"),
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
    (work / ".checkpoints").symlink_to(cache, target_is_directory=True)
    env["HARMONY_CHECKPOINT_CACHE"] = str(cache)
    from urllib.parse import urlparse

    env["NO_PROXY"] = ",".join(
        filter(None, [env.get("NO_PROXY"), urlparse(qwen).hostname, "127.0.0.1", "localhost"])
    )
    env["no_proxy"] = env["NO_PROXY"]
    seed = config.get("new_generation_seed")
    if seed is not None:
        env.update(
            REPLACEMENT_AUDIO_SEED=str(seed),
            HARMONY_STABLE_TARGET=str(local / "audio_edit/tools/run_stable_audio.py"),
            STABLE_AUDIO_SCRIPT=str(root / "harmony/runtime/stable_entry.py"),
        )
    else:
        env.pop("REPLACEMENT_AUDIO_SEED", None)
    common = json.loads((root / "configs/runtime.json").read_text())["common_source"]
    env["HARMONY_COMMON_SOURCE"] = str(root / common)
    bridge = local / "audio_edit/tools/basic_bridge.py"
    value = bridge.read_text()
    before = value
    value = value.replace(
        "Path('/home/ztong/agent/evaluation/mmae/harmony_basic_execution_v372_20261023')",
        "Path(" + repr(str(root / "harmony/runtime/service")) + ")",
    )
    value = value.replace("'/home/ztong/agent/.venvs/mixer/bin/python'", repr(command[0]))
    bridge.write_text(value)
    if value == before:
        raise RuntimeError("Common service relocation did not match pinned bridge")
    (out / "run_config.json").write_text(
        json.dumps(
            dict(
                profile=config,
                source_manifest=str(source / "SOURCE.json"),
                resource_root=str(resource),
                adapter_changes=["basic_bridge service path and interpreter only"],
                snapshot_result_claim=False,
            ),
            indent=2,
        )
        + "\n"
    )
    with (out / "execution.log").open("w") as log:
        result = subprocess.run(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print("Edit failed; see", out / "execution.log", "exit", result.returncode)
    else:
        print("New edit completed:", out / "report.json")
    return result.returncode
