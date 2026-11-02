"""Run the bounded delivery route using the shared editing implementation."""

from pathlib import Path
import argparse
import json
import os
import sys
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("request")
    p.add_argument("response")
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required")
    if not (os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS")):
        raise RuntimeError("No GPU granted by Slurm to this job/step")
    print(subprocess.check_output(["nvidia-smi"], text=True), flush=True)
    request = json.loads(Path(a.request).read_text())
    assert set(request) == {"original_instruction", "input_audio", "output_dir"}
    source = Path(os.environ["HARMONY_COMMON_SOURCE"])
    sys.path.insert(0, str(source))
    folder = Path(request["output_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    os.chdir(folder)
    cache = folder / ".checkpoints"
    if not cache.exists():
        cache.symlink_to(os.environ["HARMONY_CHECKPOINT_CACHE"], target_is_directory=True)
    os.environ.update(
        LLM_BACKEND="openai",
        OPENAI_MODEL="gpt-5-mini",
        SAMAUDIO_SCRIPT=str(Path(__file__).parent / "sam_once.py"),
        STABLE_AUDIO_SCRIPT=str(source / "audio_edit/tools/run_stable_audio.py"),
        BASIC_EXECUTION_SEED="20261018",
        PYTHONPATH=os.pathsep.join(
            [
                str(source),
                str(Path(os.environ["HARMONY_RESOURCE_ROOT"]) / "externals/sam-audio"),
                str(Path(os.environ["HARMONY_RESOURCE_ROOT"]) / "externals/stable-audio-3"),
            ]
        ),
    )
    from audio_edit.models import AudioEditContext
    from audio_edit.tools.basic_execution import recover

    ctx = recover(
        AudioEditContext(request["original_instruction"], input_audio=request["input_audio"]),
        request["input_audio"],
    )
    Path(a.response).write_text(
        json.dumps(
            dict(
                output_audio=ctx.output_audio,
                params=ctx.params,
                warnings=ctx.warnings,
                execution_status=ctx.execution_status,
                execution_error=ctx.execution_error,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
