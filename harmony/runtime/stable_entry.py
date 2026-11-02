"""Seeded audio generation for the shared editing pipeline."""

import os
import runpy
import subprocess

if not os.environ.get("SLURM_JOB_ID"):
    raise RuntimeError("Slurm allocation required")
if not (os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS")):
    raise RuntimeError("No GPU granted by Slurm to this job/step")
print(subprocess.check_output(["nvidia-smi"], text=True), flush=True)
from stable_audio_3 import StableAudioModel

original = StableAudioModel.generate
seed = int(os.environ["REPLACEMENT_AUDIO_SEED"])


def fixed_seed(self, *args, **kwargs):
    kwargs.setdefault("seed", seed)
    if kwargs["seed"] != seed:
        raise ValueError("Declared seed mismatch")
    return original(self, *args, **kwargs)


StableAudioModel.generate = fixed_seed
runpy.run_path(os.environ["HARMONY_STABLE_TARGET"], run_name="__main__")
