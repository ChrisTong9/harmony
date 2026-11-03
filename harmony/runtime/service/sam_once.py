"""Seeded source separation for the bounded delivery route."""

from pathlib import Path
import os
import random
import runpy
import subprocess
import sys

if not os.environ.get("SLURM_JOB_ID"):
    raise RuntimeError("Slurm allocation required")
if not (os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS")):
    raise RuntimeError("No GPU granted by Slurm to this job/step")
print(subprocess.check_output(["nvidia-smi"], text=True), flush=True)
source = Path(os.environ["HARMONY_COMMON_SOURCE"])
sys.path.insert(0, str(source))
sys.path.insert(0, str(Path(os.environ["HARMONY_RESOURCE_ROOT"]) / "externals/sam-audio"))
import numpy as np
import torch

random.seed(20261018)
np.random.seed(20261018)
torch.manual_seed(20261018)
torch.cuda.manual_seed_all(20261018)
runpy.run_path(str(source / "audio_edit/tools/run_sam_audio.py"), run_name="__main__")
