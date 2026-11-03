"""Check GPU access without making cluster scheduling mandatory on workstations."""

import os
import socket
import subprocess


def on_turing():
    """Recognize the WPI cluster where GPU work requires a Slurm allocation."""
    names = (socket.gethostname().lower(), socket.getfqdn().lower())
    return os.environ.get("SLURM_CLUSTER_NAME", "").lower() == "turing" or any(
        name == "turing.wpi.edu" or name.endswith(".turing.wpi.edu") for name in names
    )


def _job_id():
    return os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")


def _devices(value):
    if value is None:
        return None
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens or any(token.lower() in {"-1", "none", "void", "nodevfiles"} for token in tokens):
        return []
    return tokens


def gpu_policy_error():
    """Return a scheduling/visibility error without probing or initializing CUDA."""
    if on_turing() and not _job_id():
        return "GPU inference on Turing requires a Slurm GPU allocation."
    if _job_id():
        assigned = os.environ.get("SLURM_STEP_GPUS", os.environ.get("SLURM_JOB_GPUS"))
        if not _devices(assigned):
            return "No GPU granted by Slurm to this job/step."
    if _devices(os.environ.get("CUDA_VISIBLE_DEVICES")) == []:
        return "CUDA_VISIBLE_DEVICES hides all GPU devices."
    return None


def require_compute_allocation():
    """Keep optional CPU model work off Turing login nodes; allow other hosts."""
    if on_turing() and not _job_id():
        raise RuntimeError("Model inference on Turing requires a Slurm allocation.")


def require_gpu(*, emit=False):
    """Check scheduling and nvidia-smi before any local GPU inference.

    Slurm device variables describe the allocation; nvidia-smi may list additional
    host GPUs and is never treated as permission to use them. CUDA visibility and
    Slurm's device isolation remain unchanged.
    """
    error = gpu_policy_error()
    if error:
        raise RuntimeError(error)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("nvidia-smi could not verify an available NVIDIA GPU.") from exc
    devices = [line.split(",", 2) for line in result.stdout.splitlines() if line.strip()]
    if not devices or any(len(row) != 3 or not row[0].strip().isdigit() for row in devices):
        raise RuntimeError("nvidia-smi reported no NVIDIA GPUs.")
    if emit:
        print(result.stdout.strip(), flush=True)
    return result.stdout.strip()
