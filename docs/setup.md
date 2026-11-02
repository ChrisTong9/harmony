# Setup

## CPU-only interface

Python 3.10 or later is sufficient for the CLI, dry runs, source verification, table reconstruction, and release tests. No package installation is required for these commands. Run them from the repository root; this release is a source checkout, not a standalone inference wheel.

```bash
python -m harmony profiles
python -m harmony tables --out outputs/tables
python -m unittest discover -s tests -v
```

## Model execution

Inference was developed on Linux with Slurm and separate Python environments. The launcher intentionally requires a Slurm GPU allocation. Model installation and GPU inference have not been rerun as part of this repository reorganization.

Provide a resource directory with this layout:

```text
resources/
├── .venvs/
│   ├── mixer/bin/python           # planner and audio composition, Python 3.11
│   ├── sam-audio/bin/python       # SAM Audio and speech tools, Python 3.11
│   ├── stable-audio-3/bin/python  # Stable Audio, Python 3.10
│   └── music-render311/bin/python
├── externals/
│   ├── sam-audio/
│   ├── stable-audio-3/
│   └── music-render/GeneralUser-GS.sf2
├── models/
│   ├── Qwen3-Omni-30B-A3B-Instruct/
│   └── whisper-large-v3-turbo/
└── .checkpoints/                 # checkpoint cache used by model backends
```

The resource directory can be anywhere. Set `HARMONY_RESOURCE_ROOT` or pass `--resource-root`. The checkpoint cache defaults to `RESOURCE_ROOT/.checkpoints`; use `HARMONY_CHECKPOINT_CACHE` or `--checkpoint-cache` to point to an existing cache. The public configuration has no dependency on a particular researcher's workspace.

The optional legacy video/MMAudio branches remain in frozen sources but are outside the four audio tasks documented here. They require their own resources if used directly.

## Dependencies

`requirements/planner.txt` records direct planner dependencies observed at packaging time. It is not a complete lockfile for every model environment. The full observations, Python versions, and external repository commit IDs are in `reproducibility/environments/`.

Use the corresponding model project's installation instructions and recorded revision for SAM Audio and Stable Audio. Install matching PyTorch/Torchaudio builds for your CUDA environment; the recorded planner used `2.7.1+cu126`. Do not install every model into one environment by concatenating all observed package lists.

| Component | Upstream setup | Recorded resource |
|---|---|---|
| Audio understanding | [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni) | `Qwen3-Omni-30B-A3B-Instruct`, served separately |
| Source separation | [SAM Audio](https://github.com/facebookresearch/sam-audio) | `facebook/sam-audio-large`; source revision in `reproducibility/environments/external_repositories.json` |
| Audio generation | [Stable Audio 3](https://github.com/Stability-AI/stable-audio-3) | Separate Python 3.10 environment; recorded source revision in the same file |
| Speech timing | [Whisper large v3 turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | Model revision and hashes in `configs/profiles.json` |

For the planner environment, install `requirements/planner.txt` into `RESOURCE_ROOT/.venvs/mixer`, then install the matching PyTorch/Torchaudio build. The observed versions provide a reproducibility reference; fresh installation of all model environments has not been validated for this release.

The Qwen audio model is served separately through an OpenAI-compatible endpoint. This checkout does not start that server. The configured model aliases are `Qwen3Omni-Perception` and `Qwen3Omni-Instruct`; the endpoint must expose the alias used by the selected profile. The recorded weights are `Qwen3-Omni-30B-A3B-Instruct`.

The planner requests `gpt-5-mini-2025-08-07`; the common execution service requests `gpt-5-mini`. Hosted aliases can change over time. Access credentials and any model access agreements must be supplied separately.

## Credentials and execution

The `.env.example` file lists variable names only. The launcher reads the process environment; export the variables in your shell or provide them through your job environment.

```bash
export HARMONY_RESOURCE_ROOT=/path/to/resources
export HARMONY_CHECKPOINT_CACHE=/path/to/checkpoint-cache
export AUDIO_ANALYSIS_BASE_URL=http://YOUR_QWEN_HOST:PORT/v1
# Supply OPENAI_API_KEY through your shell or job's secret environment.
# Set AUDIO_ANALYSIS_API_KEY if your Qwen endpoint requires it.
```

Inside an active Slurm GPU allocation:

```bash
nvidia-smi
python -m harmony run \
  --task replace \
  --input /path/to/input.wav \
  --instruction "Replace the background noise with soft piano music." \
  --output-dir outputs/new-replacement
```

The launcher verifies the source tree, copies it into the fresh output directory, and adapts resource/service paths in that copy. It preserves the frozen source on disk. Each run writes its configuration, execution log, evidence, and report into the selected output directory. An existing nonempty output directory is rejected to avoid overwriting previous runs.

A successful dry run confirms profile selection and command construction. It does not establish that model installations, credentials, endpoint compatibility, or GPU memory are sufficient. Actual inference requires those resources and may incur API charges.
