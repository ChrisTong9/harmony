# Setup

## Command-line interface

Use Python 3.10 or later and run commands from the repository root. The public CLI and `--dry-run` use the standard library:

```bash
python -m harmony --help
python -m harmony profiles
```

Inference uses separately installed models and environments. The launcher requires Linux with an active Slurm GPU allocation and checks `nvidia-smi` before starting model execution.

## Model resources

Provide a resource directory with this layout:

```text
resources/
├── .venvs/
│   ├── mixer/bin/python           # planner and audio composition
│   ├── sam-audio/bin/python       # separation and speech tools
│   ├── stable-audio-3/bin/python   # audio generation
│   └── music-render311/bin/python # optional instrument rendering
├── externals/
│   ├── sam-audio/
│   ├── stable-audio-3/
│   └── music-render/GeneralUser-GS.sf2
├── models/
│   ├── Qwen3-Omni-30B-A3B-Instruct/
│   └── whisper-large-v3-turbo/
└── .checkpoints/
```

Set `HARMONY_RESOURCE_ROOT` or pass `--resource-root`. The checkpoint cache defaults to `RESOURCE_ROOT/.checkpoints`; override it with `HARMONY_CHECKPOINT_CACHE` or `--checkpoint-cache`.

Install the planner dependencies from [`requirements/planner.txt`](../requirements/planner.txt) in the `mixer` environment. Install matching PyTorch and Torchaudio builds for your CUDA environment. The development setup used Python 3.11 for the planner and SAM Audio, and Python 3.10 for Stable Audio. The requirements file covers the planner's direct dependencies, not every model environment.

| Component | Upstream project | Configured model or resource |
|---|---|---|
| Audio understanding | [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni) | `Qwen3-Omni-30B-A3B-Instruct` |
| Source separation | [SAM Audio](https://github.com/facebookresearch/sam-audio) | `facebook/sam-audio-large` |
| Audio generation | [Stable Audio 3](https://github.com/Stability-AI/stable-audio-3) | Separate generation environment |
| Speech timing | [Whisper large v3 turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | Local weights; revision in `configs/profiles.json` |

Follow the corresponding model project's installation instructions and access requirements. Optional video/MMAudio branches require additional resources and are outside the four audio tasks covered here.

## Services and credentials

Serve Qwen separately through an OpenAI-compatible endpoint. This checkout does not start the service. The profiles use `Qwen3Omni-Perception` and `Qwen3Omni-Instruct`; the endpoint must expose the model alias required by the selected profile.

The main planner is configured with `gpt-5-mini-2025-08-07`; basic execution uses `gpt-5-mini`. Configuration and endpoint access must be supplied before inference. Model/API calls may incur charges.

The `.env.example` file lists environment variable names. Export values in your shell or job environment:

```bash
export HARMONY_RESOURCE_ROOT=/path/to/resources
export HARMONY_CHECKPOINT_CACHE=/path/to/checkpoint-cache
export AUDIO_ANALYSIS_BASE_URL=http://YOUR_QWEN_HOST:PORT/v1
```

Supply `OPENAI_API_KEY` through your shell or job's secret environment. Set `AUDIO_ANALYSIS_API_KEY` if the Qwen endpoint requires it. No credentials are included in the repository.

## Run an edit

First preview the command:

```bash
python -m harmony run \
  --task replace \
  --input /path/to/input.wav \
  --instruction "Replace the background noise with soft piano music." \
  --output-dir outputs/new-replacement \
  --dry-run
```

Then run it without `--dry-run` inside your Slurm GPU allocation. Verify the allocation first:

```bash
nvidia-smi
```

Use `--cell A1B1` for both mechanisms, or `A0B0`, `A1B0`, and `A0B1` to disable the corresponding mechanisms. All configurations use the same editing implementation.

Each run saves its configuration, execution log, and report locally under `--output-dir`. Use a fresh output directory for every run. Evidence and intermediate audio remain local and are not part of the published data list.

A dry run confirms command construction only. It does not validate model installations, credentials, endpoint compatibility, GPU memory, or editing quality. Fresh GPU inference has not been rerun as part of this code consolidation.
