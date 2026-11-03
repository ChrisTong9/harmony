# Setup

## Install the interface

The CLI uses Python 3.10+ and the standard library. The planner and separation environments below use Python 3.11; generation uses Python 3.10. Run from the checkout or install the interface:

```bash
python -m pip install -e .
python -m harmony --help
python -m harmony config
```

Model inference requires Linux with a compatible NVIDIA GPU. Install FFmpeg with both `ffmpeg` and `ffprobe` on `PATH`; SAM Audio also needs TorchCodec and compatible FFmpeg/CUDA shared libraries.

## Resource layout

```text
resources/
├── .venvs/
│   ├── mixer/bin/python
│   ├── sam-audio/bin/python
│   ├── stable-audio-3/bin/python
│   ├── qwen/bin/vllm
│   └── music-render311/bin/python # optional
├── externals/
│   ├── sam-audio/
│   ├── stable-audio-3/
│   └── music-render/GeneralUser-GS.sf2 # optional
└── models/
    ├── Qwen3-Omni-30B-A3B-Instruct/
    └── whisper-large-v3-turbo/
        └── speech_model_manifest.json
```

Set `HARMONY_RESOURCE_ROOT` or pass `--resource-root`. Existing installations can be linked into this layout; the launcher does not download models or start Qwen for you.

```bash
export HARMONY_RESOURCE_ROOT=/path/to/resources
mkdir -p "$HARMONY_RESOURCE_ROOT/.venvs" "$HARMONY_RESOURCE_ROOT/externals" "$HARMONY_RESOURCE_ROOT/models"
```

SAM Audio and Stable Audio use the Hugging Face cache. Use `HF_HOME` or `HF_HUB_CACHE` to select that cache, and provision the required model access before running. `--checkpoint-cache` and `HARMONY_CHECKPOINT_CACHE` refer to the auxiliary `.checkpoints` directory used by optional legacy tools; they do not redirect Hugging Face downloads. Qwen and Whisper use the explicit local model paths above.

## Planner and model environments

The commands below specify the component environments and reference versions. A fresh installation of every environment and an end-to-end model run have not yet been validated for this release.

### Planner and audio composition

```bash
python3.11 -m venv "$HARMONY_RESOURCE_ROOT/.venvs/mixer"
"$HARMONY_RESOURCE_ROOT/.venvs/mixer/bin/python" -m pip install -r requirements/planner.txt
"$HARMONY_RESOURCE_ROOT/.venvs/mixer/bin/python" -m pip install \
  torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

### Source separation and speech evidence

Use [SAM Audio](https://github.com/facebookresearch/sam-audio) commit `bb4c6999d2677c7402360e426afc01ddfad6dce0`. The configured separator is `facebook/sam-audio-large`.

```bash
git clone https://github.com/facebookresearch/sam-audio.git "$HARMONY_RESOURCE_ROOT/externals/sam-audio"
git -C "$HARMONY_RESOURCE_ROOT/externals/sam-audio" checkout bb4c6999d2677c7402360e426afc01ddfad6dce0
python3.11 -m venv "$HARMONY_RESOURCE_ROOT/.venvs/sam-audio"
"$HARMONY_RESOURCE_ROOT/.venvs/sam-audio/bin/python" -m pip install \
  torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 torchcodec==0.5 --index-url https://download.pytorch.org/whl/cu126
"$HARMONY_RESOURCE_ROOT/.venvs/sam-audio/bin/python" -m pip install \
  -e "$HARMONY_RESOURCE_ROOT/externals/sam-audio" \
  torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 torchcodec==0.5 \
  transformers==4.57.1 huggingface-hub==0.36.0 numpy==1.26.4 \
  librosa==0.11.0 soundfile==0.14.0 scipy==1.17.1
```

The reference TorchCodec build is `0.5+cu126`. Follow the [TorchCodec installation instructions](https://github.com/meta-pytorch/torchcodec#installing-torchcodec) for the selected Torch/CUDA combination and native FFmpeg libraries. Upstream SAM dependencies include additional source repositories; the two commands above are not a full transitive lock.

Speech evidence requires a pinned Whisper model **and its manifest**. Merely downloading the weights is insufficient. This provided command downloads the declared revision and writes the required file checksums; it does not run inference:

```bash
"$HARMONY_RESOURCE_ROOT/.venvs/sam-audio/bin/python" -m audio_edit.tools.prepare_speech_model \
  --output "$HARMONY_RESOURCE_ROOT/models/whisper-large-v3-turbo"
```

The Whisper revision is `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`. Model identities and shared settings are declared in [`harmony/configs/defaults.json`](../harmony/configs/defaults.json). Speech evidence is unavailable if the manifest or runtime cannot be validated; check this before running an edit.

### Audio generation

Use [Stable Audio 3](https://github.com/Stability-AI/stable-audio-3) commit `a0b57f5483c4588f827f3552b7d5c6ca2a9687be`. Harmony uses its `medium` model, corresponding to `stabilityai/stable-audio-3-medium`.

```bash
git clone https://github.com/Stability-AI/stable-audio-3.git "$HARMONY_RESOURCE_ROOT/externals/stable-audio-3"
git -C "$HARMONY_RESOURCE_ROOT/externals/stable-audio-3" checkout a0b57f5483c4588f827f3552b7d5c6ca2a9687be
python3.10 -m venv "$HARMONY_RESOURCE_ROOT/.venvs/stable-audio-3"
"$HARMONY_RESOURCE_ROOT/.venvs/stable-audio-3/bin/python" -m pip install \
  torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
"$HARMONY_RESOURCE_ROOT/.venvs/stable-audio-3/bin/python" -m pip install \
  -e "$HARMONY_RESOURCE_ROOT/externals/stable-audio-3" \
  transformers==5.16.1 huggingface-hub==1.28.0 numpy==2.2.6 soundfile==0.14.0
```

The reference cache revisions are SAM `5f2cd3a9471a08c7282c06036be6893e18de8b70` and Stable Audio medium `d4d3579395a8ac0df4ab13b5b1d7138ce7290f81`. The current loaders resolve upstream defaults rather than enforce those revisions. Keep your model cache fixed and record its revisions when comparing runs; future downloads can change.

### Optional instrument rendering

Some musical replacements use Basic Pitch or continuous-pitch analysis with a SoundFont. Install this component separately:

```bash
python3.11 -m venv "$HARMONY_RESOURCE_ROOT/.venvs/music-render311"
"$HARMONY_RESOURCE_ROOT/.venvs/music-render311/bin/python" -m pip install -r requirements/music-render.txt
```

Normal package installation may also install TensorFlow and PyAudio. PyAudio may require PortAudio development headers on Linux. Harmony selects the packaged ONNX model explicitly and runs it with the CPU provider, regardless of whether TensorFlow is installed. Obtain [GeneralUser GS](https://www.schristiancollins.com/generaluser) under its own license and place the SoundFont at `externals/music-render/GeneralUser-GS.sf2`. Missing optional rendering resources leave those branches unavailable.

## Qwen service

The audio-language model is [Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct). Download its weights into the local model directory separately. If you already have a compatible endpoint, skip server installation and set `AUDIO_ANALYSIS_BASE_URL`.

The reference server environment uses vLLM `0.29.0`, Python 3.11, Torch `2.13.0`, and Transformers `5.17.0`. Keep it separate from the editing environments:

```bash
python3.11 -m venv "$HARMONY_RESOURCE_ROOT/.venvs/qwen"
"$HARMONY_RESOURCE_ROOT/.venvs/qwen/bin/python" -m pip install vllm==0.29.0
```

Start the audio service with the shared model alias in a separate terminal:

```bash
nvidia-smi
VLLM_WORKER_MULTIPROC_METHOD=spawn "$HARMONY_RESOURCE_ROOT/.venvs/qwen/bin/vllm" serve \
  "$HARMONY_RESOURCE_ROOT/models/Qwen3-Omni-30B-A3B-Instruct" \
  --served-model-name Qwen3Omni-Instruct \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 \
  --tensor-parallel-size 1 --max-model-len 32768 --max-num-seqs 1 \
  --enforce-eager --limit-mm-per-prompt '{"audio":3,"image":0,"video":0}' \
  --gpu-memory-utilization 0.9 \
  --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'
```

A loopback endpoint is reachable only on its server host. Use a reachable service address if editing runs on another host. For an existing endpoint with a different served alias, pass `--qwen-model` to both `doctor` and `run`.

## Credentials and checks

The main planner uses `gpt-5-mini-2025-08-07`; the basic execution route uses `gpt-5-mini`. Speech generation, when requested, uses `gpt-4o-mini-tts` and voice `alloy` by default, configurable through `OPENAI_TTS_MODEL` and `OPENAI_TTS_VOICE`. Hosted model calls require access and may incur charges.

Export credentials through the shell or job environment; `.env.example` contains names only:

```bash
export AUDIO_ANALYSIS_BASE_URL=http://YOUR_QWEN_HOST:8000/v1
```

Set `OPENAI_API_KEY` and, if your endpoint requires it, `AUDIO_ANALYSIS_API_KEY`. No credentials are included. Run the CPU installation check before inference:

```bash
python -m harmony doctor \
  --resource-root "$HARMONY_RESOURCE_ROOT"
```

Add `--check-service` to check the endpoint's model listing. This sends no audio or editing request. A passing installation check does not establish editing quality.

## Run an edit

```bash
python -m harmony run \
  --input /path/to/input.wav \
  --instruction "Replace the background noise with soft piano music." \
  --output-dir outputs/new-replacement
```

The command above performs the edit after the resources, credentials, and Qwen service are configured. Add `--dry-run` to preview it without model calls. The LLM router selects one primary operation from the instruction: addition, removal, replacement, or volume adjustment. You do not supply a task label. All operations use the shared settings; tool use and bounded corrections are selected from the current request and evidence. Split requests involving separate primary operations into separate edits.

Stable Audio generation uses seed `20261018` by default. Add `--seed 123` to choose another generation seed; this applies to Stable Audio in the primary and basic execution routes. It does not control the hosted planner, speech generation, or every source of runtime nondeterminism.

Each run saves its configuration, execution log, evidence, and report locally under `--output-dir`. Use a fresh output directory for every run. Use `report.json` to locate the output audio and inspect `execution_status`, `execution_path`, and `fallback_used`. The basic execution route is recorded explicitly. An otherwise completed operation that leaves the waveform unchanged is reported as `no_op`, with a nonzero exit status; producing a file does not establish that the instruction was fulfilled.

A dry run confirms command construction only.
