# Setup

## Prerequisites and interface

Use Linux with a compatible NVIDIA GPU, Python 3.10 and 3.11, Git, and FFmpeg (`ffmpeg` and `ffprobe` on `PATH`). The model components use separate environments to accommodate their dependencies.

Run the following commands from the Harmony checkout. The interface itself requires Python 3.10+:

```bash
python -m pip install -e .
python -m harmony --help
python -m harmony config
```

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

Set `HARMONY_RESOURCE_ROOT` or pass `--resource-root`. Existing environments and model directories can be linked into this layout. Prepare the resources below before running an edit; Harmony does not install environments or start Qwen automatically.

```bash
export HARMONY_RESOURCE_ROOT=/path/to/resources
mkdir -p "$HARMONY_RESOURCE_ROOT/.venvs" "$HARMONY_RESOURCE_ROOT/externals" "$HARMONY_RESOURCE_ROOT/models"
```

SAM Audio and Stable Audio use the Hugging Face cache. Set `HF_HOME` or `HF_HUB_CACHE` if you want a custom cache location. Obtain access to [SAM Audio](https://huggingface.co/facebook/sam-audio-large) and [Stable Audio 3 medium](https://huggingface.co/stabilityai/stable-audio-3-medium), accept any required model terms, and set `HF_TOKEN` for downloads that require authentication. Qwen and Whisper use the explicit local model directories above.

## Planner and audio composition

```bash
python3.11 -m venv "$HARMONY_RESOURCE_ROOT/.venvs/mixer"
"$HARMONY_RESOURCE_ROOT/.venvs/mixer/bin/python" -m pip install -r requirements/planner.txt
"$HARMONY_RESOURCE_ROOT/.venvs/mixer/bin/python" -m pip install \
  torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

## Source separation and speech evidence

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

These commands use PyTorch 2.7.1 and TorchCodec 0.5 with CUDA 12.6. TorchCodec also needs compatible native FFmpeg libraries; see its [installation instructions](https://github.com/meta-pytorch/torchcodec#installing-torchcodec) if loading the decoder fails.

Download the separator weights into the Hugging Face cache:

```bash
"$HARMONY_RESOURCE_ROOT/.venvs/sam-audio/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("facebook/sam-audio-large")
PY
```

Prepare Whisper with the supplied command. It downloads the pinned model and creates the manifest required by the speech tools:

```bash
"$HARMONY_RESOURCE_ROOT/.venvs/sam-audio/bin/python" -m audio_edit.tools.prepare_speech_model \
  --output "$HARMONY_RESOURCE_ROOT/models/whisper-large-v3-turbo"
```

Whisper uses revision `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`. Keep `speech_model_manifest.json` with its downloaded files.

## Audio generation

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

Download the generation weights into the Hugging Face cache:

```bash
"$HARMONY_RESOURCE_ROOT/.venvs/stable-audio-3/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("stabilityai/stable-audio-3-medium")
PY
```

The SAM and Stable Audio loaders use their upstream default model revisions. Additional model dependencies may download on first use.

## Qwen service

The audio-language model is [Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct). If you already have a compatible endpoint, skip server installation and set `AUDIO_ANALYSIS_BASE_URL` in the configuration section below.

Otherwise, install vLLM in a separate environment and download the model:

```bash
python3.11 -m venv "$HARMONY_RESOURCE_ROOT/.venvs/qwen"
"$HARMONY_RESOURCE_ROOT/.venvs/qwen/bin/python" -m pip install vllm==0.29.0
"$HARMONY_RESOURCE_ROOT/.venvs/qwen/bin/python" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

destination = Path(os.environ["HARMONY_RESOURCE_ROOT"]) / "models/Qwen3-Omni-30B-A3B-Instruct"
snapshot_download("Qwen/Qwen3-Omni-30B-A3B-Instruct", local_dir=destination)
PY
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

A loopback endpoint is reachable only on its server host. Use a reachable service address if Harmony runs on another host. For an endpoint with a different served alias, pass `--qwen-model` to both `doctor` and `run`.

## Optional instrument rendering

To enable Basic Pitch and SoundFont rendering for supported musical replacements:

```bash
python3.11 -m venv "$HARMONY_RESOURCE_ROOT/.venvs/music-render311"
"$HARMONY_RESOURCE_ROOT/.venvs/music-render311/bin/python" -m pip install -r requirements/music-render.txt
mkdir -p "$HARMONY_RESOURCE_ROOT/externals/music-render"
```

Obtain [GeneralUser GS](https://www.schristiancollins.com/generaluser) and place the SoundFont at `$HARMONY_RESOURCE_ROOT/externals/music-render/GeneralUser-GS.sf2`. PyAudio may require PortAudio development headers on Linux. Harmony uses Basic Pitch's ONNX model with the CPU provider; TensorFlow may also be installed as a package dependency. The instrument-rendering component is optional.

## Configure credentials and endpoints

Set the following values in the shell where you run Harmony:

```bash
export OPENAI_API_KEY=your_api_key
export AUDIO_ANALYSIS_BASE_URL=http://YOUR_QWEN_HOST:8000/v1
```

Set `AUDIO_ANALYSIS_API_KEY` if your Qwen endpoint requires authentication. The planner uses `gpt-5-mini-2025-08-07` and `gpt-5-mini`; your OpenAI account needs access to these models. Requests to generate speech additionally use `gpt-4o-mini-tts`, with voice `alloy` by default. Set `OPENAI_TTS_MODEL` and `OPENAI_TTS_VOICE` to change the speech settings. Hosted API calls may incur charges.

Shared model settings are in [`harmony/configs/defaults.json`](../harmony/configs/defaults.json). Display them and check the configured resources and service:

```bash
harmony config
harmony doctor --check-service
```

`doctor` checks files, imports, and the endpoint's model listing without sending audio. Resolve any reported missing resources before running an edit.

## Run an edit

```bash
python -m harmony run \
  --input /path/to/input.wav \
  --instruction "Replace the background noise with soft piano music." \
  --output-dir outputs/new-replacement
```

The command performs an edit using the configured models. The operation is selected from the instruction; no task label is needed. Use one primary operation per request and a fresh output directory for each run.

Useful options:

- `--resource-root /path/to/resources`: override `HARMONY_RESOURCE_ROOT`.
- `--qwen-url http://host:8000/v1`: override `AUDIO_ANALYSIS_BASE_URL`.
- `--qwen-model model-alias`: match a different model alias served by your endpoint.
- `--seed 123`: change the Stable Audio generation seed (default: `20261018`). This does not seed hosted planning or speech generation.
- `--dry-run`: preview the command without model calls.

Open `report.json` in the output directory to find the output audio and execution status. The directory also contains `execution.log` and `run_config.json`. A failed, partial, or unchanged edit returns a nonzero exit status.
