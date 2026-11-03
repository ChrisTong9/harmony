<div align="center">

<img src=".github/assets/mark.svg" alt="Harmony" width="72">

# Harmony
### Bridging Audio Understanding and Editing through Agentic Collaboration

[Audio Demo](https://christong9.github.io/harmony/) · [Setup](docs/setup.md) · [420-case CSV](data/mmae_420.csv) · arXiv (coming soon)

Addition · Removal · Replacement · Volume adjustment

</div>

Harmony edits an audio file from a natural-language instruction and selects the editing operation automatically.

## Install

Install the command-line interface with **Python 3.10+**:

```bash
git clone https://github.com/ChrisTong9/harmony.git
cd harmony
python -m pip install -e .
```

## Configure

Follow the [setup guide](docs/setup.md) to install the model environments, download weights, and start or connect to a Qwen audio service. Set the resource directory and credentials in your shell:

```bash
export HARMONY_RESOURCE_ROOT=/path/to/resources
export OPENAI_API_KEY=your_api_key
export AUDIO_ANALYSIS_BASE_URL=http://localhost:8000/v1
harmony doctor --check-service
```

## Run

Provide your audio and one editing instruction:

```bash
python -m harmony run \
  --input /path/to/input.wav \
  --instruction "Make the door slamming sound louder." \
  --output-dir outputs/door-volume
```

Use a fresh output directory for each edit. The generated `report.json` identifies the output audio; `execution.log` contains runtime messages. Use one primary operation per request.

Run `harmony config` to view settings, add `--seed 123` to set the Stable Audio generation seed, or add `--dry-run` to preview a command without model calls. See [setup](docs/setup.md) for all configuration steps.
