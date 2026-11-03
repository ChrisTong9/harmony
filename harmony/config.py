"""Resolve model and resource settings for the complete Harmony pipeline."""

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "configs/defaults.json"


def load_config(*, qwen_model=None, seed=None):
    config = json.loads(CONFIG_PATH.read_text())
    if qwen_model is not None:
        if not qwen_model.strip():
            raise ValueError("The Qwen served model name cannot be empty")
        config["perception"]["model"] = qwen_model.strip()
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
            raise ValueError("The generation seed must be an integer in [0, 2147483647]")
        config["generation_seed"] = seed
    config.update(
        preedit_environment={
            "AUDIO_PREEDIT_SPEECH": "1",
            "AUDIO_PREEDIT_SEPARATION": "1",
            "AUDIO_ADD_TIMING_RECOVERY": "overlap_preserve",
            "HARMONY_BASIC_EXECUTION": "1",
        },
    )
    return config
