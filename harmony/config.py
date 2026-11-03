"""Resolve model and resource settings for the complete Harmony pipeline."""

from copy import deepcopy
import json
from pathlib import Path

TASKS = ("add", "remove", "replace", "volume")
CONFIG_PATH = Path(__file__).parent / "configs/profiles.json"


def load_profile(task):
    if task not in TASKS:
        raise ValueError(f"Unsupported task: {task}")
    settings = json.loads(CONFIG_PATH.read_text())
    config = deepcopy(settings["defaults"])
    options = settings["tasks"][task]
    config["perception"]["model"] = options["perception_model"]
    config["new_generation_seed"] = options["generation_seed"]
    config.update(
        task=task,
        preedit_environment={
            "AUDIO_PREEDIT_SPEECH": "1",
            "AUDIO_PREEDIT_SEPARATION": "1",
            "AUDIO_ADD_TIMING_RECOVERY": "overlap_preserve",
            "HARMONY_BASIC_EXECUTION": "1",
        },
    )
    return config
