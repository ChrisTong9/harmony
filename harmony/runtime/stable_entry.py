"""Seeded audio generation for the shared editing pipeline."""

import os
import runpy
from harmony.runtime.gpu import require_gpu

require_gpu()
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
