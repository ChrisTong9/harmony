# Example requests

These requests use the unchanged input files already present in the public demo. They are examples of the interface, not a new evaluation set or newly generated outputs.

| Task | Instruction | Input |
|---|---|---|
| Add | Add four beats of drum kits before the drum start, the speed in line with the drum sound. | `audio/69e89805bd490e32ff0d3c9d/input.wav` |
| Remove | Remove the sound similar to "ewww" at the end of the audio. | `audio/69e8981f6508941e55c185a6/input.wav` |
| Replace | Replace the ending laughter with applause. | `audio/69e89817910050e595461416/input.wav` |
| Volume | Make the door slamming sound louder. | `audio/69e8980c910050e59545dd12/input.wav` |

`requests.json` contains the same four requests in machine-readable form.

```bash
python -m harmony run \
  --task add \
  --input audio/69e89805bd490e32ff0d3c9d/input.wav \
  --instruction "Add four beats of drum kits before the drum start, the speed in line with the drum sound." \
  --output-dir outputs/four-drums \
  --dry-run
```

Use `--cell A0B0`, `A1B0`, `A0B1`, or `A1B1` to inspect a profile. A dry run does not load models, create an edit, or call a paid API. Actual inference requires the setup described in `docs/setup.md` and a Slurm GPU allocation.
