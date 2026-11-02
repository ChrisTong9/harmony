# Example requests

The following commands illustrate the four supported operations. Supply your own audio containing the referenced sounds; no input recordings are bundled with this code branch.

| Task | Example instruction |
|---|---|
| Add | Add a short bell sound at 2 seconds. |
| Remove | Remove the background noise while keeping the speech. |
| Replace | Replace the ending laughter with applause. |
| Volume | Make the door slamming sound louder. |

[`requests.json`](requests.json) contains the same interface examples. `/path/to/input.wav` is a placeholder for your own recording.

```bash
python -m harmony run \
  --task add \
  --input /path/to/input.wav \
  --instruction "Add a short bell sound at 2 seconds." \
  --output-dir outputs/example-add \
  --dry-run
```

Use `--cell A0B0`, `A1B0`, `A0B1`, or `A1B1` to select the mechanism settings. A dry run does not load models, create an edit, or call an API. Actual inference requires the [model setup](../docs/setup.md) and a Slurm GPU allocation.

For audio comparisons, visit the [project demo](https://christong9.github.io/harmony/). The benchmark sample list is [`data/mmae_420.csv`](../data/mmae_420.csv).
