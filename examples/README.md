# Example requests

The following requests illustrate the four supported operations. Supply your own audio containing the referenced sounds; the planner selects the primary operation from your instruction. No input recordings are bundled with this code branch.

| Task | Example instruction |
|---|---|
| Add | Add a short bell sound at 2 seconds. |
| Remove | Remove the background noise while keeping the speech. |
| Replace | Replace the ending laughter with applause. |
| Volume | Make the door slamming sound louder. |

[`requests.json`](requests.json) contains the same interface examples. `/path/to/input.wav` is a placeholder for your own recording.

```bash
python -m harmony run \
  --input /path/to/input.wav \
  --instruction "Add a short bell sound at 2 seconds." \
  --output-dir outputs/example-add
```

Complete the [model setup](../docs/setup.md) before running. Add `--dry-run` to preview a command without loading models, creating an edit, or calling an API.

Use one primary operation per request. Replace the example instruction with any row above; no task argument is needed. Optionally add `--seed 123` to set the Stable Audio generation seed.

For audio comparisons, visit the [project demo](https://christong9.github.io/harmony/).
