# Method and code map

Harmony supports addition, removal, replacement, and volume adjustment through a shared planning-and-execution interface. The public wrapper selects a task/configuration profile; the underlying task implementation interprets the natural-language request.

## Collaborative planning

The planner and audio-language model exchange targeted observations. With A enabled, the agent may request additional evidence such as speech timing, event grouping, source separation evidence, rhythm measurements, or source/activity measurements. Availability does not imply that every tool is called on every input.

Relevant modules under a selected backend's `audio_edit/` tree:

- `analyzers/`: listening and measurement evidence.
- `plans/`: source, temporal, and operation constraints.
- `agents/music_edit_agent.py`: agent coordination.
- `run_edit.py`: editing entry point.

## Evidence-guided execution

With B enabled, source-specific execution branches can inspect initial separated/generated material and make bounded changes. These checks can revise extraction, generation, or the execution plan before mixing. Bounds are stage-dependent; they should not be described as a single universal correction call.

- `runners/`: editing and mixing execution.
- `tools/`: model subprocess interfaces and measurement tools.
- `harmony/runtime/service/`: adapter to the common execution service.
- `reproducibility/correction_budgets.json`: observed/configured correction limits.

The common basic execution service is enabled in all four configuration profiles. It is separate from the A/B mechanism switches.

## Runtime code layout

| Backend | Role |
|---|---|
| `harmony/backends/addition` | Addition profiles |
| `harmony/backends/removal` | A1B1 removal primary profile |
| `harmony/backends/removal_control` | Other removal profiles |
| `harmony/backends/replacement` | A1B1 replacement primary profile |
| `harmony/backends/replacement_control` | Other replacement profiles |
| `harmony/backends/volume` | Volume profiles |
| `harmony/backends/execution` | Common basic execution service |

These are frozen source trees, not independent published methods. Their duplication preserves task-specific source identity. The 20 additional historical snapshots in `reproducibility/history/` document earlier result-producing revisions and are not selected by benchmark ID during new-input inference.

`configs/profiles.json` is the executable mapping. `reproducibility/SOURCES.json` records the source identities and old-to-new directory mapping. Do not treat all four configuration results as a new, controlled measurement of this reorganized checkout.
