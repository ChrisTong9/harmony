# Method and code map

Harmony supports addition, removal, replacement, and volume adjustment through one shared implementation. A user provides an audio file and a natural-language instruction. The system resolves the target and scope, executes the requested operation, and mixes the result while retaining surrounding content.

## Edit-driven collaborative planning

The LLM planner asks the audio-language model targeted questions about the edit. With mechanism A enabled, the agent can request additional evidence from tools, including speech alignment, source separation, activity measurements, and rhythm analysis. Tool use is selected for the current request; enabling A does not require every tool to run on every input.

Evidence is represented as source descriptions, temporal references, and operation constraints. Explicit file times, selected events, and particular participants have different requirements and are resolved by the corresponding planning modules.

## Evidence-guided execution

With mechanism B enabled, applicable editing branches inspect the first separated or generated material and may make bounded corrections before mixing. These can revise a source query, material generation, or execution parameters. Correction limits are specific to the branch; there is no single universal correction count.

A common basic execution route is available across all four A/B configurations when the primary route cannot finish. It compiles the original instruction into a bounded edit plan without reusing the failed route's listening evidence. The runtime retains the execution status and any delivery error in the local report.

## Configuration

| Configuration | Collaborative tool evidence (A) | Execution correction (B) |
|---|---|---|
| `A0B0` | Off | Off |
| `A1B0` | On | Off |
| `A0B1` | Off | On |
| `A1B1` | On | On |

Select a configuration with `--cell`; `A1B1` is the default. `configs/profiles.json` supplies the task settings. All configurations execute the same `audio_edit/` package.

## Code map

| Entry point | Responsibility |
|---|---|
| `harmony/cli.py` | Public `profiles` and `run` commands |
| `harmony/runtime/driver.py` | Configuration, resource paths, and GPU allocation checks |
| `audio_edit/run_edit.py` | Editing entry point |
| `audio_edit/agents/` | Coordination and operation dispatch |
| `audio_edit/analyzers/` | Listening, tool evidence, and evidence binding |
| `audio_edit/plans/` | Source, temporal, and operation constraints |
| `audio_edit/specialists/` | Operation-specific planning |
| `audio_edit/runners/` | Execution and composition |
| `audio_edit/tools/` | Audio tools, subprocess interfaces, and basic execution |
| `harmony/runtime/service/` | Isolated basic execution adapter |

The unified code is intended for new inputs. CPU checks of configuration and command construction do not establish fresh GPU editing results or guarantee identical outputs from stochastic model calls.
