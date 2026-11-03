# Method and code map

Harmony supports addition, removal, replacement, and volume adjustment through one shared implementation. A user provides an audio file and a natural-language instruction. The system resolves the target and scope, executes the requested operation, and mixes the result while retaining surrounding content.

## Edit-driven collaborative planning

The LLM planner asks the audio-language model targeted questions about the edit. The agent can request additional evidence from tools, including speech alignment, source separation, activity measurements, and rhythm analysis. Tool use is selected for the current request; a tool is called only when its evidence is needed.

Evidence is represented as source descriptions, temporal references, and operation constraints. Explicit file times, selected events, and particular participants have different requirements and are resolved by the corresponding planning modules.

## Evidence-guided execution

Applicable editing branches inspect the first separated or generated material and may make bounded corrections before mixing. These can revise a source query, material generation, or execution parameters. Correction limits are specific to the branch; there is no single universal correction count.

A common basic execution route is available when the primary route cannot finish. It compiles the original instruction into a bounded edit plan without reusing the failed route's listening evidence. The runtime retains the execution status and any delivery error in the local report.

## Task settings

The public interface runs the complete Harmony pipeline. `harmony/configs/profiles.json` contains shared model settings and task defaults. All four tasks execute the same `audio_edit/` package, and the agent selects relevant tools and corrections from the current instruction and evidence.

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

Provide an input recording and instruction to run a new edit. Tool availability and resource requirements are described in the setup guide.

## Reading the implementation

Start with `harmony/cli.py` and `harmony/runtime/driver.py` for configuration and resource setup. Follow `audio_edit/run_edit.py` into `agents/music_edit_agent.py` for dispatch, then the selected `user_scripts/` module for task orchestration. Specialists create plans; runners execute them; tools perform audio operations.

`analyzers/edit_listening.py` coordinates evidence requests. Its schemas and synthetic demonstrations are separated into the adjacent listening modules. `plans/temporal.py` and `plans/add_intent.py` define the timeline contracts shared by planning and mixing. The detailed evidence and correction branches are intentionally explicit; they should be read with their corresponding plan types.
