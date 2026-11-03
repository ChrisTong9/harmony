"""Run audio editing or inspect the listening dialogue."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from audio_edit.agents.music_edit_agent import MusicEditAgent
from audio_edit.analyzers.edit_listening import listen_for_edit
from audio_edit.analyzers.qwen_audio_analyzer import QwenAudioAnalyzer
from audio_edit.llm.client import LLMClient
from audio_edit.models import AudioEditContext
from audio_edit.scripts.registry import ScriptRegistry
from audio_edit.scripts.runner import ScriptRunner
from audio_edit.utils.audio_input import validate_audio_input
from audio_edit.utils.execution_outcome import execution_exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--listen-only",
        action="store_true",
        help="Save questions, answers and review without editing audio",
    )
    args = parser.parse_args()
    ctx = AudioEditContext(args.instruction, input_audio=str(validate_audio_input(args.input)))
    registry, llm = ScriptRegistry(), LLMClient()
    agent = MusicEditAgent(ScriptRunner(registry), registry, llm)
    try:
        if args.listen_only:
            invocation = agent.choose_script(ctx.user_prompt)
            ctx.selected_script = invocation.script_name
            ctx.params.update(invocation.params)
            listen_for_edit(ctx, llm, QwenAudioAnalyzer(), operation=invocation.script_name)
            ctx.execution_status = "listening_ready"
        else:
            ctx = agent.execute(ctx)
    except Exception as exc:
        ctx.execution_status = "error"
        ctx.execution_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(asdict(ctx), ensure_ascii=False, indent=2) + "\n")
    print(f"{ctx.execution_status}: report={args.report.resolve()} output={ctx.output_audio}")
    return execution_exit_code(ctx)


if __name__ == "__main__":
    raise SystemExit(main())
