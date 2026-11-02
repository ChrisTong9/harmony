from audio_edit.models import AudioEditContext
from audio_edit.scripts.invocation import ScriptInvocation
from audio_edit.llm.client import LLMClient
from pathlib import Path
import traceback
from audio_edit.plans.errors import UnresolvedEditError
from audio_edit.utils.execution_outcome import annotate_outcome


class MusicEditAgent:
    def __init__(self, runner, registry, llm: LLMClient):
        self.runner = runner
        self.registry = registry
        self.llm = llm

    def choose_script(self, user_prompt: str) -> ScriptInvocation:
        script_catalog = self.registry.list_scripts()
        decision = self.llm.choose_script(user_prompt=user_prompt, script_catalog=script_catalog)

        return ScriptInvocation(
            script_name=decision.script_name,
            params=decision.params,
        )

    def execute(self, ctx: AudioEditContext) -> AudioEditContext:
        original_audio = ctx.input_audio
        try:
            invocation = self.choose_script(ctx.user_prompt)

            result = self.runner.run(ctx, invocation)
            if not result.output_audio or not Path(result.output_audio).is_file():
                raise RuntimeError("Editing did not produce an output file")
            reference = ctx.input_audio if ctx.input_video else original_audio
            return annotate_outcome(result, reference)
        except Exception as exc:
            if not original_audio or not Path(original_audio).is_file():
                raise
            import os

            if os.environ.get("HARMONY_BASIC_EXECUTION", "0") == "1":
                from audio_edit.tools.basic_bridge import recover

                return recover(ctx, original_audio, llm=self.llm)
            traceback.print_exc()
            ctx.output_audio = original_audio
            ctx.fallback_used = True
            ctx.execution_status = (
                "unresolved" if isinstance(exc, UnresolvedEditError) else "fallback"
            )
            if ctx.params.get("best_effort"):
                ctx.params["failure_kind"] = ctx.execution_status
                ctx.execution_status = "fallback"
            ctx.execution_error = f"{type(exc).__name__}: {exc}"
            ctx.warnings.append({"code": "fallback_original", "reason": ctx.execution_error})
            print("[agent] Returning original input as fallback:", ctx.execution_error)
            return ctx
