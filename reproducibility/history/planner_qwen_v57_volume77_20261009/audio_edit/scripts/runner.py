import importlib
from audio_edit.scripts.invocation import ScriptInvocation
from audio_edit.models import AudioEditContext


class ScriptRunner:
    def __init__(self, registry):
        self.registry = registry

    def run(self, ctx: AudioEditContext, invocation: ScriptInvocation) -> AudioEditContext:
        script_info = self.registry.get(invocation.script_name)
        module_name = script_info["module"]

        module = importlib.import_module(module_name)

        if not hasattr(module, "run"):
            raise ValueError(f"Script {invocation.script_name} has no run(ctx, params) function")

        ctx.selected_script = invocation.script_name


        previous = ctx.params


        routed_params = {key:value for key,value in invocation.params.items() if key != 'perception_mode'}
        params = {**previous, **routed_params}
        if 'best_effort' in previous:
            params['best_effort'] = previous['best_effort']
        params['dispatch_audit'] = dict(script=invocation.script_name,
            caller_best_effort=previous.get('best_effort'),
            routed_best_effort=invocation.params.get('best_effort'),
            effective_best_effort=params.get('best_effort'),
            policy='Caller execution policy survives routing; edit arguments come from invocation')
        ctx.params = params
        return module.run(ctx, params)
