"""One same-query continuation for explicit all-singing removal and first residue.

Checkpoint-only: requires a bound first neutral residual observation. No listening,
new source identity, time inference, automatic retries or repeated decisions.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib,re
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def eligible(ctx,plan,fact):
    d=ctx.params.get('edit_listening_dialogue',{})
    if (ctx.params.get('first_singing_completion') or ctx.input_video
        or d.get('selection')!={'target':'named_source','scope':'whole_file','occurrence':'unselected'}
        or d.get('participant_selection') or d.get('placement_windows')
        or d.get('original_instruction')!=ctx.user_prompt
        or plan.mode!='text' or plan.result_mode!='residual' or plan.time_span
        or not re.fullmatch(r'singing(?: voices?| vocals?)?',plan.text_prompt.strip(),re.I)
        or plan.use_video_prompt):return False
    if not isinstance(fact,dict)or fact.get('first_observation')is not True or fact.get('finish_reason')!='stop':return False
    audio=fact.get('input',{})
    if (audio.get('material_type')!='saved_first_native_SAM_residual_unmodified'
        or audio.get('original_sha256')!=sha(ctx.input_audio)
        or audio.get('sha256')!=sha(ctx.output_audio) or sha(audio['path'])!=audio['sha256']):return False
    paragraph=next((p for p in fact.get('response','').splitlines()if '**Singing:**'in p),'')
    return bool(re.search(r'\bvoice (?:begins to sing|is singing)\b',paragraph,re.I)
        and not re.search(r'\b(no|not|uncertain|possibly|perhaps|might|could|if|absent)\b',paragraph,re.I))

def apply(ctx,plan,fact):
    if not eligible(ctx,plan,fact):return ctx
    from audio_edit.runners.samaudio_runner import SAMAudioRunner
    first=ctx.output_audio;continuation=deepcopy(ctx);continuation.input_audio=first;continuation.output_audio=None;continuation.params={}
    follow=deepcopy(plan);follow.removal_evidence=None
    result=SAMAudioRunner().run_remove(continuation,follow)
    ctx.output_audio=result.output_audio;ctx.warnings.extend(result.warnings)
    ctx.params['first_singing_completion']=dict(first_residual=first,first_residual_sha256=sha(first),
        first_observation=fact,query_unchanged=True,scope_unchanged=True,new_SAM_backend_requests=1,
        plan=asdict(follow),separation_calls=result.params.get('separation_calls',[]))
    return ctx
