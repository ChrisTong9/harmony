"""Complete an audible old whole-music residual once, retaining the first new layer."""
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict
import base64, hashlib, json, os, tempfile
SYSTEM = 'Describe only audible facts. Do not infer an editing intention or grade a task.'
QUESTION = 'Describe the audible sound sources and their relative prominence across this recording. Identify speech, singing, sustained tonal sounds, impulsive or rhythmic sounds, and other sources only if heard. State when a source is faint or uncertain. Do not infer missing sources from an unavailable original recording.'
PROMPT = """Check whether ONE completion of an existing whole-file music replacement is supported by first observations. The first generated replacement layer will be kept exactly. The residual is the saved result of removing the old source, BEFORE adding the generated layer. Only the original separation query, scope and settings may be applied once to that residual. No score or alternative render is available.
Support only if the original request replaces the complete old instrumental/background music class, the original observation clearly identifies that old music, and the residual clearly retains that same old musical source. The generated-layer observation must clearly identify the requested new instrument without old-instrument contamination. A selected instrument among protected instruments, vague/possible/faint residual identity, or contradictory facts does not support completion. Do not infer inaudible targets, rewrite a query, choose times or change the generated material. Protected nonmusic sounds must still be retained by separation.
For each quote, copy one contiguous exact substring from its corresponding observation. Return the required JSON only."""
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def key(value): return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(value, f, ensure_ascii=False, indent=2)
def root(): return Path(os.environ.get('AUDIO_EDIT_EVIDENCE_DIR', 'audio_edit/evidence')).resolve()
def sample_digest(path):
    from audio_edit.analyzers.replacement_material_guard import sample_digest as digest
    return digest(path)

def first_fact(path, supplied=None):
    identity = dict(protocol='neutral_component_inventory_768_v1', audio_sample_sha256=sample_digest(path))
    if supplied is not None:
        artifact = Path(supplied['artifact_path'])
        if sha(artifact) != supplied['artifact_sha256']: raise ValueError('First fact artifact changed')
        value = read(artifact); original = value['input']['path']
        if sha(original) != value['input']['sha256'] or sample_digest(original) != identity['audio_sample_sha256']:
            raise ValueError('First fact belongs to different decoded audio')
        if value.get('first_observation') is not True or value['raw_response']['choices'][0]['finish_reason'] != 'stop':
            raise ValueError('Incomplete first material observation')
        if value['request_sha256'] != sha(value['input']['request_path']): raise ValueError('First fact request changed')
        request = read(value['input']['request_path'])
        if request['system'] != SYSTEM or request['user_text'] != QUESTION or request['max_tokens'] != 768:
            raise ValueError('Different first fact protocol')
        return dict(identity=identity, text=value['response'], provenance=deepcopy(supplied), imported_first_fact=True)
    folder = root() / 'replace_music_residual_facts'; saved = folder / (key(identity) + '.json')
    if saved.exists():
        fact = read(saved)
        if fact['identity'] != identity: raise ValueError('First fact cache mismatch')
        return fact
    claim = saved.with_suffix('.claim.json')
    if claim.exists(): raise ValueError('A first observation was already attempted')
    from audio_edit.analyzers.residual_source_query import local_analyzer
    analyzer = local_analyzer()
    if analyzer is None: raise ValueError('Local allocated analyzer unavailable')
    write(claim, identity)
    payload = dict(model=analyzer.model, temperature=0, max_tokens=768, messages=[
        dict(role='system', content=SYSTEM), dict(role='user', content=[
        dict(type='audio_url', audio_url=dict(url='data:audio/wav;base64,'+base64.b64encode(Path(path).read_bytes()).decode())),
        dict(type='text', text=QUESTION)])])
    response = analyzer.client.with_options(max_retries=0).chat.completions.create(**payload)
    raw = response.model_dump(); write(saved.with_suffix('.raw.json'), raw)
    text = response.choices[0].message.content
    if response.choices[0].finish_reason != 'stop' or not isinstance(text, str) or not text.strip():
        raise ValueError('Incomplete first material observation')
    fact = dict(identity=identity, text=text, first_observation=True, audio_path=path, audio_sha256=sha(path))
    write(saved, fact); return fact

def candidate(ctx, plan, original, instruction):
    d=ctx.params.get('edit_listening_dialogue',{});p=plan.remove_plan
    return (p.backend=='samaudio' and p.mode=='text' and p.result_mode=='residual'
        and p.text_prompt.strip().lower() in {'music','background music','instrumental music','instrumental background music'}
        and p.time_span is None and p.temporal_status=='resolved' and not p.temporal_schema_error
        and p.temporal is not None and p.temporal.kind=='unspecified' and len(p.placements)==1
        and p.placements[0].start_time==0 and p.placements[0].end_time==d.get('media_duration')
        and d.get('status')=='not_required' and d.get('operation')=='replace'
        and d.get('original_instruction')==instruction and plan.user_prompt==instruction
        and d.get('selection')==dict(target='named_source',scope='whole_file',occurrence='unselected')
        and not d.get('participant_selection') and not ctx.input_video
        and len(ctx.params.get('separation_calls',[]))==1
        and not any(ctx.params.get(k) for k in ['replacement_residual_correction','replacement_music_backing_execution','replace_music_residual']))

def evidence(ctx, plan, original):
    import numpy as np
    import soundfile as sf
    call=ctx.params['separation_calls'][0];req=read(call['request']);res=read(call['response'])
    if res!=call['result'] or req['input_audio']!=original:raise ValueError('First separation provenance mismatch')
    if req['plan']['text_prompt']!=plan.remove_plan.text_prompt or req['plan']['result_mode']!='residual' or req['plan']['time_span'] is not None:
        raise ValueError('First separation query/scope mismatch')
    residual=res['output_audio']
    original_fact=ctx.params['replacement_source_facts']
    if not original_fact['first_observation'] or original_fact['audio_sha256']!=sha(original):raise ValueError('First original fact mismatch')
    generated=list(ctx.params.get('replacement_material_first_facts',{}).values())
    if len(generated)!=1 or not generated[0]['first_observation']:raise ValueError('Exactly one first material observation required')
    audits=ctx.params.get('mix_audit',[]);timeline=[a for a in audits if a.get('code')=='timeline_placement']
    mapping=[a for a in audits if a.get('code')=='original_timeline_mapping']
    if len(timeline)!=1 or len(mapping)!=1:raise ValueError('Single identity mixing grid required')
    t=timeline[0];m=mapping[0]
    if (t['timeline_op']!='overlay' or t['headroom_scale']!=1 or t['original_gain']!=1
        or t['headroom_policy']!='no_master_normalization' or t['actual_start']!=0
        or t['actual_end']!=plan.remove_plan.placements[0].end_time
        or m['original_samples']!=m['output_samples']
        or m['segments']!=[dict(original_start=0,original_end=m['original_samples'],output_start=0)]):
        raise ValueError('First mix is not an unscaled identity residual plus generated layer')
    arrays=[sf.read(p,dtype='float32',always_2d=True) for p in [original,residual,ctx.output_audio]]
    if len({(x.shape,sr) for x,sr in arrays})!=1 or not all(np.isfinite(x).all()for x,sr in arrays):raise ValueError('Grid mismatch')
    fact=first_fact(residual,ctx.params.get('replace_music_residual_first_fact'))
    return dict(original=original_fact,generated=generated[0],residual=fact,path=residual)

def binding_request(ctx, plan, instruction, facts):
    return dict(original_instruction=instruction,existing_old_query=plan.remove_plan.text_prompt,new_target=plan.target_sound,
        first_original_observation=facts['original']['text'],first_residual_observation=facts['residual']['text'],
        first_generated_observation=facts['generated']['text'])

def payload_for(request,model):
    props=dict(supported=dict(type='boolean'),original_quote=dict(type='string'),residual_quote=dict(type='string'),generated_quote=dict(type='string'),reason=dict(type='string'))
    return dict(model=model,reasoning_effort='low',messages=[dict(role='system',content=PROMPT),dict(role='user',content=json.dumps(request,ensure_ascii=False))],
        response_format=dict(type='json_schema',json_schema=dict(name='replace_music_residual',strict=True,
            schema=dict(type='object',additionalProperties=False,properties=props,required=list(props)))))

def first_binding(ctx,plan,original,instruction,facts):
    from audio_edit.llm.client import LLMClient
    client=LLMClient();request=binding_request(ctx,plan,instruction,facts);payload=payload_for(request,client.model)
    identity=dict(payload=payload,original_sha256=sha(original),first_output_sha256=sha(ctx.output_audio),residual_sha256=sha(facts['path']))
    path=root()/'replace_music_residual_bindings'/(key(identity)+'.json')
    if path.exists():
        value=read(path)
        if value['identity']!=identity:raise ValueError('First decision cache mismatch')
        return value
    if path.with_suffix('.claim.json').exists():raise ValueError('First decision already attempted')
    write(path.with_suffix('.claim.json'),identity)
    response=client.client.with_options(max_retries=0).chat.completions.create(**payload)
    write(path.with_suffix('.raw.json'),response.model_dump())
    if response.choices[0].finish_reason!='stop':raise ValueError('Incomplete first decision')
    answer=json.loads(response.choices[0].message.content)
    if set(answer)!={'supported','original_quote','residual_quote','generated_quote','reason'} or type(answer['supported'])is not bool:raise ValueError('Invalid first decision')
    valid=answer['supported'] and all(isinstance(answer[k],str) and answer[k].strip() and answer[k]in request[f]
        for k,f in [('original_quote','first_original_observation'),('residual_quote','first_residual_observation'),('generated_quote','first_generated_observation')])
    value=dict(identity=identity,response=answer,supported=valid,first_decision=True);write(path,value);return value

def apply(ctx,plan,original,instruction):
    if not candidate(ctx,plan,original,instruction):return ctx
    before=ctx.output_audio
    try:
        facts=evidence(ctx,plan,original);decision=first_binding(ctx,plan,original,instruction,facts)
    except Exception as exc:
        ctx.params['replace_music_residual']=dict(status='first_evidence_unavailable',error=str(exc),adopted=False);return ctx
    audit=dict(status='first_completion_not_supported',decision=decision,facts=facts,adopted=False,first_output=before,first_output_sha256=sha(before))
    ctx.params['replace_music_residual']=audit
    if not decision['supported']:return ctx
    from audio_edit.models import AudioEditContext
    from audio_edit.runners.samaudio_runner import SAMAudioRunner
    import numpy as np
    import soundfile as sf
    second=AudioEditContext(instruction,input_audio=facts['path'])
    second=SAMAudioRunner().run_remove(second,deepcopy(plan.remove_plan))
    if second.params.get('removal_execution',{}).get('backend')!='sam_audio' or len(second.params.get('separation_calls',[]))!=1:
        raise ValueError('One original-query SAM execution required')
    call=second.params['separation_calls'][0]
    if not all(a.get('delta_scale')==1 for a in call['result']['warnings'] if a.get('code')=='removal_alignment'):
        raise ValueError('Residual removal rescaled input')
    y,sr=sf.read(before,dtype='float32',always_2d=True);r,rr=sf.read(facts['path'],dtype='float32',always_2d=True)
    z,zr=sf.read(second.output_audio,dtype='float32',always_2d=True)
    if sr!=rr or sr!=zr or y.shape!=r.shape or y.shape!=z.shape:raise ValueError('Correction changed grid')
    final=(y.astype(np.float64)+z.astype(np.float64)-r.astype(np.float64)).astype(np.float32)
    if not np.isfinite(final).all():raise ValueError('Nonfinite correction')
    folder=Path('audio_edit/generated').resolve();folder.mkdir(parents=True,exist_ok=True)
    output=Path(tempfile.mkdtemp(prefix='music-residual-',dir=folder))/'corrected.wav';sf.write(output,final,sr,subtype='FLOAT')
    audit.update(status='first_residual_completion_executed',adopted=True,second_call=deepcopy(call),formula='first_output + corrected_residual - first_residual',query_unchanged=True,first_generated_layer_unchanged=True,output_sha256=sha(output))
    ctx.params['separation_calls'].append(deepcopy(call));ctx.warnings.extend(second.warnings);ctx.output_audio=str(output)
    return ctx
