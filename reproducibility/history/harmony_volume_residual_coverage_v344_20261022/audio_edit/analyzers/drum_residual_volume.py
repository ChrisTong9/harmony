"""One source-preserving correction for clearly unextracted named drums.

Keep the first separated component at its requested gain. Extract once from its
complement, then apply that same gain only to the newly extracted component.
"""
from pathlib import Path
from copy import deepcopy
import base64, hashlib, json, os, tempfile

SYSTEM = 'Describe only audible facts. Do not infer an editing intention or grade a task.'
QUESTION = 'Describe the audible sound sources and their relative prominence across this recording. Identify speech, singing, sustained tonal sounds, impulsive or rhythmic sounds, and other sources only if heard. State when a source is faint or uncertain. Do not infer missing sources from an unavailable original recording.'
PROMPT = '''Decide whether one bounded extraction from the first separation's complementary residual is supported by these actual first neutral observations.
The original request names drums. The first component has already received the requested gain. The residual is the complement BEFORE applying that component's gain; it is not the attenuated final mix. No score or alternative render is provided.
Return supported=true only if the component observation clearly describes the requested drums without a protected vocal or melodic source, AND the residual observation clearly identifies remaining drums/percussion belonging to that target. A generic rhythm, bassline, electronic effect, or merely possible/faint drum does not establish the latter. Do not infer missing drums from the request. If either fact is uncertain or contradictory, return false. No new query, gain, times, instrument, or observation is requested.
Quote literal words from both observations supporting your decision. Return only the required JSON.'''

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

def candidate(ctx, plan):
    d = ctx.params.get('edit_listening_dialogue', {})
    targets = plan.effective_target_sounds()
    return (targets in (['drums'], ['drum']) and not plan.changes
        and plan.apply_to == 'target_source' and -24 <= plan.gain_db < 0
        and plan.gain_profile == 'static' and not plan.placements
        and not plan.temporal_schema_error and plan.temporal_status == 'not_requested'
        and (plan.temporal is None or plan.temporal.kind == 'unspecified')
        and d.get('operation') == 'volume_change' and d.get('original_instruction') == ctx.user_prompt
        and d.get('status') == 'not_required'
        and d.get('selection') == dict(target='named_source', scope='whole_file', occurrence='unselected')
        and not d.get('participant_selection') and not ctx.params.get('drum_residual_volume'))

def materials(ctx, plan, directory):
    import numpy as np
    import soundfile as sf
    calls = ctx.params.get('separation_calls', [])
    if ctx.params.get('volume_backend') != 'sam_source_gain' or len(calls) != 1:
        raise ValueError('Residual completion requires one first native source-gain execution')
    request, result = read(calls[0]['request']), read(calls[0]['response'])
    if (request['input_audio'] != ctx.input_audio or request['target_sounds'] != plan.effective_target_sounds()
            or request['gain_db'] != plan.gain_db or request['gain_profile'] != 'static' or request['placements']):
        raise ValueError('First source execution does not match the requested plan')
    audit = next(a for a in result['warnings'] if a.get('code') == 'source_volume_execution')
    if (audit['actual_delta_scale'] != 1 or audit['normalization_applied'] is not False
            or audit['output_policy'] != 'preserve_requested_gain_float_wav'):
        raise ValueError('First edit has no invertible unclipped gain execution')
    x, sr = sf.read(ctx.input_audio, dtype='float32', always_2d=True)
    y, yr = sf.read(ctx.output_audio, dtype='float32', always_2d=True)
    native, nr = sf.read(result['output_audio'], dtype='float32', always_2d=True)
    if sr != yr or sr != nr or x.shape != y.shape or not np.array_equal(native, y):
        raise ValueError('First output and native execution do not share exact samples')
    if not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError('Nonfinite first audio')
    factor = 10 ** (plan.gain_db / 20) - 1
    component = (y.astype(np.float64) - x.astype(np.float64)) / factor
    residual = (x.astype(np.float64) - component).astype(np.float32)
    paths = {}
    for label, data in [('component', component.astype(np.float32)), ('residual', residual)]:
        path = directory / (label + '.wav'); sf.write(path, data, sr, subtype='FLOAT'); paths[label] = str(path)
    return paths

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
    folder = root() / 'drum_residual_facts'; saved = folder / (key(identity) + '.json')
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

def binding_request(ctx, plan, facts):
    return dict(original_instruction=ctx.user_prompt, named_target=plan.effective_target_sounds()[0],
        first_component_observation=facts['component']['text'], first_residual_observation=facts['residual']['text'])

def payload_for(request, model):
    props = dict(supported=dict(type='boolean'), component_fact_quote=dict(type='string'),
                 residual_fact_quote=dict(type='string'), reason=dict(type='string'))
    return dict(model=model, reasoning_effort='low', messages=[dict(role='system',content=PROMPT),
        dict(role='user',content=json.dumps(request,ensure_ascii=False))],
        response_format=dict(type='json_schema',json_schema=dict(name='drum_residual_completion',strict=True,
        schema=dict(type='object',additionalProperties=False,properties=props,required=list(props)))))

def first_binding(ctx, plan, facts):
    from audio_edit.llm.client import LLMClient
    client = LLMClient(); request = binding_request(ctx, plan, facts); payload = payload_for(request, client.model)
    identity = dict(payload=payload, original_audio_sha256=sha(ctx.input_audio), first_output_sha256=sha(ctx.output_audio),
        component_sample_sha256=facts['component']['identity']['audio_sample_sha256'],
        residual_sample_sha256=facts['residual']['identity']['audio_sample_sha256'])
    path = root() / 'drum_residual_bindings' / (key(identity) + '.json')
    if path.exists():
        result = read(path)
        if result['identity'] != identity: raise ValueError('First binding cache mismatch')
        return result
    if path.with_suffix('.claim.json').exists(): raise ValueError('First binding already attempted')
    write(path.with_suffix('.claim.json'), identity)
    response = client.client.with_options(max_retries=0).chat.completions.create(**payload)
    write(path.with_suffix('.raw.json'), response.model_dump())
    if response.choices[0].finish_reason != 'stop': raise ValueError('Incomplete first binding')
    answer = json.loads(response.choices[0].message.content)
    expected = {'supported','component_fact_quote','residual_fact_quote','reason'}
    if set(answer) != expected or type(answer['supported']) is not bool: raise ValueError('Invalid first binding')
    supported = answer['supported'] and all(isinstance(answer[k],str) and answer[k].strip() and answer[k] in request[f]
        for k,f in [('component_fact_quote','first_component_observation'),('residual_fact_quote','first_residual_observation')])
    result = dict(identity=identity, response=answer, supported=supported, first_decision=True)
    write(path,result); return result

def apply(ctx, plan):
    if not candidate(ctx, plan): return ctx
    folder = Path('audio_edit/generated').resolve(); folder.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='drum-residual-', dir=folder)).resolve()
    before = ctx.output_audio
    try:
        paths = materials(ctx, plan, directory)
        supplied = ctx.params.get('drum_residual_first_facts', {})
        facts = {role:first_fact(path, supplied.get(role)) for role,path in paths.items()}
        decision = first_binding(ctx, plan, facts)
    except Exception as exc:
        ctx.params['drum_residual_volume'] = dict(status='first_completion_evidence_unavailable', error=str(exc), adopted=False)
        return ctx
    audit = dict(status='first_completion_not_supported', decision=decision, facts=facts, first_output=before,
        first_output_sha256=sha(before), materials=paths, adopted=False, maximum_extra_SAM=1)
    ctx.params['drum_residual_volume'] = audit
    if not decision['supported']: return ctx
    from audio_edit.runners.samaudio_runner import SAMAudioRunner
    import numpy as np
    import soundfile as sf
    runner = SAMAudioRunner()
    corrected = runner.run_source_volume(original_path=paths['residual'], target_sounds=plan.effective_target_sounds(),
        gain_db=plan.gain_db, placements=None)
    second = next(a for a in runner.last_warnings if a.get('code') == 'source_volume_execution')
    if (second['actual_delta_scale'] != 1 or second['normalization_applied'] is not False
            or second['output_policy'] != 'preserve_requested_gain_float_wav'):
        raise ValueError('Residual source gain was not preserved exactly')
    y, sr = sf.read(before, dtype='float32', always_2d=True)
    r, rr = sf.read(paths['residual'], dtype='float32', always_2d=True)
    z, zr = sf.read(corrected, dtype='float32', always_2d=True)
    if sr != rr or sr != zr or y.shape != r.shape or y.shape != z.shape: raise ValueError('Residual correction changed the sample grid')
    final = (y.astype(np.float64) + z.astype(np.float64) - r.astype(np.float64)).astype(np.float32)
    if not np.isfinite(final).all(): raise ValueError('Residual correction produced nonfinite samples')
    output = directory / 'corrected.wav'; sf.write(output, final, sr, subtype='FLOAT')
    audit.update(status='first_residual_completion_executed', adopted=True, second_call=deepcopy(runner.last_call),
        second_warnings=deepcopy(runner.last_warnings), output=str(output), output_sha256=sha(output),
        formula='first_output + corrected_residual - first_residual', requested_gain_db=plan.gain_db,
        query_unchanged=True, no_master_normalization=True)
    ctx.params.setdefault('separation_calls', []).append(deepcopy(runner.last_call))
    ctx.warnings.extend(runner.last_warnings); ctx.output_audio = str(output)
    return ctx
