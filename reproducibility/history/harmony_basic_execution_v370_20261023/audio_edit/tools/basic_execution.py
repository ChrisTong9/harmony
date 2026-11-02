"""Common bounded execution route from the original request alone.

It does not observe audio, revise a previous plan, inspect errors, or inspect scores.
The same route is available independently of both experimental A/B switches.
"""
from pathlib import Path
from copy import deepcopy
import hashlib, json, math, os, subprocess, sys, tempfile
import numpy as np
import soundfile as sf

PROMPT = '''Compile the original user request into a simple executable audio edit.
You have only the request and file duration, no listening observations. Return one
bounded first plan. Never ask questions, request tools, or claim to have heard audio.
Use 1-4 actions. remove deletes the named source; retain keeps the named source;
replace deletes target and adds material; add adds material; volume changes target.
Extract/isolate/keep-only requests normally use retain for the desired sound.
Each target/material is a short positive acoustic noun phrase, not an instruction.
Keep requested voice gender, activity and identity adjectives in the target phrase.
Use source_scope=mix only if the request explicitly changes all audio, otherwise named.
Specify numerical start/end in seconds, with end > start. Use the whole file for
unqualified source removal, retention or volume. Explicit file times are literal.
For a source at a given time with no end, use a 3-second window from that time,
clipped to the file; mark timing=estimated. Beginning/end with no duration uses
the first/last min(3 seconds, file duration). If an event-relative time cannot be
known, make a conservative approximate placement from the request, mark estimated;
never present estimates as observed event boundaries. Do not refuse execution.
For addition at the very end, append; at the beginning overlay unless insertion is
explicit. Continuous background sound covers its requested range; a discrete sound
defaults to 1 second. Repeat count is the requested count, or 1. For unspecified
repeated events use 3. Repeats are evenly spaced within the declared interval.
Ordinary add/replace uses overlay. Only add may append or insert and extend time;
an append/insert must be the final action, with at most one such action.
Volume defaults to -6 dB for decrease, +6 dB for increase. Retain explicit gains.
For gradual volume changes set gain_start_db and gain_end_db differently; otherwise
set both equal. For non-volume actions set them to 0. Background material uses
background=true; other additions false. Material must describe the requested new
sound alone. Do not invent content, evaluation claims or factual listening results.
Use empty target only for add; empty material unless add/replace. All fields required.
'''

ACTION_FIELDS = {
    'operation': {'type':'string','enum':['remove','retain','add','replace','volume']},
    'target': {'type':'string'}, 'material': {'type':'string'},
    'source_scope': {'type':'string','enum':['named','mix']},
    'start': {'type':'number'}, 'end': {'type':'number'},
    'timeline': {'type':'string','enum':['overlay','insert','append']},
    'timing': {'type':'string','enum':['literal','default','estimated']},
    'count': {'type':'integer','minimum':1,'maximum':16},
    'continuous': {'type':'boolean'}, 'background': {'type':'boolean'},
    'gain_start_db': {'type':'number'}, 'gain_end_db': {'type':'number'},
}
SCHEMA = {'type':'object','properties':{'actions':{'type':'array','minItems':1,'maxItems':4,
    'items':{'type':'object','properties':ACTION_FIELDS,'required':list(ACTION_FIELDS),
             'additionalProperties':False}}},'required':['actions'],'additionalProperties':False}

def save(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2))

def planning_request(instruction, duration, model='gpt-5-mini'):
    return dict(model=model,reasoning_effort='low',max_completion_tokens=4000,
        messages=[dict(role='system',content=PROMPT),dict(role='user',content=json.dumps(
            dict(original_instruction=instruction,duration_seconds=duration),ensure_ascii=False))],
        response_format=dict(type='json_schema',json_schema=dict(name='basic_audio_edit',strict=True,schema=SCHEMA)))

def validate(plan, duration):
    assert type(plan)is dict and set(plan)=={'actions'}
    assert type(plan['actions'])is list and 1<=len(plan['actions'])<=4
    for index,a in enumerate(plan['actions']):
        assert type(a)is dict and set(a)==set(ACTION_FIELDS)
        for k,s in ACTION_FIELDS.items():
            v=a[k]
            if s['type']=='string':assert isinstance(v,str) and ('enum'not in s or v in s['enum'])
            elif s['type']=='number':assert type(v)in(int,float) and math.isfinite(v)
            elif s['type']=='integer':assert type(v)is int and s['minimum']<=v<=s['maximum']
            elif s['type']=='boolean':assert type(v)is bool
        assert 0<=a['start']<a['end'] and a['start']<=duration
        assert a['timeline']=='overlay' or a['operation']=='add'
        if a['timeline']=='overlay':assert a['end']<=duration+1e-5
        else:
            assert a['end']-a['start']<=30 and index==len(plan['actions'])-1
        if a['timeline']=='append':assert abs(a['start']-duration)<1e-5
        if a['operation']!='add':assert 0<len(a['target'].split())<=35
        if a['operation']in('add','replace'):assert 0<len(a['material'].split())<=40
        assert -60<=a['gain_start_db']<=24 and -60<=a['gain_end_db']<=24
        if a['operation']!='volume':assert a['gain_start_db']==a['gain_end_db']==0
    return deepcopy(plan)

def first_plan(instruction, duration, folder, llm=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    request=planning_request(instruction,duration,getattr(llm,'model','gpt-5-mini'))
    claim=folder/'plan_claim.json'
    with claim.open('x')as f:json.dump(dict(maximum_calls=1,request_sha256=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()),f)
    save(folder/'plan_request.json',request)
    if llm is None:
        from audio_edit.llm.client import LLMClient
        llm=LLMClient()

    client=llm.client.with_options(max_retries=0,timeout=150)
    response=client.chat.completions.create(**request)
    save(folder/'plan_first_response.json',response.model_dump(mode='json'))
    choice=response.choices[0]
    assert choice.finish_reason=='stop' and not getattr(choice.message,'refusal',None)
    plan=validate(json.loads(choice.message.content),duration)
    save(folder/'plan.json',plan)
    return plan

def align(data, rate, sample_rate, frames, channels):
    data=np.asarray(data,dtype=np.float64)
    if data.ndim==1:data=data[:,None]
    assert np.isfinite(data).all() and len(data)>0
    if rate!=sample_rate:
        import torch
        from torchaudio.functional import resample
        data=resample(torch.as_tensor(data.T.copy()),rate,sample_rate).numpy().T
    if data.shape[1]!=channels:
        if channels==1:data=data.mean(axis=1,keepdims=True)
        elif data.shape[1]==1:data=np.repeat(data,channels,axis=1)
        else:data=np.tile(data,(1,math.ceil(channels/data.shape[1])))[:,:channels]
    if frames is not None:
        data=data[:frames]
        if len(data)<frames:data=np.pad(data,((0,frames-len(data)),(0,0)))
    return data

def edges(length, rate):
    w=np.ones(length);n=min(round(.005*rate),length//2)
    if n:
        w[:n]=np.linspace(0,1,n);w[-n:]=np.linspace(1,0,n)
    return w[:,None]

class NativeBackend:
    def __init__(self, folder):
        if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Basic editing requires Slurm GPU allocation')
        print(subprocess.check_output(['nvidia-smi'],text=True),flush=True)
        self.folder=Path(folder);self.calls=[]
        self.source=Path(__file__).resolve().parents[2]

    def separate(self, original, query, mode, index):
        from audio_edit.plans.remove_sound_plan import RemoveSoundPlan
        from audio_edit.runners.samaudio_runner import SAMAudioRunner
        d=self.folder/f'separation-{index}';d.mkdir()
        plan=RemoveSoundPlan(backend='samaudio',mode='text',result_mode=mode,text_prompt=query)
        payload=dict(input_audio=str(original),output_dir=str(d),plan=SAMAudioRunner()._plan_to_json_dict(plan))
        request=d/'request.json';response=d/'response.json';save(request,payload)
        cmd=[os.environ.get('SAMAUDIO_PYTHON',str(self.source/'.venvs/sam-audio/bin/python')),
            os.environ.get('SAMAUDIO_SCRIPT',str(self.source/'audio_edit/tools/run_sam_audio.py')),
            '--plan-json',str(request),'--result-json',str(response),'--model-id','facebook/sam-audio-large']
        self.calls.append(dict(kind='SAM',request=str(request),response=str(response)))
        subprocess.run(cmd,check=True,timeout=900)
        result=json.loads(response.read_text());return sf.read(result['output_audio'],always_2d=True)

    def generate(self, query, duration, index):
        output=self.folder/f'material-{index}.wav'
        cmd=[os.environ.get('STABLE_AUDIO_PYTHON',str(self.source/'.venvs/stable-audio-3/bin/python')),
            os.environ.get('STABLE_AUDIO_SCRIPT',str(self.source/'audio_edit/tools/run_stable_audio.py')),
            '--prompt',query,'--duration',str(min(30.,max(.5,duration))),
            '--seed',str(int(os.environ.get('BASIC_EXECUTION_SEED','20261018'))),'--output',str(output)]
        self.calls.append(dict(kind='StableAudio',prompt=query,duration=min(30.,max(.5,duration)),output=str(output)))
        subprocess.run(cmd,check=True,timeout=900)
        return sf.read(output,always_2d=True)

def render(plan, original, folder, backend=None):
    folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=True)
    x,sr=sf.read(original,always_2d=True);assert len(x)>0 and np.isfinite(x).all()
    plan=validate(plan,len(x)/sr);backend=backend or NativeBackend(folder)
    y=x.copy();reference=x.copy();cache={};events=[];offset=0

    for i,a in enumerate(plan['actions']):
        start=round(a['start']*sr);end=round(a['end']*sr);length=end-start
        if length<=0:raise ValueError('Empty numerical action')
        event=dict(index=i,operation=a['operation'],timing=a['timing'])
        try:
            delta=np.zeros_like(x)
            if a['operation'] in ('remove','retain','replace','volume'):
                mode='result' if a['operation']in('retain','volume')else'residual'
                current=y.copy()
                if a['source_scope']=='mix':stem=current.copy()if mode=='result'else np.zeros_like(current)
                else:


                    working=folder/f'working-{i}.wav';sf.write(working,current,sr,subtype='FLOAT')
                    key=(hashlib.sha256(working.read_bytes()).hexdigest(),a['target'],mode)
                    if key not in cache:
                        raw,rate=backend.separate(working,a['target'],mode,i)
                        cache[key]=align(raw,rate,sr,len(x),x.shape[1])
                    stem=cache[key]
                if a['operation']=='volume':
                    gains=10**(np.linspace(a['gain_start_db'],a['gain_end_db'],length)/20)
                    delta[start:end]=stem[start:end]*(gains[:,None]-1)
                else:delta[start:end]=stem[start:end]-current[start:end]
                delta[start:end]*=edges(length,sr)
                y[start+offset:end+offset]+=delta[start:end]
            if a['operation'] in ('add','replace'):
                unit=length if a['continuous']else max(1,length//a['count'])
                raw,rate=backend.generate(a['material'],unit/sr,i)
                raw=align(raw,rate,sr,None,x.shape[1])
                if a['continuous'] and len(raw)<unit:raw=np.tile(raw,(math.ceil(unit/len(raw)),1))
                raw=align(raw,sr,sr,unit,x.shape[1])
                rms=float(np.sqrt(np.mean(raw**2)));peak=float(np.max(np.abs(raw)))
                if rms>1e-7:
                    level=10**((-24 if a['background']else-18)/20)
                    raw*=min(level/rms,10**(30/20),.8/max(peak,1e-9))
                raw*=edges(len(raw),sr)
                layer=np.zeros((length,x.shape[1]))
                starts=[0]if a['continuous']else[round(k*(length-unit)/max(1,a['count']-1))for k in range(a['count'])]
                for pos in starts:layer[pos:pos+unit]+=raw[:min(unit,length-pos)]
                if a['timeline']=='overlay':y[start+offset:end+offset]+=layer
                else:
                    pos=start+offset;y=np.concatenate([y[:pos],layer,y[pos:]],axis=0)
                    reference=np.concatenate([reference[:pos],np.zeros_like(layer),reference[pos:]],axis=0)
                    offset+=length
            assert np.isfinite(y).all()
            event['completed']=True
            sf.write(folder/'partial.wav',y,sr,subtype='FLOAT')
        except Exception as exc:
            event.update(completed=False,reason=f'{type(exc).__name__}: {exc}')
        events.append(event)

    changed=y!=reference
    clipped=int(np.count_nonzero((np.abs(y)>1)&changed))
    y[changed]=np.clip(y[changed],-1,1)
    output=folder/'edited.wav';sf.write(output,y,sr,subtype='FLOAT')
    save(folder/'execution.json',dict(actions=events,backend_calls=backend.calls,clipped_samples=clipped,
        sample_rate=sr,input_frames=len(x),output_frames=len(y),channels=y.shape[1]))
    return str(output),events

def recover(ctx, original, llm=None):
    """Always deliver valid audio for a valid local audio input; never fake a pass."""
    root=Path('./audio_edit/outputs');root.mkdir(parents=True,exist_ok=True)
    folder=Path(tempfile.mkdtemp(prefix='basic-execution-',dir=root)).resolve()
    info=sf.info(original);audit=dict(route='basic_execution',directory=str(folder),original_request_only=True)
    try:
        plan=first_plan(ctx.user_prompt,info.frames/info.samplerate,folder,llm)
        output,events=render(plan,original,folder);audit['actions']=events
    except Exception as exc:
        output=original;audit['delivery_note']=f'{type(exc).__name__}: {exc}'
    ctx.output_audio=output;ctx.fallback_used=False;ctx.execution_error=None
    ctx.params.pop('execution_outcome',None);ctx.params['basic_execution']=audit
    ctx.params['temporal_resolution_status']='estimated'
    from audio_edit.utils.execution_outcome import annotate_outcome

    return annotate_outcome(ctx,original)
