"""Remove positively observed unrequested singing from a generated instrumental bed."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,re,subprocess
from tempfile import mkdtemp
PROMPT='''Determine whether the FIRST generated replacement material contains an unrequested singing voice. Use only the original instruction, requested replacement source, and first neutral material facts. The material is a NEW isolated asset; original speech is in a separate, fixed residual and must remain unchanged. Select cleanup ONLY when the requested new source is instrumental music, no new singing/humming/choir/vocals are requested, the facts positively identify a singing voice in the generated asset, and a requested instrumental source is also heard. A soft male vocalist accompanying guitar is still singing. Do not infer singing from a synthesizer, sustained chord, string texture, breath, or uncertainty. If no singing is positively heard or the new target includes voices, return false. Cleanup removes only singing voice from this generated asset with the existing separator; it never alters the original or retained residual. Return JSON: eligible(boolean), fact_quote(exact continuous substring of the first material facts supporting singing, or null), reason. No timing, level, generator, instrument, or target changes.'''
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def applicable(ctx,plan,placement):
    replacement=ctx.params.get('replacement_plan')
    if not isinstance(replacement,dict) or not placement.get('shared_window'):return False
    target=replacement.get('target_sound','')
    if not isinstance(target,str) or not re.search(r'\b(piano|guitar|violin|viola|cello|flute|oboe|clarinet|trumpet|saxophone|harp|strings)\b',target,re.I):return False
    if re.search(r'\b(vocals?|voices?|singing|singer|humming|choir|a cappella)\b',target,re.I):return False
    if plan.is_speech or plan.is_on_screen or placement.get('generator')!='audio':return False
    if any(ctx.params.get(key,{}).get('status')=='rendered' for key in ['replacement_music_render','replacement_independent_piano']):return False
    return True

def decide(ctx,plan,placement,generated,*,llm=None):
    audit=dict(eligible=False,status='not_applicable',new_material_observations=0)
    if not applicable(ctx,plan,placement):return audit
    digest=sha(generated);facts=ctx.params.get('replacement_material_first_facts',{}).get(digest)
    if facts is None:
        from audio_edit.analyzers.qwen_audio_analyzer import QwenAudioAnalyzer
        try:text=QwenAudioAnalyzer().describe_sources(generated)
        except Exception as exc:return dict(audit,status='first_material_facts_unavailable',error=f'{type(exc).__name__}: {exc}')
        facts=dict(audio_sha256=digest,text=text,first_observation=True)
        ctx.params.setdefault('replacement_material_first_facts',{})[digest]=deepcopy(facts);audit['new_material_observations']=1
    if (not isinstance(facts,dict) or facts.get('audio_sha256')!=digest or facts.get('first_observation') is not True or not isinstance(facts.get('text'),str)):
        return dict(audit,status='unbound_first_material_facts')
    replacement=ctx.params['replacement_plan'];payload=dict(original_instruction=replacement['user_prompt'],requested_new_source=replacement['target_sound'],first_material_facts=facts['text'])
    if llm is None:
        from audio_edit.llm.client import LLMClient
        llm=LLMClient()
    props=dict(eligible=dict(type='boolean'),fact_quote=dict(type=['string','null']),reason=dict(type='string'))
    kwargs=dict(model=llm.model,messages=[dict(role='system',content=PROMPT),dict(role='user',content=json.dumps(payload,ensure_ascii=False))])
    if getattr(llm,'backend',None)=='openai':kwargs['response_format']=dict(type='json_schema',json_schema=dict(name='instrumental_material_voice',strict=True,schema=dict(type='object',additionalProperties=False,properties=props,required=list(props))))
    from audio_edit.analyzers.qwen_audio_analyzer import extract_json_object
    try:
        response=llm.client.chat.completions.create(**kwargs);answer=extract_json_object(response.choices[0].message.content)
    except Exception as exc:return dict(audit,status='first_material_decision_unavailable',error=f'{type(exc).__name__}: {exc}')
    audit.update(status='first_material_retained',material_sha256=digest,request=payload,response=deepcopy(answer),first_decision=True)
    if isinstance(answer,dict) and set(answer)==set(props) and answer.get('eligible') is True:
        quote=answer.get('fact_quote')
        if isinstance(quote,str) and quote.strip() and quote in facts['text']:audit.update(eligible=True,status='unrequested_singing_confirmed',sam_query='singing voice')
    return audit

def clean(ctx,plan,placement,generated):
    audit=decide(ctx,plan,placement,generated)
    ctx.params.setdefault('replacement_material_voice_cleanup',[]).append(audit)
    if not audit['eligible']:return generated
    from audio_edit.models import AudioEditContext
    from audio_edit.plans.remove_sound_plan import RemoveSoundPlan
    from audio_edit.runners.samaudio_runner import SAMAudioRunner
    import numpy as np
    import soundfile as sf
    root=Path('audio_edit/generated');root.mkdir(parents=True,exist_ok=True);folder=Path(mkdtemp(prefix='material-instrumental-',dir=root)).resolve()
    removal=RemoveSoundPlan(backend='samaudio',mode='text',result_mode='residual',text_prompt='singing voice',temporal_status='resolved',notes='Remove observed unrequested singing only from the first generated instrumental material.')
    result=SAMAudioRunner(output_dir=str(folder)).run_remove(AudioEditContext('Remove unrequested singing from generated instrumental material',input_audio=generated),removal)
    original,rate=sf.read(generated,always_2d=True,dtype='float32');output,sr=sf.read(result.output_audio,always_2d=True,dtype='float32')
    if sr!=rate:
        resampled=folder/'resampled.wav';subprocess.run(['ffmpeg','-nostdin','-v','error','-i',result.output_audio,'-ar',str(rate),'-c:a','pcm_f32le',str(resampled)],check=True)
        output,sr=sf.read(resampled,always_2d=True,dtype='float32')
    if output.shape[1]==1 and original.shape[1]>1:output=np.repeat(output,original.shape[1],axis=1)
    elif original.shape[1]==1 and output.shape[1]>1:output=output.mean(axis=1,keepdims=True)
    assert output.shape[1]==original.shape[1] and abs(len(output)-len(original))<=2 and np.isfinite(output).all()
    aligned=np.zeros_like(original);aligned[:min(len(output),len(aligned))]=output[:len(aligned)];path=folder/'instrumental.wav';sf.write(path,aligned,rate,subtype='FLOAT')
    audit.update(executed=True,input_audio=str(generated),input_sha256=sha(generated),output_audio=str(path),output_sha256=sha(path),separation_calls=result.params.get('separation_calls'),first_sample_grid=True,original_residual_unchanged=True)
    return str(path)
