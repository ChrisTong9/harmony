"""Do not discard the old residual for an audibly wrong replacement material."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,shutil,tempfile

PROMPT='''Check only whether the first generated material has a source-class
contradiction with the original requested NEW sound. material_facts are one neutral
description of the isolated generated layer; they contain no original mixture.
No scores, alternative renders, or second listening are available.

Return contradicts_new_source=true ONLY when these facts positively describe an
audible source class incompatible with the requested new source. An instrument
is not a human voice, and a sustained tonal source is not automatically a discrete
noise/click source. Use the actual request; do not invent extra restrictions.
Do not reject a merely uncertain subtype within a compatible broad source family,
or differences in melody, technique, loudness, room tone, quality or fidelity.
Those do not establish that the wrong source class was rendered. If the facts are
compatible or insufficient, return false. This check never regenerates material.
When true, quote exact original words specifying the requested new sound and exact
material-fact words establishing the contradiction. Return JSON with
contradicts_new_source (boolean), instruction_quote (string|null),
material_fact_quote (string|null), reason (string).'''

def sample_digest(path):
    import soundfile as sf
    audio,rate=sf.read(path,always_2d=True,dtype='float32')
    digest=hashlib.sha256(json.dumps(dict(rate=rate,shape=audio.shape),sort_keys=True).encode())
    digest.update(audio.astype('<f4',copy=False).tobytes())
    return digest.hexdigest()

def decide(ctx,plan,path,*,llm=None,original_instruction=None):
    digest=sample_digest(path)
    audit=dict(contradiction=False,status='class_not_contradicted',new_material_queries=0,
        material_sample_sha256=digest)
    facts=ctx.params.get('replacement_material_first_facts')
    if facts is None:
        from audio_edit.analyzers.qwen_audio_analyzer import QwenAudioAnalyzer
        try:
            audit['new_material_queries']=1
            text=QwenAudioAnalyzer().describe_sources(path)
            facts=dict(audio_sample_sha256=digest,text=text,first_observation=True)
            ctx.params['replacement_material_first_facts']=deepcopy(facts)
            audit['new_material_queries']=1
        except Exception as exc:
            return dict(audit,status='material_observation_unavailable_policy_unchanged',error=f'{type(exc).__name__}: {exc}')
    if (not isinstance(facts,dict) or facts.get('audio_sample_sha256')!=digest
            or facts.get('first_observation') is not True or not isinstance(facts.get('text'),str)
            or not facts['text'].strip()):
        return dict(audit,status='unbound_material_facts_policy_unchanged')
    instruction=ctx.user_prompt if original_instruction is None else original_instruction
    request=dict(original_instruction=instruction,new_source=plan.target_sound,material_facts=facts['text'])
    if llm is None:
        from audio_edit.llm.client import LLMClient
        llm=LLMClient()
    props=dict(contradicts_new_source=dict(type='boolean'),instruction_quote=dict(type=['string','null']),
        material_fact_quote=dict(type=['string','null']),reason=dict(type='string'))
    kwargs=dict(model=llm.model,messages=[dict(role='system',content=PROMPT),dict(role='user',content=json.dumps(request,ensure_ascii=False))])
    if getattr(llm,'backend',None)=='openai':
        kwargs['response_format']=dict(type='json_schema',json_schema=dict(name='replacement_material_class',strict=True,
            schema=dict(type='object',additionalProperties=False,properties=props,required=list(props))))
    from audio_edit.analyzers.qwen_audio_analyzer import extract_json_object
    try:
        response=llm.client.chat.completions.create(**kwargs)
        answer=extract_json_object(response.choices[0].message.content)
    except Exception as exc:
        return dict(audit,status='first_material_decision_unavailable_policy_unchanged',
            request=request,first_decision=True,error=f'{type(exc).__name__}: {exc}')
    audit.update(request=request,response=deepcopy(answer),first_decision=True,source_facts=deepcopy(facts))
    if isinstance(answer,dict) and set(answer)==set(props) and answer.get('contradicts_new_source') is True:
        quote,evidence=answer.get('instruction_quote'),answer.get('material_fact_quote')
        if (isinstance(quote,str) and quote.strip() and quote in instruction
                and isinstance(evidence,str) and evidence.strip() and evidence in facts['text']):
            audit.update(contradiction=True,status='positive_source_class_contradiction')
    return audit

def retain_residual_if_contradicted(ctx,plan,mixer,prior_residual,*,original_instruction):
    audit=decide(ctx,plan,ctx.output_audio,original_instruction=original_instruction)
    ctx.params['replacement_material_class_guard']=audit
    if not audit['contradiction']:return ctx

    root=Path('audio_edit/generated');root.mkdir(parents=True,exist_ok=True)
    folder=Path(tempfile.mkdtemp(prefix='material-class-',dir=root)).resolve()
    isolated=folder/'first_isolated_material.wav';shutil.copy2(ctx.output_audio,isolated)
    placements=deepcopy(ctx.params['executed_placements'])
    features=dict(plan.add_plan.composer or {})
    if plan.add_plan.layer_role:features['layer_role']=plan.add_plan.layer_role
    if ctx.params.get('replacement_reference_path'):features['replacement_reference_path']=ctx.params['replacement_reference_path']
    ctx.output_audio=mixer.mix_background(original_path=prior_residual,background_path=placements[0]['background_path'],
        start_time=plan.add_plan.start_time,end_time=plan.add_plan.end_time,acoustic_feature=features,placements=placements)
    audit.update(action='retain_original_residual_with_same_first_target',first_isolated_material=str(isolated),
        retained_residual=str(prior_residual),new_generator_calls=0,new_separation_calls=0)
    ctx.params['replacement_residual_correction']['adopted']=False
    ctx.params['mix_audit']=getattr(mixer,'last_audit',[])
    ctx.warnings.extend(getattr(mixer,'last_warnings',[]))
    return ctx
