"""Render a positively wrong isolated generated instrument from its first composition."""
from copy import deepcopy
import json,re
from audio_edit.tools.instrumental_material import sha
PROMPT = 'Check the FIRST isolated replacement music, not the original mixed recording. The user requests an instrument/timbre substitution. Decide whether a one-time note transcription and deterministic instrumental rendering of that same FIRST generated composition is justified by a positive wrong-instrument fact. This never transcribes the original mixture, removes another source, changes existing residual, invents a new composition, regenerates diffusion, changes mixing, or retries analysis. It preserves the first generated pitches, note times and dynamics to the extent supported by the fixed existing transcriber. Do not claim recovery of the original melody if it is missing from the first composition.\nRequire that the original request names one specific replacement instrument, and first_material_facts positively identify a different instrumental timbre without the requested instrument. The material must be exclusively pitched instrumental music: no voice, drums, crowd, environment or uncertain additional source. An absent explicit name alone is insufficient; generic musical descriptions or a possibly compatible acoustic family are insufficient. If the desired instrument is already positively present, return false. Exact continuous slides/vibrato/technique preservation is outside discrete-note rendering. No preset selection, audio feedback, score, or variant is available.\nReturn eligible:boolean, requested_instrument:string or null, request_quote:exact substring or null, wrong_material_quote:exact substring of first_material_facts or null, reason:string. True requires direct quoted authority and a positively incompatible isolated instrumental timbre. False fields may be null. This check only authorizes one first capability test; empty note transcription closes it without another model, threshold, patch or generation.'

def decide(ctx,plan,placement,generated,*,llm=None):
    from audio_edit.tools.instrumental_material import applicable
    audit=dict(eligible=False,status='not_applicable',new_material_observations=0)
    if not applicable(ctx,plan,placement):return audit
    replacement=ctx.params['replacement_plan']

    target=' '.join(str(replacement.get('target_sound','')).lower().split())
    if re.fullmatch(r'(?:solo |acoustic )?violin(?: (?:timbre|tone|sound|performance|playing))?(?: replacing .+)?',target)is None:return audit
    if re.search(r'\b(?:slides?|glissand[oi]|portamento|vibrato)\b',replacement.get('user_prompt',''),re.I):return audit
    digest=sha(generated);facts=ctx.params.get('replacement_material_first_facts',{}).get(digest)


    if (not isinstance(facts,dict) or facts.get('audio_sha256')!=digest or facts.get('first_observation')is not True
            or not isinstance(facts.get('text'),str) or not facts['text'].strip()):return dict(audit,status='no_bound_first_material_facts')
    payload=dict(original_instruction=replacement['user_prompt'],requested_new_source=replacement['target_sound'],first_material_facts=facts['text'])
    props=dict(eligible=dict(type='boolean'),requested_instrument=dict(type=['string','null']),request_quote=dict(type=['string','null']),wrong_material_quote=dict(type=['string','null']),reason=dict(type='string'))
    if llm is None:
        from audio_edit.llm.client import LLMClient
        llm=LLMClient()
    kwargs=dict(model=llm.model,messages=[dict(role='system',content=PROMPT),dict(role='user',content=json.dumps(payload,ensure_ascii=False))])
    if getattr(llm,'backend',None)=='openai':kwargs.update(reasoning_effort='low',response_format=dict(type='json_schema',json_schema=dict(name='first_material_instrument_projection',strict=True,schema=dict(type='object',additionalProperties=False,properties=props,required=list(props)))))
    try:
        from audio_edit.analyzers.qwen_audio_analyzer import extract_json_object
        response=llm.client.chat.completions.create(**kwargs);answer=extract_json_object(response.choices[0].message.content)
    except Exception as exc:return dict(audit,status='first_projection_binding_unavailable',error=f'{type(exc).__name__}: {exc}')
    audit.update(status='first_material_retained',first_decision=True,request=payload,response=deepcopy(answer))
    if (isinstance(answer,dict)and set(answer)==set(props)and answer.get('eligible')is True
            and str(answer.get('requested_instrument','')).lower()=='violin'
            and isinstance(answer.get('request_quote'),str)and answer['request_quote'].strip()and answer['request_quote']in replacement['user_prompt']
            and isinstance(answer.get('wrong_material_quote'),str)and answer['wrong_material_quote'].strip()and answer['wrong_material_quote']in facts['text']):
        audit.update(eligible=True,status='first_generated_wrong_instrument_confirmed',program=40)
    return audit

def project(ctx,plan,placement,generated):
    audit=decide(ctx,plan,placement,generated)
    ctx.params.setdefault('replacement_generated_instrument_projection',[]).append(audit)
    if not audit['eligible']:return generated
    from audio_edit.tools.independent_piano import render_first
    result=render_first(generated,audit['program']);audit['first_render']=deepcopy(result)
    if result.get('status')!='rendered':
        audit.update(status='first_transcription_or_render_failed_original_material_retained',executed=False)
        return generated
    audit.update(status='first_generated_composition_projected',executed=True,first_material_sha256=sha(generated),original_residual_unchanged=True,new_source_SAM=0,new_diffusion=0,new_CPU_transcriptions=1)
    return result['output_audio']
