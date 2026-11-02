"""Clear only a complete instrumental residual, bound to its first neutral facts."""
from copy import deepcopy
import json,re
PROMPT = 'Bind a requested complete instrumental replacement to the audible sources in its EXISTING retained residual. Use only the original instruction and supplied first original/residual descriptions. Decide whether EVERY audible source that remains in the residual belongs to the class of instruments the user requested replacing. This is residual-source membership, not equality between the original and residual inventories: instruments already absent from the residual need not still be heard. A missing originally described instrument is not itself a reason to preserve other unwanted instruments. Never infer an instrument from a desired future sound.\nRequire an explicit replacement of all instruments, the complete band/backing, or a complete instrumental arrangement. A selected instrument, selected occurrence, ambiguous target class, or whole-file timing alone does not authorize clearing every sound. Vocals, speech, crowds, environment, noise and ambiguous nonmusical effects are protected unless the original instruction explicitly replaces them. Their presence or uncertainty in the residual prevents complete residual clearing. A residual described as instrumental music with explicitly no other discernible sources may consist of any subset of the requested instrumental classes. Keep every contradictory or uncertain original observation in consideration; do not relabel voices or noises as instruments.\nThis decision can only clear the residual while retaining the same first replacement material, placement and level policy. It changes no original source extraction, reference, generation, timing, duration or seed. Return eligible:boolean, residual_only_requested_sources:boolean or null, retained_or_uncertain_sources:array of strings or null, instruction_quote:exact substring or null, residual_fact_quote:exact substring or null, reason:string. eligible=true requires residual_only_requested_sources=true, retained_or_uncertain_sources=[], and literal quotations supporting both requested scope and observed residual inventory. If unsupported or uncertain, false; no follow-up listening or alternate proposal.'

def inspect_residual_subset(ctx,plan,residual,duration,*,llm=None,analyzer=None):
    from audio_edit.plans.music_backing import backing_request
    from audio_edit.analyzers.removal_focus import file_sha256,sole_source_decision
    from audio_edit.analyzers.replacement_material_guard import sample_digest
    from audio_edit.plans.remove_sound_plan import shared_removal_windows
    audit=dict(eligible=False,status='not_applicable',new_observations=0)
    if ctx.selected_script!='replace' or backing_request(ctx,plan,duration) is None:return audit


    source=' '.join(str(plan.source_sound).lower().split())
    if not re.search(r'\b(?:all (?:the )?(?:musical )?instruments|(?:complete |entire |full )?(?:band|instrumental (?:backing|arrangement|accompaniment)))\b',source):return audit
    original=ctx.params.get('replacement_source_facts')
    if (not isinstance(original,dict) or original.get('audio_sha256')!=file_sha256(ctx.input_audio)
            or original.get('first_observation')is not True or not isinstance(original.get('text'),str) or not original['text'].strip()):
        return dict(audit,status='no_bound_first_original_facts')
    digest=sample_digest(residual);facts=ctx.params.get('replacement_residual_first_facts')
    if facts is None:
        if analyzer is None:
            from audio_edit.analyzers.qwen_audio_analyzer import QwenAudioAnalyzer
            analyzer=QwenAudioAnalyzer()
        try:
            audit['new_observations']=1
            facts=dict(audio_sample_sha256=digest,text=analyzer.describe_sources(residual),first_observation=True)
            ctx.params['replacement_residual_first_facts']=deepcopy(facts)
        except Exception as exc:return dict(audit,status='first_residual_observation_unavailable',error=f'{type(exc).__name__}: {exc}')
    if (not isinstance(facts,dict) or facts.get('audio_sample_sha256')!=digest
            or facts.get('first_observation')is not True or not isinstance(facts.get('text'),str) or not facts['text'].strip()):
        return dict(audit,status='unbound_first_residual_facts')
    payload=dict(original_instruction=ctx.user_prompt,requested_old_source=plan.source_sound,
        first_original_facts=original['text'],first_residual_facts=facts['text'])
    props=dict(eligible=dict(type='boolean'),residual_only_requested_sources=dict(type=['boolean','null']),retained_or_uncertain_sources=dict(type=['array','null'],items=dict(type='string')),instruction_quote=dict(type=['string','null']),residual_fact_quote=dict(type=['string','null']),reason=dict(type='string'))
    if llm is None:
        from audio_edit.llm.client import LLMClient
        llm=LLMClient()
    kwargs=dict(model=llm.model,messages=[dict(role='system',content=PROMPT),dict(role='user',content=json.dumps(payload,ensure_ascii=False))])
    if getattr(llm,'backend',None)=='openai':kwargs.update(reasoning_effort='low',response_format=dict(type='json_schema',json_schema=dict(name='residual_target_subset',strict=True,schema=dict(type='object',additionalProperties=False,properties=props,required=list(props)))))
    try:
        from audio_edit.analyzers.qwen_audio_analyzer import extract_json_object
        response=llm.client.chat.completions.create(**kwargs);answer=extract_json_object(response.choices[0].message.content)
    except Exception as exc:return dict(audit,status='first_subset_binding_unavailable',first_decision=True,error=f'{type(exc).__name__}: {exc}')
    audit.update(status='residual_subset_not_confirmed',first_decision=True,request=payload,response=deepcopy(answer))
    if (not isinstance(answer,dict) or set(answer)!=set(props) or answer.get('eligible')is not True
            or answer.get('residual_only_requested_sources')is not True or answer.get('retained_or_uncertain_sources')!=[]
            or not isinstance(answer.get('instruction_quote'),str) or not answer['instruction_quote'].strip() or answer['instruction_quote']not in ctx.user_prompt
            or not isinstance(answer.get('residual_fact_quote'),str) or not answer['residual_fact_quote'].strip() or answer['residual_fact_quote']not in facts['text']):return audit
    windows=shared_removal_windows(plan.remove_plan,duration)
    evidence=dict(source='first_residual_target_subset',target_sound=plan.remove_plan.text_prompt,audio=dict(sha256=file_sha256(residual)),windows=[dict(start_time=a,end_time=b,observation=dict(target_present=True,only_target=True,other_sounds=[],target_is_single_instrument=False,evidence=answer['residual_fact_quote']))for a,b in windows],binding=dict(first_original_facts=deepcopy(original),first_residual_facts=deepcopy(facts),first_response=deepcopy(answer)))
    decision=sole_source_decision(evidence,residual,duration,windows,plan.remove_plan.text_prompt,source='first_residual_target_subset')
    return dict(audit,status='confirmed_residual_target_subset',eligible=decision['eligible'],decision=decision)
