"""Consume one pre-bound local analysis decision; never issue a new observation."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path

def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def bind(ctx,plan,analysis):
    evidence=ctx.params.get('source_query_first_binding')
    if not isinstance(evidence,dict):return plan
    contract=ctx.params.get('edit_listening_dialogue',{})
    selection=contract.get('selection',{})
    if not (plan.backend=='samaudio' and plan.mode=='text' and plan.result_mode=='residual'
        and not plan.use_video_prompt and not plan.placements and not plan.time_span
        and not plan.temporal_schema_error and plan.temporal_status=='not_requested'
        and (plan.temporal is None or plan.temporal.kind=='unspecified')
        and selection.get('target')=='named_source' and selection.get('scope')=='whole_file'
        and selection.get('occurrence')=='unselected' and not contract.get('participant_selection')
        and contract.get('operation')=='remove_sound' and not (analysis or {}).get('edit_listening',{}).get('participant_selection')):
        return plan
    if plan.temporal is not None and any(getattr(plan.temporal,k,None) is not None for k in
            ('start_time','end_time','count','reference_event','end_reference_event','boundary_role')):
        return plan
    if plan.temporal is not None and plan.temporal.offset_seconds!=0:return plan
    facts=evidence.get('facts',{});request=evidence.get('request',{});decision=evidence.get('response',{})
    if not (evidence.get('first_decision') is True and evidence.get('analysis_provider')=='local_qwen'
        and decision.get('supported') is True and request.get('original_instruction')==ctx.user_prompt
        and request.get('current_source_query')==plan.text_prompt):return plan
    try:
        for role,key in [('original','first_original_facts'),('residual','first_residual_facts')]:
            item=facts[role]
            if not (item.get('first_observation') is True and isinstance(item.get('text'),str)
                and item['text'].strip() and request.get(key)==item['text']
                and file_sha(item['audio_path'])==item['audio_sha256']
                and file_sha(item['observation_path'])==item['observation_sha256']):return plan
        if facts['original']['audio_sha256']!=file_sha(ctx.input_audio):return plan
    except (KeyError,OSError,TypeError):return plan
    target=decision.get('target_description')
    if not isinstance(target,str) or not 0<len(target.split())<=24 or target.strip()==plan.text_prompt:return plan
    if any(word.lower().strip('.,:;') in {'remove','mute','extract','replace','erase','delete','silence'} for word in target.split()):return plan
    for key,text in [('instruction_quote',ctx.user_prompt),('original_fact_quote',facts['original']['text']),('residual_fact_quote',facts['residual']['text'])]:
        quote=decision.get(key)
        if not isinstance(quote,str) or not quote.strip() or quote not in text:return plan
    if not isinstance(decision.get('preserved_sources'),list) or not all(isinstance(x,str) and x.strip() for x in decision['preserved_sources']):return plan
    new=deepcopy(plan);new.text_prompt=target.strip()
    ctx.params['source_query_binding_audit']=dict(status='first_fact_bound_query',original_plan=asdict(plan),
        execution_query=new.text_prompt,first_response=deepcopy(decision),first_facts=deepcopy(facts),
        changed_fields=['text_prompt'],new_model_calls=0,original_plan_unchanged=True)
    return new
