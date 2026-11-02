"""Bind the first original-source facts to whole-file independent replacement residuals."""
from copy import deepcopy
import json

PROMPT = "Bind existing FIRST original-audio facts to a whole-file old-source replacement scope. Decide whether the positively observed old target is the ONLY audible source throughout the entire original recording, with no other speech, music, accompaniment, ambient sound or noise to preserve. A dominant source or an incomplete event list alone is insufficient. Use only the original instruction, literal old source, original first description/assessment, and original first event facts. Do not use a requested replacement as evidence. Do not infer a source from an editing goal. If any other sound is mentioned, any retained source is uncertain, or the request selects only one occurrence, return false. Silence between a confirmed single source's events is not a separate source. Return eligible:boolean, target_present:boolean or null, only_target:boolean or null, other_sounds:array of strings or null, fact_quote:exact continuous substring of first_source_facts supporting the decision or null, reason:string. The proposed operation clears the original first residual only after this complete-target claim is bound; it never changes the first generated replacement material, its placement or any level policy. No source re-listening, new source query, timing, gain or generation changes."


def inspect_first_sole(ctx, plan, duration, *, llm=None):
    from audio_edit.plans.music_backing import backing_request
    from audio_edit.analyzers.removal_focus import file_sha256, sole_source_decision
    from audio_edit.plans.remove_sound_plan import shared_removal_windows
    scope = backing_request(ctx, plan, duration)
    audit = dict(eligible=False, status='not_applicable', new_observations=0)
    if scope is None or scope.get('replacement_material') != 'independent_texture':
        return audit
    facts = ctx.params.get('replacement_source_facts')
    if (not isinstance(facts, dict) or facts.get('audio_sha256') != file_sha256(ctx.input_audio)
            or facts.get('first_observation') is not True or not isinstance(facts.get('text'), str)
            or not facts['text'].strip()):
        return dict(audit, status='no_bound_first_source_facts_existing_path_retained')
    payload = dict(original_instruction=ctx.user_prompt, old_source=plan.source_sound,
                   scope='whole_file', first_source_facts=facts['text'])
    props = dict(eligible=dict(type='boolean'), target_present=dict(type=['boolean','null']),
                 only_target=dict(type=['boolean','null']), other_sounds=dict(type=['array','null'],items=dict(type='string')),
                 fact_quote=dict(type=['string','null']), reason=dict(type='string'))
    if llm is None:
        from audio_edit.llm.client import LLMClient
        llm = LLMClient()
    kwargs = dict(model=llm.model, messages=[dict(role='system',content=PROMPT),
                                           dict(role='user',content=json.dumps(payload,ensure_ascii=False))])
    if getattr(llm,'backend',None)=='openai':
        kwargs.update(reasoning_effort='low',response_format=dict(type='json_schema',json_schema=dict(
            name='first_original_sole_event',strict=True,schema=dict(type='object',additionalProperties=False,properties=props,required=list(props)))))
    try:
        from audio_edit.analyzers.qwen_audio_analyzer import extract_json_object
        response=llm.client.chat.completions.create(**kwargs)
        answer=extract_json_object(response.choices[0].message.content)
    except Exception as exc:
        return dict(audit,status='first_binding_failed_existing_path_retained',first_decision=True,error=f'{type(exc).__name__}: {exc}')
    audit.update(status='first_facts_do_not_confirm_sole_target',first_decision=True,request=payload,response=deepcopy(answer))
    if (not isinstance(answer,dict) or set(answer)!=set(props) or answer.get('eligible') is not True
            or answer.get('target_present') is not True or answer.get('only_target') is not True
            or answer.get('other_sounds')!=[] or not isinstance(answer.get('fact_quote'),str)
            or not answer['fact_quote'].strip() or answer['fact_quote'] not in facts['text']):
        return audit
    windows=shared_removal_windows(plan.remove_plan,duration)
    evidence=dict(source='first_original_source_facts_binding',target_sound=plan.remove_plan.text_prompt,
        audio=dict(sha256=file_sha256(ctx.input_audio)),windows=[dict(start_time=a,end_time=b,
            observation=dict(target_present=True,only_target=True,other_sounds=[],target_is_single_instrument=False,evidence=answer['fact_quote']))for a,b in windows],
        binding=dict(first_facts=deepcopy(facts),first_response=deepcopy(answer)))
    decision=sole_source_decision(evidence,ctx.input_audio,duration,windows,plan.remove_plan.text_prompt,source='first_original_source_facts_binding')
    return dict(audit,status='first_facts_confirm_complete_target',eligible=decision['eligible'],decision=decision)
