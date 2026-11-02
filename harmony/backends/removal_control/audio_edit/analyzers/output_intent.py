"""Resolve a bounded conflict between global extraction and discarded-person scope."""
from copy import deepcopy
import json
import re

from audio_edit.analyzers.qwen_audio_analyzer import extract_json_object

PROMPT = """Review one audio edit's requested OUTPUT from the original instruction.
The first brief selected a person's events across the entire file, but the user
also explicitly requested extraction or an only-output source. Decide whether
the named person selects the RETURNED source, or is merely discarded while an
independent named source is extracted. Do not infer anything heard in the audio.

independent_named_source: the request explicitly asks to extract a named sound
class/instrument or keep ONLY it; no person, occurrence, event, or conditional
timing selects the retained source. Unwanted people/words may describe discarded
audio. Quote the exact original words requesting that returned source, and give
a short positive source description copied from that quote. This permits the
existing separator to extract that source across the original whole file.

selected_retained_source: the returned source is one particular participant,
selected occurrence, or source chosen relative to another event. Keep all prior
evidence requirements. A woman's/man's/ordinal person's voice is not an
independent sound-class extraction when it selects WHO is retained.

other_or_uncertain: ordinary subtraction with preservation constraints, unclear
output intent, or an actual partial/conditional scope. Keeping other sounds
natural does not mean keeping ONLY one sound. Never turn an instruction to
remove one person's events into all-voice removal. Never invent permission to
change unselected sources. The original instruction outranks the brief.

Return the structured decision once. No quality score or audio is available.
"""


def candidate(brief, payload):
    selection = brief.get('selection', {})
    return (payload.get('operation') == 'remove_sound'
            and selection.get('scope') == 'whole_file'
            and selection.get('occurrence') == 'unselected'
            and bool(brief.get('participant_selection'))
            and not brief.get('placement_windows')
            and bool(re.search(r'\b(?:extract|isolate)\b|\b(?:keep|keeping|retain|retaining|leave|leaving|output)\s+only\b',
                               payload.get('original_instruction', ''), flags=re.I)))


def _request(llm, payload):
    props = dict(decision=dict(type='string', enum=[
        'independent_named_source', 'selected_retained_source', 'other_or_uncertain']),
        source_description=dict(type=['string', 'null']),
        instruction_quote=dict(type=['string', 'null']), reason=dict(type='string'))
    options = {}
    if getattr(llm, 'backend', None) == 'openai':
        options['response_format'] = dict(type='json_schema', json_schema=dict(
            name='remove_output_intent_review', strict=True,
            schema=dict(type='object', additionalProperties=False, properties=props, required=list(props))))
    response = llm.client.chat.completions.create(model=llm.model, messages=[
        dict(role='system', content=PROMPT),
        dict(role='user', content=json.dumps(payload, ensure_ascii=False))], **options)
    return extract_json_object(response.choices[0].message.content)


def review(llm, brief, payload, audit):
    if not candidate(brief, payload):
        return brief
    request = dict(original_instruction=payload['original_instruction'],
                   first_brief=deepcopy(brief), original_scope='whole_file')
    response = _request(llm, request)
    entry = dict(request=request, response=deepcopy(response), accepted=False)
    audit['output_intent_review'] = entry
    if not isinstance(response, dict) or response.get('decision') != 'independent_named_source':
        return brief
    quote, source = response.get('instruction_quote'), response.get('source_description')
    if (not isinstance(quote, str) or not quote.strip() or quote not in payload['original_instruction']
            or not isinstance(source, str) or not source.strip() or len(source.split()) > 16
            or source.casefold() not in quote.casefold()):
        entry['rejection_reason'] = 'Named source must be a literal phrase within the exact output request quote'
        return brief
    corrected = deepcopy(brief)
    corrected.update(selection=dict(target='named_source', scope='whole_file', occurrence='unselected'),
        participant_selection=None, questions=[], placement_windows=[], timeline_policy=None,
        reason=response['reason'], output_intent=dict(mode='extract_source', source_description=source,
             instruction_quote=quote, reason=response['reason']))
    entry.update(accepted=True, corrected_brief=deepcopy(corrected))
    return corrected
