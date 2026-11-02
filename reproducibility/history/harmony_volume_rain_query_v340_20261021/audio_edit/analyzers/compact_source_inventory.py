"""One perceptual verdict per source region, with no duplicate event payload.

The wire representation changes; identity, timing and coverage validation do not.
Only an explicitly matched region can produce the existing measured event form.
"""
from copy import deepcopy

from audio_edit.analyzers.source_inventory import required_regions, inventory_schema


def eligible(question):
    return (question.get('focus') == 'timing'
            and set(question.get('required_fields', [])) == {'interval'}
            and question.get('measured_regions', {}).get('tool') == 'source_activity'
            and not question.get('speech_references')
            and isinstance(question.get('subject'), str) and bool(question['subject'].strip())
            and bool(required_regions(question)))


def compact_response_format(response_format, questions):
    result = deepcopy(response_format)
    from audio_edit.analyzers.listening_protocol import obj
    for q in questions:
        if not eligible(q):
            continue
        inventory = inventory_schema(q)
        for verdict in inventory['properties'].values():
            verdict['properties']['evidence']['maxLength'] = 160
        fact = obj(dict(status=dict(type='string', enum=['answered', 'uncertain', 'not_found']),
                        answer=dict(type='string', minLength=1, maxLength=320),
                        region_assessments=inventory))
        result['json_schema']['schema']['properties']['facts']['properties'][q['id']] = fact
    return result


def compact_instruction(questions):
    ids = [q['id'] for q in questions if eligible(q)]
    if not ids:
        return ''
    return ('\nCOMPACT SOURCE INVENTORY for questions ' + ', '.join(ids) + ': '
        'Return only status, a short answer, and region_assessments in each such fact. '
        'Do NOT also output events, occurrence numbers, times, or retained_event_ids for these questions. '
        'This replaces the events output requirement for these questions only. '
        'Classify EVERY offered region exactly once as match/no_match/unknown, based on the attached audio. '
        'Give a brief AUDIBLE identity cue for each verdict (at most 160 characters); repeating its times '
        'is not an audible cue. The program binds each explicitly matched region once using the supplied '
        'region times. Never repeat a region to reach a previously estimated count. '
        'A separated region is not automatically a target. Use uncertain if any candidate is unknown '
        'or the offered regions do not cover all requested occurrences you hear. '
        'A missing candidate, unresolved tail or extra audible occurrence must be reported in the answer; '
        'do not invent a region or call a partial inventory complete. Keep other question formats unchanged.')


def expand_compact_inventories(data, questions):
    """Expand only fully explicit, structurally valid verdicts; never fix identity."""
    for q in questions:
        if not eligible(q):
            continue
        fact = data.get('facts', {}).get(q['id'])
        if not isinstance(fact, dict) or set(fact) != {'status', 'answer', 'region_assessments'}:
            raise ValueError('Compact source fact must contain status, answer and region_assessments only')
        inventory = fact['region_assessments']
        region_ids = required_regions(q)
        if (fact['status'] not in ('answered', 'uncertain', 'not_found')
                or not isinstance(fact['answer'], str) or not fact['answer'].strip()
                or not isinstance(inventory, dict) or set(inventory) != set(region_ids)):
            raise ValueError('Compact source fact must explicitly classify every offered region')
        for verdict in inventory.values():
            if (not isinstance(verdict, dict) or set(verdict) != {'status', 'evidence'}
                    or verdict['status'] not in ('match', 'no_match', 'unknown')
                    or not isinstance(verdict['evidence'], str) or not verdict['evidence'].strip()):
                raise ValueError('Compact source verdict requires a status and audible evidence')
        events = []
        if fact['status'] == 'answered':
            for region_id in region_ids:
                verdict = inventory[region_id]
                if verdict['status'] == 'match':
                    events.append(dict(event=q['subject'], occurrence_index=len(events)+1,
                        timing_kind='measured_region', region_id=region_id,
                        association_status='match', association_evidence=verdict['evidence']))


        fact['events'] = events
        data.setdefault('schema_adaptations', []).append(dict(
            kind='compact_source_inventory', question_id=q['id'],
            matched_region_ids=[e['region_id'] for e in events]))
