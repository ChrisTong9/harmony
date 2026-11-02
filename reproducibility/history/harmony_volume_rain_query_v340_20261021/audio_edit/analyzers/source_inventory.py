"""An 'each' source request needs a verdict for every offered candidate.

Coverage is not identity: only explicit positive associations may become events.
"""
from copy import deepcopy


def required_regions(question):
    measured = question.get('measured_regions', {})
    if question.get('occurrence') == 'each' and measured.get('inventory_required'):
        return [r['region_id'] for r in measured.get('regions', [])]
    return []


def inventory_schema(question):
    def obj(properties):
        return dict(type='object', additionalProperties=False, properties=properties, required=list(properties))
    verdict = obj(dict(status=dict(type='string', enum=['match','no_match','unknown']),
                       evidence=dict(type='string', minLength=1)))
    return obj({key: deepcopy(verdict) for key in required_regions(question)})


def validate_source_inventories(data, questions):
    """Reject omitted candidates without manufacturing any source identity."""
    facts = data.get('facts', {})
    for question in questions:
        regions = required_regions(question)
        if not regions:
            continue
        key = question['id']
        fact = facts.get(key)
        if not isinstance(fact, dict):
            continue
        inventory = fact.get('region_assessments')
        error = None
        if not isinstance(inventory, dict) or set(inventory) != set(regions):
            error = 'Each-occurrence source selection needs an explicit verdict for every offered region.'
        elif any(not isinstance(v, dict) or set(v) != {'status','evidence'}
                 or v.get('status') not in ('match','no_match','unknown')
                 or not isinstance(v.get('evidence'), str) or not v['evidence'].strip()
                 for v in inventory.values()):
            error = 'Every source-region verdict needs match/no_match/unknown and audible evidence.'
        elif fact.get('status') == 'answered':
            matches = {region for region, verdict in inventory.items() if verdict['status']=='match'}
            events = fact.get('events', [])
            selected = [e.get('region_id') for e in events if isinstance(e,dict)
                        and e.get('timing_kind')=='measured_region' and e.get('association_status')=='match']
            if any(v['status']=='unknown' for v in inventory.values()):
                error = 'Unknown source candidates leave the requested each-occurrence inventory unresolved.'
            elif (not matches or len(selected)!=len(events) or len(selected)!=len(set(selected))
                  or set(selected)!=matches or 'retained_event_ids' in fact):
                error = 'Events must represent exactly the explicitly matched source regions, once each.'
        data.setdefault('source_region_assessments', []).append(dict(
            question_id=key, audio_sha256=question['measured_regions']['audio_sha256'],
            assessments=deepcopy(inventory), complete=error is None and fact.get('status')=='answered'))
        if error:
            facts[key] = dict(status='uncertain', answer=error, events=[])
            data.setdefault('measurement_reference_errors', []).append(dict(question_id=key,
                audio_sha256=question['measured_regions']['audio_sha256'],
                region_id=None, association_status='incomplete_inventory', reason=error))


def source_inventory_gaps(analysis, measurement, questions, pending_ids):
    """Planner bindings cannot discharge an each-source request with a subset."""
    if not measurement or measurement.get('tool') != 'source_activity':
        return {}
    missing = {}
    regions = {r['region_id'] for r in measurement['regions'] if r['complete']}
    for q in questions:
        if q['id'] not in pending_ids or q.get('occurrence') != 'each':
            continue
        records = [r for r in (analysis or {}).get('source_region_assessments', [])
                   if r.get('question_id')==q['id'] and r.get('audio_sha256')==measurement['audio_sha256']]
        record = records[-1] if records else {}
        inventory = record.get('assessments') or {}
        events = [e for e in (analysis or {}).get('events', []) if q['id'] in e.get('question_ids', [])]
        selected = [e.get('measurement_provenance',{}).get('region_id') for e in events]
        matches = {k for k,v in inventory.items() if isinstance(v,dict) and v.get('status')=='match'}
        if (not record.get('complete') or set(inventory)!=regions or not matches
                or any(not isinstance(v,dict) or v.get('status') not in ('match','no_match')
                       or not isinstance(v.get('evidence'),str) or not v['evidence'].strip()
                       for v in inventory.values())
                or len(selected)!=len(set(selected)) or set(selected)!=matches):
            missing[q['id']] = ['The each-occurrence request needs a verdict for every complete source candidate '
                'and executable events for exactly the positively matched candidates. Confirm the full inventory '
                'in the remaining observation; a planner binding or a single-example probe cannot replace coverage.']
    return missing
