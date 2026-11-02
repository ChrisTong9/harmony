"""One heard event sequence may span several independently measured components."""

from audio_edit.analyzers.measurement_tools import source_question_eligible


GROUP_FIELDS = {'timing_kind', 'event', 'occurrence_index', 'region_ids',
                'group_relation', 'association_status', 'association_evidence'}


def supports_group(question):
    return (source_question_eligible(question)
            and question.get('occurrence') == 'one'
            and question.get('required_fields') == ['interval']
            and question.get('measured_regions', {}).get('tool') == 'source_activity')


def measured_group(question, event):
    """Compile a positively heard group; never choose/merge candidates by distance."""
    if not supports_group(question) or set(event) != GROUP_FIELDS:
        raise ValueError('Measured groups require one sound-event interval and source-region IDs only')
    ids = event['region_ids']
    if (not isinstance(ids, list) or len(ids) < 2
            or any(not isinstance(x, str) for x in ids) or len(ids) != len(set(ids))):
        raise ValueError('A measured group requires at least two distinct region IDs')
    if event['group_relation'] != 'one_event_sequence':
        raise ValueError('Grouping requires an explicit same-event sequence relation')
    regions = question['measured_regions']['regions']
    indexed = {r['region_id']: r for r in regions}
    if any(key not in indexed or not indexed[key]['complete'] for key in ids):
        raise ValueError('Every grouped component must be an available complete source region')
    selected = [indexed[key] for key in ids]
    if any(a['end_time'] > b['start_time'] for a, b in zip(selected, selected[1:])):
        raise ValueError('Grouped components must be ordered and non-overlapping')
    start, end = selected[0]['start_time'], selected[-1]['end_time']
    covered = [r['region_id'] for r in regions if r['start_time'] < end and r['end_time'] > start]
    if set(covered) != set(ids):
        raise ValueError('A group cannot skip intervening source candidates; resolve their identity first')
    return dict(start_time=start, end_time=end, complete=True, region_ids=list(ids),
                components=[{k: r[k] for k in ('region_id', 'start_time', 'end_time')} for r in selected])
