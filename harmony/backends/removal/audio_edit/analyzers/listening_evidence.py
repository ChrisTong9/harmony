"""Small factual listening contract, using the executor's existing boundaries."""
from copy import deepcopy

from audio_edit.plans.event_ids import assign_event_ids
from audio_edit.plans.temporal_boundaries import event_boundary, finite, select_event
from audio_edit.plans.boundary_activity import verify_boundary_activity


FACTS = {
    'identity': 'whether the existing source/reference is audible and what it is; distinguish similar sounds',
    'onset': 'its onset, recorded as boundaries.onset with time and audible evidence',
    'action_end': 'the end of the action itself, recorded as boundaries.action_end, separately from its decay',
    'sound_end': 'the end of its audible sound/decay, recorded as boundaries.sound_end',
    'utterance_end': 'the end of the spoken utterance, recorded as boundaries.utterance_end; a generic sound end is insufficient',
    'interval': 'each complete start_time/end_time interval of the requested existing reference',
    'overlap': 'which other original sources overlap it, including quiet voices and ambience; report uncertainty',
    'rhythm': 'the heard attacks as rhythm.onsets_seconds with evidence; do not infer a regular grid',
}

FOCUS_FIELDS = {'target': {'identity'}, 'timing': set(FACTS) - {'identity', 'overlap'},
                'preservation': {'overlap'}}

SELECTION_FIELDS = {
    'target': ('whole_mix', 'named_source', 'ambiguous_source'),
    'scope': ('whole_file', 'literal_time', 'event_condition'),
    'occurrence': ('unselected', 'selected'),
}


REFERENCE_KINDS = ('sound_event', 'speech', 'participant', 'unknown')

def selection_needs_audio(selection):
    """Gate questions by instruction selection, not hypothetical source overlap."""
    if not isinstance(selection, dict) or any(selection.get(k) not in values
                                             for k, values in SELECTION_FIELDS.items()):
        raise ValueError('Listening selection must classify target, scope and occurrence')
    return (selection['target'] == 'ambiguous_source' or selection['scope'] == 'event_condition'
            or selection['occurrence'] == 'selected')


def answer_contract(question):
    """Specify machine-readable evidence without replacing the planner's question."""
    facts = '; '.join(FACTS[field] for field in question['required_fields'])
    return (f"For the existing sound/reference {question['subject']!r}, occurrence "
            f"{question['occurrence']!r}, record: {facts}. "
            f"Put its observations in facts[{question['id']!r}].events. "
            'Use full-file seconds, mark estimates, and say when a fact cannot be heard. '
            'Report original-audio facts only; do not choose editing parameters or user preferences.')


def questions_from_brief(data, *, force_audio=False):
    questions = deepcopy(data.get('questions'))
    if not isinstance(questions, list) or not 0 <= len(questions) <= 6:
        raise ValueError('Listening needs 0-6 decision-relevant questions')


    if 'selection' in data:
        needs_audio = selection_needs_audio(data['selection']) or force_audio
        if not needs_audio:
            questions = []
        elif not questions:
            raise ValueError('An unresolved event/source selection needs listening questions')
    if not questions and (not isinstance(data.get('reason'), str) or not data['reason'].strip()):
        raise ValueError('Skipping listening needs a reason grounded in the instruction')
    if force_audio and not questions:
        raise ValueError('A participant selection needs listening questions')
    for index, q in enumerate(questions):
        if not isinstance(q, dict) or any(not isinstance(q.get(k), str) or not q[k].strip()
                                         for k in ('focus', 'subject', 'question', 'why')):
            raise ValueError('Each listening requirement needs focus, subject, question and decision relevance')
        fields = q.get('required_fields')
        if not isinstance(fields, list) or not fields or any(f not in FACTS for f in fields):
            raise ValueError('Listening requirements need supported factual fields')
        if q['focus'] not in FOCUS_FIELDS or not set(fields) <= FOCUS_FIELDS[q['focus']]:
            raise ValueError('Listening focus and factual fields disagree')
        occurrence = q.get('occurrence')
        if not (type(occurrence) is int and occurrence > 0) and occurrence not in ('one', 'first', 'last', 'each'):
            raise ValueError('Invalid listening occurrence')
        q.setdefault('reference_kind', 'unknown')
        if q['reference_kind'] not in REFERENCE_KINDS:
            raise ValueError('Unsupported listening reference kind')
        if q.get('speech_scope') not in (None,'all_spoken_activity','selected_content'):
            raise ValueError('Unsupported speech scope')
        if q.get('speech_scope')=='all_spoken_activity' and not (
                q['reference_kind']=='speech' and q['focus']=='timing'
                and q['occurrence']=='each' and set(fields)=={'interval'}):
            raise ValueError('All-spoken-activity scope requires each speech interval, not a selected boundary')
        q['id'] = 'q' + str(index + 1)
        q['required_fields'] = list(dict.fromkeys(fields))
        q['answer_contract'] = answer_contract(q)
    return questions


def observation_view(analysis, duration):
    """Keep observation fields, not Qwen's proposed edit/mix or planner decisions."""
    fields = ('events', 'answers', 'source_assessments', 'uncertainties',
              'current_audio_profile', 'audio_evidence', 'schema_adaptations', 'schema_normalizations',
              'measurement_reference_errors', 'unlocated_observations', 'evidence_needs',
              'speech_reference_errors', 'speech_evidence', 'source_region_assessments', 'speech_segment_assessments')
    data = {k: deepcopy(analysis[k]) for k in fields if k in analysis}
    data.update(media_duration=duration, mixing_recommendations={},
                evidence_source='qwen_observation', instruction_authority=False)
    for event in data.get('events', []):
        event['approximate'] = True
        source = ('speech_reference' if event.get('speech_reference_provenance') else
                  event.get('measurement_provenance', {}).get('tool', 'qwen_observation'))
        event['timing_source'] = source
        if source == 'speech_reference':
            data['evidence_source'] = 'qwen_semantics_with_approximate_lexical_and_acoustic_references'
        elif source != 'qwen_observation':
            data['evidence_source'] = 'qwen_identity_with_independent_measurement'
        for phase in (event.get('boundaries') or {}).values():
            if isinstance(phase, dict):
                phase.update(approximate=True, timing_source=source)
                if phase.get('status') == 'resolved':
                    phase['status'] = 'approximate'
    assign_event_ids(data, 'audio', duration)
    return data


def placement_windows_from_brief(brief, questions, operation):
    """Carry paired placement dependencies from the instruction, before hearing audio."""
    windows = deepcopy(brief.get('placement_windows', []))
    if not isinstance(windows, list) or len(windows) > 3:
        raise ValueError('placement_windows must be a list of up to three paired references')
    if windows and operation != 'add_sound':
        raise ValueError('Addition placement windows cannot change another operation scope')
    known = {q['id']: q for q in questions}
    for window in windows:
        if not isinstance(window, dict) or set(window) != {'start_question_id', 'end_question_id'}:
            raise ValueError('A placement window needs exactly two question references')
        if window['start_question_id'] == window['end_question_id']:
            q = known.get(window['start_question_id'], {})
            if (q.get('focus') == 'timing' and q.get('required_fields') == ['interval']
                    and q.get('reference_kind') in ('sound_event', 'speech', 'participant')
                    and q.get('occurrence') != 'each'):


                continue
            raise ValueError('Flanking references need separate questions or one complete selected interval')
        for key in window.values():
            q = known.get(key, {})
            roles = q.get('required_fields', [])
            if (q.get('focus') != 'timing' or len(roles) != 1
                    or roles[0] not in ('onset', 'action_end', 'sound_end', 'utterance_end')
                    or q.get('occurrence') == 'each'):
                raise ValueError('A paired placement endpoint needs one selected semantic boundary')
    return windows


def timeline_policy_from_brief(brief, instruction, operation):
    """Interpret timeline permission once from user text, never from listening."""
    if operation != 'add_sound':
        return None
    policy = deepcopy(brief.get('timeline_policy') or dict(mode='preserve', user_quote=None))
    if not isinstance(policy, dict) or set(policy) != {'mode', 'user_quote'}:
        raise ValueError('Timeline policy needs mode and the supporting user quote')
    if policy['mode'] not in ('preserve', 'retime'):
        raise ValueError('Unknown timeline policy')
    quote = policy['user_quote']
    if policy['mode'] == 'retime':
        if not isinstance(quote, str) or not quote.strip() or quote not in instruction:
            raise ValueError('Retiming needs an exact instruction quote, not a planner justification')
    elif quote is not None:
        raise ValueError('Preserving the timeline is the default, not a quoted retiming request')
    return policy


def check_placement_windows(windows, questions, analysis):
    """Check the relation while there is still a chance to clarify it with Qwen."""
    if not analysis:
        return {}
    known = {q['id']: q for q in questions}
    missing = {}
    for window in windows:
        if window['start_question_id'] == window['end_question_id']:


            continue
        times = []
        for key in (window['start_question_id'], window['end_question_id']):
            q = known[key]
            events = {e['event_id']: e for e in analysis.get('events', [])
                      if key in (e.get('question_ids') or [])}
            try:
                time, _, _ = event_boundary(dict(event_id=list(events),
                    occurrence=None if q['occurrence'] == 'one' else q['occurrence'],
                    boundary_role=q['required_fields'][0]), events)
                times.append(time)
            except (ValueError, TypeError, KeyError):
                break
        if len(times) == 2 and times[1] <= times[0]:
            missing[window['end_question_id']] = [
                'The reported flanking boundaries leave no positive placement window. '
                'Recheck their order/overlap; do not move either reference to create space.']
    return missing


def check_evidence(questions, analysis, duration):
    """A fluent 'ready' cannot bypass missing fields or executor boundary checks."""
    if analysis is None:
        return {q['id']: ['No valid listening analysis'] for q in questions}
    answers = {a['question_id']: a for a in analysis.get('answers', [])}
    missing = {}
    for q in questions:
        issues = []
        answer = answers.get(q['id'], {})
        fields = q['required_fields']

        accepted = ('answered', 'uncertain') if fields == ['overlap'] else ('answered',)
        if answer.get('status') not in accepted:
            issues.append('Factual answer is missing, uncertain, or not found')
        if q['focus'] == 'target':
            linked = [e for e in analysis.get('events', [])
                      if q['id'] in (e.get('question_ids') or [])
                      and e.get('resolution_status') != 'unresolved']
            if not linked:
                issues.append('Identity is asserted only in prose: identify the actual observed '
                              'source/occurrences in this question\'s events with distinguishing audible cues')
        if q['focus'] == 'timing':
            linked = [e for e in analysis.get('events', [])
                      if isinstance(e.get('question_ids'), list) and q['id'] in e['question_ids']]
            events = {e['event_id']: e for e in linked
                      if e.get('resolution_status') != 'unresolved'}
            participant_identity = q.get('participant_identity_question_id')
            if participant_identity and any(participant_identity not in e.get('question_ids', [])
                                             for e in events.values()):
                issues.append('The located events are not linked to the selected participant identity. '
                    'Identify that person\'s target events and link the same observations to both questions; '
                    'a voice inventory or another speaker\'s interval cannot authorize this edit.')
            try:
                ids = list(events)
                if not linked:
                    raise ValueError(f"No event links to {q['id']}; include this ID in the relevant event's question_ids")
                if q['occurrence'] == 'each':
                    if not ids or len(events) != len(linked):
                        raise ValueError('No complete occurrences linked to this question')
                    selected = list(events.values())
                else:
                    occurrence = None if q['occurrence'] == 'one' else q['occurrence']
                    event, _ = select_event(dict(event_id=ids, occurrence=occurrence), events)
                    selected = [event]
                for event in selected:
                    for field in fields:
                        if field == 'interval':
                            a, b = event.get('start_time'), event.get('end_time')
                            if not (finite(a) and finite(b) and 0 <= a < b <= duration
                                    and not event.get('partial_start') and not event.get('partial_end')):
                                raise ValueError('Missing complete interval')
                            checks = [c for c in event.get('boundary_activity_audit', [])
                                      if c.get('boundary_role') == 'sound_end'
                                      and c.get('claimed_boundary') == b]
                            if checks and checks[-1].get('status') == 'conflict':
                                raise ValueError('Event interval end conflicts with isolated activity')
                        elif field == 'rhythm':
                            rhythm = event.get('rhythm') or {}
                            onsets = rhythm.get('onsets_seconds', [])
                            if (not isinstance(onsets, list) or len(onsets) < 3 or not rhythm.get('evidence')
                                    or not all(finite(t) and 0 <= t <= duration for t in onsets)
                                    or any(a >= b for a, b in zip(onsets, onsets[1:]))):
                                raise ValueError('Missing measured rhythm attacks')
                        else:
                            event_boundary(dict(event_id=event['event_id'], boundary_role=field), events)
            except (ValueError, TypeError, KeyError) as exc:
                issues.append(str(exc))
        if issues:
            missing[q['id']] = issues
    return missing


def check_boundary_activity(questions, analysis, input_audio):
    """Check a requested envelope without inventing a semantic endpoint."""
    requests = []
    for q in questions:
        ids = [e['event_id'] for e in analysis.get('events', [])
               if isinstance(e.get('question_ids'), list) and q['id'] in e['question_ids']]
        for role in _required_end_roles(q):
            if ids:
                requests.append(dict(event_id=ids, boundary_role=role,
                    occurrence=None if q['occurrence'] == 'one' else q['occurrence']))
    return verify_boundary_activity(analysis, requests, input_audio, repair=False)


def _required_end_roles(question):
    roles = set(question['required_fields']) & {'sound_end', 'utterance_end'}


    if 'interval' in question['required_fields'] and not roles:
        roles.add('sound_end')
    return sorted(roles)


def boundary_decision_risks(questions, analysis, activity):
    """Expose limitations and conflicts for the planner's next evidence decision.

    Only requested event boundaries count. Fixed file scopes and source overlap
    do not depend on such an endpoint and must not acquire one here.
    These are review obligations on the first turn, not proof of a wrong time.
    """
    if analysis is None:
        return {}
    risks = {}
    for q in questions:
        if q['focus'] != 'timing':
            continue
        ids = {e['event_id'] for e in analysis.get('events', [])
               if q['id'] in (e.get('question_ids') or [])}
        checks = [c for c in activity if c.get('event_id') in ids
                  and c.get('boundary_role') in _required_end_roles(q)
                  and c.get('status') in ('ambiguous', 'conflict')]
        if checks:
            conflict = any(c.get('status') == 'conflict' for c in checks)
            risks[q['id']] = dict(
                decision=q['why'], reference=q['subject'], checks=deepcopy(checks),
                evidence_effect='measured_conflict_with_reported_claim' if conflict else 'inconclusive',
                issue=('A measured activity/quiet transition disagrees with the reported claim. '
                       'Identify the source and assess whether a later observation corrects or explains that claim.'
                       if conflict else
                       'The tool cannot establish or refute the target endpoint in this mixture. '
                       'Do not turn this into missing evidence or a contradiction by itself; '
                       'assess the current source-specific cues and required scope.'))
    return risks
