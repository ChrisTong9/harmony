"""Ask about the requested volume targets without changing the owned edit plan."""
from copy import deepcopy
from dataclasses import asdict
import math
from audio_edit.analyzers.removal_focus import attach_evidence, sole_source_decision

VOLUME_OBSERVATION_PROMPT = '''
VOLUME OBSERVATION: the instruction requests gain changes, not source deletion.
Find the existing timing references in events. Preserve each action's own
reference, occurrence and interval; do not combine regions from different actions.
For each source_question also listen for the requested target and all OTHER
sounds throughout every assessment window. Decide whether directly changing that
window's waveform would affect any non-target sound. A dominant instrument, one
event label, or a failed separation is not evidence that only the target exists.
Include quiet speech, other instruments, ambience and background noise in the
other-source check. The target's own decay and silence between notes are allowed.
Additionally return in the SAME JSON response:
"volume_assessment": {"actions": [{"action_index": integer, "windows": [
{"window_index": integer, "target_present": true/false/null,
 "only_target": true/false/null, "other_sounds": ["other audible sources"] or null,
 "evidence": "concrete audible basis for the entire window"}]}]}.
Use the supplied zero-based action/window indices. only_target=true requires the
target to be present and no other audible source to preserve ANYWHERE in the
window. An absent target is not a sole target. Use null when uncertain or masked;
other_sounds=[] means none heard with sufficient certainty, not unchecked.
The user instruction and old candidate observations are search context, not proof.
Return no mixing advice, revised gains, or revised edit ranges.
'''


def volume_windows(plan, duration):
    if plan.placements:
        windows = [(p.start_time, p.end_time) for p in plan.placements]
    elif plan.temporal is None or plan.temporal.kind == 'unspecified':
        windows = [(0., duration)]
    else:
        raise ValueError('Volume scope has no resolved placements')
    if not windows or any(not all(type(v) in (int, float) and math.isfinite(v) for v in (a, b))
                          or not 0 <= a < b <= duration for a, b in windows):
        raise ValueError('Volume scope has invalid boundaries')
    return windows


def observation_context(actions, duration):
    references, questions = [], []
    for index, plan in enumerate(actions):
        if plan.temporal:
            references.append(dict(action_index=index, gain_profile=plan.gain_profile,
                                   temporal=asdict(plan.temporal)))
        targets = plan.effective_target_sounds()


        if plan.apply_to != 'target_source' or len(targets) != 1:
            continue
        try:
            windows = volume_windows(plan, duration)
        except ValueError:
            windows = [(0., duration)]
        questions.append(dict(action_index=index, target_sound=targets[0],
            gain_profile=plan.gain_profile, gain_db=plan.gain_db,
            assessment_windows=[dict(window_index=i, start_time=a, end_time=b)
                                for i, (a, b) in enumerate(windows)]))
    return dict(operation='volume', references=references, source_questions=questions)


def attach_volume_evidence(data, context, audio_evidence):
    raw = data.get('volume_assessment')
    actions = raw.get('actions', []) if isinstance(raw, dict) else []
    actions = actions if isinstance(actions, list) else []
    evidence = []
    for question in context['source_questions']:
        matches = [a for a in actions if isinstance(a, dict)
                   and type(a.get('action_index')) is int
                   and a['action_index'] == question['action_index']]
        if len(matches) != 1:
            continue
        attached = attach_evidence(dict(source_assessment=matches[0]), question, audio_evidence,
            assessment_key='source_assessment', evidence_key='source_evidence', source='qwen_directed_volume')
        evidence.append(dict(action_index=question['action_index'], source_evidence=attached['source_evidence']))
    data['volume_evidence'] = evidence
    return data


def direct_gain_decision(plan, input_audio, duration):
    decision = dict(eligible=False, reason='Source separation required without confirmed sole-source evidence')
    targets = plan.effective_target_sounds()
    if (plan.apply_to != 'target_source' or len(targets) != 1 or plan.temporal_schema_error
            or plan.temporal_status not in ('not_requested', 'resolved')):
        return decision
    try:
        windows = volume_windows(plan, duration)
    except ValueError as exc:
        return dict(decision, reason=str(exc))
    return sole_source_decision(plan.source_evidence, input_audio, duration, windows,
                                targets[0], source='qwen_directed_volume')
