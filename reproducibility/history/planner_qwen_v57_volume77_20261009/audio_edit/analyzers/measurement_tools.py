"""Bounded pre-edit measurements with explicit, auditable semantic binding.

Waveform regions are NOT named events. The planner must associate an existing
heard source with a measured region; unknown identity, merged activity and
semantic utterance/action endings cannot be inferred from this tool.
"""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import subprocess

import numpy as np

from audio_edit.plans.temporal_boundaries import finite
from audio_edit.utils.execution_outcome import open_decoded_audio

PHYSICAL_FIELDS = {'onset', 'sound_end', 'interval'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def activity_regions(path, *, separated=False):
    """Measure the entire bounded file, independent of Qwen's timestamp guesses."""
    with open_decoded_audio(path) as source:
        sr, duration = source.samplerate, len(source)/source.samplerate
        if not 0 < duration <= 90:
            raise ValueError('Activity inventory supports recordings up to 90 seconds')
        samples = source.read(dtype='float32', always_2d=True)
    if not np.isfinite(samples).all():
        raise ValueError('Non-finite waveform')
    hop = max(1, round(.005*sr))
    energy = np.array([np.sqrt(np.mean(samples[i:i+hop].astype('float64')**2))
                       for i in range(0, len(samples), hop)])
    peak = float(energy.max())
    threshold = max(1e-7, peak*(.01 if separated else .0001))
    active = energy > threshold
    transitions = np.diff(np.pad(active.astype(int), (1, 1)))
    groups = []
    for a, b in zip(np.flatnonzero(transitions==1), np.flatnonzero(transitions==-1)):
        if groups and (a-groups[-1][1])*hop/sr <= .03:
            groups[-1][1] = int(b)
        else:
            groups.append([int(a), int(b)])
    groups = [g for g in groups if (g[1]-g[0])*hop/sr >= .015]
    if len(groups) > 64:
        raise ValueError('Too many activity regions for this bounded inventory')
    regions, gaps = [], []
    digest = sha256(path)
    for i, (a, b) in enumerate(groups):
        start, end = a*hop/sr, min(len(samples), b*hop)/sr
        previous_end = groups[i-1][1]*hop/sr if i else 0.
        next_start = groups[i+1][0]*hop/sr if i+1 < len(groups) else duration
        quiet_before, quiet_after = start-previous_end, next_start-end
        regions.append(dict(region_id=f'{digest[:10]}_r{i+1}', start_time=start, end_time=end,
            quiet_before=quiet_before, quiet_after=quiet_after,

            complete=quiet_before >= .06 and quiet_after >= .06,
            peak_rms=float(energy[a:b].max()), kind='unclassified_activity'))
        if quiet_before >= .06:
            gaps.append(dict(start_time=previous_end, end_time=start))
    if groups and duration-regions[-1]['end_time'] >= .06:
        gaps.append(dict(start_time=regions[-1]['end_time'], end_time=duration))
    return dict(tool='source_activity' if separated else 'activity_regions',
        audio_path=str(Path(path).resolve()), audio_sha256=digest, duration=duration,
        sample_rate=sr, resolution_seconds=hop/sr, threshold=threshold,
        regions=regions, quiet_gaps=gaps,
        method='5 ms multichannel RMS; bridge <=30 ms gaps; complete requires >=60 ms quiet on both sides',
        limitation=('Separated-source candidates may contain leakage or lose quiet tails; identity requires perceptual evidence.'
                    if separated else 'Mixed activity regions are not source identities, speech turns or semantic event counts.'))


def measurement_question_ids(questions):
    return [q['id'] for q in questions if q['focus']=='timing'
            and set(q['required_fields']) <= PHYSICAL_FIELDS]


def source_tool_available():


    return os.environ.get('AUDIO_PREEDIT_SEPARATION') == '1' and bool(os.environ.get('SLURM_JOB_ID'))


def source_question_eligible(question):
    """A source label cannot erase an inherited participant-identity dependency."""
    return (question.get('reference_kind') == 'sound_event'
            and 'participant_reference' not in question
            and 'participant_identity_question_id' not in question)


def run_measurement(name, path, questions, ctx):
    original_hash = sha256(path)
    if name == 'activity_regions':
        result = activity_regions(path)
    elif name == 'source_activity':
        if any(not source_question_eligible(q) for q in questions):
            raise ValueError('Source activity cannot locate speech, participants or unclassified references')
        if not source_tool_available() or len(questions) != 1:
            raise ValueError('Source activity needs an enabled dedicated GPU and one selected source')
        subprocess.run(['nvidia-smi'], check=True, capture_output=True)
        from audio_edit.models import AudioEditContext
        from audio_edit.plans.remove_sound_plan import RemoveSoundPlan
        from audio_edit.runners.samaudio_runner import SAMAudioRunner
        extraction = AudioEditContext('Pre-edit source measurement', input_audio=str(path))
        target = questions[0]['subject']
        extraction = SAMAudioRunner().run_remove(extraction, RemoveSoundPlan(
            backend='samaudio', mode='text', result_mode='result', text_prompt=target,
            notes='Analysis-only source extraction; original audio and user edit remain unchanged'))
        result = activity_regions(extraction.output_audio, separated=True)
        with open_decoded_audio(path) as original:
            original_duration = len(original)/original.samplerate
        if abs(result['duration']-original_duration) > .02:
            raise ValueError('Separated source does not preserve the original time axis')
        result.update(source_description=target, original_audio_sha256=original_hash,
                      time_axis=dict(origin='original_audio_start', offset_seconds=0,
                          original_duration=original_duration, units='seconds', cropped=False),
                      separation_calls=extraction.params.get('separation_calls', []))
        ctx.params.setdefault('preedit_source_measurements', []).append(deepcopy(result))
    elif name == 'speech_transcription':
        from audio_edit.analyzers.speech_evidence import run_speech_evidence
        result = run_speech_evidence(path, ctx=ctx)
    else:
        raise ValueError('Unknown evidence tool')
    if sha256(path) != original_hash:
        raise ValueError('The original recording changed while acquiring evidence')
    result['question_ids'] = [q['id'] for q in questions]
    return result


def measurement_candidates(analysis, measurement, questions):
    """Only a unique complete component intersecting a heard span can be bound.

    No nearest timestamp snapping or ordinal inference from anonymous energy.
    A full-file envelope crossing several components remains ambiguous.
    """
    if not measurement or measurement.get('tool') not in ('activity_regions','source_activity'):
        return []
    eligible = set(measurement_question_ids(questions)) & set(measurement['question_ids'])
    answers = {a['question_id']:a for a in (analysis or {}).get('answers', [])}
    events = (analysis or {}).get('events', [])
    candidates = []
    for event in events:
        a, b = event.get('start_time'), event.get('end_time')
        if not (finite(a) and finite(b) and a < b):
            continue
        matches = [r for r in measurement['regions'] if r['start_time'] < b and r['end_time'] > a]
        if len(matches) != 1 or not matches[0]['complete']:
            continue
        region = matches[0]


        if any(e is not event and not e.get('review_context_only')
               and set(e.get('question_ids', [])).intersection(event.get('question_ids', []))
               and finite(e.get('start_time')) and finite(e.get('end_time'))
               and e['start_time'] < region['end_time'] and e['end_time'] > region['start_time']
               and (e.get('occurrence_index') != event.get('occurrence_index')
                    or e['end_time'] <= a or e['start_time'] >= b) for e in events):
            continue


        if measurement['tool']=='activity_regions' and any(
            e is not event and not e.get('review_context_only')
            and finite(e.get('start_time')) and finite(e.get('end_time'))
            and e['start_time'] < region['end_time'] and e['end_time'] > region['start_time']
            and (e.get('event'),e.get('start_time'),e.get('end_time'),e.get('occurrence_index')) !=
                (event.get('event'),a,b,event.get('occurrence_index')) for e in events):
            continue
        for key in eligible.intersection(event.get('question_ids', [])):
            if answers.get(key, {}).get('status') != 'answered':
                continue
            candidates.append(dict(question_id=key, event_id=event['event_id'],
                region_id=region['region_id'], event=event['event'],
                observed_span=[a,b], measured_span=[region['start_time'],region['end_time']],
                prerequisite='Confirm this entire isolated activity is the heard target, not another source, combined events or a reflection.'))
    return candidates


def apply_measurement_bindings(analysis, measurement, questions, bindings):
    """Copy tool samples by validated ID; preserve Qwen identity and old estimate."""
    if not isinstance(bindings, list):
        raise ValueError('measurement_bindings must be an array')
    if not analysis:
        raise ValueError('A measurement cannot supply an unheard target identity')
    result = deepcopy(analysis)
    allowed = {(c['question_id'],c['event_id'],c['region_id'])
               for c in measurement_candidates(analysis, measurement, questions)}
    events = {e['event_id']:e for e in result.get('events', [])}
    regions = {r['region_id']:r for r in (measurement or {}).get('regions', [])}
    selected = {}
    for binding in bindings:
        if not isinstance(binding, dict) or set(binding) != {'question_id','event_id','region_id'}:
            raise ValueError('A measurement binding needs exactly question_id, event_id and region_id')
        key = (binding.get('question_id'),binding.get('event_id'),binding.get('region_id'))
        if key not in allowed:
            raise ValueError('Measurement binding has no unique isolated region supporting this observed event')
        if key[1] in selected and selected[key[1]] != key[2]:
            raise ValueError('The same event cannot bind conflicting regions')
        if key[2] in selected.values() and selected.get(key[1]) != key[2]:
            raise ValueError('One activity region cannot certify multiple separate events')
        selected[key[1]] = key[2]
    for event_id, region_id in selected.items():
        event, region = events[event_id], regions[region_id]
        before = deepcopy(event)
        event.update(start_time=region['start_time'],end_time=region['end_time'],timing_kind='interval',
                     timing_source=measurement['tool'],resolution_status='estimated',approximate=True)
        event['measurement_provenance'] = dict(tool=measurement['tool'],region_id=region_id,
            audio_sha256=measurement['audio_sha256'], original_audio_sha256=measurement.get('original_audio_sha256'),
            method=measurement['method'],resolution_seconds=measurement['resolution_seconds'],
            original_observation=before,
            identity_source='Qwen observation plus planner association; energy does not identify sources')
        event['boundaries'] = {role:dict(time=event[edge],status='estimated',approximate=True,
            timing_source=measurement['tool'], evidence='Measured isolated activity boundary; semantic identity from the linked Qwen observation',
            region_id=region_id) for role, edge in [('onset','start_time'),('sound_end','end_time')]}
        event.pop('boundary_activity_audit', None)
    if selected:
        result['evidence_source'] = 'qwen_identity_with_independent_measurement'
    return result


def unresolved_measurement_conflicts(analysis, measurement, questions):
    """A ready flag cannot ignore an applicable independent quiet transition."""
    missing = {}
    known = {q['id']:q for q in questions}
    for candidate in measurement_candidates(analysis, measurement, questions):
        event = next(e for e in analysis['events'] if e['event_id']==candidate['event_id'])
        if event.get('measurement_provenance', {}).get('region_id')==candidate['region_id']:
            continue
        q = known[candidate['question_id']]
        for role, index in [('onset',0),('sound_end',1)]:
            if role not in q['required_fields'] and 'interval' not in q['required_fields']:
                continue
            claim = event.get('boundaries', {}).get(role, {}).get('time', candidate['observed_span'][index])
            if abs(claim-candidate['measured_span'][index]) > .08:
                missing.setdefault(q['id'], []).append(
                    'Independent isolated activity differs from the claimed '+role+'. '
                    'Associate the measured region with the target, or resolve the source mismatch; do not ignore the measurement.')
    return missing


def source_handoff_gaps(analysis, measurement, question_ids):
    """A requested source measurement cannot be satisfied by stale Qwen times."""
    missing = {}
    for key in question_ids:
        events = [e for e in (analysis or {}).get('events',[]) if key in e.get('question_ids',[])]
        bound = bool(measurement and measurement.get('tool')=='source_activity' and events) and all(
            e.get('measurement_provenance',{}).get('tool')=='source_activity'
            and e['measurement_provenance'].get('audio_sha256')==measurement['audio_sha256']
            and e.get('resolution_status')!='unresolved' for e in events)
        if not bound:
            missing[key] = ['Source measurement was requested to resolve this timing dependency. '
                'Its result has not been bound to executable events. Bind a supported complete candidate, '
                'or request an available audio observation to resolve the location/association. '
                'Tool failure, intervals in review prose, and the unchanged pre-tool estimate cannot close this dependency.']
    return missing
