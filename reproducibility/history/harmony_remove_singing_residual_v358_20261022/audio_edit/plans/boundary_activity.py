"""Conservative waveform contradiction checks for existing event END references.

This is not source recognition or forced alignment. Only isolated activity with
observed quiet context supports a correction, and corrections remain estimated.
No asset duration, sound-name rules, evaluation labels or EOF-position heuristics.
"""
from copy import deepcopy
import math

from audio_edit.plans.temporal_boundaries import boundary_measurement, finite, select_event


def _anchors(value):
    if isinstance(value, dict):
        if 'event_id' in value:
            yield value
        else:
            for child in value.values():
                yield from _anchors(child)
    elif isinstance(value, list):
        for child in value:
            yield from _anchors(child)


def verify_boundary_activity(analysis, requests, input_audio, *, repair=False):
    """Flag a contradiction before review; optionally estimate AFTER that review.

    sound_end requires near-digital silence (preserving faint decay). For
    utterance_end, an isolated activity/quiet transition is a conservative speech
    completion proxy, not transcription. Overlapping sources make it ambiguous.
    Explicit timestamps and media edges have no event references and are untouched.
    """
    import numpy as np
    from audio_edit.utils.execution_outcome import open_decoded_audio
    result = deepcopy(analysis or {})
    events = {e.get('event_id'):e for e in result.get('events', [])}
    targets = {}
    for anchor in _anchors(requests):
        role = anchor.get('boundary_role')
        if role not in ('sound_end', 'utterance_end'):
            continue
        ids = anchor.get('event_id')
        ids = ids if isinstance(ids, list) else [ids]
        selectors = [dict(anchor, event_id=i, occurrence=None) for i in ids] if anchor.get('occurrence')=='each' else [anchor]
        for selector in selectors:
            try:
                event, _ = select_event(selector, events)
            except (ValueError, TypeError, KeyError):
                continue
            targets[(event['event_id'],role)] = event
    audits = []
    if not targets:
        return result, audits
    with open_decoded_audio(input_audio) as source:
        sr, duration = source.samplerate, len(source)/source.samplerate
        hop = max(1, round(.01*sr))
        for (event_id,role), event in targets.items():
            measurement = boundary_measurement(event,role)
            a,b = event.get('start_time'),event.get('end_time')
            phases = event.get('boundaries')
            raw = phases.get(role) if isinstance(phases, dict) else None


            raw_usable = (isinstance(raw, dict) and finite(raw.get('time'))
                and finite(a) and finite(b) and a <= raw['time'] <= b
                and not raw.get('partial') and raw.get('evidence'))
            claimed = raw['time'] if raw_usable else event.get('end_time')
            if not all(finite(x) for x in (a,b,claimed)) or not 0 <= a < b <= duration:
                continue
            audit = dict(event_id=event_id, boundary_role=role, claimed_boundary=claimed,
                stage='after_review' if repair else 'before_review', status='unchecked')
            audits.append(audit)
            def uncertain(reason, **details):
                audit.update(status='ambiguous', reason=reason, **details)


                if measurement is not None:
                    event.setdefault('boundaries', {})[role] = dict(measurement,
                        status='estimated' if repair else 'approximate', approximate=True,
                        activity_verification=deepcopy(audit))
                event.setdefault('boundary_activity_audit', []).append(deepcopy(audit))


            lo,hi = max(a, claimed-12.), min(duration, max(b,claimed+3.))
            audit['search_context'] = dict(start_seconds=lo, end_seconds=hi)
            if hi-lo > 15:
                uncertain('Claim and observed envelope require a wider search than this bounded check')
                continue
            concurrent = [other['event_id'] for other in events.values()
                if other is not event and not other.get('review_context_only')
                and other.get('refines_event_id') != event_id
                and finite(other.get('start_time')) and finite(other.get('end_time'))
                and other.get('resolution_status') != 'unresolved'

                and other['start_time'] < claimed and other['end_time'] > claimed]
            if concurrent:
                uncertain('Other observed sources overlap the claimed boundary; mixed energy is not target evidence',
                    overlapping_event_ids=concurrent)
                continue
            first,last = max(0,math.floor(lo*sr)),min(len(source),math.ceil(hi*sr))
            source.seek(first)
            data=source.read(last-first,dtype='float32',always_2d=True)
            energy=np.array([np.sqrt(np.mean(data[i:i+hop].astype('float64')**2)) for i in range(0,len(data),hop)])
            if not len(energy) or not np.isfinite(energy).all() or energy.max()<1e-7:
                uncertain('No usable activity in the candidate observation; silence is not invalid audio')
                continue

            threshold=max(1e-7,float(energy.max())*.01) if role=='utterance_end' else 1e-7
            active=energy>threshold
            transitions=np.diff(np.pad(active.astype(int),(1,1)))
            groups=[]
            for start,end in zip(np.flatnonzero(transitions==1),np.flatnonzero(transitions==-1)):
                if groups and (start-groups[-1][1])*hop/sr <= .12:
                    groups[-1][1]=int(end)
                else:
                    groups.append([int(start),int(end)])


            audit['activity_components'] = [dict(
                start_seconds=(first+start*hop)/sr,
                end_seconds=min(last, first+end*hop)/sr,
                left_clipped=start == 0 and first > 0,
                right_clipped=first+end*hop >= last and last < len(source))
                for start, end in groups[:64]]
            audit['activity_components_truncated'] = len(groups) > 64


            groups=[g for g in groups if (first+g[0]*hop)/sr < min(b,claimed)-1/sr
                    and min(last,first+g[1]*hop)/sr > a]
            if not groups:
                uncertain('No measured activity overlaps the reported sound span; nearby components are not identified sources',
                          issue_code='no_activity_in_reported_span')
                continue
            if len(groups) > 1:
                uncertain('The reported span crosses separate activity components with intervening quiet; cannot assign them to one source or call the later component a continuation',
                          issue_code='separated_activity_components')
                continue


            component = groups[-1]
            component_start = (first+component[0]*hop)/sr
            component_end = min(last,first+component[1]*hop)/sr
            neighbours = [other['event_id'] for other in events.values()
                if other is not event and not other.get('review_context_only')
                and other.get('refines_event_id') != event_id
                and other.get('resolution_status') != 'unresolved'
                and finite(other.get('start_time')) and finite(other.get('end_time'))
                and other['start_time'] <= claimed + .02 + 1/sr
                and other['start_time'] < component_end and other['end_time'] > component_start]
            if neighbours:
                uncertain('The same energy component contains another observed event; its end is not target-only evidence',
                    overlapping_component_event_ids=neighbours)
                continue
            end=groups[-1][1]
            candidate=min(last,first+end*hop)/sr
            quiet_frames=max(1,math.ceil(.12*sr/hop))
            quiet=energy[end:end+quiet_frames]
            if len(quiet)<quiet_frames or np.any(quiet>threshold):
                if abs(candidate-duration) <= 1/sr and abs(claimed-duration) <= 1/sr:
                    audit.update(status='consistent', reason='Activity and observed event reach the actual media boundary; no trailing silence required')
                    event.setdefault('boundary_activity_audit', []).append(deepcopy(audit))
                else:
                    uncertain('No sustained quiet within bounded context; retain an uncertain perceptual estimate, not a verified end')
                continue
            if candidate <= a:
                uncertain('Activity proxy is outside the reference envelope')
                continue


            contradiction=(candidate-claimed > .02+1/sr or claimed-candidate > .08+1/sr)
            audit.update(status='conflict' if contradiction else 'consistent',
                activity_end=candidate, energy_threshold=threshold, peak_frame_rms=float(energy.max()),
                quiet_context_seconds=len(quiet)*hop/sr, difference_seconds=candidate-claimed,
                reason='Isolated activity followed by sustained quiet contradicts the claimed boundary' if contradiction
                    else 'Claim agrees with the isolated activity/quiet transition within check tolerance',
                method='10 ms RMS; sound_end uses near-digital silence, utterance_end uses 1% candidate peak; not source separation or forced alignment')
            audit['search_context']['end_seconds'] = min(hi, max(b, candidate+.12))
            if repair and (contradiction or measurement is None):
                event.setdefault('boundaries', {})[role]=dict(time=candidate,status='estimated',approximate=True,
                    evidence='After one bounded perceptual review: isolated waveform activity followed by quiet; phase proxy, not exact semantic alignment',
                    activity_verification=deepcopy(audit))
                if candidate>b:
                    event['activity_original_envelope']=[a,b]
                    event['end_time']=candidate
                audit['status']='estimated_correction'
            elif contradiction and measurement is not None:


                phases=event.setdefault('boundaries',{})
                phases[role]=dict(phases[role],status='unresolved',activity_conflict=deepcopy(audit))
            event.setdefault('boundary_activity_audit',[]).append(deepcopy(audit))
    return result,audits
