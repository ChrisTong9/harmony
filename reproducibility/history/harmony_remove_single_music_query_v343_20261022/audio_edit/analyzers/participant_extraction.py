"""Explicit selected-person extraction from already compiled speech identity."""
from copy import deepcopy
import re
from pathlib import Path

from audio_edit.analyzers.participant_selection import participant_placements
from audio_edit.analyzers.speech_references import speech_reference_gaps
from audio_edit.analyzers.speech_segments import uses_segments
from audio_edit.plans.remove_sound_plan import RemoveSoundPlan, TimeSpan
from audio_edit.plans.temporal import TemporalConstraint


def _text(value):
    return ' '.join(str(value or '').translate(str.maketrans({'’':"'",'‘':"'"})).casefold().split())


def explicit_selected_extraction(instruction, first_brief):
    """A literal first clause supplies output intent, never speaker identity."""
    selection=first_brief.get('selection') or {}
    participant=first_brief.get('participant_selection') or {}
    if selection.get('scope')!='whole_file' or selection.get('occurrence')!='unselected':
        return False
    reference=_text(participant.get('reference_text'))
    if not reference:
        return False
    clause=re.match(r'^\s*(?:please\s+)?(?:extract|isolate)\s+only\s+([^.!?\n]+)',_text(instruction))
    if not clause:
        return False
    phrase=clause.group(1).strip()
    reference=re.sub(r'^the\s+','',reference)
    phrase=re.sub(r'^the\s+','',phrase)
    if not phrase.startswith(reference):
        return False
    suffix=phrase[len(reference):]
    if suffix and not suffix.startswith(' '):
        return False
    suffix=suffix.strip()
    return (not suffix or bool(re.fullmatch(
        r'(?:from|in|across|throughout) (?:the )?(?:entire|whole|full) (?:audio|clip|recording|file)',suffix)))


def compile_participant_extraction(ctx, analysis, duration):
    """Local plan compilation requires the existing complete identity links."""
    ctx.params.pop('participant_extraction',None)
    dialogue=ctx.params.get('edit_listening_dialogue') or {}
    calls=dialogue.get('planner_calls') or []
    first=(calls[0].get('response') or {}) if calls else {}
    if not explicit_selected_extraction(ctx.user_prompt,first):
        return None
    contract=(analysis or {}).get('edit_listening') or {}
    participant=contract.get('participant_selection')
    if (contract.get('status')!='ready' or not participant
            or participant!=first.get('participant_selection')):
        return None
    questions=contract.get('requirements') or []
    timing=next((q for q in questions if q.get('id')==participant['timing_question_id']),None)
    if (timing is None or not uses_segments(timing)
            or timing.get('participant_identity_question_id')!=participant['identity_question_id']
            or speech_reference_gaps(questions,analysis)):
        return None
    from audio_edit.analyzers.removal_focus import file_sha256
    if timing['speech_references']['original_audio_sha256']!=file_sha256(ctx.input_audio):
        return None
    from audio_edit.plans.errors import UnresolvedEditError
    try:
        placements=participant_placements(analysis,duration)
    except UnresolvedEditError:
        return None
    if not placements:
        return None
    reference=participant['reference_text']
    ctx.params['participant_extraction']=dict(eligible=True,
        intent_source='Literal Extract/Isolate only first clause and unchanged whole-file first brief',
        participant_selection=deepcopy(participant),source_description=reference,
        input_sha256=timing['speech_references']['original_audio_sha256'],
        windows=[dict(start_time=p.start_time,end_time=p.end_time,event_id=p.event_id) for p in placements],
        outside_policy='silence',fade_seconds=.005,new_planning=0,
        limitation='Uses first approximate lexical intervals and identity observations; no new diarization or boundary inference.')
    return RemoveSoundPlan(backend='samaudio',mode='span',result_mode='result',text_prompt=reference,
        time_span=[TimeSpan(p.start_time,p.end_time) for p in placements],
        use_video_prompt=False,visual_prompt=None,
        notes='Extract the explicitly selected participant within the existing identity-linked speech intervals.',
        temporal=TemporalConstraint(kind='unspecified'),temporal_status='approximate',
        temporal_reason='Existing selected speech intervals; output intent explicitly requests extraction',placements=placements)


def retain_selected_intervals(ctx):
    """Keep the first SAM result inside selected spans, with no widening."""
    import numpy as np
    import soundfile as sf
    from audio_edit.analyzers.removal_focus import file_sha256
    audit=ctx.params.get('participant_extraction') or {}
    if audit.get('eligible') is not True or audit.get('outside_policy')!='silence':
        return ctx
    if audit.get('input_sha256')!=file_sha256(ctx.input_audio):
        raise ValueError('Participant extraction input changed after identity compilation')
    original,rate=sf.read(ctx.input_audio,always_2d=True,dtype='float32')
    selected,selected_rate=sf.read(ctx.output_audio,always_2d=True,dtype='float32')
    if selected_rate!=rate or selected.shape!=original.shape or not np.isfinite(selected).all():
        raise ValueError('Participant extraction must preserve the original finite sample grid')
    mask=np.zeros(len(selected),dtype=np.float32)
    ranges=[]
    for window in audit['windows']:
        start,end=round(window['start_time']*rate),round(window['end_time']*rate)
        if not 0<=start<end<=len(selected):
            raise ValueError('Participant extraction interval is outside the original recording')
        envelope=np.ones(end-start,dtype=np.float32)
        fade=min(round(.005*rate),len(envelope)//2)
        if fade>1:
            ramp=np.sin(np.linspace(0,np.pi/2,fade,dtype=np.float32))**2
            envelope[:fade]=ramp;envelope[-fade:]=ramp[::-1]
        mask[start:end]=np.maximum(mask[start:end],envelope)
        ranges.append(dict(start_sample=start,end_sample=end))
    output=selected*mask[:,None]
    source=Path(ctx.output_audio)
    target=source.with_name(source.stem+'_selected_participant.wav')
    if target.exists():
        raise ValueError('Refusing to overwrite a first participant extraction output')
    sf.write(target,output,rate,subtype='FLOAT')
    ctx.params['participant_extraction_execution']=dict(input_sam_result=str(source),output_audio=str(target),
        sample_rate=rate,channels=selected.shape[1],samples=len(selected),sample_ranges=ranges,
        outside_silent=True,fade_seconds=.005,normalization_applied=False)
    ctx.output_audio=str(target)
    return ctx
