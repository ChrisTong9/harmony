"""Authored synthetic few-shots; no benchmark instructions, audio or answers.

These teach a question/decision relationship, not task-specific execution rules.
The toy observations and times below are invented ONLY inside these examples.
"""

BRIEF_EXAMPLES = [
    {
        'input': dict(original_instruction='Place a soft tick between the first and second bell rings.',
                      operation='add_sound', media_duration=14),
        'output': dict(selection=dict(target='named_source', scope='event_condition', occurrence='selected'),
                       reason='Placement depends on two different audible anchors, not a user preference for where a tick sounds best.', questions=[
            dict(focus='timing', subject='first bell ring', occurrence=1, required_fields=['sound_end'],
                 question='When does the first bell finish sounding, including its decay? Distinguish continued ringing from the start of the next bell.',
                 why='This determines the earlier boundary of the requested between relation.'),
            dict(focus='timing', subject='second bell ring', occurrence=2, required_fields=['onset'],
                 question='When does the second bell begin, and what audible attack distinguishes it from the first bell decay?',
                 why='This determines the later boundary; the planner must check the relation before placing the new tick.')]),
    },
    {
        'input': dict(original_instruction='Turn the audio down from 2 to 7 seconds.',
                      operation='volume_change', media_duration=16),
        'output': dict(selection=dict(target='whole_mix', scope='literal_time', occurrence='unselected'),
                       reason='No source is selected, so the existing default is whole-mix gain over the literal range. There is no authorized exception for voices and no need to discover boundaries or ask how far to move them.',
                       questions=[]),
    },
    {
        'input': dict(original_instruction='Swap the continuous instrumental bed for a gentle synth texture; leave the spoken narration in place.',
                      operation='replace', media_duration=18),
        'output': dict(selection=dict(target='named_source', scope='whole_file', occurrence='unselected'),
                       reason='The source and whole-file scope are explicit. Existing source separation and mixing can preserve narration and its timing; preserving it does not require identifying its individual utterances.',
                       questions=[]),
    },
    {
        'input': dict(original_instruction='Make the rain quieter only while the window is being opened or closed.',
                      operation='volume_change', media_duration=19),
        'output': dict(selection=dict(target='named_source', scope='event_condition', occurrence='selected'),
                       reason='The source is named, but the action-conditioned scope must be observed; separation already preserves other sounds.', questions=[
            dict(focus='timing', subject='window opening or closing', occurrence='each', required_fields=['interval'],
                 question='Locate every complete window movement, distinguishing movement sounds from rain and stationary pauses. For each movement give its start/end, identifying cue, and the gap before the next movement.',
                 why='The rain gain applies only during those actions; treating the whole file as one action would also change the intervening pauses.')]),
    },
    {
        'input': dict(original_instruction="Remove the applause throughout the recording, keeping the presenter's words and breaths unchanged.",
                      operation='remove_sound', media_duration=12),
        'output': dict(selection=dict(target='named_source', scope='whole_file', occurrence='unselected'),
                       reason="Applause is the named sound-class target. The existing separator can extract it from the mix while retaining other sources. The presenter is protected, not selected for editing; no person identity or speech-window question is needed.",
                       questions=[]),
    },
    {
        'input': dict(original_instruction='Add a wooden click after the third ceramic tap; preserve the footsteps.',
                      operation='add_sound', media_duration=11),
        'output': dict(selection=dict(target='named_source', scope='event_condition', occurrence='selected'),
                       reason='The ordinal reference and audible completion determine placement.', questions=[
            dict(focus='target', subject='ceramic taps', occurrence='each', required_fields=['identity'],
                 question='From the beginning of the recording, identify the ceramic taps in order, distinguishing their attacks from footsteps and ringing repeats. Describe what surrounds the third tap and whether more taps follow; third does not mean three taps in total.',
                 why='The count and identity determine which existing event is the third tap.'),
            dict(focus='timing', subject='third ceramic tap', occurrence=3, required_fields=['sound_end'],
                 question='Give the third tap\'s full audible span and where it finishes sounding. What cue separates its decay from the following sound?',
                 why='The new click must follow the correct tap rather than cut into its decay or follow another event.')]),
    },
]

for _example in BRIEF_EXAMPLES:
    for _question in _example['output']['questions']:
        _question['reference_kind'] = 'sound_event'
    _example['output']['placement_windows'] = []
    _example['output']['timeline_policy'] = (dict(mode='preserve', user_quote=None)
        if _example['input']['operation'] == 'add_sound' else None)
BRIEF_EXAMPLES[0]['output']['placement_windows'] = [
    dict(start_question_id='q1', end_question_id='q2')]

BRIEF_EXAMPLES.extend([
    dict(input=dict(original_instruction='Lower only the spoken words "please close the gate".',
            operation='volume_change', media_duration=15),
        output=dict(selection=dict(target='ambiguous_source',scope='event_condition',occurrence='selected'),
            participant_selection=None, placement_windows=[], timeline_policy=None,
            reason='Quoted lexical content selects the existing phrase; locate its spoken interval without identifying a person by ordinal.',
            questions=[dict(focus='target',reference_kind='speech',subject='spoken phrase please close the gate',occurrence='one',required_fields=['identity'],
                question='Which spoken phrase contains these requested words, and what neighboring words distinguish it? Say uncertain if you cannot recognize the words.',
                why='The lexical content selects the speech to adjust.'),
                dict(focus='timing',reference_kind='speech',subject='spoken phrase please close the gate',occurrence='one',required_fields=['interval'],
                question='Locate the audible span of that same spoken phrase, separating neighboring words and any background source.',
                why='Only the requested words receive the gain change.')])) ,
    dict(input=dict(original_instruction='Soften the third caller\'s coughs while preserving the other callers.',
            operation='volume_change', media_duration=18,
            routed_selection=dict(target_speaker='third caller',target_sound='coughs')),
        output=dict(selection=dict(target='ambiguous_source',scope='whole_file',occurrence='unselected'),
            participant_selection=dict(reference_text='third caller',instruction_quote='third caller',
                identity_question_id='q1',timing_question_id='q2'),
            reason='The third distinct caller must be identified; all coughs by that person are selected, not the third cough overall. Text separation cannot determine who coughed.',
            questions=[
                dict(focus='target',reference_kind='participant',subject='coughs by the third caller',occurrence='each',required_fields=['identity'],
                    question='Distinguish the callers by their voices and first appearances. Which coughs can be attributed to the third distinct caller, and what audible cues support that attribution? Say uncertain if the cough cannot be linked to that voice.',
                    why='Only this person\'s coughs may change; a count of speech turns or coughs cannot identify the person.'),
                dict(focus='timing',reference_kind='participant',subject='coughs by the third caller',occurrence='each',required_fields=['interval'],
                    question='Locate each complete cough identified for that caller, retaining the same event identities and intervening gaps; do not include other callers\' coughs.',
                    why='These attributed event windows bound the source gain while preserving the rest.')],
            placement_windows=[],timeline_policy=None)),
    dict(input=dict(original_instruction='Insert a short swish at 2 seconds.', operation='add_sound', media_duration=9),
        output=dict(selection=dict(target='named_source', scope='literal_time', occurrence='unselected'),
            reason='Insert describes adding content here; no new timeline space is requested. Use ordinary mixing at the literal time.',
            questions=[], placement_windows=[], timeline_policy=dict(mode='preserve', user_quote=None))),
    dict(input=dict(original_instruction='Insert a short swish at 2 seconds by adding half a second of new time and shifting later audio.', operation='add_sound', media_duration=9),
        output=dict(selection=dict(target='named_source', scope='literal_time', occurrence='unselected'),
            reason='The instruction explicitly requests new time and shifting the later recording. No acoustic localization is needed.',
            questions=[], placement_windows=[], timeline_policy=dict(mode='retime',
                user_quote='adding half a second of new time and shifting later audio'))),
])

BRIEF_EXAMPLES.append(dict(
    input=dict(original_instruction='Play a soft notification after the spoken instruction and before the reply.',
               operation='add_sound', media_duration=17),
    output=dict(selection=dict(target='ambiguous_source', scope='event_condition', occurrence='selected'),
        participant_selection=None, timeline_policy=dict(mode='preserve', user_quote=None),
        placement_windows=[dict(start_question_id='q2', end_question_id='q3')],
        reason='The spoken instruction and reply are lexical references with two independently supported boundaries.',
        questions=[
            dict(focus='target', reference_kind='speech', subject='spoken instruction and reply', occurrence='one',
                required_fields=['identity'], question='What words and audible voices distinguish the instruction from its reply? Keep any uncertainty about their relation explicit.',
                why='These audible phrases identify the two requested reference events.'),
            dict(focus='timing', reference_kind='speech', subject='spoken instruction', occurrence='one',
                required_fields=['utterance_end'], question='Where does the complete spoken instruction finish? Include its span as context and distinguish an internal pause from completion.',
                why='The added notification must begin after this instruction has finished.'),
            dict(focus='timing', reference_kind='speech', subject='spoken reply', occurrence='one',
                required_fields=['onset'], question='Where does the reply begin, and what first word or attack distinguishes it?',
                why='The notification must finish before that independent onset without shifting the recording.')])) )


BRIEF_EXAMPLES.append(dict(
    input=dict(original_instruction="Duck the ventilation noise during the speaker's spoken phrases, preserving the gaps between phrases.",
               operation='volume_change', media_duration=21,
               routed_selection=dict(target_sound='ventilation noise')),
    output=dict(selection=dict(target='named_source', scope='event_condition', occurrence='selected'),
        participant_selection=None, timeline_policy=None, placement_windows=[],
        reason='Spoken activity controls when the named noise is ducked. No distinguishing person is selected; locating all speech phrases is required, not choosing a particular voice.',
        questions=[dict(focus='timing', reference_kind='speech', speech_scope='all_spoken_activity', subject='spoken phrases', occurrence='each',
            required_fields=['interval'], question='Locate each spoken phrase with its recognized words and audible start/end. Keep distinct phrases and their intervening gaps separate; say uncertain when words or boundaries cannot be identified.',
            why='The noise gain changes only over those speech windows while pauses and the rest remain unchanged.')])) )

BRIEF_EXAMPLES.extend([
    dict(input=dict(original_instruction="Remove the camera shutter beside the actor's sigh, leaving the sigh and dialogue unchanged.",
            operation='remove_sound', media_duration=13),
        output=dict(selection=dict(target='named_source',scope='event_condition',occurrence='selected'),
            participant_selection=None, timeline_policy=None, placement_windows=[],
            reason='The shutter is edited; the sigh and dialogue are protected residual sources. The relation selects one shutter, not a person to edit. Ask its span and audible overlap rather than the actor identity or exact sigh span.',
            questions=[
                dict(focus='timing',reference_kind='sound_event',subject='camera shutter beside the sigh',
                    occurrence='one',required_fields=['interval'],
                    question='Locate the complete shutter click beside the sigh. Describe its mechanical attack and decay and distinguish it from the adjacent vocalization.',
                    why='This window selects the shutter for source-specific removal.'),
                dict(focus='preservation',reference_kind='sound_event',subject='camera shutter and overlapping voices',
                    occurrence='one',required_fields=['overlap'],
                    question='Does the shutter overlap a sigh or spoken voice? Describe any co-occurring sounds that must remain separate from the shutter target; no speaker identity is required.',
                    why='Overlapping voice rules out whole-mix muting; the existing source separator targets the shutter.') ])),
    dict(input=dict(original_instruction='Replace the rattling passage between two door creaks with flowing water.',
            operation='replace', media_duration=16),
        output=dict(selection=dict(target='ambiguous_source',scope='event_condition',occurrence='selected'),
            participant_selection=None, timeline_policy=None, placement_windows=[],
            reason='The neighboring creaks select one acoustic passage. Its internal attacks may be parts of one sequence rather than separate alternative target events.',
            questions=[
                dict(focus='target',reference_kind='sound_event',subject='rattling passage between the door creaks',
                    occurrence='one',required_fields=['identity'],
                    question='Identify the selected passage by its sound and the creaks on either side. Are its rattles one coherent sequence or distinct alternative events? Explain the audible relation.',
                    why='The relational description selects which original source passage to replace.'),
                dict(focus='timing',reference_kind='sound_event',subject='rattling passage between the door creaks',
                    occurrence='one',required_fields=['interval'],
                    question='Locate that complete passage, including internal short pauses and its audible ending. Keep unrelated surrounding creaks outside it; do not pick one attack as the whole passage.',
                    why='The replacement must cover the selected passage, while source separation preserves unrelated sounds.') ])),
])

from copy import deepcopy


def _speech_review_examples(payload):
    """Teach independent lexical/physical cues before their final audio interpretation."""
    names = {tool['name'] for tool in payload.get('available_tools', [])}
    can_listen = bool(payload.get('observations_available'))
    has_listen_budget = can_listen or bool(payload.get('observation_prerequisites')
        and payload.get('remaining_budget', {}).get('qwen_listens', 0))
    questions = [dict(id='q1', focus='timing', reference_kind='speech', subject='spoken instruction',
                     occurrence='one', required_fields=['utterance_end']),
                 dict(id='q2', focus='timing', reference_kind='speech', subject='spoken reply',
                     occurrence='one', required_fields=['onset'])]
    cue = 'I hear an instruction followed by a reply, but their words and the dividing pause remain unclear.'
    analysis = dict(events=[], answers=[dict(question_id=q['id'],status='uncertain',answer=cue) for q in questions])
    base = dict(requirements=questions, observations_available=can_listen,
        available_audio_sources=['original'], remaining_budget=dict(qwen_listens=int(has_listen_budget),tool_requests=2),
        missing_evidence={'q1':['Instruction completion remains unclear'], 'q2':['Reply onset remains unclear']},
        turns=[dict(action='qwen', analysis=deepcopy(analysis))])
    output = dict(evidence_assessment={q['id']:dict(cue_quote=cue,problem='The speech boundary is not grounded.') for q in questions},
        status='unresolved', reason='No available evidence can yet establish the requested speech relation.',
        observation_request=None, tool_request=None, measurement_bindings=[])
    lexical = dict(tool='speech_transcription', kind='tool_evidence',
        duration=10., audio_sha256='illustrative_original', original_audio_sha256='illustrative_original',
        supports_diarization=False,
        time_axis=dict(origin='original_audio_start',offset_seconds=0,cropped=False,units='seconds',original_duration=10.),
        words=[dict(word_id='w1',text='corridor',start_time=7.10,end_time=7.42,timestamp_status='estimated'),
               dict(word_id='w2',text='understood',start_time=8.06,end_time=8.51,timestamp_status='estimated')],
        groups=[dict(text='Take the left corridor',word_ids=['w1']),
                dict(text='Understood',word_ids=['w2'])],
        limitation='Illustrative lexical anchors only; groups are not people and word ends are not utterance endings.')
    activity = dict(tool='activity_regions',
        duration=10., audio_sha256='illustrative_original',
        regions=[dict(region_id='demo_r1',start_time=5.6,end_time=7.59,complete=True),
                 dict(region_id='demo_r2',start_time=8.02,end_time=9.4,complete=False)],
        quiet_gaps=[dict(start_time=7.59,end_time=8.02)],
        limitation='Anonymous candidates; no phrase identity or semantic completion.')
    examples = []
    if 'speech_transcription' in names and has_listen_budget:
        item = dict(input=deepcopy(base),output=deepcopy(output))
        item['input']['available_tools']=[dict(name='speech_transcription',question_ids=['q1','q2'],max_questions=2),
                                        dict(name='activity_regions',question_ids=['q1','q2'],max_questions=2)]
        item['output'].update(status='measure', tool_request=dict(name='speech_transcription',question_ids=['q1','q2']),
            reason='Independent words and approximate word times can locate the instruction and reply when Qwen is uncertain. Save the remaining audio observation to interpret these cues; a separator cannot identify either speech turn.')
        examples.append(item)
    if 'activity_regions' in names and has_listen_budget:
        item = dict(input=deepcopy(base),output=deepcopy(output))
        item['input']['turns'].append(dict(action='speech_transcription',tool_evidence=lexical,analysis=deepcopy(analysis)))
        item['input'].update(available_tools=[dict(name='activity_regions',question_ids=['q1','q2'],max_questions=2)],
                             remaining_budget=dict(qwen_listens=1,tool_requests=1))
        item['output'].update(status='measure', tool_request=dict(name='activity_regions',question_ids=['q1','q2']),
            reason='ASR supplies phrase anchors, but neither the end of a word nor a lexical group proves a between-window. Obtain original-waveform gap candidates before the final listening call, without assigning them to a person.')
        examples.append(item)
    if can_listen:
        item = dict(input=deepcopy(base),output=deepcopy(output))

        tools_present = payload.get('perception_mode') != 'dialogue_only' and any(
            t.get('action') in ('speech_transcription','activity_regions') for t in payload.get('turns',[]))
        if tools_present:
            item['input']['turns'].extend([
                dict(action='speech_transcription',tool_evidence=lexical,analysis=deepcopy(analysis)),
                dict(action='activity_regions',tool_evidence=activity,analysis=deepcopy(analysis))])
            from audio_edit.analyzers.speech_references import with_speech_references
            item['input']['requirements'] = with_speech_references(questions, [lexical, activity],
                duration=10., original_hash='illustrative_original')
        question = ('Using the recognized phrase anchors and candidate gap, which audible words flank the pause? '
                    'Does it separate the complete instruction from its reply or interrupt a phrase? '
                    'Select the word range IDs and endpoint words that the original audio supports; code joins adjacent gaps. '
                    'Report unknown/no match if none fits; do not supply new seconds.' if tools_present else
                    'What audible words end the instruction and begin its reply? Is the pause between those phrases or inside one phrase?')
        item['input'].update(available_tools=[],remaining_budget=dict(qwen_listens=1,tool_requests=0))
        item['output'].update(status='listen',
            reason='Only an audio observation can associate the phrase cues with the requested semantic boundary; neither words nor anonymous gaps certify it.',
            observation_request=dict(question_id='q1',question=question,
                why='Resolve the instruction completion and its relation to the already requested reply onset.',audio_source='original'))
        examples.append(item)
    if not examples:
        examples.append(dict(input=dict(base,available_tools=[]),output=output))
    end_cue = 'The instruction finishes with corridor and its trailing voice ends before a quiet pause.'
    start_cue = 'The reply starts with understood after that pause; it is not another word of the instruction.'
    supported = dict(events=[
        dict(event_id='heard_instruction',event='spoken instruction',question_ids=['q1'],
             start_time=5.6,end_time=7.6,occurrence_index=1,
             boundaries=dict(utterance_end=dict(time=7.6,evidence=end_cue))),
        dict(event_id='heard_reply',event='spoken reply',question_ids=['q2'],
             start_time=8.0,end_time=9.4,occurrence_index=1,
             boundaries=dict(onset=dict(time=8.0,evidence=start_cue)))],
        answers=[dict(question_id='q1',status='answered',answer=end_cue),
                 dict(question_id='q2',status='answered',answer=start_cue)])
    examples.append(dict(input=dict(requirements=questions,
        turns=[dict(action='qwen',analysis=supported)],missing_evidence={},
        observations_available=can_listen,available_tools=[]),
        output=dict(evidence_assessment=dict(q1=dict(cue_quote=end_cue,problem=None),
            q2=dict(cue_quote=start_cue,problem=None)),status='ready',
            reason='Current audible cues ground both requested boundaries and positive placement space. These remain estimates; additional transcription or separation is unnecessary.',
            observation_request=None,tool_request=None,measurement_bindings=[])))
    if any(q.get('speech_references') for q in payload.get('requirements', [])):


        from audio_edit.analyzers.speech_references import with_speech_references, resolve_speech_references
        from audio_edit.analyzers.listening_protocol import unpack_facts
        bound = with_speech_references(questions, [lexical, activity], duration=10.,
                                      original_hash='illustrative_original')
        gap_id = bound[0]['speech_references']['gaps'][0]['gap_id']
        def choice(word, label, cue, *, start_gap=None, end_gap=None, complete=False):
            return dict(timing_kind='speech_reference',event=label,occurrence_index=1,
                first_word_id=word,last_word_id=word,start_gap_id=start_gap,end_gap_id=end_gap,
                utterance_complete=complete,association_status='match',association_evidence=cue)
        compiled = dict(facts=dict(
            q1=dict(status='answered',answer=end_cue,events=[choice('w1','instruction ending',end_cue,end_gap=gap_id,complete=True)]),
            q2=dict(status='answered',answer=start_cue,events=[choice('w2','reply',start_cue,start_gap=gap_id)])))
        resolve_speech_references(compiled, bound)
        unpack_facts(compiled)
        examples.append(dict(input=dict(requirements=bound,turns=[dict(action='qwen',analysis=compiled)],
            missing_evidence={},observations_available=False,available_tools=[]),output=dict(
                evidence_assessment=dict(q1=dict(cue_quote=end_cue,problem=None),q2=dict(cue_quote=start_cue,problem=None)),
                status='ready',reason='The original-audio interpretation selected a completed instruction and its reply; '
                    'the selected word and gap references compile to one coherent positive window. '
                    'These remain approximate evidence, not speaker identification.',
                observation_request=None,tool_request=None,measurement_bindings=[])))
    return examples


def _speech_interval_examples(payload):
    """Interval coverage does not inherit a turn-ending/acoustic-silence task."""
    from audio_edit.analyzers.speech_references import with_speech_references,resolve_speech_references
    from audio_edit.analyzers.listening_protocol import unpack_facts
    q=dict(id='q1',focus='timing',reference_kind='speech',subject='spoken phrases',
           occurrence='each',required_fields=['interval'],speech_scope='all_spoken_activity')
    speech=dict(tool='speech_transcription',duration=4.,audio_sha256='interval_demo',
        original_audio_sha256='interval_demo',supports_diarization=False,
        time_axis=dict(origin='original_audio_start',offset_seconds=0,cropped=False,units='seconds',original_duration=4.),
        words=[dict(word_id='demo_w1',text='Welcome.',start_time=.3,end_time=1.,timestamp_status='estimated'),
               dict(word_id='demo_w2',text='Please wait.',start_time=2.,end_time=3.,timestamp_status='estimated')])
    qs=with_speech_references([q],[speech],duration=4.,original_hash='interval_demo')
    cue='Both lexical candidates match audible spoken phrases; music continues through the pause.'
    compiled=dict(facts=dict(q1=dict(status='answered',answer=cue,segment_assessments={
        r['segment_id']:dict(status='match',evidence='These words are audible in the original recording.')
        for r in qs[0]['speech_references']['lexical_regions']})))
    resolve_speech_references(compiled,qs);unpack_facts(compiled)
    base=dict(requirements=qs,available_tools=[],observations_available=False,
              turns=[dict(action='qwen',analysis=compiled)],missing_evidence={})
    ready=dict(evidence_assessment=dict(q1=dict(cue_quote=cue,problem=None)),status='ready',
        reason='The matched lexical regions supply approximate speech windows while preserving pauses. '
               'This interval edit does not require a completed turn or silence in the mixed music.',
        observation_request=None,tool_request=None,measurement_bindings=[])
    examples=[dict(input=base,output=ready)]
    unknown='The apparent continuous voice has not been checked against lexical coverage.'
    pending=dict(requirements=[q],turns=[dict(action='qwen',analysis=dict(events=[],
        answers=[dict(question_id='q1',status='uncertain',answer=unknown)]))],
        missing_evidence=dict(q1=['Speech coverage remains unclear.']))
    output=dict(evidence_assessment=dict(q1=dict(cue_quote=unknown,problem='Need speech interval coverage.')),
        status='unresolved',reason='No remaining evidence can resolve coverage.',
        observation_request=None,tool_request=None,measurement_bindings=[])
    names={t['name'] for t in payload.get('available_tools',[])}
    if 'speech_transcription' in names:
        pending.update(observations_available=False,available_tools=[dict(name='speech_transcription',question_ids=['q1'],max_questions=1)])
        output.update(status='measure',tool_request=dict(name='speech_transcription',question_ids=['q1']),
            reason='Obtain approximate lexical coverage before checking the regions with the remaining listen.')
    elif payload.get('observations_available'):
        pending.update(requirements=qs,observations_available=True,available_tools=[],
            turns=pending['turns']+[dict(action='speech_transcription',tool_evidence=speech)])
        output.update(status='listen',reason='Confirm the candidate phrases, without treating music energy as speech.',
            observation_request=dict(question_id='q1',audio_source='original',
                question='Classify every supplied lexical region as matching speech, nonmatching or unknown; identify any omitted words.',
                why='Apply the gain only to confirmed speech coverage and preserve intervening pauses.'))
    examples.insert(0,dict(input=pending,output=output))
    return examples


def review_examples(payload):
    """Select short synthetic demonstrations by the current available capabilities."""
    speech=[q for q in payload.get('requirements',[]) if q.get('reference_kind')=='speech']
    if speech and all(q['focus']=='timing' and q.get('speech_scope')=='all_spoken_activity' and q.get('occurrence')=='each'
                      and q['required_fields']==['interval'] for q in speech):
        return _speech_interval_examples(payload)
    if speech:
        return _speech_review_examples(payload)
    names = {tool['name'] for tool in payload.get('available_tools', [])}
    can_listen = bool(payload.get('observations_available'))
    identity = dict(id='q1',focus='target',subject='the different tap',occurrence='one',required_fields=['identity'])
    interval = dict(id='q1',focus='timing',subject='second chime',occurrence=2,required_fields=['sound_end'])
    cue = 'The second pitched chime stops into silence.'
    base = dict(requirements=[interval], turns=[dict(action='qwen',analysis=dict(
        answers=[dict(question_id='q1',status='answered',answer=cue)],events=[dict(
            event_id='heard_chime',event='pitched chime',question_ids=['q1'],occurrence_index=2,
            start_time=2.1,end_time=2.6,boundaries=dict(sound_end=dict(time=2.6,evidence=cue)))]))])
    output = dict(evidence_assessment=dict(q1=dict(cue_quote=cue,problem=None)),
        reason='The current target and required boundary are supported.',status='ready',
        observation_request=None,tool_request=None,measurement_bindings=[])
    examples = [dict(input=dict(requirements=[identity],missing_evidence={'q1':['Target distinction remains uncertain']},
        observations_available=can_listen,available_tools=[],turns=[dict(action='qwen',analysis=dict(
            answers=[dict(question_id='q1',status='uncertain',answer='Two similar taps cannot yet be distinguished.')],events=[]))]),
        output=dict(evidence_assessment=dict(q1=dict(cue_quote=None,problem='The requested contrasting source is not identified.')),
            reason=('A contrast in attack or resonance would identify the target; anonymous activity cannot.' if can_listen else
                    'The remaining tools cannot identify the contrasting source and no audio observation remains.'),
            status='listen' if can_listen else 'unresolved',
            observation_request=dict(question_id='q1',question='Which tap has a different attack or resonance from its neighbors, and what audible cue distinguishes it?',
                why='The identified source selects the event to edit.',audio_source='original') if can_listen else None,
            tool_request=None,measurement_bindings=[]))]
    if payload.get('measurement'):
        example = dict(input=deepcopy(base),output=deepcopy(output))
        example['input']['measurement_candidates']=[dict(question_id='q1',event_id='heard_chime',region_id='isolated_r2',
            observed_span=[2.1,2.6],measured_span=[2.08,2.84])]
        example['output'].update(reason='The one complete isolated component corresponds to the already heard second chime. Bind its measured end to the event.',
            measurement_bindings=[dict(question_id='q1',event_id='heard_chime',region_id='isolated_r2')])
        examples.append(example)
    elif 'activity_regions' in names:
        example = dict(input=deepcopy(base),output=deepcopy(output))
        example['input']['available_tools']=[dict(name='activity_regions',question_ids=['q1'],max_questions=1)]
        example['output'].update(status='measure',reason='The isolated physical end controls placement; a waveform measurement can check the rough estimate without another audio call.',
            tool_request=dict(name='activity_regions',question_ids=['q1']))
        examples.append(example)
    if 'source_activity' in names:
        example = dict(input=deepcopy(base),output=deepcopy(output))
        example['input'].update(measurement=dict(tool='activity_regions',regions=[dict(region_id='mixed_r1',start_time=0,end_time=7,complete=False)]),
            available_tools=[dict(name='source_activity',question_ids=['q1'],max_questions=1)])
        example['output'].update(status='measure',reason='The mixed component cannot locate the identified chime over steady noise. Extract that one source before measuring its physical completion.',
            tool_request=dict(name='source_activity',question_ids=['q1']))
        examples.append(example)
    if 'speech_transcription' in names:
        lexical = dict(id='q1',focus='target',subject='spoken phrase please close the gate',
            occurrence='one',required_fields=['identity'])
        examples.append(dict(input=dict(requirements=[lexical],
            missing_evidence={'q1':['The quoted words are not identified']},
            available_tools=[dict(name='speech_transcription',question_ids=['q1'],max_questions=1)],
            turns=[dict(action='qwen',analysis=dict(events=[],answers=[dict(question_id='q1',status='uncertain',
                answer='I hear a short request but cannot distinguish the last few words.')],
                evidence_needs=[dict(question_id='q1',capability='lexical_content',
                    reason='Recognized words could distinguish this phrase.')]))]),
            output=dict(evidence_assessment=dict(q1=dict(cue_quote=None,problem='The requested words remain unidentified.')),
                status='measure',reason='ASR can provide the missing lexical content before it is identified; anonymous energy and separation cannot recognize the quoted words.',
                observation_request=None,tool_request=dict(name='speech_transcription',question_ids=['q1']),
                measurement_bindings=[])))
    if payload.get('measurement') and can_listen:
        example = dict(input=deepcopy(base),output=deepcopy(output))
        example['input'].update(missing_evidence={'q1':['Extracted regions remain unassociated with the target event']},
            measurement_candidates=[],observations_available=True)
        source = 'original_and_source' if 'original_and_source' in payload.get('available_audio_sources',[]) else 'original'
        example['output'].update(status='listen',reason='The measured activity has no supported target association; a new audible observation can resolve that specific identity-to-region link.',
            observation_request=dict(question_id='q1',question='Which activity region belongs to the already identified chime, including its quiet tail? Report uncertainty if leakage prevents a reliable association.',
                why='Only a supported source-to-region association can replace the event boundary.',audio_source=source))
        examples.append(example)
    if not examples[1:]:
        examples.append(dict(input=deepcopy(base),output=deepcopy(output)))
    return examples
