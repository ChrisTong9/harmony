"""Edit evidence supervision with bounded dialogue and capability-based tools."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path

from audio_edit.analyzers.qwen_audio_analyzer import extract_json_object
from audio_edit.plans.errors import UnresolvedEditError
from audio_edit.utils.media_util import get_media_duration
from audio_edit.analyzers.listening_evidence import (FACTS, FOCUS_FIELDS, SELECTION_FIELDS, REFERENCE_KINDS, questions_from_brief,
    observation_view, check_evidence, check_boundary_activity,
    boundary_decision_risks, placement_windows_from_brief, check_placement_windows,
    timeline_policy_from_brief)
from audio_edit.analyzers.listening_capabilities import (CAPABILITY_RULES,
    available_tools, tool_request_schema, tool_request_error)
from audio_edit.analyzers.measurement_tools import (run_measurement, measurement_candidates,
    apply_measurement_bindings, unresolved_measurement_conflicts, source_handoff_gaps, sha256)
from audio_edit.analyzers.measured_references import with_measured_regions
from audio_edit.analyzers.speech_references import with_speech_references, speech_reference_gaps
from audio_edit.analyzers.speech_dependencies import (update_speech_dependencies,
    speech_dependency_gaps, speech_prerequisites)
from audio_edit.analyzers.listening_tool_evidence import merge_tool_evidence
from audio_edit.analyzers.participant_selection import (routed_selection, routed_participant,
    participant_selection_schema, participant_from_brief, bind_participant_questions)


QUESTION_PROMPT = """You are the existing edit planner deciding what must be heard
before the requested edit can be executed. You have NOT heard the recording.
Use the original instruction and executor_defaults. Identify WHAT is edited and
WHERE: whole recording, explicit file windows, or event-conditioned windows. For
replace, identify old and new content; remove and add share the same scope. Whole
file is a time scope, not permission to affect every source. A particular speaker
or ordinal sound remains a selection even if its event type is named. First distinguish decisions
already settled by the instruction/defaults from facts still needed from audio.
Return JSON {"selection":{"target":"named_source","scope":"whole_file","occurrence":"unselected"},
"reason":"why listening is or is not needed","questions":[{"focus":"target","reference_kind":"unknown","subject":"existing reference",
"occurrence":"one","required_fields":["identity"],"question":"audible inquiry",
"why":"the specific editing decision that depends on this answer"}],"placement_windows":[],
"timeline_policy":{"mode":"preserve","user_quote":null},"participant_selection":null}.

The routed_selection fields are earlier request-extraction hints, NOT observations
or new instructions. Preserve the ORIGINAL request when interpreting them. If an
existing participant is selected, return participant_selection={"reference_text":
"the exact participant reference", "instruction_quote":"exact words from the user",
"identity_question_id":"q1", "timing_question_id":"q2"}. Link the questions by their
positions in your question list (the program numbers them q1, q2, ...).
Select target=ambiguous_source. The identity question must identify the selected
person's REQUESTED EVENTS with distinguishing audible cues; the timing question
must locate those SAME events, including attribution when they are coughs/laughs
or other non-speech sounds. Removal/replacement/volume require complete intervals;
add may need an existing speaker's boundary. SAM cannot recognize a participant
from an ordinal label. Do not skip this dependency because a sound class is named,
or because the temporal range is whole-file/literal. A second PERSON is a distinct
voice ordered by first appearance, not the second speech segment; q.occurrence
selects events OF that person, never the person's ordinal. If attribution cannot
be heard, keep it uncertain. Merely preserving a narrator while editing BGM does
not select that person, so participant_selection=null and the simple path remains.
More generally, a person or vocalization named ONLY to be KEPT is a preservation
constraint, not the source to edit and not a timing reference. Do not identify
that person's voice or locate their protected words/breath/laugh just to protect
them while removing another sound. Set participant_selection=null in that case.
Ask about the actual removal/replacement/gain target and, only if needed, its
audible overlap with protected sounds. Existing source separation preserves the
residual. It does not require a person label or the protected sound's exact span.
Participant selection remains required when WHOSE events should change selects
the target, or when a particular person's events define WHERE another source
changes. Keeping that same person's OTHER sounds still requires this selection.
For add, a voice requested for NEW generated speech is not an existing participant;
only an existing participant used to select a reference needs this dependency.

participant_selection is for choosing WHOSE events, such as a name, ordinal,
distinguishing voice description or a role that excludes other participants.
Generic references to speech/the speaker in a request to duck another source
during speech do not by themselves select a distinct person. When the requested
condition is simply spoken activity, use participant_selection=null and ask for
the speech intervals with reference_kind=speech and occurrence=each. Keep pauses
outside those intervals; do not simplify them to the whole file. If the original
instruction actually selects one person, retain that identity dependency even
when another source is what gets edited. Do not invent a person selector merely
because an event-conditioned window refers to speech.

For ADD, decide timeline_policy from the ORIGINAL instruction before listening.
Default mode=preserve, user_quote=null: mix onto the original timeline without
moving any original samples or extending file duration. Ordinary verbs such as
"add" or "insert a sound after an event" describe adding content, not new time.
Use mode=retime ONLY for an explicit request to shift later audio, create/insert
new timeline time, append/prepend beyond the file, or extend duration; user_quote
must be the exact phrase requesting that change. Unspecified placement or lack
of space never grants this permission. Non-add operations use timeline_policy=null.

For ADD with an explicit file range, FIRST distinguish the layer's placement
from its musical content. A continuous accompaniment starts and stops at the
stated times. Following existing harmony, root notes, chord changes or rhythm is
content conditioning through the generator's audio reference; it is NOT a missing
placement boundary. Set selection={"target":"named_source","scope":"literal_time",
"occurrence":"unselected"}, participant_selection=null and questions=[] for that
case. Do not turn its musical alignment into chord transcription or per-attack
measurement. A stated start through the file end is also an explicit range.
Keep listening when the USER additionally selects an existing participant/event,
limits activity to existing event spans, or asks for separate sounds at measured
attacks or a counted sequence before/after an event. These are actual placement
dependencies, unlike synchronizing the content of one fixed-range musical layer.

Classify selection from the USER WORDS first, without inferring acoustic events:
- target: whole_mix when no source is selected; named_source for an ordinary
  described source (including music/accompaniment/noise); ambiguous_source only
  for a referential identity requiring observation, such as an unnamed odd sound
  relative to other events or a particular participant. The theoretical presence
  of other instruments does not make a named source ambiguous.
- scope: whole_file by default when no region is selected; literal_time for a
  stated file time/range/fraction; event_condition only when the user conditions
  scope/placement on an existing event or relation. Keeping another source's
  timing unchanged does NOT make scope event_condition. For ADD, musical content
  following a reference does not override an already explicit file range.
- occurrence: unselected unless the user chooses an ordinal, each occurrence,
  a subset or count tied to existing events. A whole named source is unselected;
  never turn it into "each passage" just to align a replacement.
When selection is named_source/whole_mix + whole_file/literal_time + unselected,
the program uses existing defaults and SKIPS all listening questions. For those
cases return questions=[]. For other selections ask the minimal missing facts.

Use questions=[] when existing execution is sufficient. A named source does not
need a new identity confirmation just because it overlaps other sources: the
separator already accepts its description. Whole-file background source edits use
separation to preserve other sources and their timing. Preserving narration's
EXACT timing is an invariant, not a request to locate its individual utterances.
Ordinary volume changes with no selected source affect the whole mix. A literal
range already fixes where to act. An independent addition at a literal time also
needs no listening just to describe its surroundings. Do not ask how to implement
fades, measure noise fingerprints, or select an unavailable editing backend.

Ask only if an audio answer changes a necessary decision: which ambiguous source,
which/how many occurrences, an event-conditioned edit interval, or an event-relative
anchor. In particular, "during a source's events" requires intervals, whereas
"keep that source and its timing unchanged" does not. Identify these dependencies
in why using the original instruction, not an extra goal invented by the question.
There are 0-6 questions; target/timing/preservation are not a mandatory checklist.
Unknown overlap can use separation. Skipping listening does not assert isolation.

Write the actual question yourself, tailored to the instruction's unresolved
references. Request distinguishing audible cues and neighboring sounds as well as
the required identity/time/count; a bare timestamp or confident label is weak
support. For each requested occurrence, include its actual sound span. For event
segments, request intervening gaps as well as activity, not a blanket whole-file
summary. Do not invent observations, expected counts or candidate times.
An ordinal N selects an occurrence; it does not say there are N occurrences in
the file. Ask for the ordered matching attacks from the start and the selected
event's neighboring context, distinguishing other transients and repetitions.
Do not ask Qwen to confirm the user's intended source. For relational descriptions
such as "the different sound in the middle", identify it by contrast with the
surrounding sounds; do not silently replace the relation with the numeric midpoint.
A selected passage may contain multiple attacks and internal pauses. Ask whether
it is one coherent passage or several distinct candidate events. Describe the
flanking sounds and complete passage extent. Do not force one activity component
to stand for a whole passage, and do not treat every component as an alternative
occurrence when they belong to the same heard sequence.
Preserve the requested event class: speech includes short words and interjections
unless the user excludes them. Do not invent minimum lengths or omit brief events.
Ask for approximate positions and audible cues, never exact/sample-accurate timing.
For a requested end, also ask for that same event's full audible span when both
edges can be heard. This lets an available waveform check inspect the estimate;
an end-only observation must not discard an onset already heard for that event.

Qwen hears facts; it is NOT the user or the editing planner. Never ask what the user
means/wants, which sources SHOULD stay at original level, whether to copy/generate
new material, or whether explicit times may move. Ask which audible interpretation
is supported instead. A replacement sound need not exist in the original. Keep
source qualifiers and counts. Include faint sources when relevant; prominence
does not authorize removing them. Do not request opinions about separation quality.
Do not ask for an appropriate insertion position or an early/centered/late choice.
For placement BETWEEN two events, ask separately for the first event's required
end and the second event's onset, with explicit reference/occurrence for each.
Two flanking events are not one detected interval. The planner computes their
relation and chooses placement after hearing those facts; Qwen does not choose it.
For an ADD constrained between two references on the original timeline, record
placement_windows=[{"start_question_id":"q1","end_question_id":"q2"}] using
their actual question positions (q1, q2, ...). This preserves BOTH constraints
through generation and mixing. Use [] for other operations, one-sided anchors,
literal scopes, and explicit requests to create new timeline space. Never invent
a paired window merely because two unrelated timing questions exist. For several
separate bounded additions, record their pairs separately. Each pair needs one
selected boundary per endpoint, not occurrence=each or a blanket interval.
Ordinary additions default to mixing on the unchanged original timeline. A short
pause is not permission to shift the following audio. If spoken content must be
preserved, ask for utterance_end and the next utterance's onset, distinguishing a
breath/internal pause from a completed turn. Ask what is heard, never which turn
the user intended. A target identity answer must link actual observed events;
a generic statement such as 'there are two voices' does not ground the selection.

Classify each question's reference_kind from the instruction: sound_event for a
non-speech sound or source, speech for spoken words/phrases and their starts/ends,
participant for identifying a person or attributing their non-speech events, or
unknown when the kind itself is unclear. A speech reference does not require a
literal quotation: a spoken exchange or earlier/later speech is still speech.
For a named person's spoken phrase, the identity question can be participant and
its lexical/timing question speech. Never label an ordinal person's laugh/cough
sound_event to obtain separation. This classification limits eligible evidence
tools; it does not assert the source exists or settle the person's identity.

The following metadata connects evidence to the unchanged executors; it is not a
menu of fixed questions. subject identifies the existing source/reference using
the user's wording. occurrence is one, first, last, each, or a positive integer;
preserve any user-requested occurrence. focus is target, timing or preservation.
The occurrence metadata must match the question's selector: for a fifth event use
5 even when asking only about that one event. "one" means an unnumbered occurrence;
never leave the ordinal only in prose. Each requested occurrence uses "each".
Set speech_scope=all_spoken_activity only for a timing/each/interval question
selecting spoken activity, such as changing BGM during speech. Set selected_content
for a particular quoted phrase or selected spoken content; set null otherwise.
Do not turn editing a quoted sentence into selecting all speech. This describes
the requested scope, not whether the sound is actually present.
Choose only required_fields necessary for this decision:
Each question declares ONE primary execution fact. Use interval when both
endpoints are needed for remove/replace/volume; use only sound_end for placing
after a sound. A full span requested to help a measurement is optional context,
not another required field. Do not make onset a hard dependency just to check
an after anchor. Preserve any genuine identity/ordinal uncertainty separately.
- target: identity.
- timing: onset, action_end, sound_end, utterance_end, interval, rhythm.
  Fixed file scopes need NO timing question. Use interval for event-conditioned
  scopes and complete occurrences being removed/replaced; onset alone or an action
  completion is not a complete removal interval. Use a semantic boundary for an
  event-relative placement, matching the requested relation. Each timing question
  refers to ONE reference; split distinct start/end references. Timing questions
  declare execution dependencies: missing facts need appropriate observation or measurement.
  Use occurrence=each for every requested event/segment. Require rhythm only when
  measured attacks affect placement; do not infer a grid.
- preservation: overlap (including quiet sources and uncertain source isolation).
Unknown source isolation can use existing separation; it does not require consent.
Return concise, concrete questions and dependencies, not an executable plan.
"""

REPLACE_QUESTION_PROMPT = 'For ordinary REPLACE, the old acoustic source description is already a\nvalid query for the existing source separator. A named sound class does not become\nan ambiguous identity simply because its occurrences have not been listened to.\nPlural source nouns or possessives alone do not select an occurrence. If no\nordinal, count, each-occurrence mapping, rhythm or event-conditioned scope is\nrequested, the default replaces that named source throughout the file; no\nidentity or per-attack timing question is needed merely to implement replacement.\nPreserve all actual user selectors. Each occurrence, a required count or rhythm,\na selected participant, a relational passage and a last/first/nth event still\nrequire the corresponding evidence and original occurrence metadata described\nbelow. This clarification does not authorize dropping any such dependency.\n' + QUESTION_PROMPT


REPLACE_QUESTION_PROMPT = 'For REPLACE, distinguish preserving a source\'s rhythmic CONTENT from\nselecting separate edit intervals. A named source transformation with "the same\nrhythm" or "maintaining the rhythm", with no ordinal/count/per-occurrence mapping\nor event-conditioned scope, uses the existing structure-conditioned audio\ngenerator over the normal whole-file (or explicit file) scope. Rhythm preservation\ndoes not alone require measuring every attack or cutting the edit into transient\nwindows. Use named_source/whole_file/unselected and questions=[] when those are\nthe only requirements; do not invent a particular source identity from "this".\nThe original rhythm still must be retained through structure conditioning; this\nis not permission to generate unrelated rhythm. Keep every explicit selection:\n"each bark", one new sound per original event, numeric counts, first/last/nth,\nselected participants, and event-relative scopes still require their own\nidentity/timing evidence. A count/each mapping never becomes a whole-file texture.\nThis clarifies edit scope, not audible source facts or the user\'s requested sound.\n\n' + REPLACE_QUESTION_PROMPT.replace(
    'ordinal, count, each-occurrence mapping, rhythm or event-conditioned scope is',
    'ordinal, count, each-occurrence mapping or event-conditioned scope is').replace(
    'a required count or rhythm,', 'a required count,')

VOLUME_QUESTION_PROMPT = QUESTION_PROMPT.replace(
    'Return JSON {"selection":{"target":"named_source","scope":"whole_file","occurrence":"unselected"},\n"reason":"why listening is or is not needed","questions":[{"focus":"target","reference_kind":"unknown","subject":"existing reference",',
    'Write reason FIRST: identify the source being changed, what defines its scope,\nand whether the instruction distinguishes a particular person from other voices.\nFor speech-conditioned edits, "speaker" is an activity label, not a person\'s\nname or a distinguishing role. The definite article and possessive in "the\nspeaker\'s" or "the speaker’s" do not exclude other voices. Without a qualifying\nselector, ask for spoken intervals with participant_selection=null. A name, ordinal,\ndistinctive voice or contrasting role DOES select a person; retain that identity\ndependency even when another source is edited. Do not invent a person selector\nfrom the possessive form alone. Keeping speech unchanged requires no identity\nquestion unless whose events are edited or used as references is actually selected.\nReturn JSON {"reason":"edit target, scope and any distinguishing person selector",\n"selection":{"target":"named_source","scope":"whole_file","occurrence":"unselected"},\n"questions":[{"focus":"target","reference_kind":"unknown","subject":"existing reference",')


REVIEW_PROMPT = """Supervise the evidence needed to execute the ORIGINAL audio edit.
Qwen observes sound; you choose the next action. Identify the unresolved operation
argument: target/occurrence (what), executable scope/anchor (where), or preservation.
Do not acquire more evidence for gain, new content, or a literal window already
specified by the user. All tools act before editing; there is no output inspection.

Return JSON with evidence_assessment, reason, status, observation_request,
tool_request, measurement_bindings. For each requirement, evidence_assessment
contains cue_quote (an exact current observation or null) and problem (the specific
blocking gap or null). A fluent reason is not an observation. Assess current facts;
historical snapshots are fallible context, not cumulative constraints. Uncertain
source overlap alone may use the existing execution separator without more evidence.

Choose exactly one status:
- ready: all requested selections, event identities and executable windows are
  supported. Both requests are null. Do not ignore code-reported missing evidence.
- listen: a new audible distinction can resolve a missing edit dependency. Set
  observation_request={"question_id":"an existing requirement ID",
  "question":"a targeted question about the missing audible fact",
  "why":"the execution argument this observation would resolve",
  "audio_source":"original or original_and_source when offered"}; tool_request=null.
  The question may use prior answers and returned tool evidence, but must not ask
  Qwen to make user choices or accept a proposed edit. There is no automatic second
  round: choose listening only when the available audio observation can help.
  original_and_source supplies BOTH the original context and the current validated
  SAM extraction. The extraction is fallible evidence, not a replacement original.
- measure: choose an offered tool for a concrete unresolved fact or a physical
  boundary whose estimate needs independent evidence. Set tool_request={"name":
  "an available tool", "question_ids":[eligible requirement IDs]};
  observation_request=null. Explain why that tool's actual output can resolve
  the dependency. Obey max_questions and the remaining budget.
- unresolved: no available action can settle an essential dependency within budget.
  Both requests are null. Do not select an unsupported target, invent a window,
  broaden the edit, or reinterpret a missing gap as permission to move original audio.

There are at most two Qwen observations and two external tool calls in total.
The order is selected by the missing evidence. A tool may be used once; SAM source
extraction is at most once. A failed action still consumes its call budget.
When tools can provide needed cues that Qwen must interpret, obtain those cues
before the last listen. In particular, uncertain speech boundaries can use ASR
lexical landmarks and anonymous activity/gaps together. Do not spend the final
listen repeating unsupported times and only afterward discover the candidate gaps.
No fixed tool chain is required: stop when supported, and skip irrelevant tools.
speech_dependencies records timing uncertainty that has already occurred, even
if a later answer supplies more confident seconds. For these questions, collect
the capabilities in observation_prerequisites before the final listen. Only the
offered tools are allowed, in either order; do not spend the last observation
before the needed evidence exists. Such a dependency closes when the original
audio interpretation selects supported word/gap references, not when a new
timestamp merely makes the window positive. If a tool is unavailable or fails,
acknowledge the remaining limitation. A quiet subinterval inside a larger proposed
window does not certify that the entire window is between complete utterances.
In tool_augmented mode an insertion bounded by one spoken turn's end and the
next spoken turn's onset also needs lexical and acoustic evidence from the start,
even when the initial approximate times look positive. This is a requirement of
the requested bounded gap, not a reason to analyze whole-file BGM or literal times.
An each-speech-segment condition reported as a single whole-recording interval
also requires lexical evidence to verify that no actual pauses were discarded.
This does not apply to a user who directly requested a whole-file edit.
When speech_references is present, the last listen selects word IDs; code joins
unique flanking acoustic gaps. Do not ask Qwen to repeat gap IDs or event indices.
Boundary questions select the endpoint word of a selected target event. Ask which lexical
range matches each requested event, and whether a candidate pause follows a
completed utterance or interrupts the same utterance. The program compiles the
selected references into times. Existing free timestamps cannot close this
dependency. ASR is approximate and can be wrong: no suitable reference means
uncertain, not permission to invent a different time. Identity and timing
questions about the same utterance must share the same selected observation;
an unrelated utterance needs its own observed word range.
For each speech interval, ask Qwen to classify the supplied lexical_regions with
segment_assessments. Code creates the matched word-coverage windows and keeps
their pauses. This task does not require completed turns or acoustic silence in
the mixed recording: continuous BGM masks quiet speech gaps. Do not demand a
turn-ending tool or reject supported lexical regions merely because their timing
is approximate. Qwen must still confirm the spoken content from the original.
In dialogue_only mode external perception tools are unavailable; the same dialogue
budget and evidence requirements apply. This does not disable editing executors.

Measurements return candidates, not semantic identities. measurement_bindings may
contain only supported {question_id,event_id,region_id} entries from the offered
measurement_candidates. Binding replaces those event times with measured samples;
mentioning times in reason has no effect. Do not align by nearest timestamp or
infer target count from anonymous activity. A component covering multiple events
cannot authorize separate occurrences. Incomplete components cannot certify an
entire event. Requested source measurement must reach actual event evidence, or
be resolved by a new permitted audio observation; unchanged pre-tool estimates
and tool errors cannot close that dependency.
An each-occurrence source request must still cover every occurrence after a
targeted listen. Ask for all candidate verdicts when an inventory is supplied,
not merely confirmation of the first component. Match only positively heard
targets; rejected/unknown components are not automatic edit events.

For participant selection, locate only the events linked to that person's audible
identity. Person order is not event order. A separator or energy map cannot
establish who spoke, laughed or coughed. A reported boundary still needs the correct
semantic role; activity_end does not prove utterance completion. A between-window
requires two distinct supported anchors in the requested order and positive space.
If tools do not provide the needed kind of evidence, state that limitation.
""" + CAPABILITY_RULES

EXECUTOR_DEFAULTS = {
    'add_sound': 'Independent additions use existing generation and mixing defaults. '
        'Literal times/ranges need no event detection. Event-relative placement does. '
        'Default to overlay on the unchanged timeline, including between events. '
        'Adding sound never authorizes removing a source or shifting later original audio.',
    'remove_sound': 'A named source can be separated by its description across the '
        'whole recording or explicit range; other sources and the timeline remain. '
        'Selected occurrences or ambiguous identities need observed evidence.',
    'replace': 'A named source is separated, then new material covers the same scope '
        'without shifting other sources. No supplied range normally means the whole '
        'recording. Keeping narration timing needs no utterance measurements. '
        'Selected/each occurrences and ambiguous source identities need evidence.',
    'volume_change': 'No named/selected source means whole_mix gain over the explicit '
        'range or whole recording; do not infer a target from dominant audio. '
        'A named background/source uses separation, preserving the rest and timing. '
        'Gain and fades already have defaults. Only event-conditioned ranges, '
        'selected occurrences or ambiguous identities need audio facts.',
}

ANSWER_PROMPT = """Listen to the attached ORIGINAL recording. Supply only audible
facts needed for the requested edit. The instruction identifies the task, not
proof that a source is present. The planner decides the edit and its parameters.
Answer the requested question IDs. Distinguish requested
occurrences and source identities with audible cues; do not guess counts, speaker
turns, or exact times. For 'between' supply the two independently heard anchors,
not an invented gap event. Required sound intervals must include their full audible
extent. All estimates remain approximate. Unknown facts stay uncertain.
During further dialogue, previous answers are fallible context. Use the new question to
resolve the actual missing distinction; explain corrections and retain only
still-supported facts. Waveform activity cannot identify a source or speaker.
"""


EDIT_EVIDENCE_RULES = '''
The supplied audio_analysis contains completed planner/Qwen listening evidence.
Read answers, uncertainties and edit_listening.requirements as well as events.
Preserve question provenance: an audio anchor can carry question_id from the
matching timing requirement, with its unchanged reference and occurrence. Code
then binds that answer's measured events instead of asking another model to
rediscover the reference from a flattened inventory. For add placement_windows,
set listening_window_index on each affected edit (zero-based); code supplies its
immutable two-sided within constraint. A sole edit/window is linked automatically.
Use them to identify the original target, requested timing and sounds to preserve.
An event with timing_kind=boundary supplies only its explicitly measured semantic
roles. Reference that role through event_id; its null start/end cannot authorize
a removal interval or a generic event edge. Do not invent a span around a point.
The original user instruction remains authoritative; observations cannot broaden
the requested scope, change an occurrence or turn a future added sound into a
heard source. Preserve other sounds and explicit preservation constraints. A
non-null edit_listening.participant_selection selects the observed person's target
events. Preserve its identity-qualified target and linked timing windows; never
replace them with whole-file text separation of an ordinal speaker label. The
person's ordinal is separate from the occurrence of that person's sound. A
dominant source is not a sole source. Use existing source separation when uncertain
about overlap. Bind observed timing through this specialist's existing reference
schema; never invent missing boundaries.
Only user_prompt is the original instruction. planner_subgoal is a derived hint,
not a user constraint. Qwen timestamps are approximate observations, never explicit
user times. Use event_id references for them; do not turn them into literal user
intervals. Neither Qwen nor a reviewer can grant permission or state user choices.
Choose unspecified mix parameters using the existing defaults, not listening advice.
No more audio listening is available. If essential timing is missing, keep it
unresolved. Explicit file times do not require event detections.
'''


def _questions(data, *, force_audio=False):
    return questions_from_brief(data, force_audio=force_audio)


def _review_view(payload):
    """Put each verbatim answer next to its supporting observations, without transport metadata."""
    result = deepcopy(payload)
    result['requirements'] = [{k:v for k,v in q.items() if k != 'answer_contract'}
                              for q in payload.get('requirements', [])]
    turns = result.get('turns', [])
    current = next((i for i in range(len(turns)-1, -1, -1) if turns[i].get('analysis')), None)
    if turns and turns[-1].get('action') == 'qwen' and not turns[-1].get('analysis'):
        current = None
    result['evidence_policy'] = dict(
        current_fact_turn=None if current is None else current+1,
        historical_facts='Superseded snapshots, retained for comparison; not cumulative constraints on corrected facts.',
        tool_scope='Mixed waveform measurements of the claim at that turn; inconclusive is not contrary evidence.')
    for index, turn in enumerate(turns):
        turn['facts_are_current'] = index == current
        if index != current:


            turn.pop('missing_evidence', None)
            turn.pop('decision_risks', None)
        for check in turn.get('boundary_activity_checks', []):
            check['evidence_effect'] = {
                'ambiguous': 'inconclusive', 'consistent': 'mixed_energy_agreement',
                'conflict': 'measured_conflict_with_that_reported_claim',
            }.get(check.get('status'), 'no_conclusion')
        analysis = turn.pop('analysis', None)
        turn.pop('questions', None)                                                      
        if not analysis:
            continue
        if index != current:


            turn['observation_status'] = 'superseded; use the current fact turn'
            continue
        events = analysis.get('events', [])
        turn['facts'] = []
        for answer in analysis.get('answers', []):
            key = answer.get('question_id')
            observations = []
            for event in events:
                if key not in (event.get('question_ids') or []):
                    continue
                observations.append({k:deepcopy(event[k]) for k in ('event_id', 'event', 'occurrence_index',
                    'timing_kind', 'start_time', 'end_time', 'boundaries', 'resolution_status',
                    'timing_source', 'measurement_provenance', 'speech_reference_provenance') if k in event})
            turn['facts'].append(dict(answer, observations=observations))
        turn['unlinked_observations'] = [{k:deepcopy(e[k]) for k in ('event', 'occurrence_index',
            'start_time', 'end_time', 'boundaries', 'resolution_status') if k in e}
            for e in events if not e.get('question_ids')]
        for key in ('source_assessments', 'uncertainties', 'unlocated_observations', 'measurement_reference_errors',
                    'speech_reference_errors', 'speech_evidence', 'source_region_assessments', 'speech_segment_assessments', 'evidence_needs'):
            if analysis.get(key):
                turn[key] = deepcopy(analysis[key])
    return result


def _quote_text(value):
    """Ignore balanced quotation wrappers and whitespace, not the quoted facts."""
    if not isinstance(value, str):
        return ''
    value = ' '.join(value.split())
    pairs = {'"': '"', "'": "'", '“': '”', '‘': '’', '「': '」', '『': '』'}
    while len(value) >= 2 and pairs.get(value[0]) == value[-1]:
        value = value[1:-1].strip()
    return value


def _assessment_gaps(review, questions, analysis):
    """A review cannot claim sufficient support after explicitly reporting none."""
    checks = review.get('evidence_assessment')
    if checks is None:
        return {}
    quotes = [a.get('answer', '') for a in (analysis or {}).get('answers', [])]
    quotes += [p.get('evidence', '') for e in (analysis or {}).get('events', [])
               for p in (e.get('boundaries') or {}).values() if isinstance(p, dict)]
    quotes = [_quote_text(text) for text in quotes]
    missing = {}
    for q in questions:
        check = checks.get(q['id'], {}) if isinstance(checks, dict) else {}
        cue, problem = _quote_text(check.get('cue_quote')), check.get('problem')
        issues = []
        if q['required_fields'] != ['overlap'] and (not cue or not any(cue in source for source in quotes)):
            issues.append('Review did not cite a current audible cue supporting this selection or boundary')
        if isinstance(problem, str) and problem.strip():
            issues.append(problem)
        if issues:
            missing[q['id']] = issues
    return missing


def _current_cue_choices(payload):
    """Constrain quotation to existing current text, without judging its meaning."""
    import re
    turns = payload.get('turns', [])
    analysis = turns[-1].get('analysis') if turns else None
    sources = [a.get('answer', '') for a in (analysis or {}).get('answers', [])]
    sources += [p.get('evidence', '') for e in (analysis or {}).get('events', [])
                for p in (e.get('boundaries') or {}).values() if isinstance(p, dict)]
    choices = []
    for source in sources:
        if not isinstance(source, str):
            continue
        for sentence in re.split(r'(?<=[.!?])\s+', source.strip()):
            if sentence and sentence not in choices:
                choices.append(sentence)
    return [None, *choices]


def _planner(llm, system, payload, audit):
    volume_brief = system == QUESTION_PROMPT and payload.get('operation') == 'volume_change'
    replace_brief = system == QUESTION_PROMPT and payload.get('operation') == 'replace'
    effective_system = (REPLACE_QUESTION_PROMPT if replace_brief else
                        VOLUME_QUESTION_PROMPT if volume_brief else system)
    if system == REVIEW_PROMPT and payload.get('operation') == 'replace':
        effective_system = REPLACE_INTERVAL_REVIEW_NOTE + effective_system
    record = dict(system_prompt=effective_system, payload=deepcopy(payload))
    audit['planner_calls'].append(record)
    from audio_edit.analyzers.listening_examples import BRIEF_EXAMPLES, VOLUME_BRIEF_EXAMPLES, review_examples
    examples = BRIEF_EXAMPLES if system == QUESTION_PROMPT else review_examples(payload)
    if volume_brief:
        examples = VOLUME_BRIEF_EXAMPLES
    if system == QUESTION_PROMPT:
        examples = deepcopy(examples)
        for example in examples:
            example['output'].setdefault('participant_selection', None)
            if volume_brief:
                output = example['output']
                example['output'] = dict(reason=output['reason'], **{k:v for k,v in output.items() if k!='reason'})
            for q in example['output'].get('questions',[]):q.setdefault('speech_scope',None)
    if system == REVIEW_PROMPT:
        examples = deepcopy(examples)
        for example in examples:
            output = example['output']
            output.setdefault('measurement_bindings', [])
            example['output'] = dict(**({'evidence_assessment':output['evidence_assessment']} if 'evidence_assessment' in output else {}),
                reason=output['reason'], **{k:v for k,v in output.items() if k not in ('reason','evidence_assessment')})
            example['input'] = _review_view(example['input'])
    record['few_shot_examples'] = deepcopy(examples)
    options = {}
    if getattr(llm, 'backend', None) == 'openai':


        if system == REVIEW_PROMPT:
            statuses = ['unresolved']
            if payload.get('ready_available', True):
                statuses.insert(0, 'ready')
            if payload.get('observations_available'):
                statuses.append('listen')
            if payload.get('available_tools'):
                statuses.append('measure')
            observation_properties = dict(
                question_id=dict(type='string', enum=[q['id'] for q in payload['requirements']]),
                question=dict(type='string'), why=dict(type='string'),
                audio_source=dict(type='string', enum=payload.get('available_audio_sources',['original'])))
            observation_shape = dict(type='object', additionalProperties=False,
                properties=observation_properties, required=list(observation_properties))
            assessment = dict(type='object', additionalProperties=False,
                properties={q['id']: dict(type='object', additionalProperties=False,
                    properties=dict(cue_quote=dict(type=['string','null'], enum=_current_cue_choices(payload)),
                                    problem=dict(type=['string','null'])),
                    required=['cue_quote','problem']) for q in payload['requirements']},
                required=[q['id'] for q in payload['requirements']])
            properties = dict(evidence_assessment=assessment, reason=dict(type='string'), status=dict(type='string', enum=statuses),
                observation_request=(dict(anyOf=[dict(type='null'),observation_shape])
                    if payload.get('observations_available') else dict(type='null')),
                tool_request=tool_request_schema(payload.get('available_tools', [])))
            candidates = payload.get('measurement_candidates', [])
            properties['measurement_bindings'] = dict(type='array', maxItems=len(candidates), items=dict(
                type='object', additionalProperties=False,
                properties={key:dict(type='string', enum=list(dict.fromkeys(c[key] for c in candidates)) or ['unavailable'])
                            for key in ('question_id','event_id','region_id')},
                required=['question_id','event_id','region_id']))
        else:
            question_properties = dict(focus=dict(type='string', enum=['target', 'timing', 'preservation']),
                reference_kind=dict(type='string', enum=list(REFERENCE_KINDS)),
                speech_scope=dict(type=['string','null'],enum=[None,'all_spoken_activity','selected_content']),
                subject=dict(type='string'), question=dict(type='string'), why=dict(type='string'),
                occurrence=dict(anyOf=[dict(type='integer'), dict(type='string', enum=['one', 'first', 'last', 'each'])]),
                required_fields=dict(type='array', items=dict(type='string', enum=list(FACTS))))
            variants = []
            for focus, fields in FOCUS_FIELDS.items():
                props = deepcopy(question_properties)
                props['focus'] = dict(type='string', enum=[focus])
                props['required_fields'] = dict(type='array', minItems=1, maxItems=1,
                    items=dict(type='string', enum=[f for f in FACTS if f in fields]))
                if payload.get('operation') == 'remove_sound':


                    props['speech_scope'] = dict(type=['string','null'], enum=[None,'selected_content'])
                variants.append(dict(type='object', additionalProperties=False,
                                     properties=props, required=list(props)))
            if payload.get('operation') == 'remove_sound':
                props = deepcopy(question_properties)
                props.update(focus=dict(type='string', enum=['timing']),
                    reference_kind=dict(type='string', enum=['speech']),
                    speech_scope=dict(type='string', enum=['all_spoken_activity']),
                    occurrence=dict(type='string', enum=['each']),
                    required_fields=dict(type='array', minItems=1, maxItems=1,
                        items=dict(type='string', enum=['interval'])))
                variants.append(dict(type='object', additionalProperties=False,
                                     properties=props, required=list(props)))
            question = dict(anyOf=variants)
            properties = dict(selection=dict(type='object', additionalProperties=False,
                                  properties={k: dict(type='string', enum=list(v)) for k, v in SELECTION_FIELDS.items()},
                                  required=list(SELECTION_FIELDS)),
                              participant_selection=participant_selection_schema(payload),
                              reason=dict(type='string'),
                              timeline_policy=dict(anyOf=[dict(type='null'), dict(type='object',
                                  additionalProperties=False, properties=dict(
                                      mode=dict(type='string', enum=['preserve', 'retime']),
                                      user_quote=dict(type=['string', 'null'])), required=['mode', 'user_quote'])]),
                              placement_windows=dict(type='array', maxItems=3, items=dict(
                                  type='object', additionalProperties=False,
                                  properties={k: dict(type='string') for k in ('start_question_id', 'end_question_id')},
                                  required=['start_question_id', 'end_question_id'])),
                              questions=dict(type='array', minItems=0, maxItems=6, items=question))
            if volume_brief:
                properties = dict(reason=properties['reason'], **{k:v for k,v in properties.items() if k!='reason'})
            if routed_participant(payload):
                properties['selection']['properties']['target']['enum'] = ['ambiguous_source']
                properties['questions']['minItems'] = 2
        options['response_format'] = dict(type='json_schema', json_schema=dict(
            name='edit_listening_review' if system == REVIEW_PROMPT else 'edit_listening_brief',
            strict=True, schema=dict(type='object', additionalProperties=False,
                properties=properties, required=list(properties))))
        record['response_format'] = deepcopy(options['response_format'])
    messages = [dict(role='system', content=effective_system + '\nThe following demonstrations are synthetic examples, not observations of the current audio. Never transfer their sources, times or answers to the current request.')]
    for example in examples:
        messages.extend([dict(role='user', content=json.dumps(example['input'], ensure_ascii=False)),
                         dict(role='assistant', content=json.dumps(example['output'], ensure_ascii=False))])
    model_payload = _review_view(payload) if system == REVIEW_PROMPT else payload
    record['model_payload'] = deepcopy(model_payload)
    messages.append(dict(role='user', content=json.dumps(model_payload, ensure_ascii=False)))
    response = llm.client.chat.completions.create(model=llm.model, messages=messages, **options)
    record['raw_response'] = response.choices[0].message.content
    record['response'] = extract_json_object(record['raw_response'])
    return record['response']


def _observation_questions(required, request, missing):
    """A new observation targets one existing edit dependency, without draft plans."""
    if not isinstance(request, dict) or set(request) not in (
            {'question_id','question','why'}, {'question_id','question','why','audio_source'}) or any(
            not isinstance(request.get(key), str) or not request[key].strip()
            for key in ('question_id','question','why')):
        raise UnresolvedEditError('Listening needs one existing question ID, a targeted question and its purpose')
    question = next((q for q in required if q['id']==request['question_id']), None)
    if question is None:
        raise UnresolvedEditError('Observation request cannot invent a new edit dependency')
    result = deepcopy(question)
    result['question'] = request['question']
    result['probe_reason'] = request['why']
    result['missing_evidence'] = deepcopy(missing.get(question['id'], []))
    if question.get('occurrence') == 'each':
        result['question'] += (' The original requirement still selects EACH matching occurrence in the full recording. '
            'Check the complete candidate inventory and original audio; do not narrow the answer to the single '
            'example mentioned in this probe. Give every offered source region a match/no_match/unknown verdict '
            'when region_assessments is required. A region is not a target merely because the separator returned it.')
    return [result]


def _source_audio_available(measurement, original_hash, duration):
    """Only a current, unchanged source artifact may be offered for joint listening."""
    if not isinstance(measurement, dict) or measurement.get('tool')!='source_activity':
        return False
    path = measurement.get('audio_path')
    if not isinstance(path, str) or not Path(path).is_file() or measurement.get('original_audio_sha256')!=original_hash:
        return False
    try:
        return (sha256(path)==measurement.get('audio_sha256')
                and abs(get_media_duration(path)-duration)<=.02)
    except (OSError,ValueError):
        return False


def _measurement_catalogs(questions, measurement, source_measurement, pending_source_ids):
    """Keep the source catalog for its questions after another waveform check.

    A mixed-waveform inventory remains available for other questions. It cannot
    replace a requested source measurement, including when that extraction failed.
    Both artifacts were validated by merge_tool_evidence before entering here.
    """
    catalogs = []
    for value, keys in ((source_measurement, set(pending_source_ids)),
                        (measurement, {q['id'] for q in questions} - set(pending_source_ids))):
        if not value:
            continue
        ids = [q['id'] for q in questions if q['id'] in keys
               and q['id'] in value.get('question_ids', [])]
        if ids:
            catalogs.append(dict(value, question_ids=ids))
    return catalogs


def _catalog_questions(questions, catalogs, rejected):
    result = []
    for question in questions:
        catalog = next((m for m in catalogs if question['id'] in m['question_ids']), None)
        result.extend(with_measured_regions([question], catalog, rejected=rejected))
    return result


def _catalog_candidates(analysis, catalogs, questions):
    return [c for m in catalogs for c in measurement_candidates(analysis, m, questions)]


def _apply_catalog_bindings(analysis, catalogs, questions, bindings):
    """Resolve each binding against its question's original validated artifact."""
    if not isinstance(bindings, list):
        raise ValueError('measurement_bindings must be an array')
    groups, selected = {}, {}
    for binding in bindings:
        if not isinstance(binding, dict) or set(binding) != {'question_id','event_id','region_id'}:
            raise ValueError('A measurement binding needs exactly question_id, event_id and region_id')
        matches = [i for i, m in enumerate(catalogs) if binding['question_id'] in m['question_ids']]
        if len(matches) != 1:
            raise ValueError('Measurement binding needs its question-specific catalog')
        index = matches[0]
        identity = (index, binding['region_id'])
        event = binding['event_id']
        if event in selected and selected[event] != identity:
            raise ValueError('The same event cannot bind conflicting measurement artifacts')
        selected[event] = identity
        groups.setdefault(index, []).append(binding)
    result = analysis
    for index, group in groups.items():
        result = apply_measurement_bindings(result, catalogs[index], questions, group)
    return result


def listen_for_edit(ctx, llm, analyzer, *, operation, input_audio=None):
    """Supervise bounded audio dialogue and optional tools until the edit is supported."""
    from audio_edit.analyzers.source_inventory import source_inventory_gaps
    path = input_audio or ctx.input_audio
    mode = ctx.params.get('perception_mode', os.environ.get('AUDIO_EDIT_PERCEPTION_MODE','tool_augmented'))
    if mode not in ('tool_augmented','dialogue_only'):
        raise ValueError('perception_mode must be tool_augmented or dialogue_only')
    payload = dict(original_instruction=ctx.user_prompt, operation=operation,
        media_duration=get_media_duration(path), routed_selection=routed_selection(ctx.params),
        executor_defaults=EXECUTOR_DEFAULTS.get(operation, 'Preserve the original instruction and timeline.'))
    audit = dict(version=25, status='pending', perception_mode=mode, **payload,
        planner_calls=[], turns=[], tool_calls=[],
        budget=dict(max_qwen_listens=2, max_tool_calls=2, max_cpu_checks=2,
                    max_preedit_separations=1, initial_activity_maps=0),
        usage=dict(qwen_listens=0, tool_calls=0, source_extractions=0))
    ctx.params['edit_listening_dialogue'] = audit
    try:

        brief = _planner(llm, QUESTION_PROMPT, payload, audit)
        from audio_edit.analyzers.output_intent import review as review_output_intent
        brief = review_output_intent(llm, brief, payload, audit)
        participant = participant_from_brief(brief, payload)
        required = _questions(brief, force_audio=bool(participant))
        bind_participant_questions(participant, required, operation)
        windows = placement_windows_from_brief(brief, required, operation)
        timeline_policy = timeline_policy_from_brief(brief, ctx.user_prompt, operation)
        audit.update(participant_selection=deepcopy(participant), timeline_policy=deepcopy(timeline_policy),
            placement_windows=deepcopy(windows), selection=deepcopy(brief.get('selection')),
            requirements=deepcopy(required), listening_reason=brief.get('reason'))
        if not required and brief.get('questions'):
            audit['unnecessary_questions'] = deepcopy(brief['questions'])
        if not required:
            analysis = dict(events=[], answers=[], uncertainties=[], source_assessments=[],
                media_duration=payload['media_duration'], mixing_recommendations={},
                evidence_source='not_observed', instruction_authority=False,
                edit_listening=dict(status='not_required', perception_mode=mode,
                    reason=brief['reason'], requirements=[], timeline_policy=deepcopy(timeline_policy),
                    instruction_source='original_user_instruction'))
            audit.update(status='not_required', final_analysis=deepcopy(analysis))
            return analysis

        analysis = measurement = source_measurement = None
        pending_source_ids = set()
        pending_tool_errors = {}
        speech_dependencies = {}
        used_tools = set()
        qwen_calls = tool_calls = 0
        action, questions, audio_source = 'qwen', required, 'original'
        original_hash = sha256(path)
        for turn in range(4):


            required = with_speech_references(required, audit['turns'],
                duration=payload['media_duration'], original_hash=original_hash)
            current = {q['id']: q for q in required}
            questions = [dict(deepcopy(current[q['id']]), **{k: deepcopy(q[k])
                for k in ('question', 'probe_reason', 'missing_evidence') if k in q}) for q in questions]
            audit['requirements'] = deepcopy(required)
            previous = [{key:deepcopy(value) for key,value in record.items() if key in (
                'round','action','questions','analysis','error','missing_evidence',
                'decision_risks','boundary_activity_checks','measurement','tool_evidence')}
                for record in audit['turns']]
            record = dict(round=turn+1, action=action, questions=deepcopy(questions))
            audit['turns'].append(record)
            if action == 'qwen':
                qwen_calls += 1
                audit['usage']['qwen_listens'] = qwen_calls
                try:
                    rejected = [error for old in audit['turns'] for error in
                        (old.get('analysis') or {}).get('measurement_reference_errors', [])]
                    catalogs = _measurement_catalogs(questions, measurement, source_measurement, pending_source_ids)
                    sent = _catalog_questions(questions, catalogs, rejected)
                    record['questions'] = deepcopy(sent)
                    options = dict(previous_turns=previous, operation=operation)
                    record['audio_source'] = audio_source
                    if audio_source=='original_and_source':
                        if not _source_audio_available(source_measurement,original_hash,payload['media_duration']):
                            raise ValueError('Source artifact changed or is not aligned with this original audio')
                        options['evidence_audio'] = deepcopy(source_measurement)
                    raw = analyzer.analyze_edit(path, ctx.user_prompt, sent, **options)
                    analysis = observation_view(raw, payload['media_duration'])


                    fresh_gaps = check_evidence(required, analysis, payload['media_duration'])
                    fresh_answers = {a.get('question_id'):a for a in analysis.get('answers', [])}
                    for key in {q['id'] for q in questions} & pending_tool_errors.keys():
                        if fresh_answers.get(key,{}).get('status')=='answered' and key not in fresh_gaps:
                            record.setdefault('tool_error_recoveries', []).append(dict(
                                question_id=key, failed_tool=pending_tool_errors[key]['tool'],
                                recovery_source='qwen_observation', approximate=True,
                                limitation='New original-audio observation; no support claimed from the failed tool'))
                            del pending_tool_errors[key]
                    record['analysis'] = deepcopy(analysis)
                except Exception as exc:
                    analysis = None
                    record['error'] = f'{type(exc).__name__}: {exc}'
            else:
                tool_calls += 1
                used_tools.add(action)
                audit['usage'].update(tool_calls=tool_calls,
                    source_extractions=int('source_activity' in used_tools))
                if action == 'source_activity':
                    pending_source_ids = {q['id'] for q in questions}
                call = dict(name=action, question_ids=[q['id'] for q in questions])
                try:
                    if action == 'boundary_activity':
                        analysis, result = check_boundary_activity(questions, analysis, path)
                        record['boundary_activity_checks'] = deepcopy(result)
                    else:
                        result = run_measurement(action, path, questions, ctx)
                        analysis, measured = merge_tool_evidence(action, result, analysis, required,
                            payload['media_duration'], original_hash=original_hash)
                        if measured is not None:
                            measurement = measured
                            record['measurement'] = deepcopy(measured)
                            if action == 'source_activity':
                                source_measurement = measured
                    record['tool_evidence'] = deepcopy(result)
                    call['results'] = deepcopy(result)
                except Exception as exc:
                    record['error'] = call['error'] = f'{type(exc).__name__}: {exc}'
                    for q in questions:
                        pending_tool_errors[q['id']] = dict(tool=action, error=call['error'])

                record['analysis'] = deepcopy(analysis)
                audit['tool_calls'].append(call)

            required = with_speech_references(required, audit['turns'],
                duration=payload['media_duration'], original_hash=original_hash)
            audit['requirements'] = deepcopy(required)
            if mode == 'tool_augmented':
                speech_dependencies = update_speech_dependencies(speech_dependencies, required, analysis,
                    windows=windows, turns=audit['turns'], duration=payload['media_duration'])
            audit['speech_dependencies'] = deepcopy(speech_dependencies)

            def gaps():
                missing = check_evidence(required, analysis, payload['media_duration'])
                for key, error in pending_tool_errors.items():
                    missing.setdefault(key, []).append(
                        'Requested '+error['tool']+' failed; old observations cannot ignore this missing evidence. '
                        'A new permitted audio observation must explicitly resolve this requirement; '
                        'source-measurement binding and actual physical conflicts remain separate constraints.')
                for source in (check_placement_windows(windows, required, analysis),
                        speech_reference_gaps(required, analysis),
                        speech_dependency_gaps(speech_dependencies, required, analysis),
                        unresolved_measurement_conflicts(analysis, measurement, required),
                        source_inventory_gaps(analysis, source_measurement, required, pending_source_ids),
                        source_handoff_gaps(analysis, source_measurement, pending_source_ids)):
                    for key, issues in source.items():
                        missing.setdefault(key, []).extend(issues)
                return missing

            missing = gaps()
            record['pending_source_questions'] = sorted(pending_source_ids)
            record['missing_evidence'] = deepcopy(missing)
            risks = boundary_decision_risks(required, analysis, record.get('boundary_activity_checks', []))
            record['decision_risks'] = deepcopy(risks)
            prerequisites = speech_prerequisites(speech_dependencies, required, audit['turns'])
            observations_available = qwen_calls < 2 and not prerequisites
            record['observation_prerequisites'] = deepcopy(prerequisites)
            audio_sources = ['original']
            if mode=='tool_augmented' and _source_audio_available(source_measurement,original_hash,payload['media_duration']):
                audio_sources.append('original_and_source')
            tools_available = available_tools(required, analysis,
                enabled=mode=='tool_augmented' and tool_calls<2, duration=payload['media_duration'])

            tools_available = [tool for tool in tools_available if tool['name'] not in used_tools]
            if prerequisites:
                tools_available = [tool for tool in tools_available if tool['name'] in prerequisites]
            record['available_tools'] = deepcopy(tools_available)
            catalogs = _measurement_catalogs(required, measurement, source_measurement, pending_source_ids)
            catalog_context = {}
            if source_measurement is not None and measurement is not source_measurement:


                catalog_context['measurement_catalogs'] = deepcopy(catalogs)
            review = _planner(llm, REVIEW_PROMPT, dict(**payload, **catalog_context, perception_mode=mode,
                requirements=required, placement_windows=windows, timeline_policy=timeline_policy,
                participant_selection=deepcopy(participant), turns=previous+[deepcopy(record)],
                measurement=deepcopy(measurement),
                measurement_candidates=_catalog_candidates(analysis, catalogs, required),
                missing_parameters={key:dict(parameter='where' if next(q for q in required if q['id']==key)['focus']=='timing'
                    else 'what_or_preserve', issues=value) for key,value in missing.items()},
                available_tools=tools_available, observations_available=observations_available,
                ready_available=not speech_dependency_gaps(speech_dependencies, required, analysis),
                speech_dependencies=deepcopy(speech_dependencies), observation_prerequisites=prerequisites,
                available_audio_sources=audio_sources,
                remaining_budget=dict(qwen_listens=2-qwen_calls, tool_requests=2-tool_calls,
                    source_extractions=int('source_activity' not in used_tools)),
                missing_evidence=missing, decision_risks=risks), audit)
            record['review'] = deepcopy(review)
            bindings = review.get('measurement_bindings', [])
            if bindings:
                analysis = _apply_catalog_bindings(analysis, catalogs, required, bindings)
                record['measurement_bindings'] = deepcopy(bindings)
                record['analysis'] = deepcopy(analysis)
                missing = gaps()
            for key, issues in _assessment_gaps(review, required, analysis).items():
                missing.setdefault(key, []).extend(issues)
            record['missing_evidence'] = deepcopy(missing)
            status = review.get('status')
            if status not in ('ready','listen','measure','unresolved'):
                raise UnresolvedEditError('Unsupported evidence action; use ready, listen, measure or unresolved')
            if not isinstance(review.get('reason'), str) or not review['reason'].strip():
                raise ValueError('Evidence supervision needs a reason')
            observation_request, request = review.get('observation_request'), review.get('tool_request')
            if status == 'listen':
                if not observations_available or request is not None:
                    raise UnresolvedEditError('Audio observation is unavailable or combined with another action')
                questions = _observation_questions(required, observation_request, missing)
                audio_source = observation_request.get('audio_source','original')
                if audio_source not in audio_sources:
                    raise UnresolvedEditError('The requested observation audio source is unavailable')
                action = 'qwen'
                record['effective_status'] = 'listen'
                continue
            if observation_request is not None:
                raise UnresolvedEditError('Only listen status may request an audio observation')
            if status == 'measure':
                error = tool_request_error(request, tools_available)
                if error:
                    record['rejected_tool_request'] = dict(request=deepcopy(request),error=error)
                    raise UnresolvedEditError(error)
                action = request['name']
                questions = [q for q in required if q['id'] in request['question_ids']]
                record['effective_status'] = 'measure'
                continue
            if request is not None:
                raise UnresolvedEditError('Only measure status may request a tool')
            if status == 'unresolved' or missing or analysis is None:
                record['effective_status'] = 'unresolved'
                raise UnresolvedEditError('Missing factual evidence: '+json.dumps(missing)
                                          if missing else review['reason'])
            if sha256(path) != original_hash:
                raise UnresolvedEditError('Original audio changed during evidence supervision')
            record['effective_status'] = 'ready'
            analysis['edit_listening'] = dict(status='ready', perception_mode=mode,
                requirements=deepcopy(required), participant_selection=deepcopy(participant),
                selection=deepcopy(brief.get('selection')), placement_windows=deepcopy(windows),
                timeline_policy=deepcopy(timeline_policy), missing_evidence={},
                instruction_source='original_user_instruction',
                timing_source='mixed_measured_and_perceptual' if any(e.get('measurement_provenance')
                    for e in analysis.get('events',[])) else 'speech_reference' if any(
                        e.get('speech_reference_provenance') for e in analysis.get('events', [])) else 'qwen_observation')
            if measurement:
                analysis['preedit_measurement'] = deepcopy(measurement)
            audit.update(status='ready', final_analysis=deepcopy(analysis))
            return analysis
        raise UnresolvedEditError('The evidence action budget is exhausted')
    except Exception as exc:
        audit['status'] = 'unresolved' if isinstance(exc,UnresolvedEditError) else 'error'
        audit['error'] = f'{type(exc).__name__}: {exc}'
        raise


def source_evidence(analysis, target, source):
    """Adapt exact, file-bound observations to the existing separator bypass guard.

    A renamed or ambiguous target simply uses separation. Never infer sole-source
    status from a prose answer or a dominant event. The executor rechecks coverage.
    """
    if not analysis or not isinstance(analysis.get('audio_evidence'), dict):
        return None
    duration = analysis.get('media_duration')
    windows = []
    for row in analysis.get('source_assessments', []):
        if not isinstance(row, dict) or row.get('target_sound') != target:
            continue
        a, b = row.get('start_time'), row.get('end_time')
        if (not all(type(v) in (int, float) and math.isfinite(v) for v in (a, b, duration))
                or not 0 <= a < b <= duration):
            continue

        conflicts = [other for other in analysis.get('source_assessments', [])
            if isinstance(other, dict) and other.get('target_sound') == target
            and all(type(other.get(k)) in (int, float) and math.isfinite(other[k])
                    for k in ('start_time', 'end_time'))
            and max(a, other['start_time']) < min(b, other['end_time'])
            and (other.get('only_target') is not True or other.get('target_present') is not True
                 or other.get('other_sounds') != [])]
        if not conflicts:
            windows.append(dict(start_time=a, end_time=b, observation=deepcopy(row)))
    return dict(source=source, target_sound=target,
                audio=deepcopy(analysis['audio_evidence']), windows=windows)


REPLACE_INTERVAL_REVIEW_NOTE = "For REPLACE evidence review, an approximate full audible interval can be an\nexecutable interval. Do not invent an exact or sample-accurate boundary requirement\nmerely because one sound will be replaced by another. When the current original\nobservation identifies the requested occurrence with audible distinguishing cues\nand supplies its complete span, and no code-reported dependency or contradictory\nevidence remains, retain that supported approximate interval. Its approximate\nlabel alone is not a missing fact and is not a reason to extract another source\nor search for silence. Source separation can preserve overlapping audio.\nUse measurement for a concrete unresolved boundary, conflicting evidence, or a\nrequested relationship that the current facts cannot establish. Keep all genuine\nidentity, participant, ordinal, count and timing dependencies. An anonymous region\nthat contains a target is not automatically the target's complete interval. Do\nnot discard previously obtained conflicting evidence or a requested source-tool\nhandoff; this clarification supplies no new audible evidence.\n\n"
