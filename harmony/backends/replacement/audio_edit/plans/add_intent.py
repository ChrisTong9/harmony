"""Add's authoritative intent and single compiler, before and after generation.

AddSoundPlan.intent owns meaning. Observations/bindings supply evidence, never a
replacement intent. compile_add_intent is pure: pending asset length is allowed,
missing event evidence is not. The old window compiler is only the numeric safety
backend here; it receives fully specified windows, operations and offsets.
"""

from copy import deepcopy
import json
from statistics import median

from audio_edit.plans.edit_contract import compile_edits, resolve_anchor, number
from audio_edit.plans.event_ids import assign_event_ids, has_measured_timing
from audio_edit.plans.temporal_boundaries import BoundaryEvidenceError, select_event
from audio_edit.plans.listening_handoff import reference_key, question_bindings, validate_listening_windows

SOURCES = ("user_explicit", "system_default", "planner_choice")

ADD_INTENT_PROMPT = """Understand the complete addition instruction ONCE. Return
JSON with intent_version=2, edits, is_on_screen and optional top-level
audio_prompt (a default for edits), speech_text, voice_prompt, negative_audio_prompt,
layer_role, composer, notes. Each edit must have its own non-empty audio_prompt or
inherit the top-level default. Do not repeat the same description at both levels.
Do NOT output temporal, start_time, end_time, duration, window, placement_anchors,
resolved placements or a second expression of the same timing constraint.

Every edit MUST explicitly declare generator="audio"|"speech" AND
 audio_reference=null|"context"|"structure". Omitting audio_reference is invalid.
Choose "context" when the new sound must harmonize with, accompany or respond
musically to the existing recording. Choose "structure" for transforming an
existing melody/rhythm while retaining it. Choose null for an independent sound
or spoken text. Writing "follow the original chords/tempo" in audio_prompt cannot
supply the original audio: only audio_reference routes the actual samples.
Matching event count, placement, onset or duration is handled by the edit windows
and repetition controls; it is NOT a reason to initialize generation with the
old sound. For substitution by a different independent sound, use null even when
every occurrence must keep its original time. An audio reference conditions the
waveform itself and can retain the OLD source's timbre. Reserve structure for
preserving an actual musical performance or explicitly requested acoustic
structure, not the mere fact that old and new events occupy the same windows.

Explicit event counts and matching tempo are placement requirements. For N beats,
hits or repeated sounds, use placement.kind="repeat" with repeat.count=N; generate
ONE hit per asset. Never use a single multi-beat asset with the count only in
text. When the instruction requires matching an existing rate, declare
repeat.spacing_from for the existing rhythmic sequence; audio_reference alone
cannot establish or enforce its tempo. Do not guess the total sequence duration.

Each edit contains generator, audio_reference, placement, timeline_op (overlay/insert/prepend/append),
overflow_policy (error/extend), generation, audio_prompt (ONE new sound only),
layer_role, optional relative_level_db, reason, and sources. sources records only
placement/timeline_op/overflow_policy as user_explicit, system_default or
planner_choice. Operation values and provenance labels are DIFFERENT fields:
{"timeline_op":"overlay","overflow_policy":"error",
 "sources":{"placement":"user_explicit","timeline_op":"system_default","overflow_policy":"planner_choice"}}.
Never put system_default/user_explicit/planner_choice in timeline_op or overflow_policy.
timeline_op is ONLY overlay/insert/prepend/append; overflow_policy is ONLY error/extend.
Explicit constraints must be preserved. Defaults are not permission
to relax a constraint. Unknown sound length never authorizes moving a reference.
The instruction-derived edit_listening.timeline_policy owns timeline permission.
When mode=preserve, EVERY edit uses overlay/error, even if the user writes
"insert a sound". That ordinary verb does not request extra timeline space.
Only mode=retime with a quoted request to shift original audio, insert new time,
append/prepend beyond the recording or extend its duration allows retiming.
Do not relabel an insertion as user_explicit to override this prior contract.

Choose ONLY the appropriate placement shape:
- One sound: {"kind":"single","anchor":ANCHOR,"alignment":"start|end|center"}.
  One sound for EACH existing event also uses kind="single" with the anchor's
  occurrence="each". Code expands the observed references; do not add repeat.count=1
  or an empty/zero spacing. Use repetition rules only for multiple sounds per origin.
- Continuous coverage: {"kind":"cover","range":{"start":ANCHOR,"end":ANCHOR}}.
  Its length is computed from the range; generation can be omitted or null.
  Continuous sound during EACH reference event uses paired range start/end
  anchors with occurrence="each". Do not replace required coverage with a short
  discrete sound at each onset.
- Repetition from one origin: {"kind":"repeat","anchor":ANCHOR,
  "alignment":"start|end|center","repeat":{"count":integer,
  "spacing_seconds":number}}. Spacing is onset-to-onset, NOT silence duration.
  When matching an existing beat, use spacing_from=ANCHOR instead of a guessed
  spacing_seconds. Reference the audible rhythmic sequence with occurrence=first
  (or the requested occurrence); perception reports successive beat onsets, code
  computes the interval. No guessed BPM, half-second spacing, or arithmetic by you.
  For explicitly contiguous repetition use {"count":integer,"mode":"sequential"}.
  For individual different references use {"kind":"repeat","anchors":[ANCHOR,...],
  "alignment":"start|end|center"}; the list supplies the count, never repeat it.
- A sound constrained between two references can use placement.within={start,end}
  in place of anchor. A repeat within a range may use mode=uniform ONLY when the
  user explicitly requests uniform spacing. within is a fit constraint, never a
  substitute for a missing event. For ONE sound at a random point in that range,
  use {"kind":"single","within":{"start":ANCHOR,"end":ANCHOR},
  "selection":"random"}, without anchor or alignment. selection is a sibling
  of within, not a range endpoint. Code
  records one random choice and fits the complete material; never turn random
  placement into continuous coverage or invent a timestamp.
  This selection supports overlay and insert. Keep the chosen operation;
  the complete sound must fit the stated within range in either case.
  Add it alongside an anchor only if independently
  required by the instruction. Do not duplicate an anchor as a range boundary.

ANCHOR is exactly one of {"seconds":number}, {"media_fraction":number in [0,1]},
{"media_edge":"start|end"}, or {"reference_text":"existing event phrase",
"reference_source":"audio|visual|either", "reference_kind":"action|audible_event|utterance",
"occurrence":"first|last|each or positive integer or null",
"boundary_role":"onset|action_end|sound_end|utterance_end"}.
Keep repeated-event identity separate from its occurrence selector. For "after
the third bell", use reference_text="bell", occurrence=3; do not also encode
"third" in reference_text. Preserve identity qualifiers such as WHICH speaker,
instrument or quoted words: "second speaker's cough", occurrence=2 means that
speaker's second cough, not the second cough by any speaker.
Any anchor may include offset_seconds, a SIGNED offset, default zero. Never output
observed timestamps or event IDs. Code computes literal seconds, recording fractions
and actual media edges; perception is needed ONLY for existing event references.
The supplied analysis comes from the completed listening dialogue. Preserve the
requested event reference even if it remains unlocated. Never substitute a file
fraction, whole-file range or EOF for an unlocated event, conversation transition,
or beat. Between speakers refers to their utterance transition, even when no
timestamps are available. reference_text must name the EXISTING utterance to locate,
such as "first speaker's utterance before the second speaker begins", not the
relation "between the two speakers" as if it were an audible event or spoken quote.
For an added discrete sound BETWEEN consecutive existing events, default to
overlay within BOTH the preceding event's completion and the next event's onset.
Keep the original timeline unchanged. For speech use utterance_end and onset.
If the dialogue supplies placement_windows, set listening_window_index on the
affected edit; code preserves that pair even if a later plan omits one anchor.
Unknown or insufficient pause length is not permission to move the later event.
Use an adjustable complete generated sound only if it fits; otherwise retain the
constraint conflict. Explicitly creating new timeline space may use insert, but
ordinary 'add/insert a sound between' does not by itself authorize retiming.
State the overlay default in sources.timeline_op=system_default. Do not invent a midpoint.
When an audio anchor matches an edit_listening timing requirement, include its
question_id and copy its subject into reference_text. Keep its occurrence and
boundary_role. That answer is already selected evidence; do not rediscover it
from a broad inventory that may lack the requested semantic boundary.
A file midpoint is media_fraction=.5. Distinguish starting at the midpoint from
centering the material on it: alignment=start versus center. An ending REGION is
not necessarily actual EOF. For a within-only placement, alignment=start/end/center
places the material at the corresponding edge or center of that range; code does
the arithmetic. Resolve the complete instruction; record a vague choice in reason
and sources. No event reference is needed for an ordinary new
sound with no timing: single at media start is the system default. Untimed continuous
background ambience defaults to a cover from media start to media end.
Untimed musical accompaniment or harmony also defaults to that full-file cover.
The performance being accompanied describes musical context, not a requirement
to locate first/last spoken words. Require event boundaries only when the user
actually specifies an event-relative time or a restricted performance interval.

After a completed action uses action_end, including while its resonance continues.
sound_end includes ALL audible decay; utterance_end ends linguistic content;
onset begins the event. Choose meaning from the whole instruction, not sound-name
keywords. Keep first/last/nth/each and explicit offsets. Before uses alignment=end
for overlay, with NO duration subtracted by you. For insertion before an event,
use that event's insertion coordinate and alignment=start: the original moves.
Each expands the chosen phase independently; do not also multiply repeat.count
by the number of reference events. during requires the chosen start/end phases.

Generation for a discrete sound is {"target_seconds":number or null,
"source":"user_explicit|system_default|planner_choice", "adjustable":boolean,
"min_seconds":number if adjustable}. It describes ONE event. If the user fixes
length, use user_explicit and adjustable=false. If you choose a natural duration,
mark planner_choice and adjustable=true ONLY if a shorter complete sound remains
faithful; specify its minimum. null means the generator determines material length
(speech); it is NOT a missing temporal reference. For non-speech Stable Audio or
MMAudio, always choose a numeric generation target: those generators require a
duration budget even though the actual material length is not yet known. If that
budget is omitted, the entry adapter supplies a fixed 1-second system default. Code checks space before
and after generation. Do not compute arithmetic with observed times. Continuous
coverage gets its generation target from its range; never shorten that coverage.

Overlay keeps original coordinates unchanged. Extend is allowed only when the
instruction permits it; otherwise use error. For append/prepend, omit anchor and alignment: code supplies the media end/start
insertion coordinate. Only overlay has extent alignment. Insert supplies one
anchor and omits alignment. Never combine append with alignment=end. Insert/prepend/append move later original samples and preserve
explicit sequence gaps. All coordinates are ORIGINAL timeline coordinates.
Do not silently switch overlay into insertion or extension to obtain feasibility.
Decide permission from the original instruction: adding AFTER the recording or
its final completed event can require extension unless its duration is explicitly
fixed. Express that permission in this intent's overflow_policy and sources up
front; the compiler will never turn error into extend after discovering EOF.
For an addition "at the end" without a request to extend the recording, the
conservative default is overlay at media_edge=end with alignment=end; at the
beginning defaults to overlay at media_edge=start with alignment=start. State
that default in reason; explicitly requested append/prepend/insert must keep that operation.

If shared_replacement_windows is present, do NOT output placement coordinates.
Return one edit per window in order with shared_window_index, kind=single/cover/repeat
and (only for repeat) repeat. Its window is supplied by code and is immutable;
use overlay/error. Generation describes only the replacement asset. Code supplies
available_seconds for each window: the COMPLETE requested replacement, including
all repeats and their gaps, must fit it. Choose a faithful asset target/minimum
within that budget; do not invent a larger minimum and then demand a wider scope.
If no complete faithful sound fits, preserve the conflict for review. Shortening
required coverage or cropping away an asset's ending is never a substitute.
Describe one NEW event per single window in audio_prompt. The generator does not
need the old source name, ordinal, replacement instructions or repetition count;
those belong to the plan. For example, replacing each chime with a clock tick
uses audio_reference=null and a prompt for one clock tick. Matching the number
and times is already guaranteed by one placement per supplied window. In
contrast, changing a flute performance to clarinet while retaining its notes
uses structure, because the existing musical content must condition generation.

Choose generator="speech" ONLY for spoken language with explicit non-empty
speech_text (and optional voice_prompt) on that edit. Wordless humming, sung
vocalise, instrumental or environmental sounds use generator="audio"; never send
them to a text-to-speech API or invent lyrics. The legacy is_speech field is not
needed: code derives its compatibility view from each edit's generator.
For a requested transformation preserving melody/rhythm, set audio_reference=
"structure". Replacing one musical instrument with another normally keeps the
existing notes, rhythm and phrasing: use "structure" and describe the NEW solo
instrument playing the original melody. Only choose a new independent composition
when the user asks for one. The replacement reference contains the removed OLD
source; it must not be the silent/cleaned residual. For accompaniment or harmony responding to an existing track, set
it to "context". Otherwise explicitly set audio_reference=null. Code supplies actual source
samples to the existing audio generator, without changing placement. Reference
conditioning is a generation input, not a guarantee of musical quality. Keep
requested pitch/melody in the sound description even for a percussive timbre;
do not add contradictory "no pitch" restrictions when melody is required.
The generator creates only the NEW sound, never the original scene. Different
edits may have different sounds and lengths. Keep repetition out of audio_prompt.
composer is an optional JSON OBJECT of acoustic settings, for example
{"mix_level":1.0,"reverb_type":"none"}, or null. It is not an author/agent/model
name or prose instructions. Keep composer.mix_level=1.0; relative_level_db is the
one semantic loudness control.
For ADD, the default is +6 dB relative to the original audio in the placement.
Return relative_level_db=null when the user gives no loudness instruction; code
applies +6 dB. Background, BGM, accompaniment, guitar and harmony all use this
same default. layer_role="background" describes purpose, never reduced volume.
Only an explicit loudness request overrides the default: use the requested
numeric level, or a negative relative_level_db for explicitly quiet/low-volume
sound (use -12 dB if quiet is requested without a number). Calm/gentle musical
style alone is not a low-volume request. Do not invent faint, subtle or quiet
wording in audio_prompt merely because the user requested background music.
For shared_replacement_windows, retain the replacement balance instead of
applying the ADD +6 dB default.
Do not request original-track ducking. Do not use a missing instance of the NEW
sound as a timing reference. Return no extra legacy fields."""


class DurationChoiceConflict(ValueError):
    def __init__(self, index, available):
        self.index, self.available = index, available
        super().__init__(
            f"Generation target does not fit edit {index}: available={available:g}s; anchor/range/policy unchanged"
        )


def _validate_parameter_sources(sources):
    """Granular provenance keeps its structure; validate the source labels at leaves."""
    if not isinstance(sources, dict):
        raise ValueError("Temporal parameter sources must be an object")
    for key, value in sources.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Temporal parameter source needs a field name")
        if isinstance(value, dict):
            if not value:
                raise ValueError("Empty temporal parameter source")
            _validate_parameter_sources(value)
        elif not isinstance(value, str) or value not in SOURCES:
            raise ValueError("Unknown temporal parameter source")


def validate_intent(intent):
    if (
        intent.get("version") != 2
        or not isinstance(intent.get("edits"), list)
        or not intent["edits"]
    ):
        raise ValueError("Add intent v2 needs a nonempty edits list")
    for e in intent["edits"]:
        if not isinstance(e, dict) or not isinstance(e.get("placement"), dict):
            raise ValueError("Each edit needs a placement object")
        p = e["placement"]
        if e.get("generator", "audio") not in ("audio", "speech"):
            raise ValueError("generator must be audio or speech")
        if e.get("generator") == "speech" and not (
            isinstance(e.get("speech_text"), str) and e["speech_text"].strip()
        ):
            raise ValueError("Spoken generation requires non-empty speech_text")
        if e.get("audio_reference") not in (None, "structure", "context"):
            raise ValueError("Unknown audio reference mode")
        if e.get("audio_reference") and e.get("generator") == "speech":
            raise ValueError(
                "Speech generator cannot consume audio reference conditioning"
            )
        for region_key in ("within", "range"):
            region = p.get(region_key)
            if region is not None and (
                not isinstance(region, dict) or set(region) != {"start", "end"}
            ):
                raise ValueError("Placement range must contain only start and end; do not discard extra constraints")
        selection = p.get("selection")
        if selection is not None:
            if (
                p.get("kind") != "single"
                or "within" not in p
                or any(k in p for k in ("anchor", "anchors", "alignment"))
                or e.get("timeline_op") not in ("overlay", "insert")
                or not isinstance(selection, dict)
                or selection.get("mode") != "random"
                or not 0 <= number(selection.get("unit"), "recorded random choice") < 1
            ):
                raise ValueError(
                    "Random selection needs one overlay or insert within a range and one recorded choice"
                )
        rule = p.get("repeat", {})
        if not isinstance(rule, dict):
            raise ValueError("repeat must be an object")
        if p.get("kind") == "repeat" and "anchors" not in p:
            rules = [
                k
                for k in ("spacing_seconds", "spacing_from", "offsets_seconds", "mode")
                if k in rule
            ]
            if len(rules) != 1:
                raise ValueError("Repetition needs exactly one spacing rule")
            if (
                "spacing_seconds" in rule
                and number(rule["spacing_seconds"], "spacing") <= 0
            ):
                raise ValueError("Repetition spacing must be positive")
            if "spacing_from" in rule and (
                not isinstance(rule["spacing_from"], dict)
                or rule["spacing_from"].get("occurrence") == "each"
            ):
                raise ValueError("Rhythm spacing needs one selected rhythmic sequence")
        g = e.get("generation")
        if g is not None and not isinstance(g, dict):
            raise ValueError("generation must be an object or null")
        kind = p.get("kind")
        if kind not in ("single", "cover", "repeat"):
            raise ValueError("placement.kind must be single, cover or repeat")
        if any(
            k in e
            for k in (
                "temporal",
                "window",
                "placement_anchors",
                "start_time",
                "end_time",
            )
        ):
            raise ValueError("Redundant timing fields are not part of add intent v2")
        if e.get("timeline_op") not in ("overlay", "append", "prepend", "insert"):
            raise ValueError("Explicit timeline operation required")
        if e.get("overflow_policy") not in ("error", "extend"):
            raise ValueError("Explicit overflow policy required")
        if not e.get("reason"):
            raise ValueError("Add intent needs a reason")
        if kind == "cover":
            if any(
                k in p for k in ("anchor", "anchors", "repeat", "within", "alignment")
            ):
                raise ValueError(
                    "Coverage needs only its range, not additional placements"
                )
            if not isinstance(p.get("range"), dict) or p["range"].get("end") is None:
                raise ValueError(
                    "Coverage requires an explicit end anchor; until EOF is media_edge=end"
                )
        else:
            if p.get("alignment", "start") not in ("start", "end", "center"):
                raise ValueError("Unknown alignment")
            if not any(k in p for k in ("anchor", "anchors", "within")):
                raise ValueError(
                    "Discrete placement requires an anchor or a constrained range"
                )
            if "anchor" in p and "anchors" in p:
                raise ValueError("Choose one repetition representation")
            if kind == "single" and any(k in p for k in ("repeat", "anchors")):
                raise ValueError("Single placement must not also express repetition")
            if "anchors" in p and "repeat" in p:
                raise ValueError(
                    "The explicit anchor list already supplies occurrence count"
                )
            if (
                kind == "repeat"
                and "anchors" not in p
                and not p.get("repeat", {}).get("count")
            ):
                raise ValueError(
                    "Repeated placement requires its count or an explicit anchor list"
                )
            if "anchors" in p and not p["anchors"]:
                raise ValueError("An explicit anchor list must not be empty")
            if g is None:
                raise ValueError(
                    "Discrete placement needs generation settings from the entry adapter"
                )
            if g.get("source") not in SOURCES:
                raise ValueError("Generation target needs its source")
            target = g.get("target_seconds")
            if target is not None and number(target, "generation target") <= 0:
                raise ValueError("Generation target must be positive")
            if g.get("adjustable"):
                if (
                    g["source"] != "planner_choice"
                    or number(g.get("min_seconds"), "minimum complete event duration")
                    <= 0
                ):
                    raise ValueError(
                        "Only explicitly adjustable planner duration choices may be reduced"
                    )
                if target is not None and g["min_seconds"] > target:
                    raise ValueError("Minimum event length exceeds generation target")
        _validate_parameter_sources(e.get("sources", {}))
    return intent


def references(intent):
    result = {}

    def visit(value, purpose=None):
        if isinstance(value, dict):
            if "reference_text" in value and not any(
                k in value
                for k in ("seconds", "media_edge", "media_fraction", "event_id")
            ):
                request = deepcopy(value)
                if purpose:
                    request["measurement"] = purpose
                result.setdefault(reference_key(value), []).append(request)
            else:
                for k, v in value.items():
                    visit(v, "beat_onsets" if k == "spacing_from" else None)
        elif isinstance(value, list):
            for v in value:
                visit(v)

    for e in intent["edits"]:
        visit(e["placement"])
    return result


def bound_requests(intent, bindings):
    def visit(v, purpose=None):
        if isinstance(v, list):
            return [visit(x, purpose) for x in v]
        if not isinstance(v, dict):
            return v
        if "reference_text" in v and not any(
            k in v for k in ("seconds", "media_fraction", "media_edge", "event_id")
        ):
            request = dict(v, event_id=deepcopy(bindings.get(reference_key(v), [])))
            if purpose:
                request["measurement"] = purpose
            return request
        return {k: visit(x, "beat_onsets" if k == "spacing_from" else None) for k, x in v.items()}

    return visit([e["placement"] for e in intent["edits"]])


def bind_references(llm, intent, audio_analysis, visual_analysis=None, *, audit=None):
    refs = references(intent)
    if not refs:
        return {}
    grounded = question_bindings(refs, audio_analysis)
    refs = {key: value for key, value in refs.items() if key not in grounded}
    if not refs:
        if audit is not None:
            audit.append(dict(status='question_bound', bindings=deepcopy(grounded)))
        return grounded
    events = [
        e
        for a, k in ((audio_analysis, "events"), (visual_analysis, "visual_events"))
        for e in (a or {}).get(k, [])
        if has_measured_timing(e) and not e.get("review_context_only")
    ]
    if not events:
        return dict(grounded, **{key: [] for key in refs})
    available = {e["event_id"] for e in events if e.get("event_id")}
    prompt = """Group observed events by the requested IDENTITY, not by occurrence.
Return JSON {"bindings":{"exact reference_text":["event_id",...]}} with every
reference key unchanged. Each list must contain ALL observed instances of that
identity. Code alone applies first/last/nth/each AFTER this call.

An occurrence selector is NOT an identity filter, even if reference_text repeats
it. Example: reference_text="third bell", occurrence=3, with bell events b1,b2,b3,b4
requires ["b1","b2","b3","b4"] when these are the available identity observations.
A question-local observation explicitly marked occurrence_index=3 can also
represent the selected third event: code preserves that index, not list position.
Prefer records measuring the requested boundary to duplicate inventory records
without it; do not treat duplicate descriptions as additional occurrences.
If only two bells are observed,
return both; code will report the missing third. Do not invent an event or choose
a nearby different sound to make the edit feasible.

Preserve identity restrictions: speaker, source, quoted content, and reference_source.
For the second cough of a particular speaker, include all coughs of THAT speaker,
not other speakers' coughs. Distinguish the speaker's identity from the cough's
occurrence. Ignore review_context_only observations. No matching identity means [].
Do not select a phase or time here. Do not return offsets, placement, operation,
duration or a replacement instruction. Do not change the supplied selector.
IDs must be present in the observations. The NEW sound is never a reference."""
    messages = [
        dict(role="system", content=prompt),
        dict(role="user", content=json.dumps(dict(references=refs, events=events,
            listening_requirements=(audio_analysis or {}).get('edit_listening', {}).get('requirements', [])))),
    ]
    call_audit = dict(model=llm.model, messages=deepcopy(messages), status="started")
    if audit is not None:
        audit.append(call_audit)
    response = llm.client.chat.completions.create(
        model=llm.model,
        messages=messages,
    )
    from audio_edit.analyzers.qwen_audio_analyzer import extract_json_object

    raw_response = response.choices[0].message.content or ""
    call_audit.update(raw_response=raw_response, status="response_received")
    data = extract_json_object(raw_response)
    if (
        set(data) != {"bindings"}
        or not isinstance(data["bindings"], dict)
        or set(data["bindings"]) != set(refs)
    ):
        raise ValueError(
            "Reference binding must supply exactly the immutable reference identities"
        )
    for ids in data["bindings"].values():
        if (
            not isinstance(ids, list)
            or any(not isinstance(i, str) or i not in available for i in ids)
            or len(ids) != len(set(ids))
        ):
            raise ValueError(
                "Reference binding returned unknown or duplicate event IDs"
            )
    result = dict(grounded, **data['bindings'])
    call_audit.update(bindings=deepcopy(result), status="validated")
    return result


def _expand_each(edit, events):
    selectors = []

    def collect(v):
        if isinstance(v, dict):
            if v.get("occurrence") == "each" and "event_id" in v:
                selectors.append(v)
            else:
                for x in v.values():
                    collect(x)
        elif isinstance(v, list):
            for x in v:
                collect(x)

    collect(edit["placement"])
    if not selectors:
        return [edit]
    ids = selectors[0]["event_id"]
    ids = ids if isinstance(ids, list) else [ids]
    if not ids:
        raise BoundaryEvidenceError("each reference has no observed occurrences")
    if any(
        set(v["event_id"] if isinstance(v["event_id"], list) else [v["event_id"]])
        != set(ids)
        for v in selectors
    ):
        raise BoundaryEvidenceError(
            "Paired each references need the same observed identity set"
        )
    if any(i not in events for i in ids):
        raise BoundaryEvidenceError("Unknown each event ID")

    def select(v, key):
        if isinstance(v, list):
            return [select(x, key) for x in v]
        if not isinstance(v, dict):
            return v
        if v.get("occurrence") == "each" and "event_id" in v:
            return dict(v, event_id=key, occurrence=None)
        return {k: select(x, key) for k, x in v.items()}

    return [select(edit, i) for i in sorted(ids, key=lambda k: events[k]["start_time"])]


def compile_add_intent(
    intent,
    *,
    duration,
    audio_analysis=None,
    visual_analysis=None,
    bindings=None,
    asset_durations=None,
    replacement_extent_limits=None,
):
    """Same entry for target feasibility and final actual-material realization.

    asset_durations is keyed by the deterministic placement key. No inference,
    default event substitution, duration mutation or semantic replanning occurs.
    """
    validate_intent(intent)
    validate_listening_windows(intent, audio_analysis)
    analyses = [deepcopy(audio_analysis), deepcopy(visual_analysis)]
    for a, source in zip(analyses, ("audio", "visual")):
        assign_event_ids(a, source, duration)
    events = {
        e["event_id"]: e
        for a, k in zip(analyses, ("events", "visual_events"))
        for e in (a or {}).get(k, [])
    }
    bound = bound_requests(intent, bindings or {})
    output, windows = [], []
    final = asset_durations is not None
    for edit_index, original in enumerate(intent["edits"]):
        linked = dict(deepcopy(original), placement=bound[edit_index])
        for group_index, e in enumerate(_expand_each(linked, events)):
            p = e["placement"]
            kind = p["kind"]
            op = e["timeline_op"]
            overflow = e["overflow_policy"]
            evidence = []
            statuses = []

            def time(a):
                value, status, ev = resolve_anchor(
                    a, events, duration, best_effort=False
                )
                evidence.append(ev)
                statuses.append(status)
                return value

            region = p.get("range") if kind == "cover" else p.get("within")
            lo, hi = (
                (time(region["start"]), time(region["end"]))
                if region
                else (0.0, duration)
            )
            if region and (lo < 0 or hi <= lo):
                raise ValueError(
                    f"Constraint conflict: invalid/empty associated range {lo:g}..{hi:g}; no endpoint repair or clipping"
                )
            if kind == "cover":
                origins = [lo]
                count = 1
                repeat = {}
                alignment = "start"
                target = hi - lo
                g = e.get("generation") or {}
                if (
                    g.get("source") == "user_explicit"
                    and g.get("target_seconds") is not None
                    and abs(g["target_seconds"] - target) > 1e-8
                ):
                    raise ValueError(
                        "Explicit generation duration conflicts with required continuous coverage"
                    )
            else:
                alignment = p.get("alignment", "start")
                repeat = deepcopy(p.get("repeat", {}))
                if "spacing_from" in repeat:
                    step, rhythm_evidence = _rhythm_spacing(
                        repeat.pop("spacing_from"), events, duration
                    )
                    repeat["spacing_seconds"] = step
                    evidence.append(rhythm_evidence)
                    statuses.append("approximate")
                    p = dict(
                        p, repeat=repeat
                    )
                origins = (
                    [time(a) for a in p["anchors"]]
                    if "anchors" in p
                    else [
                        time(p["anchor"])
                        if "anchor" in p
                        else {"start": lo, "end": hi, "center": (lo + hi) / 2}[
                            alignment
                        ]
                    ]
                )
                count = len(origins) if "anchors" in p else repeat.get("count", 1)
                target = e["generation"].get("target_seconds")
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or not 1 <= count <= 100
            ):
                raise ValueError("Repetition count must be an integer in [1,100]")
            if op != "overlay" and alignment != "start":
                raise ValueError(
                    "Insert/prepend/append use an insertion coordinate, not extent alignment"
                )
            for a in origins:
                if a < 0 or (op in ("insert", "prepend") and a > duration):
                    raise ValueError("Invalid insertion/placement anchor")
                if op == "prepend" and abs(a) > 1e-8:
                    raise ValueError("Prepend must start at zero")
                if op == "append" and abs(a - duration) > 1e-8:
                    raise ValueError("Append must reference original media end")
            keys = [f"{edit_index}:{group_index}:{j}" for j in range(count)]
            actual = (
                [number(asset_durations[k], "actual asset duration") for k in keys]
                if final
                else [None] * count
            )
            if final and any(d <= 0 for d in actual):
                raise ValueError("Actual asset length must be positive")
            fixed = (
                kind == "cover"
                or e.get("generation", {}).get("source") == "user_explicit"
            )
            sizes = [target if fixed or not final else d for d in actual]
            if final and kind == "cover" and any(d + 1e-4 < target for d in actual):
                raise ValueError(
                    "Generated continuous sound is shorter than required coverage"
                )


            if any(d is None for d in sizes):
                if (
                    op == "overlay"
                    and any(
                        (
                            a >= hi
                            if alignment == "start"
                            else a <= lo
                            if alignment == "end"
                            else not lo < a < hi
                        )
                        for a in origins
                    )
                    and (region or overflow == "error")
                ):
                    raise ValueError(
                        "No positive space for pending material at the fixed anchor"
                    )
                for j, key in enumerate(keys):
                    output.append(
                        dict(
                            key=key,
                            edit_index=edit_index,
                            group_index=group_index,
                            generation_target_seconds=target,
                            start_time=origins[min(j, len(origins) - 1)],
                            end_time=None,
                            timeline_op=op,
                            overflow_policy=overflow,
                            coverage="single",
                            compiled_phase="pending_asset",
                            contract_version=2,
                            audio_prompt=e.get("audio_prompt"),
                            **_generation_fields(e),
                            status=_status(statuses, e),
                            evidence=deepcopy(evidence),
                        )
                    )
                continue
            if p.get("selection"):
                room = hi - lo - sizes[0]
                if room < -1e-8:
                    if not final:
                        raise DurationChoiceConflict(edit_index, hi - lo)
                    raise ValueError(
                        "Complete material does not fit the requested random placement range"
                    )
                origins = [lo + p["selection"]["unit"] * max(0.0, room)]
                evidence.append(
                    dict(
                        random_choice=p["selection"]["unit"],
                        feasible_start=[lo, hi - sizes[0]],
                        reason="One entry choice, reused with measured material length",
                    )
                )
            if "anchors" in p:
                starts = [
                    a - ({"start": 0.0, "end": d, "center": d / 2}[alignment])
                    for a, d in zip(origins, sizes)
                ]
            else:
                if kind != "repeat" or count == 1:
                    offsets = [0.0]
                elif "offsets_seconds" in repeat:
                    offsets = [
                        number(x, "repetition offset")
                        for x in repeat["offsets_seconds"]
                    ]
                elif "spacing_seconds" in repeat:
                    step = number(repeat["spacing_seconds"], "onset spacing")
                    offsets = [j * step for j in range(count)]
                elif repeat.get("mode") == "sequential":
                    offsets = [sum(sizes[:j]) for j in range(count)]
                elif (
                    repeat.get("mode") == "uniform"
                    and region
                    and repeat.get("explicitly_requested")
                ):
                    step = (hi - lo - max(sizes)) / (count - 1)
                    offsets = [j * step for j in range(count)]
                else:
                    raise ValueError(
                        "Repeated placement needs explicit spacing/offsets or a declared sequence rule"
                    )
                if len(offsets) != count or any(x < 0 for x in offsets):
                    raise ValueError("Invalid repetition offsets")
                if op != "overlay" and any(
                    offsets[j] + sizes[j] > offsets[j + 1] + 1e-8
                    for j in range(count - 1)
                ):
                    available = min(
                        offsets[j + 1] - offsets[j] for j in range(count - 1)
                    )
                    if not final and available > 0:
                        raise DurationChoiceConflict(edit_index, available)
                    raise ValueError(
                        "Inserted repetitions overlap at the required onset spacing"
                    )
                extent = max(o + d for o, d in zip(offsets, sizes))
                origin = (
                    origins[0]
                    - {"start": 0.0, "end": extent, "center": extent / 2}[alignment]
                )
                starts = [origin + o for o in offsets]
            ends = [a + d for a, d in zip(starts, sizes)]
            extent_limit = (replacement_extent_limits or {}).get(edit_index)
            extended_extent = None
            if extent_limit is not None:
                from audio_edit.plans.replacement_extent import validate_extent_limit
                fit_end = validate_extent_limit(extent_limit, e, lo, hi, duration)
                if final and max(ends) > hi + 1e-8:


                    hi = fit_end
                    extended_extent = deepcopy(extent_limit)
            constrained = region is not None or (
                op == "overlay" and overflow == "error"
            )
            if op == "overlay" and (
                min(starts) < lo - 1e-8 or (constrained and max(ends) > hi + 1e-8)
            ):
                available = _available(origins, lo, hi, alignment, p, count)
                if (
                    not final
                    and kind != "cover"
                    and target is not None
                    and available > 0
                ):
                    raise DurationChoiceConflict(edit_index, available)
                raise ValueError(
                    f"Constraint conflict: complete material {min(starts):g}..{max(ends):g} does not fit {lo:g}..{hi:g}; anchor and policy unchanged"
                )
            status = _status(statuses, e)
            previous_end = 0.0
            for j, (key, a, b, d) in enumerate(zip(keys, starts, ends, sizes)):
                insertion_origin = origins[0] if "anchors" not in p else origins[j]
                anchor = insertion_origin if op != "overlay" else a
                numeric = dict(
                    window=dict(start={"seconds": anchor}, end={"seconds": anchor + d}),
                    timeline_op=op,
                    overflow_policy=overflow,
                    coverage="continuous" if kind == "cover" else "single",
                    event_duration=d,
                    count=1,
                    distribution="explicit",
                    offsets_seconds=[0.0],
                    placement_anchors=None,
                    reason=e["reason"],
                    audio_prompt=e.get("audio_prompt"),
                    layer_role=e.get("layer_role"),
                    relative_level_db=e.get("relative_level_db"),
                )
                _, one = compile_edits([numeric], duration=duration)
                q = one[0]
                q.update(
                    key=key,
                    edit_index=edit_index,
                    group_index=group_index,
                    window_index=edit_index,
                    occurrence_index=j + 1,
                    generation_target_seconds=target,
                    actual_asset_seconds=actual[j],
                    event_duration=d,
                    render_alignment=alignment,
                    status=status,
                    evidence=deepcopy(evidence),
                    compiled_phase="final" if final else "pending_asset",
                    contract_version=2,
                    sources=deepcopy(e.get("sources", {})),
                    **_generation_fields(e),
                )
                if op != "overlay":
                    offset = a - insertion_origin
                    gap = offset - previous_end if "anchors" not in p else 0.0
                    if gap < -1e-8:
                        raise ValueError(
                            "Inserted sequence overlaps or is out of order"
                        )
                    q["insert_gap_before_seconds"] = max(0.0, gap)
                    previous_end = offset + d
                if region:
                    q.update(
                        window_start=lo, window_end=hi, allowed_start=lo, allowed_end=hi
                    )
                elif (op == "overlay" and overflow == "error" and not fixed
                      and e.get("shared_window") is None):


                    q.update(allowed_start=lo, allowed_end=hi)
                if e.get("shared_window") is not None:
                    sw = e["shared_window"]
                    if (
                        op != "overlay"
                        or overflow != "error"
                        or a < sw["start_time"] - 1e-8
                        or b > (hi if extended_extent else sw["end_time"]) + 1e-8
                    ):
                        raise ValueError(
                            "Replacement changed authoritative removal window"
                        )
                    q["shared_window"] = deepcopy(sw)
                    if extended_extent:
                        q['replacement_extent'] = deepcopy(extended_extent)
                output.append(q)
            windows.append(
                dict(
                    edit_index=edit_index,
                    group_index=group_index,
                    start_time=min(starts),
                    end_time=max(ends),
                    status=status,
                    evidence=deepcopy(evidence),
                    reason=e["reason"],
                )
            )


    inserts = [p["start_time"] for p in output if p["timeline_op"] != "overlay"]
    if any(
        p["start_time"] < a < p["end_time"]
        for p in output
        if p["timeline_op"] == "overlay" and p["end_time"] is not None
        for a in inserts
    ):
        raise ValueError(
            "Overlay crosses an insertion seam; split the intent explicitly"
        )
    return windows, output


def _generation_fields(edit):
    return {
        k: deepcopy(edit[k])
        for k in ("generator", "speech_text", "voice_prompt", "audio_reference")
        if k in edit
    }


def _rhythm_spacing(anchor, events, duration):
    event, reason = select_event(anchor, events, best_effort=False)
    rhythm = (event or {}).get("rhythm") or {}
    onsets = rhythm.get("onsets_seconds")
    if not isinstance(onsets, list) or len(onsets) < 3 or not rhythm.get("evidence"):
        raise BoundaryEvidenceError(
            "Rhythm reference needs at least three observed beat onsets and evidence; no guessed tempo"
        )
    try:
        onsets = [number(v, "observed beat onset") for v in onsets]
    except ValueError as exc:
        raise BoundaryEvidenceError(str(exc)) from exc
    gaps = [b - a for a, b in zip(onsets, onsets[1:])]
    if not 0 <= onsets[0] < onsets[-1] <= duration or min(gaps) <= 0:
        raise BoundaryEvidenceError(
            "Observed beat onsets are not ordered within the recording"
        )
    step = median(gaps)
    if any(abs(g - step) > 0.2 * step for g in gaps):
        raise BoundaryEvidenceError(
            "Observed beat intervals do not establish a stable repetition rate"
        )
    return step, dict(
        event_id=event["event_id"],
        rhythm=deepcopy(rhythm),
        spacing_seconds=step,
        reason=reason or "Median measured beat interval",
    )


def _status(statuses, edit):
    if "estimated" in statuses:
        return "estimated"
    if "approximate" in statuses or edit.get("default_reason"):
        return "approximate"
    return "resolved"


def _available(origins, lo, hi, alignment, p, count):
    if "anchors" in p:
        return min(
            a - lo
            if alignment == "end"
            else 2 * min(a - lo, hi - a)
            if alignment == "center"
            else hi - a
            for a in origins
        )
    a = origins[0]
    room = (
        a - lo
        if alignment == "end"
        else 2 * min(a - lo, hi - a)
        if alignment == "center"
        else hi - a
    )
    rule = p.get("repeat", {})
    if rule.get("mode") in ("sequential", "uniform"):
        return room / count
    offsets = rule.get("offsets_seconds")
    span = max(offsets) if offsets else (count - 1) * rule.get("spacing_seconds", 0.0)
    return room - span
