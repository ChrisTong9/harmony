"""Map natural-language references to observed event boundaries, then do arithmetic."""
from __future__ import annotations
import json
import math
from audio_edit.analyzers.qwen_audio_analyzer import extract_json_object
from audio_edit.plans.temporal import ResolvedPlacement, TemporalResolution
from audio_edit.plans.event_ids import assign_event_ids
from audio_edit.plans.temporal_boundaries import event_boundary


def regions_from_anchors(data, events, media_duration):
    if data.get('purpose') == 'search' or data.get('status') == 'unresolved':
        return TemporalResolution('unresolved', reason=data.get('reason') or 'Search region is not an observed edit target')
    used = []
    def time(anchor):
        if 'event_id' in anchor:
            index = anchor['event_id']
            if isinstance(index, str):
                matches = [e for e in events if e.get('event_id') == index]
                if len(matches) != 1:
                    raise ValueError('Grounding referenced an unknown event_id')
                event = matches[0]
            elif isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(events):
                event = events[index]
            else:
                raise ValueError('Grounding referenced an unknown event_id')
            if event.get('resolution_status') == 'unresolved':
                raise ValueError('Selected observation has invalid/unresolved boundaries')
            used.append(event)
            if anchor.get('boundary_role'):
                key = event.get('event_id', 'legacy_reference')
                value, status, evidence = event_boundary(dict(anchor, event_id=key), {key: dict(event, event_id=key)})
                used.append(dict(resolution_status=status, timing_evidence=evidence))
                result = float(value) + float(anchor.get('offset_seconds', 0))
                if not math.isfinite(result) or not 0 <= result <= media_duration:
                    raise ValueError('Grounded phase exceeds media; no clipping allowed')
                return result
            edge = anchor.get('edge')
            if edge not in ('start', 'end'):
                raise ValueError('Event anchor edge must be start or end')
            if event.get('partial_' + edge) or event.get('review_context_only'):
                raise ValueError('Selected boundary is truncated or superseded by focused review')
            value = event.get(edge + '_time')
            if value is None:
                raise ValueError('Grounding selected an event with no measured boundary')
        else:
            edge = anchor.get('media_edge')
            if edge not in ('start', 'end', 'middle'):
                raise ValueError('Grounding needs an event or media anchor')
            value = {'start': 0, 'end': media_duration, 'middle': media_duration / 2}[edge]
        result = float(value) + float(anchor.get('offset_seconds', 0))
        if not math.isfinite(result):
            raise ValueError('Grounding returned a non-finite time')
        if result < 0 or result > media_duration:
            raise ValueError('Grounded region exceeds media; no boundary clipping allowed')
        return result
    placements = []
    for region in data.get('regions', []):
        start, end = time(region['start']), time(region['end'])
        if end <= start:
            raise ValueError('Grounded region has no positive duration')
        approximate = bool(data.get('approximate', False) or data.get('status') == 'approximate') or any(e.get('resolution_status') == 'approximate' for e in used)
        placements.append(ResolvedPlacement(start, end, approximate=approximate,
            resolution_status='approximate' if approximate else 'resolved', evidence=data.get('reason') or 'Explicit observed boundary anchors'))
    return TemporalResolution(('approximate' if any(p.approximate for p in placements) else 'resolved') if placements else 'unresolved', placements,
                              data.get('reason'),
                              data.get('target_description'))


def ground_regions(llm, *, user_prompt, target_sound, media_duration, audio_analysis, operation, temporal_constraint=None):
    assign_event_ids(audio_analysis, 'audio', media_duration)
    events = (audio_analysis or {}).get('events', [])
    payload = {'instruction': user_prompt, 'target_sound': target_sound, 'operation': operation,
               'media_duration': media_duration,
               'events': events, 'temporal_constraint': temporal_constraint}
    system = '''Resolve editing regions from the observed audio events and instruction.
Understand semantic references before choosing timestamps. Pick event IDs by
meaning, preserving reference_text verbatim. Keep occurrence and relation distinct
from boundary_role (onset/action_end/sound_end/utterance_end). Action completion,
audible decay completion, utterance completion and file end are not interchangeable.
After is not universally event.end; choose a role from the complete instruction.
Optional boundary_role selects an observed phase, never a sound-keyword rule.
For remove/replace, select the EXISTING TARGET event's full start/end satisfying
the requested relation. If no target span is detected, return status=unresolved,
regions=[], and a reason. A proposed search window is NOT an executed edit region.
For "middle", prefer the matching target event nearest the media midpoint;
respect explicit target exclusions in the complete instruction.
For first/last/few seconds use MEDIA boundaries, not first/last sound occurrence.
Without a numeric amount, "few seconds" means min(3 seconds, media duration),
marked approximate. Preserve explicit numeric amounts from the instruction.
For a named target missing near the middle, return unresolved; never invent a
central-third target span. A vague timing default is allowed only when the user
actually asks for an approximate region, not to conceal a missing target.
Never fabricate an event ID or a measured boundary.
Prefer reviewed observations when they provide the required complete boundaries.
A partial fragment does not invalidate a retained full-recording event boundary.
Use that complete original observation when the fragment cannot remeasure the edge.
Events marked
review_context_only are context, not valid edit anchors. An event with partial_start
or partial_end reaches a crop edge: that edge is NOT a measured event boundary.
Never use such an edge to claim an event ended/started or a complete target was located.
For add/between, return the
gap from the earlier event's END to the later event's START. For during, return
the reference interval. Numeric offset_seconds are arithmetic offsets, not invented
observations. Use them only for explicit times or the approximate window policies above.
Return JSON: {"status":"resolved|approximate|unresolved", "purpose":"edit",
"regions":[{"start":{"event_id":"observed ID","edge":"end"},
"end":{"event_id":"observed ID","edge":"start"}}],"approximate":false,"reason":"..."}.
Each anchor is EITHER {"event_id":"opaque observed ID","edge":"start|end","offset_seconds":0}
OR {"media_edge":"start|middle|end","offset_seconds":number}. No literal timestamps.
Multiple regions are allowed for each/all occurrences. Return empty regions with
a reason when the actual target cannot be located. Search hints are not edits.
Also return target_description: a concise acoustic identity of the OLD source
when the observations identify it (especially for non-X/exclusion descriptions).
Use null when no more specific identity was observed. Never use the NEW source
as the removal target.'''
    if operation == 'extract':
        system += """
For extract, this stage returns the POSITIVE stem of target_sound: the existing
source to KEEP, not a source to remove. Preserve that source's identity. Resolve
the requested OUTPUT scope for that retained source. If the instruction requests
only that source throughout the recording, use media start and media end; do not
restrict extraction to the intervals of unwanted noises. An instruction to remove
or inpaint other sounds does not make those sounds the positive extraction target
and does not limit the retained output to their intervals. Preserve any explicit
time or occurrence restriction on the retained source. Return target_description
as null: source identity is already bound. This stage only resolves separation
scope and must not claim to have synthesized or inpainted missing content."""
    if operation == 'volume':
        system += """
For volume, target_sound is WHAT to separate and adjust, not the WHEN reference.
Source separation locates the named targets; they need not have their own observed
event boundaries. Preserve the supplied temporal_constraint. For during/when a
spoken phrase or other reference, use that reference's own start AND end. Do not
combine a phrase onset with a background source's end or a crop edge. Return
unresolved when the requested reference itself cannot be located."""
    response = llm.client.chat.completions.create(
        model=llm.model,
        messages=[{'role': 'system', 'content': system},
                  {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}])
    data = extract_json_object(response.choices[0].message.content or '')
    constraint = temporal_constraint or {}
    if (operation == 'volume' and constraint.get('kind') in ('event', 'relative')
            and constraint.get('relation') == 'during' and not constraint.get('end_reference_event')):
        for region in data.get('regions', []):
            start, end = region.get('start', {}), region.get('end', {})
            if (not start.get('event_id') or start.get('event_id') != end.get('event_id')
                    or start.get('edge') != 'start' or end.get('edge') != 'end'):
                raise ValueError('During-volume scope must use the same timing reference start and end, not a target-source boundary')
    result = regions_from_anchors(data, events, media_duration)


    return result


def ground_from_observations(llm, **kwargs):
    """Resolve against the completed dialogue without triggering another listen."""
    try:
        return ground_regions(llm, **kwargs)
    except (ValueError, TypeError, KeyError) as exc:
        return TemporalResolution('unresolved', reason=str(exc))


def ground_with_review(llm, analyzer, *, input_audio, temporal_constraint=None, perception_prompt=None, **kwargs):
    """One semantic resolution, then one actual fragment review if still missing."""
    from audio_edit.analyzers.focused_audio_review import review_audio
    try:
        resolution = ground_regions(llm, temporal_constraint=temporal_constraint, **kwargs)
    except (ValueError, TypeError, KeyError) as exc:
        resolution = TemporalResolution('unresolved', reason=str(exc))
    if resolution.status != 'unresolved':
        return resolution, kwargs['audio_analysis'], {'status': 'not_needed', 'reason': resolution.reason}
    audit = {'initial_status': resolution.status, 'initial_reason': resolution.reason}
    analysis = kwargs['audio_analysis']
    try:
        analysis, review = review_audio(llm, analyzer, input_audio=input_audio, perception_prompt=perception_prompt, **kwargs)
        audit.update(review)
        if review['status'] == 'reviewed':
            resolution = ground_regions(llm, temporal_constraint=temporal_constraint, **dict(kwargs, audio_analysis=analysis))
    except Exception as exc:


        resolution = TemporalResolution('unresolved', reason=f'Focused grounding failed: {type(exc).__name__}: {exc}')
        audit['error'] = resolution.reason
    audit.update(final_status=resolution.status, final_reason=resolution.reason)
    return resolution, analysis, audit
