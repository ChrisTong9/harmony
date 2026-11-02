"""Keep source deletion bounds separate from an independent event's decay.

Only an event-selected removal supplies this permission. Numeric/paired windows,
explicit new-event durations, structured audio and ordinary ADD retain their
existing limits. A decay may reach the next source onset or EOF, never cross it.
"""
import math


POLICY = 'complete_event_after_source_onset'


def replacement_extent_limits(plan, removal_plan, duration):
    if plan.is_on_screen or not isinstance(removal_plan, dict):
        return {}
    temporal = removal_plan.get('temporal') or {}
    retained_selection = False
    if not temporal:


        temporal = removal_plan.get('temporal_input') or {}
        retained_selection = True
    if temporal.get('kind') != 'event' or temporal.get('relation') is not None:
        return {}
    if any(temporal.get(k) is not None for k in (
        'start_time', 'end_time', 'end_reference_event', 'window_seconds', 'boundary',
    )) or temporal.get('offset_seconds', 0) not in (None, 0):
        return {}
    spans = removal_plan.get('time_span')
    edits = plan.intent.get('edits', [])
    if not isinstance(spans, list) or len(spans) != len(edits):
        return {}
    analysis = getattr(getattr(plan, 'media_analysis', None), 'audio_analysis', None) or {}
    listening = analysis.get('edit_listening') or {}
    selected_ids = temporal.get('reference_event')
    selected_ids = selected_ids if isinstance(selected_ids, list) else [selected_ids]
    original_sha = (removal_plan.get('removal_evidence') or {}).get('audio', {}).get('sha256')
    retained_events = {}
    if retained_selection:
        if (listening.get('instruction_source') != 'original_user_instruction'
                or listening.get('selection', {}).get('scope') != 'event_condition'
                or listening.get('placement_windows')
                or not original_sha or analysis.get('audio_evidence', {}).get('sha256') != original_sha):
            return {}
        retained_events = {e.get('event_id'): e for e in analysis.get('events', [])
                           if e.get('event_id') in selected_ids
                           and e.get('timing_source') in ('source_activity', 'qwen_observation')}
    scopes = []
    for edit, span in zip(edits, spans):
        sw = edit.get('shared_window')
        if not isinstance(sw, dict):
            return {}
        a, b = sw['start_time'], sw['end_time']
        if retained_selection:
            matches = [e for e in retained_events.values()
                       if (e.get('start_time'), e.get('end_time')) == (a, b)]
            if len(matches) != 1:
                return {}
        elif not sw.get('event_id'):
            return {}
        if (a, b) != (span.get('start'), span.get('end')) or not 0 <= a < b <= duration:
            return {}
        scopes.append((a, b))
    result = {}
    for index, edit in enumerate(edits):
        p = edit['placement']
        if (edit.get('replacement_material', {}).get('mode') != 'independent_event'
                or edit.get('generator') != 'audio' or edit.get('audio_reference') is not None
                or edit.get('generation', {}).get('source') != 'planner_choice'
                or edit.get('timeline_op') != 'overlay' or edit.get('overflow_policy') != 'error'
                or p.get('kind') not in ('single', 'repeat')
                or p.get('alignment', 'start') != 'start'
                or any(k in p for k in ('anchor', 'anchors', 'selection'))):
            continue


        if p['kind'] == 'repeat' and not any(
                k in p.get('repeat', {}) for k in ('spacing_seconds', 'offsets_seconds')):
            continue
        a, b = scopes[index]
        if p.get('within') != {'start': {'seconds': a}, 'end': {'seconds': b}}:
            continue
        if any(j != index and c < b and d > a for j, (c, d) in enumerate(scopes)):
            continue
        end = min([duration] + [c for j, (c, _) in enumerate(scopes) if j != index and c > a])
        if end > b + 1e-8:
            result[index] = dict(policy=POLICY, source_start=a, source_end=b,
                                 end_time=end, basis='event_selected_removal',
                                 provenance='retained_observed_selection' if retained_selection else 'normalized_event',
                                 boundary='next_source_onset' if end < duration else 'media_end')
    return result


def validate_extent_limit(limit, edit, lo, hi, duration):
    """Fail closed if an internal override disagrees with the owned source scope."""
    sw = edit.get('shared_window') or {}
    end = limit.get('end_time')
    if (limit.get('policy') != POLICY or limit.get('basis') != 'event_selected_removal'
            or (limit.get('source_start'), limit.get('source_end')) != (lo, hi)
            or (sw.get('start_time'), sw.get('end_time')) != (lo, hi)
            or not isinstance(end, (int, float)) or isinstance(end, bool)
            or not math.isfinite(end) or not hi <= end <= duration
            or edit.get('replacement_material', {}).get('mode') != 'independent_event'
            or edit.get('generation', {}).get('source') != 'planner_choice'):
        raise ValueError('Invalid independent replacement extent limit')
    return end
