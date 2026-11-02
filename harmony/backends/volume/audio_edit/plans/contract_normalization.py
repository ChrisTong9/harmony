"""Lossless compatibility at the planner/compiler boundary.

Normalize representation, never move an anchor, shrink coverage, drop an event,
or relax an explicit overflow policy. Raw plans remain available to the caller.
"""
from copy import deepcopy
import math


REFERENCE_KEYS = ('seconds', 'event_id', 'media_edge', 'media_fraction')


def normalize_edit(edit):
    result = deepcopy(edit)
    audit = []

    def record(code, before, after, reason):
        audit.append(dict(code=code, before=deepcopy(before), after=deepcopy(after), reason=reason))

    def anchor(value):
        if not isinstance(value, dict):
            return value
        value = deepcopy(value)


        populated = [k for k in REFERENCE_KEYS if value.get(k) is not None]
        if len(populated) == 1:
            before = deepcopy(value)
            for key in REFERENCE_KEYS:
                if key != populated[0] and value.get(key) is None:
                    value.pop(key, None)
            if before != value:
                record('null_reference_fields_removed', before, value, 'One non-null reference; no reference selected by precedence')
        if value.get('offset_seconds') is None and 'offset_seconds' in value:
            value['offset_seconds'] = 0.
            record('null_offset_is_zero', None, 0., 'Absent explicit offset does not move the anchor')
        return value

    window = result.get('window')
    if isinstance(window, dict) and 'start' not in window and any(k in window for k in REFERENCE_KEYS):
        result['window'] = dict(start=window, end=None)
        record('point_window_wrapped', window, result['window'], 'A point anchor is represented as window.start, not a second timeline decision')
    window = result.get('window')
    if not isinstance(window, dict) or 'start' not in window:
        raise ValueError('window requires {start: ANCHOR, end: ANCHOR or null}; start is the placement reference')
    window['start'] = anchor(window['start'])
    window['end'] = anchor(window.get('end'))
    individual = result.get('placement_anchors')
    if isinstance(individual, list):
        individual = [anchor(a) for a in individual]
        result['placement_anchors'] = individual
    if individual == []:
        result['placement_anchors'] = None
        record('empty_placement_anchors', [], None, 'Empty optional list supplies no additional occurrences')
    count = result.get('count')
    coverage = result.get('coverage')
    if coverage == 'single' and isinstance(count, int) and not isinstance(count, bool) and count > 1:
        result['coverage'] = coverage = 'repeated'
        record('explicit_count_is_repeated', 'single', 'repeated', 'Preserve all explicitly planned occurrences; single describes the asset, not its count')


    if coverage == 'continuous' and result.get('placement_anchors'):
        boundaries = [window['start'], window['end']]
        if all(a in boundaries for a in result['placement_anchors']):
            before = result['placement_anchors']
            result['placement_anchors'] = None
            record('redundant_continuous_boundaries', before, None, 'Exact duplicate window boundaries; full continuous coverage unchanged')
        else:
            raise ValueError('Continuous placement_anchors conflict with the coverage window: put each continuous range in its own window; do not discard or shorten it')
    elif count != 1 and result.get('placement_anchors') == [window['start']]:
        before = result['placement_anchors']
        result['placement_anchors'] = None
        record('repetition_origin_not_occurrence_list', before, None, 'The repeated origin is already window.start; retain count and explicit offsets')
    elif count == 1 and window['end'] is not None and result.get('placement_anchors') == [window['start'],window['end']]:
        before = result['placement_anchors']
        result['placement_anchors'] = None
        record('single_window_boundaries_not_two_occurrences', before, None,
               'Exactly one event was requested; the duplicate range boundaries are not two event starts')
    offsets = result.get('offsets_seconds')
    if coverage == 'continuous' and count == 1 and offsets in (None, [], [0], [0.]):
        if result.get('distribution', 'explicit') != 'explicit':
            record('continuous_distribution', result['distribution'], 'explicit', 'One continuous range has no distribution of discrete events')
            result['distribution'] = 'explicit'
    elif result.get('distribution') == 'sequential' and isinstance(offsets, list) and len(offsets) == count:


        if all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in offsets):
            result['distribution'] = 'explicit'
            record('explicit_offsets_preserved', 'sequential', 'explicit', 'Preserve given spacing instead of silently packing events together')
    if result.get('count') == 1 and result.get('offsets_seconds') in (None, []) and not result.get('placement_anchors'):
        result['offsets_seconds'] = [0.]
        record('single_origin_default', offsets, [0.], 'One occurrence starts at its explicitly resolved window origin')
    if result.get('timeline_op') == 'overlay' and window['end'] is None and isinstance(window['start'], dict) and window['start'].get('relation') == 'before' and 'alignment' not in result:
        result['alignment'] = 'before'
        record('before_extent_alignment', None, 'before', 'The entire requested event sequence ends before its reference, not after it')
    return result, audit
