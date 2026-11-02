"""Separate the new sound's identity from replacement and timing instructions."""
from copy import deepcopy

REPLACEMENT_MATERIAL_PROMPT = '''
For each shared-window replacement edit, also declare replacement_material:
{"mode":"independent_event|independent_texture|structured_audio","target":string or null}.
This is the authoritative generation-content decision, separate from placement.
- independent_event: one independent audible event per asset (single/repeat).
  target is a short noun phrase naming ONLY the new sound and essential acoustic
  attributes, at most 24 words. Do not include the removed source, replacement
  instructions, original envelope, occurrence counts, times, or mixing directions.
  Examples: "clock ticking", "metallic latch clicking", "owl hooting".
  Code uses this target verbatim as the generation prompt and audio_reference=null;
  code selects the first complete event from decoded material, then places/repeats it.
- independent_texture: an independent sound scene spanning the supplied window
  (cover), including ambience or a sequence of discrete sounds with natural pauses.
  Use the same target-only description rule; audio_reference=null and full coverage.
  A cover placement describes window-length content; it must not request the
  one-event extractor just because the new source is an animal call or a clap.
- structured_audio: preserve an existing musical performance, melody or internal
  rhythm. Keep the normal detailed audio_prompt and reference decision, target=null.
  This includes a rhythmic sequence whose internal events are not individually
  supplied as windows; do not discard its structure just because its timbre is
  percussive or environmental. Individually supplied event windows already own
  their times and count, and do not by themselves require waveform conditioning.
Musical instrument transformations retain structured_audio. Spoken generation
keeps its existing speech fields and does not need replacement_material.
Keep reasons concise. Do not duplicate the target phrase as editing prose.
'''

def normalize_replacement_material(data, *, required=False):
    data = deepcopy(data)
    audit = []
    for index, edit in enumerate(data.get('edits', [])):
        if edit.get('generator') == 'speech':
            continue
        material = edit.get('replacement_material')
        if material is None:
            if required:
                raise ValueError('Fresh replacement needs replacement_material')
            continue
        if not isinstance(material, dict) or set(material) - {'mode', 'target'}:
            raise ValueError('replacement_material needs only mode and target')
        mode = material.get('mode')
        if mode not in ('independent_event', 'independent_texture', 'structured_audio'):
            raise ValueError('Unknown replacement material mode')
        if mode == 'structured_audio':
            continue
        target = material.get('target')
        if not isinstance(target, str) or not target.strip() or len(target.split()) > 24:
            raise ValueError('Independent replacement needs a short target-only description')
        kind = edit.get('kind') or edit.get('placement', {}).get('kind', 'single')


        if (mode == 'independent_event' and kind == 'cover'
                and type(edit.get('shared_window_index')) is int
                and edit['shared_window_index'] >= 0
                and edit.get('generator') == 'audio'
                and edit.get('audio_reference') is None
                and edit.get('timeline_op', 'overlay') == 'overlay'
                and not edit.get('repeat')
                and (edit.get('generation') or {}).get('source') == 'planner_choice'):
            audit.append(dict(edit_index=index, code='shared_cover_material_type',
                old_mode=mode, new_mode='independent_texture',
                reason='Existing cover owns the content extent; do not extract one event from its sound scene'))
            mode = material['mode'] = 'independent_texture'
        if (mode == 'independent_texture') != (kind == 'cover'):
            raise ValueError('Independent material type disagrees with event/coverage intent')
        target = target.strip()
        audit.append(dict(edit_index=index, mode=mode, old_prompt=edit.get('audio_prompt'),
                          old_reference=edit.get('audio_reference'), target=target))
        edit.update(audio_prompt=target, audio_reference=None)
        material['target'] = target
    return data, audit
