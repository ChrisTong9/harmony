"""Whole-file extraction of two explicitly requested distinct vocal activities.

Candidate template only: this file is not installed into the accepted source.
Composition reuses the fixed unscaled source-sum implementation tested in v199.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import re
from tempfile import mkdtemp


_ACTIVITIES = {'speech', 'humming', 'singing', 'laughter', 'cheering', 'shouting', 'whistling'}


def decide(ctx, plan):
    no = dict(eligible=False, reason='Requires two explicitly retained whole-file vocal activities')
    if (plan.backend != 'samaudio' or plan.mode != 'text' or plan.result_mode != 'result'
            or plan.temporal_status not in {'not_requested', 'resolved'}
            or plan.time_span or plan.placements or plan.use_video_prompt or plan.visual_prompt
            or plan.temporal_schema_error
            or (plan.temporal and plan.temporal.kind != 'unspecified')
            or ctx.params.get('participant_extraction', {}).get('eligible')):
        return no
    query = ' '.join(plan.text_prompt.lower().strip(' .').split())
    parts = re.split(r'\s+(?:and|&)\s+', query)
    if len(parts) != 2 or len(set(parts)) != 2 or not all(p in _ACTIVITIES for p in parts):
        return no
    request = ' '.join(ctx.user_prompt.lower().split())
    phrase = r'\b(?:extract|isolate|retain|keep)\s+(?:only\s+)?(?:the\s+)?' + re.escape(parts[0]) + r'\s+(?:and|&)\s+' + re.escape(parts[1])
    if not re.search(phrase + r'(?:\s+sounds?)?(?=[.!?;,]|\s+(?:from|without)\b|$)', request):
        return no
    if re.search(r'\b(?:first|last|second|before|after|until|between|during|seconds?|minutes?|louder|quieter)\b', request):
        return no
    return dict(eligible=True, reason='Explicitly retained distinct vocal activities; extract each class once',
        source_queries=parts, input_policy='same_original_for_each_source',
        composition='sum_unscaled_source_estimates', new_model=False,
        limitation='Overlapping source estimates can double-count leakage; no gain tuning or score-selected stems.')


def render(ctx, plan, decision, runner):
    from audio_edit.models import AudioEditContext
    from audio_edit.tools.retained_source_sum import combine
    folder = Path(mkdtemp(prefix='retained-vocal-activities-', dir=runner.output_dir)).resolve()
    sources, calls = [], []
    for query in decision['source_queries']:
        single = deepcopy(plan)
        single.text_prompt = query

        single.removal_evidence = None
        child = AudioEditContext(ctx.user_prompt, input_audio=ctx.input_audio)
        result = runner.run_remove(child, single)
        sources.append(result.output_audio)
        calls.extend(result.params.get('separation_calls', []))
        ctx.warnings.extend(result.warnings)
    output, composition = combine(ctx.input_audio, sources, folder)
    ctx.params['remove_plan'] = asdict(plan)
    ctx.params['temporal_resolution_status'] = plan.temporal_status
    ctx.params['retained_vocal_activities'] = dict(decision, composition=composition)
    ctx.params['removal_execution'] = dict(backend='sam_audio', retained_source_queries=decision['source_queries'])
    ctx.params.setdefault('separation_calls', []).extend(calls)
    ctx.output_audio = output
    return ctx
