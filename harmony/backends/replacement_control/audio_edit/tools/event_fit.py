"""Fit an already generated independent event into its available decay slot.

Only the generated asset is time-compressed. Original audio, source deletion
windows, occurrence count and start times never change. No new sampling occurs.
"""
import hashlib
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import soundfile as sf


def fit_event_asset(source, destination, available_seconds):
    if not isinstance(available_seconds, (int, float)) or isinstance(available_seconds, bool) or not math.isfinite(available_seconds) or available_seconds <= 0:
        raise ValueError('Independent event needs a finite positive available extent')
    original, sr = sf.read(source, always_2d=True)
    if not len(original) or not np.isfinite(original).all():
        raise ValueError('Invalid independent event asset')
    frames = int(math.floor(available_seconds * sr + 1e-6))
    if frames < 1:
        raise ValueError('No complete output sample fits the available extent')
    source_sha = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    if len(original) <= frames:
        return dict(applied=False, path=str(Path(source).resolve()), source_sha256=source_sha)
    rate = len(original) / frames


    stages = max(1, math.ceil(math.log2(rate)))
    factor = rate ** (1 / stages)
    filters = ','.join('atempo=' + format(factor, '.15g') for _ in range(stages))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='event-fit-', dir=destination.parent) as temp:
        rendered = Path(temp) / 'tempo.wav'
        command = ['ffmpeg', '-v', 'error', '-nostdin', '-y', '-i', str(source),
                   '-af', filters, '-c:a', 'pcm_f32le', str(rendered)]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        audio, rate_out = sf.read(rendered, always_2d=True)
    if rate_out != sr or audio.shape[1] != original.shape[1] or not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Time compression changed the sample grid or produced invalid audio')
    pre_fit_frames = len(audio)
    trimmed = audio[frames:]
    trim_rms = float(np.sqrt(np.mean(trimmed ** 2))) if len(trimmed) else 0.0


    output = np.zeros((frames, audio.shape[1]), dtype=np.float32)
    count = min(frames, len(audio))
    output[:count] = audio[:count]
    fade = min(round(.01 * sr), count // 2)
    if fade:
        output[count-fade:count] *= np.linspace(1, 0, fade)[:, None]
    sf.write(destination, output, sr, subtype='FLOAT')
    assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == source_sha
    return dict(applied=True, path=str(destination.resolve()), source_sha256=source_sha,
        method='ffmpeg_atempo_independent_event_fit', rate=rate, atempo_factors=[factor] * stages,
        original_frames=len(original), output_frames=frames, sample_rate=sr,
        available_seconds=available_seconds, pitch_resampling=False,
        atempo_output_frames=pre_fit_frames, trimmed_frames=max(0, pre_fit_frames-frames),
        trimmed_tail_rms=trim_rms, padded_frames=max(0, frames-pre_fit_frames), end_fade_seconds=fade/sr,
        audible_quality='unreviewed')


def fit_replacement_assets(plan, placements, assets_by_key, actual_lengths, extent_limits, folder):
    paths, lengths, audit = dict(assets_by_key), dict(actual_lengths), []
    if plan.is_on_screen:
        return paths, lengths, audit
    for index, placement in enumerate(placements):
        edit = plan.intent['edits'][placement['edit_index']]
        limit = extent_limits.get(placement['edit_index'])
        if (not placement.get('shared_window')
                or edit.get('generator') != 'audio' or edit.get('audio_reference') is not None
                or edit.get('replacement_material', {}).get('mode') != 'independent_event'
                or edit.get('generation', {}).get('source') != 'planner_choice'
                or edit.get('placement', {}).get('kind') != 'single'):
            continue
        if not limit:
            shape, shared = edit.get('placement', {}), placement['shared_window']
            if (shared != edit.get('shared_window')
                    or shape.get('alignment', 'start') != 'start'
                    or any(k in shape for k in ('anchor', 'anchors', 'selection'))
                    or shape.get('within') != {'start': {'seconds': shared['start_time']},
                                              'end': {'seconds': shared['end_time']}}):
                continue


        end = limit['end_time'] if limit else placement['shared_window']['end_time']
        available = end - placement['start_time']
        key = placement['key']
        result = fit_event_asset(paths[key], Path(folder) / ('fit-' + str(index) + '.wav'), available)
        if result['applied']:
            paths[key] = result['path']
            lengths[key] = result['output_frames'] / result['sample_rate']
            audit.append(dict(key=key, start_time=placement['start_time'], limit=limit, **result))
    return paths, lengths, audit
