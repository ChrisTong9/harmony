"""Typed tool evidence: only measured physical regions may change event times."""
from pathlib import Path

from audio_edit.analyzers.measurement_tools import sha256
from audio_edit.plans.temporal_boundaries import finite
from audio_edit.utils.media_util import get_media_duration


def _original_axis(result, duration):
    axis = result.get('time_axis', {})
    return (isinstance(axis, dict) and axis.get('origin')=='original_audio_start'
        and axis.get('offset_seconds')==0 and axis.get('cropped') is False
        and axis.get('units')=='seconds' and finite(axis.get('original_duration'))
        and abs(axis['original_duration']-duration)<=.02)


def merge_tool_evidence(name, result, analysis, requirements, duration, *, original_hash=None):
    """Return current observations and an optional activity measurement.

    Every tool result is retained for the planner and Qwen. Transcription never
    supplies a measured physical region or an executable event by itself. An
    arbitrary result['analysis'] is not authority, even from a known tool.
    """
    if not isinstance(result, dict) or result.get('tool') != name:
        raise ValueError('Evidence tool returned the wrong result contract')
    known = {q['id'] for q in requirements}
    ids = result.get('question_ids')
    if (not isinstance(ids, list) or not ids
            or any(not isinstance(key,str) or key not in known for key in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('Tool evidence must retain eligible unique requirement IDs')
    if not finite(result.get('duration')) or abs(result['duration']-duration) > .02:
        raise ValueError('Tool evidence does not preserve the original duration')
    if name in ('activity_regions', 'source_activity'):
        path = result.get('audio_path')
        if (not isinstance(path,str) or not Path(path).is_file()
                or sha256(path) != result.get('audio_sha256')
                or abs(get_media_duration(path)-duration)>.02):
            raise ValueError('Measured audio artifact changed or is not on the original time axis')
        expected = result.get('original_audio_sha256') if name=='source_activity' else result.get('audio_sha256')
        if original_hash is not None and expected != original_hash:
            raise ValueError('Measurement belongs to a different original recording')
        if name=='source_activity' and not _original_axis(result,duration):
            raise ValueError('Source evidence needs unchanged original-axis provenance')
        regions = result.get('regions')
        if not isinstance(regions,list) or not all(isinstance(r,dict) for r in regions):
            raise ValueError('Activity evidence needs a list of measured regions')
        region_ids = [r.get('region_id') for r in regions]
        if (any(not isinstance(key,str) or not key for key in region_ids)
                or len(set(region_ids)) != len(region_ids)):
            raise ValueError('Measured regions need unique nonempty IDs')
        if any(not all(finite(r.get(k)) for k in ('start_time','end_time'))
                or not 0 <= r['start_time'] < r['end_time'] <= duration
                or not isinstance(r.get('complete'),bool) for r in regions):
            raise ValueError('Measured regions must be bounded physical intervals')
        if not finite(result.get('resolution_seconds')) or result['resolution_seconds']<=0 or not isinstance(result.get('method'),str):
            raise ValueError('Measured evidence needs its method and physical resolution')
        return analysis, result
    if name == 'speech_transcription':
        if (result.get('kind')!='tool_evidence' or not _original_axis(result,duration)
                or result.get('supports_diarization') is not False
                or not isinstance(result.get('text'),str)
                or not isinstance(result.get('words'),list)
                or not isinstance(result.get('segments'),list)
                or result.get('audio_sha256') != result.get('original_audio_sha256')
                or original_hash is not None and result.get('original_audio_sha256') != original_hash):
            raise ValueError('Speech evidence needs original-file lexical provenance, without speaker identity')
        return analysis, None
    raise ValueError('Unregistered evidence cannot authorize observations or event timing')
