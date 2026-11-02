"""Extract the observed protected class with the existing SAM Audio tool."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import subprocess
from tempfile import mkdtemp


def retain_music_vocals(input_audio, cleaned_audio, decision):
    if not decision.get('eligible'):
        return cleaned_audio, None
    from audio_edit.models import AudioEditContext
    from audio_edit.plans.remove_sound_plan import RemoveSoundPlan
    from audio_edit.runners.samaudio_runner import SAMAudioRunner
    from audio_edit.utils.media_util import get_media_duration
    import numpy as np
    import soundfile as sf
    root=Path('audio_edit/generated');root.mkdir(parents=True,exist_ok=True)
    folder=Path(mkdtemp(prefix='sam-complement-',dir=root)).resolve()
    plan=RemoveSoundPlan(backend='samaudio',mode='text',result_mode='result',text_prompt=decision['retained_source'],
        temporal_status='resolved',notes='Extract the observed protected source; whole instrumental backing is replaced.')
    ctx=AudioEditContext('Extract the retained audible source',input_audio=input_audio)
    runner=SAMAudioRunner(output_dir=str(folder))
    result=runner.run_remove(ctx,plan)
    original,rate=sf.read(input_audio,always_2d=True,dtype='float32')
    extracted,sr=sf.read(result.output_audio,always_2d=True,dtype='float32')
    if sr!=rate:
        resampled=folder/'resampled.wav'
        subprocess.run(['ffmpeg','-nostdin','-v','error','-i',result.output_audio,'-ar',str(rate),'-c:a','pcm_f32le',str(resampled)],check=True)
        extracted,sr=sf.read(resampled,always_2d=True,dtype='float32')
    if extracted.shape[1]==1 and original.shape[1]>1:
        extracted=np.repeat(extracted,original.shape[1],axis=1)
    elif original.shape[1]==1 and extracted.shape[1]>1:
        extracted=extracted.mean(axis=1,keepdims=True)
    assert extracted.shape[1]==original.shape[1] and abs(len(extracted)-len(original))<=2
    aligned=np.zeros_like(original);aligned[:min(len(extracted),len(aligned))]=extracted[:len(aligned)]
    assert np.isfinite(aligned).all()
    output=folder/'retained.wav';sf.write(output,aligned,rate,subtype='FLOAT')
    return str(output),dict(status='observed_protected_source_extracted',previous_residual=str(cleaned_audio),
        corrected_residual=str(output),existing_generator_reference_retained=True,
        execution=dict(backend='sam_audio',plan=asdict(plan),separation_calls=result.params.get('separation_calls'),
            original_sample_grid=True,normalization_applied=False))
