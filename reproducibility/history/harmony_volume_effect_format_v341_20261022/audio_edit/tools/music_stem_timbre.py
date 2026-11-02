"""Use the first removed musical component for literal instrument timbre changes."""
from pathlib import Path
from tempfile import mkdtemp
import json, os, re, subprocess
import numpy as np
import soundfile as sf

PROGRAMS={'piano':0,'harp':46,'violin':40,'guitar':24,'oboe':68}

def request_program(ctx,plan):


    if ctx.params.get('replacement_music_backing_decision',{}).get('eligible') is not True:return None
    removal=plan.remove_plan;edits=(plan.add_plan.intent or {}).get('edits',[])
    if (removal.mode!='text' or removal.result_mode!='residual' or removal.use_video_prompt
            or len(removal.placements)!=1 or len(edits)!=1 or plan.add_plan.is_speech or plan.add_plan.is_on_screen):return None
    edit=edits[0];window=edit.get('shared_window',{})
    if (edit.get('generator')!='audio' or edit.get('audio_reference')!='structure'
            or edit.get('placement',{}).get('kind')!='cover' or edit.get('timeline_op')!='overlay'
            or window.get('start_time')!=0 or any(window.get(k) is not None for k in ('reference_event','event_id','occurrence_index','reference_source'))):return None
    p=removal.placements[0]
    if p.start_time!=0 or p.end_time!=window.get('end_time'):return None


    text=ctx.user_prompt.strip().rstrip('.!?').strip().lower()
    target=r'(?:an? )?(?:acoustic )?(?P<target>piano|harp|violin|guitar|oboe)(?: (?:timbre|tone))?'
    patterns=[r'change (?:the )?[a-z ]+? (?:timbre|tone)(?: in the background)? to '+target,
              r'change the (?:timbre|tone) of (?:the )?[a-z ]+? to '+target]
    matches=[m for pattern in patterns if (m:=re.fullmatch(pattern,text))]
    if len(matches)!=1:return None
    name=matches[0]['target']
    if not re.search(r'\b'+re.escape(name)+r'\b',str(plan.target_sound).lower()):return None
    return PROGRAMS[name]

def render_stem_replacement(ctx,plan,reference):
    program=request_program(ctx,plan)
    if program is None:return dict(status='not_applicable')
    if not reference or not Path(reference).is_file():return dict(status='missing_first_stem_existing_generator_retained')
    stem,rate=sf.read(reference,always_2d=True,dtype='float32')
    original,sr=sf.read(ctx.input_audio,always_2d=True,dtype='float32')
    if (rate!=sr or stem.shape!=original.shape or not np.isfinite(stem).all()
            or float(np.sqrt(np.mean(stem.astype(np.float64)**2)))<1e-7):return dict(status='invalid_first_stem_existing_generator_retained')
    project=Path(__file__).resolve().parents[2]
    python=os.environ.get('HARMONY_MUSIC_RENDER_PYTHON',str(project/'.venvs/music-render311/bin/python'))
    soundfont=os.environ.get('HARMONY_MUSIC_SOUNDFONT',str(project/'externals/music-render/GeneralUser-GS.sf2'))
    if not Path(python).is_file() or not Path(soundfont).is_file():return dict(status='tool_unconfigured')
    folder=Path('audio_edit/generated');folder.mkdir(parents=True,exist_ok=True)
    folder=Path(mkdtemp(prefix='stem-timbre-',dir=folder)).resolve()
    request=dict(input_audio=str(Path(reference).resolve()),output_audio=str(folder/'music.wav'),program=program,soundfont=soundfont,report=str(folder/'render.json'))
    path=folder/'request.json';path.write_text(json.dumps(request,indent=2)+'\n')
    try:
        with (folder/'render.log').open('x') as log:
            subprocess.run([python,str(Path(__file__).with_name('run_music_timbre.py')),str(path)],env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
        result=json.loads(Path(request['report']).read_text())
        if result['status']!='rendered' or not Path(result['output_audio']).is_file():raise ValueError('No valid first stem render')
        result.update(input_role='existing_first_SAM_removed_component',original_audio=ctx.input_audio,new_separations=0)
        return result
    except (OSError,ValueError,subprocess.SubprocessError) as exc:
        return dict(status='tool_failed_existing_generator_retained',error=f'{type(exc).__name__}: {exc}',log=str(folder/'render.log'))
