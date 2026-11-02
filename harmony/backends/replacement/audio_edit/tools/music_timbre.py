"""Optional note-preserving renderer for confirmed isolated instrumental replacement."""
from pathlib import Path
from tempfile import mkdtemp
import json, os, re, subprocess


def target_program(plan, observation):
    if not observation.get('eligible') or observation.get('decision', {}).get('target_is_single_instrument') is not True:
        return None
    if plan.remove_plan.mode != 'text' or plan.remove_plan.result_mode != 'residual' or plan.remove_plan.use_video_prompt:
        return None
    edits=(plan.add_plan.intent or {}).get('edits', [])
    if (len(edits)!=1 or edits[0].get('generator')!='audio' or edits[0].get('audio_reference')!='structure'
            or edits[0].get('placement',{}).get('kind')!='cover' or edits[0].get('timeline_op')!='overlay'
            or plan.add_plan.is_speech or plan.add_plan.is_on_screen):
        return None


    if re.fullmatch(r'(?:(?:solo|human|wordless) )*humming(?: voice)?', str(plan.target_sound).lower().strip()):
        return 53


    label = ' '.join(str(plan.target_sound).lower().split())
    request = str(plan.user_prompt).lower()
    prefix = re.match(r'(?:(?:solo|acoustic|grand|concert|natural) )*piano accompaniment(?: |$)', label)
    preserves = re.search(r'\b(?:preserv(?:e|ing)|maintain(?:ing)?) (?:the )?original melody\b', request)
    names_piano = re.search(r'\bto (?:a |an |the )?(?:(?:solo|acoustic|grand|concert) )*piano\b', request)
    if prefix and preserves and names_piano:
        tail = label[prefix.end():].strip()
        words = set(re.findall(r'[a-z]+', tail))
        allowed = {'replicating','preserving','maintaining','the','original','melody',
                   'and','harmonic','rhythmic','support','timing','phrasing','dynamics'}
        if not tail or (re.match(r'(?:replicating|preserving|maintaining)\b', tail)
                        and words <= allowed):
            return 0
    match=re.fullmatch(r'(?:(?:solo|acoustic|grand|concert|plucked|classical|clean|natural) )*(piano|harp|violin)(?: (?:performance|rendition|music|piece|sound|playing))?',str(plan.target_sound).lower().strip())
    return {'piano':0,'harp':46,'violin':40}[match[1]] if match else None


def render_replacement(input_audio, plan, observation):
    program=target_program(plan, observation)
    if program is None:return dict(status='not_applicable')
    project=Path(__file__).resolve().parents[2]
    python=os.environ.get('HARMONY_MUSIC_RENDER_PYTHON',str(project/'.venvs/music-render311/bin/python'))
    soundfont=os.environ.get('HARMONY_MUSIC_SOUNDFONT',str(project/'externals/music-render/GeneralUser-GS.sf2'))
    if not python or not soundfont or not Path(python).is_file() or not Path(soundfont).is_file():
        return dict(status='tool_unconfigured')
    folder=Path('audio_edit/generated');folder.mkdir(parents=True,exist_ok=True)
    folder=Path(mkdtemp(prefix='music-timbre-',dir=folder)).resolve()
    request=dict(input_audio=str(Path(input_audio).resolve()),output_audio=str(folder/'music.wav'),
        program=program,soundfont=soundfont,report=str(folder/'render.json'))
    request_path=folder/'request.json';request_path.write_text(json.dumps(request,indent=2)+'\n')
    try:
        with (folder/'render.log').open('x') as log:
            subprocess.run([python,str(Path(__file__).with_name('run_music_timbre.py')),str(request_path)],
                env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
        result=json.loads(Path(request['report']).read_text())
        if result['status']!='rendered' or not Path(result['output_audio']).is_file():
            raise ValueError('Music renderer returned no valid audio')
        return result
    except (OSError,ValueError,subprocess.SubprocessError) as exc:
        return dict(status='tool_failed_existing_generator_retained',error=f'{type(exc).__name__}: {exc}',log=str(folder/'render.log'))
