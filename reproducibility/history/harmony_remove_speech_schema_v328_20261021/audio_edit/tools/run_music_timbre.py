"""CPU Basic Pitch -> unquantized notes -> SoundFont; fixed defaults, one inference."""
from pathlib import Path
import hashlib, json, os, sys
import numpy as np
import soundfile as sf


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_notes(notes, frames, rate, channels, soundfont, program):
    from tinysoundfont import Synth
    if channels not in (1,2):raise ValueError('Expected mono or stereo source')
    synth=Synth(samplerate=rate)
    sfid=synth.sfload(soundfont)
    synth.program_select(0,sfid,0,program)
    events=[]
    for note in notes:
        start=max(0,min(frames,round(note.start*rate)))
        end=max(start,min(frames,round(note.end*rate)))
        if end>start:
            events.extend([(start,1,int(note.pitch),int(note.velocity)),(end,0,int(note.pitch),0)])
    if not events:raise ValueError('Transcription produced no in-scope notes')
    events.sort()
    result=np.zeros((frames,2),dtype=np.float32);pos=0
    for frame,on,pitch,velocity in events:
        if frame>pos:
            result[pos:frame]=np.frombuffer(synth.generate_simple(frame-pos),dtype=np.float32).reshape(-1,2)
            pos=frame
        if on:synth.noteon(0,pitch,velocity)
        else:synth.noteoff(0,pitch)
    if pos<frames:
        result[pos:]=np.frombuffer(synth.generate_simple(frames-pos),dtype=np.float32).reshape(-1,2)
    if channels==1:result=result.mean(axis=1,keepdims=True)
    return result,events


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Model inference requires a scheduled allocation')
    request=json.loads(Path(sys.argv[1]).read_text());output=Path(request['output_audio'])
    assert not output.exists(),'Keep the first render; no regeneration'
    source,rate=sf.read(request['input_audio'],always_2d=True,dtype='float32')
    if not np.isfinite(source).all():raise ValueError('Non-finite source')
    import onnxruntime as ort
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model,predict

    options=ort.SessionOptions();options.intra_op_num_threads=max(1,min(4,int(os.environ.get('SLURM_CPUS_PER_TASK','1'))))
    options.inter_op_num_threads=1
    model=Model.__new__(Model);model.model_type=Model.MODEL_TYPES.ONNX
    model.model=ort.InferenceSession(str(ICASSP_2022_MODEL_PATH),sess_options=options,providers=['CPUExecutionProvider'])
    raw,midi,note_events=predict(request['input_audio'],model)
    notes=[n for instrument in midi.instruments for n in instrument.notes]
    audio,events=render_notes(notes,len(source),rate,source.shape[1],request['soundfont'],request['program'])
    source_rms=float(np.sqrt(np.mean(source.astype(np.float64)**2)))
    result_rms=float(np.sqrt(np.mean(audio.astype(np.float64)**2)))
    peak=float(np.max(np.abs(audio)))
    if result_rms<1e-7 or source_rms<1e-7:raise ValueError('Source or rendered notes are silent')
    gain=min(source_rms/result_rms,.98/max(peak,1e-7))
    audio*=gain
    if audio.shape!=source.shape or not np.isfinite(audio).all():raise ValueError('Invalid output grid')
    for instrument in midi.instruments:
        instrument.program=request['program'];instrument.pitch_bends=[]
    midi.write(str(output.with_suffix('.mid')))
    np.savez_compressed(output.with_suffix('.npz'),**raw)
    sf.write(output,audio,rate,subtype='FLOAT')
    result=dict(status='rendered',backend='basic_pitch_soundfont',input_audio=request['input_audio'],
        input_sha256=sha(request['input_audio']),output_audio=str(output),output_sha256=sha(output),
        program=request['program'],soundfont=request['soundfont'],soundfont_sha256=sha(request['soundfont']),
        model_path=str(ICASSP_2022_MODEL_PATH),model_sha256=sha(ICASSP_2022_MODEL_PATH),
        runtime='ONNX CPUExecutionProvider',transcription_defaults=True,inference_calls=1,render_calls=1,
        notes=[dict(start=float(n.start),end=float(n.end),pitch=int(n.pitch),velocity=int(n.velocity)) for n in notes],
        midi_path=str(output.with_suffix('.mid')),note_count=len(notes),note_event_count=len(events),
        source_rms=source_rms,raw_render_rms=result_rms,level_gain=gain,frames=len(source),sample_rate=rate,
        timing='Original transcription times; no grid quantization, tempo changes, pitch transposition or note additions',
        pitch_bends='Omitted for fixed-pitch piano and harp',channels=source.shape[1])
    Path(request['report']).write_text(json.dumps(result,indent=2)+'\n')
    print('RENDERED',len(notes),'notes',request['program'],flush=True)


if __name__=='__main__':main()
