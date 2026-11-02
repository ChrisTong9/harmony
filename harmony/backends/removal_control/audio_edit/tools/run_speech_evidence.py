"""Whisper tool process: read-only evidence, with no network or CPU inference."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def require_allocated_gpu(torch):
    """Inspect the allocation before any CUDA availability check or model loading."""
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Speech inference requires a Slurm GPU allocation')
    subprocess.run(['nvidia-smi'], check=True, capture_output=True, timeout=30)
    if not torch.cuda.is_available():
        raise RuntimeError('No allocated CUDA device')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--healthcheck', action='store_true')
    parser.add_argument('--model', required=True)
    parser.add_argument('--audio')
    parser.add_argument('--output')
    parser.add_argument('--language')
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from audio_edit.analyzers.speech_evidence import (
        MAX_SECONDS, file_sha256, load_model_manifest, normalize_transcription)
    manifest = load_model_manifest(args.model, verify_checksums=args.healthcheck)
    import numpy as np
    import torch
    import librosa
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
    dependencies = {name: importlib.metadata.version(name)
        for name in ('transformers', 'torch', 'librosa', 'soundfile', 'scipy')}

    processor = WhisperProcessor.from_pretrained(args.model, local_files_only=True)
    if args.healthcheck:
        print(json.dumps(dict(ready=True, dependencies=dependencies,
            validation='imports, processor and provisioned file checksums; no GPU inference')))
        return
    if not args.audio or not args.output:
        parser.error('--audio and --output are required for inference')
    require_allocated_gpu(torch)
    from audio_edit.utils.execution_outcome import open_decoded_audio
    original_hash = file_sha256(args.audio)
    with open_decoded_audio(args.audio) as source:
        duration = len(source)/source.samplerate
        if not 0 < duration <= MAX_SECONDS:
            raise ValueError('Speech evidence supports audio up to 90 seconds')
        sample_rate = source.samplerate
        audio = source.read(dtype='float32', always_2d=True)
    if not np.isfinite(audio).all():
        raise ValueError('Non-finite audio samples')
    audio = audio.mean(axis=1)
    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    started = time.monotonic()


    model = WhisperForConditionalGeneration.from_pretrained(args.model,
        local_files_only=True, dtype=torch.float16, attn_implementation='eager').to('cuda').eval()
    recognizer = pipeline('automatic-speech-recognition', model=model,
        tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
        dtype=torch.float16, device=0)
    generation = dict(task='transcribe', do_sample=False, num_beams=1,
        max_new_tokens=440, condition_on_prev_tokens=False)
    if args.language:
        generation['language'] = args.language
    with torch.inference_mode():
        raw = recognizer(dict(raw=audio, sampling_rate=16000), return_timestamps='word',
            generate_kwargs=generation)
    if file_sha256(args.audio) != original_hash:
        raise ValueError('Original audio changed during speech inference')
    result = normalize_transcription(raw, audio_sha256=original_hash, duration=duration,
        sample_rate=sample_rate, model_manifest=manifest, language=args.language)
    result.update(audio_path=str(Path(args.audio).resolve()), dependencies=dependencies,
        runtime_seconds=time.monotonic()-started,
        generation=generation, input_sample_rate=16000,
        preprocessing='Full original timeline; mono channel mean; 16 kHz resampling; no crop')
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
