"""Take the first complete event; never fit by cutting through active material."""
from pathlib import Path
import hashlib
import math
import numpy as np
import soundfile as sf

def first_event_asset(source, destination, target_seconds):
    if not math.isfinite(target_seconds) or target_seconds <= 0:
        raise ValueError('A positive event generation budget is required')
    audio, sr = sf.read(source, always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Invalid decoded event material')
    hop = max(1, round(.01 * sr))
    rms = np.array([np.sqrt(np.mean(audio[a:a+hop] ** 2)) for a in range(0, len(audio), hop)])
    threshold = max(float(rms.max()) * .01, 1e-5)
    active = np.flatnonzero(rms > threshold)
    if not len(active):
        raise ValueError('Decoded event is silent')
    gaps = np.flatnonzero(np.diff(active) - 1 >= 8)
    start = max(0, int(active[0]) * hop - round(.02 * sr))
    if len(gaps):
        left, right = int(active[gaps[0]] + 1), int(active[gaps[0] + 1])
        valley = left + int(np.argmin(rms[left:right]))
        end = min(len(audio), valley * hop + hop // 2)
        boundary = float(rms[valley])
        gap = [left * hop / sr, right * hop / sr]
    else:
        end = min(len(audio), (int(active[-1]) + 1) * hop + round(.02 * sr))
        boundary = float(rms[-1]) if end == len(audio) else float(rms[min(len(rms)-1, end//hop)])
        gap = None
        if end == len(audio) and rms[-1] > threshold:
            raise ValueError('Decoded event reaches an active file boundary; tail remains unresolved')
    event = audio[start:end].copy()
    fade = min(round(.01 * sr), len(event) // 2)
    if fade:
        event[-fade:] *= np.linspace(1, 0, fade)[:, None]
    requested = round(target_seconds * sr)


    prepared = np.zeros((max(len(event), requested), audio.shape[1]), dtype=np.float32)
    prepared[:len(event)] = event
    sf.write(destination, prepared, sr, subtype='FLOAT')
    return dict(method='first_chronological_complete_event', source=str(Path(source).resolve()),
                source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                source_seconds=len(audio)/sr, source_start=start/sr, source_end=end/sr,
                event_seconds=len(event)/sr, prepared_seconds=len(prepared)/sr,
                target_seconds=target_seconds, fits_generation_budget=len(event)<=requested,
                first_gap=gap, boundary_rms=boundary, frame_seconds=hop/sr,
                relative_rms_threshold=.01, end_fade_seconds=.01, time_stretch=False)
