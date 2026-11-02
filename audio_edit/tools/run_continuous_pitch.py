"""One fixed pYIN contour and a continuous SoundFont violin performance."""

from pathlib import Path
import hashlib, json, os, sys
import numpy as np
import soundfile as sf


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def analyze(path):
    import librosa

    y, sr = librosa.load(path, sr=22050, mono=True)
    f0, voiced, prob = librosa.pyin(
        y, fmin=65.40639132514966, fmax=2093.004522404789, sr=sr, frame_length=2048, hop_length=441
    )
    return dict(
        f0=f0,
        voiced=voiced,
        voiced_probability=prob,
        times=librosa.times_like(f0, sr=sr, hop_length=441),
    )


def render(request, track):
    from tinysoundfont import Synth
    from scipy.signal import resample_poly
    from math import gcd

    source, sr = sf.read(request["input_audio"], always_2d=True, dtype="float32")
    rate = request["sample_rate"]
    frames = request["frames"]
    channels = request["channels"]
    if sr != rate:
        g = gcd(sr, rate)
        source = resample_poly(source, rate // g, sr // g, axis=0)
    assert abs(len(source) - frames) <= 2
    aligned = np.zeros((frames, source.shape[1]), dtype="float32")
    aligned[: min(frames, len(source))] = source[:frames]
    source = aligned
    f0 = track["f0"]
    times = track["times"]
    valid = np.isfinite(f0) & track["voiced"]
    midi = 69 + 12 * np.log2(f0 / 440.0)
    synth = Synth(samplerate=rate)
    sfid = synth.sfload(request["soundfont"])
    synth.program_select(0, sfid, 0, 40)
    synth.pitchbend_range(0, 48.0)

    edges = np.diff(np.r_[False, valid, False].astype(int))
    segments = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    assert segments
    events = {}
    anchors = {}
    for a, b in segments:
        anchor = int(round(float(np.median(midi[a:b]))))
        assert 0 <= anchor <= 127
        anchors[a] = anchor
        for i in range(a, b):
            offset = float(midi[i] - anchor)
            assert abs(offset) < 48
            wheel = round(8192 + offset / 48 * 8192)
            assert 0 <= wheel <= 16383
            events.setdefault(min(frames, round(times[i] * rate)), []).append(("bend", wheel))
            if i == a:
                events[min(frames, round(times[i] * rate))].append(("on", anchor))
        end = min(frames, round((times[b] if b < len(times) else frames / rate) * rate))
        events.setdefault(end, []).append(("off", anchor))
    out = np.zeros((frames, 2), dtype="float32")
    pos = 0
    for frame, items in sorted(events.items()):
        if frame > pos:
            out[pos:frame] = np.frombuffer(
                synth.generate_simple(frame - pos), dtype=np.float32
            ).reshape(-1, 2)
            pos = frame
        for kind, value in items:
            if kind == "bend":
                synth.pitchbend(0, value)
            elif kind == "on":
                synth.noteon(0, value, 100)
            else:
                synth.noteoff(0, value)
    if pos < frames:
        out[pos:] = np.frombuffer(synth.generate_simple(frames - pos), dtype=np.float32).reshape(
            -1, 2
        )
    centers = np.minimum(frames - 1, np.round(times * rate).astype(int))
    half = max(1, round(0.01 * rate))
    rms = np.array(
        [
            np.sqrt(
                np.mean(source[max(0, k - half) : min(frames, k + half)].astype("float64") ** 2)
            )
            for k in centers
        ]
    )
    assert rms.max() > 1e-7
    envelope = np.interp(np.arange(frames), centers, rms / rms.max()).astype("float32")
    voiced = np.interp(np.arange(frames), centers, valid.astype(float))
    out *= envelope[:, None]
    out *= voiced[:, None]
    if channels == 1:
        out = out.mean(axis=1, keepdims=True)
    assert channels in (1, 2) and out.shape == (frames, channels)
    source_rms = float(np.sqrt(np.mean(source.astype("float64") ** 2)))
    render_rms = float(np.sqrt(np.mean(out.astype("float64") ** 2)))
    assert render_rms > 1e-7
    gain = min(source_rms / render_rms, 0.98 / max(float(np.max(np.abs(out))), 1e-7))
    out *= gain
    assert np.isfinite(out).all()
    path = Path(request["output_audio"])
    assert not path.exists()
    sf.write(path, out, rate, subtype="FLOAT")
    return dict(
        status="rendered",
        backend="pyin_continuous_soundfont",
        input_audio=request["input_audio"],
        input_sha256=sha(request["input_audio"]),
        output_audio=str(path),
        output_sha256=sha(path),
        program=40,
        pitch_bend_range=48,
        voiced_segments=len(segments),
        source_rms=source_rms,
        raw_render_rms=render_rms,
        global_gain=gain,
        frames=frames,
        sample_rate=rate,
        input_role="existing_first_SAM_removed_component",
        new_separations=0,
    )


if __name__ == "__main__":
    assert os.environ.get("SLURM_JOB_ID"), (
        "Music analysis/render must run inside a Slurm allocation"
    )
    request = json.loads(Path(sys.argv[1]).read_text())
    assert sha(request["input_audio"]) == request["input_sha256"]
    track = analyze(request["input_audio"])
    path = Path(request["report"]).with_suffix(".track.npz")
    assert not path.exists()
    np.savez_compressed(path, **track)
    result = render(request, track)
    result.update(pitch_track=str(path), pitch_track_sha256=sha(path), new_pitch_analysis=1)
    with Path(request["report"]).open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
