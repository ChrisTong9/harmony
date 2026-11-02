"""Bounded acoustic rhythm verification for requested tempo matching."""

from copy import deepcopy
import hashlib
import subprocess
from pathlib import Path
import numpy as np

from audio_edit.analyzers.focused_audio_review import reference_anchors
from audio_edit.models import AudioEditContext
from audio_edit.plans.add_intent import _rhythm_spacing
from audio_edit.plans.remove_sound_plan import RemoveSoundPlan
from audio_edit.plans.temporal_boundaries import select_event
from audio_edit.runners.samaudio_runner import SAMAudioRunner
from audio_edit.utils.execution_outcome import open_decoded_audio


def measured_onsets(samples, sample_rate, offset=0.0):
    """Return actual transient times in a stable run; never synthesize a beat grid."""
    x = np.asarray(samples, dtype="float64")
    if x.ndim == 2:
        x = x.mean(axis=1)
    hop = max(1, round(sample_rate * 0.01))
    size = max(4, round(sample_rate * 0.04))
    if len(x) < size + 10 * hop or not np.isfinite(x).all() or np.max(np.abs(x)) < 1e-6:
        raise ValueError("No usable rhythmic waveform")
    frames = np.lib.stride_tricks.sliding_window_view(x, size)[::hop]
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(size), axis=1))
    flux = np.maximum(np.diff(np.log1p(spectrum), axis=0), 0).mean(axis=1)
    center = float(np.median(flux))
    mad = float(np.median(np.abs(flux - center)))
    threshold = max(center + 2.5 * mad, float(flux.max()) * 0.25, 1e-7)
    candidates = (
        np.flatnonzero(
            (flux[1:-1] > flux[:-2]) & (flux[1:-1] >= flux[2:]) & (flux[1:-1] > threshold)
        )
        + 1
    )
    selected = []
    for index in sorted(candidates, key=lambda i: -flux[i]):
        if all(abs(index - j) * hop / sample_rate >= 0.15 for j in selected):
            selected.append(int(index))
    selected.sort()
    times = [offset + ((i + 1) * hop + size / 2) / sample_rate for i in selected]
    best = []
    for first in range(len(times)):
        for last in range(first + 5, len(times) + 1):
            seq = times[first:last]
            gaps = np.diff(seq)
            step = float(np.median(gaps))
            if not 0.15 <= step <= 2 or seq[-1] - seq[0] < 1.0:
                continue
            if np.max(np.abs(gaps - step)) > step * 0.15:
                continue
            if len(seq) > len(best):
                best = seq
    if not best:
        raise ValueError(
            "No stable sequence of at least five observed transients; tempo remains unresolved"
        )
    return dict(
        onsets_seconds=best,
        approximate=True,
        evidence="Measured attacks in the isolated requested source: 40 ms spectral-flux frames, 10 ms hop; at least five actual peaks with intervals within 15% of their median; no generated beat positions.",
        method="isolated_source_spectral_flux",
        all_candidate_onsets=times,
        median_interval_seconds=float(np.median(np.diff(best))),
        threshold=threshold,
    )


def complete_rhythm_evidence(ctx, analysis, requests, *, max_references=3):
    """One source extraction per located rhythm, after perception review.

    The original input and edit intent are unchanged. All extracted audio and
    requests remain available. This supplies estimated acoustic beat evidence;
    source recognition and the reference onset remain perceptual observations.
    """
    result = deepcopy(analysis or {})
    events = {e["event_id"]: e for e in result.get("events", []) if e.get("event_id")}
    audits = []
    seen = set()
    for anchor in reference_anchors(requests):
        if anchor.get("measurement") != "beat_onsets" or anchor.get("reference_source") == "visual":
            continue
        try:
            event, _ = select_event(anchor, events, best_effort=False)
            key = event["event_id"]
            if key in seen:
                continue
            seen.add(key)
            with open_decoded_audio(ctx.input_audio) as source:
                duration = len(source) / source.samplerate
            previous = deepcopy(event.get("rhythm"))

            if (
                isinstance(previous, dict)
                and previous.get("method") == "isolated_source_spectral_flux"
            ):
                _rhythm_spacing(anchor, events, duration)
                continue
            event.pop("rhythm", None)
            if len(audits) >= max_references:
                audits.append(
                    dict(
                        event_id=key,
                        status="unresolved",
                        error="Bounded rhythm measurement limit reached",
                        original_model_rhythm=previous,
                    )
                )
                continue
            audit = dict(
                event_id=key,
                reference_text=anchor.get("reference_text"),
                status="started",
                original_model_rhythm=previous,
            )
            audits.append(audit)
            extraction = AudioEditContext(
                "Measure the existing rhythmic source", input_audio=ctx.input_audio
            )
            plan = RemoveSoundPlan(
                backend="samaudio",
                mode="text",
                result_mode="result",
                text_prompt=anchor.get("reference_text") or event["event"],
                notes="Analysis-only extraction of the existing referenced rhythm; no edit to original",
            )
            extraction = SAMAudioRunner().run_remove(extraction, plan)
            path = Path(extraction.output_audio)
            audit.update(
                reference_audio=str(path.resolve()),
                reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                separation_calls=extraction.params.get("separation_calls", []),
            )
            with open_decoded_audio(str(path)) as source:
                rate = source.samplerate
                start = max(0.0, event["start_time"] - 0.5)
                end = min(len(source) / rate, start + 12.0)
                source.seek(round(start * rate))
                data = source.read(round((end - start) * rate), dtype="float32", always_2d=True)
            rhythm = measured_onsets(data, rate, start)
            rhythm["source_audio"] = str(path.resolve())
            rhythm["source_audio_sha256"] = audit["reference_sha256"]
            event["rhythm"] = rhythm
            audit.update(
                status="estimated", rhythm=deepcopy(rhythm), search_start=start, search_end=end
            )
        except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
            if audits and audits[-1].get("status") == "started":
                audits[-1].update(status="unresolved", error=f"{type(exc).__name__}: {exc}")
            else:
                audits.append(dict(status="unresolved", error=f"{type(exc).__name__}: {exc}"))
    ctx.params["rhythm_grounding"] = audits

    return result
