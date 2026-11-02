"""Reconcile one-frame resampling quantization inside unchanged edit limits."""

import math


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def placement_sample_bounds(
    placement, *, native_frames, native_rate, rendered_frames, sample_rate, diagnostics
):
    """Preserve complete resampled material without relaxing a time window.

    A native duration and separately rounded endpoints can describe one fewer
    output sample than the resampler's ceil-sized array. Only this proven case
    receives a sample-grid correction. Arbitrary long assets still reach the
    existing complete-event check; no sample is removed here.
    """
    start_seconds = float(placement.get("start_time", 0))
    end_seconds = placement.get("end_time")
    start = round(start_seconds * sample_rate)
    end = (
        round(float(end_seconds) * sample_rate)
        if end_seconds is not None
        else start + rendered_frames
    )
    if (
        native_rate == sample_rate
        or rendered_frames - (end - start) != 1
        or placement.get("coverage", "single") != "single"
        or placement.get("contract_version") != 2
        or placement.get("compiled_phase") != "final"
    ):
        return start, end
    expected_frames = (native_frames * sample_rate + native_rate - 1) // native_rate
    duration = native_frames / native_rate
    reported = placement.get("actual_asset_seconds")
    if (
        rendered_frames != expected_frames
        or not _number(reported)
        or abs(reported - duration) > 1e-9
        or end_seconds is None
        or abs((end_seconds - start_seconds) - duration) > 1e-9
    ):
        return start, end
    lo, hi = placement.get("allowed_start"), placement.get("allowed_end")
    if not (_number(lo) and _number(hi) and lo < hi):
        return start, end
    lower, upper = round(lo * sample_rate), round(hi * sample_rate)
    alignment = placement.get("render_alignment")
    if alignment is None:
        at_start = abs(start_seconds - lo) <= 1e-9
        at_end = abs(end_seconds - hi) <= 1e-9
        if at_start == at_end:
            return start, end
        alignment = "start" if at_start else "end"
    if alignment == "start":
        first, last = start, start + rendered_frames
    elif alignment == "end":
        first, last = end - rendered_frames, end
    elif alignment == "center":
        center = (start_seconds + end_seconds) * sample_rate / 2
        first = round(center - rendered_frames / 2)
        last = first + rendered_frames
    else:
        raise ValueError("Unknown compiled sample alignment")
    if first < lower or last > upper:
        raise ValueError("Complete resampled material cannot fit the unchanged allowed window")
    diagnostics.append(
        dict(
            code="resampling_sample_quantization",
            native_frames=native_frames,
            native_sample_rate=native_rate,
            rendered_frames=rendered_frames,
            sample_rate=sample_rate,
            old_start_sample=start,
            old_end_sample=end,
            start_sample=first,
            end_sample=last,
            allowed_start_sample=lower,
            allowed_end_sample=upper,
            alignment=alignment,
            removed_samples=0,
            reason="One-frame ceil/endpoint-rounding difference; retain the full resampled event inside the original allowed window",
        )
    )
    return first, last
