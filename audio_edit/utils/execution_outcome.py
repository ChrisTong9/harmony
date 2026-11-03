"""Report actual waveform change separately from file production and quality."""

import math
from contextlib import contextmanager
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
import soundfile as sf


@contextmanager
def open_decoded_audio(path):
    """Decode supported compressed audio without normalization or timeline changes."""
    try:
        source = sf.SoundFile(path)
    except RuntimeError:
        with TemporaryDirectory(prefix="audio-outcome-") as directory:
            decoded = Path(directory) / "decoded.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(path),
                    "-vn",
                    "-c:a",
                    "pcm_f32le",
                    str(decoded),
                ],
                check=True,
                capture_output=True,
            )
            with sf.SoundFile(decoded) as source:
                yield source
        return
    with source:
        yield source


def compare_waveforms(original_path, output_path):
    with open_decoded_audio(original_path) as source, open_decoded_audio(output_path) as output:
        aligned = (source.samplerate, source.channels, len(source)) == (
            output.samplerate,
            output.channels,
            len(output),
        )
        if not len(output):
            raise ValueError("Empty output audio")
        count, changed, delta_energy, source_energy, peak_delta = 0, 0, 0.0, 0.0, 0.0
        for block in output.blocks(blocksize=65536, dtype="float32", always_2d=True):
            if not np.isfinite(block).all():
                raise ValueError("Non-finite output waveform")
            if aligned:
                before = source.read(len(block), dtype="float32", always_2d=True)
                delta = block.astype("float64") - before
                changed += int(np.count_nonzero(delta))
                delta_energy += float(np.sum(delta * delta))
                source_energy += float(np.sum(before.astype("float64") ** 2))
                peak_delta = max(peak_delta, float(np.abs(delta).max()))
                count += delta.size
        return dict(
            no_op=aligned and changed == 0,
            aligned=aligned,
            original_frames=len(source),
            output_frames=len(output),
            sample_rate=output.samplerate,
            changed_samples=changed if aligned else None,
            max_abs_delta=peak_delta if aligned else None,
            delta_rms=math.sqrt(delta_energy / max(1, count)) if aligned else None,
            delta_to_original_db=10
            * math.log10(max(delta_energy, 1e-30) / max(source_energy, 1e-30))
            if aligned
            else None,
        )


def mark_basic_execution(ctx):
    """Record recovery independently of whether the recovered edit succeeds."""
    ctx.fallback_used = True
    ctx.execution_path = "basic"
    if ctx.recovery_trigger is None:
        ctx.recovery_trigger = dict(
            ctx.params.get("primary_execution_error")
            or {"reason": "Primary execution did not complete; no exception details supplied"}
        )
    return ctx


def execution_exit_code(ctx):
    """A recovered edit may succeed; unchanged or incomplete edits do not."""
    return (
        0 if ctx.execution_status in {"applied", "limited", "estimated", "listening_ready"} else 1
    )


def annotate_outcome(ctx, original_path):
    if ctx.fallback_used and ctx.execution_path != "basic":
        if ctx.execution_status not in ("unresolved", "fallback"):
            ctx.execution_status = "fallback"
        return ctx
    if ctx.params.get("execution_outcome"):
        return ctx
    audit = compare_waveforms(original_path, ctx.output_audio)
    ctx.params["execution_outcome"] = audit
    if audit["no_op"]:
        ctx.execution_status = "no_op"
        ctx.execution_error = "Output waveform is unchanged; edit was not applied."
        ctx.warnings.append(dict(code="no_op", reason=ctx.execution_error))
    else:
        headroom_limited = any(w.get("code") == "headroom_limited" for w in ctx.warnings)
        audibility_limited = any(w.get("code") == "audibility_floor_limited" for w in ctx.warnings)
        balance_limited = any(w.get("code") == "added_layer_balance_limited" for w in ctx.warnings)
        audit["added_layer_balance_limited"] = balance_limited
        audit["mix_peak_reports"] = [
            dict(w) for w in ctx.warnings if w.get("code") == "mix_peak_over_full_scale"
        ]
        audit["volume_peak_reports"] = [
            dict(w) for w in ctx.warnings if w.get("code") == "volume_peak_over_full_scale"
        ]
        audit["headroom_limited"] = headroom_limited
        audit["audibility_floor_limited"] = audibility_limited
        audit["constraint_details"] = [
            dict(w, energy_change_db=20 * math.log10(max(w.get("actual_delta_scale", 1.0), 1e-12)))
            if w.get("code") == "headroom_limited"
            else dict(w)
            for w in ctx.warnings
            if w.get("code")
            in ("headroom_limited", "audibility_floor_limited", "added_layer_balance_limited")
        ]
        ctx.execution_status = (
            "limited" if headroom_limited or audibility_limited or balance_limited else "applied"
        )
        if ctx.params.get("temporal_resolution_status") in ("approximate", "estimated"):
            audit["headroom_limited"] = headroom_limited
            ctx.execution_status = "estimated"
        if audit["aligned"] and (audit["delta_rms"] < 1e-5 or audit["delta_to_original_db"] < -60):
            ctx.warnings.append(
                dict(
                    code="weak_change",
                    reason="Nonzero waveform change is small; retained, not dropped",
                    delta_rms=audit["delta_rms"],
                    delta_to_original_db=audit["delta_to_original_db"],
                )
            )
    basic = ctx.params.get("basic_execution", {})
    delivery = ctx.params.get("basic_execution_delivery", {})
    actions = basic.get("actions", [])
    failures = [action for action in actions if action.get("completed") is False]
    delivery_error = basic.get("delivery_note") or delivery.get("note")
    if failures or delivery_error:
        audit["completed_actions"] = sum(action.get("completed") is True for action in actions)
        audit["failed_actions"] = len(failures)
        ctx.execution_status = "failed" if audit["no_op"] else "partial"
        ctx.execution_error = delivery_error or (
            f"{len(failures)} of {len(actions)} basic execution actions failed: "
            + failures[0].get("reason", "No failure details recorded")
        )
        ctx.warnings.append(dict(code=ctx.execution_status, reason=ctx.execution_error))
    return ctx


def removed_component_reference(original_path, residual_path, output_path):
    """Reuse the component actually removed; no second separation or timeline edit."""
    import hashlib
    import soundfile as sf
    from pathlib import Path

    digest = hashlib.sha256()
    with (
        open_decoded_audio(original_path) as original,
        open_decoded_audio(residual_path) as residual,
    ):
        if (original.samplerate, original.channels, original.frames) != (
            residual.samplerate,
            residual.channels,
            residual.frames,
        ):
            raise ValueError("Removed-source reference requires identical sample grids")
        with sf.SoundFile(
            output_path,
            mode="w",
            samplerate=original.samplerate,
            channels=original.channels,
            subtype="FLOAT",
            format="WAV",
        ) as out:
            while True:
                a = original.read(65536, dtype="float32", always_2d=True)
                if not len(a):
                    break
                b = residual.read(len(a), dtype="float32", always_2d=True)
                out.write(a - b)
    with Path(output_path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return dict(
        path=str(Path(output_path).resolve()),
        sha256=digest.hexdigest(),
        method="original minus executed residual on identical sample grid",
        quality="unreviewed; separation leakage may remain",
    )
