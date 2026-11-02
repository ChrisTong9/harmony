"""Local-only DSP with auditable gain constraints and original-time coordinates."""
from __future__ import annotations
import math
import torch


def active_rms(audio, sr):
    if audio.numel() == 0:
        return 0.
    x = audio.float()
    n = max(1, round(.02 * sr))
    padded = torch.nn.functional.pad(x, (0, (-x.shape[-1]) % n))
    power = padded.square().reshape(x.shape[0], -1, n).mean((0, 2))


    threshold = float(power.max()) * .01
    active = power[power >= threshold]
    return math.sqrt(float(active.mean())) if active.numel() else 0.


def finite_audio(audio):
    if audio.ndim != 2 or not audio.numel() or not torch.isfinite(audio).all():
        raise ValueError('Audio must be nonempty, finite channels x samples')


def bounded_delta(original, delta, diagnostics, label='added_layer', sample_rate=48000,
                  return_envelope=False):
    """Transparent unless needed; restrict ONLY the delta in short local frames.

    Each frame's two boundary gains are <= that frame's safe gain. Linear
    interpolation therefore stays safe without hard clipping or a global limiter.
    The returned scalar is an energy-ratio summary, NOT a uniform applied gain.
    """
    finite_audio(original)
    finite_audio(delta)
    if original.shape != delta.shape or sample_rate <= 0:
        raise ValueError('Delta shape and sample rate must match original')
    if float(original.abs().max()) > 1.000001:
        raise ValueError('Original exceeds full scale; preservation and no-overflow conflict')
    requested = original + delta
    if float(requested.abs().max()) <= 1.:
        return (requested, 1., delta.new_ones(delta.shape[-1])) if return_envelope else (requested, 1.)
    cap = torch.ones_like(delta)
    positive, negative = delta > 0, delta < 0
    cap[positive] = (1-original[positive])/delta[positive]
    cap[negative] = (-1-original[negative])/delta[negative]
    cap = cap.amin(0).clamp(0., 1.)
    width = max(2, round(.005*sample_rate))
    padded = torch.nn.functional.pad(cap, (0, (-cap.numel()) % width), value=1.)
    frames = padded.reshape(-1,width).amin(1)
    knots = torch.cat([frames[:1], torch.minimum(frames[:-1],frames[1:]), frames[-1:]])
    ramp = torch.linspace(0,1,width,device=delta.device,dtype=delta.dtype)
    gain = (knots[:-1,None]*(1-ramp)+knots[1:,None]*ramp).flatten()[:cap.numel()]
    gain = torch.where(gain < 1, gain*.999999, gain)
    actual = delta * gain
    energy = float(delta.double().square().sum())
    scale = math.sqrt(float(actual.double().square().sum())/energy) if energy else 1.
    active = delta.abs().amax(0) > 0
    affected = active & (gain < 1)
    diagnostics.append(dict(code='headroom_limited', layer=label,
        requested_delta_scale=1., actual_delta_scale=scale, delta_scale_kind='energy_ratio',
        scale_min=float(gain[active].min()) if active.any() else 1.,
        scale_max=float(gain[active].max()) if active.any() else 1.,
        affected_seconds=float(affected.sum())/sample_rate,
        affected_fraction=float(affected.sum())/max(1,int(active.sum())), frame_seconds=width/sample_rate,
        policy='local_delta_only', reason='Preserve original; smooth local delta reduction only where headroom is insufficient'))
    output = original + actual
    if float(output.abs().max()) > 1.000001:
        raise ValueError('Local delta envelope failed headroom post-check')
    return (output, scale, gain) if return_envelope else (output, scale)


def edge_envelope(length, sr, seconds=.005, device=None, dtype=None):
    mask = torch.ones(length, device=device, dtype=dtype)
    n = min(round(seconds*sr), length//2)
    if n:
        mask[:n] = torch.linspace(0, 1, n, device=device, dtype=dtype)
        mask[-n:] = torch.linspace(1, 0, n, device=device, dtype=dtype)
    return mask


def scope_envelope(length, sr, placements, device=None, dtype=None):
    mask = torch.zeros(length, device=device, dtype=dtype)
    for p in placements:
        start, end = round(float(p['start_time'])*sr), round(float(p['end_time'])*sr)
        if start < 0 or end > length or end <= start:
            raise ValueError('Local edit scope is invalid; cannot clip it')
        mask[start:end] = torch.maximum(mask[start:end], edge_envelope(end-start, sr, device=device, dtype=dtype))
    return mask


def prepare_snippet(audio, sr, length, coverage, diagnostics):
    finite_audio(audio)
    if length < 1:
        raise ValueError('Zero-length sound is not an executed edit')
    if audio.shape[-1] > length:
        if coverage == 'continuous':
            diagnostics.append(dict(code='continuous_segment_selected', generated_samples=audio.shape[-1], used_samples=length,
                                    reason='Explicit continuous coverage, not a claimed complete discrete event'))
            audio = audio[:, :length]
        else:

            amplitude = audio.abs().amax(0)
            active = torch.nonzero(amplitude > max(float(amplitude.max())*.003, 1e-7)).flatten()
            if not active.numel():
                raise ValueError('Generated sound is silent')
            a, b = int(active[0]), int(active[-1])+1
            margin = round(.005*sr)
            a, b = max(0, a-margin), min(audio.shape[-1], b+margin)
            if b-a > length:
                raise ValueError('Complete active generated event exceeds placement; refusing head truncation')
            diagnostics.append(dict(code='silent_padding_removed', source_start=a/sr, source_end=b/sr))
            audio = audio[:, a:b]
    if audio.shape[-1] < length:
        missing = length-audio.shape[-1]
        if coverage == 'continuous' and missing > max(2, round(.01*sr)):
            raise ValueError('Generated continuous sound is shorter than required coverage')
        diagnostics.append(dict(code='trailing_padding', samples=missing,
                                reason='Preserve complete event and planned placement extent'))
        audio = torch.nn.functional.pad(audio, (0, missing))
    return audio.clone()


def scene_gain(clip, original, sr, start, end, features, diagnostics, reference=None):
    active = active_rms(clip, sr)
    peak = float(clip.abs().max())
    if active < 1e-7 or peak < 1e-7:
        raise ValueError('Generated placement is effectively silent')
    local = active_rms(original[:, max(0,start):min(end,original.shape[-1])], sr)
    context = active_rms(original[:, max(0,start-sr):min(original.shape[-1],end+sr)], sr)
    scene = local if local >= 1e-4 else context
    role = features.get('layer_role') or 'foreground'


    scene_level_is_authoritative = 'relative_level_db' in features
    relative = features.get('relative_level_db')
    is_add = scene_level_is_authoritative and reference is None
    default_relative = 6. if is_add else (-12 if role == 'background' else -3)
    relative = float(relative if relative is not None else default_relative)
    if not math.isfinite(relative) or not -36 <= relative <= 12:
        raise ValueError('Relative scene gain must be bounded to [-36,12] dB')
    reference_rms = active_rms(reference, sr) if reference is not None else 0.
    basis = 'local_scene' if local >= 1e-4 else 'neighboring_context'
    if reference_rms > 1e-5:

        scene = max(reference_rms*.8, scene*.5)
        basis = 'old_target_and_residual_masking'
    if scene < 1e-4:
        fallback = float(features.get('target_rms_dbfs', -36 if role == 'background' and not is_add else -30))
        fallback = max(-48., min(-24., fallback))
        target = 10**(fallback/20)
        basis = 'bounded_silent_scene_fallback'
    else:
        target = scene * 10**(relative/20)
    level = float(features.get('mix_level', 1.))
    if not math.isfinite(level) or not 0 < level <= 4:
        raise ValueError('mix_level must be finite in (0,4]')
    applied_level = 1. if scene_level_is_authoritative else level
    if scene_level_is_authoritative and level != 1.:
        diagnostics.append(dict(code='legacy_mix_level_ignored', supplied_mix_level=level,
            applied_mix_level=1., relative_level_db=relative,
            reason='relative_level_db (or its operation default) is the sole scene-level target; do not apply legacy mix_level a second time'))
    unfloored_target = target * applied_level
    floor_db = features.get('minimum_added_rms_dbfs')
    if floor_db is not None and (isinstance(floor_db, bool) or not math.isfinite(float(floor_db)) or not -48 <= float(floor_db) <= -18):
        raise ValueError('minimum_added_rms_dbfs must be finite and within [-48,-18]')
    configured_floor_db = floor_db
    floor = 10**(float(floor_db)/20) if floor_db is not None else 0.


    if (is_add and features.get('relative_level_db') is None
            and basis == 'local_scene' and local < context * math.sqrt(.01)
            and floor > 0):
        bounded_context = min(context, floor / 10**(relative/20))
        if bounded_context > scene:
            previous_scene = scene
            scene = bounded_context
            unfloored_target = scene * 10**(relative/20)
            basis = 'bounded_neighboring_context'
            diagnostics.append(dict(code='quiet_gap_default_reference',
                original_local_reference=previous_scene, neighboring_active_rms=context,
                selected_reference=scene, relative_power_gate=.01,
                added_target_cap_dbfs=float(configured_floor_db),
                reason='Unspecified ADD level in an inactive gap; use bounded nearby activity'))


    if (scene_level_is_authoritative
            and basis != 'bounded_silent_scene_fallback' and floor > unfloored_target):
        floor = unfloored_target
        floor_db = 20*math.log10(floor)
        diagnostics.append(dict(code='audibility_floor_capped_by_relative_level',
            configured_minimum_added_rms_dbfs=configured_floor_db,
            effective_minimum_added_rms_dbfs=floor_db, relative_level_db=relative,
            reason='Preserve declared relative balance against the measured source'))


    requested_target = max(unfloored_target, floor)
    requested = requested_target / active
    max_gain = features.get('max_generated_gain_db')
    if max_gain is not None and (isinstance(max_gain, bool) or not math.isfinite(float(max_gain)) or not 0 <= float(max_gain) <= 80):
        raise ValueError('Explicit generated gain cap must be finite and within [0,80] dB')
    gain_cap = 10**(float(max_gain)/20) if max_gain is not None else float('inf')
    gain = min(requested, gain_cap)
    scaled = clip*gain


    finite_audio(scaled)
    diagnostics.append(dict(code='placement_gain', basis=basis, role=role,
        local_active_rms=local, neighboring_active_rms=context, old_target_active_rms=reference_rms,
        prepared_active_rms=active, prepared_peak=peak, requested_gain=requested,
        bounded_gain=gain, relative_level_db=relative,
        gain_control='scene_relative' if scene_level_is_authoritative else 'legacy_mix_level',
        supplied_mix_level=level, applied_mix_level=applied_level,
        unfloored_active_rms=unfloored_target, minimum_added_rms_dbfs=floor_db,
        configured_minimum_added_rms_dbfs=configured_floor_db,
        audibility_floor_applied=unfloored_target < floor,
        requested_active_rms=requested_target,
        applied_active_rms_before_headroom=active_rms(scaled,sr),
        reference_active_rms=scene, peak_handling='preserve_float_output_no_master_normalization'))
    return scaled


def audit_added_level(delta, sr, placement, diagnostics):
    """Measure the added component after effects and its requested gain."""
    floor_db = placement.get('minimum_added_rms_dbfs')
    if floor_db is None:
        return
    rms = active_rms(delta, sr)
    actual_db = 20*math.log10(max(rms, 1e-12))
    effective_floor = float(floor_db)
    met = actual_db >= effective_floor-.05
    diagnostics.append(dict(code='added_layer_level' if met else 'audibility_floor_limited',
        minimum_added_rms_dbfs=effective_floor, requested_floor_before_mix_dbfs=float(floor_db),
        actual_active_rms_dbfs=actual_db,
        actual_peak=float(delta.abs().max()), floor_met=met, tolerance_db=.05,
        reason='Active-sound floor met after rendering' if met else
            'Actual layer below its target; see explicit generated gain limits'))


def mix_added_layers(base, sr, layers, diagnostics):
    """Preserve original gain; protect default continuous context additions.

    Explicit level requests and independent/discrete additions retain their
    existing policy. Context accompaniments with an unspecified level share
    one local envelope when their raw sum would exceed playback full scale.
    """
    layers = sorted(layers, key=lambda item: (item[0], item[1]))
    envelope = None
    limited_audit = []
    eligible = [p.get('headroom_policy') == 'default_context_local_delta' for _, _, p, _ in layers]
    if any(eligible):
        fixed = base.clone()
        delta = torch.zeros_like(base)
        for protect, (start, _, _, clip) in zip(eligible, layers):
            (delta if protect else fixed)[:, start:start+clip.shape[-1]] += clip


        if float(fixed.abs().max()) <= 1. and float((fixed+delta).abs().max()) > 1.:
            _, _, envelope = bounded_delta(fixed, delta, limited_audit,
                label='default_context_layers', sample_rate=sr, return_envelope=True)
        elif float(fixed.abs().max()) > 1.:
            diagnostics.append(dict(code='context_headroom_unavailable',
                reason='Retained audio already exceeds full scale; preserve its requested gain',
                original_gain=1.))
    output = base.clone()
    for protect, (start, index, p, clip) in zip(eligible, layers):
        end = start+clip.shape[-1]
        scale = 1.
        layer_audit = []
        if protect and envelope is not None:
            requested_energy = float(clip.double().square().sum())
            clip = clip * envelope[start:end]
            scale = math.sqrt(float(clip.double().square().sum()) / requested_energy) if requested_energy else 1.
            layer_audit.extend(dict(item, layer=index, actual_delta_scale=scale) for item in limited_audit)
        output[:, start:end] += clip
        audit_added_level(clip, sr, p, layer_audit)
        target = p.get('level_target')
        if target:
            actual = active_rms(clip, sr)
            requested = target['active_rms']
            error_db = 20*math.log10(max(actual,1e-15)/max(requested,1e-15))
            reference = target['reference_active_rms']
            retained = active_rms(base[:, start:end], sr)
            layer_audit.append(dict(code='added_layer_balance' if abs(error_db) <= 1. else 'added_layer_balance_limited',
                requested_relative_level_db=target['relative_level_db'], basis=target['basis'],
                target_active_rms=requested, actual_active_rms=actual,
                actual_relative_to_reference_db=20*math.log10(max(actual,1e-15)/reference) if reference > 1e-7 else None,
                actual_relative_to_retained_audio_db=20*math.log10(max(actual,1e-15)/retained) if retained > 1e-7 else None,
                target_error_db=error_db, target_met=abs(error_db) <= 1.,
                measurement='Rendered layer after local headroom control; original gain unchanged'))
        diagnostics.append(dict(code='timeline_placement', index=index,
            timeline_op=p['timeline_op'], actual_start=start/sr, actual_end=end/sr,
            headroom_scale=scale, headroom_policy='local_delta_only' if protect and envelope is not None else 'no_master_normalization',
            original_gain=1., added_final_gain=scale,
            added_final_gain_kind='energy_ratio' if protect and envelope is not None else 'scalar', constraints=layer_audit))
    finite_audio(output)
    peak = float(output.abs().max())
    if peak > 1.:
        diagnostics.append(dict(code='mix_peak_over_full_scale', peak=peak,
            peak_dbfs=20*math.log10(peak), samples_over_full_scale=int((output.abs()>1.).sum()),
            policy='preserve_float_pcm', master_gain=1., normalization_applied=False,
            reason='Raw sum retained by policy; playback or integer PCM conversion may clip'))
    return output


def render_timeline(original, sr, prepared, diagnostics):
    """Map original coordinates; preserve the original gain and planned additions."""
    finite_audio(original)
    total = original.shape[-1]
    inserts, overlays = [], []
    for index, (p, clip) in enumerate(prepared):
        start = round(float(p['start_time'])*sr)
        op = p.get('timeline_op', 'overlay')
        if op in ('insert', 'prepend', 'append'):
            if not 0 <= start <= total or (op == 'prepend' and start != 0) or (op == 'append' and start != total):
                raise ValueError('Invalid insertion anchor')
            inserts.append((start,index,p,clip))
        elif op == 'overlay':
            if start < 0 or (start+clip.shape[-1] > total and p.get('overflow_policy') != 'extend'):
                raise ValueError('Overlay exceeds original timeline without extend policy')
            overlays.append((start,index,p,clip))
        else:
            raise ValueError('Unknown timeline operation')
    inserts.sort(key=lambda x:(x[0],x[1]))
    for start, _, _, clip in overlays:
        if any(start < anchor < start+clip.shape[-1] for anchor, *_ in inserts):
            raise ValueError('Overlay crosses an insertion seam; split the plan explicitly')
    chunks, mapping, layers, cursor, outpos = [], [], [], 0, 0
    for position, (anchor, index, p, clip) in enumerate(inserts):
        if anchor > cursor:
            chunks.append(original[:, cursor:anchor])
            mapping.append(dict(original_start=cursor, original_end=anchor, output_start=outpos))
            outpos += anchor-cursor
        first_at_anchor = position == 0 or inserts[position-1][0] != anchor
        last_at_anchor = position+1 == len(inserts) or inserts[position+1][0] != anchor
        gap_seconds = float(p.get('insert_gap_before_seconds', 0.))
        if not math.isfinite(gap_seconds) or gap_seconds < 0:
            raise ValueError('Inserted gap must be finite and nonnegative')
        gap_length = round(gap_seconds*sr)
        if gap_length:
            gap = original.new_zeros(original.shape[0], gap_length)
            n = min(round(.005*sr), gap_length)
            if first_at_anchor and anchor > 0 and n:
                gap[:, :n] = original[:, anchor-1:anchor] * torch.linspace(1,0,n,device=clip.device)
            chunks.append(gap)
            diagnostics.append(dict(code='inserted_gap', index=index,
                original_anchor=anchor/sr, actual_start=outpos/sr,
                actual_end=(outpos+gap_length)/sr, requested_seconds=gap_seconds,
                reason='Explicit sequence spacing, not asset padding or moved original audio'))
            outpos += gap_length

        base = torch.zeros_like(clip)
        n = min(round(.005*sr), clip.shape[-1]//2)
        if n:
            if first_at_anchor and not gap_length and anchor > 0:
                base[:, :n] = original[:, anchor-1:anchor] * torch.linspace(1,0,n,device=clip.device)
            if last_at_anchor and anchor < total:
                base[:, -n:] = original[:, anchor:anchor+1] * torch.linspace(0,1,n,device=clip.device)
        chunks.append(base)
        layers.append((outpos, index, p, clip))
        outpos += clip.shape[-1]
        cursor = anchor
    if cursor < total:
        chunks.append(original[:, cursor:])
        mapping.append(dict(original_start=cursor, original_end=total, output_start=outpos))
    output = torch.cat(chunks, dim=-1).clone() if chunks else original.clone()
    for start,index,p,clip in overlays:

        shifted = start + sum(c.shape[-1] + round(float(ip.get('insert_gap_before_seconds', 0.))*sr)
                              for a,_,ip,c in inserts if a <= start)
        end = shifted+clip.shape[-1]
        if end > output.shape[-1]:
            output = torch.nn.functional.pad(output,(0,end-output.shape[-1]))
        layers.append((shifted, index, p, clip))
    output = mix_added_layers(output, sr, layers, diagnostics)
    diagnostics.append(dict(code='original_timeline_mapping', sample_rate=sr, segments=mapping,
                            original_samples=total, output_samples=output.shape[-1]))
    return output
