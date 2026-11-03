"""Bounded cleanup of the first generated replacement material.

The class-specific eligibility rules live beside their implementations. The
entrypoint fixes their order and never regenerates an asset or retries an
unavailable first observation.
"""

from . import generated_instrument_projection, generated_old_instrument, instrumental_material
from ..core import PolicySpec, record_policy

POLICIES = (
    PolicySpec(
        "material.replaced_instrument",
        "material",
        "Validated request semantics replace one instrument with one instrument from a different family; ensembles, voices, and uncertain identities are excluded.",
        "Bound first generated-material facts positively identify both planned instruments with exact supporting quotes.",
        "Remove the planned old instrument from the generated asset, preserving the original residual and sample grid.",
        "Per first-asset pass: one observation if uncached, one binding decision, and at most one SAM cleanup; no generation.",
    ),
    PolicySpec(
        "material.unrequested_singing",
        "material",
        "The replacement requests a named instrument or instrumental accompaniment without voices.",
        "Bound first generated-material facts positively identify unrequested singing and the requested instrument.",
        "Remove singing from the generated asset while keeping the existing residual unchanged.",
        "Per first-asset pass: reuse first facts or obtain one observation, one binding decision, and at most one SAM cleanup.",
    ),
    PolicySpec(
        "material.generated_violin_projection",
        "material",
        "The replacement requests violin, excludes slides/vibrato, and the first asset is incompatible pitched instrumental music.",
        "Existing first-material facts and exact request/fact quotes; no new listening observation.",
        "Transcribe and render the first generated composition with a violin SoundFont voice.",
        "One binding decision and at most one transcription/render; no regeneration, source separation, or retry.",
    ),
)


def _record(ctx, spec, key, before, after):
    audits = ctx.params.get(key, [])
    audit = audits[-1] if audits else {}
    record_policy(
        ctx,
        spec,
        audit.get("status", "changed" if after != before else "retained"),
        subject=str(before),
        changed=after != before,
        audit_key=key,
    )


def clean_generated_material(ctx, plan, placement, asset):
    """Try old-instrument, vocal, then instrument correction on the first asset."""
    if not ctx.params.get("replacement_plan"):
        return asset
    current = POLICIES[0]
    try:
        generated = generated_old_instrument.clean(ctx, plan, placement, asset)
        _record(ctx, POLICIES[0], "replacement_material_old_instrument_cleanup", asset, generated)
        audit = ctx.params["replacement_material_old_instrument_cleanup"][-1]
        facts_unavailable = audit["status"] in {
            "first_material_facts_unavailable",
            "unbound_first_material_facts",
        }
        if generated != asset or facts_unavailable:
            return generated
        current = POLICIES[1]
        generated = instrumental_material.clean(ctx, plan, placement, generated)
        _record(ctx, POLICIES[1], "replacement_material_voice_cleanup", asset, generated)
        if generated != asset:
            return generated
        current = POLICIES[2]
        generated = generated_instrument_projection.project(ctx, plan, placement, generated)
        _record(ctx, POLICIES[2], "replacement_generated_instrument_projection", asset, generated)
        return generated
    except Exception as exc:
        record_policy(
            ctx, current, "error", subject=str(asset), error=f"{type(exc).__name__}: {exc}"
        )
        raise
