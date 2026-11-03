"""Optional editing policies grouped by the audio stage they can correct."""

from dataclasses import asdict

from .core import PolicySpec, record_policy

__all__ = ["PolicySpec", "catalog", "record_policy"]


def catalog() -> list[dict[str, str]]:
    """Return policy descriptions without executing editing or model calls."""
    from . import material, removal, rendering

    return [asdict(spec) for stage in (removal, material, rendering) for spec in stage.POLICIES]
