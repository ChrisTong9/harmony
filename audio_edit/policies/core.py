"""Shared descriptions and execution receipts for optional editing policies."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PolicySpec:
    """Describe a policy; its stage implementation enforces conditions and limits."""

    policy_id: str
    stage: str
    condition: str
    evidence: str
    action: str
    budget: str


def record_policy(ctx, spec: PolicySpec, status: str, *, subject=None, **details: Any) -> None:
    """Append a snapshot of a policy decision using JSON-compatible detail values.

    Recording never selects or executes a strategy. Stage functions call this
    after evaluating a condition or receiving the result of an existing action.
    """
    receipt = {"policy": spec.policy_id, "stage": spec.stage, "status": status}
    if subject is not None:
        receipt["subject"] = subject
    if details:
        receipt["details"] = details
    ctx.params.setdefault("policy_trace", []).append(deepcopy(receipt))
