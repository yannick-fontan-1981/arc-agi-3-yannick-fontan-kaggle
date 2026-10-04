"""Mechanical, read-only execution of the active normative SRC TODO projection."""

from __future__ import annotations

from typing import Mapping

from agents.yf_arc3_v5.drm.meaning import project_declared_alternatives
from agents.yf_arc3_v5.drm.registry import DrmRegistry
from agents.yf_arc3_v5.logos.types import FrozenMap
from agents.yf_arc3_v5.src.ast import ExpressionKind, StatementKind
from agents.yf_arc3_v5.src.compiler import IrWorkflow


def project_todo_alerts(
    *,
    workflow: IrWorkflow,
    drm: DrmRegistry,
    descriptive_facts: Mapping[str, object],
) -> tuple[FrozenMap, ...]:
    """Visit exactly the alternatives named by the compiled SRC/DRM pair.

    This reads no oracle and changes no scheduler, TERM, CLAIM or action state.
    Python only checks the source shape and transports DRM-authored alert facts.
    """

    if len(workflow.steps) != 1:
        raise ValueError("TODO projection must have one passive SRC step")
    statements = workflow.steps[0].statements
    if len(statements) != 2 or tuple(item.kind for item in statements) != (
        StatementKind.CALL,
        StatementKind.STOP,
    ):
        raise ValueError("TODO projection may only CALL and STOP")
    call, stop = statements
    if call.expression is None or call.expression.value != "control.project_declared_alternatives":
        raise ValueError("TODO SRC must call the declared DRM projection")
    args = {item.name: item.value for item in call.expression.arguments}
    if set(args) != {"selection_policy_ref", "alternative_refs", "descriptive_facts"}:
        raise ValueError("TODO SRC projection argument contract changed")
    policy_expr = args["selection_policy_ref"]
    refs_expr = args["alternative_refs"]
    facts_expr = args["descriptive_facts"]
    if (
        policy_expr.kind is not ExpressionKind.STRING
        or refs_expr.kind is not ExpressionKind.LIST
        or refs_expr.items
        or facts_expr.kind is not ExpressionKind.REFERENCE
        or facts_expr.value != "descriptive_todo_facts"
        or stop.expression is None
        or stop.expression.value != call.result_binding
    ):
        raise ValueError("TODO SRC projection has an unsupported source shape")
    projected = project_declared_alternatives(
        drm=drm,
        selection_policy_ref=str(policy_expr.value),
        alternative_refs=(),
        descriptive_facts=descriptive_facts,
    )
    return tuple(
        values
        for ref in projected.alternative_refs
        if isinstance((values := projected.alternative_facts.get(ref)), FrozenMap)
        and values.get("applicable") is True
    )
