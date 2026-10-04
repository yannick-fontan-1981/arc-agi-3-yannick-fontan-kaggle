"""Source-attested mechanical executor for one declared temporal tracking CALL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from agents.yf_arc3_v5.capabilities.contracts import (
    ComponentExtractionResult,
    RevisionedCapabilityRequest,
    TemporalTrackingInput,
    TemporalTrackingResult,
)
from agents.yf_arc3_v5.capabilities.registry import CapabilityRegistry
from agents.yf_arc3_v5.src.ast import ExpressionKind, StatementKind
from agents.yf_arc3_v5.src.compiler import CompiledModule


@dataclass(frozen=True, slots=True)
class CompiledTemporalTracking:
    workflow_ref: str
    src_source_hash: str
    step_id: str

    def run(
        self,
        *,
        descriptive_components: ComponentExtractionResult,
        frame_ref: str,
        previous_tracking: Mapping[str, object] | None,
        action_ref: str,
        reset_correspondence: bool,
        input_revision: int,
        capabilities: CapabilityRegistry,
    ) -> TemporalTrackingResult:
        measured = capabilities.invoke(
            RevisionedCapabilityRequest(
                capability_id="capability.temporal_component_tracking",
                input_revision=input_revision,
                payload=TemporalTrackingInput.model_validate({
                    "current": descriptive_components,
                    "frame_ref": frame_ref,
                    "previous": previous_tracking,
                    "action_ref": action_ref,
                    "reset_correspondence": reset_correspondence,
                }),
            )
        ).output
        if not isinstance(measured, TemporalTrackingResult):
            raise RuntimeError("compiled tracking output type changed")
        return measured


def compile_temporal_tracking(
    module: CompiledModule, workflow_ref: str
) -> CompiledTemporalTracking:
    workflow = next((item for item in module.workflows if item.id == workflow_ref), None)
    if workflow is None:
        raise ValueError("compiled tracking workflow is absent")
    if set(binding.name for binding in workflow.inputs) != {
        "descriptive_components", "frame_ref", "previous_tracking", "action_ref", "reset_correspondence",
    } or len(workflow.steps) != 1:
        raise ValueError("compiled tracking input or step topology changed")
    step = workflow.steps[0]
    if step.id != workflow.entry_step_id or len(step.statements) != 2:
        raise ValueError("compiled tracking entry topology changed")
    call, stop = step.statements
    if (
        call.kind is not StatementKind.CALL
        or call.expression is None
        or call.expression.value != "capability.temporal_component_tracking"
        or call.result_binding != "result"
        or stop.kind is not StatementKind.STOP
        or stop.expression is None
        or stop.expression.value != "result"
    ):
        raise ValueError("compiled tracking declared call or result changed")
    arguments = call.expression.arguments
    if {argument.name: argument.value.value for argument in arguments} != {
        "current": "descriptive_components",
        "frame_ref": "frame_ref",
        "previous": "previous_tracking",
        "action_ref": "action_ref",
        "reset_correspondence": "reset_correspondence",
    } or any(argument.value.kind is not ExpressionKind.REFERENCE for argument in arguments):
        raise ValueError("compiled tracking call binding changed")
    return CompiledTemporalTracking(
        workflow_ref=workflow_ref,
        src_source_hash=module.source_hash,
        step_id=step.id,
    )
