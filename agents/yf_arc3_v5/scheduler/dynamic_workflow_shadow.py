"""Durable scheduler observer for authority-free single-node shadow IR."""

from __future__ import annotations

from dataclasses import dataclass

from agents.yf_arc3_v5.dynamic_workflow.contracts import DynamicWorkflowIR, WorkflowBounds
from agents.yf_arc3_v5.dynamic_workflow.loader import LoadedColdComponentDocument
from agents.yf_arc3_v5.dynamic_workflow.registry import ColdComponentRegistry
from agents.yf_arc3_v5.dynamic_workflow.shadow import (
    DynamicWorkflowShadowRecord,
    ShadowEquivalenceSnapshot,
    build_shadow_snapshot,
    build_single_node_shadow_ir,
    shadow_mismatch_fields,
)
from agents.yf_arc3_v5.logos.operations import WorkflowInstanceRecord
from agents.yf_arc3_v5.logos.types import FrozenMap, Ref, stable_digest
from agents.yf_arc3_v5.scheduler.contracts import WorkflowStartRequest


@dataclass(frozen=True)
class _ShadowState:
    shadow_ir: DynamicWorkflowIR
    component_ref: Ref
    source_revision: int
    source_frame_ref: Ref
    workflow_ref: Ref
    input_fingerprint: Ref


class DynamicWorkflowShadowObserver:
    def __init__(
        self,
        loaded: LoadedColdComponentDocument,
        components: ColdComponentRegistry,
    ) -> None:
        self._loaded = loaded
        self._components = components
        self._bounds = WorkflowBounds.model_validate(loaded.document.global_bounds)
        self._component_by_workflow = {
            component.workflow_definition_ref: component
            for component in components.components
            if component.workflow_definition_ref is not None
        }
        self._states: dict[str, _ShadowState] = {}

    def begin(
        self,
        request: WorkflowStartRequest,
        *,
        source_revision: int,
    ) -> None:
        component = self._component_by_workflow.get(request.workflow_definition_id)
        if component is None:
            raise ValueError(
                "workflow start has no source-attested cold component: "
                f"{request.workflow_definition_id}"
            )
        if component.transitive_source_hash != request.expected_transitive_source_hash:
            raise ValueError("shadow component and workflow source hashes differ")
        source_frame_ref = request.frame_id or "frame:none"
        shadow_ir = build_single_node_shadow_ir(
            component=component,
            inputs=request.inputs,
            source_state_revision=source_revision,
            source_frame_ref=source_frame_ref,
            cold_registry_hash=self._components.registry_hash,
            drm_declaration_hash=self._loaded.semantic_hash,
            synthesis_policy_ref=self._loaded.document.synthesis_policy_ref,
            bounds=self._bounds,
        )
        self._states[request.workflow_instance_id] = _ShadowState(
            shadow_ir=shadow_ir,
            component_ref=component.component_ref,
            source_revision=source_revision,
            source_frame_ref=source_frame_ref,
            workflow_ref=request.workflow_definition_id,
            input_fingerprint=stable_digest(request.inputs),
        )

    def abort(self, workflow_instance_id: Ref) -> None:
        self._states.pop(workflow_instance_id, None)

    def restore(self, record: DynamicWorkflowShadowRecord) -> None:
        self._states[record.workflow_instance_id] = _ShadowState(
            shadow_ir=record.shadow_ir,
            component_ref=record.component_ref,
            source_revision=record.expected.source_revision,
            source_frame_ref=record.expected.source_frame_ref,
            workflow_ref=record.expected.workflow_ref,
            input_fingerprint=record.expected.input_fingerprint,
        )

    def observe(
        self,
        checkpoint: WorkflowInstanceRecord,
    ) -> DynamicWorkflowShadowRecord:
        workflow_instance_id = checkpoint.workflow_instance_id or ""
        state = self._states.get(workflow_instance_id)
        if state is None:
            raise ValueError(
                f"workflow checkpoint has no shadow state: {workflow_instance_id}"
            )
        projection = FrozenMap(
            {
                "awaiting_frame_binding": checkpoint.awaiting_frame_binding,
                "checkpoint_sequence": checkpoint.checkpoint_sequence,
                "control_stack_fingerprint": stable_digest(
                    checkpoint.control_stack
                ),
                "current_statement_id": checkpoint.current_statement_id,
                "current_step_id": checkpoint.current_step_id,
                "pending_action_intent_id": checkpoint.pending_action_intent_id,
                "pending_action_release_permit_id": (
                    checkpoint.pending_action_release_permit_id
                ),
                "pending_environment_adapter_ref": (
                    checkpoint.pending_environment_adapter_ref
                ),
                "pending_expectation_refs": checkpoint.pending_expectation_refs,
                "status": checkpoint.status,
                "terminal_result": checkpoint.terminal_result,
            }
        )
        observed = build_shadow_snapshot(
            source_revision=state.source_revision,
            source_frame_ref=state.source_frame_ref,
            registry_hash=self._components.registry_hash,
            workflow_ref=checkpoint.workflow_definition_id,
            input_fingerprint=state.input_fingerprint,
            checkpoint_projection=projection,
        )
        expected = observed.model_copy(
            update={
                "workflow_ref": state.workflow_ref,
                "registry_hash": state.shadow_ir.cold_registry_hash,
            }
        )
        return DynamicWorkflowShadowRecord(
            id=(
                f"{workflow_instance_id}:dynamic-shadow:"
                f"{checkpoint.checkpoint_sequence:08d}"
            ),
            source_unit=self._loaded.source_unit,
            source_hash=self._loaded.source_hash,
            state_revision=checkpoint.state_revision,
            workflow_definition_id=checkpoint.workflow_definition_id,
            workflow_instance_id=workflow_instance_id,
            timeline_id=checkpoint.timeline_id or "timeline:none",
            frame_id=checkpoint.frame_id or state.source_frame_ref,
            shadow_ir=state.shadow_ir,
            component_ref=state.component_ref,
            expected=expected,
            observed=observed,
            mismatch_fields=shadow_mismatch_fields(expected, observed),
        )
