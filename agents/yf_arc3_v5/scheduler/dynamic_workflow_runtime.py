"""Scheduler gate for source-declared dynamic-only hot activation and composition."""

from __future__ import annotations

from dataclasses import dataclass

from agents.yf_arc3_v5.dynamic_workflow.contracts import (
    ActionReleaseKind,
    HotWorkflowCheckpoint,
    HotWorkflowInstance,
    HotWorkflowOutcome,
    HotWorkflowOutcomeStatus,
    HotWorkflowStatus,
    WorkflowBounds,
)
from agents.yf_arc3_v5.dynamic_workflow.declarations import (
    DeclaredRuntimeCompositionEdge,
    load_declared_runtime_composition_edges,
)
from agents.yf_arc3_v5.dynamic_workflow.loader import (
    ColdComponentMigrationControl,
    LoadedColdComponentDocument,
)
from agents.yf_arc3_v5.dynamic_workflow.registry import ColdComponentRegistry
from agents.yf_arc3_v5.dynamic_workflow.runtime import (
    ChildOutcomeEnvelope,
    DynamicWorkflowExecutionPermit,
    RootColdExecutionRecord,
    RootColdSupervisor,
)
from agents.yf_arc3_v5.dynamic_workflow.shadow import build_single_node_shadow_ir
from agents.yf_arc3_v5.logos.operations import WorkflowInstanceRecord, WorkflowStatus
from agents.yf_arc3_v5.logos.types import FrozenMap, Ref, stable_digest
from agents.yf_arc3_v5.scheduler.contracts import WorkflowStartRequest


@dataclass(frozen=True)
class _ActiveExecution:
    control: ColdComponentMigrationControl
    hot_instance: HotWorkflowInstance
    permit: DynamicWorkflowExecutionPermit
    parent_workflow_instance_id: Ref | None = None
    declared_composition_edge_ref: Ref | None = None
    parent_hot_instance_before_attach: HotWorkflowInstance | None = None
    child_outcome: ChildOutcomeEnvelope | None = None


_HOT_STATUS_BY_SCHEDULER = {
    WorkflowStatus.ACTIVE: HotWorkflowStatus.CONTINUING,
    WorkflowStatus.SUSPENDED: HotWorkflowStatus.AWAITING_OBSERVATION,
    WorkflowStatus.COMPLETED: HotWorkflowStatus.COMPLETED,
    WorkflowStatus.DEFERRED: HotWorkflowStatus.SUSPENDED,
    WorkflowStatus.BLOCKED: HotWorkflowStatus.BLOCKED,
    WorkflowStatus.FAILED: HotWorkflowStatus.FALSIFIED,
}


class DynamicWorkflowRuntimeGate:
    def __init__(
        self,
        loaded: LoadedColdComponentDocument,
        components: ColdComponentRegistry,
    ) -> None:
        self._loaded = loaded
        self._components = components
        self._bounds = WorkflowBounds.model_validate(loaded.document.global_bounds)
        self._control_by_workflow = {
            control.workflow_ref: control
            for control in loaded.document.migration_controls
        }
        self._component_by_workflow = {
            component.workflow_definition_ref: component
            for component in components.components
            if component.workflow_definition_ref is not None
        }
        self._supervisor = RootColdSupervisor(components)
        self._active: dict[str, _ActiveExecution] = {}
        self._runtime_composition_edges = load_declared_runtime_composition_edges()
        self._runtime_composition_edge_by_ref = {
            edge.edge_ref: edge for edge in self._runtime_composition_edges
        }
        self._linked_child_instances: set[str] = set()
        self._edge_occurrences_by_parent: dict[tuple[str, str], int] = {}

    def _resolve_runtime_composition_edge(
        self,
        request: WorkflowStartRequest,
        parent: _ActiveExecution,
    ) -> DeclaredRuntimeCompositionEdge:
        edge_ref = request.declared_composition_edge_ref or ""
        edge = self._runtime_composition_edge_by_ref.get(edge_ref)
        if edge is None:
            raise ValueError("composed workflow start names an undeclared edge")
        if parent.control.workflow_ref not in edge.parent_workflow_refs:
            raise ValueError("declared composition edge rejects the parent workflow")
        if request.workflow_definition_id not in edge.child_workflow_refs:
            raise ValueError("declared composition edge rejects the child workflow")
        occurrence_key = (request.parent_workflow_instance_id or "", edge.edge_ref)
        if (
            self._edge_occurrences_by_parent.get(occurrence_key, 0)
            >= edge.maximum_occurrences_per_parent
        ):
            raise ValueError(
                "declared composition edge occurrence bound exhausted: "
                f"parent={occurrence_key[0]} edge={edge.edge_ref} "
                f"count={self._edge_occurrences_by_parent.get(occurrence_key, 0)} "
                f"maximum={edge.maximum_occurrences_per_parent} frame={request.frame_id}"
            )
        return edge

    def begin(
        self,
        request: WorkflowStartRequest,
        *,
        source_revision: int,
    ) -> None:
        control = self._control_by_workflow.get(request.workflow_definition_id)
        if control is None:
            raise ValueError("production workflow lacks a dynamic migration control")
        if (
            control.migration_state != "legacy_retired"
            or control.action_authority != "dynamic_only"
            or control.removal_authorization != "authorized_after_proof"
        ):
            raise ValueError("production workflow retains legacy action authority")
        component = self._component_by_workflow[request.workflow_definition_id]
        source_frame_ref = request.frame_id or "frame:none"
        workflow = build_single_node_shadow_ir(
            component=component,
            inputs=request.inputs,
            source_state_revision=source_revision,
            source_frame_ref=source_frame_ref,
            cold_registry_hash=self._components.registry_hash,
            drm_declaration_hash=self._loaded.semantic_hash,
            synthesis_policy_ref=self._loaded.document.synthesis_policy_ref,
            bounds=self._bounds,
        )
        parent = None
        edge = None
        parent_hot_before_attach = None
        if request.parent_workflow_instance_id is not None:
            parent = self._active.get(request.parent_workflow_instance_id)
            if parent is None:
                raise ValueError("composed workflow parent is not active in root cold")
            if parent.hot_instance.child_instance_refs:
                raise ValueError("composed workflow parent already has an open child")
            edge = self._resolve_runtime_composition_edge(request, parent)
            parent_hot_before_attach = parent.hot_instance
        hot_instance = self._supervisor.commit_single_node(
            workflow,
            workflow_instance_ref=(
                "hot-instance:" + request.workflow_instance_id
            ),
            parent_instance_ref=(
                parent.hot_instance.workflow_instance_ref if parent is not None else None
            ),
        )
        if parent is not None and edge is not None:
            attached_parent = self._supervisor.attach_child(
                parent.hot_instance,
                hot_instance,
            )
            self._active[request.parent_workflow_instance_id or ""] = _ActiveExecution(
                control=parent.control,
                hot_instance=attached_parent,
                permit=parent.permit,
                parent_workflow_instance_id=parent.parent_workflow_instance_id,
                declared_composition_edge_ref=parent.declared_composition_edge_ref,
                parent_hot_instance_before_attach=(
                    parent.parent_hot_instance_before_attach
                ),
                child_outcome=parent.child_outcome,
            )
            occurrence_key = (
                request.parent_workflow_instance_id or "",
                edge.edge_ref,
            )
            self._edge_occurrences_by_parent[occurrence_key] = (
                self._edge_occurrences_by_parent.get(occurrence_key, 0) + 1
            )
            self._linked_child_instances.add(request.workflow_instance_id)
        permit = self._supervisor.authorize_single_node_start(
            hot_instance,
            workflow_definition_ref=request.workflow_definition_id,
            expected_transitive_source_hash=request.expected_transitive_source_hash,
            inputs=request.inputs,
            source_state_revision=source_revision,
            source_frame_ref=source_frame_ref,
        )
        self._active[request.workflow_instance_id] = _ActiveExecution(
            control=control,
            hot_instance=hot_instance,
            permit=permit,
            parent_workflow_instance_id=request.parent_workflow_instance_id,
            declared_composition_edge_ref=request.declared_composition_edge_ref,
            parent_hot_instance_before_attach=parent_hot_before_attach,
        )

    def abort(self, workflow_instance_id: Ref) -> None:
        active = self._active.pop(workflow_instance_id, None)
        if active is None or active.parent_workflow_instance_id is None:
            return
        parent = self._active.get(active.parent_workflow_instance_id)
        if parent is not None and active.parent_hot_instance_before_attach is not None:
            self._active[active.parent_workflow_instance_id] = _ActiveExecution(
                control=parent.control,
                hot_instance=active.parent_hot_instance_before_attach,
                permit=parent.permit,
                parent_workflow_instance_id=parent.parent_workflow_instance_id,
                declared_composition_edge_ref=parent.declared_composition_edge_ref,
                parent_hot_instance_before_attach=parent.parent_hot_instance_before_attach,
                child_outcome=parent.child_outcome,
            )
        edge_ref = active.declared_composition_edge_ref or ""
        occurrence_key = (active.parent_workflow_instance_id, edge_ref)
        count = self._edge_occurrences_by_parent.get(occurrence_key, 0)
        if count <= 1:
            self._edge_occurrences_by_parent.pop(occurrence_key, None)
        else:
            self._edge_occurrences_by_parent[occurrence_key] = count - 1
        self._linked_child_instances.discard(workflow_instance_id)

    def restore(self, record: RootColdExecutionRecord) -> None:
        control = self._control_by_workflow.get(record.workflow_definition_id)
        if control is None or control.action_authority != "dynamic_only":
            raise ValueError("restored root-cold execution lacks dynamic authority")
        self._active[record.workflow_instance_id] = _ActiveExecution(
            control=control,
            hot_instance=record.hot_instance,
            permit=record.execution_permit,
            parent_workflow_instance_id=record.parent_workflow_instance_id,
            declared_composition_edge_ref=record.declared_composition_edge_ref,
            child_outcome=record.child_outcome,
        )
        if record.parent_workflow_instance_id is not None:
            parent = self._active.get(record.parent_workflow_instance_id)
            if parent is None:
                raise ValueError("restored composed workflow lacks its parent record")
            edge_ref = record.declared_composition_edge_ref or ""
            edge = self._runtime_composition_edge_by_ref.get(edge_ref)
            if edge is None:
                raise ValueError("restored composed workflow names an undeclared edge")
            if record.workflow_instance_id not in self._linked_child_instances:
                occurrence_key = (record.parent_workflow_instance_id, edge_ref)
                self._edge_occurrences_by_parent[occurrence_key] = (
                    self._edge_occurrences_by_parent.get(occurrence_key, 0) + 1
                )
                self._linked_child_instances.add(record.workflow_instance_id)
            if record.parent_hot_instance_after_reconciliation is not None:
                self._active[record.parent_workflow_instance_id] = _ActiveExecution(
                    control=parent.control,
                    hot_instance=record.parent_hot_instance_after_reconciliation,
                    permit=parent.permit,
                    parent_workflow_instance_id=parent.parent_workflow_instance_id,
                    declared_composition_edge_ref=parent.declared_composition_edge_ref,
                    parent_hot_instance_before_attach=(
                        parent.parent_hot_instance_before_attach
                    ),
                    child_outcome=parent.child_outcome,
                )

    def observe(
        self,
        checkpoint: WorkflowInstanceRecord,
    ) -> RootColdExecutionRecord | None:
        workflow_instance_id = checkpoint.workflow_instance_id or ""
        active = self._active.get(workflow_instance_id)
        if active is None:
            return None
        hot_status = _HOT_STATUS_BY_SCHEDULER[checkpoint.status]
        terminal = hot_status in {
            HotWorkflowStatus.COMPLETED,
            HotWorkflowStatus.BLOCKED,
            HotWorkflowStatus.FALSIFIED,
        }
        source_frame_ref = checkpoint.frame_id or active.hot_instance.frame_ref
        hot_instance = active.hot_instance
        child_outcome = active.child_outcome
        parent_hot_after_reconciliation = None
        if (
            hot_status is HotWorkflowStatus.AWAITING_OBSERVATION
            and active.permit.action_release_kind
            is ActionReleaseKind.SINGLE_PRIMITIVE
        ):
            if active.permit.node_ref not in hot_instance.released_action_node_refs:
                hot_instance = self._supervisor.release_action(
                    hot_instance,
                    active.permit,
                )
            hot_instance = hot_instance.model_copy(
                update={
                    "current_revision": checkpoint.state_revision,
                    "frame_ref": source_frame_ref,
                }
            )
        elif terminal and hot_status is HotWorkflowStatus.COMPLETED:
            outcome = HotWorkflowOutcome(
                status=HotWorkflowOutcomeStatus.COMPLETED,
                checkpoint_ref=checkpoint.id,
                evidence_refs=(checkpoint.id,),
            )
            if active.permit.action_release_kind is ActionReleaseKind.SINGLE_PRIMITIVE:
                if hot_instance.status is HotWorkflowStatus.AWAITING_OBSERVATION:
                    hot_instance = self._supervisor.reconcile_action_observation(
                        hot_instance,
                        active.permit,
                        observed_contract_ref=(
                            active.permit.awaited_observation_contract_ref
                        ),
                        outcome=outcome,
                        source_state_revision=checkpoint.state_revision,
                        source_frame_ref=source_frame_ref,
                    )
                else:
                    hot_instance = hot_instance.model_copy(
                        update={
                            "runnable_leaf_ref": None,
                            "current_revision": checkpoint.state_revision,
                            "frame_ref": source_frame_ref,
                            "status": HotWorkflowStatus.BLOCKED,
                        }
                    )
            else:
                hot_instance = self._supervisor.complete_non_action_node(
                    hot_instance,
                    active.permit,
                    outcome=outcome,
                    source_state_revision=checkpoint.state_revision,
                    source_frame_ref=source_frame_ref,
                )
        else:
            hot_instance = hot_instance.model_copy(
                update={
                    "runnable_leaf_ref": (
                        None if terminal else active.permit.node_ref
                    ),
                    "current_revision": checkpoint.state_revision,
                    "frame_ref": source_frame_ref,
                    "status": hot_status,
                }
            )
        if (
            active.parent_workflow_instance_id is not None
            and hot_instance.status
            in {
                HotWorkflowStatus.COMPLETED,
                HotWorkflowStatus.BLOCKED,
                HotWorkflowStatus.FALSIFIED,
            }
            and child_outcome is None
        ):
            parent = self._active.get(active.parent_workflow_instance_id)
            if parent is None:
                raise ValueError("composed workflow completed without its parent")
            outcome = HotWorkflowOutcome(
                status=HotWorkflowOutcomeStatus(hot_instance.status.value),
                checkpoint_ref=checkpoint.id,
                evidence_refs=(checkpoint.id,),
            )
            parent_hot_after_reconciliation, child_outcome = (
                self._supervisor.reconcile_child_outcome(
                    parent.hot_instance,
                    hot_instance,
                    outcome=outcome,
                    declared_disposition_ref=(
                        active.declared_composition_edge_ref or ""
                    ),
                )
            )
            self._active[active.parent_workflow_instance_id] = _ActiveExecution(
                control=parent.control,
                hot_instance=parent_hot_after_reconciliation,
                permit=parent.permit,
                parent_workflow_instance_id=parent.parent_workflow_instance_id,
                declared_composition_edge_ref=parent.declared_composition_edge_ref,
                parent_hot_instance_before_attach=(
                    parent.parent_hot_instance_before_attach
                ),
                child_outcome=parent.child_outcome,
            )
        self._active[workflow_instance_id] = _ActiveExecution(
            control=active.control,
            hot_instance=hot_instance,
            permit=active.permit,
            parent_workflow_instance_id=active.parent_workflow_instance_id,
            declared_composition_edge_ref=active.declared_composition_edge_ref,
            parent_hot_instance_before_attach=active.parent_hot_instance_before_attach,
            child_outcome=child_outcome,
        )
        hot_checkpoint = HotWorkflowCheckpoint(
            workflow_ir_hash=active.permit.workflow_ir_hash,
            workflow_instance_ref=hot_instance.workflow_instance_ref,
            current_node_ref=hot_instance.runnable_leaf_ref,
            completed_node_refs=hot_instance.completed_node_refs,
            inactive_node_refs=hot_instance.inactive_node_refs,
            released_action_node_refs=hot_instance.released_action_node_refs,
            reconciled_action_node_refs=hot_instance.reconciled_action_node_refs,
            state_revision=checkpoint.state_revision,
            frame_ref=checkpoint.frame_id or hot_instance.frame_ref,
            remaining_bounds=hot_instance.remaining_global_bounds,
            awaited_observation_contract=hot_instance.awaiting_observation_contract,
            status=hot_instance.status,
        )
        scheduler_projection = FrozenMap(
            {
                "checkpoint_sequence": checkpoint.checkpoint_sequence,
                "current_statement_id": checkpoint.current_statement_id,
                "current_step_id": checkpoint.current_step_id,
                "pending_action_intent_id": checkpoint.pending_action_intent_id,
                "pending_action_release_permit_id": (
                    checkpoint.pending_action_release_permit_id
                ),
                "status": checkpoint.status,
                "terminal_result": checkpoint.terminal_result,
            }
        )
        return RootColdExecutionRecord(
            id=(
                f"{workflow_instance_id}:root-cold:"
                f"{checkpoint.checkpoint_sequence:08d}"
            ),
            source_unit=self._loaded.source_unit,
            source_hash=self._loaded.source_hash,
            state_revision=checkpoint.state_revision,
            workflow_definition_id=checkpoint.workflow_definition_id,
            workflow_instance_id=workflow_instance_id,
            timeline_id=checkpoint.timeline_id or "timeline:none",
            frame_id=checkpoint.frame_id or hot_instance.frame_ref,
            hot_instance=hot_instance,
            hot_checkpoint=hot_checkpoint,
            execution_permit=active.permit,
            scheduler_checkpoint_fingerprint=stable_digest(scheduler_projection),
            parent_workflow_instance_id=active.parent_workflow_instance_id,
            declared_composition_edge_ref=active.declared_composition_edge_ref,
            child_outcome=child_outcome,
            parent_hot_instance_after_reconciliation=(
                parent_hot_after_reconciliation
            ),
        )
