"""Mechanical root-cold validation and hot execution contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from agents.yf_arc3_v5.dynamic_workflow.contracts import (
    ActionReleaseKind,
    DynamicWorkflowIR,
    HotWorkflowCheckpoint,
    HotWorkflowInstance,
    HotWorkflowOutcome,
    HotWorkflowOutcomeStatus,
    HotWorkflowStatus,
)
from agents.yf_arc3_v5.dynamic_workflow.registry import (
    ColdComponentRegistry,
    DynamicWorkflowRegistry,
)
from agents.yf_arc3_v5.logos.types import (
    FrozenMap,
    FrozenModel,
    NonNegativeRevision,
    Ref,
    stable_digest,
)


class DynamicWorkflowExecutionPermit(FrozenModel):
    workflow_ir_hash: Ref
    workflow_instance_ref: Ref
    node_ref: Ref
    component_ref: Ref
    workflow_definition_ref: Ref
    expected_transitive_source_hash: Ref
    input_fingerprint: Ref
    source_state_revision: NonNegativeRevision
    source_frame_ref: Ref
    awaited_observation_contract_ref: Ref
    action_release_kind: ActionReleaseKind
    action_authority: Literal["dynamic_only"] = "dynamic_only"
    schema_version: Ref = "yf_arc3_v5.dynamic_workflow_execution_permit.v1"


class ChildOutcomeEnvelope(FrozenModel):
    parent_instance_ref: Ref
    child_instance_ref: Ref
    declared_disposition_ref: Ref
    outcome: HotWorkflowOutcome
    schema_version: Ref = "yf_arc3_v5.child_outcome_envelope.v1"


class RootColdExecutionRecord(FrozenModel):
    record_kind: Literal["root_cold_execution"] = "root_cold_execution"
    id: Ref
    source_unit: Ref
    source_hash: Ref
    state_revision: NonNegativeRevision
    workflow_definition_id: Ref
    workflow_instance_id: Ref
    timeline_id: Ref
    frame_id: Ref
    hot_instance: HotWorkflowInstance
    hot_checkpoint: HotWorkflowCheckpoint
    execution_permit: DynamicWorkflowExecutionPermit
    scheduler_checkpoint_fingerprint: Ref
    parent_workflow_instance_id: Ref | None = None
    declared_composition_edge_ref: Ref | None = None
    child_outcome: ChildOutcomeEnvelope | None = None
    parent_hot_instance_after_reconciliation: HotWorkflowInstance | None = None
    action_authority: Literal["dynamic_only"] = "dynamic_only"
    schema_version: Ref = "yf_arc3_v5.root_cold_execution_record.v2"

    @model_validator(mode="after")
    def validate_record(self) -> "RootColdExecutionRecord":
        if self.hot_instance.workflow_ir_hash != self.execution_permit.workflow_ir_hash:
            raise ValueError("root-cold record mixes dynamic workflow hashes")
        if (
            self.hot_checkpoint.workflow_instance_ref
            != self.hot_instance.workflow_instance_ref
        ):
            raise ValueError("root-cold record mixes hot workflow instances")
        composed = self.parent_workflow_instance_id is not None
        if composed != (self.declared_composition_edge_ref is not None):
            raise ValueError("root-cold composition linkage is incomplete")
        if composed != (self.hot_instance.parent_instance_ref is not None):
            raise ValueError("root-cold composition parent linkage is inconsistent")
        if self.child_outcome is not None:
            if not composed:
                raise ValueError("root-cold child outcome lacks a declared parent")
            if self.child_outcome.child_instance_ref != self.hot_instance.workflow_instance_ref:
                raise ValueError("root-cold child outcome names another hot child")
            if self.child_outcome.declared_disposition_ref != self.declared_composition_edge_ref:
                raise ValueError("root-cold child outcome names another declared edge")
        if self.parent_hot_instance_after_reconciliation is not None:
            if self.child_outcome is None:
                raise ValueError("reconciled parent snapshot lacks a child outcome")
            if (
                self.parent_hot_instance_after_reconciliation.workflow_instance_ref
                != self.child_outcome.parent_instance_ref
            ):
                raise ValueError("reconciled parent snapshot names another parent")
        return self


class DeclaredWorkflowCommit(FrozenModel):
    committed_workflow: DynamicWorkflowIR
    discarded_candidate_hashes: tuple[Ref, ...] = ()
    declared_selection_ref: Ref
    schema_version: Ref = "yf_arc3_v5.declared_workflow_commit.v1"


def _input_ref_fingerprints(inputs: FrozenMap) -> FrozenMap:
    return FrozenMap(
        {key: stable_digest(value) for key, value in sorted(inputs.items())}
    )


class RootColdSupervisor:
    """Validate and register exact hot IR; never choose its semantic leaf."""

    def __init__(self, components: ColdComponentRegistry) -> None:
        self._components = components
        self._workflows = DynamicWorkflowRegistry()

    def commit_declared_candidate(
        self,
        candidates: tuple[DynamicWorkflowIR, ...],
        *,
        declared_semantic_hash: Ref,
        declared_selection_ref: Ref,
    ) -> DeclaredWorkflowCommit:
        matches = tuple(
            candidate
            for candidate in candidates
            if candidate.semantic_hash == declared_semantic_hash
        )
        if len(matches) != 1:
            raise ValueError("declared workflow selection must resolve exactly one candidate")
        selected = matches[0]
        return DeclaredWorkflowCommit(
            committed_workflow=selected,
            discarded_candidate_hashes=tuple(
                sorted(
                    candidate.semantic_hash
                    for candidate in candidates
                    if candidate.semantic_hash != selected.semantic_hash
                )
            ),
            declared_selection_ref=declared_selection_ref,
        )

    def _validate_workflow_components(self, workflow: DynamicWorkflowIR) -> None:
        if workflow.cold_registry_hash != self._components.registry_hash:
            raise ValueError("hot workflow cold registry hash mismatch")
        order = workflow.canonical_derivation.declared_node_order_refs
        order_index = {node_ref: index for index, node_ref in enumerate(order)}
        for node in workflow.nodes:
            component = self._components.resolve(node.component_ref)
            declared_hash = (
                workflow.canonical_derivation.cold_component_transitive_hashes.get(
                    component.component_ref
                )
            )
            if declared_hash != component.transitive_source_hash:
                raise ValueError("hot workflow component derivation hash mismatch")
        for edge in workflow.dependency_edges:
            if order_index[edge.provider_node_ref] >= order_index[edge.consumer_node_ref]:
                raise ValueError("declared workflow order must be dependency-topological")
            law_hash = workflow.canonical_derivation.composition_law_hashes.get(
                edge.composition_principle_ref
            )
            if law_hash is None:
                raise ValueError("workflow edge lacks its declared composition law hash")

    @staticmethod
    def _next_declared_ready_node(
        workflow: DynamicWorkflowIR,
        *,
        completed_node_refs: tuple[Ref, ...],
        inactive_node_refs: tuple[Ref, ...],
    ) -> Ref | None:
        completed = set(completed_node_refs)
        inactive = set(inactive_node_refs)
        providers_by_consumer: dict[str, set[str]] = {
            node.node_ref: set() for node in workflow.nodes
        }
        for edge in workflow.dependency_edges:
            providers_by_consumer[edge.consumer_node_ref].add(edge.provider_node_ref)
        for node_ref in workflow.canonical_derivation.declared_node_order_refs:
            if node_ref in completed or node_ref in inactive:
                continue
            if providers_by_consumer[node_ref].issubset(completed):
                return node_ref
        return None

    def commit_workflow(
        self,
        workflow: DynamicWorkflowIR,
        *,
        workflow_instance_ref: Ref,
        parent_instance_ref: Ref | None = None,
    ) -> HotWorkflowInstance:
        self._validate_workflow_components(workflow)
        self._workflows.register(workflow)
        runnable = self._next_declared_ready_node(
            workflow,
            completed_node_refs=(),
            inactive_node_refs=(),
        )
        if runnable is None:
            raise ValueError("committed workflow has no declared runnable node")
        return HotWorkflowInstance(
            workflow_instance_ref=workflow_instance_ref,
            workflow_ir_hash=workflow.semantic_hash,
            root_goal_ref=workflow.root_goal_ref,
            parent_instance_ref=parent_instance_ref,
            runnable_leaf_ref=runnable,
            current_revision=workflow.source_state_revision,
            frame_ref=workflow.source_frame_ref,
            remaining_global_bounds=workflow.global_resource_bounds,
            status=HotWorkflowStatus.COMMITTED,
        )

    def commit_single_node(
        self,
        workflow: DynamicWorkflowIR,
        *,
        workflow_instance_ref: Ref,
        parent_instance_ref: Ref | None = None,
    ) -> HotWorkflowInstance:
        if len(workflow.nodes) != 1:
            raise ValueError("single-node activation requires exactly one node")
        if workflow.dependency_edges or workflow.branch_conditions:
            raise ValueError("single-node activation cannot carry edges or branches")
        return self.commit_workflow(
            workflow,
            workflow_instance_ref=workflow_instance_ref,
            parent_instance_ref=parent_instance_ref,
        )

    def authorize_node_start(
        self,
        instance: HotWorkflowInstance,
        *,
        node_ref: Ref,
        workflow_definition_ref: Ref,
        expected_transitive_source_hash: Ref,
        inputs: FrozenMap,
        source_state_revision: NonNegativeRevision,
        source_frame_ref: Ref,
    ) -> DynamicWorkflowExecutionPermit:
        workflow = self._workflows.resolve(instance.workflow_ir_hash)
        if instance.status not in {
            HotWorkflowStatus.COMMITTED,
            HotWorkflowStatus.CONTINUING,
        }:
            raise ValueError("hot workflow is not ready to start a node")
        if workflow.source_state_revision > source_state_revision:
            raise ValueError("hot workflow source revision is from the future")
        if instance.current_revision != source_state_revision:
            raise ValueError("hot workflow source revision is stale")
        if instance.frame_ref != source_frame_ref:
            raise ValueError("hot workflow current frame is stale")
        if instance.runnable_leaf_ref != node_ref:
            raise ValueError("requested node is not the declared runnable leaf")
        node = next(item for item in workflow.nodes if item.node_ref == node_ref)
        component = self._components.resolve(
            node.component_ref,
            expected_transitive_source_hash=expected_transitive_source_hash,
        )
        if not component.runtime_dispatchable:
            raise ValueError("cold component is not runtime dispatchable")
        if component.workflow_definition_ref != workflow_definition_ref:
            raise ValueError("cold component workflow definition mismatch")
        if not set(inputs).issubset(component.input_contract):
            raise ValueError("hot workflow inputs contain an undeclared cold-component field")
        if node.bound_input_refs != _input_ref_fingerprints(inputs):
            raise ValueError("hot workflow input fingerprint mismatch")
        return DynamicWorkflowExecutionPermit(
            workflow_ir_hash=workflow.semantic_hash,
            workflow_instance_ref=instance.workflow_instance_ref,
            node_ref=node.node_ref,
            component_ref=component.component_ref,
            workflow_definition_ref=workflow_definition_ref,
            expected_transitive_source_hash=expected_transitive_source_hash,
            input_fingerprint=stable_digest(inputs),
            source_state_revision=source_state_revision,
            source_frame_ref=source_frame_ref,
            awaited_observation_contract_ref=component.reconciliation_contract_ref,
            action_release_kind=component.action_release_kind,
        )

    def authorize_single_node_start(
        self,
        instance: HotWorkflowInstance,
        *,
        workflow_definition_ref: Ref,
        expected_transitive_source_hash: Ref,
        inputs: FrozenMap,
        source_state_revision: NonNegativeRevision,
        source_frame_ref: Ref,
    ) -> DynamicWorkflowExecutionPermit:
        workflow = self._workflows.resolve(instance.workflow_ir_hash)
        if len(workflow.nodes) != 1:
            raise ValueError("single-node authorization requires exactly one node")
        return self.authorize_node_start(
            instance,
            node_ref=workflow.nodes[0].node_ref,
            workflow_definition_ref=workflow_definition_ref,
            expected_transitive_source_hash=expected_transitive_source_hash,
            inputs=inputs,
            source_state_revision=source_state_revision,
            source_frame_ref=source_frame_ref,
        )

    def _advance_after_completion(
        self,
        instance: HotWorkflowInstance,
        *,
        completed_node_ref: Ref,
        source_state_revision: NonNegativeRevision,
        source_frame_ref: Ref,
    ) -> HotWorkflowInstance:
        workflow = self._workflows.resolve(instance.workflow_ir_hash)
        completed = tuple(
            sorted(set(instance.completed_node_refs).union({completed_node_ref}))
        )
        runnable = self._next_declared_ready_node(
            workflow,
            completed_node_refs=completed,
            inactive_node_refs=instance.inactive_node_refs,
        )
        terminal = len(completed) + len(instance.inactive_node_refs) == len(
            workflow.nodes
        )
        if not terminal and runnable is None:
            raise ValueError("hot workflow has unresolved dependencies and no runnable node")
        return instance.model_copy(
            update={
                "completed_node_refs": completed,
                "runnable_leaf_ref": None if terminal else runnable,
                "awaiting_observation_contract": None,
                "current_revision": source_state_revision,
                "frame_ref": source_frame_ref,
                "status": (
                    HotWorkflowStatus.COMPLETED
                    if terminal
                    else HotWorkflowStatus.CONTINUING
                ),
            }
        )

    def complete_non_action_node(
        self,
        instance: HotWorkflowInstance,
        permit: DynamicWorkflowExecutionPermit,
        *,
        outcome: HotWorkflowOutcome,
        source_state_revision: NonNegativeRevision,
        source_frame_ref: Ref,
    ) -> HotWorkflowInstance:
        if permit.workflow_ir_hash != instance.workflow_ir_hash:
            raise ValueError("node permit belongs to another hot workflow")
        if permit.node_ref != instance.runnable_leaf_ref:
            raise ValueError("node permit is no longer the runnable leaf")
        if permit.action_release_kind is not ActionReleaseKind.NONE:
            raise ValueError("an action-bearing node requires observation reconciliation")
        if outcome.status is not HotWorkflowOutcomeStatus.COMPLETED:
            raise ValueError("non-completed outcome requires a declared lifecycle disposition")
        return self._advance_after_completion(
            instance,
            completed_node_ref=permit.node_ref,
            source_state_revision=source_state_revision,
            source_frame_ref=source_frame_ref,
        )

    def release_action(
        self,
        instance: HotWorkflowInstance,
        permit: DynamicWorkflowExecutionPermit,
    ) -> HotWorkflowInstance:
        if permit.workflow_ir_hash != instance.workflow_ir_hash:
            raise ValueError("action permit belongs to another hot workflow")
        if permit.node_ref in instance.released_action_node_refs:
            raise ValueError("the exact action-bearing node was already released")
        if permit.node_ref != instance.runnable_leaf_ref:
            raise ValueError("action permit is no longer the runnable leaf")
        if permit.action_release_kind is not ActionReleaseKind.SINGLE_PRIMITIVE:
            raise ValueError("non-action node cannot release an environment primitive")
        if instance.awaiting_observation_contract is not None:
            raise ValueError("a second action cannot precede observation reconciliation")
        if instance.remaining_global_bounds.maximum_actions < 1:
            raise ValueError("hot workflow action bound exhausted")
        released = tuple(
            sorted(set(instance.released_action_node_refs).union({permit.node_ref}))
        )
        remaining = instance.remaining_global_bounds.model_copy(
            update={
                "maximum_actions": (
                    instance.remaining_global_bounds.maximum_actions - 1
                )
            }
        )
        return instance.model_copy(
            update={
                "released_action_node_refs": released,
                "runnable_leaf_ref": None,
                "awaiting_observation_contract": (
                    permit.awaited_observation_contract_ref
                ),
                "remaining_global_bounds": remaining,
                "status": HotWorkflowStatus.AWAITING_OBSERVATION,
            }
        )

    def reconcile_action_observation(
        self,
        instance: HotWorkflowInstance,
        permit: DynamicWorkflowExecutionPermit,
        *,
        observed_contract_ref: Ref,
        outcome: HotWorkflowOutcome,
        source_state_revision: NonNegativeRevision,
        source_frame_ref: Ref,
    ) -> HotWorkflowInstance:
        if instance.status is not HotWorkflowStatus.AWAITING_OBSERVATION:
            raise ValueError("hot workflow is not awaiting an observation")
        if permit.node_ref not in instance.released_action_node_refs:
            raise ValueError("observation cannot reconcile an unreleased action")
        if permit.node_ref in instance.reconciled_action_node_refs:
            raise ValueError("the exact action observation was already reconciled")
        if observed_contract_ref != instance.awaiting_observation_contract:
            raise ValueError("observation does not satisfy the awaited contract")
        if (
            source_state_revision <= instance.current_revision
            and source_frame_ref == instance.frame_ref
        ):
            raise ValueError("action reconciliation requires a fresh revision or frame")
        if outcome.status is not HotWorkflowOutcomeStatus.COMPLETED:
            raise ValueError("action reconciliation outcome is not completed")
        reconciled = tuple(
            sorted(
                set(instance.reconciled_action_node_refs).union({permit.node_ref})
            )
        )
        advanced = instance.model_copy(
            update={"reconciled_action_node_refs": reconciled}
        )
        return self._advance_after_completion(
            advanced,
            completed_node_ref=permit.node_ref,
            source_state_revision=source_state_revision,
            source_frame_ref=source_frame_ref,
        )

    def apply_declared_branch(
        self,
        instance: HotWorkflowInstance,
        *,
        source_node_ref: Ref,
        declared_target_node_ref: Ref,
    ) -> HotWorkflowInstance:
        workflow = self._workflows.resolve(instance.workflow_ir_hash)
        branch = next(
            (
                item
                for item in workflow.branch_conditions
                if item.source_node_ref == source_node_ref
                and declared_target_node_ref
                in {item.true_node_ref, item.false_node_ref}
            ),
            None,
        )
        if branch is None:
            raise ValueError("declared branch target is not present in committed IR")
        if source_node_ref not in instance.completed_node_refs:
            raise ValueError("branch source must complete before branch disposition")
        inactive_ref = (
            branch.false_node_ref
            if declared_target_node_ref == branch.true_node_ref
            else branch.true_node_ref
        )
        inactive = tuple(
            sorted(set(instance.inactive_node_refs).union({inactive_ref}))
        )
        runnable = self._next_declared_ready_node(
            workflow,
            completed_node_refs=instance.completed_node_refs,
            inactive_node_refs=inactive,
        )
        return instance.model_copy(
            update={
                "inactive_node_refs": inactive,
                "runnable_leaf_ref": runnable,
                "status": (
                    HotWorkflowStatus.COMPLETED
                    if runnable is None
                    else HotWorkflowStatus.CONTINUING
                ),
            }
        )

    def attach_child(
        self,
        parent: HotWorkflowInstance,
        child: HotWorkflowInstance,
    ) -> HotWorkflowInstance:
        if child.parent_instance_ref != parent.workflow_instance_ref:
            raise ValueError("hot child must name its single execution parent")
        if child.workflow_instance_ref in parent.child_instance_refs:
            raise ValueError("hot child is already attached to this parent")
        if parent.remaining_global_bounds.maximum_children < 1:
            raise ValueError("hot workflow child bound exhausted")
        remaining = parent.remaining_global_bounds.model_copy(
            update={
                "maximum_children": (
                    parent.remaining_global_bounds.maximum_children - 1
                )
            }
        )
        return parent.model_copy(
            update={
                "child_instance_refs": tuple(
                    sorted(parent.child_instance_refs + (child.workflow_instance_ref,))
                ),
                "remaining_global_bounds": remaining,
                "status": HotWorkflowStatus.SUSPENDED,
            }
        )

    @staticmethod
    def reconcile_child_outcome(
        parent: HotWorkflowInstance,
        child: HotWorkflowInstance,
        *,
        outcome: HotWorkflowOutcome,
        declared_disposition_ref: Ref,
    ) -> tuple[HotWorkflowInstance, ChildOutcomeEnvelope]:
        if child.parent_instance_ref != parent.workflow_instance_ref:
            raise ValueError("child outcome names a different execution parent")
        if child.workflow_instance_ref not in parent.child_instance_refs:
            raise ValueError("child outcome is not open under this parent")
        remaining_children = tuple(
            item
            for item in parent.child_instance_refs
            if item != child.workflow_instance_ref
        )
        reconciled_parent = parent.model_copy(
            update={
                "child_instance_refs": remaining_children,
                "status": HotWorkflowStatus.CONTINUING,
            }
        )
        return reconciled_parent, ChildOutcomeEnvelope(
            parent_instance_ref=parent.workflow_instance_ref,
            child_instance_ref=child.workflow_instance_ref,
            declared_disposition_ref=declared_disposition_ref,
            outcome=outcome,
        )

    def checkpoint(self, instance: HotWorkflowInstance) -> HotWorkflowCheckpoint:
        self._workflows.resolve(instance.workflow_ir_hash)
        return HotWorkflowCheckpoint(
            workflow_ir_hash=instance.workflow_ir_hash,
            workflow_instance_ref=instance.workflow_instance_ref,
            parent_instance_ref=instance.parent_instance_ref,
            current_node_ref=instance.runnable_leaf_ref,
            completed_node_refs=instance.completed_node_refs,
            inactive_node_refs=instance.inactive_node_refs,
            released_action_node_refs=instance.released_action_node_refs,
            reconciled_action_node_refs=instance.reconciled_action_node_refs,
            open_child_refs=instance.child_instance_refs,
            state_revision=instance.current_revision,
            frame_ref=instance.frame_ref,
            remaining_bounds=instance.remaining_global_bounds,
            awaited_observation_contract=instance.awaiting_observation_contract,
            status=instance.status,
        )

    def restore_checkpoint(
        self,
        checkpoint: HotWorkflowCheckpoint,
    ) -> HotWorkflowInstance:
        workflow = self._workflows.resolve(checkpoint.workflow_ir_hash)
        known_nodes = {node.node_ref for node in workflow.nodes}
        referenced_nodes = set(checkpoint.completed_node_refs).union(
            checkpoint.inactive_node_refs,
            checkpoint.released_action_node_refs,
            checkpoint.reconciled_action_node_refs,
        )
        if not referenced_nodes.issubset(known_nodes):
            raise ValueError("checkpoint names a node absent from committed IR")
        return HotWorkflowInstance(
            workflow_instance_ref=checkpoint.workflow_instance_ref,
            workflow_ir_hash=checkpoint.workflow_ir_hash,
            root_goal_ref=workflow.root_goal_ref,
            parent_instance_ref=checkpoint.parent_instance_ref,
            child_instance_refs=checkpoint.open_child_refs,
            completed_node_refs=checkpoint.completed_node_refs,
            inactive_node_refs=checkpoint.inactive_node_refs,
            released_action_node_refs=checkpoint.released_action_node_refs,
            reconciled_action_node_refs=checkpoint.reconciled_action_node_refs,
            runnable_leaf_ref=checkpoint.current_node_ref,
            awaiting_observation_contract=checkpoint.awaited_observation_contract,
            current_revision=checkpoint.state_revision,
            frame_ref=checkpoint.frame_ref,
            remaining_global_bounds=checkpoint.remaining_bounds,
            status=checkpoint.status,
        )
