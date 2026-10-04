"""Authority-free single-node shadow IR and exact equivalence records."""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from agents.yf_arc3_v5.dynamic_workflow.contracts import (
    CanonicalWorkflowDerivation,
    ColdWorkflowComponent,
    DynamicWorkflowIR,
    DynamicWorkflowNode,
    WorkflowBounds,
    WorkflowCompletionContract,
    build_dynamic_workflow_ir,
)
from agents.yf_arc3_v5.logos.types import (
    FrozenMap,
    FrozenModel,
    NonNegativeRevision,
    Ref,
    require_unique,
    stable_digest,
)


class ShadowEquivalenceSnapshot(FrozenModel):
    source_revision: NonNegativeRevision
    source_frame_ref: Ref
    registry_hash: Ref
    workflow_ref: Ref
    input_fingerprint: Ref
    action_fingerprint: Ref
    checkpoint_fingerprint: Ref
    outcome_fingerprint: Ref
    parallel_join_fingerprint: Ref
    schema_version: Ref = "yf_arc3_v5.dynamic_workflow_shadow_snapshot.v1"


class DynamicWorkflowShadowRecord(FrozenModel):
    record_kind: Literal["dynamic_workflow_shadow"] = "dynamic_workflow_shadow"
    id: Ref
    source_unit: Ref
    source_hash: Ref
    state_revision: NonNegativeRevision
    workflow_definition_id: Ref
    workflow_instance_id: Ref
    timeline_id: Ref
    frame_id: Ref
    shadow_ir: DynamicWorkflowIR
    component_ref: Ref
    expected: ShadowEquivalenceSnapshot
    observed: ShadowEquivalenceSnapshot
    mismatch_fields: tuple[Ref, ...] = ()
    action_authority: Literal["none"] = "none"
    schema_version: Ref = "yf_arc3_v5.dynamic_workflow_shadow_record.v1"

    @model_validator(mode="after")
    def validate_record(self) -> "DynamicWorkflowShadowRecord":
        require_unique(self.mismatch_fields, "shadow mismatch fields")
        if self.mismatch_fields != tuple(sorted(self.mismatch_fields)):
            raise ValueError("shadow mismatch fields must use canonical order")
        actual = shadow_mismatch_fields(self.expected, self.observed)
        if actual != self.mismatch_fields:
            raise ValueError("shadow mismatch field projection is stale")
        return self

    @property
    def equivalent(self) -> bool:
        return not self.mismatch_fields


def _fingerprint_map(values: FrozenMap) -> FrozenMap:
    return FrozenMap(
        {key: stable_digest(value) for key, value in sorted(values.items())}
    )


def build_single_node_shadow_ir(
    *,
    component: ColdWorkflowComponent,
    inputs: FrozenMap,
    source_state_revision: NonNegativeRevision,
    source_frame_ref: Ref,
    cold_registry_hash: Ref,
    drm_declaration_hash: Ref,
    synthesis_policy_ref: Ref,
    bounds: WorkflowBounds,
) -> DynamicWorkflowIR:
    """Wrap one declared legacy workflow without granting the wrapper authority."""

    node_ref = "dynamic-shadow-node:" + stable_digest(
        {
            "component_ref": component.component_ref,
            "input_fingerprint": stable_digest(inputs),
            "source_state_revision": source_state_revision,
            "source_frame_ref": source_frame_ref,
        }
    )[:24]
    node = DynamicWorkflowNode(
        node_ref=node_ref,
        component_ref=component.component_ref,
        served_goal_refs=(component.served_migration_goal_ref,),
        bound_input_refs=_fingerprint_map(inputs),
        expected_output_predicate_refs=component.provides_predicate_refs,
        local_precondition_refs=component.requires_predicate_refs,
        local_invariant_refs=component.preserves_invariant_refs,
        local_falsifier_refs=component.falsifier_refs,
        local_resource_bound=FrozenMap(
            {
                "maximum_action_count": component.maximum_action_count,
                "maximum_symbolic_depth": component.maximum_symbolic_depth,
            }
        ),
    )
    return build_dynamic_workflow_ir(
        root_goal_ref=component.served_migration_goal_ref,
        source_state_revision=source_state_revision,
        source_frame_ref=source_frame_ref,
        cold_registry_hash=cold_registry_hash,
        drm_declaration_hash=drm_declaration_hash,
        synthesis_policy_ref=synthesis_policy_ref,
        nodes=(node,),
        dependency_edges=(),
        branch_conditions=(),
        preserved_invariant_refs=component.preserves_invariant_refs,
        global_falsifier_refs=component.falsifier_refs,
        global_resource_bounds=bounds,
        completion_contract=WorkflowCompletionContract(
            required_terminal_predicate_refs=component.provides_predicate_refs,
            observable_completion_evidence_refs=(
                component.reconciliation_contract_ref,
            ),
            preserved_condition_refs=component.preserves_invariant_refs,
            resource_bound_refs=component.resource_dimensions,
        ),
        canonical_derivation=CanonicalWorkflowDerivation(
            cold_component_transitive_hashes=FrozenMap(
                {component.component_ref: component.transitive_source_hash}
            ),
            composition_law_hashes=FrozenMap(),
            canonical_goal_bindings=FrozenMap(
                {"root": component.served_migration_goal_ref}
            ),
            declared_order_ref="workflow.shadow.single_node_declared_order",
            declared_node_order_refs=(node_ref,),
        ),
    )


def shadow_mismatch_fields(
    expected: ShadowEquivalenceSnapshot,
    observed: ShadowEquivalenceSnapshot,
) -> tuple[Ref, ...]:
    return tuple(
        sorted(
            field_name
            for field_name in expected.__class__.model_fields
            if field_name != "schema_version"
            and getattr(expected, field_name) != getattr(observed, field_name)
        )
    )


def build_shadow_snapshot(
    *,
    source_revision: NonNegativeRevision,
    source_frame_ref: Ref,
    registry_hash: Ref,
    workflow_ref: Ref,
    input_fingerprint: Ref,
    checkpoint_projection: FrozenMap,
) -> ShadowEquivalenceSnapshot:
    action_projection = FrozenMap(
        {
            key: checkpoint_projection.get(key)
            for key in (
                "pending_action_intent_id",
                "pending_action_release_permit_id",
                "pending_environment_adapter_ref",
            )
        }
    )
    outcome_projection = FrozenMap(
        {
            "status": checkpoint_projection.get("status"),
            "terminal_result": checkpoint_projection.get("terminal_result"),
        }
    )
    return ShadowEquivalenceSnapshot(
        source_revision=source_revision,
        source_frame_ref=source_frame_ref,
        registry_hash=registry_hash,
        workflow_ref=workflow_ref,
        input_fingerprint=input_fingerprint,
        action_fingerprint=stable_digest(action_projection),
        checkpoint_fingerprint=stable_digest(checkpoint_projection),
        outcome_fingerprint=stable_digest(outcome_projection),
        parallel_join_fingerprint=stable_digest(()),
    )
