"""Immutable contracts for cold components, dynamic IR and hot instances."""

from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from agents.yf_arc3_v5.logos.types import (
    FrozenMap,
    FrozenModel,
    NonNegativeRevision,
    Ref,
    require_unique,
    stable_digest,
)


def _require_sha256(value: str, description: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")


def _require_canonical_refs(values: tuple[Ref, ...], description: str) -> None:
    if len(values) < 2:
        return
    require_unique(values, description)
    if any(values[index - 1] > values[index] for index in range(1, len(values))):
        raise ValueError(f"{description} must use canonical reference order")


class ActionReleaseKind(str, Enum):
    NONE = "none"
    SINGLE_PRIMITIVE = "single_primitive"


class HotWorkflowStatus(str, Enum):
    PROPOSED = "proposed"
    ELIGIBLE = "eligible"
    INSTANTIATED = "instantiated"
    COMMITTED = "committed"
    AWAITING_ACTION = "awaiting_action"
    AWAITING_OBSERVATION = "awaiting_observation"
    RECONCILED = "reconciled"
    CONTINUING = "continuing"
    COMPLETED = "completed"
    SUSPENDED = "suspended"
    BLOCKED = "blocked"
    FALSIFIED = "falsified"
    BUDGET_EXHAUSTED = "budget_exhausted"


class HotWorkflowOutcomeStatus(str, Enum):
    COMPLETED = "completed"
    FALSIFIED = "falsified"
    BLOCKED = "blocked"
    SUSPENDED = "suspended"
    BUDGET_EXHAUSTED = "budget_exhausted"


class WorkflowBounds(FrozenModel):
    maximum_global_executable_candidates: int = Field(strict=True, ge=1, le=3)
    maximum_symbolic_depth: int = Field(strict=True, ge=1)
    maximum_nodes: int = Field(strict=True, ge=1)
    maximum_edges: int = Field(strict=True, ge=0)
    maximum_branches: int = Field(strict=True, ge=0)
    maximum_children: int = Field(strict=True, ge=0)
    maximum_actions: int = Field(strict=True, ge=0)
    maximum_parallel_analyses: int = Field(strict=True, ge=0)
    maximum_retrospective_transitions: int = Field(strict=True, ge=0)
    schema_version: Ref = "yf_arc3_v5.dynamic_workflow_bounds.v1"


class GameMethodActivityContract(FrozenModel):
    """Source-authored, scene-free requirements of one reusable activity.

    This is a partial method definition, not an execution record or a proof
    that all mechanisms of a game are covered. Canonical claims hold its
    learning evidence; the existing dynamic workflow IR owns execution.
    """

    method_ref: Ref
    activity_ref: Ref
    component_refs: tuple[Ref, ...] = Field(min_length=1, max_length=64)
    objective_schema_ref: Ref
    required_role_refs: tuple[Ref, ...] = Field(min_length=1, max_length=64)
    required_mechanism_refs: tuple[Ref, ...] = Field(min_length=1, max_length=64)
    observed_control_role_refs: tuple[Ref, ...] = Field(default=(), max_length=64)
    requires_predicate_refs: tuple[Ref, ...] = Field(min_length=1, max_length=64)
    provides_predicate_refs: tuple[Ref, ...] = Field(min_length=1, max_length=64)
    preserves_invariant_refs: tuple[Ref, ...] = Field(min_length=1, max_length=64)
    falsifier_refs: tuple[Ref, ...] = Field(min_length=1, max_length=64)
    schema_version: Ref = "yf_arc3_v5.game_method_activity.v1"

    @model_validator(mode="after")
    def validate_reference_sets(self) -> "GameMethodActivityContract":
        if not set(self.observed_control_role_refs).issubset(self.required_role_refs):
            raise ValueError("observed control roles must be declared activity roles")
        for name in (
            "component_refs", "required_role_refs", "required_mechanism_refs", "requires_predicate_refs",
            "provides_predicate_refs", "preserves_invariant_refs", "falsifier_refs", "observed_control_role_refs",
        ):
            _require_canonical_refs(getattr(self, name), name)
        return self


class ColdWorkflowComponent(FrozenModel):
    component_ref: Ref
    component_family_ref: Ref
    served_migration_goal_ref: Ref
    workflow_definition_ref: Ref | None = None
    runtime_dispatchable: bool = True
    runtime_profile_refs: tuple[Ref, ...] = ()
    input_contract: FrozenMap = Field(default_factory=FrozenMap)
    requires_predicate_refs: tuple[Ref, ...] = ()
    provides_predicate_refs: tuple[Ref, ...] = ()
    preserves_invariant_refs: tuple[Ref, ...] = ()
    possible_effect_refs: tuple[Ref, ...] = ()
    falsifier_refs: tuple[Ref, ...] = ()
    resource_dimensions: tuple[Ref, ...] = ()
    maximum_symbolic_depth: int = Field(strict=True, ge=1)
    maximum_action_count: int = Field(strict=True, ge=0)
    action_release_kind: ActionReleaseKind
    reconciliation_contract_ref: Ref
    source_hash: Ref
    semantic_hash: Ref
    transitive_source_hash: Ref
    schema_version: Ref = "yf_arc3_v5.cold_workflow_component.v1"

    @model_validator(mode="after")
    def validate_component(self) -> "ColdWorkflowComponent":
        for values, description in (
            (self.requires_predicate_refs, "component requirements"),
            (self.provides_predicate_refs, "component provisions"),
            (self.preserves_invariant_refs, "component invariants"),
            (self.possible_effect_refs, "component effects"),
            (self.falsifier_refs, "component falsifiers"),
            (self.resource_dimensions, "component resource dimensions"),
            (self.runtime_profile_refs, "component runtime profiles"),
        ):
            _require_canonical_refs(values, description)
        _require_sha256(self.source_hash, "component source hash")
        _require_sha256(self.semantic_hash, "component semantic hash")
        _require_sha256(
            self.transitive_source_hash,
            "component transitive source hash",
        )
        if (
            self.action_release_kind is ActionReleaseKind.NONE
            and self.maximum_action_count != 0
        ):
            raise ValueError("a non-action component must declare zero actions")
        if (
            self.action_release_kind is ActionReleaseKind.SINGLE_PRIMITIVE
            and self.maximum_action_count < 1
        ):
            raise ValueError("an action component must declare a positive action bound")
        return self


class DynamicWorkflowNode(FrozenModel):
    node_ref: Ref
    component_ref: Ref
    served_goal_refs: tuple[Ref, ...] = Field(min_length=1)
    bound_input_refs: FrozenMap = Field(default_factory=FrozenMap)
    expected_output_predicate_refs: tuple[Ref, ...] = ()
    local_precondition_refs: tuple[Ref, ...] = ()
    local_invariant_refs: tuple[Ref, ...] = ()
    local_falsifier_refs: tuple[Ref, ...] = ()
    local_resource_bound: FrozenMap
    parent_node_ref: Ref | None = None
    child_workflow_policy_ref: Ref | None = None
    schema_version: Ref = "yf_arc3_v5.dynamic_workflow_node.v1"

    @model_validator(mode="after")
    def validate_node(self) -> "DynamicWorkflowNode":
        for values, description in (
            (self.served_goal_refs, "node served goals"),
            (self.expected_output_predicate_refs, "node expected outputs"),
            (self.local_precondition_refs, "node preconditions"),
            (self.local_invariant_refs, "node invariants"),
            (self.local_falsifier_refs, "node falsifiers"),
        ):
            _require_canonical_refs(values, description)
        if self.parent_node_ref == self.node_ref:
            raise ValueError("a workflow node cannot parent itself")
        return self


class DynamicWorkflowEdge(FrozenModel):
    edge_ref: Ref
    provider_node_ref: Ref
    consumer_node_ref: Ref
    required_predicate_ref: Ref
    provided_predicate_ref: Ref
    current_absence_evidence_ref: Ref
    composition_principle_ref: Ref
    preservation_evidence_refs: tuple[Ref, ...]
    falsifier_ref: Ref
    schema_version: Ref = "yf_arc3_v5.dynamic_workflow_edge.v1"

    @model_validator(mode="after")
    def validate_edge(self) -> "DynamicWorkflowEdge":
        if self.provider_node_ref == self.consumer_node_ref:
            raise ValueError("a workflow dependency cannot be a self edge")
        if self.required_predicate_ref != self.provided_predicate_ref:
            raise ValueError("workflow edge predicates must unify exactly")
        _require_canonical_refs(
            self.preservation_evidence_refs,
            "edge preservation evidence",
        )
        return self


class WorkflowBranchCondition(FrozenModel):
    branch_ref: Ref
    source_node_ref: Ref
    predicate_ref: Ref
    true_node_ref: Ref
    false_node_ref: Ref
    falsifier_ref: Ref
    schema_version: Ref = "yf_arc3_v5.dynamic_workflow_branch.v1"


class WorkflowCompletionContract(FrozenModel):
    required_terminal_predicate_refs: tuple[Ref, ...] = Field(min_length=1)
    observable_completion_evidence_refs: tuple[Ref, ...] = Field(min_length=1)
    preserved_condition_refs: tuple[Ref, ...] = ()
    forbidden_overshoot_refs: tuple[Ref, ...] = ()
    resource_bound_refs: tuple[Ref, ...] = ()
    schema_version: Ref = "yf_arc3_v5.workflow_completion_contract.v1"

    @model_validator(mode="after")
    def validate_contract(self) -> "WorkflowCompletionContract":
        for values, description in (
            (self.required_terminal_predicate_refs, "terminal predicates"),
            (self.observable_completion_evidence_refs, "completion evidence"),
            (self.preserved_condition_refs, "preserved completion conditions"),
            (self.forbidden_overshoot_refs, "forbidden overshoots"),
            (self.resource_bound_refs, "completion resource bounds"),
        ):
            _require_canonical_refs(values, description)
        return self


class CanonicalWorkflowDerivation(FrozenModel):
    cold_component_transitive_hashes: FrozenMap
    composition_law_hashes: FrozenMap
    canonical_goal_bindings: FrozenMap
    declared_order_ref: Ref
    declared_node_order_refs: tuple[Ref, ...] = Field(min_length=1)
    schema_version: Ref = "yf_arc3_v5.dynamic_workflow_derivation.v1"

    @model_validator(mode="after")
    def validate_hashes(self) -> "CanonicalWorkflowDerivation":
        require_unique(self.declared_node_order_refs, "declared workflow node order")
        for field_name, values in (
            ("cold component transitive hashes", self.cold_component_transitive_hashes),
            ("composition law hashes", self.composition_law_hashes),
        ):
            for value in sorted(values.values()):
                if not isinstance(value, str):
                    raise ValueError(f"{field_name} must map references to hashes")
                _require_sha256(value, field_name)
        return self


class DynamicWorkflowIR(FrozenModel):
    schema_version: Ref = "yf_arc3_v5.dynamic_workflow_ir.v1"
    workflow_ir_ref: Ref
    root_goal_ref: Ref
    source_state_revision: NonNegativeRevision
    source_frame_ref: Ref
    cold_registry_hash: Ref
    drm_declaration_hash: Ref
    synthesis_policy_ref: Ref
    nodes: tuple[DynamicWorkflowNode, ...] = Field(min_length=1)
    dependency_edges: tuple[DynamicWorkflowEdge, ...] = ()
    branch_conditions: tuple[WorkflowBranchCondition, ...] = ()
    preserved_invariant_refs: tuple[Ref, ...] = ()
    global_falsifier_refs: tuple[Ref, ...] = Field(min_length=1)
    global_resource_bounds: WorkflowBounds
    completion_contract: WorkflowCompletionContract
    canonical_derivation: CanonicalWorkflowDerivation
    semantic_hash: Ref

    @model_validator(mode="after")
    def validate_ir(self) -> "DynamicWorkflowIR":
        _require_sha256(self.cold_registry_hash, "cold registry hash")
        _require_sha256(self.drm_declaration_hash, "DRM declaration hash")
        _require_sha256(self.semantic_hash, "dynamic workflow semantic hash")
        _require_canonical_refs(
            self.preserved_invariant_refs,
            "workflow preserved invariants",
        )
        _require_canonical_refs(
            self.global_falsifier_refs,
            "workflow global falsifiers",
        )
        node_refs = tuple(node.node_ref for node in self.nodes)
        edge_refs = tuple(edge.edge_ref for edge in self.dependency_edges)
        branch_refs = tuple(branch.branch_ref for branch in self.branch_conditions)
        _require_canonical_refs(node_refs, "workflow nodes")
        _require_canonical_refs(edge_refs, "workflow edges")
        _require_canonical_refs(branch_refs, "workflow branches")
        if len(self.nodes) > self.global_resource_bounds.maximum_nodes:
            raise ValueError("workflow node bound exceeded")
        if len(self.dependency_edges) > self.global_resource_bounds.maximum_edges:
            raise ValueError("workflow edge bound exceeded")
        if len(self.branch_conditions) > self.global_resource_bounds.maximum_branches:
            raise ValueError("workflow branch bound exceeded")
        known_nodes = set(node_refs)
        for node in self.nodes:
            if node.parent_node_ref is not None and node.parent_node_ref not in known_nodes:
                raise ValueError("workflow node names an unknown parent")
        for edge in self.dependency_edges:
            if (
                edge.provider_node_ref not in known_nodes
                or edge.consumer_node_ref not in known_nodes
            ):
                raise ValueError("workflow edge names an unknown node")
        for branch in self.branch_conditions:
            if not {
                branch.source_node_ref,
                branch.true_node_ref,
                branch.false_node_ref,
            }.issubset(known_nodes):
                raise ValueError("workflow branch names an unknown node")
        incoming: dict[str, int] = {node_ref: 0 for node_ref in node_refs}
        outgoing: dict[str, list[str]] = {node_ref: [] for node_ref in node_refs}
        for edge in self.dependency_edges:
            incoming[edge.consumer_node_ref] += 1
            outgoing[edge.provider_node_ref].append(edge.consumer_node_ref)
        frontier = sorted(node_ref for node_ref, count in incoming.items() if count == 0)
        visited = 0
        while frontier:
            current = frontier.pop(0)
            visited += 1
            for consumer in sorted(outgoing[current]):
                incoming[consumer] -= 1
                if incoming[consumer] == 0:
                    frontier.append(consumer)
                    frontier.sort()
        if visited != len(node_refs):
            raise ValueError("workflow dependency graph must be acyclic")
        if set(self.canonical_derivation.declared_node_order_refs) != set(node_refs):
            raise ValueError("declared workflow node order must cover every node exactly once")
        if dynamic_workflow_semantic_hash(self) != self.semantic_hash:
            raise ValueError("dynamic workflow semantic hash mismatch")
        return self


def dynamic_workflow_semantic_hash(value: DynamicWorkflowIR) -> str:
    return stable_digest(
        {
            "schema_version": value.schema_version,
            "root_goal_ref": value.root_goal_ref,
            "source_state_revision": value.source_state_revision,
            "source_frame_ref": value.source_frame_ref,
            "cold_registry_hash": value.cold_registry_hash,
            "drm_declaration_hash": value.drm_declaration_hash,
            "synthesis_policy_ref": value.synthesis_policy_ref,
            "nodes": value.nodes,
            "dependency_edges": value.dependency_edges,
            "branch_conditions": value.branch_conditions,
            "preserved_invariant_refs": value.preserved_invariant_refs,
            "global_falsifier_refs": value.global_falsifier_refs,
            "global_resource_bounds": value.global_resource_bounds,
            "completion_contract": value.completion_contract,
            "canonical_derivation": value.canonical_derivation,
        }
    )


def build_dynamic_workflow_ir(
    *,
    root_goal_ref: Ref,
    source_state_revision: NonNegativeRevision,
    source_frame_ref: Ref,
    cold_registry_hash: Ref,
    drm_declaration_hash: Ref,
    synthesis_policy_ref: Ref,
    nodes: tuple[DynamicWorkflowNode, ...],
    dependency_edges: tuple[DynamicWorkflowEdge, ...],
    branch_conditions: tuple[WorkflowBranchCondition, ...],
    preserved_invariant_refs: tuple[Ref, ...],
    global_falsifier_refs: tuple[Ref, ...],
    global_resource_bounds: WorkflowBounds,
    completion_contract: WorkflowCompletionContract,
    canonical_derivation: CanonicalWorkflowDerivation,
) -> DynamicWorkflowIR:
    values = {
        "root_goal_ref": root_goal_ref,
        "source_state_revision": source_state_revision,
        "source_frame_ref": source_frame_ref,
        "cold_registry_hash": cold_registry_hash,
        "drm_declaration_hash": drm_declaration_hash,
        "synthesis_policy_ref": synthesis_policy_ref,
        "nodes": tuple(sorted(nodes, key=lambda item: item.node_ref)),
        "dependency_edges": tuple(
            sorted(dependency_edges, key=lambda item: item.edge_ref)
        ),
        "branch_conditions": tuple(
            sorted(branch_conditions, key=lambda item: item.branch_ref)
        ),
        "preserved_invariant_refs": tuple(sorted(preserved_invariant_refs)),
        "global_falsifier_refs": tuple(sorted(global_falsifier_refs)),
        "global_resource_bounds": global_resource_bounds,
        "completion_contract": completion_contract,
        "canonical_derivation": canonical_derivation,
    }
    provisional = DynamicWorkflowIR.model_construct(
        workflow_ir_ref="dynamic-ir:pending",
        semantic_hash="0" * 64,
        **values,
    )
    semantic_hash = dynamic_workflow_semantic_hash(provisional)
    return DynamicWorkflowIR(
        workflow_ir_ref=f"dynamic-ir:{semantic_hash[:24]}",
        semantic_hash=semantic_hash,
        **values,
    )


class HotWorkflowInstance(FrozenModel):
    workflow_instance_ref: Ref
    workflow_ir_hash: Ref
    root_goal_ref: Ref
    parent_instance_ref: Ref | None = None
    child_instance_refs: tuple[Ref, ...] = ()
    completed_node_refs: tuple[Ref, ...] = ()
    inactive_node_refs: tuple[Ref, ...] = ()
    released_action_node_refs: tuple[Ref, ...] = ()
    reconciled_action_node_refs: tuple[Ref, ...] = ()
    runnable_leaf_ref: Ref | None = None
    awaiting_observation_contract: Ref | None = None
    current_revision: NonNegativeRevision
    frame_ref: Ref
    remaining_global_bounds: WorkflowBounds
    status: HotWorkflowStatus
    schema_version: Ref = "yf_arc3_v5.hot_workflow_instance.v1"

    @model_validator(mode="after")
    def validate_instance(self) -> "HotWorkflowInstance":
        _require_sha256(self.workflow_ir_hash, "hot workflow IR hash")
        _require_canonical_refs(self.child_instance_refs, "hot child instances")
        _require_canonical_refs(self.completed_node_refs, "completed hot nodes")
        _require_canonical_refs(self.inactive_node_refs, "inactive hot nodes")
        if set(self.completed_node_refs).intersection(self.inactive_node_refs):
            raise ValueError("a hot node cannot be both completed and inactive")
        _require_canonical_refs(
            self.released_action_node_refs,
            "released hot action nodes",
        )
        _require_canonical_refs(
            self.reconciled_action_node_refs,
            "reconciled hot action nodes",
        )
        if not set(self.reconciled_action_node_refs).issubset(
            self.released_action_node_refs
        ):
            raise ValueError("a reconciled action node must first be released")
        if self.workflow_instance_ref in self.child_instance_refs:
            raise ValueError("a hot workflow cannot contain itself as a child")
        if (
            self.status is HotWorkflowStatus.AWAITING_OBSERVATION
            and self.awaiting_observation_contract is None
        ):
            raise ValueError("awaiting observation requires an exact contract")
        return self


class HotWorkflowCheckpoint(FrozenModel):
    workflow_ir_hash: Ref
    workflow_instance_ref: Ref
    parent_instance_ref: Ref | None = None
    current_node_ref: Ref | None = None
    completed_node_refs: tuple[Ref, ...] = ()
    inactive_node_refs: tuple[Ref, ...] = ()
    released_action_node_refs: tuple[Ref, ...] = ()
    reconciled_action_node_refs: tuple[Ref, ...] = ()
    open_child_refs: tuple[Ref, ...] = ()
    state_revision: NonNegativeRevision
    frame_ref: Ref
    remaining_bounds: WorkflowBounds
    awaited_observation_contract: Ref | None = None
    status: HotWorkflowStatus = HotWorkflowStatus.SUSPENDED
    schema_version: Ref = "yf_arc3_v5.hot_workflow_checkpoint.v1"

    @model_validator(mode="after")
    def validate_checkpoint(self) -> "HotWorkflowCheckpoint":
        _require_sha256(self.workflow_ir_hash, "hot checkpoint IR hash")
        _require_canonical_refs(self.completed_node_refs, "checkpoint completed nodes")
        _require_canonical_refs(self.inactive_node_refs, "checkpoint inactive nodes")
        if set(self.completed_node_refs).intersection(self.inactive_node_refs):
            raise ValueError("a checkpoint node cannot be completed and inactive")
        _require_canonical_refs(
            self.released_action_node_refs,
            "checkpoint released action nodes",
        )
        _require_canonical_refs(
            self.reconciled_action_node_refs,
            "checkpoint reconciled action nodes",
        )
        if not set(self.reconciled_action_node_refs).issubset(
            self.released_action_node_refs
        ):
            raise ValueError("checkpoint reconciles an unreleased action node")
        _require_canonical_refs(self.open_child_refs, "checkpoint open children")
        return self


class HotWorkflowOutcome(FrozenModel):
    status: HotWorkflowOutcomeStatus
    satisfied_goal_refs: tuple[Ref, ...] = ()
    produced_claim_refs: tuple[Ref, ...] = ()
    contradicted_claim_refs: tuple[Ref, ...] = ()
    opened_subgoal_refs: tuple[Ref, ...] = ()
    preserved_invariant_refs: tuple[Ref, ...] = ()
    consumed_resources: FrozenMap = Field(default_factory=FrozenMap)
    checkpoint_ref: Ref
    evidence_refs: tuple[Ref, ...] = Field(min_length=1)
    schema_version: Ref = "yf_arc3_v5.hot_workflow_outcome.v1"

    @model_validator(mode="after")
    def validate_outcome(self) -> "HotWorkflowOutcome":
        for values, description in (
            (self.satisfied_goal_refs, "outcome goals"),
            (self.produced_claim_refs, "outcome produced claims"),
            (self.contradicted_claim_refs, "outcome contradicted claims"),
            (self.opened_subgoal_refs, "outcome subgoals"),
            (self.preserved_invariant_refs, "outcome invariants"),
            (self.evidence_refs, "outcome evidence"),
        ):
            _require_canonical_refs(values, description)
        return self


class WorkflowLanguageGap(FrozenModel):
    missing_predicate_ref: Ref
    observed_evidence_refs: tuple[Ref, ...]
    attempted_component_family_refs: tuple[Ref, ...]
    exact_failure_reason_refs: tuple[Ref, ...]
    extension_proposal_ref: Ref | None = None
    falsifier_ref: Ref
    schema_version: Ref = "yf_arc3_v5.workflow_language_gap.v1"

    @model_validator(mode="after")
    def validate_gap(self) -> "WorkflowLanguageGap":
        for values, description in (
            (self.observed_evidence_refs, "gap evidence"),
            (self.attempted_component_family_refs, "gap component families"),
            (self.exact_failure_reason_refs, "gap failure reasons"),
        ):
            _require_canonical_refs(values, description)
        return self
