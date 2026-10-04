"""Mechanical validation for a DRM-declared obligation DAG.

The models transport canonical references and enforce only finite structural
properties.  They do not create an obligation, dependency, activity, safety
gate, or local ordering; those meanings remain declared by Logos/DRM/SRC.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from agents.yf_arc3_v5.logos.types import (
    FrozenModel,
    NonNegativeRevision,
    ObligationEdgeKind,
    ObligationKind,
    ObligationStatus,
    PrioritySafetyGate,
    Ref,
    require_unique,
)


class ObligationNode(FrozenModel):
    obligation_ref: Ref
    kind: ObligationKind
    status: ObligationStatus
    premise_claim_refs: tuple[Ref, ...] = ()
    requires_obligation_refs: tuple[Ref, ...] = ()
    provides_predicate_refs: tuple[Ref, ...] = ()
    preserves_invariant_refs: tuple[Ref, ...] = ()
    conflicts_with_refs: tuple[Ref, ...] = ()
    completion_evidence_refs: tuple[Ref, ...] = ()
    falsifier_refs: tuple[Ref, ...] = ()
    resource_bound_refs: tuple[Ref, ...] = ()
    eligible_activity_refs: tuple[Ref, ...] = ()
    local_selection_policy_ref: Ref | None = None
    beneficiary_goal_refs: tuple[Ref, ...] = ()
    blocking_predicate_refs: tuple[Ref, ...] = ()
    readiness_evidence_refs: tuple[Ref, ...] = ()
    schema_version: Ref = "yf_arc3_v5.priority_obligation_node.v1"

    @model_validator(mode="after")
    def validate_declared_lifecycle_evidence(self) -> "ObligationNode":
        for refs, label in (
            (self.premise_claim_refs, "obligation premises"),
            (self.requires_obligation_refs, "obligation requirements"),
            (self.provides_predicate_refs, "obligation provided predicates"),
            (self.preserves_invariant_refs, "obligation invariants"),
            (self.conflicts_with_refs, "obligation conflicts"),
            (self.completion_evidence_refs, "obligation completion evidence"),
            (self.falsifier_refs, "obligation falsifiers"),
            (self.resource_bound_refs, "obligation resource bounds"),
            (self.eligible_activity_refs, "obligation activities"),
            (self.beneficiary_goal_refs, "obligation beneficiary goals"),
            (self.blocking_predicate_refs, "obligation blockers"),
            (self.readiness_evidence_refs, "obligation readiness evidence"),
        ):
            require_unique(refs, label)
        if self.obligation_ref in self.requires_obligation_refs:
            raise ValueError("an obligation cannot require itself")
        if self.status is ObligationStatus.READY:
            if self.blocking_predicate_refs:
                raise ValueError("a ready obligation cannot retain a blocker")
            if not self.readiness_evidence_refs:
                raise ValueError("a ready obligation requires declared readiness evidence")
            if not self.eligible_activity_refs or self.local_selection_policy_ref is None:
                raise ValueError("a ready obligation requires a declared local producer")
        if self.status is ObligationStatus.BLOCKED and not self.blocking_predicate_refs:
            raise ValueError("a blocked obligation requires a named blocking predicate")
        if self.status is ObligationStatus.SATISFIED and not self.completion_evidence_refs:
            raise ValueError("a satisfied obligation requires completion evidence")
        if self.status is ObligationStatus.FALSIFIED and not self.falsifier_refs:
            raise ValueError("a falsified obligation requires a declared falsifier")
        return self


class ObligationEdge(FrozenModel):
    edge_ref: Ref
    edge_kind: ObligationEdgeKind
    source_obligation_ref: Ref
    target_obligation_ref: Ref
    reason_refs: tuple[Ref, ...] = Field(min_length=1)
    falsifier_refs: tuple[Ref, ...] = Field(min_length=1)
    active: bool = True
    schema_version: Ref = "yf_arc3_v5.priority_obligation_edge.v1"

    @model_validator(mode="after")
    def validate_edge(self) -> "ObligationEdge":
        if self.source_obligation_ref == self.target_obligation_ref:
            raise ValueError("an obligation edge cannot be a self edge")
        require_unique(self.reason_refs, "obligation edge reasons")
        require_unique(self.falsifier_refs, "obligation edge falsifiers")
        return self


class PriorityObligationGraph(FrozenModel):
    nodes: tuple[ObligationNode, ...] = Field(max_length=256)
    edges: tuple[ObligationEdge, ...] = Field(max_length=768)
    source_state_revision: NonNegativeRevision
    schema_version: Ref = "yf_arc3_v5.priority_obligation_graph.v1"

    @model_validator(mode="after")
    def validate_graph(self) -> "PriorityObligationGraph":
        node_refs = tuple(node.obligation_ref for node in self.nodes)
        edge_refs = tuple(edge.edge_ref for edge in self.edges)
        require_unique(node_refs, "priority obligation nodes")
        require_unique(edge_refs, "priority obligation edges")
        known = set(node_refs)
        if any(
            edge.source_obligation_ref not in known
            or edge.target_obligation_ref not in known
            for edge in self.edges
        ):
            raise ValueError("obligation edges must reference known nodes")
        if any(
            ref not in known
            for node in self.nodes
            for ref in (*node.requires_obligation_refs, *node.conflicts_with_refs)
        ):
            raise ValueError("obligation node relations must reference known nodes")

        active_requirements = {
            (edge.source_obligation_ref, edge.target_obligation_ref)
            for edge in self.edges
            if edge.active and edge.edge_kind is ObligationEdgeKind.REQUIRES
        }
        declared_requirements = {
            (node.obligation_ref, target_ref)
            for node in self.nodes
            for target_ref in node.requires_obligation_refs
        }
        if active_requirements != declared_requirements:
            raise ValueError("requires edges must exactly match node requirements")

        adjacency = {ref: set() for ref in node_refs}
        for source_ref, target_ref in active_requirements:
            adjacency[source_ref].add(target_ref)
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(ref: str) -> None:
            if ref in visiting:
                raise ValueError("obligation requirements must be acyclic")
            if ref in visited:
                return
            visiting.add(ref)
            for target_ref in sorted(adjacency[ref]):
                visit(target_ref)
            visiting.remove(ref)
            visited.add(ref)

        for ref in sorted(node_refs):
            visit(ref)

        by_ref = {node.obligation_ref: node for node in self.nodes}
        for node in self.nodes:
            if node.status is not ObligationStatus.READY:
                continue
            if any(
                by_ref[ref].status is not ObligationStatus.SATISFIED
                for ref in node.requires_obligation_refs
            ):
                raise ValueError("a ready obligation has an unsatisfied requirement")
        return self


class ObligationFrontierProjection(FrozenModel):
    source_state_revision: NonNegativeRevision
    active_safety_gate: PrioritySafetyGate
    active_obligation_ref: Ref | None = None
    ready_frontier_refs: tuple[Ref, ...] = Field(default=(), max_length=256)
    blocked_obligation_refs: tuple[Ref, ...] = Field(default=(), max_length=256)
    blocking_predicate_refs: tuple[Ref, ...] = Field(default=(), max_length=256)
    activity_ref: Ref | None = None
    local_selection_policy_ref: Ref | None = None
    first_discriminating_criterion_ref: Ref | None = None
    preserved_invariant_refs: tuple[Ref, ...] = Field(default=(), max_length=256)
    expected_effect_ref: Ref | None = None
    falsifier_refs: tuple[Ref, ...] = Field(default=(), max_length=256)
    goal_route_stop_lifecycle_ref: Ref | None = None
    pursued_goal_ref: Ref | None = None
    engaged_plan_or_experiment_ref: Ref | None = None
    stop_condition_ref: Ref | None = None
    mechanism_use_refs: tuple[Ref, ...] = Field(default=(), max_length=64)
    mechanism_application_test_refs: tuple[Ref, ...] = Field(
        default=(), max_length=64
    )
    mechanism_utility_success_evidence_refs: tuple[Ref, ...] = Field(
        default=(), max_length=256
    )
    mechanism_utility_failure_evidence_refs: tuple[Ref, ...] = Field(
        default=(), max_length=256
    )
    mechanism_utility_stop_condition_refs: tuple[Ref, ...] = Field(
        default=(), max_length=256
    )
    mechanism_utility_reconciliation_refs: tuple[Ref, ...] = Field(
        default=(), max_length=64
    )
    graph_semantic_hash: Ref
    schema_version: Ref = "yf_arc3_v5.obligation_frontier_projection.v2"

    @model_validator(mode="after")
    def validate_projection(self) -> "ObligationFrontierProjection":
        for refs, label in (
            (self.ready_frontier_refs, "ready frontier"),
            (self.blocked_obligation_refs, "blocked obligations"),
            (self.blocking_predicate_refs, "blocking predicates"),
            (self.preserved_invariant_refs, "preserved invariants"),
            (self.falsifier_refs, "frontier falsifiers"),
            (self.mechanism_use_refs, "frontier mechanism uses"),
            (
                self.mechanism_application_test_refs,
                "frontier mechanism application tests",
            ),
            (
                self.mechanism_utility_success_evidence_refs,
                "frontier mechanism utility successes",
            ),
            (
                self.mechanism_utility_failure_evidence_refs,
                "frontier mechanism utility failures",
            ),
            (
                self.mechanism_utility_stop_condition_refs,
                "frontier mechanism utility stops",
            ),
            (
                self.mechanism_utility_reconciliation_refs,
                "frontier mechanism utility reconciliations",
            ),
        ):
            require_unique(refs, label)
        if set(self.ready_frontier_refs) & set(self.blocked_obligation_refs):
            raise ValueError("an obligation cannot be both ready and blocked")
        if self.active_obligation_ref is not None:
            if self.active_obligation_ref not in self.ready_frontier_refs:
                raise ValueError("the active obligation must belong to the ready frontier")
            if self.activity_ref is None or self.local_selection_policy_ref is None:
                raise ValueError("an active obligation requires a declared local activity")
            if self.expected_effect_ref is None or not self.falsifier_refs:
                raise ValueError("an active obligation requires effect and falsifier evidence")
        elif self.activity_ref is not None:
            raise ValueError("an activity cannot be projected without an active obligation")
        lifecycle_values = (
            self.goal_route_stop_lifecycle_ref,
            self.pursued_goal_ref,
            self.engaged_plan_or_experiment_ref,
            self.stop_condition_ref,
        )
        if any(value is not None for value in lifecycle_values) and not all(
            value is not None for value in lifecycle_values
        ):
            raise ValueError("goal-route-stop lifecycle fields must be complete")
        mechanism_utility_contract = (
            self.mechanism_use_refs,
            self.mechanism_application_test_refs,
            self.mechanism_utility_success_evidence_refs,
            self.mechanism_utility_failure_evidence_refs,
            self.mechanism_utility_stop_condition_refs,
            self.mechanism_utility_reconciliation_refs,
        )
        if any(mechanism_utility_contract) and not all(mechanism_utility_contract):
            raise ValueError(
                "a frontier mechanism utility contract must be complete or absent"
            )
        if self.blocked_obligation_refs and not self.blocking_predicate_refs:
            raise ValueError("blocked obligations require named predicates")
        return self
