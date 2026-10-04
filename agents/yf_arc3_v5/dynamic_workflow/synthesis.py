"""Pure bounded materialization of DRM/SRC-declared workflow orders."""

from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from agents.yf_arc3_v5.dynamic_workflow.contracts import (
    CanonicalWorkflowDerivation,
    DynamicWorkflowEdge,
    DynamicWorkflowIR,
    DynamicWorkflowNode,
    WorkflowBounds,
    WorkflowCompletionContract,
    build_dynamic_workflow_ir,
)
from agents.yf_arc3_v5.dynamic_workflow.registry import (
    ColdComponentRegistry,
    DynamicWorkflowRegistryError,
)
from agents.yf_arc3_v5.logos.types import (
    FrozenMap,
    FrozenModel,
    NonNegativeRevision,
    Ref,
    require_unique,
    stable_digest,
)


class SynthesisFailureCode(str, Enum):
    UNKNOWN_ROOT_COMPONENT = "unknown_root_component"
    MISSING_DECLARED_PRODUCER = "missing_declared_producer"
    NO_EXACT_PROVIDER = "no_exact_provider"
    MISSING_ABSENCE_EVIDENCE = "missing_absence_evidence"
    MISSING_PRESERVATION_EVIDENCE = "missing_preservation_evidence"
    CYCLIC_COMPONENT_DEPENDENCY = "cyclic_component_dependency"
    SYMBOLIC_DEPTH_BOUND = "symbolic_depth_bound"
    NODE_BOUND = "node_bound"
    EDGE_BOUND = "edge_bound"


class DeclaredPredicateProducerOrder(FrozenModel):
    required_predicate_ref: Ref
    eligible_component_refs: tuple[Ref, ...] = Field(min_length=1)
    composition_principle_ref: Ref
    falsifier_ref: Ref
    composition_law_hash: Ref
    schema_version: Ref = "yf_arc3_v5.declared_predicate_producer_order.v1"

    @model_validator(mode="after")
    def validate_order(self) -> "DeclaredPredicateProducerOrder":
        require_unique(self.eligible_component_refs, "eligible component order")
        if len(self.composition_law_hash) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.composition_law_hash
        ):
            raise ValueError("composition law hash must be lowercase SHA-256")
        return self


class DeclaredCompositionOrder(FrozenModel):
    order_ref: Ref
    root_component_ref: Ref
    producer_orders: tuple[DeclaredPredicateProducerOrder, ...] = ()
    schema_version: Ref = "yf_arc3_v5.declared_composition_order.v1"

    @model_validator(mode="after")
    def validate_producers(self) -> "DeclaredCompositionOrder":
        require_unique(
            tuple(item.required_predicate_ref for item in self.producer_orders),
            "declared predicate producer orders",
        )
        return self


class WorkflowSynthesisSnapshot(FrozenModel):
    source_state_revision: NonNegativeRevision
    source_frame_ref: Ref
    root_goal_ref: Ref
    active_goal_frontier_hash: Ref
    satisfied_predicate_evidence: FrozenMap = Field(default_factory=FrozenMap)
    absent_predicate_evidence: FrozenMap = Field(default_factory=FrozenMap)
    invariant_evidence: FrozenMap = Field(default_factory=FrozenMap)
    bound_input_refs_by_component: FrozenMap = Field(default_factory=FrozenMap)
    canonical_goal_bindings: FrozenMap = Field(default_factory=FrozenMap)
    small_descriptive_delta: FrozenMap = Field(default_factory=FrozenMap)
    schema_version: Ref = "yf_arc3_v5.workflow_synthesis_snapshot.v1"


class DynamicWorkflowSynthesisRequest(FrozenModel):
    cold_registry_hash: Ref
    drm_declaration_hash: Ref
    synthesis_policy_ref: Ref
    snapshot: WorkflowSynthesisSnapshot
    declared_orders: tuple[DeclaredCompositionOrder, ...] = Field(
        min_length=1,
        max_length=3,
    )
    global_resource_bounds: WorkflowBounds
    completion_contract: WorkflowCompletionContract
    preserved_invariant_refs: tuple[Ref, ...] = ()
    global_falsifier_refs: tuple[Ref, ...] = Field(min_length=1)
    schema_version: Ref = "yf_arc3_v5.dynamic_workflow_synthesis_request.v1"

    @model_validator(mode="after")
    def validate_request(self) -> "DynamicWorkflowSynthesisRequest":
        require_unique(
            tuple(item.order_ref for item in self.declared_orders),
            "declared composition orders",
        )
        if len(self.declared_orders) > (
            self.global_resource_bounds.maximum_global_executable_candidates
        ):
            raise ValueError("declared composition orders exceed the global bound")
        for value, description in (
            (self.cold_registry_hash, "cold registry hash"),
            (self.drm_declaration_hash, "DRM declaration hash"),
            (self.snapshot.active_goal_frontier_hash, "goal frontier hash"),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{description} must be lowercase SHA-256")
        if self.preserved_invariant_refs != tuple(
            sorted(set(self.preserved_invariant_refs))
        ):
            raise ValueError("preserved invariants must be unique and canonical")
        if self.global_falsifier_refs != tuple(
            sorted(set(self.global_falsifier_refs))
        ):
            raise ValueError("global falsifiers must be unique and canonical")
        return self


class DynamicWorkflowSynthesisFailure(FrozenModel):
    order_ref: Ref
    code: SynthesisFailureCode
    component_ref: Ref | None = None
    predicate_ref: Ref | None = None
    detail_refs: tuple[Ref, ...] = ()
    schema_version: Ref = "yf_arc3_v5.dynamic_workflow_synthesis_failure.v1"


class DynamicWorkflowSynthesisResult(FrozenModel):
    candidate_workflows: tuple[DynamicWorkflowIR, ...]
    rejected_orders: tuple[DynamicWorkflowSynthesisFailure, ...]
    viable_candidate_count: int = Field(strict=True, ge=0, le=3)
    source_state_revision: NonNegativeRevision
    source_frame_ref: Ref
    schema_version: Ref = "yf_arc3_v5.dynamic_workflow_synthesis_result.v1"

    @model_validator(mode="after")
    def validate_count(self) -> "DynamicWorkflowSynthesisResult":
        if self.viable_candidate_count != len(self.candidate_workflows):
            raise ValueError("viable workflow candidate count mismatch")
        return self


class DynamicWorkflowSelectionFacts(FrozenModel):
    """Compact immutable facts exposed to the declared disposition policy."""

    viable_candidate_count: int = Field(strict=True, ge=0, le=3)
    candidate_semantic_hashes: tuple[Ref, ...] = ()
    rejected_order_refs: tuple[Ref, ...] = ()
    source_state_revision: NonNegativeRevision
    source_frame_ref: Ref
    schema_version: Ref = "yf_arc3_v5.dynamic_workflow_selection_facts.v1"

    @model_validator(mode="after")
    def validate_selection_facts(self) -> "DynamicWorkflowSelectionFacts":
        if self.viable_candidate_count != len(self.candidate_semantic_hashes):
            raise ValueError("candidate semantic-hash count mismatch")
        require_unique(
            self.candidate_semantic_hashes,
            "dynamic workflow candidate semantic hashes",
        )
        require_unique(self.rejected_order_refs, "rejected dynamic workflow orders")
        return self


def reduce_dynamic_workflow_synthesis_facts(
    result: DynamicWorkflowSynthesisResult,
) -> DynamicWorkflowSelectionFacts:
    """Discard executable graphs while retaining exact selection evidence."""

    return DynamicWorkflowSelectionFacts(
        viable_candidate_count=result.viable_candidate_count,
        candidate_semantic_hashes=tuple(
            candidate.semantic_hash for candidate in result.candidate_workflows
        ),
        rejected_order_refs=tuple(
            rejection.order_ref for rejection in result.rejected_orders
        ),
        source_state_revision=result.source_state_revision,
        source_frame_ref=result.source_frame_ref,
    )


class _OrderRejected(Exception):
    def __init__(
        self,
        code: SynthesisFailureCode,
        *,
        component_ref: Ref | None = None,
        predicate_ref: Ref | None = None,
        detail_refs: tuple[Ref, ...] = (),
    ) -> None:
        self.code = code
        self.component_ref = component_ref
        self.predicate_ref = predicate_ref
        self.detail_refs = detail_refs
        super().__init__(code.value)


def _evidence_ref(values: FrozenMap, key: str) -> str | None:
    value = values.get(key)
    return value if isinstance(value, str) and value.strip() else None


def _component_input_refs(values: FrozenMap, component_ref: str) -> FrozenMap:
    value = values.get(component_ref)
    return value if isinstance(value, FrozenMap) else FrozenMap()


def _materialize_order(
    request: DynamicWorkflowSynthesisRequest,
    order: DeclaredCompositionOrder,
    components: ColdComponentRegistry,
) -> DynamicWorkflowIR:
    if components.registry_hash != request.cold_registry_hash:
        raise DynamicWorkflowRegistryError("cold component registry hash mismatch")
    producer_by_predicate = {
        item.required_predicate_ref: item for item in order.producer_orders
    }
    component_by_ref: dict[str, object] = {}
    parent_by_component: dict[str, str | None] = {}
    edge_specs: list[tuple[str, str, DeclaredPredicateProducerOrder, str, tuple[str, ...]]] = []
    active: set[str] = set()
    declared_execution_component_refs: list[str] = []

    def visit(component_ref: str, *, parent_ref: str | None, depth: int) -> None:
        if depth > request.global_resource_bounds.maximum_symbolic_depth:
            raise _OrderRejected(
                SynthesisFailureCode.SYMBOLIC_DEPTH_BOUND,
                component_ref=component_ref,
            )
        if component_ref in active:
            raise _OrderRejected(
                SynthesisFailureCode.CYCLIC_COMPONENT_DEPENDENCY,
                component_ref=component_ref,
            )
        if component_ref in component_by_ref:
            return
        try:
            component = components.resolve(component_ref)
        except DynamicWorkflowRegistryError as error:
            raise _OrderRejected(
                SynthesisFailureCode.UNKNOWN_ROOT_COMPONENT,
                component_ref=component_ref,
            ) from error
        if len(component_by_ref) >= request.global_resource_bounds.maximum_nodes:
            raise _OrderRejected(
                SynthesisFailureCode.NODE_BOUND,
                component_ref=component_ref,
            )
        active.add(component_ref)
        component_by_ref[component_ref] = component
        parent_by_component.setdefault(component_ref, parent_ref)
        for predicate_ref in component.requires_predicate_refs:
            if _evidence_ref(
                request.snapshot.satisfied_predicate_evidence,
                predicate_ref,
            ):
                continue
            absence_ref = _evidence_ref(
                request.snapshot.absent_predicate_evidence,
                predicate_ref,
            )
            if absence_ref is None:
                raise _OrderRejected(
                    SynthesisFailureCode.MISSING_ABSENCE_EVIDENCE,
                    component_ref=component_ref,
                    predicate_ref=predicate_ref,
                )
            producer_order = producer_by_predicate.get(predicate_ref)
            if producer_order is None:
                raise _OrderRejected(
                    SynthesisFailureCode.MISSING_DECLARED_PRODUCER,
                    component_ref=component_ref,
                    predicate_ref=predicate_ref,
                )
            producer = None
            for producer_ref in producer_order.eligible_component_refs:
                try:
                    candidate = components.resolve(producer_ref)
                except DynamicWorkflowRegistryError:
                    continue
                if predicate_ref in candidate.provides_predicate_refs:
                    producer = candidate
                    break
            if producer is None:
                raise _OrderRejected(
                    SynthesisFailureCode.NO_EXACT_PROVIDER,
                    component_ref=component_ref,
                    predicate_ref=predicate_ref,
                    detail_refs=producer_order.eligible_component_refs,
                )
            preservation_refs: list[str] = []
            for invariant_ref in producer.preserves_invariant_refs:
                evidence_ref = _evidence_ref(
                    request.snapshot.invariant_evidence,
                    invariant_ref,
                )
                if evidence_ref is None:
                    raise _OrderRejected(
                        SynthesisFailureCode.MISSING_PRESERVATION_EVIDENCE,
                        component_ref=producer.component_ref,
                        predicate_ref=predicate_ref,
                        detail_refs=(invariant_ref,),
                    )
                preservation_refs.append(evidence_ref)
            visit(producer.component_ref, parent_ref=component_ref, depth=depth + 1)
            edge_specs.append(
                (
                    producer.component_ref,
                    component_ref,
                    producer_order,
                    absence_ref,
                    tuple(sorted(preservation_refs)),
                )
            )
            if len(edge_specs) > request.global_resource_bounds.maximum_edges:
                raise _OrderRejected(
                    SynthesisFailureCode.EDGE_BOUND,
                    component_ref=component_ref,
                    predicate_ref=predicate_ref,
                )
        active.remove(component_ref)
        declared_execution_component_refs.append(component_ref)

    visit(order.root_component_ref, parent_ref=None, depth=1)
    node_ref_by_component = {
        component_ref: (
            "dynamic-node:"
            + stable_digest(
                {
                    "order_ref": order.order_ref,
                    "component_ref": component_ref,
                }
            )[:24]
        )
        for component_ref in component_by_ref
    }
    nodes = tuple(
        DynamicWorkflowNode(
            node_ref=node_ref_by_component[component_ref],
            component_ref=component_ref,
            served_goal_refs=(request.snapshot.root_goal_ref,),
            bound_input_refs=_component_input_refs(
                request.snapshot.bound_input_refs_by_component,
                component_ref,
            ),
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
            parent_node_ref=(
                node_ref_by_component[parent_by_component[component_ref]]
                if parent_by_component[component_ref] is not None
                else None
            ),
        )
        for component_ref, component in sorted(component_by_ref.items())
    )
    edges = tuple(
        DynamicWorkflowEdge(
            edge_ref=(
                "dynamic-edge:"
                + stable_digest(
                    {
                        "order_ref": order.order_ref,
                        "provider": provider_ref,
                        "consumer": consumer_ref,
                        "predicate": producer_order.required_predicate_ref,
                        "principle": producer_order.composition_principle_ref,
                    }
                )[:24]
            ),
            provider_node_ref=node_ref_by_component[provider_ref],
            consumer_node_ref=node_ref_by_component[consumer_ref],
            required_predicate_ref=producer_order.required_predicate_ref,
            provided_predicate_ref=producer_order.required_predicate_ref,
            current_absence_evidence_ref=absence_ref,
            composition_principle_ref=producer_order.composition_principle_ref,
            preservation_evidence_refs=preservation_refs,
            falsifier_ref=producer_order.falsifier_ref,
        )
        for (
            provider_ref,
            consumer_ref,
            producer_order,
            absence_ref,
            preservation_refs,
        ) in edge_specs
    )
    component_hashes = FrozenMap(
        {
            component_ref: component.transitive_source_hash
            for component_ref, component in sorted(component_by_ref.items())
        }
    )
    law_hashes = FrozenMap(
        {
            producer_order.composition_principle_ref: producer_order.composition_law_hash
            for _provider, _consumer, producer_order, _absence, _preserved in edge_specs
        }
    )
    derivation = CanonicalWorkflowDerivation(
        cold_component_transitive_hashes=component_hashes,
        composition_law_hashes=law_hashes,
        canonical_goal_bindings=request.snapshot.canonical_goal_bindings,
        declared_order_ref=order.order_ref,
        declared_node_order_refs=tuple(
            node_ref_by_component[component_ref]
            for component_ref in declared_execution_component_refs
        ),
    )
    return build_dynamic_workflow_ir(
        root_goal_ref=request.snapshot.root_goal_ref,
        source_state_revision=request.snapshot.source_state_revision,
        source_frame_ref=request.snapshot.source_frame_ref,
        cold_registry_hash=request.cold_registry_hash,
        drm_declaration_hash=request.drm_declaration_hash,
        synthesis_policy_ref=request.synthesis_policy_ref,
        nodes=nodes,
        dependency_edges=edges,
        branch_conditions=(),
        preserved_invariant_refs=request.preserved_invariant_refs,
        global_falsifier_refs=request.global_falsifier_refs,
        global_resource_bounds=request.global_resource_bounds,
        completion_contract=request.completion_contract,
        canonical_derivation=derivation,
    )


def synthesize_declared_workflow_candidates(
    request: DynamicWorkflowSynthesisRequest,
    components: ColdComponentRegistry,
) -> DynamicWorkflowSynthesisResult:
    """Materialize one canonical witness per declared order without selecting one."""

    candidates: list[DynamicWorkflowIR] = []
    failures: list[DynamicWorkflowSynthesisFailure] = []
    for order in request.declared_orders:
        try:
            candidate = _materialize_order(request, order, components)
        except _OrderRejected as error:
            failures.append(
                DynamicWorkflowSynthesisFailure(
                    order_ref=order.order_ref,
                    code=error.code,
                    component_ref=error.component_ref,
                    predicate_ref=error.predicate_ref,
                    detail_refs=tuple(sorted(set(error.detail_refs))),
                )
            )
            continue
        candidates.append(candidate)
    return DynamicWorkflowSynthesisResult(
        candidate_workflows=tuple(candidates),
        rejected_orders=tuple(failures),
        viable_candidate_count=len(candidates),
        source_state_revision=request.snapshot.source_state_revision,
        source_frame_ref=request.snapshot.source_frame_ref,
    )
