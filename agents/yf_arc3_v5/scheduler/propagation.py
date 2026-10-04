"""Explicit bounded dependency propagation over one revision-keyed index."""

from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from agents.yf_arc3_v5.logos.operations import PropagationReport
from agents.yf_arc3_v5.logos.types import (
    FrozenModel,
    Ref,
    require_unique,
)
from agents.yf_arc3_v5.scheduler.contracts import require_sha256
from agents.yf_arc3_v5.state import DependencyIndex, EventStore, RecordArtifact
from agents.yf_arc3_v5.state.transactions import TransactionRequest


class PropagationDisposition(str, Enum):
    PRESERVE = "preserve"
    INVALIDATE = "invalidate"
    RECOMPUTE = "recompute"
    BLOCK = "block"


class DependencyEvaluation(FrozenModel):
    dependency_claim_id: Ref
    disposition: PropagationDisposition
    reason_refs: tuple[Ref, ...] = Field(min_length=1)
    blocking_contradiction_ref: Ref | None = None
    schema_version: Ref = "yf_arc3_v5.dependency_evaluation.v1"

    @model_validator(mode="after")
    def validate_block(self) -> "DependencyEvaluation":
        if (self.disposition is PropagationDisposition.BLOCK) != bool(
            self.blocking_contradiction_ref
        ):
            raise ValueError("only a blocking evaluation carries a contradiction")
        return self


class DependencyPropagationRequest(FrozenModel):
    request_id: Ref
    run_id: Ref
    timeline_id: Ref
    frame_id: Ref | None = None
    source_unit: Ref
    source_hash: Ref
    workflow_definition_id: Ref
    workflow_instance_id: Ref
    workflow_step_id: Ref
    source_change_id: Ref
    changed_refs: tuple[Ref, ...] = Field(min_length=1)
    dependency_index: DependencyIndex
    evaluations: tuple[DependencyEvaluation, ...] = ()
    maximum_depth: int = Field(strict=True, ge=1)
    schema_version: Ref = "yf_arc3_v5.dependency_propagation_request.v1"

    @model_validator(mode="after")
    def validate_request(self) -> "DependencyPropagationRequest":
        require_sha256(self.source_hash, "dependency propagation source hash")
        require_unique(self.changed_refs, "propagation changed refs")
        require_unique(
            tuple(item.dependency_claim_id for item in self.evaluations),
            "dependency evaluations",
        )
        return self


class DependencyPropagationRuntime:
    """Applies explicit edge decisions; it never ranks or invents a policy."""

    def __init__(self, event_store: EventStore) -> None:
        self._store = event_store

    def execute(self, request: DependencyPropagationRequest) -> PropagationReport:
        if request.run_id != self._store.run_id:
            raise ValueError("propagation run identity mismatch")
        if (
            request.dependency_index.state_revision != self._store.snapshot.revision
            or request.dependency_index.state_hash != self._store.snapshot.state_hash
        ):
            raise ValueError("propagation dependency index is stale")
        evaluations = {item.dependency_claim_id: item for item in request.evaluations}
        frontier = tuple(sorted(request.changed_refs))
        seen_sources: set[Ref] = set()
        affected: set[Ref] = set()
        invalidated: set[Ref] = set()
        recomputed: set[Ref] = set()
        contradiction: Ref | None = None

        for _depth in range(request.maximum_depth):
            next_frontier: set[Ref] = set()
            for source_ref in frontier:
                if source_ref in seen_sources:
                    continue
                seen_sources.add(source_ref)
                edges = tuple(
                    edge
                    for edge in request.dependency_index.edges
                    if edge.source_ref == source_ref
                )
                for edge in edges:
                    affected.add(edge.dependent_ref)
                    evaluation = evaluations.get(edge.claim_id)
                    if evaluation is None:
                        contradiction = f"contradiction:missing_dependency_evaluation:{edge.claim_id}"
                        break
                    if evaluation.disposition is PropagationDisposition.BLOCK:
                        contradiction = evaluation.blocking_contradiction_ref
                        break
                    if evaluation.disposition is PropagationDisposition.INVALIDATE:
                        invalidated.add(edge.dependent_ref)
                        next_frontier.add(edge.dependent_ref)
                    elif evaluation.disposition is PropagationDisposition.RECOMPUTE:
                        recomputed.add(edge.dependent_ref)
                        next_frontier.add(edge.dependent_ref)
                if contradiction:
                    break
            if contradiction:
                break
            frontier = tuple(sorted(next_frontier - seen_sources))
            frontier = tuple(
                source_ref
                for source_ref in frontier
                if any(
                    edge.source_ref == source_ref
                    for edge in request.dependency_index.edges
                )
            )
            if not frontier:
                break
        else:
            if frontier:
                contradiction = "contradiction:dependency_propagation_depth_limit"

        report = PropagationReport(
            id=f"{request.request_id}:report",
            source_unit=request.source_unit,
            source_hash=request.source_hash,
            state_revision=self._store.snapshot.revision,
            workflow_definition_id=request.workflow_definition_id,
            workflow_instance_id=request.workflow_instance_id,
            workflow_step_id=request.workflow_step_id,
            timeline_id=request.timeline_id,
            frame_id=request.frame_id,
            provenance=tuple(sorted({request.source_change_id, *request.changed_refs})),
            source_change_id=request.source_change_id,
            affected_refs=tuple(sorted(affected)),
            invalidated_refs=tuple(sorted(invalidated)),
            recomputed_refs=tuple(sorted(recomputed)),
            fixpoint_reached=contradiction is None,
            blocking_contradiction_ref=contradiction,
        )
        self._store.commit(
            TransactionRequest(
                transaction_id=f"{request.request_id}:propagation",
                run_id=request.run_id,
                timeline_id=request.timeline_id,
                frame_id=request.frame_id,
                source_hash=request.source_hash,
                expected_revision=self._store.snapshot.revision,
                mutations=(RecordArtifact(artifact=report),),
            )
        )
        return report
