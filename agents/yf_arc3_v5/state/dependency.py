"""Revision-keyed dependency view derived exclusively from canonical CLAIMs."""

from __future__ import annotations

from enum import Enum

from pydantic import model_validator

from agents.yf_arc3_v5.logos.types import (
    Disposition,
    EpistemicStatus,
    FrozenModel,
    NonNegativeRevision,
    Polarity,
    Ref,
    RelationFamily,
)
from agents.yf_arc3_v5.state.snapshot import CognitiveSnapshot

DEPENDENCY_PREDICATE = "depends_on"


class DependentKind(str, Enum):
    CLAIM = "claim"
    WORKFLOW = "workflow"
    PLAN = "plan"
    EXPECTATION = "expectation"
    COMMITMENT = "commitment"
    DERIVED_VIEW = "derived_view"


class DependencyEdge(FrozenModel):
    """A dependency edge projected from one exact current claim."""

    claim_id: Ref
    source_ref: Ref
    dependent_ref: Ref
    dependent_kind: DependentKind
    schema_version: Ref = "yf_arc3_v5.dependency_edge.v1"


class DependencyIndex(FrozenModel):
    """Read-only dependency index tied to an exact semantic snapshot."""

    state_revision: NonNegativeRevision
    state_hash: Ref
    edges: tuple[DependencyEdge, ...] = ()
    schema_version: Ref = "yf_arc3_v5.dependency_index.v1"

    @classmethod
    def from_snapshot(cls, snapshot: CognitiveSnapshot) -> "DependencyIndex":
        edges: list[DependencyEdge] = []
        for claim in snapshot.current_claims:
            if (
                claim.predicate != DEPENDENCY_PREDICATE
                or claim.family is not RelationFamily.CONTROL
                or claim.polarity is not Polarity.POSITIVE
                or claim.disposition is not Disposition.ACTIVE
                or claim.epistemic_status is EpistemicStatus.DEFEATED
            ):
                continue
            if len(claim.arguments) != 2:
                raise ValueError(
                    f"dependency claim {claim.id} requires [source, dependent] arguments"
                )
            raw_kind = claim.attributes.get("dependent_kind")
            if raw_kind is None:
                raise ValueError(f"dependency claim {claim.id} requires dependent_kind")
            edges.append(
                DependencyEdge(
                    claim_id=claim.id,
                    source_ref=claim.arguments[0],
                    dependent_ref=claim.arguments[1],
                    dependent_kind=DependentKind(raw_kind),
                )
            )
        return cls(
            state_revision=snapshot.revision,
            state_hash=snapshot.state_hash,
            edges=tuple(
                sorted(
                    edges,
                    key=lambda edge: (
                        edge.source_ref,
                        edge.dependent_kind.value,
                        edge.dependent_ref,
                        edge.claim_id,
                    ),
                )
            ),
        )

    @model_validator(mode="after")
    def validate_unique_claim_sources(self) -> "DependencyIndex":
        claim_ids = tuple(edge.claim_id for edge in self.edges)
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("each dependency claim may contribute only one edge")
        return self

    def dependents_of(
        self,
        source_ref: Ref,
        *,
        kind: DependentKind | None = None,
    ) -> tuple[Ref, ...]:
        return tuple(
            edge.dependent_ref
            for edge in self.edges
            if edge.source_ref == source_ref
            and (kind is None or edge.dependent_kind is kind)
        )
