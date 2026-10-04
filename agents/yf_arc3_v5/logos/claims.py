"""Native V5 CLAIM contract."""

from __future__ import annotations

from pydantic import Field, model_validator

from agents.yf_arc3_v5.logos.types import (
    Cardinality,
    Disposition,
    EpistemicStatus,
    FrozenMap,
    FrozenModel,
    NonNegativeRevision,
    Polarity,
    Ref,
    RelationFamily,
    require_unique,
)


class Claim(FrozenModel):
    id: Ref
    predicate: Ref
    family: RelationFamily
    arguments: tuple[Ref, ...] = Field(min_length=1)
    polarity: Polarity = Polarity.POSITIVE
    epistemic_status: EpistemicStatus = EpistemicStatus.PROPOSED
    disposition: Disposition = Disposition.ACTIVE
    cardinality: Cardinality = Cardinality.SINGULAR
    scope: FrozenMap = Field(default_factory=FrozenMap)
    conditions: tuple[Ref, ...] = ()
    grounds: tuple[Ref, ...] = ()
    active_contradictions: tuple[Ref, ...] = ()
    defeated_by: tuple[Ref, ...] = ()
    proof_rule: Ref | None = None
    source_operator_results: tuple[Ref, ...] = ()
    valid_through_state_revision: NonNegativeRevision | None = None
    fresh_through_prediction_revision: NonNegativeRevision | None = None
    supersedes: Ref | None = None
    attributes: FrozenMap = Field(default_factory=FrozenMap)
    schema_version: Ref = "yf_arc3_v5.claim.v1"

    @model_validator(mode="after")
    def validate_epistemic_contract(self) -> "Claim":
        for field_name in (
            "arguments",
            "conditions",
            "grounds",
            "active_contradictions",
            "defeated_by",
            "source_operator_results",
        ):
            require_unique(getattr(self, field_name), field_name)
        if (
            self.epistemic_status is EpistemicStatus.DEFEATED
            and not self.defeated_by
            and not self.active_contradictions
        ):
            raise ValueError(
                "a defeated claim requires contradiction or defeating evidence"
            )
        if self.supersedes == self.id:
            raise ValueError("a claim cannot supersede itself")
        return self
