"""Native V5 TERM contract."""

from __future__ import annotations

from pydantic import Field, model_validator

from agents.yf_arc3_v5.logos.types import (
    FrozenMap,
    FrozenModel,
    NonNegativeRevision,
    Ref,
    TermKind,
    require_unique,
)


class Term(FrozenModel):
    id: Ref
    kind: TermKind
    label: Ref
    attributes: FrozenMap = Field(default_factory=FrozenMap)
    provenance: tuple[Ref, ...] = ()
    scope: FrozenMap = Field(default_factory=FrozenMap)
    created_at_state_revision: NonNegativeRevision = 0
    last_changed_state_revision: NonNegativeRevision = 0
    schema_version: Ref = "yf_arc3_v5.term.v1"

    @model_validator(mode="after")
    def validate_revision_identity(self) -> "Term":
        if self.last_changed_state_revision < self.created_at_state_revision:
            raise ValueError("last_changed_state_revision precedes creation")
        require_unique(self.provenance, "provenance")
        return self
