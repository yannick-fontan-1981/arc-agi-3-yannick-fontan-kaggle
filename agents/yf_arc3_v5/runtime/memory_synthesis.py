"""Bounded mechanical projection for the hot cognitive memory delta."""

from __future__ import annotations

from pydantic import Field

from agents.yf_arc3_v5.logos.types import (
    FrozenMap,
    FrozenModel,
    NonNegativeRevision,
    Ref,
    canonical_json,
    schema_ordered_digest,
    require_unique,
)


class CompactMemorySynthesisRequest(FrozenModel):
    """Fields retained by a declared source-side memory contract."""

    source_revision: NonNegativeRevision
    source_facts: FrozenMap
    retained_fact_refs: tuple[Ref, ...] = Field(default=(), max_length=128)
    descriptive_delta: FrozenMap = Field(default_factory=FrozenMap)
    maximum_bytes: int = Field(default=10240, strict=True, ge=1, le=10240)
    schema_version: Ref = "yf_arc3_v5.compact_memory_synthesis_request.v1"

    @classmethod
    def from_declared_fields(
        cls,
        *,
        source_revision: int,
        source_facts: FrozenMap,
        retained_fact_refs: tuple[str, ...],
        descriptive_delta: FrozenMap,
    ) -> "CompactMemorySynthesisRequest":
        return cls(
            source_revision=source_revision,
            source_facts=source_facts,
            retained_fact_refs=retained_fact_refs,
            descriptive_delta=descriptive_delta,
        )

    def model_post_init(self, __context: object) -> None:
        require_unique(self.retained_fact_refs, "retained memory fact refs")


class CompactMemorySynthesisResult(FrozenModel):
    memory_delta: FrozenMap
    source_revision: NonNegativeRevision
    byte_size: int = Field(strict=True, ge=0, le=10240)
    source_digest: Ref
    schema_version: Ref = "yf_arc3_v5.compact_memory_synthesis_result.v1"


def synthesize_compact_memory(
    request: CompactMemorySynthesisRequest,
) -> CompactMemorySynthesisResult:
    """Project only source-declared fields and the descriptive delta.

    The function never expands ``source_facts`` into the result.  A too-large
    declared delta fails closed because silently dropping meaning-bearing facts
    would be a semantic decision belonging to DRM/SRC.
    """

    retained = FrozenMap(
        {
            ref: request.source_facts[ref]
            for ref in request.retained_fact_refs
            if ref in request.source_facts
        }
    )
    memory_delta = FrozenMap(
        {
            "retained_facts": retained,
            "descriptive_delta": request.descriptive_delta,
        }
    )
    byte_size = len(canonical_json(memory_delta).encode("utf-8"))
    if byte_size > request.maximum_bytes:
        raise ValueError(
            "declared compact memory delta exceeds the 10 KiB hard budget"
        )
    return CompactMemorySynthesisResult(
        memory_delta=memory_delta,
        source_revision=request.source_revision,
        byte_size=byte_size,
        source_digest=schema_ordered_digest(request),
    )

