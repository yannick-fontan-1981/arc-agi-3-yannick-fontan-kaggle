"""Mechanical execution of source-declared lazy perception demands.

This module deliberately contains no perception choice.  DRM/SRC provide one
typed demand when an active reasoning branch needs evidence.  The executor
only checks immutable identity, delegates the pure capability call, and keeps
a bounded pointer cache so an already completed demand is not rebuilt.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import Field, model_validator

from agents.yf_arc3_v5.capabilities.contracts import (
    CapabilityExecution,
    RevisionedCapabilityRequest,
)
from agents.yf_arc3_v5.capabilities.registry import CapabilityRegistry
from agents.yf_arc3_v5.logos.types import (
    FrozenModel,
    NonNegativeRevision,
    Ref,
    schema_ordered_digest,
    require_unique,
)


class PerceptionDemand(FrozenModel):
    """One source-declared pure measurement request.

    ``demand_ref`` and ``capability_id`` are authored outside Python.  The
    executor never discovers or selects them.  ``invalidation_refs`` is only a
    mechanical dependency index for the descriptive delta.
    """

    demand_ref: Ref
    capability_id: Ref
    input_revision: NonNegativeRevision
    payload: FrozenModel
    invalidation_refs: tuple[Ref, ...] = Field(default=(), max_length=32)
    schema_version: Ref = "yf_arc3_v5.perception_demand.v1"

    @model_validator(mode="after")
    def validate_demand(self) -> "PerceptionDemand":
        require_unique(self.invalidation_refs, "perception demand invalidation refs")
        return self


class LazyPerceptionResult(FrozenModel):
    """Small envelope around a capability result; output remains immutable."""

    demand_ref: Ref
    execution: CapabilityExecution
    exact_input_reused: bool
    delta_intersects_invalidation: bool
    schema_version: Ref = "yf_arc3_v5.lazy_perception_result.v1"


class LazyPerceptionExecutor:
    """Execute only demands explicitly submitted by the active branch."""

    _MAX_DEMAND_POINTERS = 128

    def __init__(self, registry: CapabilityRegistry) -> None:
        self._registry = registry
        # Only small identity pointers live here; capability outputs stay in
        # the registry's bounded exact-input cache.
        self._demand_inputs: dict[str, tuple[str, str]] = {}

    def resolve(
        self,
        demand: PerceptionDemand,
        *,
        delta_refs: Iterable[str] = (),
    ) -> LazyPerceptionResult:
        delta = frozenset(delta_refs)
        intersects = bool(delta.intersection(demand.invalidation_refs))
        input_digest = schema_ordered_digest(demand.payload)
        prior = self._demand_inputs.get(demand.demand_ref)
        exact_input_reused = prior is not None and prior[0] == input_digest
        execution = self._registry.invoke(
            RevisionedCapabilityRequest(
                capability_id=demand.capability_id,
                input_revision=demand.input_revision,
                payload=demand.payload,
            )
        )
        self._demand_inputs[demand.demand_ref] = (
            input_digest,
            execution.cache_key,
        )
        while len(self._demand_inputs) > self._MAX_DEMAND_POINTERS:
            self._demand_inputs.pop(next(iter(self._demand_inputs)))
        return LazyPerceptionResult(
            demand_ref=demand.demand_ref,
            execution=execution,
            exact_input_reused=exact_input_reused,
            delta_intersects_invalidation=intersects,
        )

