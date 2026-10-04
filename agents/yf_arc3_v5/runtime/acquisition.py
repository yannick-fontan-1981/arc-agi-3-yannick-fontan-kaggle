"""Semantically neutral world-input repository and acquisition authority."""

from __future__ import annotations

import hashlib
from pathlib import Path

from agents.yf_arc3_v5.logos.types import (
    EffectClass,
    FrozenMap,
    OperatorName,
)
from agents.yf_arc3_v5.operators import (
    AcquisitionInput,
    AcquisitionResult,
    AuthorityDefinition,
    OperatorAuthorityRegistry,
    RuntimeAuthorityKind,
)
from agents.yf_arc3_v5.runtime.contracts import ObservedWorldInput
from agents.yf_arc3_v5.src.symbols import SymbolDefinition, SymbolKind, type_spec

ACQUISITION_ID = "acquisition.frame"


def acquisition_symbol_definition() -> SymbolDefinition:
    path = Path(__file__)
    return SymbolDefinition(
        id=ACQUISITION_ID,
        kind=SymbolKind.ACQUISITION_MODEL,
        output_type=type_spec("ObservationResult"),
        effect_class=EffectClass.OBSERVATION_ADAPTER,
        source_unit="agents/yf_arc3_v5/runtime/acquisition.py",
        source_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


class WorldInputRepository:
    def __init__(self) -> None:
        self._inputs: dict[str, ObservedWorldInput] = {}
        self._inputs_by_frame_id: dict[str, ObservedWorldInput] = {}

    def put(self, value: ObservedWorldInput) -> None:
        existing = self._inputs.get(value.raw_input_ref)
        if existing is not None and existing != value:
            raise ValueError(
                f"raw world input identity already has different content: "
                f"{value.raw_input_ref}"
            )
        self._inputs[value.raw_input_ref] = value
        self._inputs_by_frame_id[value.frame_id] = value

    def get(self, raw_input_ref: str) -> ObservedWorldInput:
        try:
            return self._inputs[raw_input_ref]
        except KeyError as error:
            raise KeyError(f"unknown raw world input: {raw_input_ref}") from error

    def get_by_frame_id(self, frame_id: str) -> ObservedWorldInput | None:
        return self._inputs_by_frame_id.get(frame_id)

    @property
    def values(self) -> tuple[ObservedWorldInput, ...]:
        return tuple(self._inputs[key] for key in sorted(self._inputs))

    def bounded_values(self, maximum: int) -> tuple[ObservedWorldInput, ...]:
        """Bound allocation before exposing immutable inputs to a measurement."""
        if type(maximum) is not int or maximum < 1 or len(self._inputs) > maximum:
            raise ValueError("world input history bound exceeded")
        return tuple(self._inputs.values())


def register_frame_acquisition(
    authorities: OperatorAuthorityRegistry,
    repository: WorldInputRepository,
) -> None:
    static_symbol = acquisition_symbol_definition()
    authorities.register(
        AuthorityDefinition(
            id=static_symbol.id,
            kind=RuntimeAuthorityKind.ACQUISITION_MODEL,
            supported_operators=(OperatorName.OBSERVE,),
            input_contract="yf_arc3_v5.authority_input.acquisition.v1",
            output_contract="yf_arc3_v5.authority_output.acquisition.v1",
            effect_class=EffectClass.OBSERVATION_ADAPTER,
            source_unit=static_symbol.source_unit,
            source_hash=static_symbol.source_hash,
        ),
        lambda value: _acquire(
            repository,
            AcquisitionInput.model_validate(value),
        ),
        static_symbol=static_symbol,
    )


def _acquire(
    repository: WorldInputRepository,
    value: AcquisitionInput,
) -> AcquisitionResult:
    observed = repository.get(value.raw_input_ref)
    if value.frame_id != observed.frame_id:
        raise ValueError("acquisition frame identity does not match raw input")
    return AcquisitionResult(
        observation_ref=value.requested_observation_ref,
        measurements=FrozenMap(
            {
                "raw_input_ref": observed.raw_input_ref,
                "frame_id": observed.frame_id,
                "height": len(observed.frame),
                "width": len(observed.frame[0]),
                "available_action_refs": observed.available_action_refs,
                "score": observed.score,
                "state": observed.state,
            }
        ),
        reason_refs=("reason:transport_observation_is_descriptive",),
    )
