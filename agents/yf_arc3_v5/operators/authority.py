"""Source-hashed runtime authority registry for Logos operator execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

from pydantic import Field, model_validator

from agents.yf_arc3_v5.logos.types import (
    EffectClass,
    FrozenModel,
    OperatorName,
    Ref,
    require_unique,
)
from agents.yf_arc3_v5.src.symbols import SymbolDefinition, SymbolKind


class RuntimeAuthorityKind(str, Enum):
    ACQUISITION_MODEL = "acquisition_model"
    MODEL = "model"
    PREDICATE = "predicate"
    RULE = "rule"
    CRITERION = "criterion"
    UPDATE_POLICY = "update_policy"
    SELECTION_POLICY = "selection_policy"


_ALLOWED_OPERATORS: dict[RuntimeAuthorityKind, frozenset[OperatorName]] = {
    RuntimeAuthorityKind.ACQUISITION_MODEL: frozenset({OperatorName.OBSERVE}),
    RuntimeAuthorityKind.MODEL: frozenset(
        {OperatorName.DISTINGUISH, OperatorName.DERIVE}
    ),
    RuntimeAuthorityKind.PREDICATE: frozenset({OperatorName.RELATE}),
    RuntimeAuthorityKind.RULE: frozenset({OperatorName.DERIVE}),
    RuntimeAuthorityKind.CRITERION: frozenset(
        {OperatorName.DISTINGUISH, OperatorName.COMPARE}
    ),
    RuntimeAuthorityKind.UPDATE_POLICY: frozenset({OperatorName.UPDATE}),
    RuntimeAuthorityKind.SELECTION_POLICY: frozenset({OperatorName.SELECT}),
}

_REQUIRED_EFFECT: dict[RuntimeAuthorityKind, EffectClass] = {
    RuntimeAuthorityKind.ACQUISITION_MODEL: EffectClass.OBSERVATION_ADAPTER,
    RuntimeAuthorityKind.MODEL: EffectClass.PURE,
    RuntimeAuthorityKind.PREDICATE: EffectClass.PURE,
    RuntimeAuthorityKind.RULE: EffectClass.PURE,
    RuntimeAuthorityKind.CRITERION: EffectClass.PURE,
    RuntimeAuthorityKind.UPDATE_POLICY: EffectClass.COGNITIVE_UPDATE_ADAPTER,
    RuntimeAuthorityKind.SELECTION_POLICY: EffectClass.PURE,
}

_STATIC_KIND: dict[RuntimeAuthorityKind, SymbolKind] = {
    RuntimeAuthorityKind.ACQUISITION_MODEL: SymbolKind.ACQUISITION_MODEL,
    RuntimeAuthorityKind.MODEL: SymbolKind.MODEL,
    RuntimeAuthorityKind.PREDICATE: SymbolKind.PREDICATE,
    RuntimeAuthorityKind.RULE: SymbolKind.RULE,
    RuntimeAuthorityKind.CRITERION: SymbolKind.CRITERION,
    RuntimeAuthorityKind.UPDATE_POLICY: SymbolKind.POLICY,
    RuntimeAuthorityKind.SELECTION_POLICY: SymbolKind.POLICY,
}


class AuthorityDefinition(FrozenModel):
    id: Ref
    kind: RuntimeAuthorityKind
    supported_operators: tuple[OperatorName, ...] = Field(min_length=1)
    input_contract: Ref
    output_contract: Ref
    effect_class: EffectClass
    source_unit: Ref
    source_hash: Ref
    schema_version: Ref = "yf_arc3_v5.runtime_authority.v1"

    @model_validator(mode="after")
    def validate_authority(self) -> "AuthorityDefinition":
        require_unique(self.supported_operators, "authority supported operators")
        unsupported = set(self.supported_operators) - _ALLOWED_OPERATORS[self.kind]
        if unsupported:
            raise ValueError(
                f"{self.kind.value} cannot authorize operators "
                f"{sorted(item.value for item in unsupported)}"
            )
        if self.effect_class is not _REQUIRED_EFFECT[self.kind]:
            raise ValueError(
                f"{self.kind.value} requires {_REQUIRED_EFFECT[self.kind].value} effect"
            )
        if len(self.source_hash) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_hash
        ):
            raise ValueError(
                "runtime authority source_hash must be a lowercase SHA-256 digest"
            )
        return self


class AuthorityRegistrationError(ValueError):
    """A runtime authority cannot be admitted under its declared identity."""


class AuthorityResolutionError(LookupError):
    """No exact runtime authority satisfies an operator contract."""


AuthorityImplementation = Callable[[FrozenModel], FrozenModel]


@dataclass(frozen=True)
class RegisteredAuthority:
    definition: AuthorityDefinition
    implementation: AuthorityImplementation


OutputT = TypeVar("OutputT", bound=FrozenModel)


class OperatorAuthorityRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, RegisteredAuthority] = {}

    @property
    def definitions(self) -> tuple[AuthorityDefinition, ...]:
        return tuple(
            self._entries[authority_id].definition
            for authority_id in sorted(self._entries)
        )

    def register(
        self,
        definition: AuthorityDefinition,
        implementation: AuthorityImplementation,
        *,
        static_symbol: SymbolDefinition,
    ) -> None:
        expected_static_kind = _STATIC_KIND[definition.kind]
        mismatches: list[str] = []
        if static_symbol.id != definition.id:
            mismatches.append("identity")
        if static_symbol.kind is not expected_static_kind:
            mismatches.append("kind")
        if static_symbol.effect_class is not definition.effect_class:
            mismatches.append("effect")
        if static_symbol.source_unit != definition.source_unit:
            mismatches.append("source_unit")
        if static_symbol.source_hash != definition.source_hash:
            mismatches.append("source_hash")
        if mismatches:
            raise AuthorityRegistrationError(
                f"runtime/static authority mismatch for {definition.id}: {mismatches}"
            )
        existing = self._entries.get(definition.id)
        entry = RegisteredAuthority(
            definition=definition, implementation=implementation
        )
        if existing is not None:
            if existing == entry:
                return
            raise AuthorityRegistrationError(
                f"runtime authority identity already registered: {definition.id}"
            )
        self._entries[definition.id] = entry

    def resolve(
        self,
        authority_id: Ref,
        *,
        operator: OperatorName,
        allowed_kinds: frozenset[RuntimeAuthorityKind],
        required_effect: EffectClass,
        input_contract: Ref,
        output_contract: Ref,
    ) -> RegisteredAuthority:
        entry = self._entries.get(authority_id)
        if entry is None:
            raise AuthorityResolutionError(
                f"unregistered runtime authority: {authority_id}"
            )
        definition = entry.definition
        if definition.kind not in allowed_kinds:
            raise AuthorityResolutionError(
                f"authority {authority_id} has kind {definition.kind.value}"
            )
        if operator not in definition.supported_operators:
            raise AuthorityResolutionError(
                f"authority {authority_id} does not authorize {operator.value}"
            )
        if definition.effect_class is not required_effect:
            raise AuthorityResolutionError(
                f"authority {authority_id} has forbidden {definition.effect_class.value} effect"
            )
        if (
            definition.input_contract != input_contract
            or definition.output_contract != output_contract
        ):
            raise AuthorityResolutionError(
                f"authority {authority_id} contract mismatch"
            )
        return entry

    def invoke(
        self,
        entry: RegisteredAuthority,
        value: FrozenModel,
        *,
        expected_type: type[OutputT],
    ) -> OutputT:
        if getattr(value, "schema_version", None) != entry.definition.input_contract:
            raise AuthorityResolutionError(
                f"authority {entry.definition.id} received the wrong input contract"
            )
        output = entry.implementation(value)
        if not isinstance(output, expected_type):
            raise AuthorityResolutionError(
                f"authority {entry.definition.id} returned {type(output).__name__}, "
                f"expected {expected_type.__name__}"
            )
        if getattr(output, "schema_version", None) != entry.definition.output_contract:
            raise AuthorityResolutionError(
                f"authority {entry.definition.id} returned the wrong output contract"
            )
        return output
