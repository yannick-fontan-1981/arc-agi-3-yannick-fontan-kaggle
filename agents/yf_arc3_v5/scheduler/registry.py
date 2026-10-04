"""Typed pure-call and operator-request builder registries for the scheduler."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agents.yf_arc3_v5.logos.types import EffectClass, OperatorName
from agents.yf_arc3_v5.operators.contracts import AnyOperatorRequest
from agents.yf_arc3_v5.scheduler.contracts import (
    FunctionCallContext,
    OperatorBuildContext,
)
from agents.yf_arc3_v5.src.compiler import AuthorityAttestation
from agents.yf_arc3_v5.src.symbols import SymbolDefinition, SymbolKind


class SchedulerRegistrationError(ValueError):
    """A runtime helper does not match its compiled static authority."""


class SchedulerResolutionError(LookupError):
    """A compiled statement has no exact runtime helper."""


class PureCallBlocked(RuntimeError):
    """A source-controlled pure call produced an explicit blocked outcome."""


class PureCallDeferred(RuntimeError):
    """A source-controlled pure call preserved an unresolved outcome."""


PureCallImplementation = Callable[[FunctionCallContext], object]
OperatorRequestBuilder = Callable[[OperatorBuildContext], AnyOperatorRequest]


@dataclass(frozen=True)
class RegisteredPureCall:
    definition: SymbolDefinition
    implementation: PureCallImplementation


class SchedulerRuntimeRegistry:
    def __init__(self) -> None:
        self._calls: dict[str, RegisteredPureCall] = {}
        self._operator_builders: dict[OperatorName, OperatorRequestBuilder] = {}

    @property
    def call_definitions(self) -> tuple[SymbolDefinition, ...]:
        """Expose immutable authority metadata for read-only architecture audits."""

        return tuple(self._calls[call_id].definition for call_id in sorted(self._calls))

    @property
    def operator_names(self) -> tuple[OperatorName, ...]:
        """Expose registered primitive names without exposing their implementations."""

        return tuple(sorted(self._operator_builders, key=lambda item: item.value))

    def register_call(
        self,
        definition: SymbolDefinition,
        implementation: PureCallImplementation,
    ) -> None:
        if definition.kind not in {
            SymbolKind.CAPABILITY,
            SymbolKind.WORKFLOW,
            SymbolKind.WISDOM_WORKFLOW,
            SymbolKind.ACTION_RELEASE_WORKFLOW,
            SymbolKind.PREDICATE,
        }:
            raise SchedulerRegistrationError(
                f"scheduler call cannot execute {definition.kind.value}"
            )
        if definition.effect_class is not EffectClass.PURE:
            raise SchedulerRegistrationError("scheduler calls must be pure")
        existing = self._calls.get(definition.id)
        registered = RegisteredPureCall(definition, implementation)
        if existing is not None and existing != registered:
            raise SchedulerRegistrationError(
                f"scheduler call already registered: {definition.id}"
            )
        self._calls[definition.id] = registered

    def register_operator_builder(
        self,
        operator: OperatorName,
        builder: OperatorRequestBuilder,
    ) -> None:
        existing = self._operator_builders.get(operator)
        if existing is not None and existing != builder:
            raise SchedulerRegistrationError(
                f"operator builder already registered: {operator.value}"
            )
        self._operator_builders[operator] = builder

    def resolve_call(
        self,
        callee_ref: str,
        *,
        attestation: AuthorityAttestation,
    ) -> RegisteredPureCall:
        registered = self._calls.get(callee_ref)
        if registered is None:
            raise SchedulerResolutionError(f"unregistered scheduler call: {callee_ref}")
        definition = registered.definition
        mismatches: list[str] = []
        if definition.id != attestation.authority_id:
            mismatches.append("identity")
        if definition.kind is not attestation.kind:
            mismatches.append("kind")
        if definition.effect_class is not attestation.effect_class:
            mismatches.append("effect")
        if definition.source_unit != attestation.source_unit:
            mismatches.append("source_unit")
        if definition.source_hash != attestation.source_hash:
            mismatches.append("source_hash")
        if mismatches:
            raise SchedulerResolutionError(
                f"scheduler call attestation mismatch for {callee_ref}: {mismatches}"
            )
        return registered

    def build_operator(self, context: OperatorBuildContext) -> AnyOperatorRequest:
        builder = self._operator_builders.get(context.operator)
        if builder is None:
            raise SchedulerResolutionError(
                f"unregistered operator request builder: {context.operator.value}"
            )
        return builder(context)
