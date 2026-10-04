"""Typed SRC symbol, authority, and type environment contracts."""

from __future__ import annotations

from enum import Enum
from functools import cached_property

from pydantic import model_validator

from agents.yf_arc3_v5.logos.types import (
    EffectClass,
    FrozenModel,
    Ref,
    require_unique,
    stable_digest,
)
from agents.yf_arc3_v5.src.ast import TypeExpression


class TypeSpec(FrozenModel):
    name: Ref
    arguments: tuple["TypeSpec", ...] = ()
    optional: bool = False

    @classmethod
    def from_ast(cls, value: TypeExpression) -> "TypeSpec":
        return cls(
            name=value.name,
            arguments=tuple(cls.from_ast(argument) for argument in value.arguments),
            optional=value.optional,
        )

    def display(self) -> str:
        generic = ""
        if self.arguments:
            generic = (
                "<" + ",".join(argument.display() for argument in self.arguments) + ">"
            )
        optional = "?" if self.optional else ""
        return f"{self.name}{generic}{optional}"

    def required(self) -> "TypeSpec":
        return (
            self if not self.optional else self.model_copy(update={"optional": False})
        )


class SymbolKind(str, Enum):
    VALUE = "value"
    PREDICATE = "predicate"
    RULE = "rule"
    MODEL = "model"
    CRITERION = "criterion"
    POLICY = "policy"
    CAPABILITY = "capability"
    WORKFLOW = "workflow"
    LOGOS_CONTRACT = "logos_contract"
    DRM_DECLARATION = "drm_declaration"
    WISDOM_WORKFLOW = "wisdom_workflow"
    ACTION_RELEASE_WORKFLOW = "action_release_workflow"
    ACQUISITION_MODEL = "acquisition_model"
    ENVIRONMENT_ADAPTER = "environment_adapter"


class ParameterSpec(FrozenModel):
    name: Ref
    type_spec: TypeSpec
    required: bool = True


class SymbolDefinition(FrozenModel):
    id: Ref
    kind: SymbolKind
    parameters: tuple[ParameterSpec, ...] = ()
    output_type: TypeSpec
    effect_class: EffectClass = EffectClass.PURE
    source_unit: Ref
    source_hash: Ref
    declaration_digest: Ref | None = None
    schema_version: Ref = "yf_arc3_v5.src_symbol.v0_2"

    @model_validator(mode="after")
    def validate_definition(self) -> "SymbolDefinition":
        require_unique(
            tuple(item.name for item in self.parameters), "symbol parameters"
        )
        if len(self.source_hash) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_hash
        ):
            raise ValueError("symbol source_hash must be a lowercase SHA-256 digest")
        if self.declaration_digest is not None and (
            len(self.declaration_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.declaration_digest
            )
        ):
            raise ValueError(
                "symbol declaration_digest must be a lowercase SHA-256 digest"
            )
        if self.kind is SymbolKind.ENVIRONMENT_ADAPTER:
            if self.effect_class is not EffectClass.ENVIRONMENT_ADAPTER:
                raise ValueError("environment adapter must declare environment effect")
        elif self.kind is SymbolKind.ACQUISITION_MODEL:
            if self.effect_class is not EffectClass.OBSERVATION_ADAPTER:
                raise ValueError("acquisition model must declare observation effect")
        elif (
            self.kind
            in {
                SymbolKind.RULE,
                SymbolKind.MODEL,
                SymbolKind.CRITERION,
                SymbolKind.PREDICATE,
                SymbolKind.LOGOS_CONTRACT,
                SymbolKind.DRM_DECLARATION,
            }
            and self.effect_class is not EffectClass.PURE
        ):
            raise ValueError(f"{self.kind.value} authority must be pure")
        return self


class CompilerEnvironment(FrozenModel):
    types: tuple[Ref, ...]
    symbols: tuple[SymbolDefinition, ...] = ()
    schema_version: Ref = "yf_arc3_v5.src_environment.v0_1"

    @model_validator(mode="after")
    def validate_environment(self) -> "CompilerEnvironment":
        require_unique(self.types, "environment types")
        require_unique(
            tuple(symbol.id for symbol in self.symbols), "environment symbols"
        )
        return self

    @cached_property
    def symbol_index(self) -> dict[str, SymbolDefinition]:
        return {symbol.id: symbol for symbol in self.symbols}


def type_spec(name: str, *arguments: TypeSpec, optional: bool = False) -> TypeSpec:
    return TypeSpec(name=name, arguments=arguments, optional=optional)


def _fixture_declaration_digest(authority_id: str) -> str:
    return stable_digest({"authority_id": authority_id, "contract_version": 1})


def generic_fixture_environment() -> CompilerEnvironment:
    """Generic compiler fixture, never the source authority for production calls.

    Production starts from these common types and abstract adapter contracts but
    replaces every cognitive control symbol with a file-backed definition in
    :func:`production_environment`.
    """
    type_names = (
        "Any",
        "Boolean",
        "Integer",
        "String",
        "Ref",
        "List",
        "Set",
        "Map",
        "Term",
        "Claim",
        "Entity",
        "Event",
        "Schema",
        "Procedure",
        "Goal",
        "Value",
        "Frame",
        "WorldInput",
        "WorldContext",
        "ObservationResult",
        "AcquisitionModel",
        "Criterion",
        "Rule",
        "Model",
        "Policy",
        "Predicate",
        "Capability",
        "EnvironmentAdapter",
        "Assessment",
        "StateChangeProposal",
        "Action",
        "ActionIntent",
        "ActionGroundingContract",
        "ActionReleasePermit",
        "ActionEvent",
        "ViabilityAssessment",
        "CommitmentDecision",
        "PropagationReport",
        "TerminalResult",
        "WorkflowResult",
        "RoleResolution",
        "LogosContract",
        "DrmDeclaration",
    )
    symbols = (
        SymbolDefinition(
            id="acquisition.frame",
            kind=SymbolKind.ACQUISITION_MODEL,
            parameters=(
                ParameterSpec(name="world_input", type_spec=type_spec("WorldInput")),
            ),
            output_type=type_spec("ObservationResult"),
            effect_class=EffectClass.OBSERVATION_ADAPTER,
            source_unit="logos.acquisition.frame",
            source_hash=_fixture_declaration_digest("acquisition.frame"),
            declaration_digest=_fixture_declaration_digest("acquisition.frame"),
        ),
        SymbolDefinition(
            id="criterion.symbolic_equality",
            kind=SymbolKind.CRITERION,
            parameters=(
                ParameterSpec(name="left", type_spec=type_spec("Any")),
                ParameterSpec(name="right", type_spec=type_spec("Any")),
            ),
            output_type=type_spec("Assessment"),
            source_unit="logos.criterion.symbolic_equality",
            source_hash=_fixture_declaration_digest("criterion.symbolic_equality"),
            declaration_digest=_fixture_declaration_digest(
                "criterion.symbolic_equality"
            ),
        ),
        SymbolDefinition(
            id="predicate.is_resolved",
            kind=SymbolKind.PREDICATE,
            parameters=(ParameterSpec(name="value", type_spec=type_spec("Any")),),
            output_type=type_spec("Boolean"),
            source_unit="logos.predicate.is_resolved",
            source_hash=_fixture_declaration_digest("predicate.is_resolved"),
            declaration_digest=_fixture_declaration_digest("predicate.is_resolved"),
        ),
        SymbolDefinition(
            id="workflow.operational_wisdom",
            kind=SymbolKind.WISDOM_WORKFLOW,
            parameters=(
                ParameterSpec(
                    name="change", type_spec=type_spec("StateChangeProposal")
                ),
                ParameterSpec(
                    name="principles",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
            ),
            output_type=type_spec("CommitmentDecision"),
            source_unit="src.core.operational_wisdom",
            source_hash=_fixture_declaration_digest("workflow.operational_wisdom"),
            declaration_digest=_fixture_declaration_digest(
                "workflow.operational_wisdom"
            ),
        ),
        SymbolDefinition(
            id="workflow.release_action",
            kind=SymbolKind.ACTION_RELEASE_WORKFLOW,
            parameters=(
                ParameterSpec(name="intent", type_spec=type_spec("ActionIntent")),
                ParameterSpec(name="wisdom", type_spec=type_spec("CommitmentDecision")),
                ParameterSpec(name="context", type_spec=type_spec("WorldContext")),
            ),
            output_type=type_spec("ActionReleasePermit"),
            source_unit="src.core.release_action",
            source_hash=_fixture_declaration_digest("workflow.release_action"),
            declaration_digest=_fixture_declaration_digest(
                "workflow.release_action"
            ),
        ),
        SymbolDefinition(
            id="control.validate_action_grounding",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="intent", type_spec=type_spec("ActionIntent")),
            ),
            output_type=type_spec("ActionGroundingContract"),
            source_unit="src.core.validate_action_grounding",
            source_hash=_fixture_declaration_digest(
                "control.validate_action_grounding"
            ),
            declaration_digest=_fixture_declaration_digest(
                "control.validate_action_grounding"
            ),
        ),
        SymbolDefinition(
            id="environment.primary",
            kind=SymbolKind.ENVIRONMENT_ADAPTER,
            parameters=(
                ParameterSpec(name="intent", type_spec=type_spec("ActionIntent")),
                ParameterSpec(name="context", type_spec=type_spec("WorldContext")),
            ),
            output_type=type_spec("ActionEvent"),
            effect_class=EffectClass.ENVIRONMENT_ADAPTER,
            source_unit="runtime.environment.primary",
            source_hash=_fixture_declaration_digest("environment.primary"),
            declaration_digest=_fixture_declaration_digest("environment.primary"),
        ),
    )
    return CompilerEnvironment(types=type_names, symbols=symbols)


def core_environment() -> CompilerEnvironment:
    """Backward-compatible alias for the generic non-production fixture."""

    return generic_fixture_environment()


TypeSpec.model_rebuild()
