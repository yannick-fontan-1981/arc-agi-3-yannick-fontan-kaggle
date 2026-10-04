"""Static type, effect, authority, and workflow-graph validation for SRC."""

from __future__ import annotations

from dataclasses import dataclass

from agents.yf_arc3_v5.logos.types import EffectClass, stable_digest
from agents.yf_arc3_v5.src.ast import (
    MACRO_STATEMENT_KINDS,
    OPERATOR_STATEMENT_KINDS,
    TERMINAL_STATEMENT_KINDS,
    Expression,
    ExpressionKind,
    ModuleAst,
    Statement,
    StatementKind,
    TypeExpression,
    WorkflowDecl,
)
from agents.yf_arc3_v5.src.errors import SrcValidationError
from agents.yf_arc3_v5.src.symbols import (
    CompilerEnvironment,
    SymbolDefinition,
    SymbolKind,
    TypeSpec,
    type_spec,
)

_TERMINAL_VALUES = {
    "success",
    "blocked",
    "deferred",
    "unresolved",
    "failure",
    "failed",
}
_SYMBOLIC_VALUES = {
    "viable",
    "conditional",
    "undetermined",
    "commit",
    "revise",
    "defer",
    "reject",
    "conditional_commit",
    "wait",
    "release",
    "match",
    "mismatch",
    "compatible",
    "contradicted",
    "dominates",
    "equivalent",
    "claim",
    "term",
    "schema",
    "procedure",
    "plan",
    "change",
}
_GENERIC_SENSITIVE_NAMES = {
    "game",
    "level",
    "palette",
    "coordinate",
    "oracle",
    "solution",
    "family",
}


@dataclass(frozen=True)
class ValidationResult:
    authority_ids: tuple[str, ...]


@dataclass(frozen=True)
class ProducedValue:
    definition: SymbolDefinition
    call: Expression


def _is_assignable(actual: TypeSpec, expected: TypeSpec) -> bool:
    if expected.name == "Any" or actual.name == "Any":
        return True
    if actual.name == "Null":
        return expected.optional
    if actual.optional and not expected.optional:
        return False
    if actual.name != expected.name or len(actual.arguments) != len(expected.arguments):
        return False
    return all(
        _is_assignable(actual_argument, expected_argument)
        for actual_argument, expected_argument in zip(
            actual.arguments, expected.arguments
        )
    )


def _expect_type(
    actual: TypeSpec,
    expected: TypeSpec,
    expression: Expression,
    description: str,
) -> None:
    if not _is_assignable(actual, expected):
        raise SrcValidationError(
            f"{description} expects {expected.display()}, got {actual.display()}",
            expression.span,
        )


class _WorkflowValidator:
    def __init__(
        self,
        module: ModuleAst,
        workflow: WorkflowDecl,
        environment: CompilerEnvironment,
    ) -> None:
        self.module = module
        self.workflow = workflow
        self.environment = environment
        self.authorities: dict[str, SymbolDefinition] = {}
        self.step_names = {step.name for step in workflow.steps}
        self.edges: dict[str, set[str]] = {step.name: set() for step in workflow.steps}
        self.input_names = {binding.name for binding in workflow.inputs}

    def validate(self) -> tuple[str, ...]:
        self._validate_bindings()
        for step in self.workflow.steps:
            bindings = self._base_bindings()
            produced: dict[str, ProducedValue] = {}
            self._validate_sequence(
                step.statements,
                bindings,
                produced,
                step_name=step.name,
                in_transaction=False,
                inside_loop=False,
            )
            if not self._sequence_transfers(step.statements):
                raise SrcValidationError(
                    f"step {step.name} must end in NEXT or a terminal result",
                    step.span,
                )
        self._validate_graph()
        return tuple(sorted(self.authorities))

    def _base_bindings(self) -> dict[str, TypeSpec]:
        return {
            binding.name: TypeSpec.from_ast(binding.type_expression)
            for binding in (
                *self.workflow.inputs,
                *self.workflow.outputs,
                *self.workflow.state,
            )
        }

    def _validate_bindings(self) -> None:
        all_bindings = (
            *self.workflow.inputs,
            *self.workflow.outputs,
            *self.workflow.state,
        )
        names = tuple(binding.name for binding in all_bindings)
        if len(names) != len(set(names)):
            raise SrcValidationError(
                f"duplicate binding in workflow {self.workflow.name}",
                self.workflow.span,
            )
        for binding in all_bindings:
            self._validate_type(binding.type_expression)
            if binding.default is not None:
                actual = self._infer_expression(binding.default, self._base_bindings())
                _expect_type(
                    actual,
                    TypeSpec.from_ast(binding.type_expression),
                    binding.default,
                    f"default for {binding.name}",
                )

    def _validate_type(self, value: TypeExpression) -> None:
        if value.name not in self.environment.types:
            raise SrcValidationError(f"unknown SRC type {value.name}", value.span)
        required_arity = {"List": 1, "Set": 1, "Map": 2}.get(value.name, 0)
        if len(value.arguments) != required_arity:
            raise SrcValidationError(
                f"type {value.name} requires {required_arity} type arguments",
                value.span,
            )
        for argument in value.arguments:
            self._validate_type(argument)

    def _infer_expression(
        self,
        expression: Expression,
        bindings: dict[str, TypeSpec],
    ) -> TypeSpec:
        if expression.kind is ExpressionKind.STRING:
            return type_spec("String")
        if expression.kind is ExpressionKind.INTEGER:
            return type_spec("Integer")
        if expression.kind is ExpressionKind.BOOLEAN:
            return type_spec("Boolean")
        if expression.kind is ExpressionKind.NULL:
            return type_spec("Null")
        if expression.kind is ExpressionKind.EMPTY_SET:
            return type_spec("Set", type_spec("Any"))
        if expression.kind is ExpressionKind.LIST:
            if not expression.items:
                return type_spec("List", type_spec("Any"))
            item_types = tuple(
                self._infer_expression(item, bindings) for item in expression.items
            )
            first = item_types[0]
            if any(
                not _is_assignable(item, first) or not _is_assignable(first, item)
                for item in item_types[1:]
            ):
                first = type_spec("Any")
            return type_spec("List", first)
        if expression.kind is ExpressionKind.REFERENCE:
            return self._resolve_reference(str(expression.value), expression, bindings)
        if expression.kind is ExpressionKind.CALL:
            definition = self._resolve_symbol(str(expression.value), expression)
            if definition.effect_class is not EffectClass.PURE:
                raise SrcValidationError(
                    f"non-pure authority {definition.id} cannot execute as an expression",
                    expression.span,
                )
            self._validate_call_arguments(expression, definition, bindings)
            return definition.output_type
        raise SrcValidationError("unsupported expression kind", expression.span)

    def _resolve_reference(
        self,
        reference: str,
        expression: Expression,
        bindings: dict[str, TypeSpec],
    ) -> TypeSpec:
        if reference in bindings:
            return bindings[reference]
        root = reference.split(".", 1)[0]
        if root in bindings and "." in reference:
            return type_spec("Any")
        if reference in _TERMINAL_VALUES:
            return type_spec("TerminalResult")
        if reference in _SYMBOLIC_VALUES:
            return type_spec("Value")
        definition = self.environment.symbol_index.get(reference)
        if definition is not None:
            self.authorities[definition.id] = definition
            return {
                SymbolKind.PREDICATE: type_spec("Predicate"),
                SymbolKind.RULE: type_spec("Rule"),
                SymbolKind.MODEL: type_spec("Model"),
                SymbolKind.CRITERION: type_spec("Criterion"),
                SymbolKind.POLICY: type_spec("Policy"),
                SymbolKind.CAPABILITY: type_spec("Capability"),
                SymbolKind.WORKFLOW: type_spec("Procedure"),
                SymbolKind.LOGOS_CONTRACT: type_spec("LogosContract"),
                SymbolKind.DRM_DECLARATION: type_spec("DrmDeclaration"),
                SymbolKind.WISDOM_WORKFLOW: type_spec("Procedure"),
                SymbolKind.ACTION_RELEASE_WORKFLOW: type_spec("Procedure"),
                SymbolKind.ACQUISITION_MODEL: type_spec("AcquisitionModel"),
                SymbolKind.ENVIRONMENT_ADAPTER: type_spec("EnvironmentAdapter"),
                SymbolKind.VALUE: definition.output_type,
            }[definition.kind]
        raise SrcValidationError(f"unresolved reference {reference}", expression.span)

    def _resolve_symbol(
        self,
        reference: str,
        expression: Expression,
    ) -> SymbolDefinition:
        definition = self.environment.symbol_index.get(reference)
        if definition is None:
            raise SrcValidationError(
                f"unresolved callable authority {reference}", expression.span
            )
        self.authorities[definition.id] = definition
        return definition

    def _validate_call_arguments(
        self,
        expression: Expression,
        definition: SymbolDefinition,
        bindings: dict[str, TypeSpec],
    ) -> None:
        parameters = definition.parameters
        parameter_index = {parameter.name: parameter for parameter in parameters}
        assigned: dict[str, Expression] = {}
        positional_index = 0
        for argument in expression.arguments:
            if argument.name is None:
                if positional_index >= len(parameters):
                    raise SrcValidationError(
                        f"too many arguments for {definition.id}",
                        argument.span,
                    )
                resolved_parameter = parameters[positional_index]
                positional_index += 1
            else:
                named_parameter = parameter_index.get(argument.name)
                if named_parameter is None:
                    raise SrcValidationError(
                        f"unknown argument {argument.name} for {definition.id}",
                        argument.span,
                    )
                resolved_parameter = named_parameter
            parameter = resolved_parameter
            if parameter.name in assigned:
                raise SrcValidationError(
                    f"duplicate argument {parameter.name} for {definition.id}",
                    argument.span,
                )
            assigned[parameter.name] = argument.value
            actual = self._infer_expression(argument.value, bindings)
            _expect_type(
                actual,
                parameter.type_spec,
                argument.value,
                f"argument {parameter.name} for {definition.id}",
            )
        missing = tuple(
            parameter.name
            for parameter in parameters
            if parameter.required and parameter.name not in assigned
        )
        if missing:
            raise SrcValidationError(
                f"missing arguments for {definition.id}: {list(missing)}",
                expression.span,
            )

    def _validate_sequence(
        self,
        statements: tuple[Statement, ...],
        bindings: dict[str, TypeSpec],
        produced: dict[str, ProducedValue],
        *,
        step_name: str,
        in_transaction: bool,
        inside_loop: bool,
    ) -> None:
        if not statements:
            raise SrcValidationError("empty SRC statement block", self.workflow.span)
        for index, statement in enumerate(statements):
            if self._statement_transfers(statement) and index != len(statements) - 1:
                raise SrcValidationError(
                    "statement after unconditional transfer is unreachable",
                    statements[index + 1].span,
                )
            if statement.kind is StatementKind.ACT:
                if (
                    index + 1 >= len(statements)
                    or statements[index + 1].kind is not StatementKind.AWAIT_FRAME
                ):
                    raise SrcValidationError(
                        "ACT must be followed immediately by AWAIT FRAME",
                        statement.span,
                    )
            if statement.kind is StatementKind.AWAIT_FRAME:
                if index == 0 or statements[index - 1].kind is not StatementKind.ACT:
                    raise SrcValidationError(
                        "AWAIT FRAME requires the immediately preceding ACT",
                        statement.span,
                    )
            self._validate_statement(
                statement,
                bindings,
                produced,
                step_name=step_name,
                in_transaction=in_transaction,
                inside_loop=inside_loop,
            )

    def _validate_statement(
        self,
        statement: Statement,
        bindings: dict[str, TypeSpec],
        produced: dict[str, ProducedValue],
        *,
        step_name: str,
        in_transaction: bool,
        inside_loop: bool,
    ) -> None:
        kind = statement.kind
        if kind is StatementKind.LET:
            target = statement.target
            expression = statement.expression
            assert target is not None and expression is not None
            if target in bindings:
                raise SrcValidationError(
                    f"LET cannot redefine binding {target}",
                    statement.span,
                )
            actual = self._infer_expression(expression, bindings)
            if statement.type_annotation:
                self._validate_type(statement.type_annotation)
                declared = TypeSpec.from_ast(statement.type_annotation)
                _expect_type(actual, declared, expression, "LET expression")
                bindings[target] = declared
            else:
                bindings[target] = actual
            return
        if kind is StatementKind.CALL:
            expression = statement.expression
            assert expression is not None
            definition = self._resolve_symbol(str(expression.value), expression)
            if definition.kind not in {
                SymbolKind.CAPABILITY,
                SymbolKind.WORKFLOW,
                SymbolKind.WISDOM_WORKFLOW,
                SymbolKind.ACTION_RELEASE_WORKFLOW,
            }:
                raise SrcValidationError(
                    f"CALL requires capability or workflow, got {definition.kind.value}",
                    statement.span,
                )
            if definition.effect_class is not EffectClass.PURE:
                raise SrcValidationError(
                    "CALL cannot hide observation, cognitive update, or environment effects",
                    statement.span,
                )
            self._validate_call_arguments(expression, definition, bindings)
            if statement.result_binding:
                if statement.result_binding in produced:
                    raise SrcValidationError(
                        f"CALL cannot overwrite produced lineage {statement.result_binding}",
                        statement.span,
                    )
                existing = bindings.get(statement.result_binding)
                if existing is None:
                    bindings[statement.result_binding] = definition.output_type
                elif not _is_assignable(definition.output_type, existing):
                    raise SrcValidationError(
                        f"CALL output {definition.output_type.display()} cannot bind to {existing.display()}",
                        statement.span,
                    )
                produced[statement.result_binding] = ProducedValue(
                    definition=definition,
                    call=expression,
                )
            return
        if kind in MACRO_STATEMENT_KINDS:
            raise SrcValidationError(
                f"macro {kind.value} has no registered primitive expansion",
                statement.span,
            )
        if kind in OPERATOR_STATEMENT_KINDS:
            self._validate_operator(
                statement,
                bindings,
                produced,
                in_transaction=in_transaction,
            )
            return
        if kind in {StatementKind.REQUIRE, StatementKind.IF}:
            expression = statement.expression
            assert expression is not None
            condition_type = self._infer_expression(expression, bindings)
            _expect_type(condition_type, type_spec("Boolean"), expression, kind.value)
            if kind is StatementKind.IF:
                self._validate_sequence(
                    statement.body,
                    dict(bindings),
                    dict(produced),
                    step_name=step_name,
                    in_transaction=in_transaction,
                    inside_loop=inside_loop,
                )
                if statement.else_body:
                    self._validate_sequence(
                        statement.else_body,
                        dict(bindings),
                        dict(produced),
                        step_name=step_name,
                        in_transaction=in_transaction,
                        inside_loop=inside_loop,
                    )
            return
        if kind is StatementKind.MATCH:
            expression = statement.expression
            assert expression is not None
            self._infer_expression(expression, bindings)
            labels_seen: set[str] = set()
            for arm in statement.match_arms:
                for label in arm.labels:
                    self._infer_expression(label, bindings)
                    identity = stable_digest(label.semantic_data())
                    if identity in labels_seen:
                        raise SrcValidationError("duplicate MATCH label", label.span)
                    labels_seen.add(identity)
                self._validate_statement(
                    arm.transition,
                    dict(bindings),
                    dict(produced),
                    step_name=step_name,
                    in_transaction=in_transaction,
                    inside_loop=inside_loop,
                )
            return
        if kind is StatementKind.FOR_EACH:
            expression = statement.expression
            iterator = statement.iterator
            assert expression is not None and iterator is not None
            collection_type = self._infer_expression(expression, bindings)
            if (
                collection_type.name not in {"List", "Set"}
                or len(collection_type.arguments) != 1
            ):
                raise SrcValidationError(
                    "FOR EACH requires a finite List or Set", expression.span
                )
            nested_bindings = dict(bindings)
            if iterator in nested_bindings:
                raise SrcValidationError(
                    f"FOR EACH iterator shadows binding {iterator}",
                    statement.span,
                )
            nested_bindings[iterator] = collection_type.arguments[0]
            self._validate_sequence(
                statement.body,
                nested_bindings,
                dict(produced),
                step_name=step_name,
                in_transaction=in_transaction,
                inside_loop=inside_loop,
            )
            return
        if kind is StatementKind.WHILE:
            expression = statement.expression
            contract = statement.loop_contract
            assert expression is not None and contract is not None
            condition_type = self._infer_expression(expression, bindings)
            _expect_type(condition_type, type_spec("Boolean"), expression, "WHILE")
            if contract.progress_witness.kind not in {
                ExpressionKind.REFERENCE,
                ExpressionKind.CALL,
            }:
                raise SrcValidationError(
                    "WHILE progress witness must be a reference or pure predicate call",
                    contract.progress_witness.span,
                )
            self._infer_expression(contract.progress_witness, bindings)
            stop_type = self._infer_expression(contract.stop_condition, bindings)
            _expect_type(
                stop_type,
                type_spec("Boolean"),
                contract.stop_condition,
                "WHILE STOP WHEN",
            )
            limit_type = self._infer_expression(contract.limit_result, bindings)
            _expect_type(
                limit_type,
                type_spec("TerminalResult"),
                contract.limit_result,
                "WHILE ON LIMIT DEFER",
            )
            if (
                contract.limit_result.kind is not ExpressionKind.REFERENCE
                or not isinstance(contract.limit_result.value, str)
                or contract.limit_result.value not in {"unresolved", "deferred"}
            ):
                raise SrcValidationError(
                    "WHILE limit must defer as unresolved or deferred, never success/fixpoint",
                    contract.limit_result.span,
                )
            self._validate_sequence(
                statement.body,
                dict(bindings),
                dict(produced),
                step_name=step_name,
                in_transaction=in_transaction,
                inside_loop=True,
            )
            return
        if kind is StatementKind.TRANSACTION:
            if in_transaction:
                raise SrcValidationError(
                    "nested TRANSACTION is forbidden", statement.span
                )
            if not any(item.kind is StatementKind.UPDATE for item in statement.body):
                raise SrcValidationError(
                    "TRANSACTION requires at least one UPDATE", statement.span
                )
            self._validate_sequence(
                statement.body,
                dict(bindings),
                dict(produced),
                step_name=step_name,
                in_transaction=True,
                inside_loop=inside_loop,
            )
            return
        if kind is StatementKind.AWAIT_FRAME:
            expression = statement.expression
            assert expression is not None
            reference = str(expression.value)
            existing = bindings.get(reference)
            if existing is None:
                bindings[reference] = type_spec("Frame")
            elif not _is_assignable(type_spec("Frame"), existing):
                raise SrcValidationError(
                    "AWAIT FRAME target must have type Frame", statement.span
                )
            return
        if kind is StatementKind.NEXT:
            expression = statement.expression
            assert expression is not None
            if expression.kind is not ExpressionKind.REFERENCE:
                raise SrcValidationError(
                    "NEXT requires a step reference", statement.span
                )
            target = str(expression.value)
            if target not in self.step_names:
                raise SrcValidationError(
                    f"NEXT references unknown step {target}", statement.span
                )
            self.edges[step_name].add(target)
            return
        if kind in TERMINAL_STATEMENT_KINDS:
            expression = statement.expression
            assert expression is not None
            self._infer_expression(expression, bindings)
            return
        raise SrcValidationError(f"unvalidated statement {kind.value}", statement.span)

    def _clause_index(self, statement: Statement) -> dict[str, Expression]:
        return {clause.name: clause.value for clause in statement.clauses}

    def _require_clause_shape(
        self,
        statement: Statement,
        required: tuple[str, ...],
        optional: tuple[str, ...] = (),
    ) -> dict[str, Expression]:
        names = tuple(clause.name for clause in statement.clauses)
        allowed = set(required) | set(optional)
        missing = tuple(name for name in required if name not in names)
        unexpected = tuple(name for name in names if name not in allowed)
        if missing or unexpected:
            raise SrcValidationError(
                f"{statement.kind.value} clause contract missing={list(missing)} unexpected={list(unexpected)}",
                statement.span,
            )
        return self._clause_index(statement)

    def _target_type(
        self,
        statement: Statement,
        bindings: dict[str, TypeSpec],
    ) -> TypeSpec:
        target = statement.target
        assert target is not None
        target_type = bindings.get(target)
        if target_type is None:
            raise SrcValidationError(
                f"{statement.kind.value} target is not a declared binding: {statement.target}",
                statement.span,
            )
        return target_type

    def _require_authority(
        self,
        expression: Expression,
        allowed_kinds: set[SymbolKind],
        *,
        required_effect: EffectClass | None = None,
    ) -> SymbolDefinition:
        if expression.kind is not ExpressionKind.REFERENCE:
            raise SrcValidationError(
                "authority must be one named reference", expression.span
            )
        definition = self._resolve_symbol(str(expression.value), expression)
        if definition.kind not in allowed_kinds:
            expected = sorted(item.value for item in allowed_kinds)
            raise SrcValidationError(
                f"authority {definition.id} has kind {definition.kind.value}, expected {expected}",
                expression.span,
            )
        if (
            required_effect is not None
            and definition.effect_class is not required_effect
        ):
            raise SrcValidationError(
                f"authority {definition.id} must have {required_effect.value} effect",
                expression.span,
            )
        return definition

    def _validate_operator(
        self,
        statement: Statement,
        bindings: dict[str, TypeSpec],
        produced: dict[str, ProducedValue],
        *,
        in_transaction: bool,
    ) -> None:
        kind = statement.kind
        target_type = self._target_type(statement, bindings)
        if kind is StatementKind.OBSERVE:
            clauses = self._require_clause_shape(statement, ("FROM", "USING"))
            self._infer_expression(clauses["FROM"], bindings)
            self._require_authority(
                clauses["USING"],
                {SymbolKind.ACQUISITION_MODEL},
                required_effect=EffectClass.OBSERVATION_ADAPTER,
            )
            if target_type.name not in {"ObservationResult", "Event", "Any"}:
                raise SrcValidationError(
                    "OBSERVE target must hold observation output", statement.span
                )
            return
        if kind is StatementKind.DISTINGUISH:
            clauses = self._require_clause_shape(statement, ("FROM", "USING"))
            self._infer_expression(clauses["FROM"], bindings)
            self._require_authority(
                clauses["USING"],
                {SymbolKind.CRITERION, SymbolKind.MODEL},
                required_effect=EffectClass.PURE,
            )
            return
        if kind is StatementKind.RELATE:
            clauses = self._require_clause_shape(
                statement,
                ("PREDICATE", "ARGUMENTS", "SCOPE", "BASIS"),
            )
            self._require_authority(
                clauses["PREDICATE"],
                {SymbolKind.PREDICATE},
                required_effect=EffectClass.PURE,
            )
            for name in ("ARGUMENTS", "SCOPE", "BASIS"):
                self._infer_expression(clauses[name], bindings)
            if target_type.name not in {"Claim", "Any"}:
                raise SrcValidationError(
                    "RELATE target must have type Claim", statement.span
                )
            return
        if kind is StatementKind.PROPOSE:
            clauses = self._require_clause_shape(
                statement,
                ("KIND", "BASIS", "SCOPE", "ALTERNATIVES", "MISSING", "FALSIFIERS"),
            )
            for value in (clauses[__yf_order_key] for __yf_order_key in sorted(clauses)):
                self._infer_expression(value, bindings)
            return
        if kind is StatementKind.DERIVE:
            clauses = self._require_clause_shape(
                statement,
                ("FROM", "USING"),
                optional=("CONTEXT",),
            )
            self._infer_expression(clauses["FROM"], bindings)
            if (
                clauses["FROM"].kind is ExpressionKind.LIST
                and not clauses["FROM"].items
            ):
                raise SrcValidationError(
                    "DERIVE requires at least one exact premise", clauses["FROM"].span
                )
            self._require_authority(
                clauses["USING"],
                {SymbolKind.RULE, SymbolKind.MODEL},
                required_effect=EffectClass.PURE,
            )
            if "CONTEXT" in clauses:
                self._infer_expression(clauses["CONTEXT"], bindings)
            if target_type.name not in {"Claim", "Set", "List", "Any"}:
                raise SrcValidationError(
                    "DERIVE target must hold claims", statement.span
                )
            return
        if kind is StatementKind.COMPARE:
            clauses = self._require_clause_shape(statement, ("LEFT", "RIGHT", "USING"))
            self._infer_expression(clauses["LEFT"], bindings)
            self._infer_expression(clauses["RIGHT"], bindings)
            self._require_authority(
                clauses["USING"],
                {SymbolKind.CRITERION},
                required_effect=EffectClass.PURE,
            )
            if target_type.name not in {"Assessment", "Claim", "Any"}:
                raise SrcValidationError(
                    "COMPARE target must hold an assessment", statement.span
                )
            return
        if kind is StatementKind.UPDATE:
            if not in_transaction:
                raise SrcValidationError(
                    "UPDATE is allowed only inside TRANSACTION", statement.span
                )
            clauses = self._require_clause_shape(
                statement,
                (
                    "TARGET",
                    "FROM",
                    "USING",
                    "BEFORE",
                    "INVALIDATE",
                    "RECALCULATE",
                    "PRESERVE",
                ),
            )
            for name in (
                "TARGET",
                "FROM",
                "BEFORE",
                "INVALIDATE",
                "RECALCULATE",
                "PRESERVE",
            ):
                self._infer_expression(clauses[name], bindings)
            self._require_authority(
                clauses["USING"],
                {SymbolKind.POLICY},
                required_effect=EffectClass.COGNITIVE_UPDATE_ADAPTER,
            )
            if target_type.name not in {"StateChangeProposal", "Any"}:
                raise SrcValidationError(
                    "UPDATE target must hold StateChangeProposal", statement.span
                )
            return
        if kind is StatementKind.SELECT:
            clauses = self._require_clause_shape(
                statement,
                ("FOR", "FROM", "UNDER", "REQUIRE", "PRESERVE"),
                optional=("CONTEXT", "DOMINANCE", "FACTS"),
            )
            for name in (
                "FOR",
                "FROM",
                "REQUIRE",
                "PRESERVE",
                "DOMINANCE",
                "FACTS",
                "CONTEXT",
            ):
                if name in clauses:
                    self._infer_expression(clauses[name], bindings)
            self._require_authority(
                clauses["UNDER"],
                {SymbolKind.POLICY},
                required_effect=EffectClass.PURE,
            )
            return
        if kind is StatementKind.ACT:
            if in_transaction:
                raise SrcValidationError(
                    "ACT is forbidden inside TRANSACTION", statement.span
                )
            clauses = self._require_clause_shape(
                statement,
                ("FROM", "CONTEXT", "USING", "RELEASE", "WISDOM"),
                optional=("EXPECT",),
            )
            _expect_type(
                self._infer_expression(clauses["FROM"], bindings),
                type_spec("ActionIntent"),
                clauses["FROM"],
                "ACT FROM",
            )
            _expect_type(
                self._infer_expression(clauses["CONTEXT"], bindings),
                type_spec("WorldContext"),
                clauses["CONTEXT"],
                "ACT CONTEXT",
            )
            self._require_authority(
                clauses["USING"],
                {SymbolKind.ENVIRONMENT_ADAPTER},
                required_effect=EffectClass.ENVIRONMENT_ADAPTER,
            )
            release_ref, release_value = self._require_action_boundary_reference(
                clauses["RELEASE"],
                produced,
                bindings,
                type_spec("ActionReleasePermit"),
                "ACT RELEASE",
            )
            wisdom_ref, wisdom_value = self._require_action_boundary_reference(
                clauses["WISDOM"],
                produced,
                bindings,
                type_spec("CommitmentDecision"),
                "ACT WISDOM",
            )
            if release_ref == wisdom_ref:
                raise SrcValidationError(
                    "ACT release and wisdom lineage must be distinct", statement.span
                )
            if (
                wisdom_value is not None
                and wisdom_value.definition.kind is not SymbolKind.WISDOM_WORKFLOW
            ):
                raise SrcValidationError(
                    "ACT WISDOM must come from a registered wisdom workflow",
                    clauses["WISDOM"].span,
                )
            if (
                release_value is not None
                and release_value.definition.kind
                is not SymbolKind.ACTION_RELEASE_WORKFLOW
            ):
                raise SrcValidationError(
                    "ACT RELEASE must come from a registered action-release workflow",
                    clauses["RELEASE"].span,
                )
            if release_value is not None:
                self._validate_release_lineage(
                    release_value.call,
                    wisdom_ref=wisdom_ref,
                    intent=clauses["FROM"],
                    context=clauses["CONTEXT"],
                )
            if "EXPECT" in clauses:
                self._infer_expression(clauses["EXPECT"], bindings)
            if target_type.name not in {"ActionEvent", "Event", "Any"}:
                raise SrcValidationError(
                    "ACT target must hold ActionEvent", statement.span
                )
            return
        raise SrcValidationError(
            f"missing operator contract for {kind.value}", statement.span
        )

    def _require_produced_reference(
        self,
        expression: Expression,
        produced: dict[str, ProducedValue],
        expected: TypeSpec,
        description: str,
    ) -> tuple[str, ProducedValue]:
        if expression.kind is not ExpressionKind.REFERENCE:
            raise SrcValidationError(
                f"{description} requires binding reference", expression.span
            )
        reference = str(expression.value)
        produced_value = produced.get(reference)
        if produced_value is None:
            raise SrcValidationError(
                f"{description} requires a preceding source-hashed CALL result",
                expression.span,
            )
        if not _is_assignable(produced_value.definition.output_type, expected):
            raise SrcValidationError(
                f"{description} producer returns {produced_value.definition.output_type.display()}",
                expression.span,
            )
        return reference, produced_value

    def _require_action_boundary_reference(
        self,
        expression: Expression,
        produced: dict[str, ProducedValue],
        bindings: dict[str, TypeSpec],
        expected: TypeSpec,
        description: str,
    ) -> tuple[str, ProducedValue | None]:
        """Accept a source-typed precommitted artifact at a CONTINUE boundary.

        A parent workflow may already have completed DRM grounding and release
        authorization.  Requiring a new CALL inside the action-only child would
        reintroduce cognition into CONTINUE.  Input artifacts remain typed and
        immutable; the scheduler validates their runtime lineage at dispatch.
        """

        if (
            expression.kind is ExpressionKind.REFERENCE
            and str(expression.value) in self.input_names
        ):
            reference = str(expression.value)
            actual = bindings.get(reference)
            if actual is None:
                raise SrcValidationError(
                    f"{description} boundary input is not bound", expression.span
                )
            _expect_type(actual, expected, expression, description)
            return reference, None
        return self._require_produced_reference(
            expression, produced, expected, description
        )

    def _validate_release_lineage(
        self,
        call: Expression,
        *,
        wisdom_ref: str,
        intent: Expression,
        context: Expression,
    ) -> None:
        named = {
            argument.name: argument.value
            for argument in call.arguments
            if argument.name is not None
        }
        expected_references = {
            "wisdom": wisdom_ref,
            "intent": intent.value if intent.kind is ExpressionKind.REFERENCE else None,
            "context": context.value
            if context.kind is ExpressionKind.REFERENCE
            else None,
        }
        for name, expected in sorted(expected_references.items()):
            value = named.get(name)
            if (
                expected is None
                or value is None
                or value.kind is not ExpressionKind.REFERENCE
                or value.value != expected
            ):
                raise SrcValidationError(
                    f"action-release lineage must bind {name} to the ACT {name}",
                    call.span,
                )

    def _statement_transfers(self, statement: Statement) -> bool:
        if (
            statement.kind is StatementKind.NEXT
            or statement.kind in TERMINAL_STATEMENT_KINDS
        ):
            return True
        if statement.kind is StatementKind.IF:
            return (
                bool(statement.else_body)
                and self._sequence_transfers(statement.body)
                and self._sequence_transfers(statement.else_body)
            )
        if statement.kind is StatementKind.MATCH:
            return all(
                self._statement_transfers(arm.transition)
                for arm in statement.match_arms
            )
        return False

    def _sequence_transfers(self, statements: tuple[Statement, ...]) -> bool:
        return bool(statements) and self._statement_transfers(statements[-1])

    def _validate_graph(self) -> None:
        if not self.workflow.steps:
            raise SrcValidationError(
                "workflow requires at least one STEP", self.workflow.span
            )
        start = self.workflow.steps[0].name
        reachable: set[str] = set()
        stack = [start]
        while stack:
            step = stack.pop()
            if step in reachable:
                continue
            reachable.add(step)
            stack.extend(sorted(self.edges[step] - reachable))
        unreachable = sorted(self.step_names - reachable)
        if unreachable:
            raise SrcValidationError(
                f"unreachable workflow steps: {unreachable}",
                self.workflow.span,
            )

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(step: str) -> None:
            if step in visiting:
                raise SrcValidationError(
                    "unbounded workflow cycle; use witnessed WHILE",
                    self.workflow.span,
                )
            if step in visited:
                return
            visiting.add(step)
            for target in self.edges[step]:
                visit(target)
            visiting.remove(step)
            visited.add(step)

        visit(start)


def _walk_expressions(statement: Statement) -> tuple[Expression, ...]:
    expressions: list[Expression] = []
    if statement.expression:
        expressions.append(statement.expression)
    expressions.extend(clause.value for clause in statement.clauses)
    if statement.loop_contract:
        expressions.extend(
            (
                statement.loop_contract.progress_witness,
                statement.loop_contract.stop_condition,
                statement.loop_contract.limit_result,
            )
        )
    for arm in statement.match_arms:
        expressions.extend(arm.labels)
        expressions.extend(_walk_expressions(arm.transition))
    for nested in (*statement.body, *statement.else_body):
        expressions.extend(_walk_expressions(nested))
    return tuple(expressions)


def _expression_references(expression: Expression) -> tuple[str, ...]:
    references: list[str] = []
    if expression.kind is ExpressionKind.REFERENCE:
        references.append(str(expression.value))
    elif expression.kind is ExpressionKind.CALL:
        references.append(str(expression.value))
        for argument in expression.arguments:
            references.extend(_expression_references(argument.value))
    elif expression.kind is ExpressionKind.LIST:
        for item in expression.items:
            references.extend(_expression_references(item))
    return tuple(references)


def _validate_generic_firewall(module: ModuleAst) -> None:
    if module.module_id.split(".", 1)[0] not in {"arc3", "core", "generic", "logos"}:
        return
    for workflow in module.workflows:
        for binding in (*workflow.inputs, *workflow.outputs, *workflow.state):
            if binding.default is None:
                continue
            segments = set(binding.name.lower().replace("-", "_").split("_"))
            if segments & _GENERIC_SENSITIVE_NAMES:
                raise SrcValidationError(
                    "generic module cannot declare game/level/palette/coordinate constants",
                    binding.span,
                )
        for step in workflow.steps:
            for statement in step.statements:
                if statement.kind is StatementKind.LET and statement.target:
                    segments = set(
                        statement.target.lower().replace("-", "_").split("_")
                    )
                    if segments & _GENERIC_SENSITIVE_NAMES:
                        raise SrcValidationError(
                            "generic module cannot declare game/level/palette/coordinate constants",
                            statement.span,
                        )
                for expression in _walk_expressions(statement):
                    for reference in _expression_references(expression):
                        root_segments = set(
                            reference.lower()
                            .replace("-", "_")
                            .replace(".", "_")
                            .split("_")
                        )
                        if root_segments & {
                            "game",
                            "level",
                            "oracle",
                            "solution",
                            "family",
                        }:
                            raise SrcValidationError(
                                "generic module contains benchmark/family-specific reference",
                                expression.span,
                            )


def validate_module(
    module: ModuleAst,
    environment: CompilerEnvironment,
) -> ValidationResult:
    if not module.workflows:
        raise SrcValidationError(
            "SRC module must declare at least one workflow", module.span
        )
    workflow_names = tuple(workflow.name for workflow in module.workflows)
    if len(workflow_names) != len(set(workflow_names)):
        raise SrcValidationError("duplicate workflow identity", module.span)
    aliases = tuple(item.alias or item.module for item in module.imports)
    if len(aliases) != len(set(aliases)):
        raise SrcValidationError("duplicate import alias", module.span)
    imported_modules = tuple(item.module for item in module.imports)
    if len(imported_modules) != len(set(imported_modules)):
        raise SrcValidationError("duplicate imported module", module.span)
    _validate_generic_firewall(module)
    authorities: set[str] = set()
    for workflow in module.workflows:
        if not workflow.outputs:
            raise SrcValidationError(
                f"workflow {workflow.name} requires an OUTPUT contract",
                workflow.span,
            )
        step_names = tuple(step.name for step in workflow.steps)
        if len(step_names) != len(set(step_names)):
            raise SrcValidationError("duplicate step identity", workflow.span)
        validator = _WorkflowValidator(module, workflow, environment)
        authorities.update(validator.validate())
    return ValidationResult(authority_ids=tuple(sorted(authorities)))
