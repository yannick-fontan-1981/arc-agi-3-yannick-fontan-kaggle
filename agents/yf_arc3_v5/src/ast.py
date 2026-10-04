"""Typed, recursively immutable AST for SRC v0.1."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field, model_validator

from agents.yf_arc3_v5.logos.types import FrozenModel, Ref, require_unique
from agents.yf_arc3_v5.src.tokens import SourceSpan


class ExpressionKind(str, Enum):
    REFERENCE = "reference"
    STRING = "string"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    NULL = "null"
    LIST = "list"
    EMPTY_SET = "empty_set"
    CALL = "call"


class CallArgument(FrozenModel):
    name: Ref | None = None
    value: "Expression"
    span: SourceSpan


class Expression(FrozenModel):
    kind: ExpressionKind
    value: str | int | bool | None = None
    items: tuple["Expression", ...] = ()
    arguments: tuple[CallArgument, ...] = ()
    span: SourceSpan

    @model_validator(mode="after")
    def validate_shape(self) -> "Expression":
        if self.kind in {
            ExpressionKind.REFERENCE,
            ExpressionKind.STRING,
            ExpressionKind.INTEGER,
            ExpressionKind.BOOLEAN,
        }:
            if self.value is None or self.items or self.arguments:
                raise ValueError(f"invalid scalar expression shape: {self.kind.value}")
        elif self.kind is ExpressionKind.NULL:
            if self.value is not None or self.items or self.arguments:
                raise ValueError("invalid null expression shape")
        elif self.kind is ExpressionKind.LIST:
            if self.value is not None or self.arguments:
                raise ValueError("invalid list expression shape")
        elif self.kind is ExpressionKind.EMPTY_SET:
            if self.value is not None or self.items or self.arguments:
                raise ValueError("invalid empty-set expression shape")
        elif self.kind is ExpressionKind.CALL:
            if not isinstance(self.value, str) or self.items:
                raise ValueError("invalid call expression shape")
            named = tuple(argument.name for argument in self.arguments if argument.name)
            require_unique(named, "named call arguments")
            saw_named = False
            for argument in self.arguments:
                if argument.name:
                    saw_named = True
                elif saw_named:
                    raise ValueError(
                        "positional call arguments must precede named arguments"
                    )
        return self

    @classmethod
    def reference(cls, value: Ref, span: SourceSpan) -> "Expression":
        return cls(kind=ExpressionKind.REFERENCE, value=value, span=span)

    def semantic_data(self) -> dict[str, Any]:
        if self.kind in {
            ExpressionKind.REFERENCE,
            ExpressionKind.STRING,
            ExpressionKind.INTEGER,
            ExpressionKind.BOOLEAN,
        }:
            return {"kind": self.kind.value, "value": self.value}
        if self.kind is ExpressionKind.NULL:
            return {"kind": self.kind.value}
        if self.kind is ExpressionKind.LIST:
            return {
                "kind": self.kind.value,
                "items": [item.semantic_data() for item in self.items],
            }
        if self.kind is ExpressionKind.EMPTY_SET:
            return {"kind": self.kind.value}
        return {
            "kind": self.kind.value,
            "callee": self.value,
            "arguments": [
                {"name": argument.name, "value": argument.value.semantic_data()}
                for argument in self.arguments
            ],
        }


class TypeExpression(FrozenModel):
    name: Ref
    arguments: tuple["TypeExpression", ...] = ()
    optional: bool = False
    span: SourceSpan

    def display(self) -> str:
        generic = ""
        if self.arguments:
            generic = (
                "<" + ",".join(argument.display() for argument in self.arguments) + ">"
            )
        optional = "?" if self.optional else ""
        return f"{self.name}{generic}{optional}"

    def semantic_data(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": [argument.semantic_data() for argument in self.arguments],
            "optional": self.optional,
        }


class ImportDecl(FrozenModel):
    module: Ref
    alias: Ref | None = None
    span: SourceSpan


class BindingDecl(FrozenModel):
    name: Ref
    type_expression: TypeExpression
    default: Expression | None = None
    span: SourceSpan


class Clause(FrozenModel):
    name: Ref
    value: Expression
    span: SourceSpan


class StatementKind(str, Enum):
    LET = "LET"
    CALL = "CALL"
    OBSERVE = "OBSERVE"
    ACT = "ACT"
    DISTINGUISH = "DISTINGUISH"
    RELATE = "RELATE"
    PROPOSE = "PROPOSE"
    DERIVE = "DERIVE"
    COMPARE = "COMPARE"
    UPDATE = "UPDATE"
    SELECT = "SELECT"
    EXPECT = "EXPECT"
    TEST = "TEST"
    TRACK = "TRACK"
    MEASURE = "MEASURE"
    PROPAGATE = "PROPAGATE"
    OPERATIONAL_WISDOM = "OPERATIONAL_WISDOM"
    IF = "IF"
    MATCH = "MATCH"
    FOR_EACH = "FOR_EACH"
    WHILE = "WHILE"
    TRANSACTION = "TRANSACTION"
    REQUIRE = "REQUIRE"
    AWAIT_FRAME = "AWAIT_FRAME"
    NEXT = "NEXT"
    STOP = "STOP"
    BLOCK = "BLOCK"
    DEFER = "DEFER"
    FAIL = "FAIL"


OPERATOR_STATEMENT_KINDS = frozenset(
    {
        StatementKind.OBSERVE,
        StatementKind.ACT,
        StatementKind.DISTINGUISH,
        StatementKind.RELATE,
        StatementKind.PROPOSE,
        StatementKind.DERIVE,
        StatementKind.COMPARE,
        StatementKind.UPDATE,
        StatementKind.SELECT,
    }
)

MACRO_STATEMENT_KINDS = frozenset(
    {
        StatementKind.EXPECT,
        StatementKind.TEST,
        StatementKind.TRACK,
        StatementKind.MEASURE,
        StatementKind.PROPAGATE,
        StatementKind.OPERATIONAL_WISDOM,
    }
)

TERMINAL_STATEMENT_KINDS = frozenset(
    {
        StatementKind.STOP,
        StatementKind.BLOCK,
        StatementKind.DEFER,
        StatementKind.FAIL,
    }
)


class MatchArm(FrozenModel):
    labels: tuple[Expression, ...] = Field(min_length=1)
    transition: "Statement"
    span: SourceSpan


class LoopContract(FrozenModel):
    progress_witness: Expression
    stop_condition: Expression
    maximum_symbolic_depth: int = Field(strict=True, ge=1)
    limit_result: Expression
    span: SourceSpan


class Statement(FrozenModel):
    kind: StatementKind
    target: Ref | None = None
    expression: Expression | None = None
    type_annotation: TypeExpression | None = None
    result_binding: Ref | None = None
    clauses: tuple[Clause, ...] = ()
    body: tuple["Statement", ...] = ()
    else_body: tuple["Statement", ...] = ()
    match_arms: tuple[MatchArm, ...] = ()
    iterator: Ref | None = None
    loop_contract: LoopContract | None = None
    span: SourceSpan

    @model_validator(mode="after")
    def validate_statement_shape(self) -> "Statement":
        require_unique(
            tuple(clause.name for clause in self.clauses), "statement clauses"
        )
        if self.kind is StatementKind.LET:
            if not self.target or self.expression is None:
                raise ValueError("LET requires target and expression")
        elif self.kind is StatementKind.CALL:
            if (
                self.expression is None
                or self.expression.kind is not ExpressionKind.CALL
            ):
                raise ValueError("CALL requires a call expression")
        elif self.kind in OPERATOR_STATEMENT_KINDS | MACRO_STATEMENT_KINDS:
            if not self.target:
                raise ValueError(f"{self.kind.value} requires a target")
        elif self.kind is StatementKind.IF:
            if self.expression is None or not self.body:
                raise ValueError("IF requires condition and body")
        elif self.kind is StatementKind.MATCH:
            if self.expression is None or not self.match_arms:
                raise ValueError("MATCH requires subject and arms")
        elif self.kind is StatementKind.FOR_EACH:
            if not self.iterator or self.expression is None or not self.body:
                raise ValueError("FOR EACH requires iterator, collection, and body")
        elif self.kind is StatementKind.WHILE:
            if self.expression is None or self.loop_contract is None or not self.body:
                raise ValueError("WHILE requires condition, loop contract, and body")
        elif self.kind is StatementKind.TRANSACTION:
            if not self.target or not self.body:
                raise ValueError("TRANSACTION requires identity and body")
        elif (
            self.kind
            in {
                StatementKind.REQUIRE,
                StatementKind.AWAIT_FRAME,
                StatementKind.NEXT,
            }
            | TERMINAL_STATEMENT_KINDS
        ):
            if self.expression is None:
                raise ValueError(f"{self.kind.value} requires an expression")
        return self


class StepDecl(FrozenModel):
    name: Ref
    statements: tuple[Statement, ...]
    span: SourceSpan


class WorkflowDecl(FrozenModel):
    name: Ref
    inputs: tuple[BindingDecl, ...]
    outputs: tuple[BindingDecl, ...]
    state: tuple[BindingDecl, ...]
    steps: tuple[StepDecl, ...]
    signature_form: bool = False
    span: SourceSpan


class ModuleAst(FrozenModel):
    module_id: Ref
    imports: tuple[ImportDecl, ...]
    workflows: tuple[WorkflowDecl, ...]
    span: SourceSpan
    schema_version: Ref = "yf_arc3_v5.src_ast.v0_1"


CallArgument.model_rebuild()
Expression.model_rebuild()
TypeExpression.model_rebuild()
MatchArm.model_rebuild()
Statement.model_rebuild()
