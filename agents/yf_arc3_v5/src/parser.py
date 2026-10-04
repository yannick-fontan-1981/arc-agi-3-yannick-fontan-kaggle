"""Deterministic parser for the bounded SRC v0.1 language."""

from __future__ import annotations

from pydantic import ValidationError

from agents.yf_arc3_v5.src.ast import (
    MACRO_STATEMENT_KINDS,
    OPERATOR_STATEMENT_KINDS,
    BindingDecl,
    CallArgument,
    Clause,
    Expression,
    ExpressionKind,
    ImportDecl,
    LoopContract,
    MatchArm,
    ModuleAst,
    Statement,
    StatementKind,
    StepDecl,
    TypeExpression,
    WorkflowDecl,
)
from agents.yf_arc3_v5.src.errors import SrcSyntaxError
from agents.yf_arc3_v5.src.lexer import lex
from agents.yf_arc3_v5.src.tokens import SourceSpan, Token, TokenKind

_OPERATION_KEYWORDS = {kind.value: kind for kind in OPERATOR_STATEMENT_KINDS}
_MACRO_KEYWORDS = {kind.value: kind for kind in MACRO_STATEMENT_KINDS}
_TERMINAL_KEYWORDS = {
    "STOP": StatementKind.STOP,
    "BLOCK": StatementKind.BLOCK,
    "DEFER": StatementKind.DEFER,
    "FAIL": StatementKind.FAIL,
}
_CLAUSE_KEYWORDS = {
    "ALTERNATIVES",
    "ARGUMENTS",
    "BASIS",
    "BEFORE",
    "CONTEXT",
    "DOMINANCE",
    "EXPECT",
    "FACTS",
    "FALSIFIERS",
    "FOR",
    "FROM",
    "INVALIDATE",
    "KIND",
    "LEFT",
    "MISSING",
    "PREDICATE",
    "PRESERVE",
    "RECALCULATE",
    "RELEASE",
    "REQUIRE",
    "RIGHT",
    "SCOPE",
    "TARGET",
    "UNDER",
    "USING",
    "WISDOM",
}
_REFERENCE_KEYWORDS = {"OPERATIONAL_WISDOM"}


def _merge_span(first: SourceSpan, last: SourceSpan) -> SourceSpan:
    return SourceSpan(source_name=first.source_name, start=first.start, end=last.end)


def _split_top_level(
    tokens: tuple[Token, ...],
    kind: TokenKind,
) -> tuple[tuple[Token, ...], ...]:
    groups: list[list[Token]] = [[]]
    depth = 0
    for token in tokens:
        if token.kind in {
            TokenKind.LPAREN,
            TokenKind.LBRACKET,
            TokenKind.LBRACE,
            TokenKind.LT,
        }:
            depth += 1
        elif token.kind in {
            TokenKind.RPAREN,
            TokenKind.RBRACKET,
            TokenKind.RBRACE,
            TokenKind.GT,
        }:
            depth -= 1
        if token.kind is kind and depth == 0:
            groups.append([])
        else:
            groups[-1].append(token)
    return tuple(tuple(group) for group in groups)


class _ExpressionParser:
    def __init__(self, tokens: tuple[Token, ...], description: str) -> None:
        if not tokens:
            raise SrcSyntaxError(f"expected {description}")
        self.tokens = tokens
        self.description = description
        self.index = 0

    @property
    def current(self) -> Token | None:
        return None if self.index >= len(self.tokens) else self.tokens[self.index]

    def advance(self) -> Token:
        token = self.current
        if token is None:
            raise SrcSyntaxError(
                f"unexpected end of {self.description}", self.tokens[-1].span
            )
        self.index += 1
        return token

    def match(self, kind: TokenKind) -> Token | None:
        if self.current is not None and self.current.kind is kind:
            return self.advance()
        return None

    def expect(self, kind: TokenKind, description: str) -> Token:
        token = self.match(kind)
        if token is None:
            span = self.tokens[-1].span if self.current is None else self.current.span
            raise SrcSyntaxError(f"expected {description}", span)
        return token

    def parse(self) -> Expression:
        expression = self.parse_expression()
        if self.current is not None:
            raise SrcSyntaxError(
                f"unexpected token in {self.description}", self.current.span
            )
        return expression

    def parse_expression(self) -> Expression:
        token = self.advance()
        if token.kind is TokenKind.STRING:
            return Expression(
                kind=ExpressionKind.STRING, value=token.value, span=token.span
            )
        if token.kind is TokenKind.INTEGER:
            return Expression(
                kind=ExpressionKind.INTEGER, value=int(token.value), span=token.span
            )
        if token.kind is TokenKind.LBRACKET:
            items: list[Expression] = []
            end = self.match(TokenKind.RBRACKET)
            if end is None:
                while True:
                    items.append(self.parse_expression())
                    end = self.match(TokenKind.RBRACKET)
                    if end is not None:
                        break
                    self.expect(TokenKind.COMMA, "',' or ']' in list")
            return Expression(
                kind=ExpressionKind.LIST,
                items=tuple(items),
                span=_merge_span(token.span, end.span),
            )
        if token.kind is TokenKind.LBRACE:
            end = self.expect(TokenKind.RBRACE, "'}' for empty map")
            return Expression(
                kind=ExpressionKind.EMPTY_SET,
                span=_merge_span(token.span, end.span),
            )
        if token.kind not in {TokenKind.IDENTIFIER, TokenKind.KEYWORD}:
            raise SrcSyntaxError(
                f"expected symbolic expression for {self.description}", token.span
            )
        if token.kind is TokenKind.KEYWORD and token.value not in _REFERENCE_KEYWORDS:
            raise SrcSyntaxError(
                f"keyword {token.value} is not an expression", token.span
            )

        lowered = token.value.lower()
        if lowered == "true":
            return Expression(kind=ExpressionKind.BOOLEAN, value=True, span=token.span)
        if lowered == "false":
            return Expression(kind=ExpressionKind.BOOLEAN, value=False, span=token.span)
        if lowered == "null":
            return Expression(kind=ExpressionKind.NULL, span=token.span)
        opening = self.match(TokenKind.LPAREN)
        if opening is None:
            return Expression.reference(token.value, token.span)

        arguments: list[CallArgument] = []
        end = self.match(TokenKind.RPAREN)
        if end is None:
            while True:
                argument_start = self.current
                if argument_start is None:
                    raise SrcSyntaxError("unterminated call expression", opening.span)
                name: str | None = None
                if (
                    argument_start.kind is TokenKind.IDENTIFIER
                    and self.index + 1 < len(self.tokens)
                    and self.tokens[self.index + 1].kind is TokenKind.EQUALS
                ):
                    name = self.advance().value
                    self.advance()
                value = self.parse_expression()
                arguments.append(
                    CallArgument(
                        name=name,
                        value=value,
                        span=_merge_span(argument_start.span, value.span),
                    )
                )
                end = self.match(TokenKind.RPAREN)
                if end is not None:
                    break
                self.expect(TokenKind.COMMA, "',' or ')' in call")
        return Expression(
            kind=ExpressionKind.CALL,
            value=token.value,
            arguments=tuple(arguments),
            span=_merge_span(token.span, end.span),
        )


def _parse_expression(tokens: tuple[Token, ...], description: str) -> Expression:
    return _ExpressionParser(tokens, description).parse()


class _TypeParser:
    def __init__(self, tokens: tuple[Token, ...]) -> None:
        if not tokens:
            raise SrcSyntaxError("expected type expression")
        self.tokens = tokens
        self.index = 0

    @property
    def current(self) -> Token | None:
        return None if self.index >= len(self.tokens) else self.tokens[self.index]

    def advance(self) -> Token:
        token = self.current
        if token is None:
            raise SrcSyntaxError(
                "unexpected end of type expression", self.tokens[-1].span
            )
        self.index += 1
        return token

    def match(self, kind: TokenKind) -> Token | None:
        if self.current is not None and self.current.kind is kind:
            return self.advance()
        return None

    def parse(self) -> TypeExpression:
        result = self.parse_type()
        if self.current is not None:
            raise SrcSyntaxError(
                "unexpected token in type expression", self.current.span
            )
        return result

    def parse_type(self) -> TypeExpression:
        name = self.advance()
        if name.kind is not TokenKind.IDENTIFIER:
            raise SrcSyntaxError("expected type name", name.span)
        arguments: list[TypeExpression] = []
        end_span = name.span
        if self.match(TokenKind.LT):
            while True:
                arguments.append(self.parse_type())
                closing = self.match(TokenKind.GT)
                if closing:
                    end_span = closing.span
                    break
                comma = self.match(TokenKind.COMMA)
                if comma is None:
                    span = end_span if self.current is None else self.current.span
                    raise SrcSyntaxError("expected ',' or '>' in generic type", span)
        optional = self.match(TokenKind.QUESTION)
        if optional:
            end_span = optional.span
        return TypeExpression(
            name=name.value,
            arguments=tuple(arguments),
            optional=optional is not None,
            span=_merge_span(name.span, end_span),
        )


def _parse_type(tokens: tuple[Token, ...]) -> TypeExpression:
    return _TypeParser(tokens).parse()


class _Parser:
    def __init__(self, tokens: tuple[Token, ...]) -> None:
        self.tokens = tokens
        self.index = 0

    @property
    def current(self) -> Token:
        return self.tokens[self.index]

    def advance(self) -> Token:
        token = self.current
        self.index += 1
        return token

    def match_kind(self, kind: TokenKind) -> Token | None:
        if self.current.kind is kind:
            return self.advance()
        return None

    def match_keyword(self, value: str) -> Token | None:
        if self.current.kind is TokenKind.KEYWORD and self.current.value == value:
            return self.advance()
        return None

    def expect_kind(self, kind: TokenKind, description: str) -> Token:
        token = self.match_kind(kind)
        if token is None:
            raise SrcSyntaxError(f"expected {description}", self.current.span)
        return token

    def expect_keyword(self, value: str) -> Token:
        token = self.match_keyword(value)
        if token is None:
            raise SrcSyntaxError(f"expected {value}", self.current.span)
        return token

    def expect_name(self, description: str = "identifier") -> Token:
        if self.current.kind is not TokenKind.IDENTIFIER:
            raise SrcSyntaxError(f"expected {description}", self.current.span)
        return self.advance()

    def skip_newlines(self) -> None:
        while self.match_kind(TokenKind.NEWLINE):
            pass

    def logical_line_tokens(self) -> tuple[Token, ...]:
        values: list[Token] = []
        depth = 0
        while True:
            token = self.current
            if token.kind is TokenKind.EOF:
                raise SrcSyntaxError("unexpected end of source line", token.span)
            if token.kind is TokenKind.NEWLINE:
                self.advance()
                if depth == 0:
                    return tuple(values)
                continue
            if token.kind in (TokenKind.INDENT, TokenKind.DEDENT):
                raise SrcSyntaxError(
                    "unexpected indentation inside a source line", token.span
                )
            token = self.advance()
            if token.kind in {TokenKind.LPAREN, TokenKind.LBRACKET, TokenKind.LBRACE}:
                depth += 1
            elif token.kind in {TokenKind.RPAREN, TokenKind.RBRACKET, TokenKind.RBRACE}:
                depth -= 1
            values.append(token)

    def parse(self) -> ModuleAst:
        module_keyword = self.expect_keyword("MODULE")
        module_name = self.expect_name("module identity")
        if self.logical_line_tokens():
            raise SrcSyntaxError(
                "unexpected content after module identity", self.current.span
            )

        imports: list[ImportDecl] = []
        while self.match_keyword("IMPORT"):
            import_name = self.expect_name("imported module identity")
            alias: str | None = None
            last = import_name
            if self.match_keyword("AS"):
                alias_token = self.expect_name("import alias")
                alias = alias_token.value
                last = alias_token
            trailing = self.logical_line_tokens()
            if trailing:
                raise SrcSyntaxError(
                    "unexpected content after import", trailing[0].span
                )
            imports.append(
                ImportDecl(
                    module=import_name.value,
                    alias=alias,
                    span=_merge_span(import_name.span, last.span),
                )
            )

        workflows: list[WorkflowDecl] = []
        while self.current.kind is not TokenKind.EOF:
            workflows.append(self.parse_workflow())
        eof = self.expect_kind(TokenKind.EOF, "end of source")
        return ModuleAst(
            module_id=module_name.value,
            imports=tuple(imports),
            workflows=tuple(workflows),
            span=_merge_span(module_keyword.span, eof.span),
        )

    def parse_workflow(self) -> WorkflowDecl:
        start = self.expect_keyword("WORKFLOW")
        name = self.expect_name("workflow identity")
        signature_form = self.current.kind is TokenKind.LPAREN
        signature_inputs: tuple[BindingDecl, ...] = ()
        signature_outputs: tuple[BindingDecl, ...] = ()
        if signature_form:
            signature_inputs, signature_outputs = self.parse_signature()
        else:
            trailing = self.logical_line_tokens()
            if trailing:
                raise SrcSyntaxError(
                    "unexpected workflow header content", trailing[0].span
                )
        self.expect_kind(TokenKind.INDENT, "indented workflow body")

        inputs = signature_inputs
        outputs = signature_outputs
        state: tuple[BindingDecl, ...] = ()
        steps: list[StepDecl] = []
        seen_sections: set[str] = set()
        section_order = {"INPUT": 0, "OUTPUT": 1, "STATE": 2}
        last_section_rank = -1
        steps_started = False
        while self.current.kind is not TokenKind.DEDENT:
            if self.current.kind is TokenKind.EOF:
                raise SrcSyntaxError("unterminated workflow body", self.current.span)
            if self.current.kind is TokenKind.KEYWORD and self.current.value in {
                "INPUT",
                "OUTPUT",
                "STATE",
            }:
                section = self.advance()
                if signature_form and section.value in {"INPUT", "OUTPUT"}:
                    raise SrcSyntaxError(
                        "signature form cannot mix INPUT/OUTPUT blocks",
                        section.span,
                    )
                if steps_started:
                    raise SrcSyntaxError(
                        f"{section.value} must appear before the first STEP",
                        section.span,
                    )
                if section.value in seen_sections:
                    raise SrcSyntaxError(
                        f"duplicate {section.value} section", section.span
                    )
                rank = section_order[section.value]
                if rank < last_section_rank:
                    raise SrcSyntaxError(
                        "workflow sections must be ordered INPUT, OUTPUT, STATE",
                        section.span,
                    )
                last_section_rank = rank
                seen_sections.add(section.value)
                bindings = self.parse_binding_block(section.value)
                if section.value == "INPUT":
                    inputs = bindings
                elif section.value == "OUTPUT":
                    outputs = bindings
                else:
                    state = bindings
                continue
            if self.current.kind is TokenKind.KEYWORD and self.current.value == "STEP":
                steps_started = True
                steps.append(self.parse_step())
                continue
            raise SrcSyntaxError(
                "workflow body accepts INPUT, OUTPUT, STATE, and STEP",
                self.current.span,
            )
        end = self.advance()
        return WorkflowDecl(
            name=name.value,
            inputs=inputs,
            outputs=outputs,
            state=state,
            steps=tuple(steps),
            signature_form=signature_form,
            span=_merge_span(start.span, end.span),
        )

    def parse_signature(
        self,
    ) -> tuple[tuple[BindingDecl, ...], tuple[BindingDecl, ...]]:
        self.expect_kind(TokenKind.LPAREN, "'('")
        self.skip_newlines()
        inputs: list[BindingDecl] = []
        if self.current.kind is not TokenKind.RPAREN:
            while True:
                binding_name = self.expect_name("parameter name")
                self.expect_kind(TokenKind.COLON, "':' after parameter name")
                type_start = self.index
                depth = 0
                while True:
                    token = self.current
                    if token.kind is TokenKind.NEWLINE:
                        self.advance()
                        continue
                    if token.kind is TokenKind.LT:
                        depth += 1
                    elif token.kind is TokenKind.GT:
                        depth -= 1
                    if depth == 0 and token.kind in {TokenKind.COMMA, TokenKind.RPAREN}:
                        break
                    self.advance()
                type_tokens = tuple(
                    token
                    for token in self.tokens[type_start : self.index]
                    if token.kind is not TokenKind.NEWLINE
                )
                type_expression = _parse_type(type_tokens)
                inputs.append(
                    BindingDecl(
                        name=binding_name.value,
                        type_expression=type_expression,
                        span=_merge_span(binding_name.span, type_expression.span),
                    )
                )
                if self.match_kind(TokenKind.RPAREN):
                    break
                self.expect_kind(TokenKind.COMMA, "',' between parameters")
                self.skip_newlines()
        else:
            self.advance()
        self.skip_newlines()
        self.expect_kind(TokenKind.ARROW, "'->' output type")
        output_tokens = self.logical_line_tokens()
        output_type = _parse_type(output_tokens)
        output = BindingDecl(
            name="result",
            type_expression=output_type,
            span=output_type.span,
        )
        return tuple(inputs), (output,)

    def parse_binding_block(self, section_name: str) -> tuple[BindingDecl, ...]:
        trailing = self.logical_line_tokens()
        if trailing:
            raise SrcSyntaxError(
                f"{section_name} must open an indented block", trailing[0].span
            )
        self.expect_kind(TokenKind.INDENT, f"indented {section_name} block")
        bindings: list[BindingDecl] = []
        while self.current.kind is not TokenKind.DEDENT:
            name = self.expect_name("binding name")
            self.expect_kind(TokenKind.COLON, "':' after binding name")
            line = self.logical_line_tokens()
            groups = _split_top_level(line, TokenKind.EQUALS)
            if len(groups) > 2:
                raise SrcSyntaxError("binding has more than one '='", groups[2][0].span)
            type_expression = _parse_type(groups[0])
            default = (
                None
                if len(groups) == 1
                else _parse_expression(groups[1], "default value")
            )
            last_span = default.span if default else type_expression.span
            bindings.append(
                BindingDecl(
                    name=name.value,
                    type_expression=type_expression,
                    default=default,
                    span=_merge_span(name.span, last_span),
                )
            )
        self.advance()
        return tuple(bindings)

    def parse_step(self) -> StepDecl:
        start = self.expect_keyword("STEP")
        name = self.expect_name("step identity")
        trailing = self.logical_line_tokens()
        if trailing:
            raise SrcSyntaxError(
                "unexpected content after step identity", trailing[0].span
            )
        statements, end = self.parse_block("step body")
        return StepDecl(
            name=name.value,
            statements=statements,
            span=_merge_span(start.span, end),
        )

    def parse_block(self, description: str) -> tuple[tuple[Statement, ...], SourceSpan]:
        self.expect_kind(TokenKind.INDENT, f"indented {description}")
        statements: list[Statement] = []
        while self.current.kind is not TokenKind.DEDENT:
            if self.current.kind is TokenKind.EOF:
                raise SrcSyntaxError(f"unterminated {description}", self.current.span)
            statements.append(self.parse_statement())
        end = self.advance().span
        return tuple(statements), end

    def parse_statement(self) -> Statement:
        token = self.current
        if token.kind is not TokenKind.KEYWORD:
            raise SrcSyntaxError("expected SRC statement", token.span)
        if token.value == "GENERALIZE":
            raise SrcSyntaxError("GENERALIZE is forbidden in SRC", token.span)
        if token.value == "LET":
            return self.parse_let()
        if token.value == "CALL":
            return self.parse_call()
        if token.value in _OPERATION_KEYWORDS or token.value in _MACRO_KEYWORDS:
            return self.parse_operation()
        if token.value == "IF":
            return self.parse_if()
        if token.value == "MATCH":
            return self.parse_match()
        if token.value == "FOR":
            return self.parse_for_each()
        if token.value == "WHILE":
            return self.parse_while()
        if token.value == "TRANSACTION":
            return self.parse_transaction()
        if token.value == "REQUIRE":
            return self.parse_simple_expression_statement(StatementKind.REQUIRE)
        if token.value == "AWAIT":
            return self.parse_await_frame()
        if token.value == "NEXT":
            return self.parse_simple_expression_statement(StatementKind.NEXT)
        if token.value in _TERMINAL_KEYWORDS:
            return self.parse_simple_expression_statement(
                _TERMINAL_KEYWORDS[token.value]
            )
        raise SrcSyntaxError(f"unsupported SRC statement {token.value}", token.span)

    def parse_let(self) -> Statement:
        start = self.expect_keyword("LET")
        target = self.expect_name("LET target")
        type_annotation: TypeExpression | None = None
        if self.match_kind(TokenKind.COLON):
            type_start = self.index
            while self.current.kind not in {TokenKind.EQUALS, TokenKind.NEWLINE}:
                self.advance()
            type_annotation = _parse_type(tuple(self.tokens[type_start : self.index]))
        self.expect_kind(TokenKind.EQUALS, "'=' in LET")
        expression = _parse_expression(self.logical_line_tokens(), "LET expression")
        return Statement(
            kind=StatementKind.LET,
            target=target.value,
            expression=expression,
            type_annotation=type_annotation,
            span=_merge_span(start.span, expression.span),
        )

    def parse_call(self) -> Statement:
        start = self.expect_keyword("CALL")
        line = self.logical_line_tokens()
        groups = _split_top_level(line, TokenKind.ARROW)
        if len(groups) > 2:
            raise SrcSyntaxError("CALL has more than one '->'", groups[2][0].span)
        call = _parse_expression(groups[0], "CALL expression")
        result_binding: str | None = None
        end_span = call.span
        if len(groups) == 2:
            result = _parse_expression(groups[1], "CALL result binding")
            if result.kind is not ExpressionKind.REFERENCE:
                raise SrcSyntaxError(
                    "CALL result must be a binding reference", result.span
                )
            result_binding = str(result.value)
            end_span = result.span
        return Statement(
            kind=StatementKind.CALL,
            expression=call,
            result_binding=result_binding,
            span=_merge_span(start.span, end_span),
        )

    def parse_operation(self) -> Statement:
        start = self.advance()
        kind = _OPERATION_KEYWORDS.get(start.value) or _MACRO_KEYWORDS[start.value]
        target = self.expect_name(f"{start.value} target")
        trailing = self.logical_line_tokens()
        if trailing:
            raise SrcSyntaxError(
                f"unexpected {start.value} header content", trailing[0].span
            )
        clauses: list[Clause] = []
        if self.match_kind(TokenKind.INDENT):
            while self.current.kind is not TokenKind.DEDENT:
                clause_name = self.current
                if (
                    clause_name.kind is not TokenKind.KEYWORD
                    or clause_name.value not in _CLAUSE_KEYWORDS
                ):
                    raise SrcSyntaxError(
                        "expected a named operator clause", clause_name.span
                    )
                self.advance()
                if any(clause.name == clause_name.value for clause in clauses):
                    raise SrcSyntaxError(
                        f"duplicate {start.value} clause {clause_name.value}",
                        clause_name.span,
                    )
                value = _parse_expression(
                    self.logical_line_tokens(),
                    f"{clause_name.value} clause value",
                )
                clauses.append(
                    Clause(
                        name=clause_name.value,
                        value=value,
                        span=_merge_span(clause_name.span, value.span),
                    )
                )
            self.advance()
        end_span = clauses[-1].span if clauses else target.span
        return Statement(
            kind=kind,
            target=target.value,
            clauses=tuple(clauses),
            span=_merge_span(start.span, end_span),
        )

    def parse_if(self) -> Statement:
        start = self.expect_keyword("IF")
        condition = _parse_expression(self.logical_line_tokens(), "IF condition")
        body, end = self.parse_block("IF body")
        else_body: tuple[Statement, ...] = ()
        if self.match_keyword("ELSE"):
            trailing = self.logical_line_tokens()
            if trailing:
                raise SrcSyntaxError(
                    "ELSE does not accept a condition", trailing[0].span
                )
            else_body, end = self.parse_block("ELSE body")
        return Statement(
            kind=StatementKind.IF,
            expression=condition,
            body=body,
            else_body=else_body,
            span=_merge_span(start.span, end),
        )

    def parse_match(self) -> Statement:
        start = self.expect_keyword("MATCH")
        subject = _parse_expression(self.logical_line_tokens(), "MATCH subject")
        self.expect_kind(TokenKind.INDENT, "indented MATCH arms")
        arms: list[MatchArm] = []
        while self.current.kind is not TokenKind.DEDENT:
            line = self.logical_line_tokens()
            groups = _split_top_level(line, TokenKind.ARROW)
            if len(groups) != 2:
                span = line[0].span if line else self.current.span
                raise SrcSyntaxError("MATCH arm requires one '->'", span)
            label_groups = _split_top_level(groups[0], TokenKind.COMMA)
            labels = tuple(
                _parse_expression(group, "MATCH label") for group in label_groups
            )
            transition = self.parse_inline_transition(groups[1])
            arms.append(
                MatchArm(
                    labels=labels,
                    transition=transition,
                    span=_merge_span(labels[0].span, transition.span),
                )
            )
        end = self.advance().span
        return Statement(
            kind=StatementKind.MATCH,
            expression=subject,
            match_arms=tuple(arms),
            span=_merge_span(start.span, end),
        )

    def parse_inline_transition(self, tokens: tuple[Token, ...]) -> Statement:
        if not tokens:
            raise SrcSyntaxError("empty MATCH transition")
        first = tokens[0]
        if first.kind is TokenKind.KEYWORD and first.value in {
            "NEXT",
            *tuple(_TERMINAL_KEYWORDS),
        }:
            kind = (
                StatementKind.NEXT
                if first.value == "NEXT"
                else _TERMINAL_KEYWORDS[first.value]
            )
            expression = _parse_expression(tokens[1:], f"{kind.value} result")
            return Statement(
                kind=kind,
                expression=expression,
                span=_merge_span(first.span, expression.span),
            )
        expression = _parse_expression(tokens, "MATCH target step")
        return Statement(
            kind=StatementKind.NEXT,
            expression=expression,
            span=expression.span,
        )

    def parse_for_each(self) -> Statement:
        start = self.expect_keyword("FOR")
        self.expect_keyword("EACH")
        iterator = self.expect_name("FOR EACH iterator")
        self.expect_keyword("IN")
        collection = _parse_expression(
            self.logical_line_tokens(), "FOR EACH collection"
        )
        body, end = self.parse_block("FOR EACH body")
        return Statement(
            kind=StatementKind.FOR_EACH,
            iterator=iterator.value,
            expression=collection,
            body=body,
            span=_merge_span(start.span, end),
        )

    def parse_while(self) -> Statement:
        start = self.expect_keyword("WHILE")
        condition = _parse_expression(self.logical_line_tokens(), "WHILE condition")
        self.expect_kind(TokenKind.INDENT, "indented WHILE contract")

        progress_start = self.expect_keyword("PROGRESS")
        progress = _parse_expression(self.logical_line_tokens(), "progress witness")
        self.expect_keyword("STOP")
        self.expect_keyword("WHEN")
        stop_condition = _parse_expression(
            self.logical_line_tokens(), "loop stop condition"
        )
        self.expect_keyword("MAX")
        self.expect_keyword("DEPTH")
        depth_tokens = self.logical_line_tokens()
        if len(depth_tokens) != 1 or depth_tokens[0].kind is not TokenKind.INTEGER:
            span = depth_tokens[0].span if depth_tokens else self.current.span
            raise SrcSyntaxError("MAX DEPTH requires one positive integer", span)
        maximum_depth = int(depth_tokens[0].value)
        if maximum_depth < 1:
            raise SrcSyntaxError(
                "MAX DEPTH requires one positive integer", depth_tokens[0].span
            )
        self.expect_keyword("ON")
        self.expect_keyword("LIMIT")
        self.expect_keyword("DEFER")
        limit_result = _parse_expression(
            self.logical_line_tokens(), "loop limit result"
        )
        self.expect_keyword("DO")
        trailing = self.logical_line_tokens()
        if trailing:
            raise SrcSyntaxError("DO opens a nested block", trailing[0].span)
        body, body_end = self.parse_block("WHILE body")
        end = self.expect_kind(TokenKind.DEDENT, "end of WHILE contract").span
        contract = LoopContract(
            progress_witness=progress,
            stop_condition=stop_condition,
            maximum_symbolic_depth=maximum_depth,
            limit_result=limit_result,
            span=_merge_span(progress_start.span, body_end),
        )
        return Statement(
            kind=StatementKind.WHILE,
            expression=condition,
            body=body,
            loop_contract=contract,
            span=_merge_span(start.span, end),
        )

    def parse_transaction(self) -> Statement:
        start = self.expect_keyword("TRANSACTION")
        identity = self.expect_name("transaction identity")
        trailing = self.logical_line_tokens()
        if trailing:
            raise SrcSyntaxError(
                "unexpected transaction header content", trailing[0].span
            )
        body, end = self.parse_block("TRANSACTION body")
        return Statement(
            kind=StatementKind.TRANSACTION,
            target=identity.value,
            body=body,
            span=_merge_span(start.span, end),
        )

    def parse_simple_expression_statement(self, kind: StatementKind) -> Statement:
        start = self.advance()
        expression = _parse_expression(
            self.logical_line_tokens(), f"{kind.value} expression"
        )
        return Statement(
            kind=kind,
            expression=expression,
            span=_merge_span(start.span, expression.span),
        )

    def parse_await_frame(self) -> Statement:
        start = self.expect_keyword("AWAIT")
        self.expect_keyword("FRAME")
        self.expect_keyword("AS")
        binding = self.expect_name("frame binding")
        trailing = self.logical_line_tokens()
        if trailing:
            raise SrcSyntaxError("unexpected AWAIT FRAME content", trailing[0].span)
        expression = Expression.reference(binding.value, binding.span)
        return Statement(
            kind=StatementKind.AWAIT_FRAME,
            expression=expression,
            span=_merge_span(start.span, binding.span),
        )


def parse(source: str, *, source_name: str = "<memory>") -> ModuleAst:
    parser = _Parser(lex(source, source_name=source_name))
    try:
        return parser.parse()
    except ValidationError as error:
        first = error.errors(include_url=False)[0]
        span = parser.tokens[min(parser.index, len(parser.tokens) - 1)].span
        raise SrcSyntaxError(
            f"invalid SRC structure: {first['msg']}",
            span,
        ) from error
