"""Small deterministic indentation-aware lexer for SRC v0.1."""

from __future__ import annotations

import json

from agents.yf_arc3_v5.src.errors import SrcSyntaxError
from agents.yf_arc3_v5.src.tokens import (
    KEYWORDS,
    SourcePosition,
    SourceSpan,
    Token,
    TokenKind,
)

_PUNCTUATION = {
    ":": TokenKind.COLON,
    ",": TokenKind.COMMA,
    "=": TokenKind.EQUALS,
    "?": TokenKind.QUESTION,
    "(": TokenKind.LPAREN,
    ")": TokenKind.RPAREN,
    "[": TokenKind.LBRACKET,
    "]": TokenKind.RBRACKET,
    "{": TokenKind.LBRACE,
    "}": TokenKind.RBRACE,
    "<": TokenKind.LT,
    ">": TokenKind.GT,
}


def _span(
    source_name: str,
    *,
    line: int,
    column: int,
    offset: int,
    length: int,
) -> SourceSpan:
    return SourceSpan(
        source_name=source_name,
        start=SourcePosition(line=line, column=column, offset=offset),
        end=SourcePosition(line=line, column=column + length, offset=offset + length),
    )


def lex(source: str, *, source_name: str = "<memory>") -> tuple[Token, ...]:
    tokens: list[Token] = []
    indent_stack = [0]
    absolute_offset = 0
    continuation_depth = 0

    for line_number, raw_line in enumerate(source.splitlines(keepends=True), start=1):
        line = raw_line.rstrip("\r\n")
        newline_width = len(raw_line) - len(line)
        if "\t" in line[: len(line) - len(line.lstrip(" \t"))]:
            raise SrcSyntaxError(
                "tabs are forbidden in SRC indentation",
                _span(
                    source_name,
                    line=line_number,
                    column=1,
                    offset=absolute_offset,
                    length=1,
                ),
            )

        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if not stripped or stripped.startswith("#"):
            absolute_offset += len(raw_line)
            continue
        continuation_line = continuation_depth > 0
        if not continuation_line and indent % 4:
            raise SrcSyntaxError(
                "indentation must use multiples of four spaces",
                _span(
                    source_name,
                    line=line_number,
                    column=1,
                    offset=absolute_offset,
                    length=max(indent, 1),
                ),
            )
        if not continuation_line:
            if indent > indent_stack[-1]:
                indent_stack.append(indent)
                tokens.append(
                    Token(
                        kind=TokenKind.INDENT,
                        value=str(indent),
                        span=_span(
                            source_name,
                            line=line_number,
                            column=1,
                            offset=absolute_offset,
                            length=indent,
                        ),
                    )
                )
            else:
                while indent < indent_stack[-1]:
                    indent_stack.pop()
                    tokens.append(
                        Token(
                            kind=TokenKind.DEDENT,
                            value=str(indent),
                            span=_span(
                                source_name,
                                line=line_number,
                                column=1,
                                offset=absolute_offset,
                                length=max(indent, 1),
                            ),
                        )
                    )
                if indent != indent_stack[-1]:
                    raise SrcSyntaxError(
                        "dedent does not match an earlier indentation level",
                        _span(
                            source_name,
                            line=line_number,
                            column=1,
                            offset=absolute_offset,
                            length=max(indent, 1),
                        ),
                    )

        index = indent
        while index < len(line):
            character = line[index]
            if character == " ":
                index += 1
                continue
            if character == "#":
                break
            column = index + 1
            token_offset = absolute_offset + index
            if line.startswith("->", index):
                tokens.append(
                    Token(
                        kind=TokenKind.ARROW,
                        value="->",
                        span=_span(
                            source_name,
                            line=line_number,
                            column=column,
                            offset=token_offset,
                            length=2,
                        ),
                    )
                )
                index += 2
                continue
            if character in _PUNCTUATION:
                if character in "([{":
                    continuation_depth += 1
                elif character in ")]}":
                    continuation_depth -= 1
                    if continuation_depth < 0:
                        raise SrcSyntaxError(
                            "unmatched closing delimiter",
                            _span(
                                source_name,
                                line=line_number,
                                column=column,
                                offset=token_offset,
                                length=1,
                            ),
                        )
                tokens.append(
                    Token(
                        kind=_PUNCTUATION[character],
                        value=character,
                        span=_span(
                            source_name,
                            line=line_number,
                            column=column,
                            offset=token_offset,
                            length=1,
                        ),
                    )
                )
                index += 1
                continue
            if character == '"':
                cursor = index + 1
                escaped = False
                while cursor < len(line):
                    current = line[cursor]
                    if current == '"' and not escaped:
                        break
                    escaped = current == "\\" and not escaped
                    if current != "\\":
                        escaped = False
                    cursor += 1
                if cursor >= len(line):
                    raise SrcSyntaxError(
                        "unterminated string literal",
                        _span(
                            source_name,
                            line=line_number,
                            column=column,
                            offset=token_offset,
                            length=len(line) - index,
                        ),
                    )
                literal = line[index : cursor + 1]
                try:
                    value = json.loads(literal)
                except json.JSONDecodeError as exc:
                    raise SrcSyntaxError(
                        "invalid JSON string literal",
                        _span(
                            source_name,
                            line=line_number,
                            column=column,
                            offset=token_offset,
                            length=len(literal),
                        ),
                    ) from exc
                tokens.append(
                    Token(
                        kind=TokenKind.STRING,
                        value=value,
                        span=_span(
                            source_name,
                            line=line_number,
                            column=column,
                            offset=token_offset,
                            length=len(literal),
                        ),
                    )
                )
                index = cursor + 1
                continue
            if character.isdigit():
                cursor = index + 1
                while cursor < len(line) and line[cursor].isdigit():
                    cursor += 1
                value = line[index:cursor]
                tokens.append(
                    Token(
                        kind=TokenKind.INTEGER,
                        value=value,
                        span=_span(
                            source_name,
                            line=line_number,
                            column=column,
                            offset=token_offset,
                            length=len(value),
                        ),
                    )
                )
                index = cursor
                continue
            if character.isalpha() or character == "_":
                cursor = index + 1
                while cursor < len(line) and (
                    line[cursor].isalnum() or line[cursor] in "_.-"
                ):
                    cursor += 1
                value = line[index:cursor]
                kind = TokenKind.KEYWORD if value in KEYWORDS else TokenKind.IDENTIFIER
                tokens.append(
                    Token(
                        kind=kind,
                        value=value,
                        span=_span(
                            source_name,
                            line=line_number,
                            column=column,
                            offset=token_offset,
                            length=len(value),
                        ),
                    )
                )
                index = cursor
                continue
            raise SrcSyntaxError(
                f"unexpected character {character!r}",
                _span(
                    source_name,
                    line=line_number,
                    column=column,
                    offset=token_offset,
                    length=1,
                ),
            )

        tokens.append(
            Token(
                kind=TokenKind.NEWLINE,
                value="",
                span=_span(
                    source_name,
                    line=line_number,
                    column=len(line) + 1,
                    offset=absolute_offset + len(line),
                    length=newline_width,
                ),
            )
        )
        absolute_offset += len(raw_line)

    eof_line = max(len(source.splitlines()), 1) + 1
    if continuation_depth:
        raise SrcSyntaxError(
            "unterminated expression delimiter",
            _span(
                source_name,
                line=eof_line,
                column=1,
                offset=len(source),
                length=0,
            ),
        )
    while len(indent_stack) > 1:
        indent_stack.pop()
        tokens.append(
            Token(
                kind=TokenKind.DEDENT,
                value="0",
                span=_span(
                    source_name,
                    line=eof_line,
                    column=1,
                    offset=len(source),
                    length=0,
                ),
            )
        )
    tokens.append(
        Token(
            kind=TokenKind.EOF,
            value="",
            span=_span(
                source_name,
                line=eof_line,
                column=1,
                offset=len(source),
                length=0,
            ),
        )
    )
    return tuple(tokens)
