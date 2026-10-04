"""SRC token and source-span contracts."""

from __future__ import annotations

from enum import Enum

from agents.yf_arc3_v5.logos.types import (
    FrozenModel,
    NonNegativeRevision,
    PositiveSequence,
    Ref,
)


class SourcePosition(FrozenModel):
    line: PositiveSequence
    column: PositiveSequence
    offset: NonNegativeRevision


class SourceSpan(FrozenModel):
    source_name: Ref
    start: SourcePosition
    end: SourcePosition

    def compact(self) -> str:
        return f"{self.source_name}:{self.start.line}:{self.start.column}"


class TokenKind(str, Enum):
    KEYWORD = "keyword"
    IDENTIFIER = "identifier"
    STRING = "string"
    INTEGER = "integer"
    COLON = "colon"
    COMMA = "comma"
    EQUALS = "equals"
    ARROW = "arrow"
    QUESTION = "question"
    LPAREN = "lparen"
    RPAREN = "rparen"
    LBRACKET = "lbracket"
    RBRACKET = "rbracket"
    LBRACE = "lbrace"
    RBRACE = "rbrace"
    LT = "lt"
    GT = "gt"
    NEWLINE = "newline"
    INDENT = "indent"
    DEDENT = "dedent"
    EOF = "eof"


class Token(FrozenModel):
    kind: TokenKind
    value: str
    span: SourceSpan


KEYWORDS = frozenset(
    {
        "ACT",
        "AS",
        "ALTERNATIVES",
        "ARGUMENTS",
        "AWAIT",
        "BASIS",
        "BEFORE",
        "BLOCK",
        "CALL",
        "COMPARE",
        "CONTEXT",
        "DOMINANCE",
        "DEFER",
        "DERIVE",
        "DISTINGUISH",
        "DO",
        "DEPTH",
        "EACH",
        "ELSE",
        "EXPECT",
        "FACTS",
        "FAIL",
        "FALSIFIERS",
        "FOR",
        "FROM",
        "FRAME",
        "GENERALIZE",
        "IF",
        "IMPORT",
        "IN",
        "INVALIDATE",
        "INPUT",
        "KIND",
        "LET",
        "LEFT",
        "LIMIT",
        "MAX",
        "MATCH",
        "MEASURE",
        "MISSING",
        "MODULE",
        "NEXT",
        "OBSERVE",
        "OPERATIONAL_WISDOM",
        "OUTPUT",
        "ON",
        "PRESERVE",
        "PROPAGATE",
        "PROGRESS",
        "PROPOSE",
        "PREDICATE",
        "RECALCULATE",
        "RELEASE",
        "RELATE",
        "REQUIRE",
        "RIGHT",
        "SCOPE",
        "SELECT",
        "STATE",
        "STEP",
        "STOP",
        "TEST",
        "TRACK",
        "TRANSACTION",
        "TARGET",
        "UNDER",
        "UPDATE",
        "USING",
        "WHEN",
        "WHILE",
        "WISDOM",
        "WORKFLOW",
    }
)
