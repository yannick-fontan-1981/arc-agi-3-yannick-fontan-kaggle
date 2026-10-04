"""Structured SRC compiler errors with compact source locations."""

from __future__ import annotations

from agents.yf_arc3_v5.src.tokens import SourceSpan


class SrcError(ValueError):
    """Base error for rejected SRC source."""

    def __init__(self, message: str, span: SourceSpan | None = None) -> None:
        self.message = message
        self.span = span
        location = "" if span is None else f" at {span.compact()}"
        super().__init__(f"{message}{location}")


class SrcSyntaxError(SrcError):
    """The source cannot be parsed as SRC."""


class SrcValidationError(SrcError):
    """The parsed program violates a static SRC contract."""
