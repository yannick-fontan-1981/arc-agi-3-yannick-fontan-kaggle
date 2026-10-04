"""Optional local observation of authority calls; never selects or changes results."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any


AuthorityObserver = Callable[[Any, Any, Any, str, str], None]
ExpressionObserver = Callable[[Any, Any, str, Any, Any], Callable[[Any, tuple, str | None], None]]
_observer: ContextVar[AuthorityObserver | None] = ContextVar(
    "v5_authority_observer", default=None
)
_expression_observer: ContextVar[ExpressionObserver | None] = ContextVar(
    "v5_expression_observer", default=None
)


@contextmanager
def capture_authority_calls(observer: AuthorityObserver) -> Iterator[None]:
    """Capture in this execution context only, including nested controller calls."""
    token = _observer.set(observer)
    try:
        yield
    finally:
        _observer.reset(token)


def authority_observer() -> AuthorityObserver | None:
    return _observer.get()


@contextmanager
def capture_expression_evaluations(observer: ExpressionObserver) -> Iterator[None]:
    """Observe exact pre-evaluation bindings without changing SRC execution."""
    token = _expression_observer.set(observer)
    try:
        yield
    finally:
        _expression_observer.reset(token)


def expression_observer() -> ExpressionObserver | None:
    return _expression_observer.get()
