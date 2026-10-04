"""Deterministic JSON-value evaluation for compiled SRC expressions and bindings."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import Enum
from functools import lru_cache
from typing import Any

from pydantic import BaseModel

from agents.yf_arc3_v5.logos.types import (
    FrozenMap,
    FrozenModel,
    canonical_json,
    freeze_json,
)
from agents.yf_arc3_v5.src.ast import ExpressionKind
from agents.yf_arc3_v5.src.compiler import IrExpression
from agents.yf_arc3_v5.src.symbols import TypeSpec


class ExpressionEvaluationError(ValueError):
    """An expression could not be evaluated from declared runtime values."""


class RuntimeTypeError(TypeError):
    """A JSON-compatible runtime value violates its compiled binding type."""


@lru_cache(maxsize=512)
def _canonical_model_field_names(model_type: type[FrozenModel]) -> tuple[str, ...]:
    """The schema fixes field names; never sort them for each model instance."""

    return tuple(sorted(model_type.model_fields))


# Declarative workflow frames reuse many immutable tuple bindings during one
# reflection.  Keep the cache bounded, but large enough not to churn through
# the same raster/route tuples before the next checkpoint.
_RUNTIME_TUPLE_CACHE_LIMIT = 4096
_RUNTIME_TUPLE_CACHE: dict[int, tuple[tuple[object, ...], tuple[object, ...]]] = {}
_RUNTIME_MODEL_MAP_CACHE_LIMIT = 1024
_RUNTIME_MODEL_MAP_CACHE: dict[
    int, tuple[FrozenModel, "RuntimeModelMap"]
] = {}


def _runtime_tuple(value: tuple[object, ...]) -> tuple[object, ...]:
    identity = id(value)
    cached = _RUNTIME_TUPLE_CACHE.get(identity)
    if cached is not None and cached[0] is value:
        return cached[1]
    # A tuple of exact JSON scalar leaves is already recursively immutable.
    # Preserve its identity instead of rebuilding large raster/pixel sequences
    # and visiting every scalar through the general model projection walker.
    if all(item is None or type(item) in {str, bool, int} for item in value):
        converted = value
    elif value and all(
        isinstance(item, tuple)
        and all(member is None or type(member) in {str, bool, int} for member in item)
        for item in value
    ):
        # Raster rows and compact route signatures are commonly represented as
        # tuples of scalar tuples.  They are already recursively immutable; do
        # not re-enter runtime_value once per row and once per cell.
        converted = value
    elif value and all(
        isinstance(item, list)
        and all(member is None or type(member) in {str, bool, int} for member in item)
        for item in value
    ):
        # Lists can still occur at the edge of a validated JSON payload.  The
        # only required normalization is their immutable tuple wrapper.
        converted = tuple(tuple(item) for item in value)
    else:
        converted = tuple(runtime_value(item) for item in value)
    if len(_RUNTIME_TUPLE_CACHE) >= _RUNTIME_TUPLE_CACHE_LIMIT:
        _RUNTIME_TUPLE_CACHE.pop(next(iter(_RUNTIME_TUPLE_CACHE)))
    _RUNTIME_TUPLE_CACHE[identity] = (value, converted)
    return converted


class RuntimeModelMap(FrozenMap):
    """Serialized runtime facts retaining their already-validated local model."""

    __slots__ = ("source_model",)

    @classmethod
    def from_model_items(
        cls,
        value: Mapping[str, object],
        *,
        source_model: FrozenModel,
    ) -> "RuntimeModelMap":
        frozen = object.__new__(cls)
        frozen._data = {
            key: value[key] for key in _canonical_model_field_names(type(source_model))
        }
        frozen._items = tuple(frozen._data.items())
        frozen._hash = None
        frozen._digest = None
        frozen.source_model = source_model
        return frozen


def restore_runtime_model_refs(value: FrozenMap) -> FrozenMap:
    """Restore direct typed children without walking their immutable payloads."""

    restored: dict[str, object] = {}
    for name, item in sorted(value.items()):
        if isinstance(item, RuntimeModelMap):
            restored[name] = item.source_model
        elif isinstance(item, tuple) and any(
            isinstance(member, RuntimeModelMap) for member in item
        ):
            restored[name] = tuple(
                member.source_model
                if isinstance(member, RuntimeModelMap)
                else member
                for member in item
            )
        else:
            restored[name] = item
    return FrozenMap.from_frozen_items(restored)


def runtime_value(value: object) -> Any:
    """Convert typed immutable results into recursively immutable JSON data."""

    # Raster and delta payloads are overwhelmingly primitive leaves and tuples.
    # Check those concrete built-in types before Pydantic/ABC-backed classes;
    # the former ordering performed millions of costly negative isinstance
    # checks while recursively transporting an otherwise unchanged frame.
    value_type = type(value)
    if value is None or value_type in {str, bool, int}:
        return value
    if value_type is float:
        return freeze_json(value)
    if value_type is tuple:
        return _runtime_tuple(value)
    if value_type is list:
        return tuple(runtime_value(item) for item in value)
    # FrozenMap is the dominant already-validated binding value.  The exact
    # type check avoids an ABC instance check on every expression leaf while
    # preserving the subclass path below for RuntimeModelMap and extensions.
    if value_type is FrozenMap:
        return value
    if isinstance(value, FrozenMap):
        return value
    if isinstance(value, tuple):
        return _runtime_tuple(tuple(value))
    if isinstance(value, list):
        return tuple(runtime_value(item) for item in value)
    if isinstance(value, Enum):
        return runtime_value(value.value)
    if isinstance(value, FrozenModel):
        identity = id(value)
        cached = _RUNTIME_MODEL_MAP_CACHE.get(identity)
        if cached is not None and cached[0] is value:
            return cached[1]
        converted = RuntimeModelMap.from_model_items(
            {
                name: runtime_value(getattr(value, name))
                for name in value.__class__.model_fields
            },
            source_model=value,
        )
        if len(_RUNTIME_MODEL_MAP_CACHE) >= _RUNTIME_MODEL_MAP_CACHE_LIMIT:
            _RUNTIME_MODEL_MAP_CACHE.pop(next(iter(_RUNTIME_MODEL_MAP_CACHE)))
        _RUNTIME_MODEL_MAP_CACHE[identity] = (value, converted)
        return converted
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return FrozenMap.from_frozen_items(
            {str(key): runtime_value(item) for key, item in sorted(value.items())}
        )
    return freeze_json(value)


def validate_runtime_type(value: object, type_spec: TypeSpec, description: str) -> None:
    if value is None:
        if type_spec.optional:
            return
        raise RuntimeTypeError(
            f"{description} requires {type_spec.display()}, got null"
        )
    required = type_spec.required()
    name = required.name
    valid = True
    if name == "Any":
        return
    if name == "Boolean":
        valid = isinstance(value, bool)
    elif name == "Integer":
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif name in {"String", "Ref"}:
        valid = isinstance(value, str)
    elif name in {"List", "Set"}:
        valid = isinstance(value, (list, tuple))
        sequence = value if isinstance(value, (list, tuple)) else ()
        if valid and name == "Set":
            identities = tuple(canonical_json(item) for item in sequence)
            valid = len(identities) == len(set(identities))
        if valid and required.arguments:
            for index, item in enumerate(sequence):
                validate_runtime_type(
                    item,
                    required.arguments[0],
                    f"{description}[{index}]",
                )
    elif name == "Map":
        valid = isinstance(value, Mapping)
    else:
        # Domain values are source-typed opaque references or serialized records.
        valid = isinstance(value, (str, Mapping, FrozenMap))
    if not valid:
        raise RuntimeTypeError(
            f"{description} requires {type_spec.display()}, got {type(value).__name__}"
        )


def evaluate_expression(
    expression: IrExpression,
    bindings: Mapping[str, object],
    *,
    call: Callable[[str, tuple[object, ...], FrozenMap], object] | None = None,
) -> Any:
    kind = expression.kind
    if kind is ExpressionKind.REFERENCE:
        reference = str(expression.value)
        if reference in bindings:
            return bindings[reference]
        root, separator, remainder = reference.partition(".")
        if separator and root in bindings:
            current = bindings[root]
            for member in remainder.split("."):
                if isinstance(current, Mapping) and member in current:
                    current = current[member]
                else:
                    raise ExpressionEvaluationError(
                        f"runtime reference {reference} has no member {member}"
                    )
            return current
        return reference
    if kind in {
        ExpressionKind.STRING,
        ExpressionKind.INTEGER,
        ExpressionKind.BOOLEAN,
    }:
        return runtime_value(expression.value)
    if kind is ExpressionKind.NULL:
        return None
    if kind is ExpressionKind.EMPTY_SET:
        return ()
    if kind is ExpressionKind.LIST:
        return tuple(
            evaluate_expression(item, bindings, call=call) for item in expression.items
        )
    if kind is ExpressionKind.CALL:
        if call is None:
            raise ExpressionEvaluationError(
                f"no pure function registry for expression call {expression.value}"
            )
        positional: list[object] = []
        named: dict[str, object] = {}
        for argument in expression.arguments:
            value = evaluate_expression(argument.value, bindings, call=call)
            if argument.name is None:
                positional.append(value)
            else:
                named[argument.name] = value
        return runtime_value(
            call(str(expression.value), tuple(positional), FrozenMap(named))
        )
    raise ExpressionEvaluationError(f"unsupported expression kind: {kind.value}")
