"""Bounded exact object assembly on an already retained logical grid."""

from __future__ import annotations

from agents.yf_arc3_v5.capabilities.contracts import (
    KnownGridCellObject,
    KnownGridObjectAssemblyInput,
    KnownGridObjectAssemblyResult,
    KnownGridShapeClass,
)
from agents.yf_arc3_v5.logos.types import stable_digest


_MAX_OBJECTS = 256


def _shape_ref(shape: tuple[tuple[int, int], ...]) -> str:
    return f"measurement.grid_shape.{stable_digest(shape)[:20]}"


def assemble_known_grid_objects(
    value: KnownGridObjectAssemblyInput,
) -> KnownGridObjectAssemblyResult:
    """Group equal-type orthogonal cells and index shapes without pair expansion."""

    rows = value.logical_rows
    columns = value.logical_columns
    type_ids = value.recount.cell_type_ids
    visited = [False] * len(type_ids)
    object_rows: list[tuple[int, tuple[int, ...], tuple[tuple[int, int], ...], str]] = []
    shape_members: dict[tuple[tuple[int, int], ...], list[str]] = {}

    for start, cell_type_id in enumerate(type_ids):
        if visited[start]:
            continue
        if len(object_rows) >= _MAX_OBJECTS:
            raise ValueError("retained-grid object count exceeds hard bound")
        pending = [start]
        visited[start] = True
        members: list[int] = []
        while pending:
            current = pending.pop()
            members.append(current)
            row, column = divmod(current, columns)
            neighbors: tuple[int, ...] = ()
            if row > 0:
                neighbors += (current - columns,)
            if row + 1 < rows:
                neighbors += (current + columns,)
            if column > 0:
                neighbors += (current - 1,)
            if column + 1 < columns:
                neighbors += (current + 1,)
            for neighbor in neighbors:
                if not visited[neighbor] and type_ids[neighbor] == cell_type_id:
                    visited[neighbor] = True
                    pending.append(neighbor)

        ordered_members = tuple(sorted(members))
        positions = tuple(divmod(index, columns) for index in ordered_members)
        top = min(row for row, _ in positions)
        left = min(column for _, column in positions)
        normalized_shape = tuple(
            sorted((row - top, column - left) for row, column in positions)
        )
        object_ref = (
            f"measurement.grid_object."
            f"{stable_digest((value.recount.grid_ref, cell_type_id, ordered_members))[:20]}"
        )
        object_rows.append(
            (cell_type_id, ordered_members, normalized_shape, object_ref)
        )
        shape_members.setdefault(normalized_shape, []).append(object_ref)

    shape_refs = {shape: _shape_ref(shape) for shape in shape_members}
    shape_classes: list[KnownGridShapeClass] = []
    for shape in sorted(shape_members):
        maximum_row = max(row for row, _ in shape)
        maximum_column = max(column for _, column in shape)
        horizontal = tuple(
            sorted((row, maximum_column - column) for row, column in shape)
        )
        vertical = tuple(
            sorted((maximum_row - row, column) for row, column in shape)
        )
        shape_classes.append(
            KnownGridShapeClass(
                shape_class_ref=shape_refs[shape],
                normalized_cells=shape,
                object_refs=tuple(shape_members[shape]),
                horizontal_reflection_class_ref=shape_refs.get(horizontal),
                vertical_reflection_class_ref=shape_refs.get(vertical),
            )
        )

    objects = tuple(
        KnownGridCellObject(
            object_ref=object_ref,
            cell_type_id=cell_type_id,
            member_cell_indexes=members,
            top=min(index // columns for index in members),
            left=min(index % columns for index in members),
            height=(
                max(index // columns for index in members)
                - min(index // columns for index in members)
                + 1
            ),
            width=(
                max(index % columns for index in members)
                - min(index % columns for index in members)
                + 1
            ),
            shape_class_ref=_shape_ref(shape),
            visibility="visible",
        )
        for cell_type_id, members, shape, object_ref in object_rows
    )
    singleton_count = sum(len(item.member_cell_indexes) == 1 for item in objects)
    return KnownGridObjectAssemblyResult(
        grid_ref=value.recount.grid_ref,
        logical_rows=rows,
        logical_columns=columns,
        objects=objects,
        shape_classes=tuple(shape_classes),
        singleton_object_count=singleton_count,
        multicell_object_count=len(objects) - singleton_count,
    )
