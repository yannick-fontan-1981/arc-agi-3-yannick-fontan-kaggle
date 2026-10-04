"""Bounded, semantic-free groups of pixels changed between two frames."""

from __future__ import annotations

from dataclasses import dataclass

from agents.yf_arc3_v5.capabilities.cell_recount import (
    measure_grid_footprint_transport,
)
from agents.yf_arc3_v5.capabilities.contracts import (
    BoundingBox,
    FrameGrid,
    PeriodicCellGridResult,
)


@dataclass(frozen=True, slots=True)
class ChangedPixelGroup:
    pixel_count: int
    bbox: BoundingBox


@dataclass(frozen=True, slots=True)
class EqualShapeChangePair:
    first: ChangedPixelGroup
    second: ChangedPixelGroup
    delta_row: int
    delta_col: int


@dataclass(frozen=True, slots=True)
class PixelDeltaGroups:
    changed_pixel_count: int
    changed_bbox: BoundingBox | None
    group_count: int
    groups: tuple[ChangedPixelGroup, ...]
    equal_shape_pairs: tuple[EqualShapeChangePair, ...]
    enumeration_truncated: bool


@dataclass(frozen=True, slots=True)
class PeriodicChangeWitness:
    """Exact cell transport facts; no cursor or grid commitment."""

    grid_ref: str
    from_cell: int
    to_cell: int
    changed_pixel_count: int
    from_cell_singleton_multivalue: bool
    cell_pattern_equal: bool
    cell_bbox_equals_change_bbox: bool


@dataclass(frozen=True, slots=True)
class PeriodicChangeWitnesses:
    witnesses: tuple[PeriodicChangeWitness, ...]
    enumeration_truncated: bool


def measure_periodic_change_witnesses(
    before: FrameGrid,
    after: FrameGrid,
    delta: PixelDeltaGroups,
    grids: PeriodicCellGridResult,
    *,
    interface_delta: tuple[int, int],
    max_witnesses: int = 3,
) -> PeriodicChangeWitnesses:
    """Compare source-supplied interface direction with exact cell transport."""

    if before.height != after.height or before.width != after.width:
        raise ValueError("periodic change requires equal frame dimensions")
    if max_witnesses < 1:
        raise ValueError("periodic change witness bound must be positive")
    if delta.enumeration_truncated or grids.enumeration_truncated or len(grids.candidates) > 3:
        return PeriodicChangeWitnesses((), True)
    if abs(interface_delta[0]) + abs(interface_delta[1]) != 1:
        return PeriodicChangeWitnesses((), False)

    witnesses: list[PeriodicChangeWitness] = []
    for pair in delta.equal_shape_pairs:
        pair_axis = (
            (pair.delta_row > 0) - (pair.delta_row < 0),
            (pair.delta_col > 0) - (pair.delta_col < 0),
        )
        if pair_axis == interface_delta:
            old_group, new_group = pair.first, pair.second
        elif (-pair_axis[0], -pair_axis[1]) == interface_delta:
            old_group, new_group = pair.second, pair.first
        else:
            continue
        for grid in grids.candidates:
            measurement = measure_grid_footprint_transport(
                grid, ((old_group.bbox, new_group.bbox),)
            )
            if not (measurement.same_moving_cell and measurement.lattice_displacement):
                continue
            from_cell = measurement.before_cells[0]
            to_cell = measurement.after_cells[0]
            if from_cell is None or to_cell is None:
                raise RuntimeError("contained grid transport lacks cell indexes")
            from_row, from_col = divmod(from_cell, grid.logical_columns)
            to_row, to_col = divmod(to_cell, grid.logical_columns)
            before_top = grid.row_offset + from_row * grid.row_pitch
            before_left = grid.col_offset + from_col * grid.col_pitch
            after_top = grid.row_offset + to_row * grid.row_pitch
            after_left = grid.col_offset + to_col * grid.col_pitch
            before_pattern = tuple(
                row[before_left:before_left + grid.cell_width]
                for row in before.rows[before_top:before_top + grid.cell_height]
            )
            after_pattern = tuple(
                row[after_left:after_left + grid.cell_width]
                for row in after.rows[after_top:after_top + grid.cell_height]
            )
            old_cell = grid.cells[from_cell]
            new_cell = grid.cells[to_cell]
            witnesses.append(PeriodicChangeWitness(
                grid_ref=grid.candidate_ref,
                from_cell=from_cell,
                to_cell=to_cell,
                changed_pixel_count=old_group.pixel_count,
                from_cell_singleton_multivalue=(
                    old_cell.pattern_occurrence_count == 1
                    and old_cell.palette_value_count > 1
                ),
                cell_pattern_equal=before_pattern == after_pattern,
                cell_bbox_equals_change_bbox=(
                    old_cell.bbox == old_group.bbox
                    and new_cell.bbox == new_group.bbox
                ),
            ))
            if len(witnesses) > max_witnesses:
                return PeriodicChangeWitnesses((), True)
    return PeriodicChangeWitnesses(tuple(witnesses), False)


def measure_pixel_delta_groups(
    before: FrameGrid,
    after: FrameGrid,
    *,
    max_groups: int = 32,
    max_pairs: int = 3,
) -> PixelDeltaGroups:
    """Measure exact 4-neighbor change masks; never infer old/new or roles.

    Pair geometry is retained only when the complete bounded enumeration fits.
    Shape masks and changed coordinates are transient and never enter the result.
    """

    if before.height != after.height or before.width != after.width:
        raise ValueError("pixel delta requires equal frame dimensions")
    changed = {
        (row, col)
        for row, (old_row, new_row) in enumerate(zip(before.rows, after.rows))
        for col, (old, new) in enumerate(zip(old_row, new_row))
        if old != new
    }
    return summarize_changed_positions(
        changed,
        frame_height=before.height,
        frame_width=before.width,
        max_groups=max_groups,
        max_pairs=max_pairs,
    )


def summarize_changed_positions(
    changed_positions: set[tuple[int, int]] | frozenset[tuple[int, int]],
    *,
    frame_height: int,
    frame_width: int,
    max_groups: int = 32,
    max_pairs: int = 3,
) -> PixelDeltaGroups:
    """Reuse a caller's already measured pixel positions without rescanning."""

    if frame_height < 1 or frame_width < 1:
        raise ValueError("pixel delta frame dimensions must be positive")
    if max_groups < 1 or max_pairs < 1:
        raise ValueError("pixel delta bounds must be positive")
    changed = set(changed_positions)
    if any(
        row < 0 or row >= frame_height or col < 0 or col >= frame_width
        for row, col in changed
    ):
        raise ValueError("changed pixel position is outside the frame")
    if not changed:
        return PixelDeltaGroups(0, None, 0, (), (), False)

    changed_bbox = BoundingBox(
        top=min(row for row, _ in changed),
        left=min(col for _, col in changed),
        bottom=max(row for row, _ in changed),
        right=max(col for _, col in changed),
    )
    remaining = set(changed)
    bounded_groups: list[tuple[ChangedPixelGroup, frozenset[tuple[int, int]]]] = []
    group_count = 0
    for seed in sorted(changed):
        if seed not in remaining:
            continue
        remaining.remove(seed)
        frontier = [seed]
        positions = [seed]
        while frontier:
            row, col = frontier.pop()
            for adjacent in (
                (row - 1, col), (row + 1, col),
                (row, col - 1), (row, col + 1),
            ):
                if adjacent in remaining:
                    remaining.remove(adjacent)
                    frontier.append(adjacent)
                    positions.append(adjacent)
        group_count += 1
        if group_count > max_groups:
            continue
        top = min(row for row, _ in positions)
        left = min(col for _, col in positions)
        group = ChangedPixelGroup(
            pixel_count=len(positions),
            bbox=BoundingBox(
                top=top,
                left=left,
                bottom=max(row for row, _ in positions),
                right=max(col for _, col in positions),
            ),
        )
        shape = frozenset(
            (row - top, col - left) for row, col in positions
        )
        bounded_groups.append((group, shape))

    if group_count > max_groups:
        return PixelDeltaGroups(
            len(changed), changed_bbox, group_count, (), (), True
        )

    pairs: list[EqualShapeChangePair] = []
    for first_index, (first, first_shape) in enumerate(bounded_groups):
        for second, second_shape in bounded_groups[first_index + 1:]:
            if first_shape != second_shape:
                continue
            pairs.append(EqualShapeChangePair(
                first=first,
                second=second,
                delta_row=second.bbox.top - first.bbox.top,
                delta_col=second.bbox.left - first.bbox.left,
            ))
            if len(pairs) > max_pairs:
                return PixelDeltaGroups(
                    len(changed), changed_bbox, group_count,
                    tuple(group for group, _ in bounded_groups), (), True,
                )
    return PixelDeltaGroups(
        len(changed), changed_bbox, group_count,
        tuple(group for group, _ in bounded_groups), tuple(pairs), False,
    )
