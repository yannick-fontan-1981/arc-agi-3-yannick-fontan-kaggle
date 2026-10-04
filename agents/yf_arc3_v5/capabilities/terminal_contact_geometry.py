"""Bounded cell-level geometry around an animated terminal transition.

This measures the last visible mover and its five-cell neighborhood. It does
not name a hazard, infer a cause of failure, or decide on restoration.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

from agents.yf_arc3_v5.capabilities.contracts import FrameGrid, PeriodicCellGridInput
from agents.yf_arc3_v5.capabilities.frame import (
    _cell_pattern_ref,
    enumerate_periodic_cell_grids,
)
from agents.yf_arc3_v5.logos.types import stable_digest


@dataclass(frozen=True)
class TerminalNeighborhoodCell:
    relation: str
    row: int
    col: int
    pattern_ref: str
    palette_values: tuple[int, ...]


@dataclass(frozen=True)
class TerminalContactGeometry:
    approach_row: int
    approach_col: int
    moving_start_cell_row: int
    moving_start_cell_col: int
    moving_start_pattern_ref: str
    moving_start_palette_values: tuple[int, ...]
    last_visible_frame_index: int
    last_visible_cell_row: int
    last_visible_cell_col: int
    neighborhood: tuple[TerminalNeighborhoodCell, ...]
    adjacent_cell_row: int
    adjacent_cell_col: int
    adjacent_pattern_ref: str
    adjacent_palette_values: tuple[int, ...]
    adjacent_face: str
    adjacent_face_values: tuple[int, ...]
    adjacent_cell_unchanged: bool
    local_extent_collapse: bool
    local_transition_count: int


def measure_cell_configuration_hash(frame: Sequence[Sequence[int]]) -> str:
    """Hash cell types on the leading measured grid, excluding unrelated HUD pixels."""

    grids = enumerate_periodic_cell_grids(
        PeriodicCellGridInput(frame=FrameGrid(rows=frame))
    )
    if not grids.candidates:
        return ""
    grid = grids.candidates[0]
    return stable_digest(
        (
            grid.row_offset,
            grid.col_offset,
            grid.row_pitch,
            grid.col_pitch,
            grid.logical_rows,
            grid.logical_columns,
            tuple(cell.pattern_ref for cell in grid.cells),
        )
    )[:16]


def measure_exact_observed_fatal_replay(
    *,
    claims: Sequence[Any],
    world: Any,
    action_ref: str,
    action_data: Mapping[str, object],
) -> bool:
    """Match an observed fatal action in an identical cell configuration.

    This is an exact safety measurement only. The source decides whether it
    blocks action release and how to reopen the remaining frontier.
    """

    metadata = world.metadata
    frame_hash = str(metadata.get("frame_hash") or "")
    game_ref = str(metadata.get("game_id") or "")
    if not game_ref:
        return False
    bounded_data = {
        key: int(action_data[key])
        for key in ("x", "y")
        if type(action_data.get(key)) is int
    }
    cell_configuration_hash: str | None = None
    for claim in claims:
        if claim.predicate != "hostile_contact_relation":
            continue
        attributes = claim.attributes
        if (
            getattr(claim.epistemic_status, "value", claim.epistemic_status)
            != "established"
            or attributes.get("identity_tier") != "SPRITE"
            or attributes.get("official_outcome") != "GAME_OVER"
            or attributes.get("game_ref") != game_ref
            or attributes.get("fatal_action_ref") != action_ref
        ):
            continue
        stored_data = attributes.get("fatal_action_data") or {}
        if not isinstance(stored_data, Mapping) or dict(stored_data) != bounded_data:
            continue
        if frame_hash and attributes.get("fatal_before_frame_hash") == frame_hash:
            return True
        prior_grid_hash = str(attributes.get("fatal_before_cell_configuration_hash") or "")
        if prior_grid_hash:
            if cell_configuration_hash is None:
                cell_configuration_hash = measure_cell_configuration_hash(world.frame)
            if cell_configuration_hash and prior_grid_hash == cell_configuration_hash:
                return True
    return False


def _field(value: object, key: str) -> Any:
    return value.get(key) if isinstance(value, Mapping) else getattr(value, key)


def _changed_bbox(
    before: Sequence[Sequence[int]], after: Sequence[Sequence[int]]
) -> tuple[int, int, int, int, int] | None:
    changed = [
        (row, col)
        for row, (old_row, new_row) in enumerate(zip(before, after))
        for col, (old, new) in enumerate(zip(old_row, new_row))
        if old != new
    ]
    if not changed:
        return None
    return (
        min(row for row, _ in changed),
        min(col for _, col in changed),
        max(row for row, _ in changed),
        max(col for _, col in changed),
        len(changed),
    )


def _cell_pattern(
    frame: Sequence[Sequence[int]], cell: object
) -> tuple[tuple[int, ...], ...]:
    bbox = _field(cell, "bbox")
    top, left, bottom, right = (
        int(_field(bbox, name)) for name in ("top", "left", "bottom", "right")
    )
    return tuple(tuple(frame[row][left : right + 1]) for row in range(top, bottom + 1))


def measure_edge_indicator_exhaustion(
    before: Sequence[Sequence[int]],
    visible: Sequence[Sequence[int]],
    component: object,
    *,
    decrement_unit: int,
    border_padding: int,
) -> bool:
    """Measure disappearance of the last established edge-gauge quantum."""

    if decrement_unit <= 0 or border_padding < 0 or not before or not before[0]:
        return False
    height, width = len(before), len(before[0])
    if len(visible) != height or any(len(row) != width for row in visible):
        return False
    bbox = _field(component, "bbox")
    if bbox is None:
        return False
    top, left, bottom, right = (
        int(_field(bbox, name)) for name in ("top", "left", "bottom", "right")
    )
    if not (0 <= top <= bottom < height and 0 <= left <= right < width):
        return False
    if min(top, left, height - bottom - 1, width - right - 1) > border_padding:
        return False
    value = _field(component, "value")
    if not isinstance(value, int):
        return False
    old_count = sum(
        before[row][col] == value
        for row in range(top, bottom + 1)
        for col in range(left, right + 1)
    )
    new_count = sum(
        visible[row][col] == value
        for row in range(top, bottom + 1)
        for col in range(left, right + 1)
    )
    return old_count == decrement_unit and new_count == 0


def measure_terminal_contact_geometry(
    frames: Sequence[Sequence[Sequence[int]]],
    grid: object,
) -> TerminalContactGeometry | None:
    """Find one local approach ending in collapse against an invariant cell.

    Global animation/camera changes end the local interval. Ambiguous diagonal
    motion, multiple separated movers, or an absent adjacent cell yield no fact.
    All thresholds are structural multiples of the declared logical cell size.
    """

    if not 4 <= len(frames) <= 64:
        return None
    height, width = len(frames[0]), len(frames[0][0])
    if not height or not width or height > 128 or width > 128:
        return None
    if any(
        len(frame) != height or any(len(row) != width for row in frame)
        for frame in frames
    ):
        return None
    cell_height = int(_field(grid, "cell_height"))
    cell_width = int(_field(grid, "cell_width"))
    row_pitch = int(_field(grid, "row_pitch"))
    col_pitch = int(_field(grid, "col_pitch"))
    row_offset = int(_field(grid, "row_offset"))
    col_offset = int(_field(grid, "col_offset"))
    if min(cell_height, cell_width, row_pitch, col_pitch) <= 0:
        return None

    local: list[tuple[int, int, int, int, int, int]] = []
    for index in range(1, len(frames)):
        bbox = _changed_bbox(frames[index - 1], frames[index])
        if bbox is None:
            continue
        top, left, bottom, right, count = bbox
        if bottom - top + 1 > 2 * row_pitch or right - left + 1 > 2 * col_pitch:
            break
        local.append((index, top, left, bottom, right, count))
    if len(local) < 4:
        return None

    first, last = local[0], local[-1]
    delta_row_twice = (last[1] + last[3]) - (first[1] + first[3])
    delta_col_twice = (last[2] + last[4]) - (first[2] + first[4])
    if abs(delta_row_twice) >= 2 * row_pitch and abs(delta_col_twice) < col_pitch:
        approach_row, approach_col = (1 if delta_row_twice > 0 else -1), 0
    elif abs(delta_col_twice) >= 2 * col_pitch and abs(delta_row_twice) < row_pitch:
        approach_row, approach_col = 0, (1 if delta_col_twice > 0 else -1)
    else:
        return None

    first_motion = next(
        (
            item
            for item in local
            if item[3] - item[1] + 1 > cell_height
            or item[4] - item[2] + 1 > cell_width
        ),
        None,
    )
    if first_motion is None:
        return None
    start_row_pixel = (
        first_motion[3]
        if approach_row < 0
        else first_motion[1]
        if approach_row > 0
        else (first_motion[1] + first_motion[3]) // 2
    )
    start_col_pixel = (
        first_motion[4]
        if approach_col < 0
        else first_motion[2]
        if approach_col > 0
        else (first_motion[2] + first_motion[4]) // 2
    )
    start_row = (start_row_pixel - row_offset) // row_pitch
    start_col = (start_col_pixel - col_offset) // col_pitch
    start_cell = next(
        (
            item
            for item in _field(grid, "cells")
            if int(_field(item, "row")) == start_row
            and int(_field(item, "col")) == start_col
        ),
        None,
    )
    if start_cell is None:
        return None

    # The last local change is a disappearing/retracting mover only when its
    # spatial extent is smaller than the immediately preceding full-cell sweep.
    prior_extent = max(
        (item[3] - item[1] + 1) * (item[4] - item[2] + 1)
        for item in local[-4:-1]
    )
    last_extent = (last[3] - last[1] + 1) * (last[4] - last[2] + 1)
    collapsed = last_extent < prior_extent
    if not collapsed:
        return None

    # Search backward through the local animation: the published GAME_OVER
    # frame may have erased the mover. A shrinking cursor can retain only one
    # of its distinctive colors, so an exact full-cell match alone is too late.
    source_pattern = _cell_pattern(frames[0], start_cell)
    terminal_row = (last[1] + last[3]) // 2
    terminal_col = (last[2] + last[4]) // 2
    cells_by_position = {
        (int(_field(item, "row")), int(_field(item, "col"))): item
        for item in _field(grid, "cells")
    }
    source_values = {value for line in source_pattern for value in line}
    other_values = {
        int(value)
        for position, item in sorted(cells_by_position.items())
        if position != (start_row, start_col)
        for value in _field(item, "palette_values")
    }
    distinctive_values = source_values.difference(other_values)
    last_visible: tuple[int, int, int] | None = None
    for frame_index in range(last[0], -1, -1):
        matches = []
        for (row, col), item in sorted(cells_by_position.items()):
            pattern = _cell_pattern(frames[frame_index], item)
            marker_count = sum(
                value in distinctive_values for line in pattern for value in line
            )
            if not marker_count and pattern != source_pattern:
                continue
            center_row = row_offset + row * row_pitch + cell_height // 2
            center_col = col_offset + col * col_pitch + cell_width // 2
            distance = abs(center_row - terminal_row) + abs(center_col - terminal_col)
            if distance <= 2 * (row_pitch + col_pitch):
                matches.append((distance, -marker_count, row, col))
        if matches:
            matches.sort()
            if len(matches) > 1 and matches[0][:2] == matches[1][:2]:
                return None
            last_visible = (frame_index, matches[0][2], matches[0][3])
            break
    if last_visible is None:
        return None
    visible_index, mover_row, mover_col = last_visible
    neighborhood = []
    for relation, offset_row, offset_col in (
        ("center", 0, 0),
        ("up", -1, 0),
        ("right", 0, 1),
        ("down", 1, 0),
        ("left", 0, -1),
    ):
        row, col = mover_row + offset_row, mover_col + offset_col
        item = cells_by_position.get((row, col))
        if item is None:
            continue
        pattern = _cell_pattern(frames[visible_index], item)
        pattern_ref = (
            str(_field(item, "pattern_ref"))
            if pattern == _cell_pattern(frames[0], item)
            else _cell_pattern_ref(pattern)
        )
        neighborhood.append(
            TerminalNeighborhoodCell(
                relation=relation,
                row=row,
                col=col,
                pattern_ref=pattern_ref,
                palette_values=tuple(sorted({value for line in pattern for value in line})),
            )
        )
    if not neighborhood or neighborhood[0].relation != "center":
        return None
    adjacent_row = mover_row + approach_row
    adjacent_col = mover_col + approach_col
    cell = cells_by_position.get((adjacent_row, adjacent_col))
    if cell is None:
        return None
    bbox = _field(cell, "bbox")
    top, left, bottom, right = (
        int(_field(bbox, name)) for name in ("top", "left", "bottom", "right")
    )
    first_frame, last_local_frame = frames[0], frames[last[0]]
    unchanged = all(
        first_frame[row][col] == last_local_frame[row][col]
        for row in range(top, bottom + 1)
        for col in range(left, right + 1)
    )
    if not unchanged:
        return None
    # A border/gap may occupy the outermost raster row. Keep the two nearest
    # rows or columns as a compact face profile, never the whole pixel history.
    if approach_row < 0:
        face = "bottom"
        face_values = [
            first_frame[row][col]
            for row in range(max(top, bottom - 1), bottom + 1)
            for col in range(left, right + 1)
        ]
    elif approach_row > 0:
        face = "top"
        face_values = [
            first_frame[row][col]
            for row in range(top, min(bottom, top + 1) + 1)
            for col in range(left, right + 1)
        ]
    elif approach_col < 0:
        face = "right"
        face_values = [
            first_frame[row][col]
            for row in range(top, bottom + 1)
            for col in range(max(left, right - 1), right + 1)
        ]
    else:
        face = "left"
        face_values = [
            first_frame[row][col]
            for row in range(top, bottom + 1)
            for col in range(left, min(right, left + 1) + 1)
        ]
    return TerminalContactGeometry(
        approach_row=approach_row,
        approach_col=approach_col,
        moving_start_cell_row=start_row,
        moving_start_cell_col=start_col,
        moving_start_pattern_ref=str(_field(start_cell, "pattern_ref")),
        moving_start_palette_values=tuple(
            int(v) for v in _field(start_cell, "palette_values")
        ),
        last_visible_frame_index=visible_index,
        last_visible_cell_row=mover_row,
        last_visible_cell_col=mover_col,
        neighborhood=tuple(neighborhood),
        adjacent_cell_row=adjacent_row,
        adjacent_cell_col=adjacent_col,
        adjacent_pattern_ref=str(_field(cell, "pattern_ref")),
        adjacent_palette_values=tuple(int(v) for v in _field(cell, "palette_values")),
        adjacent_face=face,
        adjacent_face_values=tuple(sorted({int(v) for v in face_values})),
        adjacent_cell_unchanged=True,
        local_extent_collapse=True,
        local_transition_count=len(local),
    )
