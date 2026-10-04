"""Sparse, deterministic cell-type deltas for an already measured lattice."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CellFrameDelta:
    ordinal: int
    changes: tuple[tuple[int, int, int], ...]  # cell index, old type, new type


@dataclass(frozen=True)
class CellDeltaPacket:
    positions: tuple[tuple[int, int], ...]
    types: tuple[tuple[tuple[int, ...], ...], ...]
    initial_types: tuple[int, ...]
    deltas: tuple[CellFrameDelta, ...]


def measure_cell_deltas(*, before, frames, row_phase: int, col_phase: int,
                        stride: int, size: int, max_cells: int = 256,
                        max_frames: int = 256) -> CellDeltaPacket | None:
    """Compare rows first, then classify only cells touched by changed pixels.

    Tuple equality runs in native code. No dependencies, repeated segmentation,
    role inference, retained raster frames or copies of successive cell grids.
    """
    if not before or not before[0] or not 0 < size <= stride or len(frames) > max_frames:
        return None
    height, width = len(before), len(before[0])
    if any(len(row) != width for row in before):
        return None
    positions = tuple((r, c) for r in range(row_phase, height - size + 1, stride)
                      for c in range(col_phase, width - size + 1, stride))
    if len(positions) > max_cells:
        return None
    indexes = {position: i for i, position in enumerate(positions)}
    type_indexes = {}
    types = []
    def cell_type(frame, position):
        r, c = position
        patch = tuple(tuple(row[c:c + size]) for row in frame[r:r + size])
        if patch not in type_indexes:
            type_indexes[patch] = len(types)
            types.append(patch)
        return type_indexes[patch]
    current_types = [cell_type(before, p) for p in positions]
    initial_types = tuple(current_types)
    previous = before
    deltas = []
    for ordinal, frame in enumerate(frames):
        if len(frame) != height or any(len(row) != width for row in frame):
            return None
        changed = set()
        for row_index, (old_row, new_row) in enumerate(zip(previous, frame)):
            if old_row == new_row:
                continue
            cell_row, inside_row = divmod(row_index - row_phase, stride)
            if row_index < row_phase or inside_row >= size:
                continue
            for col_index, (old, new) in enumerate(zip(old_row, new_row)):
                if old == new or col_index < col_phase:
                    continue
                cell_col, inside_col = divmod(col_index - col_phase, stride)
                if inside_col < size:
                    position = (row_phase + cell_row * stride, col_phase + cell_col * stride)
                    if position in indexes:
                        changed.add(indexes[position])
        changes = []
        for index in sorted(changed):
            new_type = cell_type(frame, positions[index])
            old_type = current_types[index]
            if new_type != old_type:
                changes.append((index, old_type, new_type))
                current_types[index] = new_type
        deltas.append(CellFrameDelta(ordinal, tuple(changes)))
        previous = frame
    return CellDeltaPacket(positions, tuple(types), initial_types, tuple(deltas))
