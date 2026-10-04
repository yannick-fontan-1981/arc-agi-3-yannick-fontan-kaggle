"""Bounded raster measurements of repeated parallel separators and square cells.

No palette role, game dimensions, goal, action selection or mechanism inference.
Unknown/occluded cells remain None. A field is an exact geometric description,
not a claim that the raster is interactive or subject to gravity.
"""

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json

from agents.yf_arc3_v5.capabilities.components import detect_components
from agents.yf_arc3_v5.capabilities.contracts import ComponentExtractionInput, FrameGrid

MAX_AREA = 4096
MAX_FIELDS = 16
MAX_CELLS = 256
MAX_PACKET_FRAMES = 64


@dataclass(frozen=True)
class ParallelCellField:
    top: int
    left: int
    pitch: int
    extent: int
    cells: tuple[tuple[int | None, ...], ...]

    @property
    def height(self) -> int:
        return len(self.cells)

    @property
    def width(self) -> int:
        return len(self.cells[0])


@dataclass(frozen=True)
class CellLine:
    value: int
    positions: tuple[tuple[int, int], ...]
    step: tuple[int, int]
    adjacent_positions: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class AxialPacketMeasure:
    action_ref: str
    field_top: int
    field_left: int
    column: int
    value: int
    prior_value: int | None
    distinct_row_positions: tuple[int, ...]
    final_cell_row: int
    retained: bool
    restored: bool
    other_final_changed_pixels: int
    other_peak_changed_pixels: int
    before_field_hash: str
    after_field_hash: str
    score_delta: int


def _field_hash(frame: FrameGrid, field: ParallelCellField) -> str:
    rows = tuple(tuple(row[field.left:field.left + field.width * field.pitch - (field.pitch - field.extent)])
                 for row in frame.rows[field.top:field.top + field.height * field.pitch - (field.pitch - field.extent)])
    return sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


def measure_axial_packet(
    before: FrameGrid, frames: tuple[FrameGrid, ...], action_ref: str,
    point: tuple[int, int] | None, score_delta: int = 0,
) -> tuple[AxialPacketMeasure, ...]:
    """Exact same-column square displacement and final raster difference.

    Intermediate objects and flashing pixels are counted, never interpreted.
    Empty packets, oversized inputs or multiple fields under the point fail closed.
    """
    if point is None or not frames or len(frames) > MAX_PACKET_FRAMES:
        return ()
    if any((f.height, f.width) != (before.height, before.width) for f in frames):
        return ()
    fields = tuple(f for f in measure_parallel_cell_fields(before)
                   if f.top <= point[0] < f.top + f.height * f.pitch
                   and f.left <= point[1] < f.left + f.width * f.pitch)
    if len(fields) != 1:
        return ()
    field, = fields
    column = (point[1] - field.left) // field.pitch
    left = field.left + column * field.pitch
    paths = defaultdict(list)
    simultaneous_values = set()
    for frame in frames:
        positions_by_value = defaultdict(list)
        components = detect_components(ComponentExtractionInput(frame=frame, connectivity=4)).components
        for component in components:
            box = component.bbox
            if (box.left != left or box.height != field.extent or box.width != field.extent
                    or component.area != field.extent ** 2
                    or box.top > field.top + (field.height - 1) * field.pitch):
                continue
            value = frame.rows[box.top][box.left]
            if all(before.rows[y][x] == value for y, x in component.pixels):
                continue
            positions_by_value[value].append(box.top)
        for value, positions in sorted(positions_by_value.items()):
            if len(positions) != 1:
                simultaneous_values.add(value)
                continue
            top, = positions
            if not paths[value] or paths[value][-1] != top:
                paths[value].append(top)
    result = []
    for value, tops in sorted(paths.items()):
        if value in simultaneous_values or len(tops) < 2 or any(b <= a for a, b in zip(tops, tops[1:])):
            continue
        row, remainder = divmod(tops[-1] - field.top, field.pitch)
        if remainder or not 0 <= row < field.height:
            continue
        body = {(y, x) for y in range(tops[-1], tops[-1] + field.extent)
                for x in range(left, left + field.extent)}
        swept = {(y, x) for top in tops for y in range(top, top + field.extent)
                 for x in range(left, left + field.extent)}
        retained = score_delta == 0 and all(frames[-1].rows[y][x] == value for y, x in body)
        restored = score_delta == 0 and all(frames[-1].rows[y][x] == before.rows[y][x] for y, x in body)
        # Same local support only: unrelated concurrent scene changes are not
        # silently associated with this object.
        others = tuple((y, x)
                       for y in range(max(0, field.top - 1), field.top + field.height * field.pitch - (field.pitch - field.extent))
                       for x in range(max(0, field.left - 1), field.left + field.width * field.pitch - (field.pitch - field.extent))
                       if (y, x) not in swept)
        counts = [sum(frame.rows[y][x] != before.rows[y][x] for y, x in others) for frame in frames]
        result.append(AxialPacketMeasure(action_ref, field.top, field.left, column, value,
                                        field.cells[row][column], tuple(tops), row,
                                        retained, restored, counts[-1], max(counts),
                                        _field_hash(before, field), _field_hash(frames[-1], field), score_delta))
    return tuple(result)


def measure_parallel_cell_fields(frame: FrameGrid) -> tuple[ParallelCellField, ...]:
    """Measure maximal equal-pitch separator runs; fail closed at hard bounds.

    The square-cell description requires exact isotropic pitch and uniform
    cell interiors. A nonuniform interior is recorded as unknown, never voted
    into a palette value. Whole-frame periodic aliases are not materialized.
    """
    if frame.height * frame.width > MAX_AREA:
        return ()
    groups = defaultdict(list)
    components = detect_components(ComponentExtractionInput(frame=frame, connectivity=4)).components
    for component in components:
        box = component.bbox
        if box.height > box.width and component.area == box.height * box.width:
            groups[(box.top, box.bottom, box.width, frame.rows[box.top][box.left])].append(box.left)
    fields = []
    for (top, bottom, separator_width, _value), starts in sorted(groups.items()):
        starts.sort()
        index = 0
        while index + 2 < len(starts):
            pitch = starts[index + 1] - starts[index]
            end = index + 1
            while end + 1 < len(starts) and starts[end + 1] - starts[end] == pitch:
                end += 1
            extent = pitch - separator_width
            total_height = bottom - top + 1
            left = starts[index] - extent
            columns = end - index + 2
            rows, remainder = divmod(total_height + separator_width, pitch)
            if (end - index >= 2 and extent > 0 and remainder == 0
                    and rows >= 2 and rows * columns <= MAX_CELLS
                    and left >= 0 and left + columns * pitch - separator_width <= frame.width):
                cells = []
                for row in range(rows):
                    values = []
                    for col in range(columns):
                        pixels = {frame.rows[y][x]
                                  for y in range(top + row * pitch, top + row * pitch + extent)
                                  for x in range(left + col * pitch, left + col * pitch + extent)}
                        values.append(next(iter(pixels)) if len(pixels) == 1 else None)
                    cells.append(tuple(values))
                # The second axis must also be visibly separated. Parallel bars
                # alone describe stripes, not a two-dimensional cell field.
                horizontal_gaps = {
                    frame.rows[y][x]
                    for row in range(rows - 1)
                    for col in range(columns)
                    for y in range(top + row * pitch + extent, top + (row + 1) * pitch)
                    for x in range(left + col * pitch, left + col * pitch + extent)
                }
                interior_values = {v for row in cells for v in row if v is not None}
                if len(horizontal_gaps) != 1 or horizontal_gaps & interior_values:
                    index = end
                    continue
                fields.append(ParallelCellField(top, left, pitch, extent, tuple(cells)))
                if len(fields) > MAX_FIELDS:
                    return ()
            index = end if end > index + 1 else index + 1
    return tuple(fields)


def measure_cell_lines(field: ParallelCellField) -> tuple[CellLine, ...]:
    """Maximal homogeneous contiguous runs, without assigning a desired length.

    Each undirected axis is traversed once. Adjacent positions are geometric
    endpoints only: no assertion that they are vacant or reachable.
    """
    result = []
    for row in range(field.height):
        for col in range(field.width):
            value = field.cells[row][col]
            if value is None:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                previous = row - dr, col - dc
                if (0 <= previous[0] < field.height and 0 <= previous[1] < field.width
                        and field.cells[previous[0]][previous[1]] == value):
                    continue
                positions = []
                r, c = row, col
                while 0 <= r < field.height and 0 <= c < field.width and field.cells[r][c] == value:
                    positions.append((r, c))
                    r, c = r + dr, c + dc
                if len(positions) >= 2:
                    adjacent = tuple((y, x) for y, x in (previous, (r, c))
                                     if 0 <= y < field.height and 0 <= x < field.width)
                    result.append(CellLine(value, tuple(positions), (dr, dc), adjacent))
    return tuple(result)


def measure_parallel_probe_context(value):
    frame = value.current_observed_frame or value.frame
    fields = measure_parallel_cell_fields(frame)
    packets = value.axial_packet_measurements
    return {
        "parallel_field_count": len(fields),
        "parallel_field_present": bool(fields),
        "parallel_packet_count": len(packets),
        "parallel_moving_value_count": len({p.value for p in packets}),
        "parallel_prior_value_count": len({p.prior_value for p in packets if p.prior_value is not None}),
    }, tuple((field, measure_cell_lines(field)) for field in fields)


def measure_parallel_candidate(value, descriptions, candidate):
    fields = tuple((f, lines) for f, lines in descriptions
                   if candidate.point is not None
                   and f.top <= candidate.point[0] < f.top + f.height * f.pitch - (f.pitch - f.extent)
                   and f.left <= candidate.point[1] < f.left + f.width * f.pitch - (f.pitch - f.extent))
    facts = {"parallel_point_field_count": len(fields)}
    if len(fields) != 1:
        return facts
    field, lines = fields[0]
    row, row_offset = divmod(candidate.point[0] - field.top, field.pitch)
    column, column_offset = divmod(candidate.point[1] - field.left, field.pitch)
    facts["parallel_point_uniform_cell"] = (row_offset < field.extent and column_offset < field.extent
                                           and field.cells[row][column] is not None)
    packets = value.axial_packet_measurements
    current_hash = _field_hash(value.current_observed_frame or value.frame, field)
    local = tuple(p for p in packets if (p.field_top, p.field_left) == (field.top, field.left))
    facts["parallel_local_retained_count"] = sum(p.retained and p.other_final_changed_pixels > 0 and p.after_field_hash == current_hash for p in local)
    facts["parallel_column_restored_count"] = sum(p.column == column and p.restored and p.other_peak_changed_pixels > 0 and p.before_field_hash == current_hash for p in local)
    facts["parallel_extension_length"] = 0
    moving_values = {p.value for p in packets}
    prior_values = {p.prior_value for p in packets if p.prior_value is not None}
    if len(moving_values) != 1 or len(prior_values) != 1:
        return facts
    unique_moving_value, = moving_values
    unique_prior_value, = prior_values
    # Conditional geometry only: the DRM must authorize the axial interpretation.
    landing = -1
    for r in range(field.height):
        if field.cells[r][column] != unique_prior_value:
            break
        landing = r
    lengths = tuple(len(line.positions) for line in lines
                    if line.value == unique_moving_value and (landing, column) in line.adjacent_positions)
    facts["parallel_extension_length"] = max(lengths, default=0)
    facts["parallel_conditional_landing_row"] = landing
    facts["parallel_column"] = column
    return facts
