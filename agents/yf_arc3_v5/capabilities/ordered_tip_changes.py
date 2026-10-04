"""Bounded raster measurements for repeated squares with edge marks.

No palette roles or propagation rules: retain the order of marker loss and
marker-preserving recolouring, including packets whose final picture returns.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from math import gcd

from agents.yf_arc3_v5.capabilities.components import detect_components
from agents.yf_arc3_v5.capabilities.contracts import ComponentExtractionInput, FrameGrid
from agents.yf_arc3_v5.capabilities.cell_deltas import CellDeltaPacket, measure_cell_deltas


MAX_SQUARES = 256
MAX_PACKET_FRAMES = 256


@dataclass(frozen=True)
class MarkedSquare:
    top: int
    left: int
    size: int
    value: int
    marker_value: int
    offsets: tuple[tuple[int, int], ...]

    @property
    def position(self) -> tuple[int, int]:
        return self.top, self.left


@dataclass(frozen=True)
class SquareField:
    squares: tuple[MarkedSquare, ...]
    stride: int
    size: int
    row_phase: int
    col_phase: int


@dataclass(frozen=True)
class OrderedSquareChange:
    square: MarkedSquare
    first_marker_loss: int | None
    first_marker_preserving_recolor: int | None
    final_matches_initial: bool


@dataclass(frozen=True)
class OrderedSquarePacket:
    action_ref: str
    field: SquareField
    changes: tuple[OrderedSquareChange, ...]
    frame_count: int
    distinct_loss_step_count: int
    equal_cardinality_external_component_group_count: int
    cell_changes: CellDeltaPacket


def _patch(frame: FrameGrid, top: int, left: int, size: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(row[left:left + size]) for row in frame.rows[top:top + size])


def measure_square_field(frame: FrameGrid) -> SquareField | None:
    """Infer square extent, marker value and lattice from repeated morphology."""
    components = detect_components(ComponentExtractionInput(frame=frame, connectivity=4)).components
    groups: dict[tuple[int, int], list[MarkedSquare]] = {}
    for component in components:
        box = component.bbox
        size = box.height
        if size < 3 or size % 2 == 0 or box.width != size or component.area != size * size - 1:
            continue
        patch = _patch(frame, box.top, box.left, size)
        offsets = tuple((r, c) for r in range(size) for c in range(size)
                        if patch[r][c] != component.value)
        middle = size // 2
        if len(offsets) != 1 or offsets[0] not in (
            (0, middle), (size - 1, middle), (middle, 0), (middle, size - 1)
        ):
            continue
        r, c = offsets[0]
        square = MarkedSquare(box.top, box.left, size, component.value, patch[r][c], offsets)
        groups.setdefault((size, patch[r][c]), []).append(square)
    repeated = [items for _, items in sorted(groups.items()) if len(items) >= 2]
    if len(repeated) != 1:
        return None
    base = repeated[0]
    if len(base) > MAX_SQUARES:
        return None
    size, marker = base[0].size, base[0].marker_value
    stride = 0
    for coordinate in (sorted({s.top for s in base}), sorted({s.left for s in base})):
        for a, b in zip(coordinate, coordinate[1:]):
            stride = gcd(stride, b - a)
    if stride <= size:
        return None
    row_phase, col_phase = base[0].top % stride, base[0].left % stride
    height, width = len(frame.rows), len(frame.rows[0])
    if len(range(row_phase, height - size + 1, stride)) * len(range(col_phase, width - size + 1, stride)) > MAX_SQUARES:
        return None
    squares = []
    for top in range(row_phase, height - size + 1, stride):
        for left in range(col_phase, width - size + 1, stride):
            patch = _patch(frame, top, left, size)
            counts = Counter(v for row in patch for v in row)
            if len(counts) != 2 or marker not in counts:
                continue
            offsets = tuple((r, c) for r in range(size) for c in range(size) if patch[r][c] == marker)
            value = next(v for v in counts if v != marker)
            middle = size // 2
            edge_midpoints = {(0, middle), (size - 1, middle), (middle, 0), (middle, size - 1)}
            if (len(offsets) == 1 and offsets[0] in edge_midpoints) or set(offsets) == edge_midpoints | {(middle, middle)}:
                squares.append(MarkedSquare(top, left, size, value, marker, offsets))
    return SquareField(tuple(squares), stride, size, row_phase, col_phase)


def measure_ordered_square_packet(
    *, before: FrameGrid, ordered_frames: tuple[FrameGrid, ...], action_ref: str
) -> OrderedSquarePacket | None:
    if not ordered_frames or len(ordered_frames) > MAX_PACKET_FRAMES:
        return None
    field = measure_square_field(before)
    if field is None:
        return None
    sparse = measure_cell_deltas(before=before.rows,
        frames=tuple(frame.rows for frame in ordered_frames),
        row_phase=field.row_phase, col_phase=field.col_phase,
        stride=field.stride, size=field.size)
    if sparse is None:
        return None
    squares = {s.position: s for s in field.squares}
    first_loss, first_recolor = {}, {}
    current_types = list(sparse.initial_types)
    for delta in sparse.deltas:
        for index, _old_type, new_type in delta.changes:
            current_types[index] = new_type
            square = squares.get(sparse.positions[index])
            if square is None:
                continue
            patch = sparse.types[new_type]
            offsets = tuple((r, c) for r in range(square.size) for c in range(square.size)
                            if patch[r][c] == square.marker_value)
            if offsets != square.offsets:
                first_loss.setdefault(square.position, delta.ordinal)
            values = {patch[r][c] for r in range(square.size) for c in range(square.size)
                      if (r, c) not in square.offsets}
            if offsets == square.offsets and len(values) == 1 and square.value not in values:
                first_recolor.setdefault(square.position, delta.ordinal)
    indexes = {p: i for i, p in enumerate(sparse.positions)}
    changes = [OrderedSquareChange(square, first_loss.get(square.position),
               first_recolor.get(square.position),
               current_types[indexes[square.position]] == sparse.initial_types[indexes[square.position]])
               for square in field.squares]
    # A translated overlay alone changes too few disjoint marked squares to
    # provide a distributed temporal relation. Preserve no such packet here.
    if sum(item.first_marker_loss is not None for item in changes) < 2:
        return None
    step_count = len({c.first_marker_loss for c in changes if c.first_marker_loss is not None})
    top, bottom = min(s.top for s in field.squares), max(s.top + s.size for s in field.squares)
    external_groups = Counter()
    for component in detect_components(ComponentExtractionInput(frame=before, connectivity=4)).components:
        if component.area > 1 and not component.touches_frame_boundary and (component.bbox.bottom < top or component.bbox.top >= bottom):
            external_groups[(component.value, component.relative_pixels)] += 1
    matching_count = sum(n == step_count for _, n in sorted(external_groups.items()))
    return OrderedSquarePacket(action_ref, field, tuple(changes), len(ordered_frames), step_count, matching_count, sparse)


def _tip_vector(square: MarkedSquare) -> tuple[int, int] | None:
    if len(square.offsets) != 1:
        return None
    r, c = square.offsets[0]
    middle = square.size // 2
    return (r > middle) - (r < middle), (c > middle) - (c < middle)


def measure_tip_probe_geometry(value: object) -> tuple[dict, dict]:
    """Describe temporal incidences and one bounded shortest geometric witness.

    An incidence is measured from an actual packet, never from a game model.
    DRM decides whether covering its earlier incoming chain is an experiment.
    """
    frame = value.current_observed_frame or value.frame
    field = measure_square_field(frame)
    facts = {"ordered_tip_square_count": 0, "ordered_tip_temporal_incidence_count": 0,
             "ordered_tip_unique_ring_present": False,
             "ordered_tip_ring_on_earlier_chain": False,
             "ordered_tip_route_present": False, "ordered_tip_same_initial_value_count": 0}
    per_action = {}
    if field is None:
        return facts, per_action
    facts["ordered_tip_square_count"] = len(field.squares)
    size, stride = field.size, field.stride
    rings, uniform = [], Counter()
    patches = {}
    for top in range(field.row_phase, len(frame.rows) - size + 1, stride):
        for left in range(field.col_phase, len(frame.rows[0]) - size + 1, stride):
            patch = _patch(frame, top, left, size)
            patches[top, left] = patch
            border = {patch[r][c] for r in range(size) for c in range(size)
                      if r in (0, size - 1) or c in (0, size - 1)}
            inner = {patch[r][c] for r in range(1, size - 1) for c in range(1, size - 1)}
            if len(border) == len(inner) == 1 and border != inner:
                rings.append((top, left))
            if len(border | inner) == 1:
                uniform[next(iter(border))] += 1
    # Associate only rings in the repeated-mark field's one-cell neighbourhood;
    # unrelated equal-sized display glyphs remain outside this local relation.
    # A marker disappearing does not erase the measured grid extent. Retain
    # compatible observation geometry while a distributed change advances.
    scope_squares = tuple(field.squares) + tuple(
        square for packet in value.ordered_square_packets
        if packet.field.stride == stride and packet.field.size == size
        and packet.field.row_phase == field.row_phase and packet.field.col_phase == field.col_phase
        for square in packet.field.squares
    )
    top_bound = min(s.top for s in scope_squares) - stride
    bottom_bound = max(s.top for s in scope_squares) + stride
    left_bound = min(s.left for s in scope_squares) - stride
    right_bound = max(s.left for s in scope_squares) + stride
    rings = [p for p in rings if top_bound <= p[0] <= bottom_bound
             and left_bound <= p[1] <= right_bound]
    if len(rings) != 1 or not uniform:
        return facts, per_action
    ring = rings[0]
    facts["ordered_tip_unique_ring_present"] = True
    facts["ordered_tip_ring_position"] = ring
    # Exact mode of lattice patches; no colour is assigned traversability here.
    modes = [v for v, n in sorted(uniform.items()) if n == max(uniform.values())]
    if len(modes) != 1:
        return facts, per_action
    mode = modes[0]
    current_positions = {s.position for s in field.squares}
    allowed = {p for p, patch in sorted(patches.items()) if all(v == mode for row in patch for v in row)}
    allowed.update(current_positions)
    allowed.add(ring)
    packets = value.ordered_square_packets
    for packet in packets:
        changes = {c.square.position: c for c in packet.changes}
        earlier_positions = set()
        incidences = []
        for cross in packet.changes:
            if len(cross.square.offsets) <= 1 or cross.first_marker_preserving_recolor is None:
                continue
            incoming = []
            for item in packet.changes:
                vector = _tip_vector(item.square)
                if vector is None or item.first_marker_loss is None:
                    continue
                successor = (item.square.top + vector[0] * stride, item.square.left + vector[1] * stride)
                if successor == cross.square.position:
                    incoming.append((item, vector))
            for early, early_vector in incoming:
                for late, late_vector in incoming:
                    if early_vector[0] * late_vector[0] + early_vector[1] * late_vector[1] != 0:
                        continue
                    if not (early.first_marker_loss < cross.first_marker_preserving_recolor < late.first_marker_loss):
                        continue
                    # Follow the late incoming vector through the marked centre;
                    # retain a witness only if its distinct terminal stayed intact.
                    position = (cross.square.top + late_vector[0] * stride,
                                cross.square.left + late_vector[1] * stride)
                    suffix = []
                    while position in changes and len(suffix) < MAX_SQUARES:
                        item = changes[position]
                        vector = _tip_vector(item.square)
                        if vector is None or position in suffix:
                            break
                        suffix.append(position)
                        position = (position[0] + vector[0] * stride, position[1] + vector[1] * stride)
                    if not suffix or len(suffix) == MAX_SQUARES:
                        continue
                    terminal = changes[suffix[-1]]
                    if terminal.first_marker_loss is not None or terminal.square.value == late.square.value:
                        continue
                    incidences.append((cross.square.position, early.square.position, late.square.position, terminal.square.position))
                    position = early.square.position
                    for _ in range(MAX_SQUARES):
                        if position not in changes or position in earlier_positions:
                            break
                        earlier_positions.add(position)
                        predecessors = [c.square.position for c in packet.changes
                                        if (v := _tip_vector(c.square)) is not None
                                        and (c.square.top + v[0] * stride, c.square.left + v[1] * stride) == position]
                        if len(predecessors) != 1:
                            break
                        position = predecessors[0]
        if not incidences:
            continue
        facts["ordered_tip_temporal_incidence_count"] = len(incidences)
        facts["ordered_tip_temporal_incidences"] = tuple(incidences)
        facts["ordered_tip_packet_action_ref"] = packet.action_ref
        facts["ordered_tip_packet_frame_count"] = packet.frame_count
        facts["ordered_tip_cell_type_count"] = len(packet.cell_changes.types)
        facts["ordered_tip_cell_delta_count"] = sum(len(d.changes) for d in packet.cell_changes.deltas)
        facts["ordered_tip_distinct_loss_step_count"] = packet.distinct_loss_step_count
        facts["ordered_tip_equal_cardinality_external_group_count"] = packet.equal_cardinality_external_component_group_count
        facts["ordered_tip_changed_then_returned_count"] = sum(c.first_marker_loss is not None and c.final_matches_initial for c in packet.changes)
        facts["ordered_tip_ring_on_earlier_chain"] = ring in earlier_positions
        losses = [c.first_marker_loss for c in packet.changes if c.first_marker_loss is not None]
        initial_values = {c.square.value for c in packet.changes if c.first_marker_loss == min(losses)}
        facts["ordered_tip_same_initial_value_count"] = sum(s.value in initial_values for s in field.squares)
        if ring in earlier_positions:
            break
        # One BFS, at most one shortest witness, no path-family enumeration.
        queue = deque([ring])
        previous = {ring: None}
        edge = {}
        endpoint = None
        while queue and len(previous) <= MAX_SQUARES:
            position = queue.popleft()
            if position in earlier_positions:
                endpoint = position
                break
            for action_ref, dr, dc in value.interface_action_translation_deltas:
                nxt = position[0] + dr * stride, position[1] + dc * stride
                if nxt in allowed and nxt not in previous:
                    previous[nxt] = position
                    edge[nxt] = action_ref
                    queue.append(nxt)
        if endpoint is not None and len(previous) <= MAX_SQUARES:
            actions = []
            while previous[endpoint] is not None:
                actions.append(edge[endpoint])
                endpoint = previous[endpoint]
            actions.reverse()
            if actions:
                facts["ordered_tip_route_present"] = True
                facts["ordered_tip_route_length"] = len(actions)
                per_action[actions[0]] = {"ordered_tip_first_shortest_route_edge": True}
        break
    return facts, per_action
