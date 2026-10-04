"""Exact bounded geometry and transition measurements for intersecting square cycles."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Iterable

from agents.yf_arc3_v5.logos.types import stable_digest

from .components import detect_components
from .contracts import ComponentDescription, ComponentExtractionInput, FrameGrid


PointTwice = tuple[int, int]


def _digest(prefix: str, payload: object) -> str:
    return f"{prefix}.{stable_digest(payload)[:12]}"


def _centre_twice(component: ComponentDescription) -> PointTwice:
    return (
        component.bbox.top + component.bbox.bottom,
        component.bbox.left + component.bbox.right,
    )


def _dominant_solid_square_family(
    components: Iterable[ComponentDescription],
) -> tuple[ComponentDescription, ...]:
    families: dict[tuple[int, int, int], list[ComponentDescription]] = defaultdict(list)
    for component in components:
        height = component.bbox.height
        width = component.bbox.width
        if height >= 2 and height == width and component.area == height * width:
            families[(height, width, component.area)].append(component)
    return tuple(
        max(
            (families[key] for key in sorted(families)),
            key=lambda members: (len(members), -members[0].area),
            default=(),
        )
    )


def _repeated_pitch_twice(points: Iterable[PointTwice]) -> int:
    by_row: dict[int, list[int]] = defaultdict(list)
    by_column: dict[int, list[int]] = defaultdict(list)
    for row, column in points:
        by_row[row].append(column)
        by_column[column].append(row)
    differences: Counter[int] = Counter()
    for coordinates in (
        *(by_row[key] for key in sorted(by_row)),
        *(by_column[key] for key in sorted(by_column)),
    ):
        ordered = sorted(set(coordinates))
        differences.update(
            right - left
            for left, right in zip(ordered, ordered[1:])
            if right > left
        )
    return min(sorted(differences.items()), key=lambda item: (-item[1], item[0]))[0] if differences else 0


def _runs(values: Iterable[int], pitch: int) -> tuple[tuple[int, ...], ...]:
    ordered = sorted(set(values))
    runs: list[list[int]] = []
    current: list[int] = []
    for value in ordered:
        if current and value - current[-1] != pitch:
            if len(current) >= 3:
                runs.append(current)
            current = []
        current.append(value)
    if len(current) >= 3:
        runs.append(current)
    return tuple(tuple(run) for run in runs)


def _rectangular_cycles(points: frozenset[PointTwice], pitch: int) -> tuple[tuple[PointTwice, ...], ...]:
    rows = sorted({point[0] for point in points})
    columns = sorted({point[1] for point in points})
    cycles: dict[frozenset[PointTwice], tuple[PointTwice, ...]] = {}
    for top_index, top in enumerate(rows):
        for bottom in rows[top_index + 2 :]:
            if (bottom - top) % pitch:
                continue
            for left_index, left in enumerate(columns):
                for right in columns[left_index + 2 :]:
                    if (right - left) % pitch:
                        continue
                    top_edge = tuple((top, column) for column in range(left, right + 1, pitch))
                    right_edge = tuple((row, right) for row in range(top + pitch, bottom + 1, pitch))
                    bottom_edge = tuple((bottom, column) for column in range(right - pitch, left - 1, -pitch))
                    left_edge = tuple((row, left) for row in range(bottom - pitch, top, -pitch))
                    cycle = (*top_edge, *right_edge, *bottom_edge, *left_edge)
                    key = frozenset(cycle)
                    if len(cycle) >= 8 and key.issubset(points):
                        cycles[key] = cycle
    maximal = []
    for key, cycle in sorted(cycles.items(), key=lambda item: tuple(sorted(item[0]))):
        if any(key < other for other in cycles):
            continue
        maximal.append(cycle)
    if not maximal:
        return ()
    maximum_slot_count = max(len(cycle) for cycle in maximal)
    return tuple(
        sorted(
            (cycle for cycle in maximal if len(cycle) == maximum_slot_count),
            key=lambda cycle: cycle,
        )
    )


def _collinear_sequences(
    points: frozenset[PointTwice],
    pitch: int,
    cycle_sets: tuple[frozenset[PointTwice], ...],
) -> tuple[tuple[PointTwice, ...], ...]:
    sequences: list[tuple[PointTwice, ...]] = []
    by_row: dict[int, list[int]] = defaultdict(list)
    by_column: dict[int, list[int]] = defaultdict(list)
    for row, column in points:
        by_row[row].append(column)
        by_column[column].append(row)
    for row, columns in sorted(by_row.items()):
        sequences.extend(tuple((row, column) for column in run) for run in _runs(columns, pitch))
    for column, rows in sorted(by_column.items()):
        sequences.extend(tuple((row, column) for row in run) for run in _runs(rows, pitch))
    return tuple(
        sequence
        for sequence in sequences
        if not any(frozenset(sequence) <= cycle for cycle in cycle_sets)
    )


def _bbox_gap(left: ComponentDescription, right: ComponentDescription) -> int:
    row_gap = max(0, left.bbox.top - right.bbox.bottom - 1, right.bbox.top - left.bbox.bottom - 1)
    column_gap = max(0, left.bbox.left - right.bbox.right - 1, right.bbox.left - left.bbox.right - 1)
    return max(row_gap, column_gap)


def _horizontal_tip_sign(component: ComponentDescription) -> int:
    counts = Counter(column for _row, column in component.relative_pixels)
    left = counts.get(0, 0)
    right = counts.get(component.bbox.width - 1, 0)
    return -1 if left < right else 1 if right < left else 0


def measure_distributed_square_structure(
    frame: FrameGrid | None,
    components: Iterable[ComponentDescription],
) -> dict[str, object]:
    components = tuple(components)
    members = _dominant_solid_square_family(components)
    points = frozenset(_centre_twice(component) for component in members)
    pitch = _repeated_pitch_twice(points)
    if len(members) < 8 or pitch <= 0:
        return {}
    by_point = {_centre_twice(component): component for component in members}
    cycles = _rectangular_cycles(points, pitch)
    cycle_sets = tuple(frozenset(cycle) for cycle in cycles)
    lines = _collinear_sequences(points, pitch, cycle_sets)
    raw_structures = tuple(("closed_rectangular_perimeter", item) for item in cycles) + tuple(
        ("collinear_wrap_candidate", item) for item in lines
    )
    structures = []
    for kind, slots in raw_structures:
        ref = _digest("measurement.distributed_square_structure", (kind, slots))
        structures.append(
            {
                "structure_ref": ref,
                "geometry_kind": kind,
                "slot_centers_twice": slots,
                "slot_count": len(slots),
                "slot_values": tuple(by_point[slot].value for slot in slots),
            }
        )
    memberships: Counter[PointTwice] = Counter(
        slot for structure in structures for slot in structure["slot_centers_twice"]
    )
    shared = frozenset(point for point, count in sorted(memberships.items()) if count > 1)

    palette_counts: Counter[int] = Counter()
    for component in components:
        palette_counts[component.value] += component.area
    excluded_values = frozenset(value for value, _count in palette_counts.most_common(2))
    small = tuple(component for component in components if 0 < component.area < members[0].area)
    markers = []
    for slot in sorted(points):
        nearby: dict[int, list[ComponentDescription]] = defaultdict(list)
        for component in small:
            centre = _centre_twice(component)
            distance = max(abs(centre[0] - slot[0]), abs(centre[1] - slot[1]))
            if 0 < distance <= pitch and component.value not in excluded_values:
                nearby[component.value].append(component)
        for value, candidates in sorted(nearby.items()):
            centres = {_centre_twice(component) for component in candidates}
            paired = tuple(
                component
                for component in candidates
                if (2 * slot[0] - _centre_twice(component)[0], 2 * slot[1] - _centre_twice(component)[1])
                in centres
            )
            if len(paired) >= 2 and sum(component.area for component in paired) >= members[0].area:
                markers.append(
                    {
                        "marker_ref": _digest("measurement.reflection_marker", (slot, value)),
                        "slot_center_twice": slot,
                        "visual_value": value,
                        "component_count": len(paired),
                    }
                )
                break

    controls = tuple(
        component
        for component in components
        if members[0].area < component.area <= members[0].area * 4
        and component.bbox.height > 1
        and component.bbox.width > 1
        and component not in members
    )
    associations = []
    for control in controls:
        centre = _centre_twice(control)
        for structure in structures:
            slots = tuple(structure["slot_centers_twice"])
            slot_components = tuple(by_point[slot] for slot in slots)
            adjacent = any(_bbox_gap(control, component) <= 1 for component in slot_components)
            if structure["geometry_kind"] == "collinear_wrap_candidate":
                same_row = len({slot[0] for slot in slots}) == 1 and centre[0] == slots[0][0]
                same_column = len({slot[1] for slot in slots}) == 1 and centre[1] == slots[0][1]
                outside = (
                    centre[1] < min(slot[1] for slot in slots)
                    or centre[1] > max(slot[1] for slot in slots)
                    or centre[0] < min(slot[0] for slot in slots)
                    or centre[0] > max(slot[0] for slot in slots)
                )
                adjacent = adjacent and outside and (same_row or same_column)
            if not adjacent:
                continue
            associations.append(
                {
                    "component_ref": control.component_id,
                    "point": (centre[0] // 2, centre[1] // 2),
                    "structure_ref": structure["structure_ref"],
                    "horizontal_tip_sign": _horizontal_tip_sign(control),
                    "shape_signature": (
                        control.bbox.height,
                        control.bbox.width,
                        control.area,
                        control.relative_pixels,
                    ),
                }
            )
    target_counts = Counter()
    for marker in markers:
        for structure in structures:
            if marker["slot_center_twice"] in structure["slot_centers_twice"]:
                target_counts[str(structure["structure_ref"])] += 1
    for structure in structures:
        structure["shared_slot_count"] = sum(
            slot in shared for slot in structure["slot_centers_twice"]
        )
        structure["marker_count"] = target_counts[str(structure["structure_ref"])]
    return {
        "pitch_twice": pitch,
        "carrier_count": len(members),
        "structures": tuple(structures),
        "shared_slot_centers_twice": tuple(sorted(shared)),
        "markers": tuple(markers),
        "associations": tuple(associations),
    }


def measure_distributed_square_transition(
    before: FrameGrid,
    after: FrameGrid,
    point: tuple[int, int],
) -> dict[str, object]:
    before_components = detect_components(ComponentExtractionInput(frame=before)).components
    after_components = detect_components(ComponentExtractionInput(frame=after)).components
    structure = measure_distributed_square_structure(before, before_components)
    if not structure:
        return {}
    after_members = _dominant_solid_square_family(after_components)
    after_values = {_centre_twice(component): component.value for component in after_members}
    matches = []
    for item in structure["structures"]:
        slots = tuple(item["slot_centers_twice"])
        prior = tuple(item["slot_values"])
        if any(slot not in after_values for slot in slots):
            continue
        current = tuple(after_values[slot] for slot in slots)
        for offset in (-1, 1):
            if all(current[(index + offset) % len(slots)] == prior[index] for index in range(len(slots))):
                matches.append((str(item["structure_ref"]), offset))
    associations = tuple(
        item for item in structure["associations"] if tuple(item["point"]) == tuple(point)
    )
    exact = tuple(
        (structure_ref, offset)
        for structure_ref, offset in matches
        if any(item["structure_ref"] == structure_ref for item in associations)
    )
    if len(exact) != 1:
        return {}
    structure_ref, offset = exact[0]
    return {
        "point": tuple(point),
        "structure_ref": structure_ref,
        "signed_slot_offset": offset,
        "scene_changed_cell_count": sum(
            left != right
            for before_row, after_row in zip(before.rows, after.rows)
            for left, right in zip(before_row, after_row)
        ),
    }


def enumerate_distributed_square_routes(
    structure: dict[str, object],
    operators: tuple[dict[str, object], ...],
    *,
    maximum_depth: int = 24,
    maximum_routes: int = 3,
) -> tuple[tuple[str, ...], ...]:
    structures = {str(item["structure_ref"]): item for item in structure.get("structures", ())}
    markers = tuple(structure.get("markers", ()))
    if not structures or not markers or not operators:
        return ()
    slots = tuple(
        sorted(
            {
                slot
                for structure_ref in sorted(structures)
                for slot in structures[structure_ref]["slot_centers_twice"]
            }
        )
    )
    slot_index = {slot: index for index, slot in enumerate(slots)}
    values: dict[PointTwice, int] = {}
    for structure_ref in sorted(structures):
        item = structures[structure_ref]
        values.update(zip(item["slot_centers_twice"], item["slot_values"]))
    required_values = tuple(sorted({int(marker["visual_value"]) for marker in markers}))
    initial = tuple(
        tuple(sorted(slot_index[slot] for slot, value in sorted(values.items()) if value == required))
        for required in required_values
    )
    requirements = tuple(
        (slot_index[marker["slot_center_twice"]], required_values.index(int(marker["visual_value"])))
        for marker in markers
    )
    locked = frozenset((slot, value_index) for slot, value_index in requirements if slot in initial[value_index])

    def complete(state: tuple[tuple[int, ...], ...]) -> bool:
        return all(slot in state[value_index] for slot, value_index in requirements)

    def preserves(state: tuple[tuple[int, ...], ...]) -> bool:
        return all(slot in state[value_index] for slot, value_index in locked)

    ordered_operators = tuple(
        sorted(
            operators,
            key=lambda item: (
                str(item["structure_ref"]),
                int(item["signed_slot_offset"]),
                str(item["component_ref"]),
            ),
        )
    )

    def advance(state: tuple[tuple[int, ...], ...], operator: dict[str, object]):
        item = structures[str(operator["structure_ref"])]
        indices = tuple(slot_index[slot] for slot in item["slot_centers_twice"])
        local = {global_index: index for index, global_index in enumerate(indices)}
        offset = int(operator["signed_slot_offset"])
        return tuple(
            tuple(
                sorted(
                    indices[(local[position] + offset) % len(indices)] if position in local else position
                    for position in positions
                )
            )
            for positions in state
        )

    queue = deque([(initial, ())])
    visited_depth = {initial: 0}
    routes: list[tuple[str, ...]] = []
    shortest: int | None = None
    while queue:
        state, path = queue.popleft()
        if shortest is not None and len(path) > shortest:
            break
        if complete(state):
            shortest = len(path)
            routes.append(path)
            if len(routes) >= maximum_routes:
                break
            continue
        if len(path) >= maximum_depth:
            continue
        for operator in ordered_operators:
            next_state = advance(state, operator)
            if not preserves(next_state):
                continue
            next_depth = len(path) + 1
            if visited_depth.get(next_state, next_depth) < next_depth:
                continue
            visited_depth[next_state] = next_depth
            queue.append((next_state, (*path, str(operator["component_ref"]))))
    return tuple(routes)
