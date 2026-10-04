"""Exact bounded geometry for oriented relation transport.

This module does not assign roles, goals, mechanisms, or strategy.  It measures
one structural description made from a repeated square family, a full
sufficient-capacity rectangle, and one asymmetric square control carrier.  It
then keeps only the first deterministic shortest witness under the route bounds
declared by DRM.  Meaning and action eligibility remain declarative.
"""

from __future__ import annotations

from collections import deque
import hashlib
from typing import Iterable, Mapping

from agents.yf_arc3_v5.capabilities.components import detect_components
from agents.yf_arc3_v5.capabilities.contracts import (
    BoundingBox,
    ComponentExtractionInput,
    ComponentExtractionResult,
    FrameGrid,
    InteractionProbeInput,
)
from agents.yf_arc3_v5.logos.types import FrozenMap


Cell = tuple[int, int]
Delta = tuple[int, int]


def _single_carrier_translation_lower_bound(
    *, actor: Cell, members: tuple[Cell, ...], capacity_cells: tuple[Cell, ...],
    maximum_assignment_expansions: int,
) -> tuple[int | None, int]:
    """Measure a relaxed rigid-offset model, not a route or action decision.

    Translations loaded with a remaining member equal that cargo's translations,
    including staged segments. All other carrier translations supply the
    difference to its endpoint (empty or carrying an already placed member).
    Ignore obstacles/orientation and relax the final pose to the capacity extent
    enlarged by the model's one-cell offset. Retain one minimum cost per mask.
    A reached bound yields unknown, never an impossible-plan conclusion.
    """
    capacity = tuple(sorted(set(capacity_cells)))
    remaining = tuple(cell for cell in members if cell not in capacity)
    if not remaining:
        return 0, 0
    if not capacity or len(remaining) > len(capacity) or maximum_assignment_expansions <= 0:
        return None, 0
    endpoint_intervals = tuple(
        (min(cell[axis] for cell in capacity) - 1 - actor[axis],
         max(cell[axis] for cell in capacity) + 1 - actor[axis])
        for axis in (0, 1)
    )
    states = {0: 0}
    expansions = 0
    for member in remaining:
        following: dict[int, int] = {}
        for mask, cost in sorted(states.items()):
            for index, target in enumerate(capacity):
                if mask & (1 << index):
                    continue
                if expansions >= maximum_assignment_expansions:
                    return None, expansions
                expansions += 1
                next_mask = mask | (1 << index)
                next_cost = cost + sum(abs(member[axis] - target[axis]) for axis in (0, 1))
                previous_cost = following.get(next_mask)
                if previous_cost is None or next_cost < previous_cost:
                    following[next_mask] = next_cost
        states = following
    minimum_cost: int | None = None
    for mask, loaded_cost in sorted(states.items()):
        cargo_delta = tuple(
            sum(cell[axis] for index, cell in enumerate(capacity) if mask & (1 << index))
            - sum(cell[axis] for cell in remaining)
            for axis in (0, 1)
        )
        empty_cost = sum(
            max(0, low - delta, delta - high)
            for delta, (low, high) in zip(cargo_delta, endpoint_intervals)
        )
        total = loaded_cost + empty_cost
        minimum_cost = total if minimum_cost is None else min(minimum_cost, total)
    return minimum_cost, expansions


def _box(component: object) -> BoundingBox:
    return component.bbox


def _height(box: BoundingBox) -> int:
    return int(box.bottom - box.top + 1)


def _width(box: BoundingBox) -> int:
    return int(box.right - box.left + 1)


def _inside(box: BoundingBox, height: int, width: int) -> bool:
    return bool(
        0 <= box.top <= box.bottom < height
        and 0 <= box.left <= box.right < width
    )


def _overlaps(first: BoundingBox, second: BoundingBox) -> bool:
    return not (
        first.bottom < second.top
        or second.bottom < first.top
        or first.right < second.left
        or second.right < first.left
    )


def _same_current_translation_covers_regions(
    *, frame: FrameGrid, components: ComponentExtractionResult,
    translations: tuple[tuple[str, int, int], ...],
    first: BoundingBox, second: BoundingBox | None, background_value: int,
) -> bool:
    """Compare current exact pixel coverage under one non-zero vector.

    Bounding-box overlap and historical control membership are insufficient.
    All occupied pixels of both measured core regions must share the vector.
    """
    if second is None:
        return False
    required = []
    for region in (first, second):
        pixels = frozenset(
            (row, col)
            for row in range(region.top, region.bottom + 1)
            for col in range(region.left, region.right + 1)
            if frame.rows[row][col] != background_value
        )
        if not pixels:
            return False
        required.append(pixels)
    required_pixels = required[0] | required[1]
    current_components = {item.component_id: item for item in components.components}
    coverage: dict[Delta, set[Cell]] = {}
    for component_ref, delta_row, delta_col in translations:
        delta = (delta_row, delta_col)
        component = current_components.get(component_ref)
        if delta == (0, 0) or component is None:
            continue
        covered = coverage.setdefault(delta, set())
        covered.update(
            (row, col)
            for row, col in component.pixels
            if (row, col) in required_pixels
        )
    return any(required_pixels.issubset(coverage[delta]) for delta in sorted(coverage))


def _full_non_background(
    rows: tuple[tuple[int, ...], ...],
    box: BoundingBox,
    background_value: int,
) -> bool:
    return all(
        int(rows[row][column]) != background_value
        for row in range(box.top, box.bottom + 1)
        for column in range(box.left, box.right + 1)
    )


def _shape_key(component: object) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    relative = tuple(
        (int(row), int(column)) for row, column in component.relative_pixels
    )
    return (_height(component.bbox), _width(component.bbox), relative)


def _actor_measurements(
    value: InteractionProbeInput,
    *,
    quantum: int,
    background_value: int,
) -> tuple[tuple[BoundingBox, BoundingBox, Delta], ...]:
    current_frame = value.current_observed_frame or value.frame
    rows = current_frame.rows
    height = current_frame.height
    width = current_frame.width
    results: list[tuple[BoundingBox, BoundingBox, Delta]] = []
    for component in value.components.components:
        body = _box(component)
        if component.touches_frame_boundary or int(component.area) != quantum * (
            quantum - 1
        ):
            continue
        candidates: tuple[tuple[BoundingBox, Delta], ...]
        if (_height(body), _width(body)) == (quantum - 1, quantum):
            candidates = (
                (
                    BoundingBox(
                        top=body.top - 1,
                        left=body.left,
                        bottom=body.bottom,
                        right=body.right,
                    ),
                    (-1, 0),
                ),
                (
                    BoundingBox(
                        top=body.top,
                        left=body.left,
                        bottom=body.bottom + 1,
                        right=body.right,
                    ),
                    (1, 0),
                ),
            )
        elif (_height(body), _width(body)) == (quantum, quantum - 1):
            candidates = (
                (
                    BoundingBox(
                        top=body.top,
                        left=body.left - 1,
                        bottom=body.bottom,
                        right=body.right,
                    ),
                    (0, -1),
                ),
                (
                    BoundingBox(
                        top=body.top,
                        left=body.left,
                        bottom=body.bottom,
                        right=body.right + 1,
                    ),
                    (0, 1),
                ),
            )
        else:
            continue
        for square, facing in candidates:
            if not _inside(square, height, width):
                continue
            if facing == (-1, 0):
                face = BoundingBox(
                    top=square.top,
                    left=square.left,
                    bottom=square.top,
                    right=square.right,
                )
            elif facing == (1, 0):
                face = BoundingBox(
                    top=square.bottom,
                    left=square.left,
                    bottom=square.bottom,
                    right=square.right,
                )
            elif facing == (0, -1):
                face = BoundingBox(
                    top=square.top,
                    left=square.left,
                    bottom=square.bottom,
                    right=square.left,
                )
            else:
                face = BoundingBox(
                    top=square.top,
                    left=square.right,
                    bottom=square.bottom,
                    right=square.right,
                )
            if _full_non_background(rows, face, background_value):
                results.append((square, body, facing))
    unique = {
        (
            item[0].top,
            item[0].left,
            item[0].bottom,
            item[0].right,
            item[2],
        ): item
        for item in results
    }
    return tuple(unique[key] for key in sorted(unique))


def _shortest_path(
    *,
    start: Cell,
    start_facing: Delta,
    terminal: Cell,
    terminal_facing: Delta,
    occupied: frozenset[Cell],
    deltas: tuple[tuple[str, Delta], ...],
    height: int,
    width: int,
    maximum_expanded_states: int,
) -> tuple[str, ...] | None:
    action_by_delta = {delta: action_ref for action_ref, delta in deltas}
    if start == terminal:
        return (
            ()
            if start_facing == terminal_facing
            else (action_by_delta[terminal_facing],)
        )
    queue: deque[tuple[Cell, Delta, tuple[str, ...]]] = deque(
        ((start, start_facing, ()),)
    )
    visited = {(start, start_facing)}
    expanded = 0
    best: tuple[str, ...] | None = None
    while queue and expanded < maximum_expanded_states:
        current, current_facing, route = queue.popleft()
        expanded += 1
        if best is not None and len(route) >= len(best):
            continue
        for action_ref, (delta_row, delta_col) in deltas:
            next_facing = (delta_row, delta_col)
            after = (current[0] + delta_row, current[1] + delta_col)
            if not (0 <= after[0] < height and 0 <= after[1] < width):
                continue
            state = (after, next_facing)
            if after in occupied or state in visited:
                continue
            next_route = (*route, action_ref)
            if after == terminal:
                complete = (
                    next_route
                    if next_facing == terminal_facing
                    else (*next_route, action_by_delta[terminal_facing])
                )
                if best is None or (len(complete), complete) < (len(best), best):
                    best = complete
            visited.add(state)
            queue.append((after, next_facing, next_route))
    return best


def _shortest_joint_path(
    *,
    actor: Cell,
    member: Cell,
    terminal_member: Cell,
    occupied: frozenset[Cell],
    deltas: tuple[tuple[str, Delta], ...],
    height: int,
    width: int,
    maximum_expanded_states: int,
) -> tuple[str, ...] | None:
    if member == terminal_member:
        return ()
    start = (actor, member)
    queue: deque[tuple[tuple[Cell, Cell], tuple[str, ...]]] = deque(((start, ()),))
    visited = {start}
    expanded = 0
    while queue and expanded < maximum_expanded_states:
        (current_actor, current_member), route = queue.popleft()
        expanded += 1
        for action_ref, (delta_row, delta_col) in deltas:
            after_actor = (
                current_actor[0] + delta_row,
                current_actor[1] + delta_col,
            )
            after_member = (
                current_member[0] + delta_row,
                current_member[1] + delta_col,
            )
            if not (
                0 <= after_actor[0] < height
                and 0 <= after_actor[1] < width
                and 0 <= after_member[0] < height
                and 0 <= after_member[1] < width
            ):
                continue
            if after_actor in occupied or after_member in occupied:
                continue
            state = (after_actor, after_member)
            if state in visited:
                continue
            next_route = (*route, action_ref)
            if after_member == terminal_member:
                return next_route
            visited.add(state)
            queue.append((state, next_route))
    return None


def _line_values(
    rows: tuple[tuple[int, ...], ...], box: BoundingBox
) -> tuple[int, ...]:
    return tuple(
        int(rows[row][column])
        for row in range(box.top, box.bottom + 1)
        for column in range(box.left, box.right + 1)
    )


def _edge_box(box: BoundingBox, delta: Delta, *, near: bool) -> BoundingBox:
    delta_row, delta_col = delta
    if delta_row < 0:
        row = box.top if near else box.bottom
        return BoundingBox(top=row, left=box.left, bottom=row, right=box.right)
    if delta_row > 0:
        row = box.bottom if near else box.top
        return BoundingBox(top=row, left=box.left, bottom=row, right=box.right)
    if delta_col < 0:
        column = box.left if near else box.right
        return BoundingBox(top=box.top, left=column, bottom=box.bottom, right=column)
    column = box.right if near else box.left
    return BoundingBox(top=box.top, left=column, bottom=box.bottom, right=column)


def _witness(
    *,
    actor: Cell,
    facing: Delta,
    members: tuple[Cell, ...],
    capacity_cells: tuple[Cell, ...],
    equal_value_bridge_present: bool,
    front_member: Cell | None,
    nondirectional_action_ref: str,
    deltas: tuple[tuple[str, Delta], ...],
    height: int,
    width: int,
    maximum_expanded_states: int,
    maximum_actions: int,
    externally_reserved_members: frozenset[Cell] = frozenset(),
) -> tuple[str, ...]:
    occupied_capacity = frozenset(set(members).intersection(capacity_cells))
    available_capacity = tuple(
        cell for cell in capacity_cells if cell not in occupied_capacity
    )
    if equal_value_bridge_present and front_member is not None:
        if front_member in capacity_cells:
            return (nondirectional_action_ref,)
        other_members = frozenset(cell for cell in members if cell != front_member)
        minimum_route: tuple[str, ...] = ()
        for terminal in available_capacity:
            route = _shortest_joint_path(
                actor=actor,
                member=front_member,
                terminal_member=terminal,
                occupied=other_members,
                deltas=deltas,
                height=height,
                width=width,
                maximum_expanded_states=maximum_expanded_states,
            )
            if route is not None and len(route) + 1 <= maximum_actions:
                witness = (*route, nondirectional_action_ref)
                if not minimum_route or (len(witness), witness) < (len(minimum_route), minimum_route):
                    minimum_route = witness
        return minimum_route

    open_members = tuple(cell for cell in members if cell not in capacity_cells and cell not in externally_reserved_members)
    minimum_route = ()
    action_by_delta = {delta: action_ref for action_ref, delta in deltas}
    for member in open_members:
        other_members = frozenset(cell for cell in members if cell != member)
        for terminal in available_capacity:
            for relation_delta in action_by_delta:
                support = (
                    member[0] - relation_delta[0],
                    member[1] - relation_delta[1],
                )
                if support in other_members or not (
                    0 <= support[0] < height and 0 <= support[1] < width
                ):
                    continue
                approach = _shortest_path(
                    start=actor,
                    start_facing=facing,
                    terminal=support,
                    terminal_facing=relation_delta,
                    occupied=frozenset(members),
                    deltas=deltas,
                    height=height,
                    width=width,
                    maximum_expanded_states=maximum_expanded_states,
                )
                if approach is None:
                    continue
                cargo = _shortest_joint_path(
                    actor=support,
                    member=member,
                    terminal_member=terminal,
                    occupied=other_members,
                    deltas=deltas,
                    height=height,
                    width=width,
                    maximum_expanded_states=maximum_expanded_states,
                )
                if cargo is None:
                    continue
                route = (
                    *approach,
                    nondirectional_action_ref,
                    *cargo,
                    nondirectional_action_ref,
                )
                if len(route) <= maximum_actions:
                    if not minimum_route or (len(route), route) < (len(minimum_route), minimum_route):
                        minimum_route = route
    return minimum_route


def measure_oriented_square_transport(
    value: InteractionProbeInput,
    *, include_witness: bool = True,
) -> tuple[FrozenMap, FrozenMap]:
    """Return shared and per-action measurements for one unique description."""

    current_frame = value.current_observed_frame or value.frame
    rows = current_frame.rows
    height = current_frame.height
    width = current_frame.width
    # This measurement is defined over exact current-frame geometry.  Temporal
    # perception may intentionally transport a prior semantic decomposition;
    # recompute the bounded 4-connected components locally so current pixels
    # are never combined with stale component coordinates.
    current_components = detect_components(
        ComponentExtractionInput(frame=current_frame, connectivity=4)
    )
    measurement_value = value.model_copy(
        update={"frame": current_frame, "components": current_components}
    )
    background_component = max(
        current_components.components,
        key=lambda component: (int(component.area), str(component.component_id)),
    )
    background_value = int(background_component.value)
    groups: dict[
        tuple[int, int, tuple[tuple[int, int], ...]], list[object]
    ] = {}
    for component in current_components.components:
        box = _box(component)
        if (
            not component.touches_frame_boundary
            and _height(box) == _width(box)
            and int(component.area) == _height(box) * _width(box)
        ):
            groups.setdefault(_shape_key(component), []).append(component)

    descriptions: list[dict[str, object]] = []
    for (core_height, core_width, _relative), components in sorted(groups.items()):
        member_count = len(components)
        if not 2 <= member_count <= 8 or core_height != core_width:
            continue
        quantum = core_height * 2
        offset = (quantum - core_height) // 2
        member_boxes = tuple(
            BoundingBox(
                top=_box(component).top - offset,
                left=_box(component).left - offset,
                bottom=_box(component).top - offset + quantum - 1,
                right=_box(component).left - offset + quantum - 1,
            )
            for component in components
        )
        if any(
            not _inside(box, height, width)
            or not _full_non_background(rows, box, background_value)
            for box in member_boxes
        ):
            continue
        row_residue = member_boxes[0].top % quantum
        col_residue = member_boxes[0].left % quantum
        if any(
            box.top % quantum != row_residue
            or box.left % quantum != col_residue
            for box in member_boxes
        ):
            continue
        # Quantify observed rectangles, not an assumed one-row arrangement.
        # The enclosing component must share an observed value with the cores;
        # neither that value nor its numeric palette index assigns a role.
        core_values = frozenset(int(component.value) for component in components)
        rectangle_boxes = [
            _box(component)
            for component in current_components.components
            if int(component.value) in core_values
        ]
        # Preserve the established equal-capacity descriptor when a member
        # occludes the contour and splits its observed colour component.
        # These are bounded geometric boxes, not additional route witnesses.
        for horizontal in (True, False):
            strip_height = quantum if horizontal else quantum * member_count
            strip_width = quantum * member_count if horizontal else quantum
            for top in range(row_residue, height - strip_height + 1, quantum):
                for left in range(col_residue, width - strip_width + 1, quantum):
                    rectangle_boxes.append(BoundingBox(
                        top=top, left=left,
                        bottom=top + strip_height - 1,
                        right=left + strip_width - 1,
                    ))
        prior_capacity_bbox = value.prior_oriented_square_transport_capacity_bbox
        if prior_capacity_bbox:
            rectangle_boxes.append(BoundingBox(
                top=int(prior_capacity_bbox[0]), left=int(prior_capacity_bbox[1]),
                bottom=int(prior_capacity_bbox[2]), right=int(prior_capacity_bbox[3]),
            ))
        maximum_capacity_cells = min(64, max(1, int(
            value.route_generation_policy.get("oriented_transport_max_capacity_cells", 64)
        )))
        capacity_boxes: list[BoundingBox] = []
        for candidate in rectangle_boxes:
            box_height, box_width = _height(candidate), _width(candidate)
            if (
                not _inside(candidate, height, width)
                or candidate.bottom >= height - 1
                or candidate.right >= width - 1
                or candidate.top % quantum != row_residue
                or candidate.left % quantum != col_residue
                or box_height % quantum != 0
                or box_width % quantum != 0
                or (box_height // quantum) * (box_width // quantum) > maximum_capacity_cells
                or not _full_non_background(rows, candidate, background_value)
                or len(frozenset(_line_values(rows, candidate))) < 2
            ):
                continue
            capacity_boxes.append(candidate)
        capacity_boxes = list(
            {
                (box.top, box.left, box.bottom, box.right): box
                for box in capacity_boxes
            }.values()
        )
        # A fragment left visible beside an occluder is not a second extent
        # when the same fully supported rectangle strictly contains it.
        capacity_boxes = [
            candidate for candidate in capacity_boxes
            if not any(
                enclosing != candidate
                and enclosing.top <= candidate.top
                and enclosing.left <= candidate.left
                and enclosing.bottom >= candidate.bottom
                and enclosing.right >= candidate.right
                for enclosing in capacity_boxes
            )
        ]
        prior_capacity_bbox = value.prior_oriented_square_transport_capacity_bbox
        if prior_capacity_bbox:
            prior_matches = [
                candidate
                for candidate in capacity_boxes
                if (
                    candidate.top,
                    candidate.left,
                    candidate.bottom,
                    candidate.right,
                )
                == tuple(int(item) for item in prior_capacity_bbox)
            ]
            if prior_matches:
                capacity_boxes = prior_matches
        if capacity_boxes:
            minimum_member_overlap = min(
                sum(_overlaps(candidate, member_box) for member_box in member_boxes)
                for candidate in capacity_boxes
            )
            capacity_boxes = [
                candidate
                for candidate in capacity_boxes
                if sum(
                    _overlaps(candidate, member_box) for member_box in member_boxes
                )
                == minimum_member_overlap
            ]
        actors = _actor_measurements(
            measurement_value,
            quantum=quantum,
            background_value=background_value,
        )
        for capacity_box in capacity_boxes:
            for actor_box, actor_body_box, facing in actors:
                if (
                    actor_box.top % quantum != row_residue
                    or actor_box.left % quantum != col_residue
                ):
                    continue
                if any(
                    _overlaps(actor_box, member_box) for member_box in member_boxes
                ):
                    continue
                descriptions.append(
                    {
                        "quantum": quantum,
                        "core_value": int(components[0].value),
                        "member_boxes": tuple(
                            sorted(
                                member_boxes,
                                key=lambda box: (box.top, box.left),
                            )
                        ),
                        "core_boxes": tuple(
                            sorted(
                                (_box(component) for component in components),
                                key=lambda box: (box.top, box.left),
                            )
                        ),
                        "capacity_box": capacity_box,
                        "actor_box": actor_box,
                        "actor_body_box": actor_body_box,
                        "facing": facing,
                    }
                )

    descriptions.sort(
        key=lambda item: (
            int(item["quantum"]),
            item["capacity_box"].top,
            item["capacity_box"].left,
            item["actor_box"].top,
            item["actor_box"].left,
        )
    )
    if len(descriptions) != 1:
        return (
            FrozenMap(
                {
                    "oriented_square_transport_description_count": len(
                        descriptions[:3]
                    ),
                    "oriented_square_transport_measurement_present": False,
                    "oriented_square_transport_cardinality_equal": False,
                    "oriented_square_transport_capacity_count_at_least_member_count": False,
                    "oriented_square_transport_route_present": False,
                }
            ),
            FrozenMap(),
        )

    description = descriptions[0]
    quantum = int(description["quantum"])
    member_boxes = tuple(description["member_boxes"])
    core_boxes = tuple(description["core_boxes"])
    capacity_box = description["capacity_box"]
    actor_box = description["actor_box"]
    actor_body_box = description["actor_body_box"]
    facing = tuple(description["facing"])
    origin_row = member_boxes[0].top % quantum
    origin_col = member_boxes[0].left % quantum

    to_cell = lambda box: (
        (box.top - origin_row) // quantum,
        (box.left - origin_col) // quantum,
    )
    members = tuple(to_cell(box) for box in member_boxes)
    # These reservations originate in DRM. Compile only exact current-frame
    # geometry into the supplied route domain; never infer an owner here.
    external_core_constraints = {}
    external_clock_constraints = {}
    current_core_boxes = frozenset(
        (box.top, box.left, box.bottom, box.right) for box in description["core_boxes"]
    )
    for record in value.canonical_external_transport_records:
        attributes = record.get("attributes", FrozenMap())
        core_box = attributes.get("cargo_core_bbox")
        if (attributes.get("cargo_reservation_current") is True
            and value.current_observation_frame_ref is not None
            and attributes.get("observation_frame_ref") == value.current_observation_frame_ref
            and attributes.get("evidence_scope_ref") == f"scope.level:{value.observed_level_index}"
            and isinstance(core_box, tuple) and len(core_box) == 4
            and core_box in current_core_boxes
            and isinstance(attributes.get("source_reservation_claim_ref"), str)):
            external_core_constraints[core_box] = attributes.get("source_reservation_claim_ref")
            if (attributes.get("external_stationary_clock_supported") is True
                and isinstance(attributes.get("stationary_clock_action_ref"), str)
                and isinstance(attributes.get("source_stationary_clock_claim_ref"), str)):
                external_clock_constraints.setdefault(attributes["stationary_clock_action_ref"], set()).add(
                    attributes["source_stationary_clock_claim_ref"]
                )
    externally_reserved_members = frozenset(
        to_cell(member_box) for member_box, core_box in zip(member_boxes, description["core_boxes"])
        if (core_box.top, core_box.left, core_box.bottom, core_box.right) in external_core_constraints
    )
    actor = to_cell(actor_box)
    capacity_cells = tuple(
        (row, col)
        for row in range(
            (capacity_box.top - origin_row) // quantum,
            (capacity_box.top - origin_row) // quantum + _height(capacity_box) // quantum,
        )
        for col in range(
            (capacity_box.left - origin_col) // quantum,
            (capacity_box.left - origin_col) // quantum + _width(capacity_box) // quantum,
        )
    )

    front_cell = (actor[0] + facing[0], actor[1] + facing[1])
    front_index = members.index(front_cell) if front_cell in members else -1
    front_box = member_boxes[front_index] if front_index >= 0 else None
    face_box = _edge_box(actor_box, facing, near=True)
    equal_value_bridge_present = False
    if front_box is not None:
        peer_edge = _edge_box(front_box, facing, near=False)
        face_values = _line_values(rows, face_box)
        peer_values = _line_values(rows, peer_edge)
        equal_value_bridge_present = bool(
            face_values
            and peer_values
            and len(set(face_values)) == 1
            and face_values == peer_values
            and face_values[0] != background_value
        )

    joint_translation_observed = _same_current_translation_covers_regions(
        frame=current_frame,
        components=current_components,
        translations=value.current_exact_component_translations,
        first=actor_body_box,
        second=core_boxes[front_index] if front_index >= 0 else None,
        background_value=background_value,
    )

    interface_deltas = tuple(
        sorted(
            (
                (str(action_ref), (int(delta_row), int(delta_col)))
                for action_ref, delta_row, delta_col
                in value.interface_action_translation_deltas
                if (int(delta_row), int(delta_col)) != (0, 0)
            ),
            key=lambda item: item[0],
        )
    )
    nondirectional_refs = tuple(
        sorted(
            str(action_ref)
            for action_ref in value.available_action_refs
            if action_ref not in {ref for ref, _delta in interface_deltas}
            and action_ref != value.level_start_checkpoint_action_ref
            and action_ref != "ACTION6"
        )
    )
    maximum_expanded_states = min(
        8192,
        max(
            1,
            int(
                value.route_generation_policy.get(
                    "oriented_transport_max_expanded_states", 4096
                )
            ),
        ),
    )
    maximum_actions = min(
        40,
        max(
            1,
            int(
                value.route_generation_policy.get(
                    "oriented_transport_max_actions", 40
                )
            ),
        ),
    )
    route = (
        _witness(
            actor=actor,
            facing=facing,
            members=members,
            capacity_cells=capacity_cells,
            equal_value_bridge_present=equal_value_bridge_present,
            front_member=front_cell if front_index >= 0 else None,
            nondirectional_action_ref=nondirectional_refs[0],
            deltas=interface_deltas,
            height=height // quantum,
            width=width // quantum,
            maximum_expanded_states=maximum_expanded_states,
            maximum_actions=maximum_actions,
            externally_reserved_members=externally_reserved_members,
        )
        if include_witness and len(nondirectional_refs) == 1 and len(interface_deltas) == 4
        else ()
    )
    route_ref = (
        "measurement.oriented_square_route."
        + hashlib.sha256("|".join(route).encode("utf-8")).hexdigest()[:16]
        if route
        else ""
    )
    occupied_count = len(set(members).intersection(capacity_cells))
    assignment_expansion_limit = min(8192, max(0, int(
        value.route_generation_policy.get("oriented_transport_max_assignment_expansions", 0)
    )))
    if {delta for _ref, delta in interface_deltas} != {(0, 1), (0, -1), (1, 0), (-1, 0)}:
        assignment_expansion_limit = 0
    translation_lower_bound, assignment_expansions = _single_carrier_translation_lower_bound(
        actor=actor, members=members, capacity_cells=capacity_cells,
        maximum_assignment_expansions=assignment_expansion_limit,
    )
    current_member_in_capacity = bool(
        front_index >= 0 and front_cell in capacity_cells
    )
    capacity_bbox = (
        capacity_box.top,
        capacity_box.left,
        capacity_box.bottom,
        capacity_box.right,
    )
    shared = FrozenMap(
        {
            "oriented_square_transport_description_count": 1,
            "oriented_square_transport_measurement_present": True,
            "oriented_square_transport_quantum": quantum,
            "oriented_square_transport_actor_body_value": int(current_frame.rows[actor_body_box.top][actor_body_box.left]),
            "oriented_square_transport_member_count": len(members),
            "oriented_square_transport_external_constraint_member_count": len(externally_reserved_members),
            "oriented_square_transport_external_constraint_source_refs": tuple(
                ref for ref in sorted(external_core_constraints.values()) if isinstance(ref, str)
            ),
            "oriented_square_transport_member_core_value": int(description["core_value"]),
            "oriented_square_transport_member_core_height": _height(description["core_boxes"][0]),
            "oriented_square_transport_member_core_width": _width(description["core_boxes"][0]),
            "oriented_square_transport_remaining_member_count": len(members) - occupied_count,
            "oriented_square_transport_unreserved_open_member_count": len(
                set(members).difference(capacity_cells).difference(externally_reserved_members)
            ),
            "oriented_square_transport_single_carrier_cost_bound_computed": translation_lower_bound is not None,
            "oriented_square_transport_single_carrier_translation_lower_bound": translation_lower_bound,
            "oriented_square_transport_assignment_expansions": assignment_expansions,
            "oriented_square_transport_capacity_cell_count": len(capacity_cells),
            "oriented_square_transport_cardinality_equal": len(members)
            == len(capacity_cells),
            "oriented_square_transport_capacity_count_at_least_member_count": (
                len(capacity_cells) >= len(members)
            ),
            "oriented_square_transport_actor_facing_row": int(facing[0]),
            "oriented_square_transport_actor_facing_col": int(facing[1]),
            "oriented_square_transport_actor_cell": tuple(int(x) for x in actor),
            "oriented_square_transport_member_cells": tuple(
                tuple(int(x) for x in cell) for cell in members
            ),
            "oriented_square_transport_capacity_cells": tuple(
                tuple(int(x) for x in cell) for cell in capacity_cells
            ),
            "oriented_square_transport_capacity_bbox": capacity_bbox,
            "oriented_square_transport_capacity_bbox_matches_prior": (
                capacity_bbox == value.prior_oriented_square_transport_capacity_bbox
            ),
            "oriented_square_transport_front_member_present": front_index >= 0,
            "oriented_square_transport_equal_value_bridge_present": (
                equal_value_bridge_present
            ),
            "oriented_square_transport_joint_translation_observed": (
                joint_translation_observed
            ),
            "oriented_square_transport_occupied_capacity_count": occupied_count,
            "oriented_square_transport_all_capacity_occupied": occupied_count
            == len(capacity_cells),
            "oriented_square_transport_all_members_in_capacity": occupied_count
            == len(members),
            "oriented_square_transport_current_member_in_capacity": (
                current_member_in_capacity
            ),
            "oriented_square_transport_route_present": bool(route),
            "oriented_square_transport_route_ref": route_ref,
            "oriented_square_transport_route_action_count": len(route),
            "oriented_square_transport_route_action_refs": route,
            "oriented_square_transport_route_candidate_refs": tuple(
                f"probe:discrete:{action_ref}" for action_ref in route
            ),
        }
    )
    per_action: dict[str, FrozenMap] = {}
    for action_ref in value.available_action_refs:
        is_directional = action_ref in {ref for ref, _delta in interface_deltas}
        is_nondirectional = action_ref in nondirectional_refs
        action_delta = dict(interface_deltas).get(str(action_ref))
        per_action[str(action_ref)] = FrozenMap(
            {
                "candidate_has_current_external_stationary_clock_evidence": str(action_ref) in external_clock_constraints,
                "candidate_is_blocked_free_direction_against_member": bool(
                    is_directional and action_delta == facing and front_index >= 0
                    and not equal_value_bridge_present
                ),
                "oriented_square_transport_stationary_clock_claim_refs": (
                    tuple(sorted(external_clock_constraints[str(action_ref)]))
                    if str(action_ref) in external_clock_constraints else ()
                ),
                "oriented_square_transport_route_action_occurrence_present": bool(
                    route and action_ref == route[0]
                ),
                "candidate_is_first_oriented_square_transport_route_action": bool(
                    route and action_ref == route[0]
                ),
                "candidate_is_nondirectional_at_front_without_equal_value_bridge": bool(
                    is_nondirectional
                    and front_index >= 0
                    and not equal_value_bridge_present
                    and not current_member_in_capacity
                ),
                "candidate_is_directional_at_equal_value_bridge_before_joint_translation": bool(
                    is_directional
                    and equal_value_bridge_present
                    and not joint_translation_observed
                    and route
                    and action_ref == route[0]
                ),
                "candidate_is_directional_at_equal_value_bridge_after_joint_translation": bool(
                    is_directional
                    and equal_value_bridge_present
                    and joint_translation_observed
                    and route
                    and action_ref == route[0]
                ),
                "candidate_is_nondirectional_at_capacity_with_equal_value_bridge": bool(
                    is_nondirectional
                    and equal_value_bridge_present
                    and current_member_in_capacity
                ),
                "oriented_square_transport_candidate_route_ref": (
                    route_ref if route and action_ref == route[0] else ""
                ),
                "oriented_square_transport_candidate_route_action_count": (
                    len(route) if route and action_ref == route[0] else 0
                ),
                "oriented_square_transport_candidate_route_action_refs": (
                    route if route and action_ref == route[0] else ()
                ),
                "oriented_square_transport_capacity_bbox": capacity_bbox,
            }
        )
    return shared, FrozenMap(per_action)
