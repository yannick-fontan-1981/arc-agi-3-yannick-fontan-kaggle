"""Bounded push-state search over an already declared cell interpretation.

The caller supplies immutable traversability, carrier, item, terminal, and
action-delta facts.  This module only searches and measures them.  It does not
infer what any cell means and it emits no cognitive references.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count


Cell = tuple[int, int]


@dataclass(frozen=True)
class CellAction:
    action_ref: str
    delta_row: int
    delta_col: int

    @property
    def delta(self) -> Cell:
        return self.delta_row, self.delta_col


@dataclass(frozen=True)
class SupportTransportRequest:
    traversable_cells: frozenset[Cell]
    carrier_cell: Cell
    item_cells: tuple[Cell, ...]
    terminal_cells: tuple[Cell, ...]
    actions: tuple[CellAction, ...]
    action_orders: tuple[tuple[str, ...], ...] = ()
    maximum_expanded_states: int = 4096
    maximum_primitive_actions: int = 120
    maximum_witnesses: int = 3

    def __post_init__(self) -> None:
        if not 1 <= self.maximum_witnesses <= 3:
            raise ValueError("maximum witnesses must be between one and three")
        if self.maximum_expanded_states < 1 or self.maximum_primitive_actions < 1:
            raise ValueError("search bounds must be positive")
        if len(self.item_cells) != len(self.terminal_cells) or not self.item_cells:
            raise ValueError("item and terminal cardinalities must be equal and nonzero")
        if len(set(self.item_cells)) != len(self.item_cells):
            raise ValueError("item cells must be unique")
        if len(set(self.terminal_cells)) != len(self.terminal_cells):
            raise ValueError("terminal cells must be unique")
        if self.carrier_cell not in self.traversable_cells:
            raise ValueError("carrier cell must be traversable")
        if any(cell not in self.traversable_cells for cell in self.item_cells):
            raise ValueError("every item cell must be traversable")
        if any(cell not in self.traversable_cells for cell in self.terminal_cells):
            raise ValueError("every terminal cell must be traversable")
        if len({action.action_ref for action in self.actions}) != len(self.actions):
            raise ValueError("action references must be unique")
        if len({action.delta for action in self.actions}) != len(self.actions):
            raise ValueError("action deltas must be unique")
        if any(abs(action.delta_row) + abs(action.delta_col) != 1 for action in self.actions):
            raise ValueError("actions must be unit cardinal translations")
        known_refs = {action.action_ref for action in self.actions}
        if any(
            len(order) != len(self.actions)
            or set(order) != known_refs
            for order in self.action_orders
        ):
            raise ValueError("each action order must contain every action exactly once")
        if len(self.action_orders) > 3:
            raise ValueError("at most three declared action orders are allowed")


@dataclass(frozen=True)
class SupportTransportWitness:
    action_refs: tuple[str, ...]
    push_count: int
    walk_count: int
    expanded_state_count: int
    final_item_cells: tuple[Cell, ...]
    action_order: tuple[str, ...]


@dataclass(frozen=True)
class SupportTransportResult:
    witnesses: tuple[SupportTransportWitness, ...]
    static_unreachable_cells: frozenset[Cell]
    rejection_counts: tuple[tuple[str, int], ...]
    expanded_state_count: int
    bound_reached: bool


@dataclass(frozen=True)
class _QueueNode:
    item_cells: tuple[Cell, ...]
    carrier_cell: Cell
    action_refs: tuple[str, ...]
    push_count: int


def enumerate_support_transport_witnesses(
    value: SupportTransportRequest,
) -> SupportTransportResult:
    """Keep the first bounded witness for each of at most three declared orders."""

    distances = _reverse_push_distances(
        value.traversable_cells,
        value.terminal_cells,
        value.actions,
    )
    static_unreachable = frozenset(
        cell
        for cell in value.traversable_cells
        if all(cell not in terminal_distances for terminal_distances in distances)
    )
    orders = value.action_orders or (
        tuple(action.action_ref for action in value.actions),
    )
    witnesses: list[SupportTransportWitness] = []
    rejections: Counter[str] = Counter()
    total_expanded = 0
    bound_reached = False
    seen_routes: set[tuple[str, ...]] = set()
    for order in orders:
        if len(witnesses) >= value.maximum_witnesses:
            break
        witness, expanded, reached = _first_witness(
            value,
            order=order,
            distances=distances,
            static_unreachable=static_unreachable,
            rejections=rejections,
        )
        total_expanded += expanded
        bound_reached = bound_reached or reached
        if witness is not None and witness.action_refs not in seen_routes:
            witnesses.append(witness)
            seen_routes.add(witness.action_refs)
    return SupportTransportResult(
        witnesses=tuple(witnesses),
        static_unreachable_cells=static_unreachable,
        rejection_counts=tuple(sorted(rejections.items())),
        expanded_state_count=total_expanded,
        bound_reached=bound_reached,
    )


def canonical_support_state(
    traversable_cells: frozenset[Cell],
    carrier_cell: Cell,
    item_cells: tuple[Cell, ...],
    actions: tuple[CellAction, ...],
) -> tuple[tuple[Cell, ...], Cell]:
    """Canonicalize equivalent carrier positions by their reachable component."""

    reachable, _parents = _reachable_with_paths(
        traversable_cells,
        carrier_cell,
        frozenset(item_cells),
        actions,
    )
    anchor = min(reachable) if reachable else carrier_cell
    return tuple(sorted(item_cells)), anchor


def _first_witness(
    value: SupportTransportRequest,
    *,
    order: tuple[str, ...],
    distances: tuple[dict[Cell, int], ...],
    static_unreachable: frozenset[Cell],
    rejections: Counter[str],
) -> tuple[SupportTransportWitness | None, int, bool]:
    action_by_ref = {action.action_ref: action for action in value.actions}
    ordered_actions = tuple(action_by_ref[action_ref] for action_ref in order)
    initial_items = tuple(sorted(value.item_cells))
    initial_cost = _minimum_assignment_cost(initial_items, distances)
    if initial_cost is None:
        rejections["minimum_assignment_unreachable"] += 1
        return None, 0, False
    serial = count()
    queue: list[tuple[tuple[object, ...], int, _QueueNode]] = []
    root = _QueueNode(
        item_cells=initial_items,
        carrier_cell=value.carrier_cell,
        action_refs=(),
        push_count=0,
    )
    heappush(
        queue,
        (
            _queue_key(root, initial_cost),
            next(serial),
            root,
        ),
    )
    best_push_count: dict[tuple[tuple[Cell, ...], Cell], int] = {
        canonical_support_state(
            value.traversable_cells,
            value.carrier_cell,
            initial_items,
            value.actions,
        ): 0
    }
    expanded = 0
    terminals = frozenset(value.terminal_cells)
    while queue:
        if expanded >= value.maximum_expanded_states:
            return None, expanded, True
        _key, _serial, node = heappop(queue)
        expanded += 1
        if frozenset(node.item_cells) == terminals:
            return (
                SupportTransportWitness(
                    action_refs=node.action_refs,
                    push_count=node.push_count,
                    walk_count=len(node.action_refs) - node.push_count,
                    expanded_state_count=expanded,
                    final_item_cells=node.item_cells,
                    action_order=order,
                ),
                expanded,
                False,
            )
        occupied = frozenset(node.item_cells)
        reachable, parents = _reachable_with_paths(
            value.traversable_cells,
            node.carrier_cell,
            occupied,
            ordered_actions,
        )
        for item_cell in node.item_cells:
            for action in ordered_actions:
                dr, dc = action.delta
                support = item_cell[0] - dr, item_cell[1] - dc
                destination = item_cell[0] + dr, item_cell[1] + dc
                if support not in reachable:
                    rejections["support_unreachable"] += 1
                    continue
                if destination not in value.traversable_cells or destination in occupied:
                    rejections["destination_blocked"] += 1
                    continue
                walk = _path_to(support, node.carrier_cell, parents)
                route = (*node.action_refs, *walk, action.action_ref)
                if len(route) > value.maximum_primitive_actions:
                    rejections["primitive_action_bound"] += 1
                    continue
                next_items = tuple(
                    sorted(
                        destination if cell == item_cell else cell
                        for cell in node.item_cells
                    )
                )
                if any(
                    cell in static_unreachable and cell not in terminals
                    for cell in next_items
                ):
                    rejections["static_unreachable_cell"] += 1
                    continue
                if _has_locked_two_by_two(
                    value.traversable_cells,
                    frozenset(next_items),
                    terminals,
                ):
                    rejections["locked_two_by_two"] += 1
                    continue
                assignment_cost = _minimum_assignment_cost(next_items, distances)
                if assignment_cost is None:
                    rejections["minimum_assignment_unreachable"] += 1
                    continue
                next_node = _QueueNode(
                    item_cells=next_items,
                    carrier_cell=item_cell,
                    action_refs=route,
                    push_count=node.push_count + 1,
                )
                state_key = canonical_support_state(
                    value.traversable_cells,
                    next_node.carrier_cell,
                    next_items,
                    value.actions,
                )
                prior_push_count = best_push_count.get(state_key)
                if prior_push_count is not None and prior_push_count <= next_node.push_count:
                    rejections["transposition_not_improved"] += 1
                    continue
                best_push_count[state_key] = next_node.push_count
                heappush(
                    queue,
                    (
                        _queue_key(next_node, assignment_cost),
                        next(serial),
                        next_node,
                    ),
                )
    return None, expanded, False


def _queue_key(node: _QueueNode, assignment_cost: int) -> tuple[object, ...]:
    """A* push lower bound, then explicit resource and deterministic tie-breaks."""

    return (
        node.push_count + assignment_cost,
        assignment_cost,
        node.push_count,
        len(node.action_refs),
        node.action_refs,
        node.item_cells,
    )


def _reverse_push_distances(
    traversable_cells: frozenset[Cell],
    terminal_cells: tuple[Cell, ...],
    actions: tuple[CellAction, ...],
) -> tuple[dict[Cell, int], ...]:
    result: list[dict[Cell, int]] = []
    for terminal in terminal_cells:
        distances = {terminal: 0}
        queue = deque((terminal,))
        while queue:
            current = queue.popleft()
            for action in actions:
                dr, dc = action.delta
                previous = current[0] - dr, current[1] - dc
                support = previous[0] - dr, previous[1] - dc
                if (
                    previous in traversable_cells
                    and support in traversable_cells
                    and previous not in distances
                ):
                    distances[previous] = distances[current] + 1
                    queue.append(previous)
        result.append(distances)
    return tuple(result)


def _minimum_assignment_cost(
    item_cells: tuple[Cell, ...],
    distances: tuple[dict[Cell, int], ...],
) -> int | None:
    """Exact bounded minimum-cost perfect matching by terminal bitmask."""

    costs: dict[int, int] = {0: 0}
    for item in item_cells:
        next_costs: dict[int, int] = {}
        for mask, prior_cost in sorted(costs.items()):
            for terminal_index, terminal_distances in enumerate(distances):
                bit = 1 << terminal_index
                distance = terminal_distances.get(item)
                if mask & bit or distance is None:
                    continue
                next_mask = mask | bit
                total = prior_cost + distance
                existing = next_costs.get(next_mask)
                if existing is None or total < existing:
                    next_costs[next_mask] = total
        costs = next_costs
        if not costs:
            return None
    return costs.get((1 << len(distances)) - 1)


def _reachable_with_paths(
    traversable_cells: frozenset[Cell],
    start: Cell,
    occupied: frozenset[Cell],
    actions: tuple[CellAction, ...],
) -> tuple[frozenset[Cell], dict[Cell, tuple[Cell, str]]]:
    if start in occupied or start not in traversable_cells:
        return frozenset(), {}
    reached = {start}
    parents: dict[Cell, tuple[Cell, str]] = {}
    queue = deque((start,))
    while queue:
        row, col = queue.popleft()
        for action in actions:
            cell = row + action.delta_row, col + action.delta_col
            if (
                cell in traversable_cells
                and cell not in occupied
                and cell not in reached
            ):
                reached.add(cell)
                parents[cell] = ((row, col), action.action_ref)
                queue.append(cell)
    return frozenset(reached), parents


def _path_to(
    cell: Cell,
    start: Cell,
    parents: dict[Cell, tuple[Cell, str]],
) -> tuple[str, ...]:
    reversed_actions: list[str] = []
    current = cell
    while current != start:
        parent, action_ref = parents[current]
        reversed_actions.append(action_ref)
        current = parent
    return tuple(reversed(reversed_actions))


def _has_locked_two_by_two(
    traversable_cells: frozenset[Cell],
    item_cells: frozenset[Cell],
    terminal_cells: frozenset[Cell],
) -> bool:
    if all(cell in terminal_cells for cell in item_cells):
        return False
    rows = [row for row, _col in traversable_cells | item_cells]
    cols = [col for _row, col in traversable_cells | item_cells]
    if not rows or not cols:
        return False
    for row in range(min(rows) - 1, max(rows) + 1):
        for col in range(min(cols) - 1, max(cols) + 1):
            square = frozenset(
                ((row, col), (row + 1, col), (row, col + 1), (row + 1, col + 1))
            )
            if square.isdisjoint(item_cells):
                continue
            if all(cell not in traversable_cells or cell in item_cells for cell in square):
                if any(cell in item_cells and cell not in terminal_cells for cell in square):
                    return True
    return False
