"""Exact measurements for repeated symbol carriers and discrete cell changes.

The module intentionally assigns no source, output, dictionary, cursor, or goal
meaning.  It enumerates bounded structural orientations and action-conditioned
deltas; DRM alone decides which orientation has symbolic-editor meaning.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

from agents.yf_arc3_v5.capabilities.components import detect_components
from agents.yf_arc3_v5.capabilities.contracts import (
    ComponentDescription,
    ComponentExtractionInput,
    FrameGrid,
    InteractionProbeAgenda,
    InteractionProbeCandidate,
)
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest


MAX_TILE_EXTENT = 16
MAX_SYSTEM_MEASUREMENTS = 3
MAX_BAND_CELLS = 16
MAX_TRANSITION_MEASUREMENTS = 64
MAX_EDIT_COST_SEARCH_STATES = 4096
MAX_WORD_DERIVATIONS = 3
MAX_WORD_PARSE_STATES = 256


@dataclass(frozen=True, slots=True)
class SymbolTileMeasurement:
    bbox: tuple[int, int, int, int]
    carrier_value: int
    inner_value: int
    pattern_ref: str


@dataclass(frozen=True, slots=True)
class SymbolPairMeasurement:
    first: SymbolTileMeasurement
    second: SymbolTileMeasurement
    bridge_bbox: tuple[int, int, int, int]
    bridge_value: int
    first_word: tuple[SymbolTileMeasurement, ...] = ()
    second_word: tuple[SymbolTileMeasurement, ...] = ()


@dataclass(frozen=True, slots=True)
class SymbolBandMeasurement:
    ordinal: int
    bbox: tuple[int, int, int, int]
    carrier_value: int
    cells: tuple[SymbolTileMeasurement, ...]


@dataclass(frozen=True, slots=True)
class SymbolTileSystemMeasurement:
    measurement_ref: str
    tile_extent: int
    inner_extent: int
    inner_value: int
    carrier_values: tuple[int, int]
    pairs: tuple[SymbolPairMeasurement, ...]
    bands: tuple[SymbolBandMeasurement, ...]
    cursor_band_ordinal: int | None
    cursor_cell_ordinal: int | None


@dataclass(frozen=True, slots=True)
class SymbolTileTransitionMeasurement:
    measurement_ref: str
    system_ref: str
    action_ref: str
    before_cursor_band_ordinal: int | None
    before_cursor_cell_ordinal: int | None
    after_cursor_band_ordinal: int | None
    after_cursor_cell_ordinal: int | None
    changed_band_ordinals: tuple[int, ...]
    changed_cell_ordinals: tuple[int, ...]
    before_pattern_refs: tuple[str, ...]
    after_pattern_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SymbolTileEditCostMeasurement:
    mutation_cost: int | None
    focus_route_cost: int | None
    total_cost: int | None
    correct_unit_count: int
    focus_transit_correct_unit_count: int | None
    search_state_count: int
    search_complete: bool
    bound_reached: bool
    first_action_refs: tuple[str, ...]


def measure_symbol_tile_systems(
    frame: FrameGrid,
) -> tuple[SymbolTileSystemMeasurement, ...]:
    """Enumerate bounded carrier/mask/band/bridge descriptions exactly."""

    components = detect_components(
        ComponentExtractionInput(frame=frame, connectivity=4)
    ).components
    square_groups: dict[tuple[int, int], list[ComponentDescription]] = {}
    for component in components:
        extent = component.bbox.height
        if (
            component.touches_frame_boundary
            or component.bbox.width % extent
            or component.bbox.width // extent > MAX_BAND_CELLS
            or extent < 3
            or extent > MAX_TILE_EXTENT
            or not all(
                _border_is_value(
                    frame,
                    (component.bbox.top, component.bbox.left + index * extent,
                     component.bbox.bottom, component.bbox.left + (index + 1) * extent - 1),
                    component.value,
                )
                for index in range(component.bbox.width // extent)
            )
        ):
            continue
        square_groups.setdefault((extent, component.value), []).append(component)

    described: list[SymbolTileSystemMeasurement] = []
    extents = sorted({extent for extent, _value in square_groups})
    for extent in extents:
        value_groups = {
            value: tuple(sorted(items, key=_component_order))
            for (group_extent, value), items in sorted(square_groups.items())
            if group_extent == extent and len(items) >= 3
        }
        for first_value, second_value in combinations(sorted(value_groups), 2):
            first_components = value_groups[first_value]
            second_components = value_groups[second_value]
            inner_values = _common_inner_values(
                frame,
                (*first_components, *second_components),
                carrier_values=(first_value, second_value),
            )
            for inner_value in inner_values:
                pairs = _bridge_pairs(
                    frame,
                    components=components,
                    first_components=first_components,
                    second_components=second_components,
                    inner_value=inner_value,
                )
                if len(pairs) < 3:
                    continue
                bands = _symbol_bands(
                    frame,
                    components=components,
                    extent=extent,
                    carrier_values=(first_value, second_value),
                    inner_value=inner_value,
                )
                if len(bands) < 2:
                    continue
                cursor_band, cursor_cell = _cursor_position(
                    components=components,
                    bands=bands,
                    excluded_values=(first_value, second_value, inner_value),
                    extent=extent,
                )
                structural_key = (
                    extent,
                    inner_value,
                    (first_value, second_value),
                    tuple(
                        (
                            pair.first.bbox,
                            pair.second.bbox,
                            pair.bridge_bbox,
                            pair.bridge_value,
                        )
                        for pair in pairs
                    ),
                    tuple((band.bbox, band.carrier_value) for band in bands),
                )
                if any(len(pair.first_word) > 1 or len(pair.second_word) > 1 for pair in pairs):
                    structural_key += (
                        tuple((tuple(cell.bbox for cell in pair.first_word),
                               tuple(cell.bbox for cell in pair.second_word)) for pair in pairs),
                    )
                described.append(
                    SymbolTileSystemMeasurement(
                        measurement_ref=(
                            "measurement.symbol_tile_system."
                            + stable_digest(structural_key)[:16]
                        ),
                        tile_extent=extent,
                        inner_extent=extent - 2,
                        inner_value=inner_value,
                        carrier_values=(first_value, second_value),
                        pairs=pairs,
                        bands=bands,
                        cursor_band_ordinal=cursor_band,
                        cursor_cell_ordinal=cursor_cell,
                    )
                )
    unique = {item.measurement_ref: item for item in described}
    ordered = sorted(
        unique.values(),
        key=lambda item: (
            -len(item.pairs),
            -sum(len(band.cells) for band in item.bands),
            item.measurement_ref,
        ),
    )
    return tuple(ordered[:MAX_SYSTEM_MEASUREMENTS])


def measure_symbol_tile_transitions(
    *,
    before: FrameGrid,
    after: FrameGrid,
    action_ref: str,
) -> tuple[SymbolTileTransitionMeasurement, ...]:
    """Compare structurally stable systems without interpreting the delta."""

    before_by_ref = {
        item.measurement_ref: item for item in measure_symbol_tile_systems(before)
    }
    after_by_ref = {
        item.measurement_ref: item for item in measure_symbol_tile_systems(after)
    }
    measured: list[SymbolTileTransitionMeasurement] = []
    for system_ref in sorted(set(before_by_ref).intersection(after_by_ref)):
        first = before_by_ref[system_ref]
        second = after_by_ref[system_ref]
        before_bands = {band.ordinal: band for band in first.bands}
        after_bands = {band.ordinal: band for band in second.bands}
        changed_bands: list[int] = []
        changed_cells: list[int] = []
        before_patterns: list[str] = []
        after_patterns: list[str] = []
        for band_ordinal in sorted(set(before_bands).intersection(after_bands)):
            before_band = before_bands[band_ordinal]
            after_band = after_bands[band_ordinal]
            if len(before_band.cells) != len(after_band.cells):
                continue
            for cell_ordinal, (before_cell, after_cell) in enumerate(
                zip(before_band.cells, after_band.cells), start=1
            ):
                if before_cell.pattern_ref == after_cell.pattern_ref:
                    continue
                changed_bands.append(band_ordinal)
                changed_cells.append(cell_ordinal)
                before_patterns.append(before_cell.pattern_ref)
                after_patterns.append(after_cell.pattern_ref)
        key = (
            system_ref,
            action_ref,
            first.cursor_band_ordinal,
            first.cursor_cell_ordinal,
            second.cursor_band_ordinal,
            second.cursor_cell_ordinal,
            tuple(changed_bands),
            tuple(changed_cells),
            tuple(before_patterns),
            tuple(after_patterns),
        )
        measured.append(
            SymbolTileTransitionMeasurement(
                measurement_ref=(
                    "measurement.symbol_tile_transition."
                    + stable_digest(key)[:16]
                ),
                system_ref=system_ref,
                action_ref=action_ref,
                before_cursor_band_ordinal=first.cursor_band_ordinal,
                before_cursor_cell_ordinal=first.cursor_cell_ordinal,
                after_cursor_band_ordinal=second.cursor_band_ordinal,
                after_cursor_cell_ordinal=second.cursor_cell_ordinal,
                changed_band_ordinals=tuple(changed_bands),
                changed_cell_ordinals=tuple(changed_cells),
                before_pattern_refs=tuple(before_patterns),
                after_pattern_refs=tuple(after_patterns),
            )
        )
    return tuple(measured[:MAX_TRANSITION_MEASUREMENTS])


def enumerate_symbol_tile_actions(
    *,
    frame: FrameGrid,
    systems: tuple[SymbolTileSystemMeasurement, ...],
    transitions: tuple[SymbolTileTransitionMeasurement, ...],
    available_action_refs: tuple[str, ...],
    direction_vectors: Mapping[str, tuple[int, int]],
    recent_local_action_ref: str | None,
    recent_local_band_ordinal: int | None,
    recent_local_cell_ordinal: int | None,
    observation_ref: str | None = None,
) -> InteractionProbeAgenda:
    """Enumerate action alternatives for every exact structural orientation."""

    current_by_ref = {item.measurement_ref: item for item in systems}
    candidates: list[InteractionProbeCandidate] = []
    facts_by_ref: dict[str, FrozenMap] = {}
    for system_ref in sorted(current_by_ref):
        system = current_by_ref[system_ref]
        relations = _relation_orientations(system)
        for relation_ordinal, relation in enumerate(relations, start=1):
            first_band, second_band, projected_patterns, contribution_spans, word_relation = relation
            current_patterns = tuple(cell.pattern_ref for cell in second_band.cells)
            if len(projected_patterns) != len(current_patterns):
                continue
            cursor_cell = (
                system.cursor_cell_ordinal
                if system.cursor_band_ordinal == second_band.ordinal
                else None
            )
            if cursor_cell is None or cursor_cell < 1 or cursor_cell > len(current_patterns):
                continue
            cell_index = cursor_cell - 1
            current_equal = current_patterns[cell_index] == projected_patterns[cell_index]
            remaining = sum(
                current != projected
                for current, projected in zip(current_patterns, projected_patterns)
            )
            system_transitions = tuple(
                item for item in transitions if item.system_ref == system_ref
            )
            edit_cost = _measure_symbol_tile_edit_cost(
                current_patterns=current_patterns,
                projected_patterns=projected_patterns,
                cursor_cell=cursor_cell,
                band_ordinal=second_band.ordinal,
                transitions=system_transitions,
                available_action_refs=available_action_refs,
                direction_vectors=direction_vectors,
            )
            for action_ref in available_action_refs:
                vector = direction_vectors.get(action_ref)
                if vector is None:
                    continue
                delta_row, delta_col = vector
                action_measurements = tuple(
                    item for item in system_transitions if item.action_ref == action_ref
                )
                local_measurements = tuple(
                    item
                    for item in action_measurements
                    if len(item.changed_band_ordinals) == 1
                    and len(item.changed_cell_ordinals) == 1
                    and item.changed_band_ordinals[0] == second_band.ordinal
                    and item.changed_cell_ordinals[0]
                    == item.before_cursor_cell_ordinal
                    and item.before_cursor_cell_ordinal
                    == item.after_cursor_cell_ordinal
                )
                cursor_measurements = tuple(
                    item
                    for item in action_measurements
                    if not item.changed_band_ordinals
                    and item.before_cursor_band_ordinal == second_band.ordinal
                    and item.after_cursor_band_ordinal == second_band.ordinal
                    and item.before_cursor_cell_ordinal is not None
                    and item.after_cursor_cell_ordinal is not None
                    and item.before_cursor_cell_ordinal
                    != item.after_cursor_cell_ordinal
                )
                edge_after = {
                    item.after_pattern_refs[0]
                    for item in local_measurements
                    if item.before_pattern_refs[0] == current_patterns[cell_index]
                }
                exact_edge_after = next(iter(edge_after)) if len(edge_after) == 1 else ""
                cursor_deltas = {
                    int(item.after_cursor_cell_ordinal)
                    - int(item.before_cursor_cell_ordinal)
                    for item in cursor_measurements
                }
                exact_cursor_delta = (
                    next(iter(cursor_deltas)) if len(cursor_deltas) == 1 else 0
                )
                next_cursor = (
                    ((cell_index + exact_cursor_delta) % len(current_patterns)) + 1
                    if exact_cursor_delta
                    else None
                )
                next_cursor_differs = bool(
                    next_cursor is not None
                    and current_patterns[next_cursor - 1]
                    != projected_patterns[next_cursor - 1]
                )
                another_known_nonprojection_edge = any(
                    item.action_ref != action_ref
                    and len(item.changed_band_ordinals) == 1
                    and item.changed_band_ordinals[0] == second_band.ordinal
                    and item.before_pattern_refs
                    and item.before_pattern_refs[0] == current_patterns[cell_index]
                    and item.after_pattern_refs[0] != projected_patterns[cell_index]
                    for item in system_transitions
                )
                same_recent_cell = bool(
                    recent_local_action_ref == action_ref
                    and recent_local_band_ordinal == second_band.ordinal
                    and recent_local_cell_ordinal == cursor_cell
                )
                candidate_ref = (
                    f"probe:symbol_tile:{system_ref}:o{relation_ordinal}:"
                    f"b{second_band.ordinal}:c{cursor_cell}:{action_ref}"
                )
                candidate = InteractionProbeCandidate(
                    candidate_ref=candidate_ref,
                    action_ref=action_ref,
                    action_data=FrozenMap(),
                    point=None,
                    basis_refs=(system_ref,),
                )
                candidates.append(candidate)
                facts_by_ref[candidate_ref] = FrozenMap(
                    {
                        "alternative_ref": candidate_ref,
                        "candidate_ref": candidate_ref,
                        "interaction_measurement_kind": "discrete",
                        "action_ref": action_ref,
                        "current_action_available": True,
                        "symbol_tile_system_measurement": True,
                        "symbol_tile_system_ref": system_ref,
                        "observation_ref": observation_ref or ("measurement.frame." + stable_digest(frame.rows)[:16]),
                        "relation_first_band_bbox": first_band.bbox,
                        "relation_second_band_bbox": second_band.bbox,
                        "tile_extent": system.tile_extent,
                        "inner_extent": system.inner_extent,
                        "bridge_pair_count": len(system.pairs),
                        "first_band_cell_count": len(first_band.cells),
                        "second_band_cell_count": len(second_band.cells),
                        "relation_projection_complete": True,
                        "relation_word_transduction": word_relation,
                        "relation_derivation_count": sum(
                            item[0].ordinal == first_band.ordinal
                            and item[1].ordinal == second_band.ordinal
                            for item in relations
                        ),
                        "relation_contribution_spans": contribution_spans,
                        "relation_source_pattern_refs": tuple(cell.pattern_ref for cell in first_band.cells),
                        "relation_projected_pattern_refs": projected_patterns,
                        "relation_bridge_bboxes": tuple(pair.bridge_bbox for pair in system.pairs),
                        "cursor_on_second_band": True,
                        "cursor_cell_ordinal": cursor_cell,
                        "current_cell_projection_equal": current_equal,
                        "remaining_projection_difference_count": remaining,
                        "symbol_tile_edit_cost_measurement": True,
                        "edit_mutation_cost": edit_cost.mutation_cost,
                        "edit_focus_route_cost": edit_cost.focus_route_cost,
                        "edit_total_cost": edit_cost.total_cost,
                        "edit_correct_unit_count": edit_cost.correct_unit_count,
                        "edit_focus_transit_correct_unit_count": (
                            edit_cost.focus_transit_correct_unit_count
                        ),
                        "edit_cost_search_state_count": edit_cost.search_state_count,
                        "edit_cost_search_complete": edit_cost.search_complete,
                        "edit_cost_bound_reached": edit_cost.bound_reached,
                        "action_starts_minimum_edit_cost_route": (
                            action_ref in edit_cost.first_action_refs
                        ),
                        "supported_correct_units_preserved": True,
                        "action_delta_row": delta_row,
                        "action_delta_col": delta_col,
                        "action_axis_parallel_to_band": delta_row == 0 and delta_col != 0,
                        "action_axis_orthogonal_to_band": delta_row != 0 and delta_col == 0,
                        "positive_axis_direction": delta_row > 0 or delta_col > 0,
                        "local_pattern_change_support_count": len(local_measurements),
                        "cursor_change_support_count": len(cursor_measurements),
                        "edge_from_current_observed": bool(exact_edge_after),
                        "edge_reaches_projection_pattern": bool(
                            exact_edge_after
                            and exact_edge_after == projected_patterns[cell_index]
                        ),
                        "another_action_has_known_nonprojection_edge_from_current": (
                            another_known_nonprojection_edge
                        ),
                        "action_unobserved": not action_measurements,
                        "same_as_recent_local_cell_action": same_recent_cell,
                        "cursor_action_moves_to_next_difference": next_cursor_differs,
                        "output_projection_complete_after_current_action": bool(
                            remaining == 1
                            and exact_edge_after == projected_patterns[cell_index]
                        ),
                    }
                )
    candidates.sort(key=lambda item: item.candidate_ref)
    refs = tuple(item.candidate_ref for item in candidates)
    return InteractionProbeAgenda(
        candidates=tuple(candidates),
        alternative_refs=refs,
        alternative_facts=FrozenMap.from_frozen_items(
            {ref: facts_by_ref[ref] for ref in refs}
        ),
        context_facts=FrozenMap(
            {
                "symbol_tile_system_measurement_count": len(systems),
                "symbol_tile_action_candidate_count": len(candidates),
            }
        ),
        has_candidates=bool(candidates),
    )


def _measure_symbol_tile_edit_cost(
    *,
    current_patterns: tuple[str, ...],
    projected_patterns: tuple[str, ...],
    cursor_cell: int,
    band_ordinal: int,
    transitions: tuple[SymbolTileTransitionMeasurement, ...],
    available_action_refs: tuple[str, ...],
    direction_vectors: Mapping[str, tuple[int, int]],
) -> SymbolTileEditCostMeasurement:
    """Measure a bounded edit route without assigning symbolic meaning to it."""

    mutation_actions = tuple(
        sorted(
            action_ref
            for action_ref in available_action_refs
            if (vector := direction_vectors.get(action_ref)) is not None
            and vector[0] != 0
            and vector[1] == 0
        )
    )
    focus_actions = tuple(
        sorted(
            action_ref
            for action_ref in available_action_refs
            if (vector := direction_vectors.get(action_ref)) is not None
            and vector[0] == 0
            and vector[1] != 0
        )
    )
    local_successors: dict[tuple[str, str], set[str]] = {}
    focus_deltas: dict[str, set[int]] = {}
    unit_count = len(current_patterns)
    for item in transitions:
        if (
            len(item.changed_band_ordinals) == 1
            and len(item.changed_cell_ordinals) == 1
            and item.changed_band_ordinals[0] == band_ordinal
            and item.before_cursor_cell_ordinal == item.after_cursor_cell_ordinal
            and len(item.before_pattern_refs) == 1
            and len(item.after_pattern_refs) == 1
        ):
            local_successors.setdefault(
                (item.action_ref, item.before_pattern_refs[0]), set()
            ).add(item.after_pattern_refs[0])
        if (
            not item.changed_band_ordinals
            and item.before_cursor_band_ordinal == band_ordinal
            and item.after_cursor_band_ordinal == band_ordinal
            and item.before_cursor_cell_ordinal is not None
            and item.after_cursor_cell_ordinal is not None
            and unit_count > 0
        ):
            delta = (
                int(item.after_cursor_cell_ordinal)
                - int(item.before_cursor_cell_ordinal)
            ) % unit_count
            if delta:
                focus_deltas.setdefault(item.action_ref, set()).add(delta)

    mutation_cost = 0
    mutation_complete = True
    mutation_bound_reached = False
    mutation_states = 0
    current_first_actions: tuple[str, ...] = ()
    for ordinal, (current, projected) in enumerate(
        zip(current_patterns, projected_patterns), start=1
    ):
        cost, first_actions, complete, bound_reached, states = (
            _bounded_pattern_route_cost(
                current=current,
                projected=projected,
                action_refs=mutation_actions,
                successors=local_successors,
            )
        )
        mutation_states += states
        mutation_bound_reached = mutation_bound_reached or bound_reached
        if cost is None or not complete:
            mutation_complete = False
        else:
            mutation_cost += cost
        if ordinal == cursor_cell and current != projected:
            current_first_actions = first_actions if complete else ()

    required_cells = frozenset(
        ordinal
        for ordinal, (current, projected) in enumerate(
            zip(current_patterns, projected_patterns), start=1
        )
        if current != projected
    )
    (
        focus_cost,
        focus_first_actions,
        focus_transit_correct,
        focus_complete,
        focus_bound_reached,
        focus_states,
    ) = _bounded_focus_visit_cost(
        cursor_cell=cursor_cell,
        unit_count=unit_count,
        required_cells=required_cells,
        action_refs=focus_actions,
        deltas=focus_deltas,
    )
    search_complete = bool(
        mutation_complete
        and focus_complete
        and not mutation_bound_reached
        and not focus_bound_reached
    )
    total_cost = (
        mutation_cost + int(focus_cost)
        if search_complete and focus_cost is not None
        else None
    )
    first_actions = (
        current_first_actions
        if cursor_cell in required_cells
        else focus_first_actions
    )
    return SymbolTileEditCostMeasurement(
        mutation_cost=mutation_cost if mutation_complete else None,
        focus_route_cost=focus_cost if focus_complete else None,
        total_cost=total_cost,
        correct_unit_count=unit_count - len(required_cells),
        focus_transit_correct_unit_count=(
            focus_transit_correct if focus_complete else None
        ),
        search_state_count=mutation_states + focus_states,
        search_complete=search_complete,
        bound_reached=mutation_bound_reached or focus_bound_reached,
        first_action_refs=first_actions if search_complete else (),
    )


def _bounded_pattern_route_cost(
    *,
    current: str,
    projected: str,
    action_refs: tuple[str, ...],
    successors: Mapping[tuple[str, str], set[str]],
) -> tuple[int | None, tuple[str, ...], bool, bool, int]:
    if current == projected:
        return 0, (), True, False, 1
    queue = deque([(current, 0, "")])
    best_depth_by_pattern = {current: 0}
    best_cost: int | None = None
    first_actions: set[str] = set()
    incomplete_before_best = False
    state_count = 0
    while queue:
        pattern, depth, first_action = queue.popleft()
        state_count += 1
        if state_count > MAX_EDIT_COST_SEARCH_STATES:
            return None, (), False, True, state_count
        if best_cost is not None and depth >= best_cost:
            continue
        for action_ref in action_refs:
            next_patterns = successors.get((action_ref, pattern), set())
            if len(next_patterns) != 1:
                incomplete_before_best = True
                continue
            next_pattern = next(iter(next_patterns))
            next_depth = depth + 1
            route_first = first_action or action_ref
            if next_pattern == projected:
                if best_cost is None or next_depth < best_cost:
                    best_cost = next_depth
                    first_actions = {route_first}
                elif next_depth == best_cost:
                    first_actions.add(route_first)
                continue
            prior_depth = best_depth_by_pattern.get(next_pattern)
            if prior_depth is None or next_depth < prior_depth:
                best_depth_by_pattern[next_pattern] = next_depth
                queue.append((next_pattern, next_depth, route_first))
    if best_cost is None:
        return None, (), False, False, state_count
    complete = best_cost == 1 or not incomplete_before_best
    return best_cost, tuple(sorted(first_actions)), complete, False, state_count


def _bounded_focus_visit_cost(
    *,
    cursor_cell: int,
    unit_count: int,
    required_cells: frozenset[int],
    action_refs: tuple[str, ...],
    deltas: Mapping[str, set[int]],
) -> tuple[int | None, tuple[str, ...], int | None, bool, bool, int]:
    pending = required_cells - {cursor_cell}
    if not pending:
        return 0, (), 0, True, False, 1
    exact_deltas = {
        action_ref: next(iter(values))
        for action_ref in action_refs
        if len(values := deltas.get(action_ref, set())) == 1
    }
    start_mask = 1 << (cursor_cell - 1) if cursor_cell in required_cells else 0
    required_mask = sum(1 << (cell - 1) for cell in required_cells)
    queue = deque([(cursor_cell, start_mask, 0, "", 0)])
    seen = {(cursor_cell, start_mask)}
    best_cost: int | None = None
    best_transit: int | None = None
    first_actions: set[str] = set()
    state_count = 0
    while queue:
        cell, visited_mask, cost, first_action, transit_correct = queue.popleft()
        state_count += 1
        if state_count > MAX_EDIT_COST_SEARCH_STATES:
            return None, (), None, False, True, state_count
        if best_cost is not None and cost >= best_cost:
            continue
        for action_ref in sorted(exact_deltas):
            next_cell = ((cell - 1 + exact_deltas[action_ref]) % unit_count) + 1
            next_mask = visited_mask
            if next_cell in required_cells:
                next_mask |= 1 << (next_cell - 1)
            next_cost = cost + 1
            route_first = first_action or action_ref
            next_transit = transit_correct + int(next_cell not in required_cells)
            if next_mask & required_mask == required_mask:
                if best_cost is None or next_cost < best_cost:
                    best_cost = next_cost
                    best_transit = next_transit
                    first_actions = {route_first}
                elif next_cost == best_cost:
                    best_transit = min(int(best_transit), next_transit)
                    first_actions.add(route_first)
                continue
            state = (next_cell, next_mask)
            if state not in seen:
                seen.add(state)
                queue.append(
                    (next_cell, next_mask, next_cost, route_first, next_transit)
                )
    if best_cost is None:
        return None, (), None, False, False, state_count
    lower_bound = len(pending)
    closed_navigation = len(exact_deltas) == len(action_refs)
    complete = best_cost == lower_bound or closed_navigation
    return (
        best_cost,
        tuple(sorted(first_actions)),
        best_transit,
        complete,
        False,
        state_count,
    )


def _common_inner_values(
    frame: FrameGrid,
    components: Sequence[ComponentDescription],
    *,
    carrier_values: tuple[int, int],
) -> tuple[int, ...]:
    common: set[int] | None = None
    for component in components:
        bbox = _bbox_tuple(component)
        values = {
            frame.rows[row][col]
            for row in range(bbox[0] + 1, bbox[2])
            for col in range(bbox[1] + 1, bbox[3])
            if frame.rows[row][col] not in carrier_values
        }
        if len(values) != 1:
            continue
        common = values if common is None else common.intersection(values)
    return tuple(sorted(common or ()))


def _bridge_pairs(
    frame: FrameGrid,
    *,
    components: Sequence[ComponentDescription],
    first_components: Sequence[ComponentDescription],
    second_components: Sequence[ComponentDescription],
    inner_value: int,
) -> tuple[SymbolPairMeasurement, ...]:
    measured: list[SymbolPairMeasurement] = []
    for first in first_components:
        for second in second_components:
            if first.bbox.top != second.bbox.top or first.bbox.bottom != second.bbox.bottom:
                continue
            left, right = (
                (first, second) if first.bbox.left < second.bbox.left else (second, first)
            )
            gap = right.bbox.left - left.bbox.right - 1
            if gap < 1 or gap > first.bbox.height:
                continue
            center_row = (left.bbox.top + left.bbox.bottom) // 2
            bridge_values = tuple(
                frame.rows[center_row][col]
                for col in range(left.bbox.right + 1, right.bbox.left)
            )
            if not bridge_values or len(set(bridge_values)) != 1:
                continue
            bridge_value = bridge_values[0]
            if bridge_value in {first.value, second.value, inner_value}:
                continue
            bridge_bbox = (
                center_row,
                left.bbox.right + 1,
                center_row,
                right.bbox.left - 1,
            )
            if not any(
                component.value == bridge_value
                and _bbox_tuple(component) == bridge_bbox
                and not component.touches_frame_boundary
                for component in components
            ):
                continue
            first_word = _component_word(frame, first, inner_value)
            second_word = _component_word(frame, second, inner_value)
            if not first_word or not second_word:
                continue
            measured.append(
                SymbolPairMeasurement(
                    first=first_word[0],
                    second=second_word[0],
                    bridge_bbox=bridge_bbox,
                    bridge_value=bridge_value,
                    first_word=first_word,
                    second_word=second_word,
                )
            )
    unique = {
        (item.first.bbox, item.second.bbox, item.bridge_bbox): item
        for item in measured
    }
    return tuple(
        unique[key]
        for key in sorted(unique, key=lambda item: (item[0][0], item[0][1], item[1][1]))
    )


def _symbol_bands(
    frame: FrameGrid,
    *,
    components: Sequence[ComponentDescription],
    extent: int,
    carrier_values: tuple[int, int],
    inner_value: int,
) -> tuple[SymbolBandMeasurement, ...]:
    raw: list[tuple[ComponentDescription, tuple[SymbolTileMeasurement, ...]]] = []
    for component in components:
        if (
            component.value not in carrier_values
            or component.bbox.height != extent
            or component.bbox.width % extent
            or component.bbox.width // extent < 2
            or component.bbox.width // extent > MAX_BAND_CELLS
        ):
            continue
        cells: list[SymbolTileMeasurement] = []
        for cell_index in range(component.bbox.width // extent):
            bbox = (
                component.bbox.top,
                component.bbox.left + cell_index * extent,
                component.bbox.bottom,
                component.bbox.left + (cell_index + 1) * extent - 1,
            )
            if not _border_is_value(frame, bbox, component.value):
                cells = []
                break
            values = {
                frame.rows[row][col]
                for row in range(bbox[0] + 1, bbox[2])
                for col in range(bbox[1] + 1, bbox[3])
                if frame.rows[row][col] != component.value
            }
            if values != {inner_value}:
                cells = []
                break
            cells.append(
                SymbolTileMeasurement(
                    bbox=bbox,
                    carrier_value=component.value,
                    inner_value=inner_value,
                    pattern_ref=_pattern_ref(frame, bbox, inner_value),
                )
            )
        if cells:
            raw.append((component, tuple(cells)))
    raw.sort(key=lambda item: _component_order(item[0]))
    return tuple(
        SymbolBandMeasurement(
            ordinal=index,
            bbox=_bbox_tuple(component),
            carrier_value=component.value,
            cells=cells,
        )
        for index, (component, cells) in enumerate(raw, start=1)
    )


def _cursor_position(
    *,
    components: Sequence[ComponentDescription],
    bands: tuple[SymbolBandMeasurement, ...],
    excluded_values: tuple[int, ...],
    extent: int,
) -> tuple[int | None, int | None]:
    candidates: set[tuple[int, int]] = set()
    for band in bands:
        above = tuple(
            component
            for component in components
            if component.value not in excluded_values
            and not component.touches_frame_boundary
            and component.bbox.bottom < band.bbox[0]
            and 0 <= band.bbox[0] - component.bbox.bottom - 1 <= extent
        )
        below = tuple(
            component
            for component in components
            if component.value not in excluded_values
            and not component.touches_frame_boundary
            and component.bbox.top > band.bbox[2]
            and 0 <= component.bbox.top - band.bbox[2] - 1 <= extent
        )
        for first in above:
            for second in below:
                if first.value != second.value:
                    continue
                overlap_left = max(first.bbox.left, second.bbox.left)
                overlap_right = min(first.bbox.right, second.bbox.right)
                if overlap_left > overlap_right:
                    continue
                center_col = (overlap_left + overlap_right) // 2
                if not (band.bbox[1] <= center_col <= band.bbox[3]):
                    continue
                cell = ((center_col - band.bbox[1]) // extent) + 1
                if 1 <= cell <= len(band.cells):
                    candidates.add((band.ordinal, cell))
    if len(candidates) != 1:
        return None, None
    return next(iter(candidates))


def _relation_orientations(
    system: SymbolTileSystemMeasurement,
) -> tuple[tuple[SymbolBandMeasurement, SymbolBandMeasurement, tuple[str, ...],
                 tuple[tuple[int, int, int, int, int], ...], bool], ...]:
    measured = []
    for first_value, second_value, reverse in (
        (system.carrier_values[0], system.carrier_values[1], False),
        (system.carrier_values[1], system.carrier_values[0], True),
    ):
        words = tuple(
            (tuple(cell.pattern_ref for cell in
                   ((pair.second_word or (pair.second,)) if reverse else (pair.first_word or (pair.first,)))),
             tuple(cell.pattern_ref for cell in
                   ((pair.first_word or (pair.first,)) if reverse else (pair.second_word or (pair.second,)))))
            for pair in system.pairs
        )
        word_relation = any(len(first) != 1 or len(second) != 1 for first, second in words)
        for first_band in system.bands:
            if first_band.carrier_value != first_value:
                continue
            for second_band in system.bands:
                if second_band.carrier_value != second_value:
                    continue
                if second_band.ordinal != system.cursor_band_ordinal:
                    continue
                derivations = _bounded_word_derivations(
                    tuple(cell.pattern_ref for cell in first_band.cells), words,
                    output_length=len(second_band.cells),
                )
                for patterns, spans in derivations:
                    measured.append((first_band, second_band, patterns, spans, word_relation))
    return tuple(measured)


def _bounded_word_derivations(
    source: tuple[str, ...],
    relations: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...],
    *, output_length: int,
) -> tuple[tuple[tuple[str, ...], tuple[tuple[int, int, int, int, int], ...]], ...]:
    """Exact complete covers, retaining provenance; exceed a bound => no witness."""
    pending = deque([(0, (), ())])
    complete = []
    visited_count = 0
    while pending:
        offset, patterns, spans = pending.popleft()
        visited_count += 1
        if visited_count > MAX_WORD_PARSE_STATES:
            return ()
        if offset == len(source):
            if len(patterns) == output_length:
                complete.append((patterns, spans))
                if len(complete) > MAX_WORD_DERIVATIONS:
                    return ()
            continue
        for ordinal, (first, second) in enumerate(relations, start=1):
            if not first or not second or source[offset:offset + len(first)] != first:
                continue
            if len(patterns) + len(second) > output_length:
                continue
            pending.append((offset + len(first), patterns + second,
                spans + ((offset + 1, offset + len(first), len(patterns) + 1,
                          len(patterns) + len(second), ordinal),)))
            if len(pending) + visited_count > MAX_WORD_PARSE_STATES:
                return ()
    return tuple(complete)


def _component_word(
    frame: FrameGrid, component: ComponentDescription, inner_value: int,
) -> tuple[SymbolTileMeasurement, ...]:
    extent = component.bbox.height
    if component.bbox.width % extent or component.bbox.width // extent > MAX_BAND_CELLS:
        return ()
    cells = []
    for index in range(component.bbox.width // extent):
        bbox = (component.bbox.top, component.bbox.left + index * extent,
                component.bbox.bottom, component.bbox.left + (index + 1) * extent - 1)
        if not _border_is_value(frame, bbox, component.value):
            return ()
        values = {frame.rows[row][col]
                  for row in range(bbox[0] + 1, bbox[2])
                  for col in range(bbox[1] + 1, bbox[3])
                  if frame.rows[row][col] != component.value}
        if values != {inner_value}:
            return ()
        cells.append(SymbolTileMeasurement(bbox=bbox, carrier_value=component.value,
            inner_value=inner_value, pattern_ref=_pattern_ref(frame, bbox, inner_value)))
    return tuple(cells)


def _tile_measurement(
    frame: FrameGrid,
    component: ComponentDescription,
    inner_value: int,
) -> SymbolTileMeasurement:
    bbox = _bbox_tuple(component)
    return SymbolTileMeasurement(
        bbox=bbox,
        carrier_value=component.value,
        inner_value=inner_value,
        pattern_ref=_pattern_ref(frame, bbox, inner_value),
    )


def _pattern_ref(
    frame: FrameGrid,
    bbox: tuple[int, int, int, int],
    inner_value: int,
) -> str:
    mask = tuple(
        tuple(
            frame.rows[row][col] == inner_value
            for col in range(bbox[1] + 1, bbox[3])
        )
        for row in range(bbox[0] + 1, bbox[2])
    )
    rotations = [mask]
    for _ in range(3):
        prior = rotations[-1]
        rotations.append(tuple(tuple(row) for row in zip(*prior[::-1])))
    canonical = min(rotations)
    return "measurement.symbol_pattern." + stable_digest(canonical)[:16]


def _border_is_value(
    frame: FrameGrid,
    bbox: tuple[int, int, int, int],
    value: int,
) -> bool:
    top, left, bottom, right = bbox
    return all(frame.rows[top][col] == value for col in range(left, right + 1)) and all(
        frame.rows[bottom][col] == value for col in range(left, right + 1)
    ) and all(frame.rows[row][left] == value for row in range(top, bottom + 1)) and all(
        frame.rows[row][right] == value for row in range(top, bottom + 1)
    )


def _bbox_tuple(component: ComponentDescription) -> tuple[int, int, int, int]:
    bbox = component.bbox
    return (bbox.top, bbox.left, bbox.bottom, bbox.right)


def _component_order(component: ComponentDescription) -> tuple[int, int, int, int, int]:
    return (
        component.bbox.top,
        component.bbox.left,
        component.bbox.bottom,
        component.bbox.right,
        component.value,
    )
