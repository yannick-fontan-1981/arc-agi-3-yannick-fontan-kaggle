"""Bounded finite viability propagation under an already-declared terminal set."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import combinations
from math import comb

from agents.yf_arc3_v5.capabilities.contracts import (
    EliminationStateViability,
    EliminationViabilityInput,
    EliminationViabilityMeasurements,
    EliminationStateObservation,
    EliminationTransitionObservation,
    RelationalEliminationGraphInput,
    RelationalEliminationGraphMeasurements,
)
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest


Position = tuple[int, int]


def collinear_elimination_moves(cells, state, half_quantum=None, *, landing_cells=None):
    found = set()
    for source in state:
        for landing in (cells if landing_cells is None else landing_cells) - state:
            if source[0] != landing[0] and source[1] != landing[1]:
                continue
            if half_quantum is not None and sum(abs(a - b) for a, b in zip(source, landing)) != 2 * half_quantum:
                continue
            row_sum, col_sum = source[0] + landing[0], source[1] + landing[1]
            if row_sum % 2 or col_sum % 2:
                continue
            middle = (row_sum // 2, col_sum // 2)
            if middle in state and middle in cells and middle != source:
                found.add((source, middle, landing))
    return tuple(sorted(found))


@dataclass(frozen=True)
class CollinearEliminationGraph:
    states: tuple[frozenset[Position], ...]
    transitions: tuple[tuple[int, int, Position, Position, Position], ...]
    complete: bool


def enumerate_collinear_elimination_graph(cells, start, *, half_quantum=None,
                                         max_states=128, max_transitions=512, landing_cells=None):
    """The shared bounded mechanical graph; no terminal criterion or role choice."""
    states, indexes = [start], {start: 0}
    transitions = []
    pending = deque((start,))
    bound_reached = False
    while pending and len(states) < max_states and len(transitions) < max_transitions:
        before = pending.popleft()
        before_index = indexes[before]
        for source, middle, landing in collinear_elimination_moves(cells, before, half_quantum,
                landing_cells=landing_cells):
            after = frozenset((before - {source, middle}) | {landing})
            after_index = indexes.get(after)
            if after_index is None:
                if len(states) >= max_states:
                    bound_reached = True
                    break
                after_index = len(states)
                indexes[after] = after_index
                states.append(after)
                pending.append(after)
            transitions.append((before_index, after_index, source, middle, landing))
            if len(transitions) >= max_transitions:
                bound_reached = True
                break
    return CollinearEliminationGraph(tuple(states), tuple(transitions),
                                    not pending and not bound_reached)


def measure_relational_elimination_graph(value: RelationalEliminationGraphInput):
    """Instantiate the SRC-supplied atomic operator and terminal criterion."""
    occupied = frozenset(value.occupied_positions)
    cells = occupied | frozenset(position for position, palette in value.cell_palette_rows
                                if palette in value.free_palette_values)
    static_landings = cells & frozenset(value.cell_positions)
    terminal_count = comb(len(cells), value.terminal_occupancy_count)
    if not 1 <= terminal_count <= 32:
        raise ValueError("declared relational terminal configuration set exceeds its bound")
    graph = enumerate_collinear_elimination_graph(cells, occupied,
        half_quantum=value.half_translation_quantum, max_states=128 - terminal_count,
        landing_cells=static_landings)
    states = list(graph.states)
    # Isolated terminal configurations are necessary to distinguish an unreachable
    # declared set from an empty set. They add no fabricated transition.
    terminals = tuple(frozenset(positions) for positions in
                      combinations(sorted(cells), value.terminal_occupancy_count))
    known = set(states)
    states.extend(state for state in terminals if state not in known)
    refs = {state: "state:relational:" + stable_digest(tuple(sorted(state)))[:24] for state in states}
    request = EliminationViabilityInput(
        measurement_context_ref=value.measurement_context_ref,
        states=tuple(EliminationStateObservation(state_ref=refs[state],
            occupied_ref_set=tuple("cell:" + stable_digest(position)[:24] for position in sorted(state))) for state in states),
        transitions=tuple(EliminationTransitionObservation(
            transition_ref="transition:relational:" + stable_digest((before, after, source, middle, landing))[:24],
            before_state_ref=refs[graph.states[before]], after_state_ref=refs[graph.states[after]],
            operator_ref=value.operator_ref,
            removed_ref_set=tuple("cell:" + stable_digest(position)[:24] for position in (source, middle)))
            for before, after, source, middle, landing in graph.transitions),
        start_state_ref=refs[graph.states[0]], declared_terminal_state_refs=tuple(refs[state] for state in terminals),
        transition_graph_complete=graph.complete, operator_ref=value.operator_ref,
        terminal_condition_ref=value.terminal_condition_ref, premise_claim_refs=value.premise_claim_refs,
        frame_ref=value.frame_ref, analysis_scope=value.analysis_scope)
    # These are absent edges, not admissible jumps or receiving-body roles.
    # Keep one shortest graph witness per geometric triple, with a hard bound
    # of three descriptions. Overflow invalidates the whole bounded inventory.
    parents = {}
    for before, after, source, middle, landing in graph.transitions:
        parents.setdefault(after, (before, source, middle, landing))
    rows = {}
    complete = graph.complete
    for index, state in enumerate(graph.states):
        for source in sorted(state):
            for middle in sorted(state):
                delta = (middle[0] - source[0], middle[1] - source[1])
                if (delta[0] and delta[1]) or sum(map(abs, delta)) != value.half_translation_quantum:
                    continue
                landing = (middle[0] + delta[0], middle[1] + delta[1])
                if landing in static_landings or landing in state:
                    continue
                key = (source, middle, landing)
                if key in rows:
                    continue
                if len(rows) == 3:
                    complete = False
                    break
                prefix = []
                cursor = index
                while cursor:
                    before_index, previous_source, previous_middle, previous_landing = parents[cursor]
                    prefix.append((previous_source, previous_middle, previous_landing))
                    cursor = before_index
                rows[key] = FrozenMap({
                    "source_position": source, "middle_position": middle,
                    "landing_position": landing, "before_occupied_positions": tuple(sorted(state)),
                    "after_occupied_positions": tuple(sorted((state - {source, middle}) | {landing})),
                    "fixed_prefix_triples": tuple(reversed(prefix)),
                    "landing_in_admitted_static_scope": False,
                })
            if not complete:
                break
        if not complete:
            break
    observations = tuple(rows.values()) if complete else ()
    reception_facts = FrozenMap({
        "context_ref": stable_digest((value.measurement_context_ref, observations))[:24],
        "frame_ref": value.frame_ref, "operator_ref": value.operator_ref,
        "terminal_condition_ref": value.terminal_condition_ref,
        "premise_claim_refs": value.premise_claim_refs, "analysis_scope": value.analysis_scope,
        "graph_complete": graph.complete, "absent_landing_inventory_complete": complete,
        "absent_landing_count": len(observations), "absent_landing_observations": observations,
        "current_occupied_positions": tuple(sorted(occupied)),
    })
    # Counterfactual source poses beside a current occupied middle, with a
    # measured free landing. These are geometry, not carrier roles or actions.
    poses = []
    pose_inventory_complete = graph.complete
    free_cells = cells - occupied
    for middle in sorted(occupied):
        for dr, dc in ((0, 1), (-1, 0), (0, -1), (1, 0)):
            source = (middle[0] - dr * value.half_translation_quantum,
                middle[1] - dc * value.half_translation_quantum)
            landing = (middle[0] + dr * value.half_translation_quantum,
                middle[1] + dc * value.half_translation_quantum)
            if source in occupied or landing not in free_cells:
                continue
            if len(poses) == 3:
                pose_inventory_complete = False
                break
            poses.append(FrozenMap({"source_position": source, "middle_position": middle,
                "landing_position": landing, "landing_in_admitted_static_scope": True}))
        if not pose_inventory_complete:
            break
    pose_observations = tuple(poses) if pose_inventory_complete else ()
    transport_facts = FrozenMap({
        "context_ref": stable_digest((value.measurement_context_ref, pose_observations))[:24],
        "frame_ref": value.frame_ref, "operator_ref": value.operator_ref,
        "terminal_condition_ref": value.terminal_condition_ref,
        "premise_claim_refs": value.premise_claim_refs, "analysis_scope": value.analysis_scope,
        "graph_complete": graph.complete, "pose_inventory_complete": pose_inventory_complete,
        "pose_count": len(pose_observations), "pose_observations": pose_observations,
        "current_occupied_positions": tuple(sorted(occupied)),
    })
    return RelationalEliminationGraphMeasurements(viability_request=request,
        reception_observation_facts=reception_facts, transport_observation_facts=transport_facts)


def measure_elimination_viability(
    value: EliminationViabilityInput,
) -> EliminationViabilityMeasurements:
    predecessors = {item.state_ref: set() for item in value.states}
    for transition in value.transitions:
        predecessors[transition.after_state_ref].add(transition.before_state_ref)

    distances = {state_ref: 0 for state_ref in value.declared_terminal_state_refs}
    pending: deque[str] = deque(sorted(value.declared_terminal_state_refs))
    expanded_count = 0
    depth_bound_reached = False
    while pending and expanded_count < value.max_expanded_states:
        current = pending.popleft()
        expanded_count += 1
        next_distance = distances[current] + 1
        for predecessor in sorted(predecessors[current]):
            if predecessor in distances:
                continue
            if next_distance > value.max_depth:
                depth_bound_reached = True
                continue
            distances[predecessor] = next_distance
            pending.append(predecessor)

    search_complete = not pending and not depth_bound_reached and value.transition_graph_complete
    measurements: list[EliminationStateViability] = []
    for state in value.states:
        distance = distances.get(state.state_ref)
        if distance is not None:
            reachable: bool | None = True
            exact_dead_end = False
        elif search_complete:
            reachable = False
            exact_dead_end = True
        else:
            reachable = None
            exact_dead_end = False
        measurements.append(
            EliminationStateViability(
                state_ref=state.state_ref,
                terminal_reachable=reachable,
                shortest_terminal_distance=distance,
                exact_dead_end=exact_dead_end,
            )
        )

    start = next(
        item for item in measurements if item.state_ref == value.start_state_ref
    )
    return EliminationViabilityMeasurements(
        state_viability=tuple(measurements),
        descriptive_facts=FrozenMap(
            {
                "measurement_context_ref": value.measurement_context_ref,
                "operator_ref": value.operator_ref,
                "terminal_condition_ref": value.terminal_condition_ref,
                "premise_claim_refs": value.premise_claim_refs,
                "frame_ref": value.frame_ref,
                "analysis_scope": value.analysis_scope,
                "transition_graph_complete": value.transition_graph_complete,
                "state_count": len(value.states),
                "transition_count": len(value.transitions),
                "declared_terminal_state_count": len(
                    value.declared_terminal_state_refs
                ),
                "expanded_state_count": expanded_count,
                "search_complete": search_complete,
                "expansion_bound_reached": bool(pending),
                "depth_bound_reached": depth_bound_reached,
                "terminal_reachable_state_count": sum(
                    item.terminal_reachable is True for item in measurements
                ),
                "exact_dead_end_state_count": sum(
                    item.exact_dead_end for item in measurements
                ),
                "start_state_is_declared_terminal": (
                    value.start_state_ref in value.declared_terminal_state_refs
                ),
                "start_terminal_reachable": start.terminal_reachable,
                "start_exact_dead_end": start.exact_dead_end,
            }
        ),
    )
