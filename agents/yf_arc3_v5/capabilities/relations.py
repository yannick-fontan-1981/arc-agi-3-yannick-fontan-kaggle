"""Pure finite relation-graph analysis with no semantic role inference."""

from __future__ import annotations

from collections import deque

from agents.yf_arc3_v5.capabilities.contracts import (
    InteractionTopologyInput,
    InteractionTopologyMeasurements,
    LandingFitMeasurement,
    ObservedTopologyEdge,
    RelationGraphInput,
    RelationGraphResult,
    RelationReachability,
    TopologyTransitionMeasurement,
)
from agents.yf_arc3_v5.logos.types import FrozenMap


def measure_interaction_topology(
    value: InteractionTopologyInput,
) -> InteractionTopologyMeasurements:
    """Measure exact graph and landing deltas without assigning interaction roles."""

    configurations = {
        item.configuration_ref: item for item in value.configurations
    }
    transitions: list[TopologyTransitionMeasurement] = []
    added_edge_count = 0
    removed_edge_count = 0
    component_change_count = 0
    unchanged_count = 0
    for observation in value.transitions:
        before = configurations[observation.before_configuration_ref]
        after = configurations[observation.after_configuration_ref]
        before_edges = {
            (edge.left_ref, edge.right_ref) for edge in before.adjacency_edges
        }
        after_edges = {
            (edge.left_ref, edge.right_ref) for edge in after.adjacency_edges
        }
        added = tuple(
            ObservedTopologyEdge(left_ref=left, right_ref=right)
            for left, right in sorted(after_edges - before_edges)
        )
        removed = tuple(
            ObservedTopologyEdge(left_ref=left, right_ref=right)
            for left, right in sorted(before_edges - after_edges)
        )
        before_components = _connected_component_count(
            before.node_refs, before_edges
        )
        after_components = _connected_component_count(after.node_refs, after_edges)
        added_edge_count += len(added)
        removed_edge_count += len(removed)
        if before_components != after_components:
            component_change_count += 1
        if not added and not removed and before_components == after_components:
            unchanged_count += 1
        transitions.append(
            TopologyTransitionMeasurement(
                transition_ref=observation.transition_ref,
                added_adjacency_edges=added,
                removed_adjacency_edges=removed,
                before_connected_component_count=before_components,
                after_connected_component_count=after_components,
            )
        )

    landing_fits: list[LandingFitMeasurement] = []
    valid_landing_count = 0
    for projection in value.landing_projections:
        projected = set(projection.projected_positions)
        permissible = set(projection.permissible_positions)
        occupied = set(projection.occupied_positions)
        outside_count = len(projected - permissible)
        overlap_count = len(projected.intersection(occupied))
        exact_fit = outside_count == 0 and overlap_count == 0
        valid_landing_count += int(exact_fit)
        landing_fits.append(
            LandingFitMeasurement(
                candidate_ref=projection.candidate_ref,
                entity_ref=projection.entity_ref,
                region_ref=projection.region_ref,
                outside_permissible_count=outside_count,
                occupied_overlap_count=overlap_count,
                exact_fit=exact_fit,
            )
        )

    return InteractionTopologyMeasurements(
        topology_transitions=tuple(transitions),
        landing_fits=tuple(landing_fits),
        descriptive_facts=FrozenMap(
            {
                "measurement_context_ref": value.measurement_context_ref,
                "configuration_count": len(value.configurations),
                "topology_transition_count": len(value.transitions),
                "added_adjacency_edge_count": added_edge_count,
                "removed_adjacency_edge_count": removed_edge_count,
                "connected_component_change_count": component_change_count,
                "unchanged_topology_transition_count": unchanged_count,
                "landing_projection_count": len(value.landing_projections),
                "exact_valid_landing_count": valid_landing_count,
            }
        ),
    )


def _connected_component_count(
    node_refs: tuple[str, ...], edges: set[tuple[str, str]]
) -> int:
    neighbors = {node_ref: set() for node_ref in node_refs}
    for left_ref, right_ref in edges:
        neighbors[left_ref].add(right_ref)
        neighbors[right_ref].add(left_ref)
    count = 0
    visited: set[str] = set()
    for node_ref in node_refs:
        if node_ref in visited:
            continue
        count += 1
        pending = [node_ref]
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            pending.extend(neighbors[current] - visited)
    return count


def analyze_relation_graph(value: RelationGraphInput) -> RelationGraphResult:
    directed_neighbors = {node: set[str]() for node in value.node_refs}
    weak_neighbors = {node: set[str]() for node in value.node_refs}
    for edge in value.edges:
        directed_neighbors[edge.source_ref].add(edge.target_ref)
        weak_neighbors[edge.source_ref].add(edge.target_ref)
        weak_neighbors[edge.target_ref].add(edge.source_ref)
        if not value.directed:
            directed_neighbors[edge.target_ref].add(edge.source_ref)

    components: list[tuple[str, ...]] = []
    unseen = set(value.node_refs)
    while unseen:
        start = min(unseen)
        reached = _distances(start, weak_neighbors)
        component = tuple(sorted(reached))
        components.append(component)
        unseen.difference_update(component)

    reachability_items: list[RelationReachability] = []
    for query in value.queries:
        distances, predecessors = _distances_with_predecessors(
            query.source_ref, directed_neighbors
        )
        reachable = query.target_ref in distances
        path = (
            _first_shortest_path(
                source=query.source_ref,
                target=query.target_ref,
                predecessors=predecessors,
            )
            if reachable
            else ()
        )
        required = tuple(
            node
            for node in path[1:-1]
            if query.target_ref
            not in _distances(
                query.source_ref,
                {
                    current: {
                        adjacent
                        for adjacent in neighbors
                        if adjacent != node
                    }
                    for current, neighbors in sorted(directed_neighbors.items())
                    if current != node
                },
            )
        )
        reachability_items.append(
            RelationReachability(
                query_id=query.query_id,
                source_ref=query.source_ref,
                target_ref=query.target_ref,
                reachable=reachable,
                minimum_edge_count=distances.get(query.target_ref),
                first_shortest_path=path,
                required_intermediate_refs=required,
            )
        )
    return RelationGraphResult(
        graph_id=value.graph_id,
        dimension=value.dimension,
        relation_ref=value.relation_ref,
        directed=value.directed,
        weak_components=tuple(sorted(components)),
        reachability=tuple(reachability_items),
    )


def _distances(start: str, neighbors: dict[str, set[str]]) -> dict[str, int]:
    return _distances_with_predecessors(start, neighbors)[0]


def _distances_with_predecessors(
    start: str,
    neighbors: dict[str, set[str]],
) -> tuple[dict[str, int], dict[str, str]]:
    if start not in neighbors:
        return {}, {}
    distances = {start: 0}
    predecessors: dict[str, str] = {}
    pending = deque((start,))
    while pending:
        current = pending.popleft()
        for adjacent in sorted(neighbors[current]):
            if adjacent in distances:
                continue
            distances[adjacent] = distances[current] + 1
            predecessors[adjacent] = current
            pending.append(adjacent)
    return distances, predecessors


def _first_shortest_path(
    *,
    source: str,
    target: str,
    predecessors: dict[str, str],
) -> tuple[str, ...]:
    reverse_path = [target]
    while reverse_path[-1] != source:
        parent = predecessors.get(reverse_path[-1])
        if parent is None:
            return ()
        reverse_path.append(parent)
    return tuple(reversed(reverse_path))
