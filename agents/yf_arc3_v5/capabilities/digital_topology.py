"""Bounded connectivity and maximal digital arcs; no semantic recognition.

The attested source supplies the graph convention. These measurements alone
are not an appearance key: a hole count does not identify a structural family.
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator

from agents.yf_arc3_v5.logos.types import FrozenMap, FrozenModel, Ref

Point = tuple[int, int]


class DigitalTopologyContract(FrozenModel):
    foreground_connectivity: Literal[8]
    background_connectivity: Literal[4]
    graph_adjacency: Literal["eight_neighbors_without_redundant_diagonal"]
    graph_vertices: Literal["individual_pixels_of_degree_not_two"]
    closed_degree_two_component_arc_count: Literal[1]
    max_padded_domain_area: int = Field(ge=9, le=4356)
    source_hash: Ref


class DigitalTopologyInput(FrozenModel):
    support: tuple[Point, ...] = Field(min_length=1, max_length=4096)
    contract: DigitalTopologyContract

    @model_validator(mode="after")
    def validate_finite_domain(self) -> "DigitalTopologyInput":
        if len(set(self.support)) != len(self.support):
            raise ValueError("digital support repeats a pixel")
        rows, columns = zip(*self.support)
        area = (max(rows) - min(rows) + 3) * (max(columns) - min(columns) + 3)
        if area > self.contract.max_padded_domain_area:
            raise ValueError("digital topology padded domain bound exceeded")
        return self


def _neighbors(point: Point, connectivity: int) -> tuple[Point, ...]:
    row, column = point
    return tuple((row + dr, column + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                 if (dr or dc) and (connectivity == 8 or abs(dr) + abs(dc) == 1))


def _components(points: set[Point], connectivity: int) -> tuple[frozenset[Point], ...]:
    remaining = set(points)
    result = []
    while remaining:
        start = min(remaining)
        queue = deque((start,))
        component = {start}
        remaining.remove(start)
        while queue:
            for adjacent in _neighbors(queue.popleft(), connectivity):
                if adjacent in remaining:
                    remaining.remove(adjacent)
                    component.add(adjacent)
                    queue.append(adjacent)
        result.append(frozenset(component))
    return tuple(result)


def _edge(first: Point, second: Point) -> tuple[Point, Point]:
    return (first, second) if first < second else (second, first)


def measure_digital_topology(value: DigitalTopologyInput) -> FrozenMap:
    top = min(row for row, _column in value.support)
    left = min(column for _row, column in value.support)
    normalized = tuple(sorted((row - top, column - left) for row, column in value.support))
    return _measure_normalized_topology(normalized, value.contract)


@lru_cache(maxsize=128)
def _measure_normalized_topology(points: tuple[Point, ...], contract: DigitalTopologyContract) -> FrozenMap:
    """Reuse immutable geometric results, bounded independently of game memory."""
    support = set(points)
    rows, columns = zip(*support)
    top, bottom, left, right = min(rows) - 1, max(rows) + 1, min(columns) - 1, max(columns) + 1
    complement = {(row, column) for row in range(top, bottom + 1) for column in range(left, right + 1)} - support
    empty_components = _components(complement, contract.background_connectivity)
    holes = tuple(component for component in empty_components if (top, left) not in component)
    graph: dict[Point, tuple[Point, ...]] = {}
    for point in sorted(support):
        row, column = point
        graph[point] = tuple(adjacent for adjacent in _neighbors(point, contract.foreground_connectivity)
                             if adjacent in support and not (
                                 adjacent[0] != row and adjacent[1] != column
                                 and ((row, adjacent[1]) in support or (adjacent[0], column) in support)
                             ))
    vertices = tuple(point for point in sorted(graph) if len(graph[point]) != 2)
    visited: set[tuple[Point, Point]] = set()
    arcs: list[tuple[Point, ...]] = []
    for start in vertices:
        for adjacent in graph[start]:
            if _edge(start, adjacent) in visited:
                continue
            path = [start, adjacent]
            visited.add(_edge(start, adjacent))
            previous, current = start, adjacent
            while len(graph[current]) == 2:
                following = next(point for point in graph[current] if point != previous)
                edge = _edge(current, following)
                if edge in visited:
                    raise ValueError("digital graph arc unexpectedly revisits an edge")
                visited.add(edge)
                path.append(following)
                previous, current = current, following
            arcs.append(tuple(path))
    closed_arc_count = 0
    for start in sorted(graph):
        for adjacent in graph[start]:
            if _edge(start, adjacent) in visited:
                continue
            if len(graph[start]) != 2:
                raise ValueError("digital graph contains an unvisited branching edge")
            path = [start, adjacent]
            visited.add(_edge(start, adjacent))
            previous, current = start, adjacent
            while current != start:
                following = next(point for point in graph[current] if point != previous)
                edge = _edge(current, following)
                if edge in visited:
                    raise ValueError("digital closed arc unexpectedly revisits an edge")
                visited.add(edge)
                path.append(following)
                previous, current = current, following
            arcs.append(tuple(path))
            closed_arc_count += contract.closed_degree_two_component_arc_count
    return FrozenMap({
        "foreground_component_count": len(_components(support, contract.foreground_connectivity)),
        "hole_count": len(holes),
        "hole_areas": tuple(sorted(len(component) for component in holes)),
        "graph_arc_count": len(arcs),
        "graph_closed_arc_count": closed_arc_count,
        "graph_endpoint_count": sum(len(graph[point]) == 1 for point in graph),
        "graph_junction_degrees": tuple(sorted(len(graph[point]) for point in graph if len(graph[point]) >= 3)),
        "graph_isolated_vertex_count": sum(not graph[point] for point in graph),
        "contains_full_two_by_two_support": any(
            (row + 1, column) in support and (row, column + 1) in support and (row + 1, column + 1) in support
            for row, column in support
        ),
        "measurement_contract_source_hash": contract.source_hash,
    })
