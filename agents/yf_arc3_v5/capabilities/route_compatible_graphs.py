"""Bounded measurement and simulation for typed, shared-node relation graphs."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

from agents.yf_arc3_v5.capabilities.contracts import (
    ComponentDescription,
    InteractionProbeAgenda,
    InteractionProbeCandidate,
    RouteCompatibleGraphPlanInput,
    VerifiedAlignmentActionInput,
)
from agents.yf_arc3_v5.capabilities.planning import (
    POINT_ACTION_REF,
    _axis_extent,
    _axis_origin,
    _checkpoint_reason,
    _closed_interval_gap,
    _component_morphology,
    _components_are_locally_adjacent,
    _current_visual_quantum_candidates,
    _first_index_path,
    _first_shortest_route_witness,
    _folded_control_sets,
    _interior_point,
    _orthogonal_end,
    _orthogonal_start,
    _scale_relation,
    _target_extent,
)
from agents.yf_arc3_v5.logos.types import FrozenMap, canonical_json


_HARD_MAX_IDENTITIES = 4
_HARD_MAX_ROUTE_VARIANTS = 3
_HARD_MAX_EXPANDED_STATES = 8192
_HARD_MAX_TOTAL_EXPANDED_STATES = 32768
_HARD_MAX_PLAN_ACTIONS = 128


@dataclass(frozen=True)
class _MeasuredInterface:
    component_ref: str
    left_index: int
    right_index: int
    left_requirement: int
    right_requirement: int

    @property
    def edge(self) -> frozenset[int]:
        return frozenset((self.left_index, self.right_index))

    def requirement_by_index(self) -> dict[int, int]:
        return {
            self.left_index: self.left_requirement,
            self.right_index: self.right_requirement,
        }


@dataclass(frozen=True)
class _EndpointMeasurement:
    target_index: int
    target_quantity: int
    path: tuple[int, ...]


@dataclass(frozen=True)
class _IdentitySeed:
    identity_ref: str
    carrier_entity_ref: str
    attribute_entity_ref: str
    carrier_component_ref: str
    attribute_component_ref: str
    target_component_ref: str
    current_index: int
    attribute_offset: tuple[int, int]
    endpoint_measurements: tuple[_EndpointMeasurement, ...]


@dataclass(frozen=True)
class _MeasuredIdentity:
    identity_ref: str
    carrier_entity_ref: str
    attribute_entity_ref: str
    carrier_component_ref: str
    attribute_component_ref: str
    target_component_ref: str
    current_index: int
    target_index: int
    target_quantity: int
    path: tuple[int, ...]
    attribute_offset: tuple[int, int]


@dataclass(frozen=True)
class _GraphMeasurement:
    measurement_ref: str
    axis: str
    origin_side: str
    quantum: int
    material_value: int
    material: tuple[ComponentDescription, ...]
    quantities: tuple[int, ...]
    capacities: tuple[int, ...]
    controls: tuple[tuple[int, int, str], ...]
    material_edges: tuple[tuple[int, int], ...]
    interfaces: tuple[_MeasuredInterface, ...]
    identity_edges: tuple[tuple[int, int], ...]
    identities: tuple[_MeasuredIdentity, ...]
    reciprocal_identity_refs: tuple[str, str]
    independent_identity_refs: tuple[str, ...]
    reciprocal_path: tuple[int, ...]
    terminal_requirements: tuple[tuple[int, int], ...]
    material_bands: tuple[tuple[str, int, int], ...]
    zone_boxes: tuple[tuple[str, int, int, int, int], ...]
    material_origins: tuple[int, ...]


@dataclass(frozen=True)
class _PlanStep:
    source_index: int
    destination_index: int
    component_ref: str
    measurement_kind: str
    quantities_after: tuple[int, ...]
    identity_indices_after: tuple[tuple[str, int], ...]
    moved_identity_refs: tuple[str, ...]
    stage: str


@dataclass
class _Simulation:
    quantities: tuple[int, ...]
    identity_indices: dict[str, int]
    steps: list[_PlanStep]
    expanded_state_count: int = 0
    maximum_remaining_route_distance_after_crossing: int = 0
    remaining_route_distance_imbalance_after_crossing: int = 0

    def copy(self) -> "_Simulation":
        return _Simulation(
            quantities=self.quantities,
            identity_indices=dict(self.identity_indices),
            steps=list(self.steps),
            expanded_state_count=self.expanded_state_count,
            maximum_remaining_route_distance_after_crossing=(
                self.maximum_remaining_route_distance_after_crossing
            ),
            remaining_route_distance_imbalance_after_crossing=(
                self.remaining_route_distance_imbalance_after_crossing
            ),
        )


def analyze_route_compatible_orders(
    value: VerifiedAlignmentActionInput,
) -> InteractionProbeAgenda:
    """Enumerate two exact execution orders without assigning their priority."""

    if (
        POINT_ACTION_REF not in value.available_action_refs
        or value.committed_plan_facts
    ):
        return _empty_agenda()
    measurement = _unique_graph_measurement(value)
    if measurement is None:
        return _empty_agenda()
    reciprocal = measurement.reciprocal_identity_refs
    independent_groups = tuple((item,) for item in measurement.independent_identity_refs)
    orders = (
        (reciprocal, *independent_groups),
        (*independent_groups, reciprocal),
    )
    described: list[tuple[InteractionProbeCandidate, FrozenMap]] = []
    for groups in orders:
        reciprocal_first = bool(groups and tuple(groups[0]) == reciprocal)
        digest = hashlib.sha256(
            canonical_json((measurement.measurement_ref, groups)).encode("utf-8")
        ).hexdigest()[:16]
        candidate_ref = f"measurement:route-order:{digest}"
        facts = FrozenMap(
            {
                "candidate_ref": candidate_ref,
                "measurement_ref": measurement.measurement_ref,
                "ordered_identity_groups": groups,
                "typed_relation_graphs_measured": True,
                "complete_identity_route_coverage": True,
                "terminal_alignment_after_identity_transport": True,
                "reciprocal_group_precedes_independent_obligation": reciprocal_first,
                "independent_obligation_incident_to_reciprocal_shared_route": True,
                "intermediate_interface_exposure_count": (
                    0 if reciprocal_first else len(independent_groups)
                ),
            }
        )
        described.append(
            (
                InteractionProbeCandidate(
                    candidate_ref=candidate_ref,
                    action_ref=POINT_ACTION_REF,
                    basis_refs=(measurement.measurement_ref,),
                ),
                facts,
            )
        )
    ordered = tuple(sorted(described, key=lambda item: item[0].candidate_ref))
    return InteractionProbeAgenda(
        candidates=tuple(item[0] for item in ordered),
        alternative_refs=tuple(item[0].candidate_ref for item in ordered),
        alternative_facts=FrozenMap(
            {candidate.candidate_ref: facts for candidate, facts in ordered}
        ),
        context_facts=FrozenMap(
            {
                "measurement_ref": measurement.measurement_ref,
                "route_order_candidate_limit": len(ordered),
            }
        ),
        has_candidates=bool(ordered),
    )


def enumerate_route_compatible_graph_plans(
    value: RouteCompatibleGraphPlanInput,
) -> InteractionProbeAgenda:
    """Simulate only the execution order already declared by DRM/SRC."""

    if POINT_ACTION_REF not in value.available_action_refs:
        return _empty_agenda()
    measurement = _unique_graph_measurement(value)
    if measurement is None:
        return _empty_agenda()
    raw_order_facts = value.route_order_analysis.alternative_facts.get(
        value.declared_route_order_ref
    )
    if not isinstance(raw_order_facts, FrozenMap):
        return _empty_agenda()
    if str(raw_order_facts.get("measurement_ref") or "") != measurement.measurement_ref:
        return _empty_agenda()
    groups = tuple(
        tuple(str(identity_ref) for identity_ref in group)
        for group in raw_order_facts.get("ordered_identity_groups", ())
    )
    if not groups:
        return _empty_agenda()
    route_policy = value.route_generation_policy
    max_expanded_states = _bounded_integer(
        route_policy,
        "route_compatible_max_expanded_states_per_stage",
        default=4096,
        hard_maximum=_HARD_MAX_EXPANDED_STATES,
    )
    max_plan_actions = _bounded_integer(
        route_policy,
        "max_plan_actions",
        default=_HARD_MAX_PLAN_ACTIONS,
        hard_maximum=_HARD_MAX_PLAN_ACTIONS,
    )
    variant_limit = _bounded_integer(
        route_policy,
        "max_route_variants",
        default=_HARD_MAX_ROUTE_VARIANTS,
        hard_maximum=_HARD_MAX_ROUTE_VARIANTS,
    )
    attempted_suffixes = frozenset(value.current_scope_attempted_route_suffixes)
    variant_limit = min(
        variant_limit,
        max(0, _HARD_MAX_ROUTE_VARIANTS - len(attempted_suffixes)),
    )
    if variant_limit == 0:
        return _empty_agenda()
    simulations = _simulate_declared_order(
        measurement=measurement,
        groups=groups,
        max_expanded_states=max_expanded_states,
        max_plan_actions=max_plan_actions,
    )[:variant_limit]
    component_by_ref = {
        item.component_id: item for item in value.scene.blocks.components
    }
    tracked_by_component = {
        item.component.component_id: item.entity_id for item in value.tracking.entities
    }
    described: list[tuple[InteractionProbeCandidate, FrozenMap]] = []
    discarded_no_effect_refs: list[str] = []
    no_effect_refs = frozenset(value.current_context_no_effect_candidate_refs)
    for simulation in simulations:
        if not simulation.steps:
            continue
        route_digest = hashlib.sha256(
            canonical_json(
                (
                    measurement.measurement_ref,
                    groups,
                    tuple(
                        (step.component_ref, step.source_index, step.destination_index)
                        for step in simulation.steps
                    ),
                )
            ).encode("utf-8")
        ).hexdigest()[:16]
        if route_digest in attempted_suffixes:
            continue
        route_ref = f"measurement:route-compatible:{route_digest}"
        first_component_ref = simulation.steps[0].component_ref
        candidate_ref = f"probe:{first_component_ref}:route:{route_digest}"
        if candidate_ref in no_effect_refs:
            discarded_no_effect_refs.append(candidate_ref)
            continue
        first_component = component_by_ref.get(first_component_ref)
        if first_component is None:
            continue
        facts = _plan_facts(
            value=value,
            measurement=measurement,
            groups=groups,
            simulation=simulation,
            route_ref=route_ref,
            candidate_ref=candidate_ref,
            route_digest=route_digest,
            route_variants_materialized=len(simulations),
            variant_limit=variant_limit,
            component_by_ref=component_by_ref,
            tracked_by_component=tracked_by_component,
        )
        point = _interior_point(first_component)
        described.append(
            (
                InteractionProbeCandidate(
                    candidate_ref=candidate_ref,
                    action_ref=POINT_ACTION_REF,
                    action_data=FrozenMap({"x": point[1], "y": point[0]}),
                    component_ref=first_component_ref,
                    point=point,
                    basis_refs=(measurement.measurement_ref,),
                ),
                facts,
            )
        )
    ordered = tuple(sorted(described, key=lambda item: item[0].candidate_ref))
    return InteractionProbeAgenda(
        candidates=tuple(item[0] for item in ordered),
        alternative_refs=tuple(item[0].candidate_ref for item in ordered),
        alternative_facts=FrozenMap(
            {candidate.candidate_ref: facts for candidate, facts in ordered}
        ),
        context_facts=FrozenMap({"selection_mode": "declared_route_order"}),
        has_candidates=bool(ordered),
        current_context_no_effect_candidate_refs=tuple(discarded_no_effect_refs),
    )


def _unique_graph_measurement(
    value: VerifiedAlignmentActionInput,
) -> _GraphMeasurement | None:
    outputs: dict[str, _GraphMeasurement] = {}
    components = value.scene.blocks.components
    tracked_by_component = {
        item.component.component_id: item.entity_id for item in value.tracking.entities
    }
    for quantum in _current_visual_quantum_candidates(components):
        for axis in ("column", "row"):
            for origin_side in ("minimum", "maximum"):
                for measurement in _graph_measurements(
                    components=components,
                    frame_height=value.scene.frame.height,
                    frame_width=value.scene.frame.width,
                    tracked_by_component=tracked_by_component,
                    quantum=quantum,
                    axis=axis,
                    origin_side=origin_side,
                ):
                    key = canonical_json(
                        (
                            measurement.axis,
                            measurement.origin_side,
                            tuple(item.component_id for item in measurement.material),
                            measurement.controls,
                            tuple(
                                (item.identity_ref, item.path)
                                for item in measurement.identities
                            ),
                        )
                    )
                    outputs.setdefault(key, measurement)
                    if len(outputs) > 1:
                        return None
    return next(iter(outputs.values())) if len(outputs) == 1 else None


def _graph_measurements(
    *,
    components: tuple[ComponentDescription, ...],
    frame_height: int,
    frame_width: int,
    tracked_by_component: dict[str, str],
    quantum: int,
    axis: str,
    origin_side: str,
) -> tuple[_GraphMeasurement, ...]:
    by_value: dict[int, list[ComponentDescription]] = defaultdict(list)
    for component in components:
        by_value[component.value].append(component)
    outputs: list[_GraphMeasurement] = []
    for material_value, same_value in sorted(by_value.items()):
        material = tuple(
            sorted(
                (
                    item
                    for item in same_value
                    if _axis_extent(item, axis) % quantum == 0
                    and _axis_extent(item, axis) >= quantum
                    and (_orthogonal_end(item, axis) - _orthogonal_start(item, axis) + 1)
                    > quantum
                ),
                key=lambda item: (
                    _orthogonal_start(item, axis),
                    _axis_origin(item, axis, origin_side),
                    item.component_id,
                ),
            )
        )
        if not 4 <= len(material) <= 8:
            continue
        for control_set in _folded_control_sets(
            components=components,
            material=material,
            material_value=material_value,
            axis=axis,
            origin_side=origin_side,
            quantum=quantum,
        ):
            incident = tuple(
                sorted(
                    {
                        int(index)
                        for pair in control_set["pairs"]
                        for index in pair[:2]
                    }
                )
            )
            if len(incident) < 4:
                continue
            remap = {prior: current for current, prior in enumerate(incident)}
            graph_material = tuple(material[index] for index in incident)
            pairs = tuple(
                (
                    remap[int(left)],
                    remap[int(right)],
                    str(left_ref),
                    str(right_ref),
                    int(anchor),
                    int(gap_start),
                    int(gap_end),
                )
                for left, right, left_ref, right_ref, anchor, gap_start, gap_end in control_set[
                    "pairs"
                ]
            )
            controls = tuple(
                step
                for left, right, left_ref, right_ref, *_rest in pairs
                for step in (
                    (right, left, left_ref),
                    (left, right, right_ref),
                )
            )
            material_edges = tuple(
                sorted({tuple(sorted((source, destination))) for source, destination, _ in controls})
            )
            if len(material_edges) != len(graph_material) - 1:
                continue
            if not _graph_is_connected(len(graph_material), material_edges):
                continue
            material_bands, zone_boxes, origins, capacities = _zone_geometry(
                material=graph_material,
                axis=axis,
                origin_side=origin_side,
                quantum=quantum,
                frame_height=frame_height,
                frame_width=frame_width,
            )
            if not zone_boxes:
                continue
            interfaces = _measure_interfaces(
                components=components,
                material=graph_material,
                pairs=pairs,
                material_value=material_value,
                axis=axis,
                origin_side=origin_side,
                quantum=quantum,
            )
            if len(interfaces) < 2:
                continue
            identity_edges = tuple(
                sorted(tuple(sorted((item.left_index, item.right_index))) for item in interfaces)
            )
            identities = _measure_identities(
                components=components,
                material=graph_material,
                material_edges=material_edges,
                interfaces=interfaces,
                identity_edges=identity_edges,
                material_bands=material_bands,
                zone_boxes=zone_boxes,
                origins=origins,
                capacities=capacities,
                tracked_by_component=tracked_by_component,
                material_value=material_value,
                control_refs=frozenset(step[2] for step in controls),
                axis=axis,
                origin_side=origin_side,
                quantum=quantum,
            )
            if identities is None:
                continue
            measured_identities, reciprocal_refs, independent_refs, reciprocal_path = identities
            terminal_by_index: dict[int, int] = {}
            terminal_conflict = False
            for identity in measured_identities:
                prior = terminal_by_index.get(identity.target_index)
                if prior is not None and prior != identity.target_quantity:
                    terminal_conflict = True
                    break
                terminal_by_index[identity.target_index] = identity.target_quantity
            if terminal_conflict or len(terminal_by_index) >= len(graph_material):
                continue
            quantities = tuple(
                _axis_extent(item, axis) // quantum for item in graph_material
            )
            if any(quantity > capacity for quantity, capacity in zip(quantities, capacities)):
                continue
            measurement_digest = hashlib.sha256(
                canonical_json(
                    (
                        axis,
                        origin_side,
                        quantum,
                        tuple(item.component_id for item in graph_material),
                        controls,
                        tuple(
                            (item.identity_ref, item.path, item.target_quantity)
                            for item in measured_identities
                        ),
                    )
                ).encode("utf-8")
            ).hexdigest()[:16]
            outputs.append(
                _GraphMeasurement(
                    measurement_ref=f"measurement:typed-graphs:{measurement_digest}",
                    axis=axis,
                    origin_side=origin_side,
                    quantum=quantum,
                    material_value=material_value,
                    material=graph_material,
                    quantities=quantities,
                    capacities=capacities,
                    controls=controls,
                    material_edges=material_edges,
                    interfaces=interfaces,
                    identity_edges=identity_edges,
                    identities=measured_identities,
                    reciprocal_identity_refs=reciprocal_refs,
                    independent_identity_refs=independent_refs,
                    reciprocal_path=reciprocal_path,
                    terminal_requirements=tuple(sorted(terminal_by_index.items())),
                    material_bands=material_bands,
                    zone_boxes=zone_boxes,
                    material_origins=origins,
                )
            )
    return tuple(outputs)


def _zone_geometry(
    *,
    material: tuple[ComponentDescription, ...],
    axis: str,
    origin_side: str,
    quantum: int,
    frame_height: int,
    frame_width: int,
) -> tuple[
    tuple[tuple[str, int, int], ...],
    tuple[tuple[str, int, int, int, int], ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    intervals = tuple(
        (_orthogonal_start(item, axis), _orthogonal_end(item, axis))
        for item in material
    )
    remaining = set(range(len(material)))
    interval_groups: list[tuple[int, ...]] = []
    while remaining:
        root = min(remaining)
        reached = {root}
        changed = True
        while changed:
            changed = False
            for index in tuple(remaining - reached):
                if any(
                    _closed_interval_gap(*intervals[index], *intervals[other]) == 0
                    for other in reached
                ):
                    reached.add(index)
                    changed = True
        remaining -= reached
        interval_groups.append(tuple(sorted(reached)))
    ordered_groups = tuple(
        sorted(
            interval_groups,
            key=lambda group: min(intervals[index][0] for index in group),
        )
    )
    group_ranges = tuple(
        (
            min(intervals[index][0] for index in group),
            max(intervals[index][1] for index in group),
        )
        for group in ordered_groups
    )
    if any(
        left[1] >= right[0]
        for left, right in zip(group_ranges, group_ranges[1:])
    ):
        return (), (), (), ()
    separators = tuple(
        (left[1] + right[0]) // 2
        for left, right in zip(group_ranges, group_ranges[1:])
    )
    orthogonal_bands = tuple(
        (
            group_ranges[0][0] if index == 0 else separators[index - 1] + 1,
            group_ranges[-1][1] if index == len(group_ranges) - 1 else separators[index],
        )
        for index in range(len(group_ranges))
    )
    group_by_index = {
        material_index: group_index
        for group_index, group in enumerate(ordered_groups)
        for material_index in group
    }
    axis_size = frame_width if axis == "column" else frame_height
    origins = tuple(_axis_origin(item, axis, origin_side) for item in material)
    if origin_side == "minimum":
        global_axis_start = min(origins)
        global_axis_end = axis_size - global_axis_start - 1
        if global_axis_end < max(item.bbox.right if axis == "column" else item.bbox.bottom for item in material):
            return (), (), (), ()
    else:
        global_axis_end = max(origins)
        global_axis_start = axis_size - global_axis_end - 1
        if global_axis_start > min(item.bbox.left if axis == "column" else item.bbox.top for item in material):
            return (), (), (), ()
    axis_ranges: dict[int, tuple[int, int]] = {}
    for group in ordered_groups:
        ordered = tuple(
            sorted(
                group,
                key=lambda index: origins[index],
                reverse=origin_side == "maximum",
            )
        )
        for offset, index in enumerate(ordered):
            if origin_side == "minimum":
                end = origins[ordered[offset + 1]] - 1 if offset + 1 < len(ordered) else global_axis_end
                axis_ranges[index] = (origins[index], end)
            else:
                start = origins[ordered[offset + 1]] + 1 if offset + 1 < len(ordered) else global_axis_start
                axis_ranges[index] = (start, origins[index])
    bands: list[tuple[str, int, int]] = []
    boxes: list[tuple[str, int, int, int, int]] = []
    capacities: list[int] = []
    for index, item in enumerate(material):
        orth_start, orth_end = orthogonal_bands[group_by_index[index]]
        axis_start, axis_end = axis_ranges[index]
        axis_extent = axis_end - axis_start + 1
        if axis_extent < quantum or axis_extent % quantum:
            return (), (), (), ()
        bands.append((item.component_id, orth_start, orth_end))
        if axis == "column":
            boxes.append((item.component_id, orth_start, axis_start, orth_end, axis_end))
        else:
            boxes.append((item.component_id, axis_start, orth_start, axis_end, orth_end))
        capacities.append(axis_extent // quantum)
    return tuple(bands), tuple(boxes), origins, tuple(capacities)


def _measure_interfaces(
    *,
    components: tuple[ComponentDescription, ...],
    material: tuple[ComponentDescription, ...],
    pairs: tuple[tuple[int, int, str, str, int, int, int], ...],
    material_value: int,
    axis: str,
    origin_side: str,
    quantum: int,
) -> tuple[_MeasuredInterface, ...]:
    excluded_refs = frozenset(
        {
            *(item.component_id for item in material),
            *(str(item) for pair in pairs for item in pair[2:4]),
        }
    )
    by_morphology: dict[tuple[object, ...], list[ComponentDescription]] = defaultdict(list)
    for component in components:
        if (
            component.component_id not in excluded_refs
            and component.value != material_value
            and (_orthogonal_end(component, axis) - _orthogonal_start(component, axis) + 1)
            == quantum
            and _axis_extent(component, axis) > quantum
        ):
            by_morphology[_component_morphology(component)].append(component)
    outputs: list[tuple[_MeasuredInterface, ...]] = []
    for members in (by_morphology[__yf_order_key] for __yf_order_key in sorted(by_morphology)):
        by_edge: dict[frozenset[int], list[_MeasuredInterface]] = defaultdict(list)
        for component in members:
            candidates: list[tuple[tuple[int, int, int, int], _MeasuredInterface]] = []
            for left, right, _left_ref, _right_ref, _anchor, gap_start, gap_end in pairs:
                if (
                    _orthogonal_start(component, axis) != gap_start
                    or _orthogonal_end(component, axis) != gap_end
                ):
                    continue
                left_distance = _target_extent(
                    component,
                    axis=axis,
                    origin_side=origin_side,
                    origin=_axis_origin(material[left], axis, origin_side),
                )
                right_distance = _target_extent(
                    component,
                    axis=axis,
                    origin_side=origin_side,
                    origin=_axis_origin(material[right], axis, origin_side),
                )
                if (
                    left_distance <= 0
                    or right_distance <= 0
                    or left_distance % quantum
                    or right_distance % quantum
                ):
                    continue
                measured = _MeasuredInterface(
                        component_ref=component.component_id,
                        left_index=left,
                        right_index=right,
                        left_requirement=left_distance // quantum,
                        right_requirement=right_distance // quantum,
                    )
                candidates.append(
                    (
                        (
                            measured.left_requirement + measured.right_requirement,
                            max(measured.left_requirement, measured.right_requirement),
                            measured.left_index,
                            measured.right_index,
                        ),
                        measured,
                    )
                )
            if not candidates:
                continue
            minimum_rank = min(item[0] for item in candidates)
            closest = tuple(item[1] for item in candidates if item[0] == minimum_rank)
            if len(closest) == 1:
                by_edge[closest[0].edge].append(closest[0])
        measured = [items[0] for items in (by_edge[__yf_order_key] for __yf_order_key in sorted(by_edge)) if len(items) == 1]
        if len(measured) >= 2 and len({item.edge for item in measured}) == len(measured):
            outputs.append(tuple(sorted(measured, key=lambda item: tuple(sorted(item.edge)))))
    if not outputs:
        return ()
    maximum = max(len(items) for items in outputs)
    best = tuple(items for items in outputs if len(items) == maximum)
    return best[0] if len(best) == 1 else ()


def _measure_identities(
    *,
    components: tuple[ComponentDescription, ...],
    material: tuple[ComponentDescription, ...],
    material_edges: tuple[tuple[int, int], ...],
    interfaces: tuple[_MeasuredInterface, ...],
    identity_edges: tuple[tuple[int, int], ...],
    material_bands: tuple[tuple[str, int, int], ...],
    zone_boxes: tuple[tuple[str, int, int, int, int], ...],
    origins: tuple[int, ...],
    capacities: tuple[int, ...],
    tracked_by_component: dict[str, str],
    material_value: int,
    control_refs: frozenset[str],
    axis: str,
    origin_side: str,
    quantum: int,
) -> tuple[
    tuple[_MeasuredIdentity, ...],
    tuple[str, str],
    tuple[str, ...],
    tuple[int, ...],
] | None:
    excluded_refs = frozenset(
        {
            *(item.component_id for item in material),
            *control_refs,
            *(item.component_ref for item in interfaces),
        }
    )
    morphology_groups: dict[tuple[object, ...], list[ComponentDescription]] = defaultdict(list)
    for component in components:
        if component.component_id not in excluded_refs:
            morphology_groups[_component_morphology(component)].append(component)
    seed_groups: list[tuple[_IdentitySeed, ...]] = []
    for carriers in (morphology_groups[__yf_order_key] for __yf_order_key in sorted(morphology_groups)):
        if not 3 <= len(carriers) <= _HARD_MAX_IDENTITIES:
            continue
        seeds: list[_IdentitySeed] = []
        valid = True
        carrier_refs = frozenset(item.component_id for item in carriers)
        for carrier in carriers:
            carrier_entity_ref = tracked_by_component.get(carrier.component_id)
            current_index = _zone_index(carrier, zone_boxes)
            if carrier_entity_ref is None or current_index is None:
                valid = False
                break
            attribute_matches: list[
                tuple[ComponentDescription, ComponentDescription, tuple[_EndpointMeasurement, ...]]
            ] = []
            for attribute in components:
                if (
                    attribute.component_id in excluded_refs
                    or attribute.component_id in carrier_refs
                    or not _components_are_locally_adjacent(carrier, attribute, quantum)
                    or _zone_index(attribute, zone_boxes) != current_index
                ):
                    continue
                targets = tuple(
                    item
                    for item in components
                    if item.component_id not in excluded_refs
                    and item.component_id not in carrier_refs
                    and item.component_id != attribute.component_id
                    and item.value == attribute.value
                    and not any(
                        _components_are_locally_adjacent(other, item, quantum)
                        for other in carriers
                    )
                )
                if len(targets) != 1:
                    continue
                endpoints = _target_endpoint_measurements(
                    target=targets[0],
                    current_index=current_index,
                    material=material,
                    material_edges=material_edges,
                    identity_edges=identity_edges,
                    material_bands=material_bands,
                    origins=origins,
                    capacities=capacities,
                    axis=axis,
                    origin_side=origin_side,
                    quantum=quantum,
                )
                if endpoints:
                    attribute_matches.append((attribute, targets[0], endpoints))
            if len(attribute_matches) != 1:
                valid = False
                break
            attribute, target, endpoints = attribute_matches[0]
            attribute_entity_ref = tracked_by_component.get(attribute.component_id)
            if attribute_entity_ref is None:
                valid = False
                break
            seeds.append(
                _IdentitySeed(
                    identity_ref=carrier_entity_ref,
                    carrier_entity_ref=carrier_entity_ref,
                    attribute_entity_ref=attribute_entity_ref,
                    carrier_component_ref=carrier.component_id,
                    attribute_component_ref=attribute.component_id,
                    target_component_ref=target.component_id,
                    current_index=current_index,
                    attribute_offset=(
                        attribute.bbox.top - carrier.bbox.top,
                        attribute.bbox.left - carrier.bbox.left,
                    ),
                    endpoint_measurements=endpoints,
                )
            )
        if valid and len(seeds) == len(carriers):
            seed_groups.append(tuple(sorted(seeds, key=lambda item: item.identity_ref)))
    if len(seed_groups) != 1:
        return None
    seeds = seed_groups[0]
    reciprocal_matches: list[
        tuple[int, _EndpointMeasurement, int, _EndpointMeasurement]
    ] = []
    for left_index, left in enumerate(seeds):
        for right_index, right in enumerate(seeds[left_index + 1 :], left_index + 1):
            for left_endpoint in left.endpoint_measurements:
                for right_endpoint in right.endpoint_measurements:
                    if (
                        len(left_endpoint.path) >= 3
                        and left_endpoint.path == tuple(reversed(right_endpoint.path))
                    ):
                        reciprocal_matches.append(
                            (left_index, left_endpoint, right_index, right_endpoint)
                        )
    unique_matches = {
        (
            left_index,
            left_endpoint.target_index,
            left_endpoint.path,
            right_index,
            right_endpoint.target_index,
            right_endpoint.path,
        ): (left_index, left_endpoint, right_index, right_endpoint)
        for left_index, left_endpoint, right_index, right_endpoint in reciprocal_matches
    }
    if len(unique_matches) != 1:
        return None
    left_index, left_endpoint, right_index, right_endpoint = next(
        iter(unique_matches.values())
    )
    endpoint_by_seed: dict[int, _EndpointMeasurement] = {
        left_index: left_endpoint,
        right_index: right_endpoint,
    }
    independent_indices = tuple(
        index for index in range(len(seeds)) if index not in endpoint_by_seed
    )
    if not independent_indices:
        return None
    reciprocal_path_nodes = frozenset(left_endpoint.path)
    for index in independent_indices:
        endpoints = seeds[index].endpoint_measurements
        if len(endpoints) != 1 or endpoints[0].target_index not in reciprocal_path_nodes:
            return None
        endpoint_by_seed[index] = endpoints[0]
    measured = tuple(
        _MeasuredIdentity(
            identity_ref=seed.identity_ref,
            carrier_entity_ref=seed.carrier_entity_ref,
            attribute_entity_ref=seed.attribute_entity_ref,
            carrier_component_ref=seed.carrier_component_ref,
            attribute_component_ref=seed.attribute_component_ref,
            target_component_ref=seed.target_component_ref,
            current_index=seed.current_index,
            target_index=endpoint_by_seed[index].target_index,
            target_quantity=endpoint_by_seed[index].target_quantity,
            path=endpoint_by_seed[index].path,
            attribute_offset=seed.attribute_offset,
        )
        for index, seed in enumerate(seeds)
    )
    reciprocal_refs = (seeds[left_index].identity_ref, seeds[right_index].identity_ref)
    independent_refs = tuple(seeds[index].identity_ref for index in independent_indices)
    return measured, reciprocal_refs, independent_refs, left_endpoint.path


def _target_endpoint_measurements(
    *,
    target: ComponentDescription,
    current_index: int,
    material: tuple[ComponentDescription, ...],
    material_edges: tuple[tuple[int, int], ...],
    identity_edges: tuple[tuple[int, int], ...],
    material_bands: tuple[tuple[str, int, int], ...],
    origins: tuple[int, ...],
    capacities: tuple[int, ...],
    axis: str,
    origin_side: str,
    quantum: int,
) -> tuple[_EndpointMeasurement, ...]:
    orth_start = _orthogonal_start(target, axis)
    orth_end = _orthogonal_end(target, axis)
    incident_edges = tuple(
        (left, right)
        for left, right in material_edges
        if (
            max(material_bands[left][1], material_bands[right][1])
            <= orth_start
            <= orth_end
            <= min(material_bands[left][2], material_bands[right][2])
        )
        or (
            min(material_bands[left][2], material_bands[right][2]) + 1
            <= orth_start
            and orth_end
            <= max(material_bands[left][1], material_bands[right][1]) - 1
        )
    )
    if not incident_edges:
        # Material bands meet at their measured control gap.  Compare the raw
        # material intervals when a partition band owns the gap cells.
        incident_edges = tuple(
            (left, right)
            for left, right in material_edges
            if min(
                _orthogonal_end(material[left], axis),
                _orthogonal_end(material[right], axis),
            )
            < orth_start
            and orth_end
            < max(
                _orthogonal_start(material[left], axis),
                _orthogonal_start(material[right], axis),
            )
        )
    outputs: dict[int, _EndpointMeasurement] = {}
    for left, right in incident_edges:
        for endpoint in (left, right):
            distance = _target_extent(
                target,
                axis=axis,
                origin_side=origin_side,
                origin=origins[endpoint],
            )
            if distance <= 0 or distance % quantum:
                continue
            quantity = distance // quantum
            if quantity > capacities[endpoint]:
                continue
            path = _first_index_path(
                node_count=len(material),
                edges=identity_edges,
                source=current_index,
                target=endpoint,
            )
            if len(path) >= 2:
                outputs.setdefault(
                    endpoint,
                    _EndpointMeasurement(
                        target_index=endpoint,
                        target_quantity=quantity,
                        path=path,
                    ),
                )
    return tuple(outputs[index] for index in sorted(outputs))


def _simulate_declared_order(
    *,
    measurement: _GraphMeasurement,
    groups: tuple[tuple[str, ...], ...],
    max_expanded_states: int,
    max_plan_actions: int,
) -> tuple[_Simulation, ...]:
    identity_by_ref = {item.identity_ref: item for item in measurement.identities}
    if set(identity_by_ref) != {item for group in groups for item in group}:
        return ()
    simulations = (
        _Simulation(
            quantities=measurement.quantities,
            identity_indices={
                item.identity_ref: item.current_index for item in measurement.identities
            },
            steps=[],
        ),
    )
    reciprocal_set = frozenset(measurement.reciprocal_identity_refs)
    for group in groups:
        next_simulations: list[_Simulation] = []
        for simulation in simulations:
            if len(group) == 2 and frozenset(group) == reciprocal_set:
                next_simulations.extend(
                    _append_reciprocal_group(
                        simulation=simulation,
                        measurement=measurement,
                        left=identity_by_ref[group[0]],
                        right=identity_by_ref[group[1]],
                        max_expanded_states=max_expanded_states,
                        max_plan_actions=max_plan_actions,
                    )
                )
            elif len(group) == 1 and group[0] in identity_by_ref:
                current = simulation.copy()
                if _append_single_identity_route(
                    simulation=current,
                    measurement=measurement,
                    identity=identity_by_ref[group[0]],
                    max_expanded_states=max_expanded_states,
                    max_plan_actions=max_plan_actions,
                ):
                    next_simulations.append(current)
        simulations = tuple(next_simulations[:_HARD_MAX_ROUTE_VARIANTS])
        if not simulations:
            return ()
    completed: list[_Simulation] = []
    for simulation in simulations:
        if any(
            simulation.identity_indices[item.identity_ref] != item.target_index
            for item in measurement.identities
        ):
            continue
        current = simulation.copy()
        remaining_actions = max_plan_actions - len(current.steps)
        requirements = dict(measurement.terminal_requirements)
        witness = _first_shortest_route_witness(
            initial=current.quantities,
            controls=measurement.controls,
            goal_test=lambda state: all(
                state[index] == expected for index, expected in sorted(requirements.items())
            ),
            capacities=measurement.capacities,
            max_expanded_states=max_expanded_states,
            max_depth=max(0, remaining_actions),
            control_order_variant="canonical",
        )
        if (
            witness is None
            or current.expanded_state_count + witness.expanded_state_count
            > _HARD_MAX_TOTAL_EXPANDED_STATES
        ):
            continue
        current.expanded_state_count += witness.expanded_state_count
        _append_quantity_steps(current, witness.route, stage="terminal_alignment")
        if len(current.steps) <= max_plan_actions:
            completed.append(current)
    return tuple(completed[:_HARD_MAX_ROUTE_VARIANTS])


def _append_reciprocal_group(
    *,
    simulation: _Simulation,
    measurement: _GraphMeasurement,
    left: _MeasuredIdentity,
    right: _MeasuredIdentity,
    max_expanded_states: int,
    max_plan_actions: int,
) -> tuple[_Simulation, ...]:
    path = left.path
    if path != tuple(reversed(right.path)):
        return ()
    interface_by_edge = {item.edge: item for item in measurement.interfaces}
    if any(frozenset(edge) not in interface_by_edge for edge in zip(path, path[1:])):
        return ()
    outputs: list[_Simulation] = []
    for crossing_index in range(len(path) - 1):
        current = simulation.copy()
        valid = True
        for edge_index in range(crossing_index):
            interface = interface_by_edge[frozenset((path[edge_index], path[edge_index + 1]))]
            if not _prepare_and_move(
                simulation=current,
                measurement=measurement,
                interface=interface,
                moves=((left.identity_ref, path[edge_index], path[edge_index + 1]),),
                stage="pre_crossing",
                max_expanded_states=max_expanded_states,
                max_plan_actions=max_plan_actions,
            ):
                valid = False
                break
        if not valid:
            continue
        for edge_index in range(len(path) - 2, crossing_index, -1):
            interface = interface_by_edge[frozenset((path[edge_index], path[edge_index + 1]))]
            if not _prepare_and_move(
                simulation=current,
                measurement=measurement,
                interface=interface,
                moves=((right.identity_ref, path[edge_index + 1], path[edge_index]),),
                stage="pre_crossing",
                max_expanded_states=max_expanded_states,
                max_plan_actions=max_plan_actions,
            ):
                valid = False
                break
        if not valid:
            continue
        crossing = interface_by_edge[
            frozenset((path[crossing_index], path[crossing_index + 1]))
        ]
        if not _prepare_and_move(
            simulation=current,
            measurement=measurement,
            interface=crossing,
            moves=(
                (left.identity_ref, path[crossing_index], path[crossing_index + 1]),
                (right.identity_ref, path[crossing_index + 1], path[crossing_index]),
            ),
            stage="atomic_crossing",
            max_expanded_states=max_expanded_states,
            max_plan_actions=max_plan_actions,
        ):
            continue
        left_remaining = len(path) - crossing_index - 2
        right_remaining = crossing_index
        current.maximum_remaining_route_distance_after_crossing = max(
            left_remaining, right_remaining
        )
        current.remaining_route_distance_imbalance_after_crossing = abs(
            left_remaining - right_remaining
        )
        for edge_index in range(crossing_index + 1, len(path) - 1):
            interface = interface_by_edge[frozenset((path[edge_index], path[edge_index + 1]))]
            if not _prepare_and_move(
                simulation=current,
                measurement=measurement,
                interface=interface,
                moves=((left.identity_ref, path[edge_index], path[edge_index + 1]),),
                stage="post_crossing",
                max_expanded_states=max_expanded_states,
                max_plan_actions=max_plan_actions,
            ):
                valid = False
                break
        if not valid:
            continue
        for edge_index in range(crossing_index - 1, -1, -1):
            interface = interface_by_edge[frozenset((path[edge_index], path[edge_index + 1]))]
            if not _prepare_and_move(
                simulation=current,
                measurement=measurement,
                interface=interface,
                moves=((right.identity_ref, path[edge_index + 1], path[edge_index]),),
                stage="post_crossing",
                max_expanded_states=max_expanded_states,
                max_plan_actions=max_plan_actions,
            ):
                valid = False
                break
        if valid:
            outputs.append(current)
    return tuple(outputs[:_HARD_MAX_ROUTE_VARIANTS])


def _append_single_identity_route(
    *,
    simulation: _Simulation,
    measurement: _GraphMeasurement,
    identity: _MeasuredIdentity,
    max_expanded_states: int,
    max_plan_actions: int,
) -> bool:
    interface_by_edge = {item.edge: item for item in measurement.interfaces}
    for source, destination in zip(identity.path, identity.path[1:]):
        interface = interface_by_edge.get(frozenset((source, destination)))
        if interface is None or not _prepare_and_move(
            simulation=simulation,
            measurement=measurement,
            interface=interface,
            moves=((identity.identity_ref, source, destination),),
            stage="independent_transport",
            max_expanded_states=max_expanded_states,
            max_plan_actions=max_plan_actions,
        ):
            return False
    return True


def _prepare_and_move(
    *,
    simulation: _Simulation,
    measurement: _GraphMeasurement,
    interface: _MeasuredInterface,
    moves: tuple[tuple[str, int, int], ...],
    stage: str,
    max_expanded_states: int,
    max_plan_actions: int,
) -> bool:
    requirements = interface.requirement_by_index()
    remaining_actions = max_plan_actions - len(simulation.steps)
    witness = _first_shortest_route_witness(
        initial=simulation.quantities,
        controls=measurement.controls,
        goal_test=lambda state: all(
            state[index] == expected for index, expected in sorted(requirements.items())
        ),
        capacities=measurement.capacities,
        max_expanded_states=max_expanded_states,
        max_depth=max(0, remaining_actions - 1),
        control_order_variant="canonical",
    )
    if witness is None:
        return False
    if (
        simulation.expanded_state_count + witness.expanded_state_count
        > _HARD_MAX_TOTAL_EXPANDED_STATES
    ):
        return False
    simulation.expanded_state_count += witness.expanded_state_count
    _append_quantity_steps(simulation, witness.route, stage=stage)
    if len(simulation.steps) >= max_plan_actions:
        return False
    for identity_ref, source, _destination in moves:
        if simulation.identity_indices.get(identity_ref) != source:
            return False
    for identity_ref, _source, destination in moves:
        simulation.identity_indices[identity_ref] = destination
    first_source = moves[0][1]
    first_destination = moves[0][2]
    simulation.steps.append(
        _PlanStep(
            source_index=first_source,
            destination_index=first_destination,
            component_ref=interface.component_ref,
            measurement_kind="multi_identity_interface_transition_candidate",
            quantities_after=simulation.quantities,
            identity_indices_after=tuple(sorted(simulation.identity_indices.items())),
            moved_identity_refs=tuple(item[0] for item in moves),
            stage=stage,
        )
    )
    return True


def _append_quantity_steps(
    simulation: _Simulation,
    route: tuple[tuple[int, int, str], ...],
    *,
    stage: str,
) -> None:
    for source, destination, component_ref in route:
        quantities = list(simulation.quantities)
        quantities[source] -= 1
        quantities[destination] += 1
        simulation.quantities = tuple(quantities)
        simulation.steps.append(
            _PlanStep(
                source_index=source,
                destination_index=destination,
                component_ref=component_ref,
                measurement_kind="extent_exchange_candidate",
                quantities_after=simulation.quantities,
                identity_indices_after=tuple(sorted(simulation.identity_indices.items())),
                moved_identity_refs=(),
                stage=stage,
            )
        )


def _plan_facts(
    *,
    value: RouteCompatibleGraphPlanInput,
    measurement: _GraphMeasurement,
    groups: tuple[tuple[str, ...], ...],
    simulation: _Simulation,
    route_ref: str,
    candidate_ref: str,
    route_digest: str,
    route_variants_materialized: int,
    variant_limit: int,
    component_by_ref: dict[str, ComponentDescription],
    tracked_by_component: dict[str, str],
) -> FrozenMap:
    material_refs = tuple(item.component_id for item in measurement.material)
    route_steps = tuple(
        (
            material_refs[step.source_index],
            material_refs[step.destination_index],
            step.component_ref,
            step.measurement_kind,
        )
        for step in simulation.steps
    )
    route_quantity_states = tuple(
        tuple(zip(material_refs, step.quantities_after)) for step in simulation.steps
    )
    route_identity_states = tuple(
        tuple((identity_ref, material_refs[index]) for identity_ref, index in step.identity_indices_after)
        for step in simulation.steps
    )
    route_moved_identity_refs = tuple(step.moved_identity_refs for step in simulation.steps)
    route_component_refs = tuple(step.component_ref for step in simulation.steps)
    all_component_refs = tuple(dict.fromkeys(route_component_refs))
    first_component_ref = route_component_refs[0]
    first_identity = measurement.identities[0]
    first_requirement_index, first_requirement_value = measurement.terminal_requirements[0]
    display_digest = hashlib.sha256(
        canonical_json(measurement.terminal_requirements).encode("utf-8")
    ).hexdigest()[:12]
    checkpoint_interval = max(
        1, int(value.continuation_policy.get("regular_checkpoint_interval") or 1)
    )
    checkpoint_reason = _checkpoint_reason(
        steps=route_steps,
        executed_component_refs=(),
        completed_click_count=0,
        last_full_checkpoint_click_count=0,
        interval=checkpoint_interval,
        enabled_conditions=tuple(
            str(item) for item in value.continuation_policy.get("checkpoint_on", ())
        ),
    )
    phase_count = 1 + sum(
        left != right for left, right in zip(route_component_refs, route_component_refs[1:])
    )
    material_graph_edges = tuple(
        (material_refs[left], material_refs[right]) for left, right in measurement.material_edges
    )
    identity_graph_edges = tuple(
        (material_refs[left], material_refs[right]) for left, right in measurement.identity_edges
    )
    material_degrees = _node_degrees(len(material_refs), measurement.material_edges)
    identity_degrees = _node_degrees(len(material_refs), measurement.identity_edges)
    terminal_unknown_indices = tuple(
        index
        for index in range(len(material_refs))
        if index not in dict(measurement.terminal_requirements)
    )
    latest_mechanism = next(
        (
            item
            for item in reversed(value.mechanism_evidence)
            if item.quantum > 0 and item.mechanism_kind == "material_transfer"
        ),
        None,
    )
    inherited_quanta = () if latest_mechanism is None else (latest_mechanism.quantum,)
    prior_axes = () if latest_mechanism is None else (latest_mechanism.alignment_axis,)
    action_points = tuple(
        (
            component_ref,
            _interior_point(component_by_ref[component_ref])[1],
            _interior_point(component_by_ref[component_ref])[0],
        )
        for component_ref in all_component_refs
        if component_ref in component_by_ref
    )
    facts = {
        "candidate_ref": candidate_ref,
        "component_ref": first_component_ref,
        "tracked_entity_ref": tracked_by_component.get(first_component_ref),
        "route_simulation_completed": True,
        "all_intermediate_states_valid": True,
        "aggregate_conserved": all(
            sum(step.quantities_after) == sum(measurement.quantities)
            for step in simulation.steps
        ),
        "display_constraint_satisfied_by_simulation": all(
            simulation.quantities[index] == expected
            for index, expected in measurement.terminal_requirements
        ),
        "first_action_moves_reusable_extent_toward_displayed_constraint": False,
        "primitive_action_count": len(simulation.steps),
        "transfer_phase_count": phase_count,
        "actuator_switch_count": phase_count - 1,
        "route_ref": route_ref,
        "plan_structure_digest": route_digest,
        "route_variant_limit": variant_limit,
        "route_variants_materialized": route_variants_materialized,
        "current_scope_attempted_route_suffixes": tuple(
            sorted(value.current_scope_attempted_route_suffixes)
        ),
        "remaining_global_route_attempt_budget": (
            _HARD_MAX_ROUTE_VARIANTS - len(value.current_scope_attempted_route_suffixes)
        ),
        "route_control_order_variant": "canonical",
        "structural_descriptions_examined": 1,
        "expanded_state_count": simulation.expanded_state_count,
        "current_context_no_effect_observed": False,
        "route_component_refs": route_component_refs,
        "all_route_component_refs": all_component_refs,
        "route_transfer_steps": route_steps,
        "route_quantity_states": route_quantity_states,
        "route_marker_states": (),
        "route_identity_states": route_identity_states,
        "route_moved_identity_refs": route_moved_identity_refs,
        "route_component_entity_bindings": tuple(
            (component_ref, tracked_by_component[component_ref])
            for component_ref in all_component_refs
            if component_ref in tracked_by_component
        ),
        "route_action_points": action_points,
        "executed_component_refs": (),
        "current_quantities": tuple(zip(material_refs, measurement.quantities)),
        "expected_quantities_after_action": route_quantity_states[0],
        "expected_identity_zones_after_action": route_identity_states[0],
        "expected_moved_identity_refs_after_action": route_moved_identity_refs[0],
        "current_step_measurement_kind": simulation.steps[0].measurement_kind,
        "display_constraint_extent_ref": material_refs[first_requirement_index],
        "display_constraint_quantity": first_requirement_value,
        "display_constraint_digest": display_digest,
        "moving_entity_ref": first_identity.identity_ref,
        "fixed_entity_ref": tracked_by_component.get(first_identity.target_component_ref)
        or first_identity.target_component_ref,
        "alignment_axis": measurement.axis,
        "origin_side": measurement.origin_side,
        "material_value": measurement.material_value,
        "material_chain_bands": measurement.material_bands,
        "material_axis_origin_coordinate": measurement.material_origins[0],
        "material_axis_origin_coordinates": measurement.material_origins,
        "moving_marker_ref": first_identity.carrier_component_ref,
        "fixed_marker_ref": first_identity.target_component_ref,
        "moving_marker_signature": (
            component_by_ref[first_identity.carrier_component_ref].value,
            component_by_ref[first_identity.carrier_component_ref].relative_pixels,
        ),
        "quantum": measurement.quantum,
        "inherited_visual_quanta": inherited_quanta,
        "current_visual_quantum": measurement.quantum,
        "global_scale_relation": (
            "scale.unbound"
            if not inherited_quanta
            else _scale_relation(
                inherited_quanta=inherited_quanta,
                current_quantum=measurement.quantum,
            )
        ),
        "prior_local_axes": prior_axes,
        "current_local_axis": measurement.axis,
        "global_rotation_relation": (
            "rotation.local_axis_unbound"
            if not prior_axes
            else (
                "rotation.local_axis_preserved"
                if measurement.axis in prior_axes
                else "rotation.quarter_turn_local_frame"
            )
        ),
        "configuration_kind": "route_compatible_graph_hybrid",
        "descriptor_kind": "route_compatible_graph_hybrid",
        "displayed_quantity_vector": tuple(
            (material_refs[index], expected)
            for index, expected in measurement.terminal_requirements
        ),
        "displayed_marker_bindings": tuple(
            (
                item.identity_ref,
                item.target_component_ref,
                material_refs[item.target_index],
                item.target_quantity,
            )
            for item in measurement.identities
        ),
        "capacities": measurement.capacities,
        "capacity_measurement_available": True,
        "inherited_mechanism_evidence_refs": tuple(
            item.evidence_ref for item in value.mechanism_evidence
        ),
        "remaining_click_count": len(simulation.steps),
        "remaining_step_count": len(simulation.steps),
        "route_total_click_count": len(simulation.steps),
        "executed_click_count": 0,
        "current_step_index": 1,
        "last_full_checkpoint_click_count": 0,
        "checkpoint_interval": checkpoint_interval,
        "checkpoint_after_current_step": bool(checkpoint_reason),
        "checkpoint_reason_after_current_step": checkpoint_reason,
        "typed_relation_graphs_measured": True,
        "declared_route_order_applied": True,
        "declared_route_order_ref": value.declared_route_order_ref,
        "ordered_identity_groups": groups,
        "material_graph_edges": material_graph_edges,
        "identity_graph_edges": identity_graph_edges,
        "material_graph_node_degrees": tuple(zip(material_refs, material_degrees)),
        "identity_graph_node_degrees": tuple(zip(material_refs, identity_degrees)),
        "identity_route_measurements": tuple(
            (
                item.identity_ref,
                tuple(material_refs[index] for index in item.path),
                item.target_quantity,
            )
            for item in measurement.identities
        ),
        "reciprocal_identity_refs": measurement.reciprocal_identity_refs,
        "independent_identity_refs": measurement.independent_identity_refs,
        "local_interface_coordinate_measurements": tuple(
            (
                item.component_ref,
                material_refs[item.left_index],
                item.left_requirement,
                material_refs[item.right_index],
                item.right_requirement,
            )
            for item in measurement.interfaces
        ),
        "identity_composite_bindings": tuple(
            (
                item.identity_ref,
                item.carrier_entity_ref,
                item.attribute_entity_ref,
                item.attribute_offset[0],
                item.attribute_offset[1],
            )
            for item in measurement.identities
        ),
        "identity_carrier_signatures": tuple(
            (
                item.identity_ref,
                component_by_ref[item.carrier_component_ref].value,
                component_by_ref[item.carrier_component_ref].relative_pixels,
            )
            for item in measurement.identities
        ),
        "composite_link_measurements": tuple(
            (item.identity_ref, "adjacent", item.attribute_offset, False)
            for item in measurement.identities
        ),
        "terminal_constraint_kind": (
            "exact_observed_supports_with_conserved_capacity_safe_residual"
        ),
        "terminal_quantity_requirements": measurement.terminal_requirements,
        "terminal_conserved_aggregate": sum(measurement.quantities),
        "terminal_unknown_zone_indices": terminal_unknown_indices,
        "terminal_witness_quantities": tuple(
            zip(material_refs, simulation.quantities)
        ),
        "zone_partition_bands": measurement.material_bands,
        "zone_measurement_boxes": measurement.zone_boxes,
        "screen_component_boxes": tuple(
            (
                item.component_id,
                item.bbox.top,
                item.bbox.left,
                item.bbox.bottom,
                item.bbox.right,
            )
            for item in measurement.material
        ),
        "maximum_remaining_route_distance_after_crossing": (
            simulation.maximum_remaining_route_distance_after_crossing
        ),
        "remaining_route_distance_imbalance_after_crossing": (
            simulation.remaining_route_distance_imbalance_after_crossing
        ),
        "atomic_crossing_identity_count": 2,
        "all_identity_transports_before_terminal_alignment": _identity_steps_precede_terminal(
            simulation.steps
        ),
        "topology_gap_evidence": False,
    }
    return FrozenMap(facts)


def _identity_steps_precede_terminal(steps: list[_PlanStep]) -> bool:
    terminal_started = False
    for step in steps:
        if step.stage == "terminal_alignment":
            terminal_started = True
        elif terminal_started and step.measurement_kind == "multi_identity_interface_transition_candidate":
            return False
    return True


def _zone_index(
    component: ComponentDescription,
    boxes: tuple[tuple[str, int, int, int, int], ...],
) -> int | None:
    row = (component.bbox.top + component.bbox.bottom) // 2
    column = (component.bbox.left + component.bbox.right) // 2
    matches = tuple(
        index
        for index, (_ref, top, left, bottom, right) in enumerate(boxes)
        if top <= row <= bottom and left <= column <= right
    )
    return matches[0] if len(matches) == 1 else None


def _graph_is_connected(
    node_count: int,
    edges: tuple[tuple[int, int], ...],
) -> bool:
    if node_count < 1:
        return False
    neighbors = {index: set[int]() for index in range(node_count)}
    for left, right in edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    reached = {0}
    frontier = [0]
    while frontier:
        current = frontier.pop()
        for neighbor in neighbors[current] - reached:
            reached.add(neighbor)
            frontier.append(neighbor)
    return len(reached) == node_count


def _node_degrees(
    node_count: int,
    edges: tuple[tuple[int, int], ...],
) -> tuple[int, ...]:
    values = [0] * node_count
    for left, right in edges:
        values[left] += 1
        values[right] += 1
    return tuple(values)


def _bounded_integer(
    facts: FrozenMap,
    field: str,
    *,
    default: int,
    hard_maximum: int,
) -> int:
    raw = facts.get(field)
    try:
        value = default if raw is None else int(raw)
    except (TypeError, ValueError):
        value = default
    return min(max(1, value), hard_maximum)


def _empty_agenda() -> InteractionProbeAgenda:
    return InteractionProbeAgenda(
        candidates=(),
        alternative_refs=(),
        alternative_facts=FrozenMap(),
        has_candidates=False,
    )


__all__ = (
    "analyze_route_compatible_orders",
    "enumerate_route_compatible_graph_plans",
)
