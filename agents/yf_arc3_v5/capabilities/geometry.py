"""Pure pairwise geometry over descriptive components."""

from __future__ import annotations

from itertools import combinations

from agents.yf_arc3_v5.capabilities.contracts import (
    BoundingBox,
    AffinePositionMeasurement,
    ComponentDescription,
    ComponentPairGeometry,
    GeometryInput,
    GeometryResult,
    OffLatticeQuartetCandidate,
    OrthogonalPerimeterAssemblyInput,
    OrthogonalPerimeterAssemblyMeasurements,
    OrthogonalPerimeterCandidate,
    DerivedGeometryStateInput,
    DerivedGeometryStateMeasurements,
    DiscreteStateReturnMeasurement,
    DiscreteStateSample,
    DiscreteStateTraceMeasurement,
    DiscreteStateTransitionMeasurement,
)
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest


def describe_component_geometry(value: GeometryInput) -> GeometryResult:
    pairs: list[ComponentPairGeometry] = []
    for first, second in combinations(value.components.components, 2):
        first_pixels = set(first.pixels)
        second_pixels = set(second.pixels)
        overlap = bool(first_pixels & second_pixels)
        minimum_gap = min(
            abs(first_row - second_row) + abs(first_col - second_col)
            for first_row, first_col in first_pixels
            for second_row, second_col in second_pixels
        )
        edge_contact = minimum_gap == 1
        corner_contact = not edge_contact and any(
            max(abs(first_row - second_row), abs(first_col - second_col)) == 1
            for first_row, first_col in first_pixels
            for second_row, second_col in second_pixels
        )
        first_center = _center_twice(first)
        second_center = _center_twice(second)
        pairs.append(
            ComponentPairGeometry(
                first_component_id=first.component_id,
                second_component_id=second.component_id,
                overlap=overlap,
                edge_contact=edge_contact,
                corner_contact=corner_contact,
                first_bbox_contains_second=_contains(first.bbox, second.bbox),
                second_bbox_contains_first=_contains(second.bbox, first.bbox),
                same_center_row=first_center[0] == second_center[0],
                same_center_column=first_center[1] == second_center[1],
                center_delta_twice=(
                    second_center[0] - first_center[0],
                    second_center[1] - first_center[1],
                ),
                minimum_manhattan_pixel_gap=minimum_gap,
            )
        )
    return GeometryResult(pairs=tuple(pairs))


def measure_orthogonal_perimeter_assemblies(
    value: OrthogonalPerimeterAssemblyInput,
) -> OrthogonalPerimeterAssemblyMeasurements:
    """Measure bounded closed-slot and off-lattice quartet geometry only."""

    shape_groups: dict[tuple[object, ...], list[ComponentDescription]] = {}
    for component in value.components.components:
        if (
            component.bbox.height != component.bbox.width
            or component.area != component.bbox.height * component.bbox.width
        ):
            continue
        key = (
            component.bbox.height,
            component.bbox.width,
            component.relative_pixels,
        )
        shape_groups.setdefault(key, []).append(component)

    described_perimeters: list[OrthogonalPerimeterCandidate] = []
    for components in (shape_groups[__yf_order_key] for __yf_order_key in sorted(shape_groups)):
        if len(components) < 8 or len(components) > 64:
            continue
        by_center = {_center_twice(component): component for component in components}
        if len(by_center) != len(components):
            continue
        rows = tuple(sorted({center[0] for center in by_center}))
        columns = tuple(sorted({center[1] for center in by_center}))
        row_step_twice = _unique_positive_step(rows)
        column_step_twice = _unique_positive_step(columns)
        if (
            len(rows) < 3
            or len(columns) < 3
            or row_step_twice is None
            or column_step_twice is None
            or row_step_twice % 2
            or column_step_twice % 2
        ):
            continue
        boundary = frozenset(
            (row, column)
            for row in rows
            for column in columns
            if row in {rows[0], rows[-1]}
            or column in {columns[0], columns[-1]}
        )
        if frozenset(by_center) != boundary:
            continue
        ordered_centers = (
            tuple((rows[0], column) for column in columns)
            + tuple((row, columns[-1]) for row in rows[1:-1])
            + tuple((rows[-1], column) for column in reversed(columns))
            + tuple((row, columns[0]) for row in reversed(rows[1:-1]))
        )
        ordered_components = tuple(by_center[center] for center in ordered_centers)
        cell_height = ordered_components[0].bbox.height
        cell_width = ordered_components[0].bbox.width
        row_gap = row_step_twice // 2 - cell_height
        column_gap = column_step_twice // 2 - cell_width
        if row_gap < 1 or column_gap < 1:
            continue
        candidate_ref = (
            "measurement.orthogonal_perimeter."
            + stable_digest(
                (
                    ordered_centers,
                    tuple(component.component_id for component in ordered_components),
                )
            )[:16]
        )
        described_perimeters.append(
            OrthogonalPerimeterCandidate(
                candidate_ref=candidate_ref,
                component_refs=tuple(
                    component.component_id for component in ordered_components
                ),
                slot_centers_twice=ordered_centers,
                slot_values=tuple(component.value for component in ordered_components),
                cell_height=cell_height,
                cell_width=cell_width,
                row_gap=row_gap,
                column_gap=column_gap,
            )
        )
    described_perimeters.sort(
        key=lambda item: (-len(item.component_refs), item.candidate_ref)
    )
    perimeter_candidates = tuple(
        described_perimeters[: value.maximum_perimeter_candidates]
    )

    quartet_groups: dict[tuple[object, ...], list[ComponentDescription]] = {}
    for component in value.components.components:
        key = (
            component.value,
            component.bbox.height,
            component.bbox.width,
            component.relative_pixels,
        )
        quartet_groups.setdefault(key, []).append(component)
    quartets: list[OffLatticeQuartetCandidate] = []
    for components in (quartet_groups[__yf_order_key] for __yf_order_key in sorted(quartet_groups)):
        if len(components) != 4:
            continue
        centers = tuple(sorted(_center_twice(component) for component in components))
        rows = tuple(sorted({center[0] for center in centers}))
        columns = tuple(sorted({center[1] for center in centers}))
        if (
            len(rows) != 2
            or len(columns) != 2
            or frozenset(centers)
            != frozenset((row, column) for row in rows for column in columns)
        ):
            continue
        row_sum = sum(center[0] for center in centers)
        column_sum = sum(center[1] for center in centers)
        if row_sum % 4 or column_sum % 4:
            continue
        common_center = (row_sum // 4, column_sum // 4)
        for perimeter in perimeter_candidates:
            if common_center not in perimeter.slot_centers_twice:
                continue
            same_value_refs = tuple(
                component_ref
                for component_ref, component_value in zip(
                    perimeter.component_refs, perimeter.slot_values
                )
                if component_value == components[0].value
            )
            if len(same_value_refs) != 1:
                continue
            candidate_ref = (
                "measurement.off_lattice_quartet."
                + stable_digest(
                    (
                        tuple(component.component_id for component in components),
                        perimeter.candidate_ref,
                        common_center,
                    )
                )[:16]
            )
            quartets.append(
                OffLatticeQuartetCandidate(
                    candidate_ref=candidate_ref,
                    component_refs=tuple(
                        component.component_id for component in components
                    ),
                    component_centers_twice=centers,
                    shared_value=components[0].value,
                    component_height=components[0].bbox.height,
                    component_width=components[0].bbox.width,
                    common_center_twice=common_center,
                    perimeter_candidate_ref=perimeter.candidate_ref,
                    coincident_perimeter_slot_index=(
                        perimeter.slot_centers_twice.index(common_center)
                    ),
                    unique_same_value_perimeter_component_ref=same_value_refs[0],
                    fits_row_separator_gap=(
                        components[0].bbox.height <= perimeter.row_gap
                    ),
                    fits_column_separator_gap=(
                        components[0].bbox.width <= perimeter.column_gap
                    ),
                    off_perimeter_lattice=all(
                        center not in perimeter.slot_centers_twice
                        for center in centers
                    ),
                )
            )
    quartets.sort(key=lambda item: item.candidate_ref)
    quartet_candidates = tuple(quartets[: value.maximum_quartet_candidates])

    facts: dict[str, object] = {
        "orthogonal_perimeter_candidate_count": len(perimeter_candidates),
        "off_lattice_quartet_candidate_count": len(quartet_candidates),
    }
    if len(perimeter_candidates) == 1:
        perimeter = perimeter_candidates[0]
        facts.update(
            {
                "unique_orthogonal_perimeter": True,
                "orthogonal_perimeter_ref": perimeter.candidate_ref,
                "orthogonal_perimeter_component_refs": perimeter.component_refs,
                "orthogonal_perimeter_slot_centers_twice": (
                    perimeter.slot_centers_twice
                ),
                "orthogonal_perimeter_slot_values": perimeter.slot_values,
                "orthogonal_perimeter_slot_count": len(perimeter.component_refs),
                "orthogonal_perimeter_cell_height": perimeter.cell_height,
                "orthogonal_perimeter_cell_width": perimeter.cell_width,
                "orthogonal_perimeter_row_gap": perimeter.row_gap,
                "orthogonal_perimeter_column_gap": perimeter.column_gap,
            }
        )
    if len(quartet_candidates) == 1:
        quartet = quartet_candidates[0]
        facts.update(
            {
                "unique_off_lattice_quartet": True,
                "off_lattice_quartet_ref": quartet.candidate_ref,
                "off_lattice_quartet_component_refs": quartet.component_refs,
                "off_lattice_quartet_shared_value": quartet.shared_value,
                "off_lattice_quartet_common_center_twice": (
                    quartet.common_center_twice
                ),
                "off_lattice_quartet_perimeter_ref": (
                    quartet.perimeter_candidate_ref
                ),
                "off_lattice_quartet_slot_index": (
                    quartet.coincident_perimeter_slot_index
                ),
                "off_lattice_quartet_unique_same_value_perimeter_component_ref": (
                    quartet.unique_same_value_perimeter_component_ref
                ),
                "off_lattice_quartet_fits_separator_gaps": (
                    quartet.fits_row_separator_gap
                    and quartet.fits_column_separator_gap
                ),
                "off_lattice_quartet_centres_are_off_perimeter_lattice": (
                    quartet.off_perimeter_lattice
                ),
            }
        )
    if not perimeter_candidates:
        from .joint_slots import measure_joint_slot_geometry

        joint = measure_joint_slot_geometry(value.components.components)
        if joint:
            facts.update({
                "joint_slot_geometry_ref": joint["geometry_ref"],
                "joint_slot_structure_count": len(joint["structures"]),
                "joint_slot_marker_count": len(joint["markers"]),
                "joint_slot_association_count": len(joint["associations"]),
                "joint_slot_structures": joint["structures"],
                "joint_slot_marker_rows": joint["markers"],
                "joint_slot_associations": joint["associations"],
                "joint_slot_values_by_position": joint["slot_values_by_position"],
            })
    from .ordered_traits import measure_ordered_trait_arrays

    words = measure_ordered_trait_arrays(value.components.components)
    if words:
        facts.update({
            "ordered_trait_geometry_ref": words["geometry_ref"],
            "ordered_trait_array_count": len(words["arrays"]),
            "ordered_trait_arrays": words["structure"],
        })
    return OrthogonalPerimeterAssemblyMeasurements(
        perimeter_candidates=perimeter_candidates,
        off_lattice_quartet_candidates=quartet_candidates,
        descriptive_facts=FrozenMap(facts),
    )


def measure_derived_geometry_states(
    value: DerivedGeometryStateInput,
) -> DerivedGeometryStateMeasurements:
    """Measure exact finite affine positions and observed state traces only."""

    positions = {item.position_ref: item for item in value.positions}
    affine_positions: list[AffinePositionMeasurement] = []
    for request in value.affine_requests:
        row_numerator = request.row_offset_numerator + sum(
            term.integer_coefficient * positions[term.position_ref].row
            for term in request.terms
        )
        column_numerator = request.column_offset_numerator + sum(
            term.integer_coefficient * positions[term.position_ref].column
            for term in request.terms
        )
        integral = (
            row_numerator % request.positive_denominator == 0
            and column_numerator % request.positive_denominator == 0
        )
        derived_position = (
            (
                row_numerator // request.positive_denominator,
                column_numerator // request.positive_denominator,
            )
            if integral
            else None
        )
        target = (
            positions.get(request.observed_target_position_ref)
            if request.observed_target_position_ref is not None
            else None
        )
        affine_positions.append(
            AffinePositionMeasurement(
                request_ref=request.request_ref,
                row_numerator=row_numerator,
                column_numerator=column_numerator,
                positive_denominator=request.positive_denominator,
                integral_position=integral,
                derived_position=derived_position,
                observed_target_matches=(
                    None
                    if target is None or derived_position is None
                    else derived_position == (target.row, target.column)
                ),
            )
        )

    samples_by_entity: dict[str, list[DiscreteStateSample]] = {}
    for sample in value.discrete_state_samples:
        samples_by_entity.setdefault(sample.entity_ref, []).append(sample)
    traces: list[DiscreteStateTraceMeasurement] = []
    for entity_ref in sorted(samples_by_entity):
        samples = sorted(
            samples_by_entity[entity_ref], key=lambda item: item.observation_index
        )
        transitions = tuple(
            DiscreteStateTransitionMeasurement(
                entity_ref=entity_ref,
                before_observation_index=before.observation_index,
                after_observation_index=after.observation_index,
                before_state_ref=before.state_ref,
                after_state_ref=after.state_ref,
            )
            for before, after in zip(samples, samples[1:])
        )
        first_index_by_state: dict[str, int] = {}
        returns: list[DiscreteStateReturnMeasurement] = []
        for sample in samples:
            earlier = first_index_by_state.setdefault(
                sample.state_ref, sample.observation_index
            )
            if earlier != sample.observation_index:
                returns.append(
                    DiscreteStateReturnMeasurement(
                        entity_ref=entity_ref,
                        earlier_observation_index=earlier,
                        later_observation_index=sample.observation_index,
                        state_ref=sample.state_ref,
                    )
                )
        traces.append(
            DiscreteStateTraceMeasurement(
                entity_ref=entity_ref,
                observed_state_refs=tuple(sample.state_ref for sample in samples),
                transitions=transitions,
                returns=tuple(returns),
            )
        )
    transitions = tuple(
        transition for trace in traces for transition in trace.transitions
    )
    successors: dict[tuple[str, str], set[str]] = {}
    for transition in transitions:
        successors.setdefault(
            (transition.entity_ref, transition.before_state_ref), set()
        ).add(transition.after_state_ref)
    return DerivedGeometryStateMeasurements(
        affine_positions=tuple(affine_positions),
        discrete_state_traces=tuple(traces),
        descriptive_facts=FrozenMap(
            {
                "measurement_context_ref": value.measurement_context_ref,
                "affine_request_count": len(affine_positions),
                "affine_integral_count": sum(
                    item.integral_position for item in affine_positions
                ),
                "affine_target_match_count": sum(
                    item.observed_target_matches is True for item in affine_positions
                ),
                "affine_target_mismatch_count": sum(
                    item.observed_target_matches is False for item in affine_positions
                ),
                "discrete_trace_count": len(traces),
                "observed_transition_count": len(transitions),
                "observed_return_count": sum(len(trace.returns) for trace in traces),
                "transition_mapping_deterministic": bool(transitions)
                and all(len(values) == 1 for values in (successors[__yf_order_key] for __yf_order_key in sorted(successors))),
            }
        ),
    )


def _center_twice(component: ComponentDescription) -> tuple[int, int]:
    return (
        component.bbox.top + component.bbox.bottom,
        component.bbox.left + component.bbox.right,
    )


def _unique_positive_step(values: tuple[int, ...]) -> int | None:
    if len(values) < 2:
        return None
    steps = {after - before for before, after in zip(values, values[1:])}
    return next(iter(steps)) if len(steps) == 1 and next(iter(steps)) > 0 else None


def _contains(outer: BoundingBox, inner: BoundingBox) -> bool:
    return (
        outer.top <= inner.top
        and outer.left <= inner.left
        and outer.bottom >= inner.bottom
        and outer.right >= inner.right
    )
