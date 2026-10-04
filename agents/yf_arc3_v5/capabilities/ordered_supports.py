"""Bounded measurements and finite search for repeated ordered supports.

The module assigns no role to a support or layer.  It measures repeated vertical
supports and strictly nested horizontal extents, then materializes the first
shortest legal witness for each traversal order declared by DRM.  Every result
remains counterfactual until DRM/SRC establishes its meaning and commits it.
"""

from __future__ import annotations

from collections import Counter, deque

from agents.yf_arc3_v5.capabilities.contracts import (
    ComponentExtractionResult,
    FrameGrid,
    InteractionProbeCandidate,
    OrderedSupportConfiguration,
    OrderedSupportDescription,
    OrderedSupportGeometryMeasurements,
    OrderedSupportTransferRequest,
    OrderedSupportTransferResult,
    OrderedSupportTransferStep,
    OrderedSupportTransferWitness,
)
from agents.yf_arc3_v5.logos.types import FrozenMap


def materialize_ordered_support_transfer_witnesses(
    value: OrderedSupportTransferRequest,
) -> OrderedSupportTransferResult:
    """Keep at most one shortest witness per declared order, globally bounded."""

    initial = value.configuration.support_extents
    all_extents = tuple(extent for support in initial for extent in support)
    endpoint = value.endpoint_support_index
    terminal_stack = tuple(sorted(all_extents, reverse=True))
    terminal = tuple(
        terminal_stack if index == endpoint else ()
        for index in range(len(initial))
    )
    if initial == terminal:
        return OrderedSupportTransferResult(
            witnesses=(),
            total_expanded_state_count=0,
            failure_kinds=("already_at_declared_endpoint",),
        )

    witnesses: list[OrderedSupportTransferWitness] = []
    retained_paths: set[tuple[tuple[int, int, int], ...]] = set()
    total_expanded = 0
    expansion_bound_reached = False
    depth_bound_reached = False

    for ordinal, traversal_order in enumerate(value.traversal_orders):
        queue = deque([(initial, ())])
        visited = {initial}
        local_expanded = 0
        first_path: tuple[tuple[int, int, int], ...] | None = None
        while queue:
            if total_expanded >= value.maximum_state_expansions:
                expansion_bound_reached = True
                break
            state, path = queue.popleft()
            total_expanded += 1
            local_expanded += 1
            if state == terminal:
                first_path = path
                break
            if len(path) >= value.maximum_transfer_depth:
                depth_bound_reached = True
                continue
            for source_index in traversal_order:
                source = state[source_index]
                if not source:
                    continue
                extent = source[-1]
                for destination_index in traversal_order:
                    if source_index == destination_index:
                        continue
                    destination = state[destination_index]
                    if destination and destination[-1] <= extent:
                        continue
                    next_supports = [tuple(support) for support in state]
                    next_supports[source_index] = source[:-1]
                    next_supports[destination_index] = (*destination, extent)
                    next_state = tuple(next_supports)
                    if next_state in visited:
                        continue
                    visited.add(next_state)
                    queue.append(
                        (
                            next_state,
                            (*path, (extent, source_index, destination_index)),
                        )
                    )
            if len(witnesses) >= 3:
                break
        if expansion_bound_reached:
            break
        if first_path is None or first_path in retained_paths:
            continue
        retained_paths.add(first_path)
        witnesses.append(
            OrderedSupportTransferWitness(
                traversal_order_ordinal=ordinal,
                steps=tuple(
                    OrderedSupportTransferStep(
                        extent=extent,
                        source_support_index=source_index,
                        destination_support_index=destination_index,
                    )
                    for extent, source_index, destination_index in first_path
                ),
                expanded_state_count=local_expanded,
            )
        )
        if len(witnesses) >= 3:
            break

    failure_kinds: list[str] = []
    if expansion_bound_reached:
        failure_kinds.append("state_expansion_bound_reached")
    if depth_bound_reached and not witnesses:
        failure_kinds.append("transfer_depth_bound_reached")
    if not witnesses and not failure_kinds:
        failure_kinds.append("no_legal_route_within_declared_bounds")
    return OrderedSupportTransferResult(
        witnesses=tuple(witnesses),
        total_expanded_state_count=total_expanded,
        state_expansion_bound_reached=expansion_bound_reached,
        transfer_depth_bound_reached=depth_bound_reached,
        failure_kinds=tuple(failure_kinds[:3]),
    )


def declared_support_orders(
    support_count: int,
    endpoint_support_index: int,
    variant_refs: tuple[str, ...],
) -> tuple[tuple[int, ...], ...]:
    """Mechanically instantiate at most three DRM-declared support orders."""

    canonical = tuple(range(support_count))
    variants: list[tuple[int, ...]] = []
    for variant_ref in variant_refs[:3]:
        if variant_ref == "canonical":
            order = canonical
        elif variant_ref == "reverse_canonical":
            order = tuple(reversed(canonical))
        elif variant_ref == "endpoint_outward":
            order = tuple(
                sorted(
                    canonical,
                    key=lambda index: (
                        abs(index - endpoint_support_index),
                        index,
                    ),
                )
            )
        else:
            continue
        if order not in variants:
            variants.append(order)
    return tuple(variants)


def measure_ordered_support_geometry(
    frame: FrameGrid,
    components: ComponentExtractionResult,
) -> OrderedSupportGeometryMeasurements:
    """Measure a conservative repeated-support arrangement from exact raster facts."""

    background = Counter(value for row in frame.rows for value in row).most_common(1)[0][0]
    lower_half = frame.height // 2
    vertical = tuple(
        component
        for component in components.components
        if not component.touches_frame_boundary
        and component.bbox.top >= lower_half
        and component.bbox.height >= 6
        and component.bbox.width <= 3
        and component.bbox.height >= 3 * component.bbox.width
    )
    if not vertical:
        return OrderedSupportGeometryMeasurements(
            supports=(), repeated_support_count=0
        )
    greatest_bottom = max(component.bbox.bottom for component in vertical)
    floor_aligned = tuple(
        component
        for component in vertical
        if component.bbox.bottom == greatest_bottom
    )
    greatest_width = max(component.bbox.width for component in floor_aligned)
    support_components = tuple(
        sorted(
            (
                component
                for component in floor_aligned
                if component.bbox.width == greatest_width
            ),
            key=lambda component: (
                component.bbox.left + component.bbox.right,
                component.component_id,
            ),
        )
    )
    if not 3 <= len(support_components) <= 8:
        return OrderedSupportGeometryMeasurements(
            supports=(), repeated_support_count=len(support_components)
        )
    shape_keys = {
        (component.bbox.height, component.bbox.width, component.relative_pixels)
        for component in support_components
    }
    if len(shape_keys) != 1:
        return OrderedSupportGeometryMeasurements(
            supports=(), repeated_support_count=len(support_components)
        )

    centers = tuple(
        (component.bbox.left + component.bbox.right) // 2
        for component in support_components
    )
    descriptions: list[OrderedSupportDescription] = []
    for index, (component, center) in enumerate(zip(support_components, centers)):
        left_boundary = (
            0 if index == 0 else (centers[index - 1] + center) // 2 + 1
        )
        right_boundary = (
            frame.width - 1
            if index == len(centers) - 1
            else (center + centers[index + 1]) // 2
        )
        row_runs: list[tuple[int, int, tuple[int, ...]]] = []
        structural_extent = component.bbox.width
        for row_index in range(component.bbox.top, component.bbox.bottom + 1):
            row = frame.rows[row_index]
            occupied = tuple(
                column
                for column in range(left_boundary, right_boundary + 1)
                if row[column] != background
            )
            if occupied:
                structural_extent = max(
                    structural_extent, occupied[-1] - occupied[0] + 1
                )
            left = center
            while left > left_boundary and row[left - 1] != background:
                left -= 1
            right = center
            while right < right_boundary and row[right + 1] != background:
                right += 1
            width = right - left + 1
            row_runs.append((row_index, width, tuple(row[left : right + 1])))
        bands: list[tuple[int, int]] = []
        for current, following in zip(row_runs, row_runs[1:]):
            if (
                current[1] > component.bbox.width
                and current[1] == following[1]
                and current[2] == following[2]
                and following[0] == current[0] + 1
            ):
                bands.append((current[0], current[1]))
        # Consecutive equal rows describe one band, regardless of its height.
        unique_bands: dict[int, int] = {}
        for row_index, extent in bands:
            unique_bands.setdefault(extent, row_index)
        layer_extents = tuple(
            extent
            for extent, _row in sorted(
                unique_bands.items(), key=lambda item: (-item[1], -item[0])
            )
        )
        point = min(
            component.pixels,
            key=lambda item: (
                abs(2 * item[0] - (component.bbox.top + component.bbox.bottom))
                + abs(2 * item[1] - (component.bbox.left + component.bbox.right)),
                item,
            ),
        )
        descriptions.append(
            OrderedSupportDescription(
                support_ref=component.component_id,
                center_column=center,
                point=point,
                structural_extent=structural_extent,
                layer_extents=layer_extents,
            )
        )

    flattened = tuple(
        extent for support in descriptions for extent in support.layer_extents
    )
    strict = bool(flattened) and all(
        lower > upper
        for support in descriptions
        for lower, upper in zip(support.layer_extents, support.layer_extents[1:])
    )
    widest_extent = max(support.structural_extent for support in descriptions)
    widest_indices = tuple(
        index
        for index, support in enumerate(descriptions)
        if support.structural_extent == widest_extent
    )
    return OrderedSupportGeometryMeasurements(
        supports=tuple(descriptions),
        repeated_support_count=len(descriptions),
        strict_layer_order_observed=strict,
        globally_unique_layer_extents_observed=(
            bool(flattened) and len(flattened) == len(set(flattened))
        ),
        unique_widest_support_index=(
            widest_indices[0] if len(widest_indices) == 1 else None
        ),
    )


def materialize_ordered_support_click_counterfactual(
    *,
    frame: FrameGrid,
    components: ComponentExtractionResult,
    point_candidates: tuple[InteractionProbeCandidate, ...],
    point_action_ref: str,
    configuration: FrozenMap,
) -> tuple[FrozenMap, FrozenMap]:
    """Return compact route measures plus one DRM-declared-order click payload.

    The complete local search result is deliberately not transported.  At most
    one payload, identified by the ordinal already declared in ``configuration``,
    is exposed as a counterfactual for DRM/SRC commitment.
    """

    geometry = measure_ordered_support_geometry(frame, components)
    context: dict[str, object] = {
        "ordered_support_repeated_support_count": geometry.repeated_support_count,
        "ordered_support_strict_layer_order_observed": (
            geometry.strict_layer_order_observed
        ),
        "ordered_support_globally_unique_layer_extents_observed": (
            geometry.globally_unique_layer_extents_observed
        ),
        "ordered_support_unique_widest_structural_endpoint_present": (
            geometry.unique_widest_support_index is not None
        ),
        "ordered_support_measured_layer_count": sum(
            len(support.layer_extents) for support in geometry.supports
        ),
        "ordered_support_route_witness_count": 0,
        "ordered_support_route_global_witness_bound": 3,
        "ordered_support_route_state_expansion_bound_reached": False,
        "ordered_support_route_transfer_depth_bound_reached": False,
        "ordered_support_route_failure_kinds": (),
        "ordered_support_declared_route_ordinal_available": False,
    }
    endpoint = geometry.unique_widest_support_index
    variant_refs = tuple(
        str(item)
        for item in configuration.get("ordered_support_traversal_orders", ())
    )[:3]
    declared_ordinal = configuration.get(
        "ordered_support_committed_traversal_ordinal"
    )
    state_bound = int(
        configuration.get("ordered_support_max_expanded_states") or 0
    )
    depth_bound = int(
        configuration.get("ordered_support_max_transfer_depth") or 0
    )
    if (
        endpoint is None
        or not geometry.strict_layer_order_observed
        or not geometry.globally_unique_layer_extents_observed
        or not variant_refs
        or declared_ordinal is None
        or state_bound <= 0
        or depth_bound <= 0
    ):
        return FrozenMap(context), FrozenMap()
    orders = declared_support_orders(
        len(geometry.supports), endpoint, variant_refs
    )
    if not orders:
        return FrozenMap(context), FrozenMap()
    result = materialize_ordered_support_transfer_witnesses(
        OrderedSupportTransferRequest(
            configuration=OrderedSupportConfiguration(
                support_extents=tuple(
                    support.layer_extents for support in geometry.supports
                )
            ),
            endpoint_support_index=endpoint,
            traversal_orders=orders,
            maximum_state_expansions=state_bound,
            maximum_transfer_depth=depth_bound,
        )
    )
    context.update(
        {
            "ordered_support_route_witness_count": len(result.witnesses),
            "ordered_support_route_global_witness_bound_not_exceeded": (
                len(result.witnesses) <= 3
            ),
            "ordered_support_route_total_expanded_state_count": (
                result.total_expanded_state_count
            ),
            "ordered_support_route_state_expansion_bound": state_bound,
            "ordered_support_route_transfer_depth_bound": depth_bound,
            "ordered_support_route_state_expansion_bound_reached": (
                result.state_expansion_bound_reached
            ),
            "ordered_support_route_transfer_depth_bound_reached": (
                result.transfer_depth_bound_reached
            ),
            "ordered_support_route_failure_kinds": result.failure_kinds,
        }
    )
    retained = next(
        (
            witness
            for witness in result.witnesses
            if witness.traversal_order_ordinal == int(declared_ordinal)
        ),
        None,
    )
    if retained is None:
        return FrozenMap(context), FrozenMap()

    candidate_by_support_ref = {
        str(candidate.component_ref): candidate
        for candidate in point_candidates
        if candidate.action_ref == point_action_ref
        and candidate.component_ref is not None
        and candidate.point is not None
    }
    if any(
        support.support_ref not in candidate_by_support_ref
        for support in geometry.supports
    ):
        return FrozenMap(context), FrozenMap()
    click_support_indices = tuple(
        support_index
        for step in retained.steps
        for support_index in (
            step.source_support_index,
            step.destination_support_index,
        )
    )
    click_candidates = tuple(
        candidate_by_support_ref[
            geometry.supports[support_index].support_ref
        ]
        for support_index in click_support_indices
    )
    if not click_candidates:
        return FrozenMap(context), FrozenMap()
    first_candidate = click_candidates[0]
    context["ordered_support_declared_route_ordinal_available"] = True
    context["ordered_support_declared_route_ordinal"] = int(declared_ordinal)
    context["ordered_support_route_transfer_count"] = len(retained.steps)
    context["ordered_support_route_click_count"] = len(click_candidates)
    context["ordered_support_route_nonretained_payloads_discarded"] = True
    route_facts = FrozenMap(
        {
            "candidate_is_declared_ordered_support_route_first_click": True,
            "ordered_support_click_path_length": len(click_candidates),
            "ordered_support_click_path_action_refs": tuple(
                point_action_ref for _candidate in click_candidates
            ),
            "ordered_support_click_path_action_payloads": tuple(
                candidate.action_data for candidate in click_candidates
            ),
            "ordered_support_click_path_component_refs": tuple(
                candidate.component_ref for candidate in click_candidates
            ),
            "ordered_support_click_path_candidate_refs": tuple(
                candidate.candidate_ref for candidate in click_candidates
            ),
            "ordered_support_route_all_intermediate_states_valid": True,
            "ordered_support_route_conserves_measured_extents": True,
            "ordered_support_route_is_first_shortest_for_declared_order": True,
        }
    )
    return FrozenMap(context), FrozenMap(
        {first_candidate.candidate_ref: route_facts}
    )
