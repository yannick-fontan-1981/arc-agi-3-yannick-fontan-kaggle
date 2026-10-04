"""Exact finite-grid reachability without goal or route selection policy."""

from __future__ import annotations

from collections import deque

from agents.yf_arc3_v5.capabilities.contracts import (
    ConfigurationGraphEdge,
    GoalReachability,
    MobileUnderlayTransitionMeasurement,
    Position,
    PositionDistance,
    ReachabilityInput,
    ReachabilityResult,
    TerrainTransitionMeasurement,
    ViewportSupportTerrainInput,
    ViewportSupportTerrainMeasurements,
)
from agents.yf_arc3_v5.logos.types import FrozenMap


def measure_viewport_support_terrain(
    value: ViewportSupportTerrainInput,
) -> ViewportSupportTerrainMeasurements:
    """Build exact observed configuration deltas without assigning their meaning."""

    shift_pairs = tuple(
        (item.delta_row, item.delta_col) for item in value.viewport_shifts
    )
    distinct_shifts = tuple(sorted(set(shift_pairs)))
    if value.viewport_shifts:
        persistent_anchors = set(value.viewport_shifts[0].anchor_refs)
        for observation in value.viewport_shifts[1:]:
            persistent_anchors.intersection_update(observation.anchor_refs)
    else:
        persistent_anchors = set()

    configurations = {
        item.configuration_ref: item for item in value.configurations
    }
    mobile_layers = {
        item.configuration_ref: item for item in value.mobile_layers
    }
    directional_action_refs = set(value.declared_directional_action_refs)
    edges: list[ConfigurationGraphEdge] = []
    measurements: list[TerrainTransitionMeasurement] = []
    mobile_measurements: list[MobileUnderlayTransitionMeasurement] = []
    dynamic_traversability_count = 0
    support_change_count = 0
    for transition in value.transitions:
        before = configurations[transition.before_configuration_ref]
        after = configurations[transition.after_configuration_ref]
        before_traversable = set(before.traversable_positions)
        after_traversable = set(after.traversable_positions)
        before_support = set(before.support_contact_positions)
        after_support = set(after.support_contact_positions)
        added_traversable = tuple(sorted(after_traversable - before_traversable))
        removed_traversable = tuple(sorted(before_traversable - after_traversable))
        gained_support = tuple(sorted(after_support - before_support))
        lost_support = tuple(sorted(before_support - after_support))
        if added_traversable or removed_traversable:
            dynamic_traversability_count += 1
        if gained_support or lost_support:
            support_change_count += 1
        edges.append(
            ConfigurationGraphEdge(
                transition_ref=transition.transition_ref,
                before_configuration_ref=transition.before_configuration_ref,
                after_configuration_ref=transition.after_configuration_ref,
                action_ref=transition.action_ref,
            )
        )
        measurements.append(
            TerrainTransitionMeasurement(
                transition_ref=transition.transition_ref,
                added_traversable_positions=added_traversable,
                removed_traversable_positions=removed_traversable,
                gained_support_contact_positions=gained_support,
                lost_support_contact_positions=lost_support,
            )
        )
        before_mobile = mobile_layers.get(transition.before_configuration_ref)
        after_mobile = mobile_layers.get(transition.after_configuration_ref)
        if before_mobile is not None and after_mobile is not None:
            before_positions = set(before_mobile.occupied_positions)
            after_positions = set(after_mobile.occupied_positions)
            before_complete_underlay = (
                before_mobile.present
                and before_positions.issubset(before_traversable)
            )
            after_complete_underlay = (
                after_mobile.present
                and after_positions.issubset(after_traversable)
            )
            same_positions = (
                before_mobile.present
                and after_mobile.present
                and before_positions == after_positions
            )
            mobile_measurements.append(
                MobileUnderlayTransitionMeasurement(
                    transition_ref=transition.transition_ref,
                    action_ref=transition.action_ref,
                    action_is_declared_directional=(
                        transition.action_ref in directional_action_refs
                    ),
                    before_mobile_present=before_mobile.present,
                    after_mobile_present=after_mobile.present,
                    same_mobile_occupied_positions=same_positions,
                    mobile_pose_changed=(
                        before_mobile.present
                        and after_mobile.present
                        and before_positions != after_positions
                    ),
                    before_complete_traversable_underlay=before_complete_underlay,
                    after_complete_traversable_underlay=after_complete_underlay,
                    incomplete_underlay_position_count_after=(
                        len(after_positions - after_traversable)
                        if after_mobile.present
                        else None
                    ),
                )
            )

    support_loss_stationary_count = sum(
        item.before_complete_traversable_underlay is True
        and item.after_complete_traversable_underlay is False
        and item.same_mobile_occupied_positions is True
        for item in mobile_measurements
    )
    unsupported_stationary_directional_count = sum(
        item.action_is_declared_directional
        and item.before_complete_traversable_underlay is False
        and item.after_complete_traversable_underlay is False
        and item.same_mobile_occupied_positions is True
        for item in mobile_measurements
    )
    support_restoration_stationary_count = sum(
        item.before_complete_traversable_underlay is False
        and item.after_complete_traversable_underlay is True
        and item.same_mobile_occupied_positions is True
        for item in mobile_measurements
    )
    supported_effectful_directional_count = sum(
        item.action_is_declared_directional
        and item.before_complete_traversable_underlay is True
        and item.after_complete_traversable_underlay is True
        and item.mobile_pose_changed is True
        for item in mobile_measurements
    )
    disappearance_after_underlay_loss_count = sum(
        item.before_complete_traversable_underlay is True
        and item.after_mobile_present is False
        for item in mobile_measurements
    )
    ordered_layered_support_trace_count = int(
        _has_ordered_layered_support_trace(mobile_measurements)
    )

    return ViewportSupportTerrainMeasurements(
        configuration_edges=tuple(edges),
        terrain_transitions=tuple(measurements),
        mobile_underlay_transitions=tuple(mobile_measurements),
        persistent_anchor_refs=tuple(sorted(persistent_anchors)),
        descriptive_facts=FrozenMap(
            {
                "measurement_context_ref": value.measurement_context_ref,
                "viewport_shift_observation_count": len(value.viewport_shifts),
                "distinct_viewport_shift_count": len(distinct_shifts),
                "coherent_viewport_shift": bool(shift_pairs)
                and len(distinct_shifts) == 1,
                "nonzero_coherent_viewport_shift": len(distinct_shifts) == 1
                and distinct_shifts[0] != (0, 0),
                "persistent_anchor_count": len(persistent_anchors),
                "configuration_count": len(value.configurations),
                "configuration_transition_count": len(value.transitions),
                "dynamic_traversability_transition_count": (
                    dynamic_traversability_count
                ),
                "support_change_transition_count": support_change_count,
                "mobile_underlay_transition_count": len(mobile_measurements),
                "support_loss_under_stationary_mobile_count": (
                    support_loss_stationary_count
                ),
                "unsupported_stationary_directional_no_effect_count": (
                    unsupported_stationary_directional_count
                ),
                "support_restoration_under_stationary_mobile_count": (
                    support_restoration_stationary_count
                ),
                "supported_effectful_directional_transition_count": (
                    supported_effectful_directional_count
                ),
                "mobile_disappearance_after_underlay_loss_count": (
                    disappearance_after_underlay_loss_count
                ),
                "ordered_layered_support_trace_count": (
                    ordered_layered_support_trace_count
                ),
            }
        ),
    )


def _has_ordered_layered_support_trace(
    measurements: list[MobileUnderlayTransitionMeasurement],
) -> bool:
    """Measure one exact loss-lock-restoration-reactivation order."""

    stage = 0
    for item in measurements:
        if stage == 0 and (
            item.before_complete_traversable_underlay is True
            and item.after_complete_traversable_underlay is False
            and item.same_mobile_occupied_positions is True
        ):
            stage = 1
        elif stage == 1 and (
            item.action_is_declared_directional
            and item.before_complete_traversable_underlay is False
            and item.after_complete_traversable_underlay is False
            and item.same_mobile_occupied_positions is True
        ):
            stage = 2
        elif stage == 2 and (
            item.before_complete_traversable_underlay is False
            and item.after_complete_traversable_underlay is True
            and item.same_mobile_occupied_positions is True
        ):
            stage = 3
        elif stage == 3 and (
            item.action_is_declared_directional
            and item.before_complete_traversable_underlay is True
            and item.after_complete_traversable_underlay is True
            and item.mobile_pose_changed is True
        ):
            return True
    return False


def finite_grid_reachability(value: ReachabilityInput) -> ReachabilityResult:
    traversable = set(value.traversable)
    distances: dict[Position, int] = {value.start: 0}
    predecessors: dict[Position, set[Position]] = {value.start: set()}
    # First discovery is deterministic because both the FIFO frontier and the
    # declared step order are stable.  Keep that single witness separately
    # while retaining all equal shortest predecessors for neutral alternatives.
    first_predecessor: dict[Position, tuple[Position, str]] = {}
    pending: deque[Position] = deque((value.start,))
    while pending:
        current = pending.popleft()
        current_distance = distances[current]
        for step in value.steps:
            neighbor = (
                current[0] + step.delta_row,
                current[1] + step.delta_col,
            )
            if neighbor not in traversable:
                continue
            proposed_distance = current_distance + 1
            known_distance = distances.get(neighbor)
            if known_distance is None:
                distances[neighbor] = proposed_distance
                predecessors[neighbor] = {current}
                first_predecessor[neighbor] = (current, step.label)
                pending.append(neighbor)
            elif known_distance == proposed_distance:
                predecessors[neighbor].add(current)

    first_step_labels_by_position: dict[Position, tuple[str, ...]] = {}
    for position in distances:
        first_neighbors = _first_neighbors(position, value.start, predecessors)
        labels = tuple(
            step.label
            for step in value.steps
            if (
                value.start[0] + step.delta_row,
                value.start[1] + step.delta_col,
            )
            in first_neighbors
        )
        first_step_labels_by_position[position] = labels

    return ReachabilityResult(
        distances=tuple(
            PositionDistance(position=position, distance=distance)
            for position, distance in sorted(distances.items())
        ),
        goals=tuple(
            GoalReachability(
                goal=goal,
                reachable=goal in distances,
                distance=distances.get(goal),
                shortest_first_step_labels=first_step_labels_by_position.get(goal, ()),
                first_shortest_step_labels=_first_shortest_witness(
                    goal, value.start, first_predecessor
                ),
            )
            for goal in value.goals
        ),
    )


def _first_shortest_witness(
    target: Position,
    start: Position,
    first_predecessor: dict[Position, tuple[Position, str]],
) -> tuple[str, ...]:
    if target == start:
        return ()
    reverse_labels: list[str] = []
    current = target
    while current != start:
        predecessor = first_predecessor.get(current)
        if predecessor is None:
            return ()
        current, label = predecessor
        reverse_labels.append(label)
    return tuple(reversed(reverse_labels))


def _first_neighbors(
    target: Position,
    start: Position,
    predecessors: dict[Position, set[Position]],
) -> set[Position]:
    if target == start:
        return set()
    result: set[Position] = set()
    pending = [target]
    visited: set[Position] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        for predecessor in predecessors.get(current, set()):
            if predecessor == start:
                result.add(current)
            else:
                pending.append(predecessor)
    return result
