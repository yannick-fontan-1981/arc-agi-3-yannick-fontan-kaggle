"""Pure bounded measurements for source-declared regional writing."""

from __future__ import annotations

from collections import deque

from agents.yf_arc3_v5.capabilities.contracts import (
    OrderedReliefPoseInput,
    OrderedReliefPoseMeasurements,
    OrderedReliefTransitionInput,
    OrderedReliefTransitionMeasurements,
    OrderedReliefWriteInput,
    OrderedReliefWriteMeasurements,
)


def measure_ordered_relief_transition(
    value: OrderedReliefTransitionInput,
) -> OrderedReliefTransitionMeasurements:
    """Measure a sealed packet without assigning meaning to its partitions."""

    before_shape = (len(value.before_rows), len(value.before_rows[0]))
    after_shape = (len(value.after_rows), len(value.after_rows[0]))
    compared_height = min(before_shape[0], after_shape[0])
    compared_width = min(before_shape[1], after_shape[1])
    changed = tuple(
        (row, column)
        for row in range(compared_height)
        for column in range(compared_width)
        if value.before_rows[row][column] != value.after_rows[row][column]
    )
    changed_set = set(changed)
    partition_positions = {
        partition.partition_ref: set(partition.positions)
        for partition in value.partitions
    }
    assigned = set().union(*partition_positions.values())
    application_positions = tuple(
        position
        for position in value.application_positions
        if 0 <= position[0] < compared_height and 0 <= position[1] < compared_width
    )
    changed_application_positions = tuple(
        position for position in application_positions if position in changed_set
    )
    same_value_positions = ()
    if value.supplied_write_value is not None:
        same_value_positions = tuple(
            position
            for position in application_positions
            if position not in changed_set
            and value.before_rows[position[0]][position[1]]
            == value.supplied_write_value
            and value.after_rows[position[0]][position[1]]
            == value.supplied_write_value
        )
    return OrderedReliefTransitionMeasurements(
        before_shape=before_shape,
        after_shape=after_shape,
        extent_changed=before_shape != after_shape,
        changed_positions=changed,
        changed_count_by_partition=tuple(
            (partition.partition_ref, len(changed_set & partition_positions[partition.partition_ref]))
            for partition in value.partitions
        ),
        unpartitioned_changed_count=len(changed_set - assigned),
        application_position_count=len(application_positions),
        changed_application_positions=changed_application_positions,
        known_same_value_application_positions=same_value_positions,
        compared_cell_count=compared_height * compared_width,
    )


def apply_ordered_relief_write(
    value: OrderedReliefWriteInput,
) -> OrderedReliefWriteMeasurements:
    """Apply one supplied last-writer replacement and preserve every other cell."""

    rows = [list(row) for row in value.canvas_rows]
    changed: list[tuple[int, int]] = []
    same_value: list[tuple[int, int]] = []
    for row, column in value.application_positions:
        if rows[row][column] == value.write_value:
            same_value.append((row, column))
        else:
            rows[row][column] = value.write_value
            changed.append((row, column))
    resulting_rows = tuple(tuple(row) for row in rows)
    application_set = set(value.application_positions)
    preserved = all(
        resulting_rows[row][column] == value.canvas_rows[row][column]
        for row in range(len(value.canvas_rows))
        for column in range(len(value.canvas_rows[0]))
        if (row, column) not in application_set
    )
    return OrderedReliefWriteMeasurements(
        resulting_rows=resulting_rows,
        affected_positions=value.application_positions,
        changed_positions=tuple(changed),
        same_value_positions=tuple(same_value),
        non_application_cells_preserved=preserved,
    )


def measure_ordered_relief_pose(
    value: OrderedReliefPoseInput,
) -> OrderedReliefPoseMeasurements:
    """Find one deterministic shortest route using observed edges only."""

    records = {record.pose_ref: record for record in value.pose_records}
    requested = records.get(value.requested_pose_ref)
    requested_pose_observed = requested is not None
    aspect_matches = bool(
        requested is not None and requested.aspect_ref == value.required_aspect_ref
    )
    size_matches = bool(
        requested is not None
        and requested.applicator_size == value.required_applicator_size
    )
    eligible_target = requested_pose_observed and aspect_matches and size_matches
    adjacency: dict[str, list[tuple[str, str]]] = {}
    for edge in value.observed_edges:
        adjacency.setdefault(edge.source_pose_ref, []).append(
            (edge.action_ref, edge.target_pose_ref)
        )
    for pose_ref in sorted(adjacency):
        adjacency[pose_ref].sort()

    queue = deque([(value.current_pose_ref, (), (value.current_pose_ref,))])
    visited = {value.current_pose_ref}
    route_actions: tuple[str, ...] = ()
    route_poses: tuple[str, ...] = ()
    tested_edge_count = 0
    depth_bound_reached = False
    while queue and eligible_target:
        pose_ref, actions, poses = queue.popleft()
        if pose_ref == value.requested_pose_ref:
            route_actions = actions
            route_poses = poses
            break
        if len(actions) >= value.maximum_route_depth:
            if adjacency.get(pose_ref):
                depth_bound_reached = True
            continue
        for action_ref, target_ref in adjacency.get(pose_ref, ()):
            tested_edge_count += 1
            if target_ref in visited:
                continue
            visited.add(target_ref)
            queue.append((target_ref, (*actions, action_ref), (*poses, target_ref)))
    route_available = bool(route_poses) or (
        eligible_target and value.current_pose_ref == value.requested_pose_ref
    )
    if route_available and not route_poses:
        route_poses = (value.current_pose_ref,)
    return OrderedReliefPoseMeasurements(
        requested_pose_observed=requested_pose_observed,
        aspect_matches=aspect_matches,
        applicator_size_matches=size_matches,
        route_available=route_available,
        route_action_refs=route_actions,
        route_pose_refs=route_poses,
        tested_edge_count=tested_edge_count,
        route_depth_bound_reached=depth_bound_reached,
    )
