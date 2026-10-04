"""Pure bounded measurements for caller-declared reconfigurable supports."""

from __future__ import annotations

from collections import deque

from agents.yf_arc3_v5.capabilities.contracts import (
    Position,
    ReconfigurableConnectionDomainInput,
    ReconfigurableConnectionDomainMeasurements,
    ReconfigurableConnectionWitness,
    ReconfigurableRigidFit,
    ReconfigurableRigidGeometryInput,
    ReconfigurableRigidGeometryMeasurements,
    ReconfigurableSupportLayer,
    ReconfigurableSupportPacketInput,
    ReconfigurableSupportPacketMeasurements,
    ReconfigurableSupportSnapshotReport,
    ReconfigurableSupportStep,
)


def _support_union(layers: tuple[ReconfigurableSupportLayer, ...]) -> frozenset[Position]:
    return frozenset(position for layer in layers for position in layer.positions)


def _shift_pose(
    pose: tuple[Position, ...],
    step: ReconfigurableSupportStep,
) -> tuple[Position, ...]:
    return tuple(
        sorted(
            (row + step.delta_row, column + step.delta_column)
            for row, column in pose
        )
    )


def _first_shortest_supported_route(
    *,
    actor_positions: tuple[Position, ...],
    terminal_positions: tuple[Position, ...],
    support_positions: frozenset[Position],
    steps: tuple[ReconfigurableSupportStep, ...],
    maximum_tested_actor_poses: int,
) -> tuple[tuple[str, ...], tuple[tuple[Position, ...], ...], int, bool]:
    start = tuple(sorted(actor_positions))
    terminal = tuple(sorted(terminal_positions))
    pending = deque(((start, (), (start,)),))
    visited = {start}
    tested = 1
    bound_reached = False
    ordered_steps = tuple(sorted(steps, key=lambda item: item.action_ref))
    while pending:
        pose, actions, poses = pending.popleft()
        if pose == terminal:
            return actions, poses, tested, bound_reached
        for step in ordered_steps:
            successor = _shift_pose(pose, step)
            if successor in visited or not set(successor) <= support_positions:
                continue
            if tested >= maximum_tested_actor_poses:
                bound_reached = True
                return (), (), tested, bound_reached
            visited.add(successor)
            tested += 1
            pending.append(
                (successor, (*actions, step.action_ref), (*poses, successor))
            )
    return (), (), tested, bound_reached


def measure_reconfigurable_support_packet(
    value: ReconfigurableSupportPacketInput,
) -> ReconfigurableSupportPacketMeasurements:
    """Compare exact support unions and test one supplied terminal footprint."""

    reports: list[ReconfigurableSupportSnapshotReport] = []
    unions: list[frozenset[Position]] = []
    for snapshot in value.snapshots:
        support_union = _support_union(snapshot.active_layers)
        unions.append(support_union)
        unsupported = tuple(
            sorted(set(snapshot.actor_positions) - support_union)
        )
        owners_by_position = tuple(
            (
                position,
                tuple(
                    sorted(
                        layer.owner_ref
                        for layer in snapshot.active_layers
                        if position in layer.positions
                    )
                ),
            )
            for position in sorted(snapshot.actor_positions)
        )
        reports.append(
            ReconfigurableSupportSnapshotReport(
                snapshot_ref=snapshot.snapshot_ref,
                support_union_positions=tuple(sorted(support_union)),
                unsupported_actor_positions=unsupported,
                supporting_owner_refs_by_actor_position=owners_by_position,
            )
        )

    restoration_positions = tuple(
        sorted(
            position
            for position in set().union(*unions)
            if any(
                position in unions[earlier]
                and position not in unions[middle]
                and position in unions[later]
                for earlier in range(len(unions) - 2)
                for middle in range(earlier + 1, len(unions) - 1)
                for later in range(middle + 1, len(unions))
            )
        )
    )
    final_snapshot = value.snapshots[-1]
    final_union = unions[-1]
    route, _poses, tested, bound_reached = _first_shortest_supported_route(
        actor_positions=final_snapshot.actor_positions,
        terminal_positions=value.requested_terminal_actor_positions,
        support_positions=final_union,
        steps=value.movement_steps,
        maximum_tested_actor_poses=value.maximum_tested_actor_poses,
    )
    final_multi_owner_actor_position_count = sum(
        1
        for _position, owner_refs in reports[-1].supporting_owner_refs_by_actor_position
        if len(owner_refs) > 1
    )
    return ReconfigurableSupportPacketMeasurements(
        snapshot_reports=tuple(reports),
        support_union_changed=any(
            before != after for before, after in zip(unions, unions[1:])
        ),
        restoration_positions=restoration_positions,
        initial_final_support_union_equal=unions[0] == unions[-1],
        final_actor_fully_supported=set(final_snapshot.actor_positions) <= final_union,
        final_connection_present=bool(route)
        or tuple(sorted(final_snapshot.actor_positions))
        == tuple(sorted(value.requested_terminal_actor_positions)),
        final_route_action_refs=route,
        tested_actor_pose_count=tested,
        actor_pose_bound_reached=bound_reached,
        final_active_layer_count=len(final_snapshot.active_layers),
        final_multi_owner_actor_position_count=final_multi_owner_actor_position_count,
    )


def measure_reconfigurable_rigid_geometry(
    value: ReconfigurableRigidGeometryInput,
) -> ReconfigurableRigidGeometryMeasurements:
    """Fit translations and quarter turns for every supplied pivot candidate."""

    before = frozenset(value.before_positions)
    after = frozenset(value.after_positions)
    translations: set[Position] = set()
    before_anchor = min(before)
    for after_anchor in sorted(after):
        delta = (
            after_anchor[0] - before_anchor[0],
            after_anchor[1] - before_anchor[1],
        )
        if frozenset(
            (row + delta[0], column + delta[1]) for row, column in before
        ) == after:
            translations.add(delta)

    before_support = _support_union(value.before_support_layers)
    after_support = _support_union(value.after_support_layers)
    fits: list[ReconfigurableRigidFit] = []
    for candidate in sorted(
        value.pivot_candidates, key=lambda item: item.candidate_ref
    ):
        pivot_row, pivot_column = candidate.position
        for quarter_turns in (1, 2, 3):
            transformed: set[Position] = set()
            for row, column in before:
                relative_row = row - pivot_row
                relative_column = column - pivot_column
                for _ in range(quarter_turns):
                    relative_row, relative_column = (
                        relative_column,
                        -relative_row,
                    )
                transformed.add(
                    (pivot_row + relative_row, pivot_column + relative_column)
                )
            if transformed != after:
                continue
            fits.append(
                ReconfigurableRigidFit(
                    candidate_ref=candidate.candidate_ref,
                    quarter_turns_clockwise=quarter_turns,
                    pivot_outside_transformed_masks=(
                        candidate.position not in before
                        and candidate.position not in after
                    ),
                    staging_supported_before=(
                        set(candidate.staging_actor_positions) <= before_support
                    ),
                    staging_supported_after=(
                        set(candidate.staging_actor_positions) <= after_support
                    ),
                )
            )
    return ReconfigurableRigidGeometryMeasurements(
        exact_translation_deltas=tuple(sorted(translations)),
        exact_quarter_turn_fits=tuple(fits),
        tested_pivot_turn_count=len(value.pivot_candidates) * 3,
    )


def enumerate_reconfigurable_connection_domain(
    value: ReconfigurableConnectionDomainInput,
) -> ReconfigurableConnectionDomainMeasurements:
    """Return at most three first-shortest witnesses, without choosing one."""

    witnesses: list[ReconfigurableConnectionWitness] = []
    tested_pose_count = 0
    pose_bound_reached = False
    witness_bound_reached = False
    ordered_configurations = tuple(
        sorted(value.configurations, key=lambda item: item.configuration_ref)
    )
    tested_configuration_count = 0
    for index, configuration in enumerate(ordered_configurations):
        if len(witnesses) >= value.maximum_witnesses:
            witness_bound_reached = index < len(ordered_configurations)
            break
        remaining_bound = value.maximum_tested_actor_poses - tested_pose_count
        if remaining_bound <= 0:
            pose_bound_reached = True
            break
        tested_configuration_count += 1
        actions, poses, tested, reached = _first_shortest_supported_route(
            actor_positions=value.actor_positions,
            terminal_positions=value.requested_terminal_actor_positions,
            support_positions=_support_union(configuration.active_layers),
            steps=value.movement_steps,
            maximum_tested_actor_poses=remaining_bound,
        )
        tested_pose_count += tested
        pose_bound_reached = pose_bound_reached or reached
        if poses:
            witnesses.append(
                ReconfigurableConnectionWitness(
                    configuration_ref=configuration.configuration_ref,
                    action_refs=actions,
                    actor_pose_positions=poses,
                )
            )
        if pose_bound_reached:
            break
    return ReconfigurableConnectionDomainMeasurements(
        witnesses=tuple(witnesses),
        tested_configuration_count=tested_configuration_count,
        tested_actor_pose_count=max(1, tested_pose_count),
        actor_pose_bound_reached=pose_bound_reached,
        witness_bound_reached=witness_bound_reached,
    )
