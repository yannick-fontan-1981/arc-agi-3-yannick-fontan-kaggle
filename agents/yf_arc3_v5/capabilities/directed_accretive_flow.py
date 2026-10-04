"""Pure bounded measurements for joint directed-flow ticks and recovery packets."""

from __future__ import annotations

from agents.yf_arc3_v5.capabilities.contracts import (
    DirectedFlowColliderUnionInput,
    DirectedFlowColliderUnionMeasurements,
    DirectedFlowJointTickInput,
    DirectedFlowJointTickMeasurements,
    DirectedFlowJointTickSuccessor,
    DirectedFlowProjectedFront,
    DirectedFlowQuantityDelta,
    DirectedFlowRecoveryInput,
    DirectedFlowRecoveryMeasurements,
    DirectedFlowTraceCell,
)


Position = tuple[int, int]


def measure_directed_flow_collider_union(
    value: DirectedFlowColliderUnionInput,
) -> DirectedFlowColliderUnionMeasurements:
    """Union supplied masks while preserving every owner/control/attachment."""

    owners: dict[Position, set[str]] = {}
    for mask in value.collider_masks:
        for position in mask.positions:
            owners.setdefault(position, set()).add(mask.collider_ref)
    owner_rows = tuple(
        (position, tuple(sorted(owner_refs)))
        for position, owner_refs in sorted(owners.items())
    )
    return DirectedFlowColliderUnionMeasurements(
        union_positions=tuple(sorted(owners)),
        owner_refs_by_position=owner_rows,
        overlap_positions=tuple(
            position for position, owner_refs in owner_rows if len(owner_refs) > 1
        ),
        control_refs=tuple(
            sorted(
                {
                    control_ref
                    for mask in value.collider_masks
                    for control_ref in mask.control_refs
                }
            )
        ),
        attachment_refs=tuple(
            sorted(
                {
                    attachment_ref
                    for mask in value.collider_masks
                    for attachment_ref in mask.attachment_refs
                }
            )
        ),
    )


def measure_directed_flow_joint_tick(
    value: DirectedFlowJointTickInput,
) -> DirectedFlowJointTickMeasurements:
    """Project every supplied joint schedule without selecting one."""

    front_by_ref = {front.front_ref: front for front in value.fronts}
    initial_trace = {
        cell.position: set(cell.lineage_refs) for cell in value.existing_trace
    }
    destinations: dict[Position, list[str]] = {}
    for front in value.fronts:
        destination = (
            front.position[0] + front.delta_row,
            front.position[1] + front.delta_column,
        )
        destinations.setdefault(destination, []).append(front.front_ref)
    simultaneous = tuple(
        position
        for position, front_refs in sorted(destinations.items())
        if len(front_refs) > 1
    )

    successors: list[DirectedFlowJointTickSuccessor] = []
    bound_reached = False
    for schedule in value.schedules:
        trace = {
            position: set(refs) for position, refs in sorted(initial_trace.items())
        }
        projected: list[DirectedFlowProjectedFront] = []
        preexisting_contacts: set[Position] = set()
        same_tick_contacts: set[Position] = set()
        for front_ref in schedule.ordered_front_refs:
            front = front_by_ref[front_ref]
            destination = (
                front.position[0] + front.delta_row,
                front.position[1] + front.delta_column,
            )
            contacted_preexisting = destination in initial_trace
            contacted_same_tick = destination in trace and not contacted_preexisting
            if contacted_preexisting:
                preexisting_contacts.add(destination)
            if contacted_same_tick:
                same_tick_contacts.add(destination)
            trace.setdefault(destination, set()).add(front.source_ref)
            projected.append(
                DirectedFlowProjectedFront(
                    front_ref=front.front_ref,
                    source_ref=front.source_ref,
                    before_position=front.position,
                    after_position=destination,
                    contacted_preexisting_trace=contacted_preexisting,
                    contacted_same_tick_trace=contacted_same_tick,
                )
            )
        if len(trace) > value.maximum_resulting_trace_positions:
            bound_reached = True
            continue
        successors.append(
            DirectedFlowJointTickSuccessor(
                schedule_ref=schedule.schedule_ref,
                projected_fronts=tuple(projected),
                resulting_trace=tuple(
                    DirectedFlowTraceCell(
                        position=position,
                        lineage_refs=tuple(sorted(lineage_refs)),
                    )
                    for position, lineage_refs in sorted(trace.items())
                ),
                simultaneous_arrival_positions=simultaneous,
                preexisting_trace_contact_positions=tuple(
                    sorted(preexisting_contacts)
                ),
                same_tick_trace_contact_positions=tuple(sorted(same_tick_contacts)),
            )
        )

    return DirectedFlowJointTickMeasurements(
        successors=tuple(successors),
        source_ref_count=len({front.source_ref for front in value.fronts}),
        front_count=len(value.fronts),
        schedule_count=len(value.schedules),
        alternative_schedule_count=max(0, len(successors) - 1),
        resulting_trace_bound_reached=bound_reached,
    )


def measure_directed_flow_recovery(
    value: DirectedFlowRecoveryInput,
) -> DirectedFlowRecoveryMeasurements:
    """Compare configuration, transient trace, receptacle and quantities exactly."""

    configuration_before = set(value.configuration_before_release)
    configuration_after = set(value.configuration_after_recovery)
    trace_failed = set(value.trace_after_failed_release)
    trace_after = set(value.trace_after_recovery)
    receptacle_before = set(value.receptacle_before_release)
    receptacle_failed = set(value.receptacle_after_failed_release)
    receptacle_after = set(value.receptacle_after_recovery)
    after_quantities = {
        item.quantity_ref: item.value for item in value.quantities_after_recovery
    }
    return DirectedFlowRecoveryMeasurements(
        retained_configuration_positions=tuple(
            sorted(configuration_before & configuration_after)
        ),
        removed_configuration_positions=tuple(
            sorted(configuration_before - configuration_after)
        ),
        added_configuration_positions=tuple(
            sorted(configuration_after - configuration_before)
        ),
        configuration_exactly_preserved=configuration_before == configuration_after,
        cleared_trace_positions=tuple(sorted(trace_failed - trace_after)),
        retained_trace_positions=tuple(sorted(trace_failed & trace_after)),
        added_trace_positions=tuple(sorted(trace_after - trace_failed)),
        receptacle_exactly_restored=(
            receptacle_after == receptacle_before
            and receptacle_failed != receptacle_before
        ),
        restored_receptacle_positions=tuple(
            sorted((receptacle_before - receptacle_failed) & receptacle_after)
        ),
        quantity_deltas=tuple(
            DirectedFlowQuantityDelta(
                quantity_ref=item.quantity_ref,
                before_value=item.value,
                after_value=after_quantities[item.quantity_ref],
                signed_delta=after_quantities[item.quantity_ref] - item.value,
            )
            for item in value.quantities_before_release
        ),
    )
