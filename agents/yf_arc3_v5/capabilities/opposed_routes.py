"""Bounded exact simulation of two opposed composite identities on one chain."""

from __future__ import annotations

import hashlib

from agents.yf_arc3_v5.logos.types import canonical_json

from agents.yf_arc3_v5.capabilities.contracts import (
    CompositeIdentityRouteMeasurement,
    MultiIdentityOpposedRoutesAnalysis,
    MultiIdentityOpposedRoutesInput,
    MultiIdentityRouteRejection,
    MultiIdentityRouteStep,
    MultiIdentityRouteWitness,
    QuantizedInterfaceMeasurement,
)
from agents.yf_arc3_v5.capabilities.planning import (
    _first_shortest_route_witness,
)


def simulate_multi_identity_opposed_routes(
    value: MultiIdentityOpposedRoutesInput,
) -> MultiIdentityOpposedRoutesAnalysis:
    """Return the first exact witness for each of at most three interfaces.

    No route is selected here. Each retained result is one deterministic
    shortest counterfactual for a distinct measured crossing interface.
    """

    interfaces = tuple(sorted(value.interfaces, key=lambda item: item.lower_index))
    identities = tuple(sorted(value.identities, key=lambda item: item.current_zone_index))
    witnesses: list[MultiIdentityRouteWitness] = []
    rejections: list[MultiIdentityRouteRejection] = []
    for interface in interfaces[: min(value.max_witnesses, 3)]:
        witness = _simulate_crossing_witness(
            value=value,
            interfaces=interfaces,
            identities=identities,
            crossing=interface,
        )
        if witness is None:
            rejections.append(
                MultiIdentityRouteRejection(
                    interface_ref=interface.interface_ref,
                    failure_kind="material_preparation_unreachable",
                )
            )
        else:
            witnesses.append(witness)
    return MultiIdentityOpposedRoutesAnalysis(
        witnesses=tuple(witnesses),
        rejections=tuple(rejections),
        witness_limit=min(value.max_witnesses, 3),
    )


def _simulate_crossing_witness(
    *,
    value: MultiIdentityOpposedRoutesInput,
    interfaces: tuple[QuantizedInterfaceMeasurement, ...],
    identities: tuple[CompositeIdentityRouteMeasurement, ...],
    crossing: QuantizedInterfaceMeasurement,
) -> MultiIdentityRouteWitness | None:
    controls = tuple(
        (item.source_index, item.destination_index, item.component_ref)
        for item in value.controls
    )
    quantity_state = value.initial_quantities
    identity_state = {
        item.identity_ref: item.current_zone_index for item in identities
    }
    offsets = tuple(
        (item.identity_ref, item.attribute_offset) for item in identities
    )
    steps: list[MultiIdentityRouteStep] = []
    expanded_state_count = 0
    material_action_count = 0
    interface_event_count = 0
    crossing_distances: tuple[int, int] | None = None

    left_identity, right_identity = identities
    crossing_index = crossing.lower_index
    pre_events = (
        *tuple(
            ("pre_crossing", interfaces[index], (left_identity,))
            for index in range(crossing_index)
        ),
        *tuple(
            ("pre_crossing", interfaces[index], (right_identity,))
            for index in range(len(interfaces) - 1, crossing_index, -1)
        ),
    )
    crossing_event = (("atomic_crossing", crossing, identities),)
    post_events = (
        *tuple(
            ("post_crossing", interfaces[index], (left_identity,))
            for index in range(crossing_index + 1, len(interfaces))
        ),
        *tuple(
            ("post_crossing", interfaces[index], (right_identity,))
            for index in range(crossing_index - 1, -1, -1)
        ),
    )

    for stage, interface, moved_identities in (
        *pre_events,
        *crossing_event,
        *post_events,
    ):
        remaining_actions = value.max_actions_per_witness - len(steps)
        if remaining_actions <= 0:
            return None
        preparation = _first_shortest_route_witness(
            initial=quantity_state,
            controls=controls,
            goal_test=lambda state, item=interface: (
                state[item.lower_index] == item.activation_threshold
                and state[item.upper_index] == item.activation_threshold
            ),
            capacities=value.capacities,
            max_expanded_states=value.max_expanded_states_per_stage,
            max_depth=remaining_actions,
            control_order_variant=value.control_order_variant,
        )
        if preparation is None:
            return None
        expanded_state_count += preparation.expanded_state_count
        quantity_state, added = _append_quantity_steps(
            steps=steps,
            route=preparation.route,
            initial=quantity_state,
            identity_state=identity_state,
            offsets=offsets,
            stage=stage,
        )
        material_action_count += added
        if len(steps) >= value.max_actions_per_witness:
            return None

        moved_refs = tuple(item.identity_ref for item in moved_identities)
        if stage == "atomic_crossing":
            if (
                identity_state[left_identity.identity_ref] != interface.lower_index
                or identity_state[right_identity.identity_ref] != interface.upper_index
            ):
                return None
            identity_state[left_identity.identity_ref] = interface.upper_index
            identity_state[right_identity.identity_ref] = interface.lower_index
            crossing_distances = (
                left_identity.target_zone_index - interface.upper_index,
                interface.lower_index - right_identity.target_zone_index,
            )
            source_index = interface.lower_index
            destination_index = interface.upper_index
        else:
            moved = moved_identities[0]
            current_index = identity_state[moved.identity_ref]
            if moved.target_zone_index > moved.current_zone_index:
                source_index, destination_index = (
                    interface.lower_index,
                    interface.upper_index,
                )
            else:
                source_index, destination_index = (
                    interface.upper_index,
                    interface.lower_index,
                )
            if current_index != source_index:
                return None
            identity_state[moved.identity_ref] = destination_index
        interface_event_count += 1
        steps.append(
            _step(
                steps=steps,
                stage=stage,
                operation_kind="interface_transition",
                component_ref=interface.interface_ref,
                source_index=source_index,
                destination_index=destination_index,
                moved_identity_refs=moved_refs,
                quantities=quantity_state,
                identity_state=identity_state,
                offsets=offsets,
            )
        )

    if any(
        identity_state[item.identity_ref] != item.target_zone_index
        for item in identities
    ):
        return None
    remaining_actions = value.max_actions_per_witness - len(steps)
    terminal_requirements = dict(value.terminal_quantity_requirements)
    terminal = _first_shortest_route_witness(
        initial=quantity_state,
        controls=controls,
        goal_test=(
            (lambda state: state == value.terminal_quantities)
            if value.terminal_quantities
            else lambda state: all(
                state[index] == expected
                for index, expected in sorted(terminal_requirements.items())
            )
        ),
        capacities=value.capacities,
        max_expanded_states=value.max_expanded_states_per_stage,
        max_depth=remaining_actions,
        control_order_variant=value.control_order_variant,
    )
    if terminal is None:
        return None
    expanded_state_count += terminal.expanded_state_count
    quantity_state, added = _append_quantity_steps(
        steps=steps,
        route=terminal.route,
        initial=quantity_state,
        identity_state=identity_state,
        offsets=offsets,
        stage="terminal_alignment",
    )
    material_action_count += added
    if len(steps) > value.max_actions_per_witness or crossing_distances is None:
        return None

    digest = hashlib.sha256(
        canonical_json(
            (
                value.material_refs,
                tuple(item.identity_ref for item in identities),
                crossing.interface_ref,
                tuple((item.component_ref, item.source_index, item.destination_index) for item in steps),
            )
        ).encode("utf-8")
    ).hexdigest()[:16]
    return MultiIdentityRouteWitness(
        witness_ref=f"measurement:multi-route:{digest}",
        crossing_interface_ref=crossing.interface_ref,
        crossing_interface_index=crossing_index,
        steps=tuple(steps),
        material_action_count=material_action_count,
        interface_event_count=interface_event_count,
        primitive_action_count=len(steps),
        maximum_remaining_route_distance_after_crossing=max(crossing_distances),
        remaining_route_distance_imbalance_after_crossing=abs(
            crossing_distances[0] - crossing_distances[1]
        ),
        expanded_state_count=expanded_state_count,
        aggregate_conserved=all(
            sum(item.quantities_after) == sum(value.initial_quantities)
            for item in steps
        ),
        all_intermediate_states_valid=all(
            all(
                0 <= quantity <= capacity
                for quantity, capacity in zip(item.quantities_after, value.capacities)
            )
            for item in steps
        ),
        all_identity_transports_before_terminal_alignment=_identity_steps_precede_terminal(
            tuple(steps)
        ),
        atomic_crossing_identity_count=2,
        final_quantities=quantity_state,
        final_identity_zone_indices=_identity_state(identity_state),
    )


def _append_quantity_steps(
    *,
    steps: list[MultiIdentityRouteStep],
    route: tuple[tuple[int, int, str], ...],
    initial: tuple[int, ...],
    identity_state: dict[str, int],
    offsets: tuple[tuple[str, tuple[int, int]], ...],
    stage: str,
) -> tuple[tuple[int, ...], int]:
    state = initial
    for source, destination, component_ref in route:
        updated = list(state)
        updated[source] -= 1
        updated[destination] += 1
        state = tuple(updated)
        steps.append(
            _step(
                steps=steps,
                stage=stage,
                operation_kind="quantity_transfer",
                component_ref=component_ref,
                source_index=source,
                destination_index=destination,
                moved_identity_refs=(),
                quantities=state,
                identity_state=identity_state,
                offsets=offsets,
            )
        )
    return state, len(route)


def _step(
    *,
    steps: list[MultiIdentityRouteStep],
    stage: str,
    operation_kind: str,
    component_ref: str,
    source_index: int,
    destination_index: int,
    moved_identity_refs: tuple[str, ...],
    quantities: tuple[int, ...],
    identity_state: dict[str, int],
    offsets: tuple[tuple[str, tuple[int, int]], ...],
) -> MultiIdentityRouteStep:
    return MultiIdentityRouteStep.model_validate(
        {
            "step_index": len(steps) + 1,
            "stage": stage,
            "operation_kind": operation_kind,
            "component_ref": component_ref,
            "source_index": source_index,
            "destination_index": destination_index,
            "moved_identity_refs": moved_identity_refs,
            "quantities_after": quantities,
            "identity_zone_indices_after": _identity_state(identity_state),
            "composite_offsets_after": offsets,
        }
    )


def _identity_state(value: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(value.items()))


def _identity_steps_precede_terminal(
    steps: tuple[MultiIdentityRouteStep, ...],
) -> bool:
    terminal_started = False
    for item in steps:
        if item.stage == "terminal_alignment":
            terminal_started = True
        elif terminal_started and item.operation_kind == "interface_transition":
            return False
    return True
