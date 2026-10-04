"""Exact finite symbolic simulations and independent-dimension comparisons."""

from __future__ import annotations

from agents.yf_arc3_v5.capabilities.contracts import (
    ConstraintViolation,
    DimensionTransitionInput,
    DimensionTransitionResult,
    QuantitySimulationInput,
    QuantitySimulationResult,
    QuantityState,
)


def simulate_quantities(value: QuantitySimulationInput) -> QuantitySimulationResult:
    quantities = {item.entity_ref: item.quantity for item in value.initial}
    capacities = {item.entity_ref: item.capacity for item in value.initial}
    group_totals = {
        group.group_ref: sum(quantities[member] for member in group.member_refs)
        for group in value.conservation_groups
    }
    states = [_state(None, quantities)]
    violations: list[ConstraintViolation] = []

    for transfer in value.transfers:
        next_quantities = dict(quantities)
        next_quantities[transfer.source_ref] -= transfer.amount
        next_quantities[transfer.target_ref] += transfer.amount
        if next_quantities[transfer.source_ref] < 0:
            violations.append(
                ConstraintViolation(
                    transition_ref=transfer.transition_ref,
                    constraint_ref="constraint.non_negative",
                    entity_refs=(transfer.source_ref,),
                )
            )
        target_capacity = capacities[transfer.target_ref]
        if (
            target_capacity is not None
            and next_quantities[transfer.target_ref] > target_capacity
        ):
            violations.append(
                ConstraintViolation(
                    transition_ref=transfer.transition_ref,
                    constraint_ref="constraint.capacity",
                    entity_refs=(transfer.target_ref,),
                )
            )
        for group in value.conservation_groups:
            observed = sum(next_quantities[member] for member in group.member_refs)
            if observed != group_totals[group.group_ref]:
                violations.append(
                    ConstraintViolation(
                        transition_ref=transfer.transition_ref,
                        constraint_ref=f"constraint.conservation:{group.group_ref}",
                        entity_refs=group.member_refs,
                    )
                )
        quantities = next_quantities
        states.append(_state(transfer.transition_ref, quantities))

    return QuantitySimulationResult(
        states=tuple(states),
        violations=tuple(violations),
        all_valid=not violations,
    )


def compare_state_dimensions(
    value: DimensionTransitionInput,
) -> DimensionTransitionResult:
    before = {
        (item.entity_ref, item.dimension_ref): item.value_ref for item in value.before
    }
    after = {
        (item.entity_ref, item.dimension_ref): item.value_ref for item in value.after
    }
    dimensions = {dimension for _entity, dimension in before}
    changed = tuple(
        sorted(
            dimension
            for dimension in dimensions
            if any(
                item != after[(entity, candidate_dimension)]
                for (entity, candidate_dimension), item in sorted(before.items())
                if candidate_dimension == dimension
            )
        )
    )
    preserved = tuple(sorted(dimensions - set(changed)))
    violations = tuple(
        sorted(
            {
                *(
                    f"expected_changed:{item}"
                    for item in value.expected_changed_dimensions
                    if item not in changed
                ),
                *(
                    f"expected_preserved:{item}"
                    for item in value.expected_preserved_dimensions
                    if item not in preserved
                ),
            }
        )
    )
    return DimensionTransitionResult(
        changed_dimensions=changed,
        preserved_dimensions=preserved,
        expectation_violations=violations,
        exact=not violations,
    )


def _state(after: str | None, quantities: dict[str, int]) -> QuantityState:
    return QuantityState(
        after_transition_ref=after,
        quantities=tuple(sorted(quantities.items())),
    )
