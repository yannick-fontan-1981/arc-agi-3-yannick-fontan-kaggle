"""Pure bounded recipe, resize, and resource-order measurements."""

from agents.yf_arc3_v5.capabilities.contracts import (
    RecipePatternDifferenceInput,
    RecipePatternDifferenceMeasurements,
    RecipeResizeSuccessorInput,
    RecipeResizeSuccessorMeasurements,
    RecipeResourceOrderInput,
    RecipeResourceOrderMeasurements,
    RecipeSuccessorInput,
    RecipeSuccessorMeasurements,
)


def measure_recipe_pattern_difference(value: RecipePatternDifferenceInput) -> RecipePatternDifferenceMeasurements:
    current, target, supported = set(value.current_active_refs), set(value.target_active_refs), set(value.supported_edit_refs)
    missing = tuple(item for item in value.lattice_refs if item in target and item not in current)
    extra = tuple(item for item in value.lattice_refs if item in current and item not in target)
    required = (*missing, *extra)
    return RecipePatternDifferenceMeasurements(
        missing_active_refs=missing,
        extra_active_refs=extra,
        supported_required_edit_refs=tuple(item for item in required if item in supported),
        unsupported_required_edit_refs=tuple(item for item in required if item not in supported),
    )


def measure_recipe_successor(value: RecipeSuccessorInput) -> RecipeSuccessorMeasurements:
    active = list(value.current_active_refs)
    if value.edit_ref in active:
        active.remove(value.edit_ref)
    else:
        active.append(value.edit_ref)
    ordered = tuple(item for item in value.lattice_refs if item in set(active))
    triggered = set(ordered) == set(value.target_active_refs)
    return RecipeSuccessorMeasurements(
        active_refs_after_edit=ordered,
        active_refs_after_transaction=() if triggered and value.auto_clear_on_trigger else ordered,
        trigger_satisfied=triggered,
        resulting_world=value.supplied_world_after if triggered else value.world_before,
    )


def measure_recipe_resize_successor(value: RecipeResizeSuccessorInput) -> RecipeResizeSuccessorMeasurements:
    after = set(value.supplied_after_body.occupied_positions)
    substrate, walls = set(value.substrate_positions), set(value.wall_positions)
    return RecipeResizeSuccessorMeasurements(
        anchor_delta=(
            value.supplied_after_body.anchor_position[0] - value.before_body.anchor_position[0],
            value.supplied_after_body.anchor_position[1] - value.before_body.anchor_position[1],
        ),
        heading_preserved=value.before_body.heading_ref == value.supplied_after_body.heading_ref,
        substrate_missing_positions=tuple(sorted(after - substrate - walls)),
        wall_conflict_positions=tuple(sorted(after & walls)),
        contacted_resource_refs=tuple(
            region.resource_ref for region in value.resources if after & set(region.positions)
        ),
    )


def measure_recipe_resource_order(value: RecipeResourceOrderInput) -> RecipeResourceOrderMeasurements:
    quantity = value.initial_quantity
    trace: list[tuple[str, int]] = []
    first_insufficient = None
    for event in value.events:
        if event.event_kind == "cost":
            if event.amount > quantity:
                first_insufficient = event.event_ref
                break
            quantity -= event.amount
        else:
            quantity += event.amount
        trace.append((event.event_ref, quantity))
    return RecipeResourceOrderMeasurements(
        quantities_after_events=tuple(trace),
        first_insufficient_event_ref=first_insufficient,
        final_quantity=quantity,
    )
