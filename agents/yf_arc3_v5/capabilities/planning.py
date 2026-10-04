"""Exact palette-neutral quantized-chain planning and reconciliation."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

from agents.yf_arc3_v5.capabilities.contracts import (
    CompositeIdentityRouteMeasurement,
    CommittedQuantizedPlanAdvanceInput,
    ComponentDescription,
    InteractionProbeAgenda,
    InteractionProbeCandidate,
    MultiIdentityOpposedRoutesInput,
    Position,
    QuantizedInterfaceMeasurement,
    QuantizedPlanReconciliationAnalysis,
    QuantizedPlanReconciliationInput,
    QuantizedTransferControlMeasurement,
    VerifiedAlignmentActionInput,
)
from agents.yf_arc3_v5.logos.types import FrozenMap, canonical_json

POINT_ACTION_REF = "ACTION6"

_TransferStep = tuple[int, int, str]

_HARD_MAX_ROUTE_VARIANTS = 3
_HARD_MAX_STRUCTURAL_DESCRIPTIONS = 32
_HARD_MAX_EXPANDED_STATES = 4096
_HARD_MAX_PLAN_ACTIONS = 128
_HARD_MAX_QUANTITY_ASSIGNMENTS = 64


@dataclass(frozen=True)
class _BoundedRouteWitness:
    """First deterministic shortest witness found inside declared bounds."""

    state: tuple[int, ...]
    route: tuple[_TransferStep, ...]
    expanded_state_count: int


@dataclass(frozen=True)
class _DisconnectedRouteWitness:
    state: tuple[int, ...]
    marker_index: int
    route: tuple[tuple[int, int, str, str], ...]
    quantity_states: tuple[tuple[tuple[str, int], ...], ...]
    marker_states: tuple[str, ...]
    expanded_state_count: int


@dataclass(frozen=True)
class _BoundedQuantityDomain:
    assignments: tuple[tuple[int, ...], ...]
    complete: bool
    examined_assignment_count: int
    free_indices: tuple[int, ...]


def _bounded_feasible_quantity_assignments(
    *,
    quantity_count: int,
    conserved_total: int,
    exact_values: dict[int, int],
    lower_bounds: dict[int, int] | None = None,
    capacities: tuple[int, ...] = (),
    max_assignments: int = _HARD_MAX_QUANTITY_ASSIGNMENTS,
) -> _BoundedQuantityDomain:
    """Enumerate a finite quantity domain without choosing an assignment."""

    supplied_lower_bounds = dict(lower_bounds or {})
    free_indices = tuple(
        index for index in range(quantity_count) if index not in exact_values
    )
    if (
        quantity_count <= 0
        or conserved_total < 0
        or max_assignments <= 0
        or (capacities and len(capacities) != quantity_count)
        or any(index < 0 or index >= quantity_count for index in exact_values)
        or any(index < 0 or index >= quantity_count for index in supplied_lower_bounds)
        or any(value < 0 for value in sorted(exact_values.values()))
        or any(value < 0 for value in sorted(supplied_lower_bounds.values()))
    ):
        return _BoundedQuantityDomain((), True, 0, free_indices)

    upper_bounds = tuple(
        capacities[index] if capacities else conserved_total
        for index in range(quantity_count)
    )
    lower_values = tuple(
        exact_values.get(index, supplied_lower_bounds.get(index, 0))
        for index in range(quantity_count)
    )
    if any(
        exact_values.get(index, lower_values[index]) < lower_values[index]
        or lower_values[index] > upper_bounds[index]
        or exact_values.get(index, 0) > upper_bounds[index]
        for index in range(quantity_count)
    ):
        return _BoundedQuantityDomain((), True, 0, free_indices)

    assignments: list[tuple[int, ...]] = []
    examined_assignment_count = 0

    def visit(index: int, remaining: int, prefix: tuple[int, ...]) -> None:
        nonlocal examined_assignment_count
        if len(assignments) > max_assignments:
            return
        if index == quantity_count:
            examined_assignment_count += 1
            if remaining == 0:
                assignments.append(prefix)
            return
        if index in exact_values:
            candidates = (exact_values[index],)
        else:
            minimum = lower_values[index]
            maximum = min(upper_bounds[index], remaining)
            candidates = range(minimum, maximum + 1)
        for candidate in candidates:
            if candidate > remaining:
                continue
            suffix_minimum = sum(lower_values[index + 1 :])
            suffix_maximum = sum(upper_bounds[index + 1 :])
            next_remaining = remaining - candidate
            if not suffix_minimum <= next_remaining <= suffix_maximum:
                continue
            visit(index + 1, next_remaining, (*prefix, candidate))
            if len(assignments) > max_assignments:
                return

    visit(0, conserved_total, ())
    complete = len(assignments) <= max_assignments
    return _BoundedQuantityDomain(
        assignments=tuple(assignments[:max_assignments]),
        complete=complete,
        examined_assignment_count=examined_assignment_count,
        free_indices=free_indices,
    )


def _route_phase_count(route: tuple[_TransferStep, ...]) -> int:
    if not route:
        return 0
    components = tuple(step[2] for step in route)
    return 1 + sum(left != right for left, right in zip(components, components[1:]))


def _first_shortest_route_witness(
    *,
    initial: tuple[int, ...],
    controls: tuple[_TransferStep, ...],
    goal_test: Callable[[tuple[int, ...]], bool],
    allowed_indices: frozenset[int] | None = None,
    capacities: tuple[int, ...] = (),
    max_expanded_states: int = _HARD_MAX_EXPANDED_STATES,
    max_depth: int = _HARD_MAX_PLAN_ACTIONS,
    control_order_variant: str = "canonical",
) -> _BoundedRouteWitness | None:
    """Return one shortest witness without retaining alternative paths.

    The queue stores one predecessor for each exact state.  Consequently the
    number of live states and the route depth are both hard bounded, and path
    permutations can never multiply into route objects or compact families.
    Control ordering is transport-only and supplied by the declared policy's
    mechanical structural order.
    """

    if goal_test(initial):
        return _BoundedRouteWitness(initial, (), 0)
    if (
        not initial
        or not controls
        or max_expanded_states <= 0
        or max_depth <= 0
    ):
        return None
    canonical_controls = tuple(sorted(dict.fromkeys(controls)))
    if control_order_variant == "reverse_canonical":
        ordered_controls = tuple(reversed(canonical_controls))
    elif control_order_variant == "canonical_rotation_1" and canonical_controls:
        ordered_controls = (*canonical_controls[1:], canonical_controls[0])
    else:
        ordered_controls = canonical_controls
    queue = deque((initial,))
    depths = {initial: 0}
    predecessor: dict[
        tuple[int, ...], tuple[tuple[int, ...], _TransferStep]
    ] = {}
    expanded_state_count = 0
    while queue and expanded_state_count < max_expanded_states:
        current_state = queue.popleft()
        expanded_state_count += 1
        current_depth = depths[current_state]
        if current_depth >= max_depth:
            continue
        for source, destination, component_ref in ordered_controls:
            if allowed_indices is not None and (
                source not in allowed_indices or destination not in allowed_indices
            ):
                continue
            if current_state[source] <= 0:
                continue
            updated = list(current_state)
            updated[source] -= 1
            updated[destination] += 1
            if capacities and updated[destination] > capacities[destination]:
                continue
            next_state = tuple(updated)
            if next_state in depths:
                continue
            if len(depths) >= max_expanded_states:
                return None
            depths[next_state] = current_depth + 1
            predecessor[next_state] = (
                current_state,
                (source, destination, component_ref),
            )
            if goal_test(next_state):
                reverse_route: list[_TransferStep] = []
                cursor = next_state
                while cursor != initial:
                    parent, step = predecessor[cursor]
                    reverse_route.append(step)
                    cursor = parent
                return _BoundedRouteWitness(
                    state=next_state,
                    route=tuple(reversed(reverse_route)),
                    expanded_state_count=expanded_state_count,
                )
            queue.append(next_state)
    return None


def _declared_control_order_variants(policy: FrozenMap) -> tuple[str, ...]:
    """Read at most three mechanical traversal variants authored by DRM."""

    supported = {
        "canonical",
        "reverse_canonical",
        "canonical_rotation_1",
    }
    declared = tuple(
        str(item)
        for item in policy.get(
            "route_control_order_variants",
            ("canonical", "reverse_canonical", "canonical_rotation_1"),
        )
        if str(item) in supported
    )
    return tuple(dict.fromkeys(declared))[:_HARD_MAX_ROUTE_VARIANTS] or ("canonical",)


def _bounded_policy_integer(
    policy: FrozenMap,
    field: str,
    *,
    default: int,
    hard_maximum: int,
) -> int:
    raw = policy.get(field)
    try:
        declared = int(raw) if raw is not None else default
    except (TypeError, ValueError):
        declared = default
    return min(max(1, declared), hard_maximum)


def _ordered_structural_descriptions(
    descriptions: tuple[dict[str, object], ...],
    policy: FrozenMap,
) -> tuple[dict[str, object], ...]:
    """Apply only the structural ordering explicitly authored by DRM."""

    objective_order = tuple(
        str(item) for item in policy.get("objective_scope_order", ())
    )
    spatial_order = tuple(str(item) for item in policy.get("spatial_order", ()))
    origin_order = tuple(str(item) for item in policy.get("origin_order", ()))
    axis_to_spatial = {"row": "top", "column": "left"}

    def rank(value: str, order: tuple[str, ...]) -> int:
        try:
            return order.index(value)
        except ValueError:
            return len(order)

    return tuple(
        sorted(
            descriptions,
            key=lambda item: (
                rank(str(item.get("configuration_kind") or "single_display_assignment"), objective_order),
                rank(axis_to_spatial.get(str(item.get("axis")), ""), spatial_order),
                rank(str(item.get("origin_side") or ""), origin_order),
            ),
        )
    )

def enumerate_transferred_quantized_plans(
    value: VerifiedAlignmentActionInput,
) -> InteractionProbeAgenda:
    """Construct at most three exact route witnesses for a transfer chain.

    The capability does not choose a route or establish semantic roles.  It
    mechanically follows the DRM-authored structural order and stops after
    the declared global candidate limit.  Each structural hypothesis retains
    only its first shortest, non-negative, conservative witness.  A DRM policy
    performs the symbolic choice and SRC commits only that selected route.

    The construction is deliberately palette- and screen-direction-neutral:
    it tries both axes, both growth directions, every repeated material value,
    and every repeated marker value.  A description survives only when the
    complete geometry supplies a quantized material chain, paired boundary
    controls, a material-attached marker, and a distinct aligned target.
    """

    if POINT_ACTION_REF not in value.available_action_refs:
        return _empty_agenda()
    if value.committed_plan_facts and (
        value.committed_plan_cursor_present
        or not value.verified_repetition_cursor_end_observed
        or str(value.committed_plan_facts.get("plan_kind") or "")
        not in {
            "orthogonal_reflection_direct_body_partial_order_series",
            "orthogonal_reflection_focus_transfer_series",
        }
    ):
        continued = advance_committed_quantized_plan(
            CommittedQuantizedPlanAdvanceInput(
                committed_plan_facts=value.committed_plan_facts,
                available_action_refs=value.available_action_refs,
                continuation_policy=value.continuation_policy,
            )
        )
        if continued.has_candidates:
            return continued
    latest_mechanism = next(
        (
            item
            for item in reversed(value.mechanism_evidence)
            if item.quantum > 0
            and (
                item.mechanism_kind == "material_transfer"
                or item.supports_transferred_quantized_plan_seed
            )
        ),
        None,
    )
    if latest_mechanism is None:
        return _empty_agenda()
    inherited_quanta = (latest_mechanism.quantum,)
    prior_local_axes = (latest_mechanism.alignment_axis,)

    components = value.scene.blocks.components
    component_by_ref = {item.component_id: item for item in components}
    tracked_by_component = {
        item.component.component_id: item.entity_id for item in value.tracking.entities
    }
    structural_descriptions: list[dict[str, object]] = []
    folded_descriptions: list[dict[str, object]] = []
    for quantum in _current_visual_quantum_candidates(components):
        for axis in ("column", "row"):
            structural_descriptions.extend(
                _quantized_chain_descriptions(
                    components=components,
                    axis=axis,
                    quantum=quantum,
                )
            )
            structural_descriptions.extend(
                _disconnected_separator_descriptions(
                    components=components,
                    axis=axis,
                    quantum=quantum,
                )
            )
            folded_descriptions.extend(
                _folded_separator_descriptions(
                    components=components,
                    axis=axis,
                    quantum=quantum,
                    tracked_by_component=tracked_by_component,
                )
            )
    structural_descriptions = list(
        _compose_complete_display_assignments(
            descriptions=tuple(structural_descriptions),
            components=components,
        )
    )
    structural_descriptions.extend(folded_descriptions)
    structural_descriptions.extend(
        _multi_identity_opposed_route_descriptions(
            descriptions=tuple(structural_descriptions),
            components=components,
            tracked_by_component=tracked_by_component,
            route_policy=value.route_generation_policy,
        )
    )

    route_policy = value.route_generation_policy
    route_variant_limit = _bounded_policy_integer(
        route_policy,
        "max_route_variants",
        default=_HARD_MAX_ROUTE_VARIANTS,
        hard_maximum=_HARD_MAX_ROUTE_VARIANTS,
    )
    attempted_route_suffixes = frozenset(
        value.current_scope_attempted_route_suffixes
    )
    route_variant_limit = min(
        route_variant_limit,
        max(0, _HARD_MAX_ROUTE_VARIANTS - len(attempted_route_suffixes)),
    )
    if route_variant_limit == 0:
        return _empty_agenda()
    structural_description_limit = _bounded_policy_integer(
        route_policy,
        "max_structural_descriptions",
        default=_HARD_MAX_STRUCTURAL_DESCRIPTIONS,
        hard_maximum=_HARD_MAX_STRUCTURAL_DESCRIPTIONS,
    )
    max_expanded_states = _bounded_policy_integer(
        route_policy,
        "max_expanded_states_per_hypothesis",
        default=_HARD_MAX_EXPANDED_STATES,
        hard_maximum=_HARD_MAX_EXPANDED_STATES,
    )
    max_plan_actions = _bounded_policy_integer(
        route_policy,
        "max_plan_actions",
        default=_HARD_MAX_PLAN_ACTIONS,
        hard_maximum=_HARD_MAX_PLAN_ACTIONS,
    )
    stop_after_first = bool(route_policy.get("stop_after_first_viable_route"))
    control_order_variants = _declared_control_order_variants(route_policy)
    route_descriptions: list[dict[str, object]] = []
    ordered_descriptions = _ordered_structural_descriptions(
        tuple(structural_descriptions),
        route_policy,
    )
    structural_descriptions_examined = 0
    for description in ordered_descriptions[:structural_description_limit]:
        structural_descriptions_examined += 1
        description_control_orders = (
            control_order_variants[:1]
            if description.get("configuration_kind")
            == "multi_identity_opposed_routes"
            else control_order_variants
        )
        for control_order_variant in description_control_orders:
            route_alternatives = (
                _multi_identity_opposed_routes(
                    description,
                    max_expanded_states=max_expanded_states,
                    max_plan_actions=max_plan_actions,
                    control_order_variant=control_order_variant,
                )
                if description.get("configuration_kind")
                == "multi_identity_opposed_routes"
                else
                _disconnected_separator_routes(
                    description,
                    max_expanded_states=max_expanded_states,
                    max_plan_actions=max_plan_actions,
                    control_order_variant=control_order_variant,
                )
                if description.get("configuration_kind")
                in {
                    "disconnected_separator_sequence",
                    "folded_separator_sequence",
                }
                else _shortest_complete_assignment_routes(
                    description,
                    max_expanded_states=max_expanded_states,
                    max_plan_actions=max_plan_actions,
                    control_order_variant=control_order_variant,
                )
                if description.get("configuration_kind")
                == "complete_display_assignment"
                else _shortest_display_match_routes(
                    description,
                    max_expanded_states=max_expanded_states,
                    max_plan_actions=max_plan_actions,
                    control_order_variant=control_order_variant,
                )
            )
            known_route_refs = {
                str(item["route_ref"]) for item in route_descriptions
            }
            for route in route_alternatives:
                if str(route["route_ref"]) in known_route_refs:
                    continue
                route_suffix = str(route["route_ref"]).rsplit(".", 1)[-1]
                if route_suffix in attempted_route_suffixes:
                    continue
                route_descriptions.append(
                    {
                        **description,
                        **route,
                        "route_control_order_variant": control_order_variant,
                        "route_variant_limit": route_variant_limit,
                        "structural_descriptions_examined": structural_descriptions_examined,
                    }
                )
                if len(route_descriptions) >= route_variant_limit:
                    break
            if route_alternatives and (
                stop_after_first or len(route_descriptions) >= route_variant_limit
            ):
                break
        if route_descriptions and (
            stop_after_first or len(route_descriptions) >= route_variant_limit
        ):
            break

    described: list[tuple[InteractionProbeCandidate, FrozenMap]] = []
    discarded_no_effect_refs: list[str] = []
    no_effect_refs = frozenset(value.current_context_no_effect_candidate_refs)
    for representative in route_descriptions[:route_variant_limit]:
        component_ref = str(representative["first_component_ref"])
        component = next(
            item for item in components if item.component_id == component_ref
        )
        quantum = int(representative["quantum"])
        route_identity = str(representative["route_ref"]).rsplit(".", 1)[-1]
        candidate_ref = f"probe:{component_ref}:route:{route_identity}"
        if candidate_ref in no_effect_refs:
            discarded_no_effect_refs.append(candidate_ref)
            continue
        moving_component_ref = str(representative["moving_marker_ref"])
        fixed_component_ref = str(representative["fixed_marker_ref"])
        display_constraint_digest = hashlib.sha256(
            canonical_json(
                (
                    representative["display_constraint_extent_ref"],
                    representative["display_constraint_quantity"],
                    moving_component_ref,
                    fixed_component_ref,
                )
            ).encode("utf-8")
        ).hexdigest()[:12]
        plan_digest = hashlib.sha256(
            canonical_json(
                (
                    representative["axis"],
                    representative["origin_side"],
                    tuple(
                        tracked_by_component.get(str(item)) or str(item)
                        for item in representative["material_refs"]
                    ),
                    tracked_by_component.get(moving_component_ref)
                    or moving_component_ref,
                    tracked_by_component.get(fixed_component_ref)
                    or fixed_component_ref,
                    representative["display_constraint_quantity"],
                )
            ).encode("utf-8")
        ).hexdigest()[:12]
        checkpoint_interval = max(
            1,
            int(value.continuation_policy.get("regular_checkpoint_interval") or 1),
        )
        checkpoint_reason = _checkpoint_reason(
            steps=tuple(representative["route_transfer_steps"]),
            executed_component_refs=(),
            completed_click_count=0,
            last_full_checkpoint_click_count=0,
            interval=checkpoint_interval,
            enabled_conditions=tuple(
                str(item)
                for item in value.continuation_policy.get("checkpoint_on", ())
            ),
        )
        facts = FrozenMap(
            {
                "candidate_ref": candidate_ref,
                "component_ref": component_ref,
                "tracked_entity_ref": tracked_by_component.get(component_ref),
                "route_simulation_completed": True,
                "all_intermediate_states_valid": True,
                "aggregate_conserved": True,
                "display_constraint_satisfied_by_simulation": True,
                "first_action_moves_reusable_extent_toward_displayed_constraint": bool(
                    representative.get(
                        "first_action_moves_reusable_extent_toward_displayed_constraint"
                    )
                ),
                "primitive_action_count": int(representative["primitive_action_count"]),
                "transfer_phase_count": int(representative["transfer_phase_count"]),
                "actuator_switch_count": int(representative["actuator_switch_count"]),
                "route_ref": representative["route_ref"],
                "plan_structure_digest": plan_digest,
                "route_variant_limit": route_variant_limit,
                "route_variants_materialized": len(route_descriptions),
                "current_scope_attempted_route_suffixes": tuple(
                    sorted(attempted_route_suffixes)
                ),
                "remaining_global_route_attempt_budget": (
                    _HARD_MAX_ROUTE_VARIANTS - len(attempted_route_suffixes)
                ),
                "route_control_order_variant": representative[
                    "route_control_order_variant"
                ],
                "structural_descriptions_examined": int(
                    representative["structural_descriptions_examined"]
                ),
                "expanded_state_count": int(
                    representative.get("expanded_state_count") or 0
                ),
                "current_context_no_effect_observed": False,
                "route_component_refs": representative["route_component_refs"],
                "all_route_component_refs": tuple(
                    dict.fromkeys(representative["route_component_refs"])
                ),
                "route_transfer_steps": representative["route_transfer_steps"],
                "route_quantity_states": representative["route_quantity_states"],
                "route_marker_states": representative.get("route_marker_states", ()),
                "route_component_entity_bindings": tuple(
                    (component_ref, tracked_by_component[component_ref])
                    for component_ref in dict.fromkeys(
                        str(step[2])
                        for step in representative["route_transfer_steps"]
                    )
                    if component_ref in tracked_by_component
                ),
                "route_action_points": tuple(
                    (
                        route_component_ref,
                        _interior_point(component_by_ref[route_component_ref])[1],
                        _interior_point(component_by_ref[route_component_ref])[0],
                    )
                    for route_component_ref in dict.fromkeys(
                        str(step[2])
                        for step in representative["route_transfer_steps"]
                    )
                    if route_component_ref in component_by_ref
                ),
                "executed_component_refs": (),
                "current_quantities": representative["current_quantities"],
                "expected_quantities_after_action": representative[
                    "expected_quantities_after_action"
                ],
                "display_constraint_extent_ref": representative[
                    "display_constraint_extent_ref"
                ],
                "display_constraint_quantity": representative[
                    "display_constraint_quantity"
                ],
                "display_constraint_digest": display_constraint_digest,
                "moving_entity_ref": (
                    tracked_by_component.get(moving_component_ref)
                    or moving_component_ref
                ),
                "fixed_entity_ref": (
                    tracked_by_component.get(fixed_component_ref)
                    or fixed_component_ref
                ),
                "alignment_axis": representative["axis"],
                "origin_side": representative["origin_side"],
                "material_value": representative["material_value"],
                "material_chain_bands": _material_chain_bands(
                    description=representative,
                    components=components,
                ),
                "material_axis_origin_coordinate": representative["origin"],
                "material_axis_origin_coordinates": representative.get(
                    "material_axis_origins", ()
                ),
                "moving_marker_ref": moving_component_ref,
                "fixed_marker_ref": fixed_component_ref,
                "moving_marker_signature": representative.get(
                    "moving_marker_signature", ()
                ),
                "quantum": quantum,
                "inherited_visual_quanta": inherited_quanta,
                "current_visual_quantum": quantum,
                "global_scale_relation": _scale_relation(
                    inherited_quanta=inherited_quanta,
                    current_quantum=quantum,
                ),
                "prior_local_axes": prior_local_axes,
                "current_local_axis": representative["axis"],
                "global_rotation_relation": _rotation_relation(
                    prior_axes=prior_local_axes,
                    current_axis=str(representative["axis"]),
                ),
                "configuration_kind": str(
                    representative.get("configuration_kind")
                    or "single_display_assignment"
                ),
                "reconfiguration_domain_assignments": representative.get(
                    "reconfiguration_domain_assignments", ()
                ),
                "reconfiguration_domain_complete": bool(
                    representative.get("reconfiguration_domain_complete")
                ),
                "reconfiguration_domain_feasible": bool(
                    representative.get("reconfiguration_domain_feasible")
                ),
                "reconfiguration_domain_assignment_count": int(
                    representative.get("reconfiguration_domain_assignment_count")
                    or 0
                ),
                "reconfiguration_domain_examined_assignment_count": int(
                    representative.get(
                        "reconfiguration_domain_examined_assignment_count"
                    )
                    or 0
                ),
                "reconfiguration_domain_enumeration_limit": int(
                    representative.get("reconfiguration_domain_enumeration_limit")
                    or 0
                ),
                "reconfiguration_domain_free_residual_refs": representative.get(
                    "reconfiguration_domain_free_residual_refs", ()
                ),
                "reconfiguration_domain_retains_free_residual_quantities": bool(
                    representative.get(
                        "reconfiguration_domain_retains_free_residual_quantities"
                    )
                ),
                "reconfiguration_domain_conserved_total": int(
                    representative.get("reconfiguration_domain_conserved_total")
                    or 0
                ),
                "reconfiguration_domain_supplied_constraint_count": int(
                    representative.get(
                        "reconfiguration_domain_supplied_constraint_count"
                    )
                    or 0
                ),
                "displayed_quantity_vector": representative.get(
                    "displayed_quantity_vector", ()
                ),
                "displayed_marker_bindings": representative.get(
                    "displayed_marker_bindings", ()
                ),
                **(
                    {
                        "capacities": representative["capacities"],
                        "capacity_measurement_available": True,
                    }
                    if representative.get("capacities")
                    else {}
                ),
                **(
                    {
                        "repeated_fill_empty_component_refs": representative[
                            "repeated_fill_empty_component_refs"
                        ],
                        "buffer_cycle_witnesses": representative.get(
                            "buffer_cycle_witnesses", ()
                        ),
                    }
                    if representative.get("repeated_fill_empty_component_refs")
                    else {}
                ),
                "inherited_mechanism_evidence_refs": tuple(
                    item.evidence_ref for item in value.mechanism_evidence
                ),
                "remaining_click_count": int(representative["primitive_action_count"]),
                "remaining_step_count": int(representative["primitive_action_count"]),
                "route_total_click_count": int(representative["primitive_action_count"]),
                "executed_click_count": 0,
                "current_step_index": 1,
                "last_full_checkpoint_click_count": 0,
                "checkpoint_interval": checkpoint_interval,
                "checkpoint_after_current_step": bool(checkpoint_reason),
                "checkpoint_reason_after_current_step": checkpoint_reason,
                **_disconnected_topology_route_facts(representative),
                **_multi_identity_route_facts(representative),
            }
        )
        described.append(
            (
                InteractionProbeCandidate(
                    candidate_ref=candidate_ref,
                    action_ref=POINT_ACTION_REF,
                    action_data=FrozenMap(
                        {
                            "x": _interior_point(component)[1],
                            "y": _interior_point(component)[0],
                        }
                    ),
                    component_ref=component_ref,
                    point=_interior_point(component),
                ),
                facts,
            )
        )
    ordered = tuple(sorted(described, key=lambda item: item[0].candidate_ref))
    candidates = tuple(item[0] for item in ordered)
    dominance_context = _multi_identity_dominance_context(
        ordered=ordered,
        route_policy=route_policy,
    )
    return InteractionProbeAgenda(
        candidates=candidates,
        alternative_refs=tuple(item.candidate_ref for item in candidates),
        alternative_facts=FrozenMap(
            {candidate.candidate_ref: facts for candidate, facts in ordered}
        ),
        context_facts=dominance_context,
        has_candidates=bool(candidates),
        current_context_no_effect_candidate_refs=tuple(discarded_no_effect_refs),
    )


def _multi_identity_dominance_context(
    *,
    ordered: tuple[tuple[InteractionProbeCandidate, FrozenMap], ...],
    route_policy: FrozenMap,
) -> FrozenMap:
    if not ordered or any(
        facts.get("configuration_kind") != "multi_identity_opposed_routes"
        for _candidate, facts in ordered
    ):
        return FrozenMap({"selection_mode": "declared_route_order"})
    supported_fields = {
        "maximum_remaining_route_distance_after_crossing",
        "remaining_route_distance_imbalance_after_crossing",
        "primitive_action_count",
    }
    criterion_order = tuple(
        str(item)
        for item in route_policy.get(
            "multi_identity_dominance_criterion_order", ()
        )
        if str(item) in supported_fields
    )
    if set(criterion_order) != supported_fields:
        return FrozenMap({"selection_mode": "declared_route_order"})
    witnesses: list[FrozenMap] = []
    for left_index, (left_candidate, left_facts) in enumerate(ordered):
        for right_candidate, right_facts in ordered[left_index + 1 :]:
            for field in criterion_order:
                left_value = int(left_facts.get(field) or 0)
                right_value = int(right_facts.get(field) or 0)
                if left_value == right_value:
                    continue
                lower_ref, higher_ref = (
                    (left_candidate.candidate_ref, right_candidate.candidate_ref)
                    if left_value < right_value
                    else (right_candidate.candidate_ref, left_candidate.candidate_ref)
                )
                witnesses.append(
                    FrozenMap(
                        {
                            "criterion_ref": f"measurement:{field}",
                            "left_ref": lower_ref,
                            "right_ref": higher_ref,
                            "preferred_order": "minimize",
                            "observed_order": "less_than",
                            "evidence_refs": (
                                f"measurement:{field}:{left_value}:{right_value}",
                            ),
                        }
                    )
                )
                break
    return FrozenMap(
        {
            "selection_mode": "explicit_symbolic_dominance",
            "dominance_criterion_order": criterion_order,
            "dominance_witnesses": tuple(witnesses),
        }
    )


def analyze_quantized_plan_reconciliation(
    value: QuantizedPlanReconciliationInput,
) -> QuantizedPlanReconciliationAnalysis:
    """Compare one committed predicted quantity delta with world feedback."""

    facts = value.selected_plan_facts
    expected_pairs = tuple(facts.get("expected_quantities_after_action") or ())
    expected_values = tuple(int(item[1]) for item in expected_pairs)
    axis = str(facts.get("alignment_axis") or "")
    origin_side = str(facts.get("origin_side") or "")
    material_value = facts.get("material_value")
    quantum = int(facts.get("quantum") or 0)
    terminal_expected = int(facts.get("remaining_click_count") or 0) == 1
    score_increased = value.after_score > value.before_score
    terminal_completed = terminal_expected and score_increased
    actual_values: tuple[int, ...] = ()
    if (
        not terminal_completed
        and value.after_scene is not None
        and axis in {"column", "row"}
        and origin_side in {"minimum", "maximum"}
        and isinstance(material_value, int)
        and quantum > 0
    ):
        raw_bands = tuple(facts.get("material_chain_bands") or ())
        raw_origins = tuple(
            int(item)
            for item in facts.get("material_axis_origin_coordinates") or ()
        )
        scalar_origin = facts.get("material_axis_origin_coordinate")
        if raw_bands and (
            len(raw_origins) == len(raw_bands)
            or isinstance(scalar_origin, int)
        ):
            observed: list[int] = []
            for index, raw_band in enumerate(raw_bands):
                band = tuple(raw_band)
                if len(band) != 3:
                    observed = []
                    break
                _material_ref, band_start, band_end = band
                origin_coordinate = (
                    raw_origins[index]
                    if len(raw_origins) == len(raw_bands)
                    else int(scalar_origin)
                )
                matches = tuple(
                    item
                    for item in value.after_scene.blocks.components
                    if item.value == material_value
                    and _orthogonal_start(item, axis) >= int(band_start)
                    and _orthogonal_end(item, axis) <= int(band_end)
                    and _axis_origin(item, axis, origin_side) == origin_coordinate
                    and _axis_extent(item, axis) % quantum == 0
                )
                observed.append(
                    max(
                        (_axis_extent(item, axis) // quantum for item in matches),
                        default=0,
                    )
                )
            actual_values = tuple(observed)
        else:
            material = tuple(
                sorted(
                    (
                        item
                        for item in value.after_scene.blocks.components
                        if item.value == material_value
                        and _axis_extent(item, axis) % quantum == 0
                        and _axis_extent(item, axis) >= quantum
                    ),
                    key=lambda item: _orthogonal_start(item, axis),
                )
            )
            actual_values = tuple(
                _axis_extent(item, axis) // quantum for item in material
            )
    exact_match: bool | None = (
        None
        if terminal_completed
        else bool(expected_values) and actual_values == expected_values
    )
    expected_marker_zone = str(
        facts.get("expected_marker_zone_after_action") or ""
    )
    observed_marker_zone = ""
    marker_identity_preserved: bool | None = None
    if expected_marker_zone and not terminal_completed:
        observed_marker_zone, marker_identity_preserved = _observed_marker_zone(
            facts=facts,
            after_scene=value.after_scene,
            after_tracking=value.after_tracking,
        )
        exact_match = (
            exact_match
            and observed_marker_zone == expected_marker_zone
            and marker_identity_preserved is True
        )
    expected_identity_zones = tuple(
        tuple(item) for item in facts.get("expected_identity_zones_after_action") or ()
    )
    expected_composite_offsets = tuple(
        (str(item[0]), (int(item[3]), int(item[4])))
        for item in facts.get("identity_composite_bindings") or ()
        if len(tuple(item)) == 5
    )
    observed_identity_zones: tuple[tuple[str, str], ...] = ()
    observed_composite_offsets: tuple[tuple[str, tuple[int, int]], ...] = ()
    nonzero_comotion_identity_refs: tuple[str, ...] = ()
    composite_mismatch_identity_refs: tuple[str, ...] = ()
    relationally_reacquired_identity_refs: tuple[str, ...] = ()
    expected_moved_identity_refs = tuple(
        str(item)
        for item in facts.get("expected_moved_identity_refs_after_action", ())
    )
    multi_identity_exact_match: bool | None = None
    if expected_identity_zones and not terminal_completed:
        (
            observed_identity_zones,
            observed_composite_offsets,
            multi_identity_exact_match,
            nonzero_comotion_identity_refs,
            composite_mismatch_identity_refs,
            relationally_reacquired_identity_refs,
        ) = _observed_multi_identity_state(
            facts=facts,
            after_tracking=value.after_tracking,
            expected_identity_zones=expected_identity_zones,
            expected_composite_offsets=expected_composite_offsets,
            expected_moved_identity_refs=expected_moved_identity_refs,
        )
    exact_plan_match = (
        exact_match
        if multi_identity_exact_match is None
        else exact_match is True and multi_identity_exact_match
    )
    failure_measurements = _quantized_transfer_failure_measurements(
        facts=facts,
        actual_values=actual_values,
        exact_plan_match=exact_plan_match,
        score_increased=score_increased,
    )
    descriptive_delta = {
        "transition_ref": value.transition_ref,
        "before_score": value.before_score,
        "after_score": value.after_score,
        "score_increased": score_increased,
        "terminal_step_expected": terminal_expected,
        "terminal_reconciliation_uses_score_boundary_only": terminal_completed,
        "expected_quantities": expected_values,
        "observed_quantities": actual_values,
        "exact_quantity_match": exact_match,
        "exact_plan_match": exact_plan_match,
        "quantity_measurement_status": (
            "not_required_after_terminal_score_boundary"
            if terminal_completed
            else "measured_from_current_scene"
        ),
        "score_transition_consistent": score_increased == terminal_expected,
        "expected_marker_zone": expected_marker_zone,
        "observed_marker_zone": observed_marker_zone,
        "marker_identity_preserved": marker_identity_preserved,
        "expected_identity_zones": expected_identity_zones,
        "observed_identity_zones": observed_identity_zones,
        "expected_composite_offsets": expected_composite_offsets,
        "observed_composite_offsets": observed_composite_offsets,
        "multi_identity_exact_match": multi_identity_exact_match,
        "nonzero_comotion_identity_refs": nonzero_comotion_identity_refs,
        "composite_mismatch_identity_refs": composite_mismatch_identity_refs,
        "relationally_reacquired_identity_refs": (
            relationally_reacquired_identity_refs
        ),
        "nonzero_comotion_observed_for_any_identity": bool(
            nonzero_comotion_identity_refs
        ),
        "nonzero_comotion_observed_for_every_expected_moved_identity": (
            bool(expected_moved_identity_refs)
            and set(expected_moved_identity_refs).issubset(
                nonzero_comotion_identity_refs
            )
        ),
        "composite_mismatch_observed": bool(composite_mismatch_identity_refs),
        "extent_vector_unchanged_during_zone_transition_candidate": (
            actual_values
            == tuple(int(item[1]) for item in facts.get("current_quantities") or ())
            if facts.get("current_step_measurement_kind")
            in {
                "separator_zone_transition_candidate",
                "multi_identity_interface_transition_candidate",
            }
            else None
        ),
        "dimension_preservation_observation": FrozenMap(
            {
                "marker_zone_before": facts.get("current_marker_zone"),
                "marker_zone_after": observed_marker_zone,
                "marker_identity_preserved": marker_identity_preserved,
                "quantities_unchanged_from_before": (
                    actual_values
                    == tuple(
                        int(item[1])
                        for item in facts.get("current_quantities") or ()
                    )
                ),
                "interacted_component_ref": facts.get("component_ref"),
            }
        ),
        **failure_measurements,
    }
    return QuantizedPlanReconciliationAnalysis(
        descriptive_delta=FrozenMap(descriptive_delta),
        evidence_refs=(
            f"evidence.transition:{value.transition_ref}",
            str(facts.get("route_ref") or facts.get("plan_ref") or "plan"),
        ),
        transition_ref=value.transition_ref,
    )


def _quantized_transfer_failure_measurements(
    *,
    facts: FrozenMap,
    actual_values: tuple[int, ...],
    exact_plan_match: bool | None,
    score_increased: bool,
) -> dict[str, object]:
    """Measure inspectable facts for DRM failure interpretation."""

    current_pairs = tuple(facts.get("current_quantities") or ())
    current_values = tuple(int(item[1]) for item in current_pairs)
    current_by_ref = {str(item[0]): int(item[1]) for item in current_pairs}
    measurement_available = bool(current_values) and len(actual_values) == len(
        current_values
    )
    unchanged = measurement_available and actual_values == current_values
    failure_packet_present = exact_plan_match is False and not score_increased
    steps = tuple(facts.get("route_transfer_steps") or ())
    source_ref = str(steps[0][0]) if steps and len(tuple(steps[0])) >= 2 else ""
    target_ref = str(steps[0][1]) if steps and len(tuple(steps[0])) >= 2 else ""
    donor_quantity_known = source_ref in current_by_ref
    donor_zero = bool(
        failure_packet_present
        and unchanged
        and donor_quantity_known
        and current_by_ref[source_ref] == 0
    )
    material_refs = tuple(str(item[0]) for item in current_pairs)
    capacities = tuple(int(item) for item in facts.get("capacities") or ())
    target_index = (
        material_refs.index(target_ref) if target_ref in material_refs else -1
    )
    capacity_measurement_available = bool(
        facts.get("capacity_measurement_available")
        and len(capacities) == len(material_refs)
        and target_index >= 0
    )
    recipient_at_capacity = bool(
        failure_packet_present
        and unchanged
        and capacity_measurement_available
        and current_by_ref[target_ref] >= capacities[target_index]
    )
    control_binding_match = facts.get("control_binding_match_observed")
    passage_ready = facts.get("passage_ready_before_action")
    return {
        "transfer_failure_packet_present": failure_packet_present,
        "transfer_no_effect_measured": bool(failure_packet_present and unchanged),
        "transfer_quantity_measurement_available": measurement_available,
        "transfer_source_ref_measured": source_ref,
        "transfer_target_ref_measured": target_ref,
        "transfer_donor_quantity_known": donor_quantity_known,
        "transfer_donor_zero_measured": donor_zero,
        "transfer_recipient_capacity_measurement_available": (
            capacity_measurement_available
        ),
        "transfer_recipient_at_capacity_measured": recipient_at_capacity,
        "transfer_control_binding_mismatch_observed": (
            control_binding_match is False
        ),
        "transfer_unready_passage_observed": passage_ready is False,
    }


def _observed_marker_zone(
    *,
    facts: FrozenMap,
    after_scene: object,
    after_tracking: object,
) -> tuple[str, bool | None]:
    raw_bands = tuple(facts.get("zone_partition_bands") or ())
    axis = str(facts.get("alignment_axis") or "")
    moving_entity_ref = str(facts.get("moving_entity_ref") or "")
    component: ComponentDescription | None = None
    identity_preserved: bool | None = None
    if after_tracking is not None and hasattr(after_tracking, "entities"):
        tracked = next(
            (
                item
                for item in after_tracking.entities
                if item.entity_id == moving_entity_ref
            ),
            None,
        )
        if tracked is not None:
            component = tracked.component
            identity_preserved = tracked.identity_status == "established"
    if component is None and hasattr(after_scene, "blocks"):
        signature = tuple(facts.get("moving_marker_signature") or ())
        if len(signature) == 2:
            marker_value, relative_pixels = signature
            matches = tuple(
                item
                for item in after_scene.blocks.components
                if item.value == int(marker_value)
                and item.relative_pixels == tuple(relative_pixels)
            )
            if len(matches) == 1:
                component = matches[0]
                identity_preserved = None
    if component is None or axis not in {"row", "column"}:
        return "", identity_preserved
    return (
        _component_zone_ref(facts=facts, component=component, axis=axis),
        identity_preserved,
    )


def _observed_multi_identity_state(
    *,
    facts: FrozenMap,
    after_tracking: object,
    expected_identity_zones: tuple[tuple[object, ...], ...],
    expected_composite_offsets: tuple[tuple[str, tuple[int, int]], ...],
    expected_moved_identity_refs: tuple[str, ...],
) -> tuple[
    tuple[tuple[str, str], ...],
    tuple[tuple[str, tuple[int, int]], ...],
    bool,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    """Measure every declared carrier zone and invariant attribute offset."""

    raw_bands = tuple(facts.get("zone_partition_bands") or ())
    axis = str(facts.get("alignment_axis") or "")
    raw_bindings = tuple(facts.get("identity_composite_bindings") or ())
    if (
        after_tracking is None
        or not hasattr(after_tracking, "entities")
        or axis not in {"row", "column"}
        or not raw_bands
    ):
        return (), (), False, (), (), ()
    tracked = {item.entity_id: item for item in after_tracking.entities}
    carrier_signature_by_identity = {
        str(item[0]): (
            int(item[1]),
            tuple(tuple(position) for position in item[2]),
        )
        for item in facts.get("identity_carrier_signatures") or ()
        if len(tuple(item)) == 3
    }
    expected_offset_by_identity = dict(expected_composite_offsets)
    observed_zones: list[tuple[str, str]] = []
    observed_offsets: list[tuple[str, tuple[int, int]]] = []
    identities_preserved = True
    nonzero_comotion_refs: list[str] = []
    mismatch_refs: list[str] = []
    relationally_reacquired_refs: list[str] = []
    for raw_binding in raw_bindings:
        binding = tuple(raw_binding)
        if len(binding) != 5:
            return (), (), False, (), (), ()
        identity_ref = str(binding[0])
        carrier = tracked.get(str(binding[1]))
        attribute = tracked.get(str(binding[2]))
        carrier_reacquired = False
        expected_offset = expected_offset_by_identity.get(identity_ref)
        signature = carrier_signature_by_identity.get(identity_ref)
        if (
            carrier is None
            and attribute is not None
            and expected_offset is not None
            and signature is not None
        ):
            carrier_value, carrier_pixels = signature
            candidates = tuple(
                item
                for item in after_tracking.entities
                if item.entity_id != attribute.entity_id
                and item.component.value == carrier_value
                and item.component.relative_pixels == carrier_pixels
                and item.component.bbox.top
                == attribute.component.bbox.top - expected_offset[0]
                and item.component.bbox.left
                == attribute.component.bbox.left - expected_offset[1]
            )
            if len(candidates) == 1:
                carrier = candidates[0]
                carrier_reacquired = True
                relationally_reacquired_refs.append(identity_ref)
        if carrier is None or attribute is None:
            return (
                tuple(observed_zones),
                tuple(observed_offsets),
                False,
                tuple(nonzero_comotion_refs),
                tuple((*mismatch_refs, identity_ref)),
                tuple(relationally_reacquired_refs),
            )
        identities_preserved = identities_preserved and (
            (carrier.identity_status == "established" or carrier_reacquired)
            and attribute.identity_status == "established"
        )
        same_delta = carrier_reacquired or (
            carrier.delta_row == attribute.delta_row
            and carrier.delta_col == attribute.delta_col
        )
        nonzero_delta = (
            (attribute.delta_row, attribute.delta_col) != (0, 0)
            if carrier_reacquired
            else (carrier.delta_row, carrier.delta_col) != (0, 0)
        )
        if same_delta and nonzero_delta:
            nonzero_comotion_refs.append(identity_ref)
        elif identity_ref in expected_moved_identity_refs:
            mismatch_refs.append(identity_ref)
        observed_zones.append(
            (
                identity_ref,
                _component_zone_ref(
                    facts=facts,
                    component=carrier.component,
                    axis=axis,
                ),
            )
        )
        observed_offsets.append(
            (
                identity_ref,
                (
                    attribute.component.bbox.top - carrier.component.bbox.top,
                    attribute.component.bbox.left - carrier.component.bbox.left,
                ),
            )
        )
    normalized_expected_zones = tuple(
        (str(item[0]), str(item[1]))
        for item in expected_identity_zones
        if len(item) == 2
    )
    for identity_ref, offset in observed_offsets:
        if expected_offset_by_identity.get(identity_ref) != offset:
            mismatch_refs.append(identity_ref)
    exact = (
        identities_preserved
        and tuple(observed_zones) == normalized_expected_zones
        and tuple(observed_offsets) == expected_composite_offsets
        and set(expected_moved_identity_refs).issubset(nonzero_comotion_refs)
    )
    return (
        tuple(observed_zones),
        tuple(observed_offsets),
        exact,
        tuple(nonzero_comotion_refs),
        tuple(dict.fromkeys(mismatch_refs)),
        tuple(relationally_reacquired_refs),
    )


def _component_zone_ref(
    *,
    facts: FrozenMap,
    component: ComponentDescription,
    axis: str,
) -> str:
    row = (component.bbox.top + component.bbox.bottom) // 2
    column = (component.bbox.left + component.bbox.right) // 2
    boxes = tuple(facts.get("zone_measurement_boxes") or ())
    box_matches = tuple(
        str(item[0])
        for item in boxes
        if len(tuple(item)) == 5
        and int(item[1]) <= row <= int(item[3])
        and int(item[2]) <= column <= int(item[4])
    )
    if len(box_matches) == 1:
        return box_matches[0]
    center = (
        _orthogonal_start(component, axis) + _orthogonal_end(component, axis)
    ) // 2
    band_matches = tuple(
        str(zone_ref)
        for zone_ref, start, end in tuple(
            facts.get("zone_partition_bands") or ()
        )
        if int(start) <= center <= int(end)
    )
    return band_matches[0] if len(band_matches) == 1 else ""


def _empty_agenda() -> InteractionProbeAgenda:
    return InteractionProbeAgenda(
        candidates=(),
        alternative_refs=(),
        alternative_facts=FrozenMap(),
        has_candidates=False,
    )


def _current_visual_quantum_candidates(
    components: tuple[ComponentDescription, ...],
) -> tuple[int, ...]:
    """Return repeated exact filled-square extents without assigning a role."""

    counts = Counter(
        item.bbox.width
        for item in components
        if item.bbox.width == item.bbox.height
        and item.area == item.bbox.width * item.bbox.height
    )
    return tuple(sorted(size for size, count in sorted(counts.items()) if count >= 2))


def _scale_relation(
    *,
    inherited_quanta: tuple[int, ...],
    current_quantum: int,
) -> str:
    if current_quantum in inherited_quanta:
        return "scale.1/1.preserved"
    prior = min(inherited_quanta, key=lambda item: (abs(item - current_quantum), item))
    ratio = Fraction(current_quantum, prior)
    direction = "zoom_out" if ratio < 1 else "zoom_in"
    return f"scale.{ratio.numerator}/{ratio.denominator}.{direction}"


def _rotation_relation(*, prior_axes: tuple[str, ...], current_axis: str) -> str:
    if not prior_axes or current_axis in prior_axes:
        return "rotation.local_axis_preserved"
    return "rotation.quarter_turn_local_frame"


def _compose_complete_display_assignments(
    *,
    descriptions: tuple[dict[str, object], ...],
    components: tuple[ComponentDescription, ...],
) -> tuple[dict[str, object], ...]:
    """Add exact joint display constraints without deleting local alternatives."""

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for item in descriptions:
        grouped[
            (
                item["axis"],
                item["origin_side"],
                item["origin"],
                item["quantum"],
                item["material_value"],
                item["material_refs"],
                item["controls"],
            )
        ].append(item)
    promoted: list[dict[str, object]] = []
    for members in (grouped[__yf_order_key] for __yf_order_key in sorted(grouped)):
        representative = members[0]
        capacities = _infer_capacity_map(
            description=representative,
            components=components,
        )
        quantities = tuple(int(item) for item in representative["quantities"])  # type: ignore[arg-type]
        display_constraints_by_index: dict[int, dict[str, object]] = {}
        contradictory = False
        for item in members:
            index = int(item["display_constraint_index"])
            constraint = {
                "material_index": index,
                "material_ref": str(item["display_constraint_extent_ref"]),
                "target_quantity": int(item["display_constraint_quantity"]),
                "moving_marker_ref": str(item["moving_marker_ref"]),
                "fixed_marker_ref": str(item["fixed_marker_ref"]),
            }
            existing = display_constraints_by_index.get(index)
            if existing is not None and existing != constraint:
                contradictory = True
                break
            display_constraints_by_index[index] = constraint
        exact_values = {
            index: int(item["target_quantity"])
            for index, item in sorted(display_constraints_by_index.items())
        }
        domain = _bounded_feasible_quantity_assignments(
            quantity_count=len(quantities),
            conserved_total=sum(quantities),
            exact_values=exact_values,
            capacities=capacities,
        )
        material_refs = tuple(str(item) for item in representative["material_refs"])  # type: ignore[arg-type]
        domain_facts: dict[str, object] = {
            "reconfiguration_domain_assignments": tuple(
                tuple(zip(material_refs, assignment))
                for assignment in domain.assignments
            ),
            "reconfiguration_domain_complete": domain.complete,
            "reconfiguration_domain_feasible": bool(domain.assignments),
            "reconfiguration_domain_assignment_count": len(domain.assignments),
            "reconfiguration_domain_examined_assignment_count": (
                domain.examined_assignment_count
            ),
            "reconfiguration_domain_enumeration_limit": (
                _HARD_MAX_QUANTITY_ASSIGNMENTS
            ),
            "reconfiguration_domain_free_residual_refs": tuple(
                material_refs[index] for index in domain.free_indices
            ),
            "reconfiguration_domain_retains_free_residual_quantities": bool(
                domain.free_indices
            ),
            "reconfiguration_domain_conserved_total": sum(quantities),
            "reconfiguration_domain_supplied_constraint_count": len(exact_values),
        }
        promoted.extend(
            {
                **member,
                "capacities": capacities,
                "capacity_measurement_available": bool(capacities),
                **domain_facts,
            }
            for member in members
        )
        explicit_total = sum(
            int(item["target_quantity"])
            for item in (display_constraints_by_index[__yf_order_key] for __yf_order_key in sorted(display_constraints_by_index))
        )
        if (
            contradictory
            or not display_constraints_by_index
            or explicit_total != sum(quantities)
        ):
            continue
        target = tuple(
            int(display_constraints_by_index[index]["target_quantity"])
            if index in display_constraints_by_index
            else 0
            for index in range(len(quantities))
        )
        if capacities and any(
            target[index] > capacities[index] for index in range(len(target))
        ):
            continue
        first_constraint = display_constraints_by_index[
            min(display_constraints_by_index)
        ]
        promoted.append(
            {
                **representative,
                "display_constraint_index": int(
                    first_constraint["material_index"]
                ),
                "display_constraint_extent_ref": str(
                    first_constraint["material_ref"]
                ),
                "display_constraint_quantity": int(
                    first_constraint["target_quantity"]
                ),
                "moving_marker_ref": str(first_constraint["moving_marker_ref"]),
                "fixed_marker_ref": str(first_constraint["fixed_marker_ref"]),
                "configuration_kind": "complete_display_assignment",
                "displayed_quantity_vector": tuple(
                    (
                        str(tuple(representative["material_refs"])[index]),
                        value,
                    )
                    for index, value in enumerate(target)
                ),
                "target_quantity_values": target,
                "capacities": capacities,
                "capacity_measurement_available": bool(capacities),
                **domain_facts,
                "displayed_marker_bindings": tuple(
                    FrozenMap(item)
                    for _index, item in sorted(
                        display_constraints_by_index.items()
                    )
                ),
            }
        )
    return tuple(promoted)


def _multi_identity_opposed_route_descriptions(
    *,
    descriptions: tuple[dict[str, object], ...],
    components: tuple[ComponentDescription, ...],
    tracked_by_component: dict[str, str],
    route_policy: FrozenMap,
) -> tuple[dict[str, object], ...]:
    """Project exact anonymous chain morphology into bounded descriptors.

    Adjacency and invariant offset remain measurements here. They do not prove
    a composite identity; DRM keeps that link revisable until non-zero
    action-conditioned co-motion is observed.
    """

    by_ref = {item.component_id: item for item in components}
    complete_structure_keys = {
        (
            item.get("axis"),
            item.get("origin_side"),
            item.get("origin"),
            item.get("quantum"),
            item.get("material_refs"),
            item.get("controls"),
        )
        for item in descriptions
        if item.get("configuration_kind") == "complete_display_assignment"
    }
    outputs: list[dict[str, object]] = []
    for description in descriptions:
        configuration_kind = description.get("configuration_kind")
        if configuration_kind not in {None, "complete_display_assignment"}:
            continue
        structure_key = (
            description.get("axis"),
            description.get("origin_side"),
            description.get("origin"),
            description.get("quantum"),
            description.get("material_refs"),
            description.get("controls"),
        )
        if (
            configuration_kind != "complete_display_assignment"
            and structure_key in complete_structure_keys
        ):
            continue
        material_refs = tuple(str(item) for item in description["material_refs"])  # type: ignore[arg-type]
        if not 2 <= len(material_refs) <= 4:
            continue
        capacities = tuple(int(item) for item in description.get("capacities") or ())
        terminal_quantities = tuple(
            int(item) for item in description.get("target_quantity_values") or ()
        )
        quantities = tuple(int(item) for item in description["quantities"])  # type: ignore[arg-type]
        directed_controls = {
            (int(source), int(destination))
            for source, destination, _component_ref in description["controls"]  # type: ignore[union-attr]
        }
        if (
            len(capacities) != len(material_refs)
            or any(value > capacity for value, capacity in zip(quantities, capacities))
            or (
                terminal_quantities
                and len(terminal_quantities) != len(material_refs)
            )
            or any(
                (index, index + 1) not in directed_controls
                or (index + 1, index) not in directed_controls
                for index in range(len(material_refs) - 1)
            )
        ):
            continue
        chain = tuple(by_ref[item] for item in material_refs)
        axis = str(description["axis"])
        origin_side = str(description["origin_side"])
        quantum = int(description["quantum"])
        zone_bands = _zone_partition_bands(chain=chain, axis=axis)
        composite_candidates = _opposed_composite_measurements(
            components=components,
            chain=chain,
            axis=axis,
            zone_bands=zone_bands,
            quantum=quantum,
            excluded_component_refs=frozenset(
                (
                    *material_refs,
                    *(
                        str(item[2])
                        for item in description["controls"]  # type: ignore[union-attr]
                    ),
                )
            ),
            tracked_by_component=tracked_by_component,
        )
        if composite_candidates is None:
            continue
        identities, carrier_component_refs, target_component_refs = composite_candidates
        terminal_quantity_requirements: tuple[tuple[int, int], ...] = ()
        terminal_constraint_kind = "exact_full_quantity_vector"
        if not terminal_quantities:
            measured_requirements: list[tuple[int, int]] = []
            for identity, target_component_ref in zip(
                identities, target_component_refs
            ):
                target_extent = _target_extent(
                    by_ref[target_component_ref],
                    axis=axis,
                    origin_side=origin_side,
                    origin=int(description["origin"]),
                )
                if target_extent <= 0 or target_extent % quantum:
                    measured_requirements = []
                    break
                measured_requirements.append(
                    (identity.target_zone_index, target_extent // quantum)
                )
            terminal_quantity_requirements = tuple(
                sorted(measured_requirements)
            )
            if (
                len(terminal_quantity_requirements) != len(identities)
                or len({index for index, _value in terminal_quantity_requirements})
                != len(identities)
            ):
                continue
            terminal_constraint_kind = (
                "exact_observed_supports_with_conserved_capacity_safe_residual"
            )
        separator_groups = _separator_morphology_groups(
            components=components,
            chain=chain,
            edge_indices=tuple(range(len(chain) - 1)),
            axis=axis,
            origin_side=origin_side,
            origin=int(description["origin"]),
            quantum=quantum,
            material_value=int(description["material_value"]),
            marker_value=by_ref[carrier_component_refs[0]].value,
        )
        for group in separator_groups:
            doors = tuple(
                sorted(
                    tuple(group["doors"]),  # type: ignore[arg-type]
                    key=lambda item: int(item[0]),
                )
            )
            if len(doors) != len(chain) - 1:
                continue
            interfaces = tuple(
                QuantizedInterfaceMeasurement(
                    interface_ref=str(component_ref),
                    lower_index=int(edge),
                    upper_index=int(edge) + 1,
                    activation_threshold=int(threshold),
                )
                for edge, component_ref, threshold in doors
            )
            descriptor = MultiIdentityOpposedRoutesInput(
                material_refs=material_refs,
                initial_quantities=quantities,
                capacities=capacities,
                terminal_quantities=terminal_quantities,
                terminal_quantity_requirements=terminal_quantity_requirements,
                controls=tuple(
                    QuantizedTransferControlMeasurement(
                        source_index=int(source),
                        destination_index=int(destination),
                        component_ref=str(component_ref),
                    )
                    for source, destination, component_ref in description["controls"]  # type: ignore[union-attr]
                ),
                interfaces=interfaces,
                identities=identities,
                control_order_variant="canonical",
                max_witnesses=min(
                    3,
                    _bounded_policy_integer(
                        route_policy,
                        "max_route_variants",
                        default=_HARD_MAX_ROUTE_VARIANTS,
                        hard_maximum=_HARD_MAX_ROUTE_VARIANTS,
                    ),
                ),
                max_expanded_states_per_stage=_bounded_policy_integer(
                    route_policy,
                    "max_expanded_states_per_hypothesis",
                    default=_HARD_MAX_EXPANDED_STATES,
                    hard_maximum=_HARD_MAX_EXPANDED_STATES,
                ),
                max_actions_per_witness=_bounded_policy_integer(
                    route_policy,
                    "max_plan_actions",
                    default=_HARD_MAX_PLAN_ACTIONS,
                    hard_maximum=_HARD_MAX_PLAN_ACTIONS,
                ),
            )
            outputs.append(
                {
                    **description,
                    "configuration_kind": "multi_identity_opposed_routes",
                    "multi_identity_descriptor": descriptor,
                    "zone_partition_bands": tuple(
                        (material_refs[index], start, end)
                        for index, (start, end) in enumerate(zone_bands)
                    ),
                    "moving_marker_ref": carrier_component_refs[0],
                    "fixed_marker_ref": target_component_refs[0],
                    "moving_marker_signature": (
                        by_ref[carrier_component_refs[0]].value,
                        by_ref[carrier_component_refs[0]].relative_pixels,
                    ),
                    "identity_carrier_component_refs": carrier_component_refs,
                    "identity_carrier_signatures": tuple(
                        (
                            identity.identity_ref,
                            by_ref[carrier_component_ref].value,
                            by_ref[carrier_component_ref].relative_pixels,
                        )
                        for identity, carrier_component_ref in zip(
                            identities, carrier_component_refs
                        )
                    ),
                    "identity_target_component_refs": target_component_refs,
                    "terminal_constraint_kind": terminal_constraint_kind,
                    "terminal_quantity_requirements": (
                        terminal_quantity_requirements
                    ),
                    "terminal_conserved_aggregate": sum(quantities),
                    "terminal_unknown_zone_indices": tuple(
                        index
                        for index in range(len(material_refs))
                        if terminal_quantity_requirements
                        and index
                        not in {
                            requirement_index
                            for requirement_index, _value in (
                                terminal_quantity_requirements
                            )
                        }
                    ),
                    "displayed_quantity_vector": (
                        description.get("displayed_quantity_vector")
                        or tuple(
                            (material_refs[index], quantity)
                            for index, quantity in terminal_quantity_requirements
                        )
                    ),
                    "separator_transition_candidates": tuple(
                        (
                            item.lower_index,
                            item.upper_index,
                            item.interface_ref,
                            item.activation_threshold,
                        )
                        for item in interfaces
                    ),
                    "topology_gap_evidence": False,
                }
            )
    unique: dict[str, dict[str, object]] = {}
    for output in outputs:
        descriptor = output["multi_identity_descriptor"]
        if isinstance(descriptor, MultiIdentityOpposedRoutesInput):
            unique.setdefault(descriptor.model_dump_json(), output)
    return tuple(unique.values())


def _opposed_composite_measurements(
    *,
    components: tuple[ComponentDescription, ...],
    chain: tuple[ComponentDescription, ...],
    axis: str,
    zone_bands: tuple[tuple[int, int], ...],
    quantum: int,
    excluded_component_refs: frozenset[str],
    tracked_by_component: dict[str, str],
) -> tuple[
    tuple[CompositeIdentityRouteMeasurement, ...],
    tuple[str, ...],
    tuple[str, ...],
] | None:
    morphology_groups: dict[tuple[object, ...], list[ComponentDescription]] = defaultdict(list)
    for component in components:
        if component.component_id not in excluded_component_refs:
            morphology_groups[_component_morphology(component)].append(component)
    endpoint = len(chain) - 1
    candidates: list[
        tuple[
            tuple[CompositeIdentityRouteMeasurement, ...],
            tuple[str, ...],
            tuple[str, ...],
        ]
    ] = []
    for carriers in (morphology_groups[__yf_order_key] for __yf_order_key in sorted(morphology_groups)):
        if len(carriers) != 2:
            continue
        ordered = tuple(
            sorted(
                carriers,
                key=lambda item: (
                    _component_zone_index(
                        component=item,
                        axis=axis,
                        zone_bands=zone_bands,
                    )
                    if _component_zone_index(
                        component=item,
                        axis=axis,
                        zone_bands=zone_bands,
                    )
                    is not None
                    else len(chain)
                ),
            )
        )
        zones = tuple(
            _component_zone_index(
                component=item,
                axis=axis,
                zone_bands=zone_bands,
            )
            for item in ordered
        )
        if zones != (0, endpoint):
            continue
        identity_items: list[CompositeIdentityRouteMeasurement] = []
        target_refs: list[str] = []
        attribute_morphologies: list[tuple[object, ...]] = []
        valid = True
        for carrier, current_zone, target_zone in (
            (ordered[0], 0, endpoint),
            (ordered[1], endpoint, 0),
        ):
            carrier_entity_ref = tracked_by_component.get(carrier.component_id)
            if carrier_entity_ref is None:
                valid = False
                break
            attribute_matches: list[tuple[ComponentDescription, ComponentDescription]] = []
            for attribute in components:
                if (
                    attribute in carriers
                    or attribute.component_id in excluded_component_refs
                    or not _components_are_locally_adjacent(
                        carrier, attribute, quantum
                    )
                    or _component_zone_index(
                        component=attribute,
                        axis=axis,
                        zone_bands=zone_bands,
                    )
                    != current_zone
                ):
                    continue
                # A displayed requirement can intentionally encode a different
                # extent from the carried attribute.  Exact morphology would
                # therefore erase the relation we need to test.  A unique
                # palette-equal detached counterpart is only a revisable
                # structural candidate here; it never establishes a role or
                # composite identity without later action-conditioned motion.
                targets = tuple(
                    item
                    for item in components
                    if item is not attribute
                    and item not in carriers
                    and item.component_id not in excluded_component_refs
                    and item.value == attribute.value
                    and not _components_are_locally_adjacent(
                        carrier, item, quantum
                    )
                )
                if len(targets) == 1:
                    attribute_matches.append((attribute, targets[0]))
            if len(attribute_matches) != 1:
                valid = False
                break
            attribute, target = attribute_matches[0]
            attribute_entity_ref = tracked_by_component.get(attribute.component_id)
            if attribute_entity_ref is None:
                valid = False
                break
            attribute_morphologies.append(_component_morphology(attribute))
            identity_items.append(
                CompositeIdentityRouteMeasurement(
                    identity_ref=carrier_entity_ref,
                    carrier_entity_ref=carrier_entity_ref,
                    attribute_entity_ref=attribute_entity_ref,
                    current_zone_index=current_zone,
                    target_zone_index=target_zone,
                    attribute_offset=(
                        attribute.bbox.top - carrier.bbox.top,
                        attribute.bbox.left - carrier.bbox.left,
                    ),
                )
            )
            target_refs.append(target.component_id)
        # The two repeated body candidates are homologous only when their
        # attached accents occupy the same relative morphology slot.  This is
        # a static pairing disambiguator, not causal identity evidence: each
        # pair's own offset still has to survive non-zero co-motion later.
        relative_slots = tuple(item.attribute_offset for item in identity_items)
        if (
            valid
            and len(set(attribute_morphologies)) == 2
            and len(relative_slots) == 2
            and len(set(relative_slots)) == 1
        ):
            candidates.append(
                (
                    tuple(identity_items),
                    tuple(item.component_id for item in ordered),
                    tuple(target_refs),
                )
            )
    return candidates[0] if len(candidates) == 1 else None


def _component_morphology(component: ComponentDescription) -> tuple[object, ...]:
    return (
        component.value,
        component.bbox.height,
        component.bbox.width,
        component.relative_pixels,
    )


def _components_are_locally_adjacent(
    left: ComponentDescription,
    right: ComponentDescription,
    quantum: int,
) -> bool:
    row_gap = max(
        left.bbox.top - right.bbox.bottom - 1,
        right.bbox.top - left.bbox.bottom - 1,
        0,
    )
    column_gap = max(
        left.bbox.left - right.bbox.right - 1,
        right.bbox.left - left.bbox.right - 1,
        0,
    )
    return row_gap <= quantum and column_gap <= quantum and (row_gap + column_gap) <= quantum


def _multi_identity_opposed_routes(
    description: dict[str, object],
    *,
    max_expanded_states: int,
    max_plan_actions: int,
    control_order_variant: str,
) -> tuple[dict[str, object], ...]:
    from agents.yf_arc3_v5.capabilities.opposed_routes import (
        simulate_multi_identity_opposed_routes,
    )

    descriptor = description.get("multi_identity_descriptor")
    if not isinstance(descriptor, MultiIdentityOpposedRoutesInput):
        return ()
    bounded_descriptor = descriptor.model_copy(
        update={
            "control_order_variant": control_order_variant,
            "max_expanded_states_per_stage": min(
                max_expanded_states, _HARD_MAX_EXPANDED_STATES
            ),
            "max_actions_per_witness": min(max_plan_actions, _HARD_MAX_PLAN_ACTIONS),
        }
    )
    analysis = simulate_multi_identity_opposed_routes(bounded_descriptor)
    material_refs = bounded_descriptor.material_refs
    routes: list[dict[str, object]] = []
    for witness in analysis.witnesses:
        route_steps = tuple(
            (
                material_refs[item.source_index],
                material_refs[item.destination_index],
                item.component_ref,
                (
                    "multi_identity_interface_transition_candidate"
                    if item.operation_kind == "interface_transition"
                    else "extent_exchange_candidate"
                ),
            )
            for item in witness.steps
        )
        quantity_states = tuple(
            tuple(zip(material_refs, item.quantities_after)) for item in witness.steps
        )
        identity_states = tuple(
            tuple(
                (identity_ref, material_refs[index])
                for identity_ref, index in item.identity_zone_indices_after
            )
            for item in witness.steps
        )
        moved_identity_states = tuple(
            item.moved_identity_refs for item in witness.steps
        )
        component_sequence = tuple(str(item[2]) for item in route_steps)
        phase_count = 1 + sum(
            left != right
            for left, right in zip(component_sequence, component_sequence[1:])
        )
        routes.append(
            {
                "route_ref": witness.witness_ref,
                "route_component_refs": component_sequence,
                "route_transfer_steps": route_steps,
                "route_quantity_states": quantity_states,
                "route_marker_states": (),
                "route_identity_states": identity_states,
                "route_moved_identity_refs": moved_identity_states,
                "first_component_ref": component_sequence[0],
                "primitive_action_count": witness.primitive_action_count,
                "transfer_phase_count": phase_count,
                "actuator_switch_count": phase_count - 1,
                "expected_quantities_after_action": quantity_states[0],
                "expected_identity_zones_after_action": identity_states[0],
                "expected_moved_identity_refs_after_action": moved_identity_states[0],
                "expanded_state_count": witness.expanded_state_count,
                "maximum_remaining_route_distance_after_crossing": (
                    witness.maximum_remaining_route_distance_after_crossing
                ),
                "remaining_route_distance_imbalance_after_crossing": (
                    witness.remaining_route_distance_imbalance_after_crossing
                ),
                "atomic_crossing_identity_count": witness.atomic_crossing_identity_count,
                "all_identity_transports_before_terminal_alignment": (
                    witness.all_identity_transports_before_terminal_alignment
                ),
            }
        )
    return tuple(routes)


def _multi_identity_route_facts(
    representative: dict[str, object],
) -> dict[str, object]:
    if representative.get("configuration_kind") != "multi_identity_opposed_routes":
        return {}
    descriptor = representative.get("multi_identity_descriptor")
    if not isinstance(descriptor, MultiIdentityOpposedRoutesInput):
        return {}
    steps = tuple(representative["route_transfer_steps"])  # type: ignore[arg-type]
    identity_states = tuple(representative["route_identity_states"])  # type: ignore[arg-type]
    moved_identity_states = tuple(representative["route_moved_identity_refs"])  # type: ignore[arg-type]
    first_step = tuple(steps[0])
    composite_bindings = tuple(
        (
            item.identity_ref,
            item.carrier_entity_ref,
            item.attribute_entity_ref,
            item.attribute_offset[0],
            item.attribute_offset[1],
        )
        for item in descriptor.identities
    )
    return {
        "descriptor_kind": "multi_identity_opposed_routes",
        "current_step_measurement_kind": str(first_step[3]),
        "route_identity_states": identity_states,
        "route_moved_identity_refs": moved_identity_states,
        "expected_identity_zones_after_action": identity_states[0],
        "expected_moved_identity_refs_after_action": moved_identity_states[0],
        "identity_composite_bindings": composite_bindings,
        "identity_carrier_signatures": representative[
            "identity_carrier_signatures"
        ],
        "composite_link_measurements": tuple(
            (
                item.identity_ref,
                "adjacent",
                item.attribute_offset,
                False,
            )
            for item in descriptor.identities
        ),
        "terminal_constraint_kind": representative["terminal_constraint_kind"],
        "terminal_quantity_requirements": representative[
            "terminal_quantity_requirements"
        ],
        "terminal_conserved_aggregate": representative[
            "terminal_conserved_aggregate"
        ],
        "terminal_unknown_zone_indices": representative[
            "terminal_unknown_zone_indices"
        ],
        "terminal_witness_quantities": tuple(
            representative["route_quantity_states"][-1]  # type: ignore[index]
        ),
        "zone_partition_bands": representative["zone_partition_bands"],
        "maximum_remaining_route_distance_after_crossing": representative[
            "maximum_remaining_route_distance_after_crossing"
        ],
        "remaining_route_distance_imbalance_after_crossing": representative[
            "remaining_route_distance_imbalance_after_crossing"
        ],
        "atomic_crossing_identity_count": representative[
            "atomic_crossing_identity_count"
        ],
        "all_identity_transports_before_terminal_alignment": representative[
            "all_identity_transports_before_terminal_alignment"
        ],
    }


def _infer_capacity_map(
    *,
    description: dict[str, object],
    components: tuple[ComponentDescription, ...],
) -> tuple[int, ...]:
    """Propose capacities from continuous limiting strips between chain nodes."""

    axis = str(description["axis"])
    origin_side = str(description["origin_side"])
    quantum = int(description["quantum"])
    by_ref = {item.component_id: item for item in components}
    chain = tuple(by_ref[str(item)] for item in description["material_refs"])  # type: ignore[arg-type]
    origin = _axis_origin(chain[0], axis, origin_side)
    boundary_capacities: list[int] = []
    for left, right in zip(chain, chain[1:]):
        bands: dict[tuple[int, int], set[int]] = defaultdict(set)
        for component in components:
            if (
                _orthogonal_extent(component, axis) != quantum
                or _orthogonal_start(component, axis) <= _orthogonal_end(left, axis)
                or _orthogonal_end(component, axis) >= _orthogonal_start(right, axis)
            ):
                continue
            band = (
                _orthogonal_start(component, axis),
                _orthogonal_end(component, axis),
            )
            bands[band].update(_axis_positions(component, axis))
        candidates: list[int] = []
        for positions in (bands[__yf_order_key] for __yf_order_key in sorted(bands)):
            if not positions or origin not in positions:
                continue
            boundary_start = min(positions) if origin_side == "maximum" else max(positions)
            expected = set(range(min(boundary_start, origin), max(boundary_start, origin) + 1))
            if positions == expected and len(positions) % quantum == 0:
                capacity = len(positions) // quantum
                if capacity > 1:
                    candidates.append(capacity)
        if len(candidates) != 1:
            return ()
        boundary_capacities.append(candidates[0])
    if len(boundary_capacities) != len(chain) - 1:
        return ()
    return (
        boundary_capacities[0],
        *(
            min(left, right)
            for left, right in zip(boundary_capacities, boundary_capacities[1:])
        ),
        boundary_capacities[-1],
    )


def _orthogonal_extent(component: ComponentDescription, axis: str) -> int:
    return component.bbox.height if axis == "column" else component.bbox.width


def _axis_positions(component: ComponentDescription, axis: str) -> tuple[int, ...]:
    return tuple(
        range(component.bbox.left, component.bbox.right + 1)
        if axis == "column"
        else range(component.bbox.top, component.bbox.bottom + 1)
    )


def _material_chain_bands(
    *,
    description: dict[str, object],
    components: tuple[ComponentDescription, ...],
) -> tuple[tuple[str, int, int], ...]:
    """Build stable zone envelopes from matter plus controls entering each zone."""

    axis = str(description["axis"])
    material_refs = tuple(str(item) for item in description["material_refs"])  # type: ignore[arg-type]
    by_ref = {item.component_id: item for item in components}
    bounds = [
        [
            _orthogonal_start(by_ref[material_ref], axis),
            _orthogonal_end(by_ref[material_ref], axis),
        ]
        for material_ref in material_refs
    ]
    for _source, raw_target, raw_component_ref in description["controls"]:  # type: ignore[union-attr]
        target = int(raw_target)
        control = by_ref[str(raw_component_ref)]
        bounds[target][0] = min(bounds[target][0], _orthogonal_start(control, axis))
        bounds[target][1] = max(bounds[target][1], _orthogonal_end(control, axis))
    return tuple(
        (material_ref, bounds[index][0], bounds[index][1])
        for index, material_ref in enumerate(material_refs)
    )


def advance_committed_quantized_plan(
    value: CommittedQuantizedPlanAdvanceInput,
) -> InteractionProbeAgenda:
    """Advance one already simulated route without reconstructing empty zones.

    `route_transfer_steps` and `route_quantity_states` are committed before the
    first click.  Reconciliation proves the first predicted state before this
    cursor is advanced.  A visually empty material zone may therefore lose its
    raw component without erasing the still-valid symbolic route.
    """

    if POINT_ACTION_REF not in value.available_action_refs:
        return _empty_agenda()
    facts = value.committed_plan_facts
    raw_steps = tuple(facts.get("route_transfer_steps") or ())
    raw_states = tuple(facts.get("route_quantity_states") or ())
    raw_marker_states = tuple(facts.get("route_marker_states") or ())
    raw_identity_states = tuple(facts.get("route_identity_states") or ())
    raw_moved_identity_refs = tuple(facts.get("route_moved_identity_refs") or ())
    if len(raw_steps) < 2 or len(raw_states) != len(raw_steps):
        return _empty_agenda()
    if raw_identity_states and len(raw_identity_states) != len(raw_steps):
        return _empty_agenda()
    if raw_moved_identity_refs and len(raw_moved_identity_refs) != len(raw_steps):
        return _empty_agenda()
    remaining_steps = raw_steps[1:]
    remaining_states = raw_states[1:]
    remaining_marker_states = raw_marker_states[1:] if raw_marker_states else ()
    remaining_identity_states = raw_identity_states[1:] if raw_identity_states else ()
    remaining_moved_identity_refs = (
        raw_moved_identity_refs[1:] if raw_moved_identity_refs else ()
    )
    next_step = tuple(remaining_steps[0])
    if len(next_step) not in {3, 4}:
        return _empty_agenda()
    planned_component_ref = str(next_step[2])
    route_entity_by_component = {
        str(component_ref): str(entity_ref)
        for component_ref, entity_ref in (
            facts.get("route_component_entity_bindings") or ()
        )
    }
    planned_entity_ref = route_entity_by_component.get(planned_component_ref)
    action_points = {
        str(component_ref): (int(x), int(y))
        for component_ref, x, y in tuple(facts.get("route_action_points") or ())
    }
    point = action_points.get(planned_component_ref)
    if point is None:
        return _empty_agenda()
    component_ref = planned_component_ref
    candidate_ref = f"probe:{component_ref}"
    component_sequence = tuple(str(item[2]) for item in remaining_steps)
    phase_count = 1 + sum(
        left != right
        for left, right in zip(component_sequence, component_sequence[1:])
    )
    prior_executed = tuple(str(item) for item in facts.get("executed_component_refs") or ())
    just_executed = str(raw_steps[0][2])
    executed = tuple(dict.fromkeys((*prior_executed, just_executed)))
    completed_click_count = int(facts.get("executed_click_count") or 0) + 1
    prior_was_checkpoint = bool(facts.get("checkpoint_after_current_step"))
    last_full_checkpoint_click_count = (
        completed_click_count
        if prior_was_checkpoint
        else int(facts.get("last_full_checkpoint_click_count") or 0)
    )
    checkpoint_interval = max(1, int(facts.get("checkpoint_interval") or 1))
    checkpoint_reason = _checkpoint_reason(
        steps=remaining_steps,
        executed_component_refs=executed,
        completed_click_count=completed_click_count,
        last_full_checkpoint_click_count=last_full_checkpoint_click_count,
        interval=checkpoint_interval,
        enabled_conditions=tuple(
            str(item)
            for item in value.continuation_policy.get("checkpoint_on", ())
        ),
    )
    continued_delta = FrozenMap(
        {
            "candidate_ref": candidate_ref,
            "component_ref": component_ref,
            "planned_component_ref": planned_component_ref,
            "tracked_entity_ref": (
                planned_entity_ref
            ),
            "route_transfer_steps": remaining_steps,
            "route_quantity_states": remaining_states,
            "route_marker_states": remaining_marker_states,
            "route_component_refs": component_sequence,
            "executed_component_refs": executed,
            "current_quantities": raw_states[0],
            "expected_quantities_after_action": remaining_states[0],
            **(
                {
                    "route_identity_states": remaining_identity_states,
                    "route_moved_identity_refs": remaining_moved_identity_refs,
                    "expected_identity_zones_after_action": remaining_identity_states[0],
                    "expected_moved_identity_refs_after_action": (
                        remaining_moved_identity_refs[0]
                    ),
                    "current_step_measurement_kind": str(next_step[3]),
                }
                if remaining_identity_states and len(next_step) == 4
                else {}
            ),
            **(
                {
                    "current_marker_zone": raw_marker_states[0],
                    "expected_marker_zone_after_action": remaining_marker_states[0],
                    "current_step_measurement_kind": str(next_step[3]),
                }
                if remaining_marker_states and len(next_step) == 4
                else {}
            ),
            "primitive_action_count": len(remaining_steps),
            "transfer_phase_count": phase_count,
            "actuator_switch_count": phase_count - 1,
            "remaining_click_count": len(remaining_steps),
            "remaining_step_count": len(remaining_steps),
            "first_use_verifies_current_binding": component_ref not in executed,
            "executed_click_count": completed_click_count,
            "current_step_index": completed_click_count + 1,
            "last_full_checkpoint_click_count": last_full_checkpoint_click_count,
            "checkpoint_interval": checkpoint_interval,
            "checkpoint_after_current_step": bool(checkpoint_reason),
            "checkpoint_reason_after_current_step": checkpoint_reason,
            "prior_step_verification": (
                "full_checkpoint"
                if prior_was_checkpoint
                else "intermediate_scene_change_guard"
            ),
        }
    )
    candidate = InteractionProbeCandidate(
        candidate_ref=candidate_ref,
        action_ref=POINT_ACTION_REF,
        action_data=FrozenMap(
            {
                "x": point[0],
                "y": point[1],
            }
        ),
        component_ref=component_ref,
        point=(point[1], point[0]),
    )
    return InteractionProbeAgenda(
        candidates=(candidate,),
        alternative_refs=(candidate_ref,),
        alternative_facts=FrozenMap({candidate_ref: continued_delta}),
        context_facts=facts,
        has_candidates=True,
    )


def _checkpoint_reason(
    *,
    steps: tuple[object, ...],
    executed_component_refs: tuple[str, ...],
    completed_click_count: int,
    last_full_checkpoint_click_count: int,
    interval: int,
    enabled_conditions: tuple[str, ...],
) -> str:
    """Return the first exact declared checkpoint condition for the next step."""

    if not steps:
        return ""
    enabled = frozenset(enabled_conditions)
    current = tuple(steps[0])
    if len(current) not in {3, 4}:
        return ""
    component_ref = str(current[2])
    next_component_ref = (
        str(tuple(steps[1])[2])
        if len(steps) > 1 and len(tuple(steps[1])) in {3, 4}
        else ""
    )
    if (
        len(current) == 4
        and current[3]
        in {
            "separator_zone_transition_candidate",
            "multi_identity_interface_transition_candidate",
        }
        and "mechanism_activation" in enabled
    ):
        return "mechanism_activation"
    if "terminal_step" in enabled and len(steps) == 1:
        return "terminal_step"
    if (
        "first_use_current_binding" in enabled
        and component_ref not in executed_component_refs
    ):
        return "first_use_current_binding"
    if (
        "transfer_phase_boundary" in enabled
        and next_component_ref
        and component_ref != next_component_ref
    ):
        return "transfer_phase_boundary"
    if (
        completed_click_count + 1 - last_full_checkpoint_click_count >= interval
    ):
        return "regular_interval"
    return ""


def _quantized_chain_descriptions(
    *,
    components: tuple[ComponentDescription, ...],
    axis: str,
    quantum: int,
) -> tuple[dict[str, object], ...]:
    """Enumerate neutral complete-structure descriptions for either local axis."""

    by_value: dict[int, list[ComponentDescription]] = defaultdict(list)
    for component in components:
        by_value[component.value].append(component)
    descriptions: list[dict[str, object]] = []
    for material_value, same_value in sorted(by_value.items()):
        for origin_side in ("minimum", "maximum"):
            material = tuple(
                item
                for item in same_value
                if _axis_extent(item, axis) % quantum == 0
                and _axis_extent(item, axis) >= quantum
                and _orthogonal_extent(item, axis) > quantum
            )
            if len(material) < 2:
                continue
            origins = Counter(_axis_origin(item, axis, origin_side) for item in material)
            for origin, count in sorted(origins.items()):
                if count < 2:
                    continue
                chain = tuple(
                    sorted(
                        (
                            item
                            for item in material
                            if _axis_origin(item, axis, origin_side) == origin
                        ),
                        key=lambda item: _orthogonal_start(item, axis),
                    )
                )
                if len(chain) < 2 or any(
                    _orthogonal_end(left, axis) >= _orthogonal_start(right, axis)
                    for left, right in zip(chain, chain[1:])
                ):
                    continue
                controls = _chain_controls(
                    components=components,
                    chain=chain,
                    axis=axis,
                    origin_side=origin_side,
                    quantum=quantum,
                )
                if len(controls) < 2 * (len(chain) - 1):
                    continue
                for marker_value, marker_components in sorted(by_value.items()):
                    if marker_value == material_value or len(marker_components) < 2:
                        continue
                    for display_index, displayed_material in enumerate(chain):
                        current_extent = _axis_extent(displayed_material, axis)
                        surface = _surface_coordinate(
                            displayed_material, axis, origin_side
                        )
                        moving_markers = tuple(
                            item
                            for item in marker_components
                            if _marker_attached_to_surface(
                                item,
                                displayed_material,
                                axis=axis,
                                origin_side=origin_side,
                                surface=surface,
                            )
                        )
                        if not moving_markers:
                            continue
                        for moving in moving_markers:
                            for fixed in marker_components:
                                if fixed.component_id == moving.component_id:
                                    continue
                                target_extent = _target_extent(
                                    fixed,
                                    axis=axis,
                                    origin_side=origin_side,
                                    origin=origin,
                                )
                                if (
                                    target_extent == current_extent
                                    or target_extent % quantum != 0
                                    or target_extent > sum(
                                        _axis_extent(item, axis) for item in chain
                                    )
                                ):
                                    continue
                                quantities = tuple(
                                    _axis_extent(item, axis) // quantum
                                    for item in chain
                                )
                                descriptions.append(
                                    {
                                        "axis": axis,
                                        "origin_side": origin_side,
                                        "origin": origin,
                                        "quantum": quantum,
                                        "material_value": material_value,
                                        "material_refs": tuple(
                                            item.component_id for item in chain
                                        ),
                                        "current_quantities": tuple(
                                            (
                                                item.component_id,
                                                quantities[index],
                                            )
                                            for index, item in enumerate(chain)
                                        ),
                                        "quantities": quantities,
                                        "controls": controls,
                                        "display_constraint_index": display_index,
                                        "display_constraint_extent_ref": displayed_material.component_id,
                                        "display_constraint_quantity": target_extent
                                        // quantum,
                                        "moving_marker_ref": moving.component_id,
                                        "fixed_marker_ref": fixed.component_id,
                                    }
                                )
    # Exact duplicate descriptions can arise from irrelevant additional members
    # of one palette group.  Content-addressing keeps the enumeration bounded.
    unique: dict[str, dict[str, object]] = {}
    for item in descriptions:
        key = canonical_json(
            (
                item["axis"],
                item["origin_side"],
                item["material_refs"],
                item["controls"],
                item["display_constraint_index"],
                item["display_constraint_quantity"],
                item["moving_marker_ref"],
                item["fixed_marker_ref"],
            )
        )
        unique[key] = item
    return tuple(unique.values())


def _chain_controls(
    *,
    components: tuple[ComponentDescription, ...],
    chain: tuple[ComponentDescription, ...],
    axis: str,
    origin_side: str,
    quantum: int,
) -> tuple[tuple[int, int, str], ...]:
    squares = tuple(
        item
        for item in components
        if item.area == quantum * quantum
        and item.bbox.height == quantum
        and item.bbox.width == quantum
    )
    controls: list[tuple[int, int, str]] = []
    for index, (before, after) in enumerate(zip(chain, chain[1:])):
        between_controls = tuple(
            item
            for item in squares
            if _orthogonal_start(item, axis)
            >= _orthogonal_end(before, axis) - quantum + 1
            and _orthogonal_end(item, axis)
            <= _orthogonal_start(after, axis) + quantum - 1
            and (
                _control_on_material_origin(item, before, axis, origin_side)
                or _control_on_material_origin(item, after, axis, origin_side)
            )
        )
        if len(between_controls) != 2:
            continue
        before_control, after_control = sorted(
            between_controls,
            key=lambda item: _orthogonal_start(item, axis),
        )
        # A boundary-side control transfers material into the zone on its own
        # side: the opposite zone is the source.
        controls.append((index + 1, index, before_control.component_id))
        controls.append((index, index + 1, after_control.component_id))
    return tuple(controls)


def _folded_separator_descriptions(
    *,
    components: tuple[ComponentDescription, ...],
    axis: str,
    quantum: int,
    tracked_by_component: dict[str, str],
) -> tuple[dict[str, object], ...]:
    """Measure non-collinear supports joined by paired controls and interfaces.

    The output contains only geometry, graph incidence, local distances and
    exact finite paths.  DRM interprets those measurements and SRC decides
    whether one counterfactual witness may be committed.
    """

    by_value: dict[int, list[ComponentDescription]] = defaultdict(list)
    for component in components:
        by_value[component.value].append(component)
    outputs: list[dict[str, object]] = []
    for material_value, same_value in sorted(by_value.items()):
        for origin_side in ("minimum", "maximum"):
            material = tuple(
                sorted(
                    (
                        item
                        for item in same_value
                        if _axis_extent(item, axis) % quantum == 0
                        and _axis_extent(item, axis) >= quantum
                        and _orthogonal_extent(item, axis) > quantum
                    ),
                    key=lambda item: (
                        _orthogonal_start(item, axis),
                        _axis_origin(item, axis, origin_side),
                        item.component_id,
                    ),
                )
            )
            if len(material) < 3:
                continue
            control_sets = _folded_control_sets(
                components=components,
                material=material,
                material_value=material_value,
                axis=axis,
                origin_side=origin_side,
                quantum=quantum,
            )
            for control_set in control_sets:
                incident = tuple(
                    sorted(
                        {
                            int(item)
                            for pair in control_set["pairs"]  # type: ignore[union-attr]
                            for item in pair[:2]
                        }
                    )
                )
                if len(incident) < 3:
                    continue
                remap = {prior: current for current, prior in enumerate(incident)}
                folded_material = tuple(material[index] for index in incident)
                pairs = tuple(
                    (
                        remap[int(left)],
                        remap[int(right)],
                        str(left_ref),
                        str(right_ref),
                        int(axis_anchor),
                        int(gap_start),
                        int(gap_end),
                    )
                    for (
                        left,
                        right,
                        left_ref,
                        right_ref,
                        axis_anchor,
                        gap_start,
                        gap_end,
                    ) in control_set["pairs"]  # type: ignore[union-attr]
                )
                controls = tuple(
                    step
                    for left, right, left_ref, right_ref, *_rest in pairs
                    for step in (
                        (right, left, left_ref),
                        (left, right, right_ref),
                    )
                )
                material_components = _material_graph_components(
                    chain_length=len(folded_material), controls=controls
                )
                if len(material_components) != 1:
                    continue
                separator_sets = _folded_separator_sets(
                    components=components,
                    material_refs=frozenset(
                        item.component_id for item in folded_material
                    ),
                    control_refs=frozenset(
                        str(item)
                        for pair in pairs
                        for item in pair[2:4]
                    ),
                    pairs=pairs,
                    material=folded_material,
                    axis=axis,
                    origin_side=origin_side,
                    quantum=quantum,
                    excluded_values=frozenset({material_value}),
                )
                if not separator_sets:
                    continue
                quantities = tuple(
                    _axis_extent(item, axis) // quantum
                    for item in folded_material
                )
                aggregate = sum(quantities)
                graph_edges = tuple(
                    sorted(
                        {
                            tuple(sorted((int(source), int(destination))))
                            for source, destination, _component_ref in controls
                        }
                    )
                )
                for separator_set in separator_sets:
                    excluded_refs = frozenset(
                        {
                            *(item.component_id for item in folded_material),
                            *(
                                str(item)
                                for pair in pairs
                                for item in pair[2:4]
                            ),
                            *(
                                str(item[2])
                                for item in separator_set["interfaces"]  # type: ignore[union-attr]
                            ),
                        }
                    )
                    for marker_value, marker_components in sorted(by_value.items()):
                        if marker_value == material_value or len(marker_components) < 2:
                            continue
                        for moving in marker_components:
                            moving_indices = tuple(
                                index
                                for index, current_material in enumerate(
                                    folded_material
                                )
                                if _marker_attached_to_surface(
                                    moving,
                                    current_material,
                                    axis=axis,
                                    origin_side=origin_side,
                                    surface=_surface_coordinate(
                                        current_material, axis, origin_side
                                    ),
                                )
                            )
                            if len(moving_indices) != 1:
                                continue
                            moving_index = moving_indices[0]
                            for fixed in marker_components:
                                if fixed.component_id == moving.component_id:
                                    continue
                                target_measurements = tuple(
                                    (index, extent // quantum)
                                    for index, current_material in enumerate(
                                        folded_material
                                    )
                                    if index != moving_index
                                    and (
                                        extent := _target_extent(
                                            fixed,
                                            axis=axis,
                                            origin_side=origin_side,
                                            origin=_axis_origin(
                                                current_material,
                                                axis,
                                                origin_side,
                                            ),
                                        )
                                    )
                                    > 0
                                    and extent % quantum == 0
                                    and extent // quantum <= aggregate
                                )
                                if len(target_measurements) != 1:
                                    continue
                                target_index, displayed_quantity = (
                                    target_measurements[0]
                                )
                                path = _first_index_path(
                                    node_count=len(folded_material),
                                    edges=graph_edges,
                                    source=moving_index,
                                    target=target_index,
                                )
                                required = _required_index_path_members(
                                    node_count=len(folded_material),
                                    edges=graph_edges,
                                    source=moving_index,
                                    target=target_index,
                                    path=path,
                                )
                                if (
                                    len(path) < 3
                                    or required != tuple(path[1:-1])
                                ):
                                    continue
                                interface_by_edge = {
                                    frozenset((int(item[0]), int(item[1]))): item
                                    for item in separator_set["interfaces"]  # type: ignore[union-attr]
                                }
                                oriented_interfaces: list[
                                    tuple[int, int, str, int, int]
                                ] = []
                                valid_path = True
                                for source, destination in zip(path, path[1:]):
                                    measured = interface_by_edge.get(
                                        frozenset((source, destination))
                                    )
                                    if measured is None:
                                        valid_path = False
                                        break
                                    left, right, component_ref, left_value, right_value = measured
                                    oriented_interfaces.append(
                                        (
                                            source,
                                            destination,
                                            str(component_ref),
                                            int(left_value)
                                            if int(left) == source
                                            else int(right_value),
                                            int(right_value)
                                            if int(right) == destination
                                            else int(left_value),
                                        )
                                    )
                                if not valid_path:
                                    continue
                                composite = _single_composite_measurement(
                                    components=components,
                                    moving=moving,
                                    excluded_refs=excluded_refs,
                                    quantum=quantum,
                                    tracked_by_component=tracked_by_component,
                                )
                                if composite is None:
                                    continue
                                carrier, identity_ref, attribute_ref = composite
                                origins = tuple(
                                    _axis_origin(item, axis, origin_side)
                                    for item in folded_material
                                )
                                description: dict[str, object] = {
                                    "axis": axis,
                                    "origin_side": origin_side,
                                    "origin": origins[0],
                                    "material_axis_origins": origins,
                                    "quantum": quantum,
                                    "material_value": material_value,
                                    "material_refs": tuple(
                                        item.component_id
                                        for item in folded_material
                                    ),
                                    "current_quantities": tuple(
                                        (item.component_id, quantities[index])
                                        for index, item in enumerate(
                                            folded_material
                                        )
                                    ),
                                    "quantities": quantities,
                                    "controls": controls,
                                    "display_constraint_index": target_index,
                                    "display_constraint_extent_ref": folded_material[
                                        target_index
                                    ].component_id,
                                    "display_constraint_quantity": displayed_quantity,
                                    "moving_marker_ref": moving.component_id,
                                    "fixed_marker_ref": fixed.component_id,
                                    "moving_marker_signature": (
                                        moving.value,
                                        moving.relative_pixels,
                                    ),
                                    "moving_zone_index": moving_index,
                                    "target_zone_index": target_index,
                                    "material_graph_edges": graph_edges,
                                    "geometric_graph_edges": tuple(
                                        (left, right)
                                        for left in range(len(folded_material))
                                        for right in range(left + 1, len(folded_material))
                                        if _orthogonal_start(
                                            folded_material[left], axis
                                        )
                                        <= _orthogonal_end(
                                            folded_material[right], axis
                                        )
                                        and _orthogonal_start(
                                            folded_material[right], axis
                                        )
                                        <= _orthogonal_end(
                                            folded_material[left], axis
                                        )
                                    ),
                                    "material_graph_components": material_components,
                                    "material_group_totals": (aggregate,),
                                    "separator_transition_candidates": tuple(
                                        oriented_interfaces
                                    ),
                                    "separator_morphology_ref": separator_set[
                                        "morphology_ref"
                                    ],
                                    "configuration_kind": "folded_separator_sequence",
                                    "topology_gap_evidence": False,
                                    "required_intermediate_indices": required,
                                    "first_shortest_zone_indices": path,
                                    "identity_ref": identity_ref,
                                    "identity_carrier_ref": carrier.component_id,
                                    "identity_attribute_ref": moving.component_id,
                                    "identity_attribute_entity_ref": attribute_ref,
                                    "identity_attribute_offset": (
                                        moving.bbox.top - carrier.bbox.top,
                                        moving.bbox.left - carrier.bbox.left,
                                    ),
                                    "identity_carrier_signature": (
                                        carrier.value,
                                        carrier.relative_pixels,
                                    ),
                                    "local_interface_coordinate_measurements": tuple(
                                        oriented_interfaces
                                    ),
                                    "screen_component_boxes": tuple(
                                        (
                                            item.component_id,
                                            item.bbox.top,
                                            item.bbox.left,
                                            item.bbox.bottom,
                                            item.bbox.right,
                                        )
                                        for item in folded_material
                                    ),
                                    "axis_order_without_relation_edges": tuple(
                                        edge
                                        for edge in (
                                            tuple(sorted((left, right)))
                                            for left, right in (
                                                (path[0], path[-1]),
                                            )
                                        )
                                        if edge not in graph_edges
                                    ),
                                    "support_to_next_origin_distance_pixels": abs(
                                        origins[path[-1]] - origins[path[0]]
                                    ),
                                    "coexisting_component_extent_pixels": (
                                        max(moving.bbox.right, carrier.bbox.right)
                                        - min(moving.bbox.left, carrier.bbox.left)
                                        + 1
                                        if axis == "column"
                                        else max(
                                            moving.bbox.bottom,
                                            carrier.bbox.bottom,
                                        )
                                        - min(
                                            moving.bbox.top,
                                            carrier.bbox.top,
                                        )
                                        + 1
                                    ),
                                    "observed_safe_quantity_lower_bound": quantities[
                                        moving_index
                                    ],
                                    "exact_limiting_boundary_extent_available": False,
                                }
                                description["zone_partition_bands"] = (
                                    _material_chain_bands(
                                        description=description,
                                        components=components,
                                    )
                                )
                                description["zone_measurement_boxes"] = (
                                    _folded_zone_measurement_boxes(
                                        description=description,
                                        components=components,
                                    )
                                )
                                outputs.append(description)
    unique: dict[str, dict[str, object]] = {}
    for item in outputs:
        key = canonical_json(
            (
                item["axis"],
                item["origin_side"],
                item["material_refs"],
                item["controls"],
                item["separator_transition_candidates"],
                item["moving_marker_ref"],
                item["fixed_marker_ref"],
            )
        )
        unique.setdefault(key, item)
    return tuple(unique.values())


def _folded_control_sets(
    *,
    components: tuple[ComponentDescription, ...],
    material: tuple[ComponentDescription, ...],
    material_value: int,
    axis: str,
    origin_side: str,
    quantum: int,
) -> tuple[dict[str, object], ...]:
    by_morphology: dict[tuple[object, ...], list[ComponentDescription]] = defaultdict(list)
    for component in components:
        if (
            component.value != material_value
            and component.area == quantum * quantum
            and component.bbox.height == quantum
            and component.bbox.width == quantum
        ):
            by_morphology[_component_morphology(component)].append(component)
    outputs: list[dict[str, object]] = []
    for morphology, controls in sorted(by_morphology.items()):
        incident: dict[str, int] = {}
        for control in controls:
            compatible = tuple(
                (
                    index,
                    abs(
                        _axis_origin(control, axis, origin_side)
                        - _axis_origin(item, axis, origin_side)
                    ),
                    _closed_interval_gap(
                        _orthogonal_start(control, axis),
                        _orthogonal_end(control, axis),
                        _orthogonal_start(item, axis),
                        _orthogonal_end(item, axis),
                    ),
                )
                for index, item in enumerate(material)
                if _closed_interval_gap(
                    _orthogonal_start(control, axis),
                    _orthogonal_end(control, axis),
                    _orthogonal_start(item, axis),
                    _orthogonal_end(item, axis),
                )
                <= quantum
            )
            if not compatible:
                continue
            minimum = min((item[2], item[1]) for item in compatible)
            matches = tuple(
                index
                for index, axis_distance, orthogonal_gap in compatible
                if (orthogonal_gap, axis_distance) == minimum
            )
            if len(matches) == 1:
                incident[control.component_id] = matches[0]
        by_axis_interval: dict[tuple[int, int], list[ComponentDescription]] = defaultdict(list)
        for control in controls:
            if control.component_id not in incident:
                continue
            if axis == "column":
                interval = (control.bbox.left, control.bbox.right)
            else:
                interval = (control.bbox.top, control.bbox.bottom)
            by_axis_interval[interval].append(control)
        pairs: list[tuple[int, int, str, str, int, int, int]] = []
        for interval, aligned in sorted(by_axis_interval.items()):
            ordered = sorted(aligned, key=lambda item: _orthogonal_start(item, axis))
            for before, after in zip(ordered, ordered[1:]):
                gap_start = _orthogonal_end(before, axis) + 1
                gap_end = _orthogonal_start(after, axis) - 1
                if (
                    gap_end - gap_start + 1 != quantum
                    or incident[before.component_id] == incident[after.component_id]
                ):
                    continue
                pairs.append(
                    (
                        incident[before.component_id],
                        incident[after.component_id],
                        before.component_id,
                        after.component_id,
                        interval[0] if origin_side == "minimum" else interval[1],
                        gap_start,
                        gap_end,
                    )
                )
        edge_keys = tuple(
            tuple(sorted((item[0], item[1]))) for item in pairs
        )
        if len(pairs) >= 2 and len(set(edge_keys)) == len(edge_keys):
            digest = hashlib.sha256(
                canonical_json(morphology).encode("utf-8")
            ).hexdigest()[:12]
            outputs.append(
                {
                    "morphology_ref": f"morphology.paired_control.{digest}",
                    "pairs": tuple(pairs),
                }
            )
    return tuple(outputs)


def _closed_interval_gap(
    left_start: int,
    left_end: int,
    right_start: int,
    right_end: int,
) -> int:
    """Count cells strictly between two closed one-dimensional intervals."""

    if left_end < right_start:
        return right_start - left_end - 1
    if right_end < left_start:
        return left_start - right_end - 1
    return 0


def _folded_separator_sets(
    *,
    components: tuple[ComponentDescription, ...],
    material_refs: frozenset[str],
    control_refs: frozenset[str],
    pairs: tuple[tuple[int, int, str, str, int, int, int], ...],
    material: tuple[ComponentDescription, ...],
    axis: str,
    origin_side: str,
    quantum: int,
    excluded_values: frozenset[int],
) -> tuple[dict[str, object], ...]:
    by_morphology: dict[tuple[object, ...], list[ComponentDescription]] = defaultdict(list)
    for component in components:
        if (
            component.component_id not in material_refs
            and component.component_id not in control_refs
            and component.value not in excluded_values
            and _orthogonal_extent(component, axis) == quantum
            and _axis_extent(component, axis) > quantum
        ):
            by_morphology[_component_morphology(component)].append(component)
    outputs: list[dict[str, object]] = []
    for morphology, members in sorted(by_morphology.items()):
        by_pair_and_offset: list[dict[int, tuple[ComponentDescription, ...]]] = []
        for _left, _right, _left_ref, _right_ref, anchor, gap_start, gap_end in pairs:
            offsets: dict[int, list[ComponentDescription]] = defaultdict(list)
            for component in members:
                if (
                    _orthogonal_start(component, axis) != gap_start
                    or _orthogonal_end(component, axis) != gap_end
                ):
                    continue
                offset = (
                    _axis_origin(component, axis, "minimum") - anchor
                    if origin_side == "minimum"
                    else anchor - _axis_origin(component, axis, "maximum")
                )
                if offset >= 0 and offset % quantum == 0:
                    offsets[offset].append(component)
            by_pair_and_offset.append(
                {key: tuple(value) for key, value in sorted(offsets.items())}
            )
        common_offsets = set(by_pair_and_offset[0]) if by_pair_and_offset else set()
        for offsets in by_pair_and_offset[1:]:
            common_offsets.intersection_update(offsets)
        for offset in sorted(common_offsets):
            matched = tuple(offsets[offset] for offsets in by_pair_and_offset)
            if any(len(items) != 1 for items in matched):
                continue
            interfaces: list[tuple[int, int, str, int, int]] = []
            valid = True
            for pair, items in zip(pairs, matched):
                left, right = int(pair[0]), int(pair[1])
                component = items[0]
                passage_coordinate = _axis_origin(
                    component, axis, origin_side
                )
                left_distance = abs(
                    passage_coordinate
                    - _axis_origin(material[left], axis, origin_side)
                )
                right_distance = abs(
                    passage_coordinate
                    - _axis_origin(material[right], axis, origin_side)
                )
                if (
                    left_distance <= 0
                    or right_distance <= 0
                    or left_distance % quantum
                    or right_distance % quantum
                ):
                    valid = False
                    break
                interfaces.append(
                    (
                        left,
                        right,
                        component.component_id,
                        left_distance // quantum,
                        right_distance // quantum,
                    )
                )
            if valid:
                digest = hashlib.sha256(
                    canonical_json((morphology, offset)).encode("utf-8")
                ).hexdigest()[:12]
                outputs.append(
                    {
                        "morphology_ref": f"morphology.repeated_interface.{digest}",
                        "interfaces": tuple(interfaces),
                    }
                )
    return tuple(outputs)


def _first_index_path(
    *,
    node_count: int,
    edges: tuple[tuple[int, int], ...],
    source: int,
    target: int,
    removed: int | None = None,
) -> tuple[int, ...]:
    if source == removed or target == removed:
        return ()
    neighbors = {index: set[int]() for index in range(node_count) if index != removed}
    for left, right in edges:
        if left == removed or right == removed:
            continue
        neighbors[left].add(right)
        neighbors[right].add(left)
    pending = deque((source,))
    predecessor: dict[int, int] = {}
    reached = {source}
    while pending:
        current = pending.popleft()
        if current == target:
            break
        for adjacent in sorted(neighbors[current]):
            if adjacent in reached:
                continue
            reached.add(adjacent)
            predecessor[adjacent] = current
            pending.append(adjacent)
    if target not in reached:
        return ()
    reverse_path = [target]
    while reverse_path[-1] != source:
        reverse_path.append(predecessor[reverse_path[-1]])
    return tuple(reversed(reverse_path))


def _required_index_path_members(
    *,
    node_count: int,
    edges: tuple[tuple[int, int], ...],
    source: int,
    target: int,
    path: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        index
        for index in path[1:-1]
        if not _first_index_path(
            node_count=node_count,
            edges=edges,
            source=source,
            target=target,
            removed=index,
        )
    )


def _single_composite_measurement(
    *,
    components: tuple[ComponentDescription, ...],
    moving: ComponentDescription,
    excluded_refs: frozenset[str],
    quantum: int,
    tracked_by_component: dict[str, str],
) -> tuple[ComponentDescription, str, str] | None:
    maximum_area = max(moving.area * 8, quantum * quantum * 8)
    moving_height = moving.bbox.height
    moving_width = moving.bbox.width
    carriers = tuple(
        component
        for component in components
        if component.component_id not in excluded_refs
        and component.component_id != moving.component_id
        and component.value != moving.value
        and component.area <= maximum_area
        and abs(component.bbox.height - moving_height) <= quantum
        and abs(component.bbox.width - moving_width) <= quantum
        and _components_are_locally_adjacent(component, moving, quantum)
        and component.component_id in tracked_by_component
    )
    attribute_ref = tracked_by_component.get(moving.component_id)
    if len(carriers) != 1 or attribute_ref is None:
        return None
    carrier = carriers[0]
    return carrier, tracked_by_component[carrier.component_id], attribute_ref


def _folded_zone_measurement_boxes(
    *,
    description: dict[str, object],
    components: tuple[ComponentDescription, ...],
) -> tuple[tuple[str, int, int, int, int], ...]:
    axis = str(description["axis"])
    origin_side = str(description["origin_side"])
    by_ref = {item.component_id: item for item in components}
    refs = tuple(str(item) for item in description["material_refs"])  # type: ignore[arg-type]
    bands = tuple(description["zone_partition_bands"])  # type: ignore[arg-type]
    global_axis_min = min(
        _axis_origin(item, axis, "minimum") for item in components
    )
    global_axis_max = max(
        _axis_origin(item, axis, "maximum") for item in components
    )
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (_ref, start, end) in enumerate(bands):
        groups[(int(start), int(end))].append(index)
    axis_ranges: dict[int, tuple[int, int]] = {}
    for indices in (groups[__yf_order_key] for __yf_order_key in sorted(groups)):
        ordered = sorted(
            indices,
            key=lambda index: _axis_origin(
                by_ref[refs[index]], axis, origin_side
            ),
            reverse=origin_side == "maximum",
        )
        origins = tuple(
            _axis_origin(by_ref[refs[index]], axis, origin_side)
            for index in ordered
        )
        for offset, index in enumerate(ordered):
            origin = origins[offset]
            if origin_side == "minimum":
                end = (
                    origins[offset + 1] - 1
                    if offset + 1 < len(origins)
                    else global_axis_max
                )
                axis_ranges[index] = (origin, end)
            else:
                start = (
                    origins[offset + 1] + 1
                    if offset + 1 < len(origins)
                    else global_axis_min
                )
                axis_ranges[index] = (start, origin)
    boxes: list[tuple[str, int, int, int, int]] = []
    for index, (ref, orth_start, orth_end) in enumerate(bands):
        axis_start, axis_end = axis_ranges[index]
        boxes.append(
            (
                str(ref),
                int(orth_start) if axis == "column" else axis_start,
                axis_start if axis == "column" else int(orth_start),
                int(orth_end) if axis == "column" else axis_end,
                axis_end if axis == "column" else int(orth_end),
            )
        )
    return tuple(boxes)


def _disconnected_separator_descriptions(
    *,
    components: tuple[ComponentDescription, ...],
    axis: str,
    quantum: int,
) -> tuple[dict[str, object], ...]:
    """Enumerate disconnected graphs with repeated separator morphology.

    The construction is morphology- and topology-driven.  It does not name a
    game, palette value, level, coordinate, semantic role, or solved answer.
    A description is emitted only when the exact geometry contains a
    disconnected extent-exchange graph, a displayed quantity outside the
    connected component's conserved total, and a repeated separator
    morphology with explicit feasible thresholds.  DRM assigns possible
    meanings to these measured relations.
    """

    by_value: dict[int, list[ComponentDescription]] = defaultdict(list)
    for component in components:
        by_value[component.value].append(component)
    outputs: list[dict[str, object]] = []
    for material_value, same_value in sorted(by_value.items()):
        for origin_side in ("minimum", "maximum"):
            material = tuple(
                item
                for item in same_value
                if _axis_extent(item, axis) % quantum == 0
                and _axis_extent(item, axis) >= quantum
                and _orthogonal_extent(item, axis) > quantum
            )
            origins = Counter(
                _axis_origin(item, axis, origin_side) for item in material
            )
            for origin, count in sorted(origins.items()):
                if count < 3:
                    continue
                chain = tuple(
                    sorted(
                        (
                            item
                            for item in material
                            if _axis_origin(item, axis, origin_side) == origin
                        ),
                        key=lambda item: _orthogonal_start(item, axis),
                    )
                )
                if len(chain) < 3 or any(
                    _orthogonal_end(left, axis) >= _orthogonal_start(right, axis)
                    for left, right in zip(chain, chain[1:])
                ):
                    continue
                controls = _chain_controls(
                    components=components,
                    chain=chain,
                    axis=axis,
                    origin_side=origin_side,
                    quantum=quantum,
                )
                if not controls or len(controls) >= 2 * (len(chain) - 1):
                    continue
                material_components = _material_graph_components(
                    chain_length=len(chain), controls=controls
                )
                component_by_index = {
                    index: component_index
                    for component_index, group in enumerate(material_components)
                    for index in group
                }
                quantities = tuple(
                    _axis_extent(item, axis) // quantum for item in chain
                )
                group_totals = tuple(
                    sum(quantities[index] for index in group)
                    for group in material_components
                )
                zone_bands = _zone_partition_bands(chain=chain, axis=axis)
                for marker_value, marker_components in sorted(by_value.items()):
                    if marker_value == material_value or len(marker_components) < 2:
                        continue
                    for moving_index, moving_material in enumerate(chain):
                        surface = _surface_coordinate(
                            moving_material, axis, origin_side
                        )
                        moving_markers = tuple(
                            item
                            for item in marker_components
                            if _marker_attached_to_surface(
                                item,
                                moving_material,
                                axis=axis,
                                origin_side=origin_side,
                                surface=surface,
                            )
                        )
                        for moving in moving_markers:
                            for fixed in marker_components:
                                if fixed.component_id == moving.component_id:
                                    continue
                                target_index = _component_zone_index(
                                    component=fixed,
                                    axis=axis,
                                    zone_bands=zone_bands,
                                )
                                if target_index is None or target_index == moving_index:
                                    continue
                                target_extent = _target_extent(
                                    fixed,
                                    axis=axis,
                                    origin_side=origin_side,
                                    origin=origin,
                                )
                                if target_extent <= 0 or target_extent % quantum:
                                    continue
                                displayed_quantity = target_extent // quantum
                                moving_group = component_by_index[moving_index]
                                target_group = component_by_index[target_index]
                                if moving_group == target_group:
                                    continue
                                missing_path_edges = tuple(
                                    range(
                                        min(moving_index, target_index),
                                        max(moving_index, target_index),
                                    )
                                )
                                if not missing_path_edges:
                                    continue
                                material_edges = {
                                    tuple(sorted((int(source), int(destination))))
                                    for source, destination, _ref in controls
                                }
                                if all(
                                    (edge, edge + 1) in material_edges
                                    for edge in missing_path_edges
                                ):
                                    continue
                                if displayed_quantity <= group_totals[moving_group]:
                                    continue
                                separator_groups = _separator_morphology_groups(
                                    components=components,
                                    chain=chain,
                                    edge_indices=missing_path_edges,
                                    axis=axis,
                                    origin_side=origin_side,
                                    origin=origin,
                                    quantum=quantum,
                                    material_value=material_value,
                                    marker_value=marker_value,
                                )
                                feasible = tuple(
                                    group
                                    for group in separator_groups
                                    if _separator_group_is_feasible(
                                        group=group,
                                        material_components=material_components,
                                        component_by_index=component_by_index,
                                        group_totals=group_totals,
                                    )
                                )
                                for separator_group in feasible:
                                    description = _describe_disconnected_separator_sequence(
                                        axis=axis,
                                        origin_side=origin_side,
                                        origin=origin,
                                        quantum=quantum,
                                        material_value=material_value,
                                        chain=chain,
                                        quantities=quantities,
                                        controls=controls,
                                        target_index=target_index,
                                        displayed_quantity=displayed_quantity,
                                        moving=moving,
                                        fixed=fixed,
                                        moving_index=moving_index,
                                        zone_bands=zone_bands,
                                        material_edges=material_edges,
                                        material_components=material_components,
                                        group_totals=group_totals,
                                        separator_group=separator_group,
                                    )
                                    if description:
                                        outputs.append(description)
    unique: dict[str, dict[str, object]] = {}
    for item in outputs:
        key = canonical_json(
            (
                item["axis"],
                item["origin_side"],
                item["material_refs"],
                item["moving_marker_ref"],
                item["fixed_marker_ref"],
                item["separator_transition_candidates"],
            )
        )
        unique[key] = item
    return tuple(unique.values())


def _describe_disconnected_separator_sequence(
    *,
    axis: str,
    origin_side: str,
    origin: int,
    quantum: int,
    material_value: int,
    chain: tuple[ComponentDescription, ...],
    quantities: tuple[int, ...],
    controls: tuple[tuple[int, int, str], ...],
    target_index: int,
    displayed_quantity: int,
    moving: ComponentDescription,
    fixed: ComponentDescription,
    moving_index: int,
    zone_bands: tuple[tuple[int, int], ...],
    material_edges: set[tuple[int, int]],
    material_components: tuple[tuple[int, ...], ...],
    group_totals: tuple[int, ...],
    separator_group: dict[str, object],
) -> dict[str, object]:
    """Normalize one measured separator ordering into descriptive facts."""

    direction = 1 if target_index > moving_index else -1
    ordered_doors = tuple(
        door
        for door in sorted(
            tuple(separator_group["doors"]),  # type: ignore[arg-type]
            key=lambda item: int(item[0]),
            reverse=direction < 0,
        )
        if int(door[0])
        in range(min(moving_index, target_index), max(moving_index, target_index))
    )
    oriented_doors = tuple(
        (
            int(edge) if direction > 0 else int(edge) + 1,
            int(edge) + 1 if direction > 0 else int(edge),
            str(component_ref),
            int(threshold),
        )
        for edge, component_ref, threshold in ordered_doors
    )
    if not oriented_doors:
        return {}
    return {
        "axis": axis,
        "origin_side": origin_side,
        "origin": origin,
        "quantum": quantum,
        "material_value": material_value,
        "material_refs": tuple(item.component_id for item in chain),
        "current_quantities": tuple(
            (item.component_id, quantities[index])
            for index, item in enumerate(chain)
        ),
        "quantities": quantities,
        "controls": controls,
        "display_constraint_index": target_index,
        "display_constraint_extent_ref": chain[target_index].component_id,
        "display_constraint_quantity": displayed_quantity,
        "moving_marker_ref": moving.component_id,
        "fixed_marker_ref": fixed.component_id,
        "moving_marker_signature": (moving.value, moving.relative_pixels),
        "moving_zone_index": moving_index,
        "target_zone_index": target_index,
        "zone_partition_bands": tuple(
            (chain[index].component_id, start, end)
            for index, (start, end) in enumerate(zone_bands)
        ),
        "material_graph_edges": tuple(sorted(material_edges)),
        "geometric_graph_edges": tuple(
            (index, index + 1) for index in range(len(chain) - 1)
        ),
        "material_graph_components": material_components,
        "material_group_totals": group_totals,
        "separator_transition_candidates": oriented_doors,
        "separator_morphology_ref": str(separator_group["morphology_ref"]),
        "configuration_kind": "disconnected_separator_sequence",
        "topology_gap_evidence": True,
    }


def _material_graph_components(
    *,
    chain_length: int,
    controls: tuple[tuple[int, int, str], ...],
) -> tuple[tuple[int, ...], ...]:
    neighbors: dict[int, set[int]] = {index: set() for index in range(chain_length)}
    for source, destination, _component_ref in controls:
        neighbors[int(source)].add(int(destination))
        neighbors[int(destination)].add(int(source))
    remaining = set(range(chain_length))
    groups: list[tuple[int, ...]] = []
    while remaining:
        root = min(remaining)
        reached = {root}
        frontier = [root]
        while frontier:
            current = frontier.pop()
            for neighbor in neighbors[current] - reached:
                reached.add(neighbor)
                frontier.append(neighbor)
        remaining -= reached
        groups.append(tuple(sorted(reached)))
    return tuple(groups)


def _zone_partition_bands(
    *, chain: tuple[ComponentDescription, ...], axis: str
) -> tuple[tuple[int, int], ...]:
    starts = tuple(_orthogonal_start(item, axis) for item in chain)
    ends = tuple(_orthogonal_end(item, axis) for item in chain)
    separators = tuple(
        (ends[index] + starts[index + 1]) // 2
        for index in range(len(chain) - 1)
    )
    return tuple(
        (
            starts[0] if index == 0 else separators[index - 1] + 1,
            ends[-1] if index == len(chain) - 1 else separators[index],
        )
        for index in range(len(chain))
    )


def _component_zone_index(
    *,
    component: ComponentDescription,
    axis: str,
    zone_bands: tuple[tuple[int, int], ...],
) -> int | None:
    center = (
        _orthogonal_start(component, axis) + _orthogonal_end(component, axis)
    ) // 2
    matches = tuple(
        index
        for index, (start, end) in enumerate(zone_bands)
        if start <= center <= end
    )
    return matches[0] if len(matches) == 1 else None


def _separator_morphology_groups(
    *,
    components: tuple[ComponentDescription, ...],
    chain: tuple[ComponentDescription, ...],
    edge_indices: tuple[int, ...],
    axis: str,
    origin_side: str,
    origin: int,
    quantum: int,
    material_value: int,
    marker_value: int,
) -> tuple[dict[str, object], ...]:
    by_morphology: dict[
        tuple[object, ...], dict[int, list[tuple[str, int]]]
    ] = defaultdict(lambda: defaultdict(list))
    for edge in edge_indices:
        left, right = chain[edge], chain[edge + 1]
        for component in components:
            if component.value in {material_value, marker_value}:
                continue
            if _orthogonal_extent(component, axis) != quantum:
                continue
            if _axis_extent(component, axis) <= quantum:
                continue
            if not (
                _orthogonal_start(component, axis)
                > _orthogonal_end(left, axis)
                and _orthogonal_end(component, axis)
                < _orthogonal_start(right, axis)
            ):
                continue
            distance = (
                origin - _axis_origin(component, axis, "maximum")
                if origin_side == "maximum"
                else _axis_origin(component, axis, "minimum") - origin
            )
            if distance <= 0 or distance % quantum:
                continue
            morphology = (
                component.value,
                component.bbox.height,
                component.bbox.width,
                component.relative_pixels,
            )
            by_morphology[morphology][edge].append(
                (component.component_id, distance // quantum)
            )
    outputs: list[dict[str, object]] = []
    for morphology, by_edge in sorted(by_morphology.items()):
        if set(by_edge) != set(edge_indices) or any(
            len(by_edge[edge]) != 1 for edge in edge_indices
        ):
            continue
        digest = hashlib.sha256(
            canonical_json(morphology).encode("utf-8")
        ).hexdigest()[:12]
        outputs.append(
            {
                "morphology_ref": f"morphology.separator.{digest}",
                "doors": tuple(
                    (edge, *by_edge[edge][0]) for edge in edge_indices
                ),
            }
        )
    return tuple(outputs)


def _separator_group_is_feasible(
    *,
    group: dict[str, object],
    material_components: tuple[tuple[int, ...], ...],
    component_by_index: dict[int, int],
    group_totals: tuple[int, ...],
) -> bool:
    for raw_edge, _component_ref, raw_threshold in group["doors"]:  # type: ignore[union-attr]
        edge, threshold = int(raw_edge), int(raw_threshold)
        left_group = component_by_index[edge]
        right_group = component_by_index[edge + 1]
        required_left = threshold
        required_right = threshold
        if left_group == right_group:
            if group_totals[left_group] < required_left + required_right:
                return False
        elif (
            group_totals[left_group] < required_left
            or group_totals[right_group] < required_right
        ):
            return False
    return True


def _disconnected_separator_routes(
    description: dict[str, object],
    *,
    max_expanded_states: int,
    max_plan_actions: int,
    control_order_variant: str,
) -> tuple[dict[str, object], ...]:
    """Construct one bounded route through measured separators.

    Separator semantics are intentionally absent here.  Each route is a pure
    counterfactual simulation under one measured transition candidate; SRC
    and DRM decide whether that candidate is an admissible ontology extension.
    """

    initial = tuple(int(item) for item in description["quantities"])  # type: ignore[arg-type]
    controls = tuple(
        sorted(
            (
                (int(source), int(destination), str(component_ref))
                for source, destination, component_ref in description["controls"]  # type: ignore[union-attr]
            ),
            key=lambda item: (item[2], item[0], item[1]),
        )
    )
    material_components = tuple(description["material_graph_components"])  # type: ignore[arg-type]
    component_by_index = {
        int(index): group_index
        for group_index, group in enumerate(material_components)
        for index in group
    }
    material_refs = tuple(str(item) for item in description["material_refs"])  # type: ignore[arg-type]
    target_index = int(description["target_zone_index"])
    partial = _DisconnectedRouteWitness(
        state=initial,
        marker_index=int(description["moving_zone_index"]),
        route=(),
        quantity_states=(),
        marker_states=(),
        expanded_state_count=0,
    )

    for raw_transition in description["separator_transition_candidates"]:  # type: ignore[union-attr]
        transition = tuple(raw_transition)
        if len(transition) == 4:
            source, destination, component_ref, source_coordinate = transition
            destination_coordinate = source_coordinate
        elif len(transition) == 5:
            (
                source,
                destination,
                component_ref,
                source_coordinate,
                destination_coordinate,
            ) = transition
        else:
            return ()
        source = int(source)
        destination = int(destination)
        source_coordinate = int(source_coordinate)
        destination_coordinate = int(destination_coordinate)
        if component_by_index[source] == component_by_index[destination]:
            stages = (
                (
                    {
                        source: source_coordinate,
                        destination: destination_coordinate,
                    },
                    frozenset(material_components[component_by_index[source]]),
                ),
            )
        else:
            stages = (
                (
                    {source: source_coordinate},
                    frozenset(material_components[component_by_index[source]]),
                ),
                (
                    {destination: destination_coordinate},
                    frozenset(material_components[component_by_index[destination]]),
                ),
            )
        if partial.marker_index != source:
            return ()
        for goals, allowed_indices in stages:
            remaining_state_budget = max_expanded_states - partial.expanded_state_count
            remaining_action_budget = max_plan_actions - len(partial.route)
            material_witness = _shortest_material_condition_route(
                initial=partial.state,
                controls=controls,
                required_values=goals,
                allowed_indices=allowed_indices,
                max_expanded_states=remaining_state_budget,
                max_plan_actions=remaining_action_budget,
                control_order_variant=control_order_variant,
            )
            if material_witness is None:
                return ()
            partial = _append_extent_exchange_steps(
                partial=partial,
                witness=material_witness,
                material_refs=material_refs,
            )
        if (
            partial.state[source] != source_coordinate
            or partial.state[destination] != destination_coordinate
            or len(partial.route) >= max_plan_actions
        ):
            return ()
        partial = _DisconnectedRouteWitness(
            state=partial.state,
            marker_index=destination,
            route=(
                *partial.route,
                (
                    source,
                    destination,
                    str(component_ref),
                    "separator_zone_transition_candidate",
                ),
            ),
            quantity_states=(
                *partial.quantity_states,
                tuple(
                    (material_refs[index], value)
                    for index, value in enumerate(partial.state)
                ),
            ),
            marker_states=(
                *partial.marker_states,
                material_refs[destination],
            ),
            expanded_state_count=partial.expanded_state_count,
        )

    if partial.marker_index != target_index:
        return ()
    target_witness = _shortest_material_condition_route(
        initial=partial.state,
        controls=controls,
        required_values={
            target_index: int(description["display_constraint_quantity"])
        },
        allowed_indices=frozenset(
            material_components[component_by_index[target_index]]
        ),
        max_expanded_states=max_expanded_states - partial.expanded_state_count,
        max_plan_actions=max_plan_actions - len(partial.route),
        control_order_variant=control_order_variant,
    )
    if target_witness is None:
        return ()
    completed = _append_extent_exchange_steps(
        partial=partial,
        witness=target_witness,
        material_refs=material_refs,
    )

    state = completed.state
    route = completed.route
    quantity_states = completed.quantity_states
    marker_states = completed.marker_states
    if (
        not route
        or state[target_index]
        != int(description["display_constraint_quantity"])
    ):
        return ()
    route_steps = tuple(
        (
            material_refs[source],
            material_refs[destination],
            component_ref,
            step_kind,
        )
        for source, destination, component_ref, step_kind in route
    )
    component_sequence = tuple(item[2] for item in route_steps)
    identity_ref = str(description.get("identity_ref") or "")
    identity_states = (
        tuple(((identity_ref, zone_ref),) for zone_ref in marker_states)
        if identity_ref
        else ()
    )
    moved_identity_states = (
        tuple(
            (identity_ref,)
            if str(step[3]) == "separator_zone_transition_candidate"
            else ()
            for step in route_steps
        )
        if identity_ref
        else ()
    )
    phase_count = 1 + sum(
        left != right
        for left, right in zip(component_sequence, component_sequence[1:])
    )
    digest = hashlib.sha256(
        canonical_json(
            (
                description["axis"],
                description["origin_side"],
                material_refs,
                route[0],
                route[-1][2],
                phase_count,
                completed.state,
                description["display_constraint_quantity"],
            )
        )
        .encode("utf-8")
    ).hexdigest()[:12]
    return (
        {
            "route_ref": f"simulation.quantized_route.disconnected.{digest}",
            "route_component_refs": component_sequence,
            "route_transfer_steps": route_steps,
            "route_quantity_states": quantity_states,
            "route_marker_states": marker_states,
            "route_identity_states": identity_states,
            "route_moved_identity_refs": moved_identity_states,
            "first_component_ref": component_sequence[0],
            "primitive_action_count": len(route_steps),
            "transfer_phase_count": phase_count,
            "actuator_switch_count": phase_count - 1,
            "expected_quantities_after_action": quantity_states[0],
            "expected_marker_zone_after_action": marker_states[0],
            **(
                {
                    "expected_identity_zones_after_action": identity_states[0],
                    "expected_moved_identity_refs_after_action": moved_identity_states[0],
                }
                if identity_states
                else {}
            ),
            "current_marker_zone": material_refs[
                int(description["moving_zone_index"])
            ],
            "expanded_state_count": completed.expanded_state_count,
        },
    )


def _append_extent_exchange_steps(
    *,
    partial: _DisconnectedRouteWitness,
    witness: _BoundedRouteWitness,
    material_refs: tuple[str, ...],
) -> _DisconnectedRouteWitness:
    current_state = partial.state
    current_route = partial.route
    current_quantity_states = partial.quantity_states
    current_marker_states = partial.marker_states
    for source, destination, component_ref in witness.route:
        updated = list(current_state)
        updated[source] -= 1
        updated[destination] += 1
        current_state = tuple(updated)
        current_route = (
            *current_route,
            (source, destination, component_ref, "extent_exchange_candidate"),
        )
        current_quantity_states = (
            *current_quantity_states,
            tuple(
                (material_refs[index], value)
                for index, value in enumerate(current_state)
            ),
        )
        current_marker_states = (
            *current_marker_states,
            material_refs[partial.marker_index],
        )
    return _DisconnectedRouteWitness(
        state=current_state,
        marker_index=partial.marker_index,
        route=current_route,
        quantity_states=current_quantity_states,
        marker_states=current_marker_states,
        expanded_state_count=(
            partial.expanded_state_count + witness.expanded_state_count
        ),
    )


def _shortest_material_condition_route(
    *,
    initial: tuple[int, ...],
    controls: tuple[tuple[int, int, str], ...],
    required_values: dict[int, int],
    allowed_indices: frozenset[int],
    max_expanded_states: int,
    max_plan_actions: int,
    control_order_variant: str,
) -> _BoundedRouteWitness | None:
    """Return the first bounded shortest witness to a quantity condition."""

    return _first_shortest_route_witness(
        initial=initial,
        controls=controls,
        goal_test=lambda state: all(
            state[index] == expected
            for index, expected in sorted(required_values.items())
        ),
        allowed_indices=allowed_indices,
        max_expanded_states=max_expanded_states,
        max_depth=max_plan_actions,
        control_order_variant=control_order_variant,
    )


def _disconnected_topology_route_facts(
    representative: dict[str, object],
) -> dict[str, object]:
    configuration_kind = str(representative.get("configuration_kind") or "")
    if configuration_kind not in {
        "disconnected_separator_sequence",
        "folded_separator_sequence",
    }:
        return {}
    steps = tuple(representative["route_transfer_steps"])  # type: ignore[arg-type]
    extent_exchange_refs = tuple(
        dict.fromkeys(
            str(item[2])
            for item in steps
            if str(item[3]) == "extent_exchange_candidate"
        )
    )
    zone_transition_refs = tuple(
        dict.fromkeys(
            str(item[2])
            for item in steps
            if str(item[3]) == "separator_zone_transition_candidate"
        )
    )
    doors = tuple(representative["separator_transition_candidates"])  # type: ignore[arg-type]
    threshold_measurements = (
        tuple(
            (
                str(item[2]),
                int(item[3]),
                int(item[4]) if len(tuple(item)) == 5 else int(item[3]),
            )
            for item in doors
        )
        if configuration_kind == "folded_separator_sequence"
        else tuple((str(item[2]), int(item[3])) for item in doors)
    )
    facts: dict[str, object] = {
        "topology_gap_evidence": bool(representative["topology_gap_evidence"]),
        "current_step_measurement_kind": str(steps[0][3]),
        "current_marker_zone": representative["current_marker_zone"],
        "expected_marker_zone_after_action": representative[
            "expected_marker_zone_after_action"
        ],
        "route_marker_states": representative["route_marker_states"],
        "zone_partition_bands": representative["zone_partition_bands"],
        "extent_exchange_component_refs": extent_exchange_refs,
        "separator_zone_transition_component_refs": zone_transition_refs,
        "all_route_component_refs": (
            tuple(dict.fromkeys(str(item[2]) for item in steps))
            if configuration_kind == "folded_separator_sequence"
            else extent_exchange_refs
        ),
        "all_action_component_refs": tuple(dict.fromkeys(str(item[2]) for item in steps)),
        "conserved_extent_graph_edges": representative["material_graph_edges"],
        "geometric_graph_edges": representative["geometric_graph_edges"],
        "conserved_extent_graph_components": representative["material_graph_components"],
        "conserved_extent_group_totals": representative["material_group_totals"],
        "moving_zone_index": int(representative["moving_zone_index"]),
        "target_zone_index": int(representative["target_zone_index"]),
        "separator_activation_thresholds": threshold_measurements,
        "separator_transition_measurements": doors,
        "separator_morphology_ref": representative["separator_morphology_ref"],
    }
    if configuration_kind == "folded_separator_sequence":
        identity_ref = str(representative["identity_ref"])
        identity_states = tuple(representative["route_identity_states"])  # type: ignore[arg-type]
        moved_identity_states = tuple(
            representative["route_moved_identity_refs"]  # type: ignore[arg-type]
        )
        facts.update(
            {
                "route_identity_states": identity_states,
                "route_moved_identity_refs": moved_identity_states,
                "expected_identity_zones_after_action": identity_states[0],
                "expected_moved_identity_refs_after_action": moved_identity_states[0],
                "identity_composite_bindings": (
                    (
                        identity_ref,
                        identity_ref,
                        str(representative["identity_attribute_entity_ref"]),
                        int(tuple(representative["identity_attribute_offset"])[0]),
                        int(tuple(representative["identity_attribute_offset"])[1]),
                    ),
                ),
                "identity_carrier_signatures": (
                    (
                        identity_ref,
                        int(tuple(representative["identity_carrier_signature"])[0]),
                        tuple(tuple(position) for position in tuple(representative["identity_carrier_signature"])[1]),
                    ),
                ),
                "composite_link_measurements": (
                    (
                        identity_ref,
                        "adjacent",
                        tuple(representative["identity_attribute_offset"]),
                        False,
                    ),
                ),
                "required_intermediate_indices": representative[
                    "required_intermediate_indices"
                ],
                "first_shortest_zone_indices": representative[
                    "first_shortest_zone_indices"
                ],
                "local_interface_coordinate_measurements": representative[
                    "local_interface_coordinate_measurements"
                ],
                "screen_component_boxes": representative[
                    "screen_component_boxes"
                ],
                "axis_order_without_relation_edges": representative[
                    "axis_order_without_relation_edges"
                ],
                "support_axis": representative["axis"],
                "support_origin_side": representative["origin_side"],
                "material_axis_origin_coordinates": representative[
                    "material_axis_origins"
                ],
                "zone_measurement_boxes": representative[
                    "zone_measurement_boxes"
                ],
                "support_to_next_origin_distance_pixels": representative[
                    "support_to_next_origin_distance_pixels"
                ],
                "coexisting_component_extent_pixels": representative[
                    "coexisting_component_extent_pixels"
                ],
                "observed_safe_quantity_lower_bound": representative[
                    "observed_safe_quantity_lower_bound"
                ],
                "exact_limiting_boundary_extent_available": representative[
                    "exact_limiting_boundary_extent_available"
                ],
            }
        )
    return facts


def _shortest_display_match_routes(
    description: dict[str, object],
    *,
    max_expanded_states: int,
    max_plan_actions: int,
    control_order_variant: str,
) -> tuple[dict[str, object], ...]:
    quantities = tuple(int(item) for item in description["quantities"])  # type: ignore[arg-type]
    controls = tuple(description["controls"])  # type: ignore[arg-type]
    display_index = int(description["display_constraint_index"])
    displayed_quantity = int(description["display_constraint_quantity"])
    if quantities[display_index] == displayed_quantity:
        return ()
    depth_limit = min(max_plan_actions, max(1, 2 * sum(quantities)))
    witness = _first_shortest_route_witness(
        initial=quantities,
        controls=tuple(
            (int(source), int(target), str(component_ref))
            for source, target, component_ref in controls
        ),
        goal_test=lambda state: state[display_index] == displayed_quantity,
        capacities=tuple(int(item) for item in description.get("capacities", ())),
        max_expanded_states=max_expanded_states,
        max_depth=depth_limit,
        control_order_variant=control_order_variant,
    )
    families = (witness,) if witness is not None else ()
    outputs: list[dict[str, object]] = []
    material_refs = tuple(str(item) for item in description["material_refs"])  # type: ignore[arg-type]
    for family in families:
        route = family.route
        simulated = list(quantities)
        quantity_states: list[tuple[tuple[str, int], ...]] = []
        transfer_steps: list[tuple[str, str, str]] = []
        for source, target, component_ref in route:
            simulated[source] -= 1
            simulated[target] += 1
            transfer_steps.append(
                (material_refs[source], material_refs[target], component_ref)
            )
            quantity_states.append(
                tuple(
                    (material_refs[index], simulated[index])
                    for index in range(len(simulated))
                )
            )
        component_sequence = tuple(item[2] for item in route)
        phase_count = 1 + sum(
            left != right
            for left, right in zip(component_sequence, component_sequence[1:])
        )
        digest = hashlib.sha256(
            canonical_json(
                (
                    description["axis"],
                    description["origin_side"],
                    material_refs,
                    route[0],
                    route[-1][2],
                    phase_count,
                    family.state,
                    description["display_constraint_extent_ref"],
                    description["display_constraint_quantity"],
                )
            ).encode("utf-8")
        ).hexdigest()[:12]
        outputs.append(
            {
                "route_ref": f"simulation.quantized_route.local.{digest}",
                "route_component_refs": component_sequence,
                "route_transfer_steps": tuple(transfer_steps),
                "route_quantity_states": tuple(quantity_states),
                "first_component_ref": component_sequence[0],
                "primitive_action_count": len(route),
                "transfer_phase_count": phase_count,
                "actuator_switch_count": phase_count - 1,
                "expected_quantities_after_action": quantity_states[0],
                "expanded_state_count": family.expanded_state_count,
            }
        )
    return tuple(outputs)


def _shortest_complete_assignment_routes(
    description: dict[str, object],
    *,
    max_expanded_states: int,
    max_plan_actions: int,
    control_order_variant: str,
) -> tuple[dict[str, object], ...]:
    """Construct the first exact shortest capacity-valid route.

    No alternative path or path family is retained.  DRM remains authoritative
    over comparisons between the at-most-three structural witnesses.
    """

    initial = tuple(int(item) for item in description["quantities"])  # type: ignore[arg-type]
    target = tuple(int(item) for item in description["target_quantity_values"])  # type: ignore[arg-type]
    controls = tuple(
        sorted(
            (
                (int(source), int(destination), str(component_ref))
                for source, destination, component_ref in description["controls"]  # type: ignore[union-attr]
            ),
            key=lambda item: (item[2], item[0], item[1]),
        )
    )
    capacities = tuple(int(item) for item in description.get("capacities", ()))
    if capacities and len(capacities) != len(initial):
        return ()
    witness = _first_shortest_route_witness(
        initial=initial,
        controls=controls,
        goal_test=lambda state: state == target,
        capacities=capacities,
        max_expanded_states=max_expanded_states,
        max_depth=max_plan_actions,
        control_order_variant=control_order_variant,
    )
    if witness is None or not witness.route:
        return ()
    return (_complete_route_output(description=description, witness=witness),)


def _complete_route_output(
    *,
    description: dict[str, object],
    witness: _BoundedRouteWitness,
) -> dict[str, object]:
    route = witness.route
    material_refs = tuple(str(item) for item in description["material_refs"])  # type: ignore[arg-type]
    simulated = [int(item) for item in description["quantities"]]  # type: ignore[arg-type]
    quantity_states: list[tuple[tuple[str, int], ...]] = []
    transfer_steps: list[tuple[str, str, str]] = []
    for source, destination, component_ref in route:
        simulated[source] -= 1
        simulated[destination] += 1
        transfer_steps.append(
            (material_refs[source], material_refs[destination], component_ref)
        )
        quantity_states.append(
            tuple(
                (material_refs[index], simulated[index])
                for index in range(len(simulated))
            )
        )
    sequence = tuple(item[2] for item in route)
    phase_count = 1 + sum(
        left != right for left, right in zip(sequence, sequence[1:])
    )
    state_values = (
        tuple(int(item) for item in description["quantities"]),  # type: ignore[arg-type]
        *(tuple(value for _ref, value in state) for state in quantity_states),
    )
    reusable_buffers: list[str] = []
    buffer_cycles: list[FrozenMap] = []
    capacities = tuple(int(item) for item in description.get("capacities", ()))
    display_indices = {
        int(item["material_index"])
        for item in description.get("displayed_marker_bindings", ())
    }
    for index in range(1, len(material_refs) - 1):
        if index in display_indices:
            continue
        values = tuple(state[index] for state in state_values)
        positive_runs = sum(
            value > 0 and (position == 0 or values[position - 1] == 0)
            for position, value in enumerate(values)
        )
        empty_after_positive = any(
            values[position] == 0 and values[position - 1] > 0
            for position in range(1, len(values))
        )
        if positive_runs >= 2 and empty_after_positive:
            reusable_buffers.append(material_refs[index])
            buffer_cycles.append(
                FrozenMap(
                    {
                        "entity_ref": material_refs[index],
                        "positive_run_count": positive_runs,
                        "emptied_after_fill": True,
                        "capacity": capacities[index] if capacities else None,
                    }
                )
            )
    digest = hashlib.sha256(
        canonical_json(
            (
                description["axis"],
                description["origin_side"],
                material_refs,
                route[0],
                route[-1][2],
                phase_count,
                description["target_quantity_values"],
                capacities,
            )
        ).encode("utf-8")
    ).hexdigest()[:12]
    return {
        "route_ref": f"simulation.quantized_route.complete.{digest}",
        "route_component_refs": sequence,
        "route_transfer_steps": tuple(transfer_steps),
        "route_quantity_states": tuple(quantity_states),
        "first_component_ref": sequence[0],
        "primitive_action_count": len(route),
        "transfer_phase_count": phase_count,
        "actuator_switch_count": phase_count - 1,
        "expected_quantities_after_action": quantity_states[0],
        "expanded_state_count": witness.expanded_state_count,
        "repeated_fill_empty_component_refs": tuple(reusable_buffers),
        "buffer_cycle_witnesses": tuple(buffer_cycles),
        "all_capacity_constraints_valid": True,
        "first_action_moves_reusable_extent_toward_displayed_constraint": bool(
            route
            and material_refs[route[0][0]] in reusable_buffers
            and route[0][1] in display_indices
            and tuple(int(item) for item in description["target_quantity_values"])[
                route[0][1]
            ]
            > tuple(int(item) for item in description["quantities"])[route[0][1]]
        ),
    }


def _axis_extent(component: ComponentDescription, axis: str) -> int:
    return component.bbox.width if axis == "column" else component.bbox.height


def _axis_origin(
    component: ComponentDescription,
    axis: str,
    origin_side: str,
) -> int:
    if axis == "column":
        return component.bbox.left if origin_side == "minimum" else component.bbox.right
    return component.bbox.top if origin_side == "minimum" else component.bbox.bottom


def _orthogonal_start(component: ComponentDescription, axis: str) -> int:
    return component.bbox.top if axis == "column" else component.bbox.left


def _orthogonal_end(component: ComponentDescription, axis: str) -> int:
    return component.bbox.bottom if axis == "column" else component.bbox.right


def _surface_coordinate(
    component: ComponentDescription,
    axis: str,
    origin_side: str,
) -> int:
    if axis == "column":
        return component.bbox.right + 1 if origin_side == "minimum" else component.bbox.left - 1
    return component.bbox.bottom + 1 if origin_side == "minimum" else component.bbox.top - 1


def _marker_attached_to_surface(
    marker: ComponentDescription,
    material: ComponentDescription,
    *,
    axis: str,
    origin_side: str,
    surface: int,
) -> bool:
    marker_axis_start = marker.bbox.left if axis == "column" else marker.bbox.top
    marker_axis_end = marker.bbox.right if axis == "column" else marker.bbox.bottom
    attached = marker_axis_start == surface if origin_side == "minimum" else marker_axis_end == surface
    overlaps_orthogonally = not (
        _orthogonal_end(marker, axis) < _orthogonal_start(material, axis)
        or _orthogonal_start(marker, axis) > _orthogonal_end(material, axis)
    )
    return attached and overlaps_orthogonally


def _target_extent(
    marker: ComponentDescription,
    *,
    axis: str,
    origin_side: str,
    origin: int,
) -> int:
    if axis == "column":
        coordinate = marker.bbox.left if origin_side == "minimum" else marker.bbox.right
    else:
        coordinate = marker.bbox.top if origin_side == "minimum" else marker.bbox.bottom
    return coordinate - origin if origin_side == "minimum" else origin - coordinate


def _control_on_material_origin(
    control: ComponentDescription,
    material: ComponentDescription,
    axis: str,
    origin_side: str,
) -> bool:
    if axis == "column":
        return (
            control.bbox.left == material.bbox.left
            if origin_side == "minimum"
            else control.bbox.right == material.bbox.right
        )
    return (
        control.bbox.top == material.bbox.top
        if origin_side == "minimum"
        else control.bbox.bottom == material.bbox.bottom
    )



def _interior_point(component: ComponentDescription) -> Position:
    pixels = set(component.pixels)
    center = (
        (component.bbox.top + component.bbox.bottom) // 2,
        (component.bbox.left + component.bbox.right) // 2,
    )
    if center in pixels:
        return center
    return min(
        component.pixels,
        key=lambda point: (
            abs(point[0] - center[0]) + abs(point[1] - center[1]),
            point,
        ),
    )


__all__ = (
    "analyze_quantized_plan_reconciliation",
    "enumerate_transferred_quantized_plans",
)
