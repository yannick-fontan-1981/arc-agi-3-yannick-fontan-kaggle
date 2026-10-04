"""Pure bounded measurements for conditional inverse-anchor extensions."""

from __future__ import annotations

from agents.yf_arc3_v5.capabilities.contracts import (
    InverseAnchorSuccessorInput,
    InverseAnchorSuccessorMeasurements,
    InverseAnchorSystemState,
    InverseAnchorTypedClearanceInput,
    InverseAnchorTypedClearanceMeasurements,
    InverseAnchorWaypointCandidateMeasurement,
    InverseAnchorWaypointCandidatesInput,
    InverseAnchorWaypointCandidatesMeasurements,
)


def measure_inverse_anchor_typed_clearance(
    value: InverseAnchorTypedClearanceInput,
) -> InverseAnchorTypedClearanceMeasurements:
    """Compare anchor, dependent and supplied swept footprints independently."""

    anchor_allowed = set(value.anchor_allowed_positions)
    dependent_allowed = set(value.dependent_allowed_positions)
    anchor_blocked = tuple(
        sorted(set(value.anchor_footprint) - anchor_allowed)
    )
    dependent_blocked = tuple(
        sorted(set(value.dependent_footprint) - dependent_allowed)
    )
    swept_positions = {
        position
        for footprint in value.dependent_swept_footprints
        for position in footprint
    }
    transition_blocked = tuple(sorted(swept_positions - dependent_allowed))
    if value.transition_path_policy == "unmeasured":
        transition_status = "unknown"
    elif transition_blocked:
        transition_status = "blocked"
    else:
        transition_status = "clear"
    return InverseAnchorTypedClearanceMeasurements(
        anchor_blocked_positions=anchor_blocked,
        dependent_blocked_positions=dependent_blocked,
        transition_blocked_positions=transition_blocked,
        anchor_destination_clear=not anchor_blocked,
        dependent_destination_clear=not dependent_blocked,
        transition_path_status=transition_status,
        examined_position_count=(
            len(value.anchor_footprint)
            + len(value.dependent_footprint)
            + sum(len(item) for item in value.dependent_swept_footprints)
        ),
    )


def measure_inverse_anchor_successor(
    value: InverseAnchorSuccessorInput,
) -> InverseAnchorSuccessorMeasurements:
    """Apply one supplied operator while copying every unaffected system exactly."""

    resulting: list[InverseAnchorSystemState] = []
    unaffected_refs: list[str] = []
    for system in value.systems:
        if system.system_ref != value.affected_system_ref:
            resulting.append(system)
            unaffected_refs.append(system.system_ref)
            continue
        if value.operator_kind == "transfer_control":
            resulting.append(
                system.model_copy(
                    update={"active_anchor_ref": value.requested_anchor_ref}
                )
            )
            continue
        anchor_positions = tuple(
            (
                anchor_ref,
                value.destination if anchor_ref == value.requested_anchor_ref else position,
            )
            for anchor_ref, position in system.anchor_positions
        )
        resulting.append(
            system.model_copy(
                update={
                    "anchor_positions": anchor_positions,
                    "dependent_position": value.supplied_dependent_position,
                    "pickup_refs": value.supplied_pickup_refs,
                    "wires": value.supplied_wires,
                }
            )
        )
    return InverseAnchorSuccessorMeasurements(
        resulting_systems=tuple(resulting),
        affected_system_ref=value.affected_system_ref,
        unaffected_system_refs=tuple(unaffected_refs),
        operator_kind=value.operator_kind,
    )


def measure_inverse_anchor_waypoint_candidates(
    value: InverseAnchorWaypointCandidatesInput,
) -> InverseAnchorWaypointCandidatesMeasurements:
    """Partition at most three supplied corridor witnesses without choosing one."""

    compatible_refs: list[str] = []
    uncertain_refs: list[str] = []
    rejected_refs: list[str] = []
    measurements: list[InverseAnchorWaypointCandidateMeasurement] = []
    for candidate in value.candidates:
        missing = tuple(
            sorted(
                set(candidate.required_corridor_constraint_refs)
                - set(candidate.satisfied_corridor_constraint_refs)
            )
        )
        measurements.append(
            InverseAnchorWaypointCandidateMeasurement(
                candidate_ref=candidate.candidate_ref,
                missing_corridor_constraint_refs=missing,
                transition_path_status=candidate.transition_path_status,
                configuration_complete=candidate.configuration_complete,
            )
        )
        if (
            missing
            or not candidate.configuration_complete
            or candidate.transition_path_status == "blocked"
        ):
            rejected_refs.append(candidate.candidate_ref)
        elif candidate.transition_path_status == "unknown":
            uncertain_refs.append(candidate.candidate_ref)
        else:
            compatible_refs.append(candidate.candidate_ref)
    return InverseAnchorWaypointCandidatesMeasurements(
        compatible_candidate_refs=tuple(compatible_refs),
        uncertain_candidate_refs=tuple(uncertain_refs),
        rejected_candidate_refs=tuple(rejected_refs),
        candidate_measurements=tuple(measurements),
        candidate_bound_respected=True,
    )
