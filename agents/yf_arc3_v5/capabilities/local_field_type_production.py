"""Pure bounded measurements for conditional local-field type production."""

from __future__ import annotations

from agents.yf_arc3_v5.capabilities.contracts import (
    LocalFieldAcceptanceWindow,
    LocalFieldAcceptanceWindowInput,
    LocalFieldAcceptanceWindowMeasurements,
    LocalFieldAutonomousScheduleMeasurement,
    LocalFieldAutonomousSuccessorInput,
    LocalFieldAutonomousSuccessorMeasurements,
    LocalFieldCandidateMeasurement,
    LocalFieldSelectiveDomainInput,
    LocalFieldSelectiveDomainMeasurements,
    LocalFieldTransactionInput,
    LocalFieldTransactionMeasurements,
    LocalFieldTransactionQuantityDelta,
    LocalFieldTypeTransition,
)


def measure_local_field_selective_domains(
    value: LocalFieldSelectiveDomainInput,
) -> LocalFieldSelectiveDomainMeasurements:
    """Classify supplied finite click candidates by exact set relations."""

    required = set(value.required_body_refs)
    excluded = set(value.excluded_body_refs)
    exact_refs: list[str] = []
    uncertain_refs: list[str] = []
    rejected_refs: list[str] = []
    rows: list[LocalFieldCandidateMeasurement] = []
    for candidate in value.candidates:
        strict = set(candidate.strictly_influenced_body_refs)
        boundary = set(candidate.boundary_body_refs)
        missing = tuple(sorted(required - strict - boundary))
        strict_excluded = tuple(sorted(excluded & strict))
        boundary_required = tuple(sorted(required & boundary))
        boundary_excluded = tuple(sorted(excluded & boundary))
        rows.append(
            LocalFieldCandidateMeasurement(
                candidate_ref=candidate.candidate_ref,
                position=candidate.position,
                missing_required_body_refs=missing,
                strictly_included_excluded_body_refs=strict_excluded,
                boundary_required_body_refs=boundary_required,
                boundary_excluded_body_refs=boundary_excluded,
            )
        )
        if missing or strict_excluded:
            rejected_refs.append(candidate.candidate_ref)
        elif boundary_required or boundary_excluded:
            uncertain_refs.append(candidate.candidate_ref)
        else:
            exact_refs.append(candidate.candidate_ref)
    return LocalFieldSelectiveDomainMeasurements(
        exact_candidate_refs=tuple(exact_refs),
        boundary_uncertain_candidate_refs=tuple(uncertain_refs),
        rejected_candidate_refs=tuple(rejected_refs),
        candidate_measurements=tuple(rows),
    )


def measure_local_field_autonomous_successors(
    value: LocalFieldAutonomousSuccessorInput,
) -> LocalFieldAutonomousSuccessorMeasurements:
    """Project each supplied complete schedule once without choosing an order."""

    occurrence_by_ref = {item.occurrence_ref: item for item in value.occurrences}
    duplicate_refs = tuple(sorted(value.already_advanced_occurrence_refs))
    schedules: list[LocalFieldAutonomousScheduleMeasurement] = []
    if not duplicate_refs:
        for schedule in value.schedules:
            positions: dict[str, tuple[int, int]] = {}
            for occurrence_ref in schedule.ordered_occurrence_refs:
                occurrence = occurrence_by_ref[occurrence_ref]
                positions[occurrence.body_ref] = occurrence.after_position
            schedules.append(
                LocalFieldAutonomousScheduleMeasurement(
                    schedule_ref=schedule.schedule_ref,
                    ordered_occurrence_refs=schedule.ordered_occurrence_refs,
                    resulting_body_positions=tuple(sorted(positions.items())),
                )
            )
    return LocalFieldAutonomousSuccessorMeasurements(
        schedule_measurements=tuple(schedules),
        event_refs=tuple(sorted({item.event_ref for item in value.occurrences})),
        duplicate_advancement_refs=duplicate_refs,
        alternative_schedule_count=max(0, len(schedules) - 1),
    )


def measure_local_field_acceptance_windows(
    value: LocalFieldAcceptanceWindowInput,
) -> LocalFieldAcceptanceWindowMeasurements:
    """Measure joint predicate intervals under caller-supplied acceptance rules."""

    required = set(value.required_predicate_refs)
    satisfied_refs: list[str] = []
    transient_refs: list[str] = []
    retained_refs: list[str] = []
    eligible_flags: list[bool] = []
    first_interruption: str | None = None
    interrupted = False
    for sample in value.samples:
        if sample.internal_interruption_observed and first_interruption is None:
            first_interruption = sample.sample_ref
            interrupted = True
        complete = required <= set(sample.satisfied_predicate_refs)
        surplus_ok = value.surplus_admissible or sample.surplus_body_count == 0
        shared_ok = (
            value.shared_zone_occupancy_admissible
            or sample.shared_zone_occupancy_count == 0
        )
        eligible = complete and surplus_ok and shared_ok and not interrupted
        eligible_flags.append(eligible)
        if not eligible:
            continue
        satisfied_refs.append(sample.sample_ref)
        retained = required <= set(sample.retained_predicate_refs)
        if retained:
            retained_refs.append(sample.sample_ref)
        else:
            transient_refs.append(sample.sample_ref)

    windows: list[LocalFieldAcceptanceWindow] = []
    start_index: int | None = None
    for index, eligible in enumerate((*eligible_flags, False)):
        if eligible and start_index is None:
            start_index = index
        if eligible or start_index is None:
            continue
        end_index = index - 1
        window_samples = value.samples[start_index:index]
        windows.append(
            LocalFieldAcceptanceWindow(
                start_sample_ref=value.samples[start_index].sample_ref,
                end_sample_ref=value.samples[end_index].sample_ref,
                sample_count=len(window_samples),
                retained_through_window=all(
                    required <= set(sample.retained_predicate_refs)
                    for sample in window_samples
                ),
            )
        )
        start_index = None
    return LocalFieldAcceptanceWindowMeasurements(
        windows=tuple(windows),
        jointly_satisfied_sample_refs=tuple(satisfied_refs),
        transient_only_sample_refs=tuple(transient_refs),
        retained_completion_sample_refs=tuple(retained_refs),
        first_interruption_sample_ref=first_interruption,
    )


def measure_local_field_transaction(
    value: LocalFieldTransactionInput,
) -> LocalFieldTransactionMeasurements:
    """Compare exact body, type, persistence and quantity deltas."""

    before = {item.body_ref: item for item in value.before.bodies}
    transient = {item.body_ref: item for item in value.transient.bodies}
    after_maps = tuple(
        {item.body_ref: item for item in snapshot.bodies}
        for snapshot in value.after_packet
    )
    final = after_maps[-1]
    before_refs = set(before)
    transient_refs = set(transient)
    final_refs = set(final)
    all_after_refs = set.intersection(*(set(items) for items in after_maps))
    stable_type_refs = {
        body_ref
        for body_ref in all_after_refs
        if len({items[body_ref].type_ref for items in after_maps}) == 1
    }
    restored_refs = {
        body_ref
        for body_ref in before_refs & final_refs
        if before[body_ref] == final[body_ref]
        and transient.get(body_ref) != before[body_ref]
    }
    after_quantities = {
        item.quantity_ref: item.value for item in value.quantities_after
    }
    return LocalFieldTransactionMeasurements(
        final_added_body_refs=tuple(sorted(final_refs - before_refs)),
        final_removed_body_refs=tuple(sorted(before_refs - final_refs)),
        final_type_transitions=tuple(
            LocalFieldTypeTransition(
                body_ref=body_ref,
                before_type_ref=before[body_ref].type_ref,
                after_type_ref=final[body_ref].type_ref,
            )
            for body_ref in sorted(before_refs & final_refs)
            if before[body_ref].type_ref != final[body_ref].type_ref
        ),
        transient_only_body_refs=tuple(
            sorted(transient_refs - before_refs - set.union(*(set(items) for items in after_maps)))
        ),
        restored_body_refs=tuple(sorted(restored_refs)),
        bodies_surviving_entire_after_packet=tuple(sorted(all_after_refs)),
        stable_type_body_refs=tuple(sorted(stable_type_refs)),
        quantity_deltas=tuple(
            LocalFieldTransactionQuantityDelta(
                quantity_ref=item.quantity_ref,
                before_value=item.value,
                after_value=after_quantities[item.quantity_ref],
                signed_delta=after_quantities[item.quantity_ref] - item.value,
            )
            for item in value.quantities_before
        ),
    )
