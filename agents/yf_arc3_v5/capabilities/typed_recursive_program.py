"""Pure bounded measurements for typed resources, edits, calls, and output prefixes."""

from __future__ import annotations

from agents.yf_arc3_v5.capabilities.contracts import (
    TypedAssignmentDomainsInput,
    TypedAssignmentDomainsMeasurements,
    TypedEditSuccessorInput,
    TypedEditSuccessorMeasurements,
    TypedEmissionProvenance,
    TypedInterpreterInput,
    TypedInterpreterMeasurements,
    TypedResourceLocation,
    TypedSocketDomain,
)


def measure_typed_assignment_domains(
    value: TypedAssignmentDomainsInput,
) -> TypedAssignmentDomainsMeasurements:
    """Expose local typed domains without composing or selecting an assignment."""

    known_definitions = set(value.definition_refs)
    incompatible_calls = tuple(
        item.token_ref
        for item in value.tokens
        if item.instruction_kind == "call"
        and item.target_definition_ref not in known_definitions
    )
    incompatible_call_set = set(incompatible_calls)
    domains: list[TypedSocketDomain] = []
    deficits: list[str] = []
    for socket in value.sockets:
        compatible = tuple(
            token.token_ref
            for token in value.tokens
            if token.instruction_kind in socket.accepted_instruction_kinds
            and token.token_ref not in incompatible_call_set
        )[: value.maximum_domain_members]
        domains.append(
            TypedSocketDomain(
                socket_ref=socket.socket_ref,
                compatible_token_refs=compatible,
            )
        )
        if not compatible:
            deficits.append(socket.socket_ref)
    return TypedAssignmentDomainsMeasurements(
        socket_domains=tuple(domains),
        deficit_socket_refs=tuple(deficits),
        incompatible_call_token_refs=incompatible_calls,
        physical_token_count=len(value.tokens),
    )


def measure_typed_edit_successor(
    value: TypedEditSuccessorInput,
) -> TypedEditSuccessorMeasurements:
    """Apply exactly one supplied token/socket edit while conserving occurrences."""

    if value.edit_kind == "token_click":
        next_selection = (
            None if value.selection_token_ref == value.edit_ref else value.edit_ref
        )
        return TypedEditSuccessorMeasurements(
            status="selection_changed",
            resulting_locations=value.locations,
            resulting_sockets=value.sockets,
            selection_token_ref=next_selection,
        )

    if value.selection_token_ref is None:
        return TypedEditSuccessorMeasurements(
            status="unchanged_without_selection",
            resulting_locations=value.locations,
            resulting_sockets=value.sockets,
        )

    locations = {item.token_ref: item.location_ref for item in value.locations}
    sockets = {item.socket_ref: item for item in value.sockets}
    moving_ref = value.selection_token_ref
    source_ref = locations[moving_ref]
    destination = sockets[value.edit_ref]
    displaced_ref = destination.occupant_token_ref

    if source_ref in sockets and sockets[source_ref].occupant_token_ref == moving_ref:
        sockets[source_ref] = sockets[source_ref].model_copy(
            update={"occupant_token_ref": None}
        )
    locations[moving_ref] = value.edit_ref
    if displaced_ref is not None and displaced_ref != moving_ref:
        locations[displaced_ref] = source_ref
    sockets[value.edit_ref] = destination.model_copy(
        update={"occupant_token_ref": moving_ref}
    )

    return TypedEditSuccessorMeasurements(
        status="exchanged" if displaced_ref is not None and displaced_ref != moving_ref else "transferred",
        resulting_locations=tuple(
            TypedResourceLocation(token_ref=item.token_ref, location_ref=locations[item.token_ref])
            for item in value.locations
        ),
        resulting_sockets=tuple(sockets[item.socket_ref] for item in value.sockets),
        selection_token_ref=None,
        displaced_token_ref=(
            displaced_ref if displaced_ref is not None and displaced_ref != moving_ref else None
        ),
    )


def measure_typed_interpreter(
    value: TypedInterpreterInput,
) -> TypedInterpreterMeasurements:
    """Interpret one supplied typed definition graph within independent hard bounds."""

    tokens = {item.token_ref: item for item in value.tokens}
    definitions = {item.definition_ref: item for item in value.definitions}
    definition_ref = value.entry_definition_ref
    instruction_index = 0
    output: list[str] = []
    emissions: list[TypedEmissionProvenance] = []
    returns: list[tuple[str, int]] = []
    call_sites: list[tuple[str, int]] = []
    active_entries: set[tuple[str, int]] = {(definition_ref, 0)}
    steps = 0
    calls = 0

    def result(
        status: str,
        *,
        mismatch: int | None = None,
        terminated: bool = False,
    ) -> TypedInterpreterMeasurements:
        prefix_match = mismatch is None and tuple(output) == value.reference_values[: len(output)]
        reached_reference = len(output) >= len(value.reference_values)
        accepted_boundary = (
            reached_reference
            if value.boundary_kind == "finite_prefix"
            else terminated and tuple(output) == value.reference_values
        )
        return TypedInterpreterMeasurements(
            status=status,
            emitted_values=tuple(output),
            emissions=tuple(emissions),
            first_mismatch_index=mismatch,
            pending_returns=tuple(returns),
            prefix_match=prefix_match,
            normal_termination=terminated,
            official_acceptance_observed=(
                value.official_acceptance_evidence and accepted_boundary
            ),
            steps_executed=steps,
            calls_executed=calls,
        )

    while True:
        if value.boundary_kind == "finite_prefix" and len(output) == len(
            value.reference_values
        ):
            return result("matched_finite_prefix")
        if steps >= value.maximum_steps or len(output) >= value.maximum_output:
            return result("budget_reached")

        definition = definitions[definition_ref]
        if instruction_index >= len(definition.instruction_token_refs):
            active_entries.discard((definition_ref, len(output)))
            if not returns:
                exact = tuple(output) == value.reference_values
                return result(
                    "normal_termination" if exact else "mismatch_localized",
                    mismatch=None if exact else len(output),
                    terminated=True,
                )
            definition_ref, instruction_index = returns.pop()
            call_sites.pop()
            continue

        token_ref = definition.instruction_token_refs[instruction_index]
        token = tokens[token_ref]
        steps += 1
        if token.instruction_kind == "literal":
            output_index = len(output)
            literal_value = token.value_ref
            assert literal_value is not None
            output.append(literal_value)
            emissions.append(
                TypedEmissionProvenance(
                    output_index=output_index,
                    value_ref=literal_value,
                    literal_token_ref=token_ref,
                    definition_ref=definition_ref,
                    instruction_index=instruction_index,
                    call_site_chain=tuple(call_sites),
                )
            )
            instruction_index += 1
            if (
                output_index >= len(value.reference_values)
                or literal_value != value.reference_values[output_index]
            ):
                return result("mismatch_localized", mismatch=output_index)
            continue

        if calls >= value.maximum_calls:
            return result("call_bound_reached")
        if len(returns) >= value.maximum_stack_depth:
            return result("stack_bound_reached")
        target_ref = token.target_definition_ref
        assert target_ref is not None
        target_entry = (target_ref, len(output))
        if target_entry in active_entries:
            return result("nonproductive_cycle")
        calls += 1
        returns.append((definition_ref, instruction_index + 1))
        call_sites.append((definition_ref, instruction_index))
        active_entries.add(target_entry)
        definition_ref = target_ref
        instruction_index = 0
