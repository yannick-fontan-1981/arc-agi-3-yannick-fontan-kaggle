"""Exact joins from an observed result to its frozen action and earlier geometry."""

from agents.yf_arc3_v5.logos.operations import ActionIntent, EnvironmentActionDispatchedRecord
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest


def bind_executed_before_observation(*, repository, executed, before_ref, after_ref, profile):
    """Join the frozen winning action to its actual predecessor in a series."""
    if repository is None or executed is None:
        return None
    if profile.get("before_observation_binding") != "actual_executed_action_frame":
        return None
    try:
        start, after = repository.get(before_ref), repository.get(after_ref)
        inputs = repository.bounded_values(profile["maximum_history_inputs"])
    except (KeyError, ValueError):
        return None
    ordinal_field = profile["history_ordinal_field"]
    end = after.metadata.get(ordinal_field)
    initial = start.metadata.get(ordinal_field)
    if type(end) is not int or type(initial) is not int or not 0 <= initial < end:
        return None
    scope_fields = profile["history_scope_fields"]
    scope = tuple(start.metadata.get(key) for key in scope_fields)
    if tuple(after.metadata.get(key) for key in scope_fields) != scope:
        return None
    candidates = tuple(value for value in inputs
        if value.metadata.get(ordinal_field) == end - 1
        and tuple(value.metadata.get(key) for key in scope_fields) == scope)
    if len(candidates) != 1:
        return None
    before, = candidates
    if (before.frame_id != executed.get("frame_id")
            or before.score != start.score or after.score <= before.score):
        return None
    return before


def measure_executed_action_contract(*, store, after_ref, profile):
    """Project the actual dispatched intent, never a reconstructed explanation."""
    if store is None or after_ref is None:
        return None
    records = store.artifacts_of_type(EnvironmentActionDispatchedRecord)
    if len(records) > profile["maximum_history_inputs"]:
        return None
    matches = tuple(record for record in records if record.raw_input_ref == after_ref)
    if len(matches) != 1:
        return None
    record, = matches
    intent = store.artifact(record.action_intent_id)
    if not isinstance(intent, ActionIntent) or intent.action_ref != record.action_ref:
        return None
    fields = profile["retained_action_fields"]
    if len(fields) > profile["maximum_action_fields"]:
        raise ValueError("terminal executed contract field bound exceeded")
    facts = {field: getattr(intent, field) for field in fields}
    for _, value in sorted(facts.items()):
        items = value if isinstance(value, tuple) else (value,)
        if len(items) > profile["maximum_action_references"] or any(
            item is not None and (type(item) not in (str, int, bool)
                or (isinstance(item, str) and len(item) > profile["maximum_reference_length"]))
            for item in items):
            raise ValueError("terminal executed contract scalar bound exceeded")
    facts.update({"action_intent_id": intent.id, "dispatch_record_ref": record.id,
        "action_release_permit_id": record.action_release_permit_id,
        "after_raw_input_ref": record.raw_input_ref,
        "frozen_intent_digest": stable_digest(intent.model_dump(mode="json"))})
    return FrozenMap(facts)


def measure_executed_support_records(*, snapshot, store, repository, action_ref,
        context_epoch, transition_ref, profile):
    """Re-read complete observed supports cited by the actual executed contract.

    No source is preferred over another. Every admitted pair/translation join is
    returned; current geometry and appearance associations are checked by the
    caller. Historical raster coordinates are evidence, never new-scene binds.
    """
    if store is None or repository is None:
        return ()
    source = profile["executed_support_source"]
    terms = snapshot.terms_for_projection_contract(source["outcome_contract"])
    links = snapshot.claims_for_predicate(source["outcome_predicate"])
    dispatched = store.artifacts_of_type(EnvironmentActionDispatchedRecord)
    if max(len(terms), len(links)) > profile["maximum_records"] or len(dispatched) > source["maximum_history_inputs"]:
        return ()

    def admitted(claim):
        return (claim is not None and claim.epistemic_status.value in profile["proof_statuses"]
            and claim.disposition.value == profile["proof_disposition"]
            and claim.polarity.value == profile["proof_polarity"]
            and not claim.active_contradictions and not claim.defeated_by)

    from agents.yf_arc3_v5.capabilities import ComponentExtractionInput, FrameGrid, detect_components
    cache = {}

    def components(value):
        if value.raw_input_ref not in cache:
            if len(value.frame) * len(value.frame[0]) > profile["maximum_pixels"]:
                return ()
            result = detect_components(ComponentExtractionInput(frame=FrameGrid(rows=value.frame),
                connectivity=profile["component_connectivity"])).components
            if len(result) > profile["maximum_components"]:
                return ()
            cache[value.raw_input_ref] = result
        return cache[value.raw_input_ref]

    records = []
    for term in terms:
        attrs = term.attributes
        if attrs.get("transition_ref") != transition_ref:
            continue
        proofs = tuple(link for link in links if admitted(link) and link.proof_rule == source["outcome_rule"]
            and link.arguments == (term.id, transition_ref))
        contract = attrs.get("executed_action_contract")
        history = attrs.get("observation_history")
        if not proofs or not contract or not history or contract.get("action_ref") != action_ref:
            continue
        # The retained projection must still match the immutable dispatched
        # intent. Never silently replace it with a newly reconstructed why.
        intent = store.artifact(contract.get("action_intent_id"))
        if not isinstance(intent, ActionIntent) or stable_digest(intent.model_dump(mode="json")) != contract.get("frozen_intent_digest"):
            continue
        refs = contract.get("canonical_premise_claim_refs", ())
        if len(refs) > source["maximum_premises"] or not set(refs) <= set(intent.premise_claim_refs):
            continue
        claims = tuple(snapshot.claim(ref) for ref in refs)
        pairs = tuple(claim for claim in claims if admitted(claim)
            and claim.predicate == source["pair_predicate"] and claim.proof_rule in source["pair_rules"])
        effects = tuple(claim for claim in claims if admitted(claim)
            and claim.predicate == source["translation_predicate"] and claim.proof_rule in source["translation_rules"]
            and claim.arguments and claim.arguments[0] == action_ref
            and claim.attributes.get("context_epoch") == context_epoch)
        observations = history.get("prior_observation_rows", ())
        if len(observations) > source["maximum_history_inputs"]:
            continue
        by_frame = {row["frame_id"]: row for row in observations}
        by_step = {row["step_index"]: row for row in observations}
        for pair in pairs:
            row = by_frame.get(pair.attributes.get("observation_ref"))
            prior = by_step.get(row["step_index"] - 1) if row else None
            if not row or not prior:
                continue
            source_dispatch = tuple(record for record in dispatched if record.raw_input_ref == row["raw_input_ref"])
            if len(source_dispatch) != 1 or source_dispatch[0].action_ref != action_ref:
                continue
            try:
                observed, previous = repository.get(row["raw_input_ref"]), repository.get(prior["raw_input_ref"])
            except KeyError:
                continue
            current = {component.component_id: component for component in components(observed)}
            previous_components = components(previous)
            for mover_fields, fixed_fields in source["endpoint_orientations"]:
                mover_ref, fixed_ref = pair.attributes.get(mover_fields[0]), pair.attributes.get(fixed_fields[0])
                mover, fixed = current.get(pair.attributes.get(mover_fields[1])), current.get(pair.attributes.get(fixed_fields[1]))
                if not mover_ref or not fixed_ref or mover is None or fixed is None or mover.value == fixed.value:
                    continue
                if mover.area + fixed.area > profile["maximum_pixels"]:
                    continue
                for effect in effects:
                    dr, dc = effect.attributes.get("delta_row"), effect.attributes.get("delta_col")
                    if mover_ref not in effect.grounds or type(dr) is not int or type(dc) is not int or dr == dc == 0:
                        continue
                    prior_pixels = frozenset((r - dr, c - dc) for r, c in mover.pixels)
                    if not any(component.value == mover.value and frozenset(component.pixels) == prior_pixels
                            for component in previous_components):
                        continue
                    if not any(component.value == fixed.value and component.pixels == fixed.pixels
                            for component in previous_components):
                        continue
                    layers = tuple(FrozenMap({"prior_entity_ref": ref,
                        "component_geometry": FrozenMap(component.model_dump(mode="json")),
                        "prediction_delta_row": dy, "prediction_delta_col": dx,
                        "last_observed_delta_row": dy, "last_observed_delta_col": dx})
                        for component, ref, dy, dx in ((mover, mover_ref, dr, dc), (fixed, fixed_ref, 0, 0)))
                    records.append(FrozenMap({"geometry_evidence_ref": term.id,
                        "layer_rows": layers, "observation_history": history,
                        "source_frame_ref": observed.frame_id,
                        "proof_refs": tuple(sorted({pair.id, effect.id} | {link.id for link in proofs}))}))
                    if len(records) > profile["maximum_rows"]:
                        raise ValueError("terminal executed support join bound exceeded")
    return tuple(records)
