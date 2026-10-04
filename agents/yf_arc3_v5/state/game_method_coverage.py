"""Bounded exact joins for Source-declared current activity requirements.

No activity, role, mechanism, priority or eligibility is invented here. The
canonical store and DRM provide every reference and comparison constraint.
Incomplete rows retain the exact missing requirements for the calling SRC.
"""

from collections.abc import Mapping

from agents.yf_arc3_v5.dynamic_workflow.contracts import GameMethodActivityContract
from agents.yf_arc3_v5.logos.claims import Claim
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest
from agents.yf_arc3_v5.state.reference_bindings import canonical_reference_bindings_current


def _claim_matches(snapshot, claim: Claim | None, profile: Mapping) -> bool:
    return bool(
        claim is not None
        and any(item.id == claim.id for item in snapshot.claims_for_predicate(claim.predicate))
        and claim.epistemic_status.value in profile["allowed_link_statuses"]
        and claim.disposition.value == profile["link_disposition"]
        and claim.polarity.value == profile["link_polarity"]
        and not (profile["no_contradictions"] and claim.active_contradictions)
        and not (profile["no_defeaters"] and claim.defeated_by)
    )


def _measure_current_dependency_refs(*, snapshot, rows, contract_variants, dependency_contracts, profile):
    """Exact retained-requirement joins; Source declares contracts and bounds.

    This does not choose an activity order or synthesize a transitive graph.
    A remembered dependency is attached only when both current measurements are
    complete and its canonical evidence still matches the current Source law.
    It never requests execution of a provider for an already satisfied input.
    """
    links = snapshot.claims_for_predicate(profile["causal_dependency_predicate"])
    if len(links) > profile["max_causal_dependency_links"]:
        raise ValueError("current method causal dependency link bound exceeded")
    if not links:
        return tuple(FrozenMap.overlay(FrozenMap({"causal_dependency_refs": ()}), row) for row in rows)
    contracts_by_field = {field: contract for field, contract, _ in contract_variants}
    rows_by_candidate = {}
    for row in rows:
        rows_by_candidate.setdefault(row["candidate_ref"], []).append(row)
    checks = 0
    result = []
    for row in rows:
        terms = {item["term_ref"]: item for item in row["canonical_term_dependency_revisions"]}
        claims = set(row["row_evidence_refs"])
        dependencies = set()
        consumer = GameMethodActivityContract.model_validate(snapshot.term(row["activity_term_ref"]).attributes["activity_contract"])
        for declaration in dependency_contracts:
            if contracts_by_field[declaration["consumer_contract_field"]] != consumer:
                continue
            provider = contracts_by_field[declaration["provider_contract_field"]]
            predicate = declaration["predicate_ref"]
            if (predicate not in provider.provides_predicate_refs
                    or predicate not in consumer.requires_predicate_refs
                    or not all(ref in provider.preserves_invariant_refs and ref in consumer.preserves_invariant_refs
                               for ref in declaration["preserved_invariant_refs"])):
                continue
            for current_provider in rows_by_candidate[row["candidate_ref"]]:
                if not all(current_provider[field] for field in (
                    "all_roles_have_rows", "all_mechanisms_have_rows", "all_predicates_true", "all_invariants_true",
                )):
                    continue
                provider_term = snapshot.term(current_provider["activity_term_ref"])
                if GameMethodActivityContract.model_validate(provider_term.attributes["activity_contract"]) != provider:
                    continue
                for link in links:
                    checks += 1
                    if checks > profile["max_causal_dependency_joins"]:
                        raise ValueError("current method causal dependency join bound exceeded")
                    if (link.arguments != (provider_term.id, row["activity_term_ref"], predicate)
                            or link.proof_rule != profile["causal_dependency_rule"]
                            or not _claim_matches(snapshot, link, profile)
                            or not link.grounds
                            or not all(_claim_matches(snapshot, snapshot.claim(ref), profile) for ref in link.grounds)
                            or link.attributes.get("method_ref") != row["method_ref"]
                            or link.attributes.get("composition_principle_ref") != declaration["composition_principle_ref"]
                            or link.attributes.get("composition_law_hash") != declaration["composition_law_hash"]
                            or link.attributes.get("preserved_invariant_refs") != declaration["preserved_invariant_refs"]
                            or link.attributes.get("falsifier_ref") != declaration["falsifier_ref"]
                            or link.attributes.get("dependency_kind") != profile["retained_dependency_kind"]
                            or link.attributes.get("requires_current_unsatisfied_predicate_before_execution") != profile["dependency_requires_absence_before_execution"]
                            or link.attributes.get("total_order_is_not_inferred") is not True):
                        continue
                    dependencies.add(link.id)
                    claims.update((link.id, *link.grounds, *current_provider["row_evidence_refs"]))
                    terms.update((item["term_ref"], item) for item in current_provider["canonical_term_dependency_revisions"])
        if len(terms) + len(claims) > profile["max_dependency_refs"]:
            raise ValueError("current method causal dependency reference bound exceeded")
        if not dependencies:
            result.append(FrozenMap.overlay(FrozenMap({"causal_dependency_refs": ()}), row))
            continue
        claim_digests = {item["claim_ref"]: item for item in row["canonical_claim_dependency_digests"]}
        for ref in sorted(claims):
            if ref not in claim_digests:
                claim_digests[ref] = FrozenMap({"claim_ref": ref, "content_digest": stable_digest(snapshot.claim(ref))})
        result.append(FrozenMap.overlay(FrozenMap({
            "causal_dependency_refs": tuple(sorted(dependencies)),
            "row_evidence_refs": tuple(sorted(claims)),
            "canonical_term_dependency_revisions": tuple(terms[ref] for ref in sorted(terms)),
            "canonical_claim_dependency_digests": tuple(claim_digests[ref] for ref in sorted(claims)),
        }), row))
    return tuple(result)


def measure_current_method_coverage(
    *, snapshot, tracking, alternative_facts: Mapping, mechanism_inputs: tuple,
    profile: Mapping, declared_contract: GameMethodActivityContract,
    contract_source_hash: str,
    component_attestations: tuple[FrozenMap, ...],
    additional_contracts: tuple = (),
    dependency_contracts: tuple = (),
) -> FrozenMap:
    contract_variants = ((profile["contract_field"], declared_contract, component_attestations),) + additional_contracts
    if len(contract_variants) > profile["max_contract_variants"]:
        raise ValueError("current method contract variant bound exceeded; no variants dropped")
    activities = snapshot.terms_for_projection_contract(profile["activity_term_contract"])
    bindings = tuple(item for item in snapshot.terms_for_projection_contract(profile["role_term_contract"])
                     if item.attributes.get("operational_scene_ref") == tracking.frame_ref)
    if (len(activities) > profile["max_activities"]
            or len(bindings) > profile["max_role_bindings"]
            or len(mechanism_inputs) > profile["max_mechanism_inputs"]
            or len(alternative_facts) > profile["max_candidates"]):
        raise ValueError("current method coverage input bound exceeded; no alternatives dropped")
    current_entities = {item.entity_id: item for item in tracking.entities}
    current_bindings = tuple(
        item for item in bindings
        if canonical_reference_bindings_current(snapshot, item, frame_ref=tracking.frame_ref)
        and item.attributes.get("current_member_entity_refs")
        and all(ref in current_entities and current_entities[ref].current_frame_ref == tracking.frame_ref
                for ref in item.attributes["current_member_entity_refs"])
    )
    method_links = snapshot.claims_for_predicate(profile["activity_link_predicate"])
    if len(method_links) > profile["max_activity_links"]:
        raise ValueError("current method proof-link bound exceeded; no proofs dropped")
    input_pairs = frozenset((item.actuator_entity_ref, item.evidence_ref) for item in mechanism_inputs)
    historical_binding = profile["historical_bearer_binding"]
    join_checks = 0
    rows = []
    evidence_refs = set()
    for activity in sorted(activities, key=lambda item: item.id):
        try:
            contract = GameMethodActivityContract.model_validate(activity.attributes["activity_contract"])
        except (KeyError, TypeError, ValueError):
            continue
        matching_variants = tuple(item for item in contract_variants if item[1] == contract)
        if len(matching_variants) != 1 or activity.attributes.get("contract_source_hash") != contract_source_hash:
            continue
        contract_field, source_contract, source_attestations = matching_variants[0]
        if (not source_attestations
                or activity.attributes.get("component_source_attestations") != source_attestations):
            continue
        links = tuple(link for link in method_links
                      if link.arguments == (contract.method_ref, activity.id)
                      and link.proof_rule == profile["activity_link_rule"]
                      and _claim_matches(snapshot, link, profile)
                      and (not profile["require_activity_grounds_current"] or all(_claim_matches(snapshot, snapshot.claim(ref), profile) for ref in link.grounds)))
        proof_refs = set()
        for link in links:
            for ref in link.grounds:
                proof = snapshot.claim(ref)
                if (_claim_matches(snapshot, proof, profile)
                        and proof.predicate == profile["completion_link_predicate"]
                        and proof.proof_rule == profile["completion_link_rule"]):
                    proof_refs.update((link.id, proof.id))
        if not proof_refs:
            continue
        for candidate_ref, facts in sorted(alternative_facts.items()):
            # Compare typed contracts, not serialization shape. Source may
            # omit a declared optional default which stored contracts include.
            # Invalid/extra fields and nondefault differences still fail closed.
            try:
                candidate_contract = GameMethodActivityContract.model_validate(facts.get(contract_field))
            except (TypeError, ValueError):
                continue
            if candidate_contract != source_contract:
                continue
            inherited = frozenset(facts.get(profile["candidate_mechanism_evidence_field"], ()))
            role_rows = []
            mechanism_refs_with_rows = set()
            role_refs_with_rows = set()
            term_refs = {activity.id}
            claim_refs = set(proof_refs)
            for binding in current_bindings:
                role_ref = binding.attributes.get("role_ref")
                if role_ref not in contract.required_role_refs:
                    continue
                association = snapshot.term(binding.attributes["association_ref"])
                if association is None:
                    continue
                source_claims = tuple(snapshot.claim(ref) for ref in association.attributes.get("source_claim_refs", ()))
                if len(source_claims) > profile["max_dependency_refs"]:
                    raise ValueError("current method source-proof bound exceeded")
                matched_inputs = []
                for source in source_claims:
                    if not _claim_matches(snapshot, source, profile):
                        continue
                    reference = source.proof_rule
                    evidence_ref = source.attributes.get(profile["source_transition_attribute"])
                    if (reference not in contract.required_mechanism_refs
                            or reference not in binding.attributes.get("mechanism_schema_refs", ())
                            or evidence_ref not in inherited):
                        continue
                    argument_index = historical_binding["source_bearer_argument_index"]
                    role_argument_index = historical_binding["source_role_argument_index"]
                    historical_pair_present = (
                        source.predicate == historical_binding["source_role_predicate"]
                        and len(source.arguments) > max(argument_index, role_argument_index)
                        and source.arguments[role_argument_index] == historical_binding["source_role_argument_by_role"].get(role_ref)
                        and (source.arguments[argument_index], evidence_ref) in input_pairs
                    )
                    for entity_ref in binding.attributes["current_member_entity_refs"]:
                        join_checks += 1
                        if join_checks > profile["max_join_checks"]:
                            raise ValueError("current method exact-join bound exceeded")
                        if (entity_ref, evidence_ref) in input_pairs or historical_pair_present:
                            matched_inputs.append(FrozenMap({
                                "mechanism_schema_ref": reference, "evidence_ref": evidence_ref,
                                "current_entity_ref": entity_ref, "source_claim_ref": source.id,
                            }))
                            mechanism_refs_with_rows.add(reference)
                            claim_refs.add(source.id)
                role_refs_with_rows.add(role_ref)
                term_refs.update((binding.id, association.id))
                claim_refs.update(binding.attributes["role_premise_claim_refs"])
                claim_refs.update(item["claim_ref"] for item in binding.attributes.get("canonical_claim_dependency_digests", ()))
                role_rows.append(FrozenMap({
                    "role_ref": role_ref, "binding_ref": binding.id,
                    "current_bearer_ref": binding.attributes["bearer_ref"],
                    "current_member_entity_refs": binding.attributes["current_member_entity_refs"],
                    "mechanism_bindings": tuple(sorted(set(matched_inputs), key=stable_digest)),
                }))
            if len(term_refs) + len(claim_refs) > profile["max_dependency_refs"]:
                raise ValueError("current method dependency bound exceeded; no ties dropped")
            absent_roles = tuple(ref for ref in contract.required_role_refs if ref not in role_refs_with_rows)
            absent_mechanisms = tuple(ref for ref in contract.required_mechanism_refs if ref not in mechanism_refs_with_rows)
            non_true_predicates = tuple(ref for ref in contract.requires_predicate_refs if facts.get(ref) is not True)
            non_true_invariants = tuple(ref for ref in contract.preserves_invariant_refs if facts.get(ref) is not True)
            row = FrozenMap({
                "frame_ref": tracking.frame_ref, "candidate_ref": candidate_ref,
                "method_ref": contract.method_ref, "activity_term_ref": activity.id,
                "role_binding_rows": tuple(role_rows), "falsifier_refs": contract.falsifier_refs,
                "missing_role_refs": absent_roles, "missing_mechanism_refs": absent_mechanisms,
                "non_true_predicate_refs": non_true_predicates, "non_true_invariant_refs": non_true_invariants,
                "all_roles_have_rows": not absent_roles, "all_mechanisms_have_rows": not absent_mechanisms,
                "all_predicates_true": not non_true_predicates, "all_invariants_true": not non_true_invariants,
                "row_evidence_refs": tuple(sorted(claim_refs)),
                "canonical_term_dependency_revisions": tuple(FrozenMap({"term_ref": ref, "revision": snapshot.term(ref).last_changed_state_revision}) for ref in sorted(term_refs)),
                "canonical_claim_dependency_digests": tuple(FrozenMap({"claim_ref": ref, "content_digest": stable_digest(snapshot.claim(ref))}) for ref in sorted(claim_refs)),
            })
            rows.append(row)
            if len(rows) > profile["max_rows"]:
                raise ValueError("current method coverage row bound exceeded; no rows dropped")
    rows = _measure_current_dependency_refs(
        snapshot=snapshot, rows=rows, contract_variants=contract_variants,
        dependency_contracts=dependency_contracts, profile=profile,
    )
    bound_rows = []
    for row in rows:
        token = stable_digest(tuple((field, row[field]) for field in profile["binding_token_fields"]))
        binding_claim_ref = profile["binding_claim_id_pattern"].format(binding_token=token)
        binding_term_ref = profile["binding_term_id_pattern"].format(binding_token=token)
        bound_rows.append(FrozenMap.overlay(FrozenMap({
            "binding_term_ref": binding_term_ref,
            "binding_claim_ref": binding_claim_ref,
            "binding_premise_claim_refs": (binding_claim_ref,),
            "binding_pair_absent": snapshot.term(binding_term_ref) is None or snapshot.claim(binding_claim_ref) is None,
        }), row))
        evidence_refs.update(row["row_evidence_refs"])
    rows = tuple(bound_rows)
    return FrozenMap({
        "coverage_rows": tuple(rows),
        "has_complete_rows": any(all(row[field] for field in (
            "all_roles_have_rows", "all_mechanisms_have_rows", "all_predicates_true", "all_invariants_true",
        )) for row in rows),
        "evidence_refs": tuple(sorted(evidence_refs)), "state_revision": snapshot.revision,
        "has_new_complete_rows": any(row["binding_pair_absent"] and all(row[field] for field in (
            "all_roles_have_rows", "all_mechanisms_have_rows", "all_predicates_true", "all_invariants_true",
        )) for row in rows),
    })
