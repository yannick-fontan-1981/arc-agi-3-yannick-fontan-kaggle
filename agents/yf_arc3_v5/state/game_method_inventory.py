"""Exact bounded inventory of canonical methods and current requirement evidence.

No action, activity order, role or requirement truth is invented here. Contracts,
proof constraints, identifiers and bounds are provided by DRM; SRC owns the
consultation transaction and subsequent decision.
"""

from agents.yf_arc3_v5.dynamic_workflow.contracts import GameMethodActivityContract
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest
from agents.yf_arc3_v5.state.reference_bindings import canonical_reference_bindings_current


def measure_current_method_inventory(*, snapshot, tracking, profile):
    activities = snapshot.terms_for_projection_contract(profile["activity_term_contract"])
    current = snapshot.terms_for_literal_attribute(profile["scene_attribute"], tracking.frame_ref)
    bindings = tuple(term for term in current
        if term.attributes.get("projection_contract") == profile["role_term_contract"])
    if len(activities) > profile["max_activities"] or len(bindings) > profile["max_role_bindings"]:
        raise ValueError("current method inventory input bound exceeded; nothing dropped")
    entities = {entity.entity_id for entity in tracking.entities
        if entity.current_frame_ref == tracking.frame_ref}
    links = snapshot.claims_for_predicate(profile["activity_link_predicate"])
    criterion_links = snapshot.claims_for_predicate(profile["control_criterion_link_predicate"])
    if len(links) > profile["max_activity_links"] or len(criterion_links) > profile["max_activity_links"]:
        raise ValueError("current method inventory proof-link bound exceeded")

    predicate_ref_cache = {}

    def admitted(claim, statuses=None):
        if claim is None:
            return False
        if claim.predicate not in predicate_ref_cache:
            candidates = snapshot.claims_for_predicate(claim.predicate)
            if len(candidates) > profile["max_requirement_claims"]:
                raise ValueError("current method admitted-proof bound exceeded")
            predicate_ref_cache[claim.predicate] = frozenset(item.id for item in candidates)
        return (claim is not None
            and claim.id in predicate_ref_cache[claim.predicate]
            and claim.epistemic_status.value in (statuses or profile["allowed_link_statuses"])
            and claim.disposition.value == profile["link_disposition"]
            and claim.polarity.value == profile["link_polarity"]
            and not (profile["no_contradictions"] and claim.active_contradictions)
            and not (profile["no_defeaters"] and claim.defeated_by))

    current_bindings = tuple(binding for binding in bindings
        if canonical_reference_bindings_current(snapshot, binding, frame_ref=tracking.frame_ref)
        and binding.attributes.get("current_member_entity_refs")
        and all(ref in entities for ref in binding.attributes["current_member_entity_refs"]))
    selection_terms = tuple(term for term in current
        if term.attributes.get(profile["selection_query_attribute"]) == profile["selection_query_value"])
    if len(selection_terms) > profile["max_selection_bindings"]:
        raise ValueError("current method selection inventory bound exceeded; ties retained")
    selection_rows = []
    for term in selection_terms:
        refs = tuple(term.attributes.get(profile["selection_member_attribute"], ()))
        if not refs and term.attributes.get(profile["selection_entity_attribute"]):
            refs = (term.attributes[profile["selection_entity_attribute"]],)
        premise_refs = term.attributes.get("role_premise_claim_refs", ())
        if (refs and premise_refs and all(admitted(snapshot.claim(ref), profile["allowed_activity_premise_statuses"]) for ref in premise_refs)
            and all(ref in entities for ref in refs)
            and canonical_reference_bindings_current(snapshot, term, frame_ref=tracking.frame_ref)):
            selection_rows.append((term, refs))
    rows = []
    joins = 0
    for activity in sorted(activities, key=lambda term: term.id):
        contract = GameMethodActivityContract.model_validate(activity.attributes["activity_contract"])
        activity_links = tuple(link for link in links
            if link.arguments == (contract.method_ref, activity.id)
            and link.proof_rule == profile["activity_link_rule"] and admitted(link)
            and all(admitted(snapshot.claim(ref), profile["allowed_activity_premise_statuses"]) for ref in link.grounds))
        claim_refs = {link.id for link in activity_links}
        claim_refs.update(ref for link in activity_links for ref in link.grounds)
        term_refs = {activity.id}
        criterion_refs = tuple(activity.attributes.get("control_criterion_evidence_refs", ()))
        criterion_associations = tuple(activity.attributes.get("control_criterion_association_refs", ()))
        recorded_control_roles = tuple(activity.attributes.get("observed_control_role_refs", ()))
        if len(criterion_refs) + len(criterion_associations) + len(recorded_control_roles) > profile["max_dependency_refs"]:
            raise ValueError("current method control criterion bound exceeded")
        admitted_criterion_links = tuple(link for link in criterion_links
            if link.arguments == (activity.id,) and link.proof_rule == profile["control_criterion_link_rule"]
            and admitted(link) and set(link.grounds) == set(criterion_refs))
        criteria_present = (bool(criterion_refs) and bool(criterion_associations) and bool(admitted_criterion_links)
            and all(admitted(snapshot.claim(ref), profile["allowed_activity_premise_statuses"]) for ref in criterion_refs)
            and all(snapshot.term(ref) is not None
                and snapshot.term(ref).attributes.get(profile["control_criterion_active_attribute"]) == profile["control_criterion_active_value"]
                and snapshot.term(ref).attributes.get("role_ref") in recorded_control_roles
                and any(snapshot.claim(proof_ref).predicate == profile["control_criterion_association_predicate"]
                    and snapshot.claim(proof_ref).proof_rule == profile["control_criterion_association_rule"]
                    and snapshot.claim(proof_ref).arguments == (ref,) for proof_ref in criterion_refs)
                for ref in criterion_associations))
        control_role_refs = recorded_control_roles if criteria_present else ()
        claim_refs.update(ref for ref in criterion_refs if snapshot.claim(ref) is not None)
        claim_refs.update(link.id for link in admitted_criterion_links)
        term_refs.update(ref for ref in criterion_associations if snapshot.term(ref) is not None)
        role_rows = []
        for role_ref in dict.fromkeys((*contract.required_role_refs, *control_role_refs)):
            matches = []
            for binding in current_bindings:
                joins += 1
                if joins > profile["max_join_checks"]:
                    raise ValueError("current method inventory exact-join bound exceeded")
                if binding.attributes.get("role_ref") != role_ref:
                    continue
                matches.append(binding)
                term_refs.add(binding.id)
                term_refs.update(item["term_ref"] for item in binding.attributes.get("canonical_term_dependency_revisions", ()))
                claim_refs.update(binding.attributes.get("role_premise_claim_refs", ()))
                claim_refs.update(item["claim_ref"] for item in binding.attributes.get("canonical_claim_dependency_digests", ()))
            criterion_matches = tuple(binding for binding in matches
                if binding.attributes.get("association_ref") in criterion_associations)
            role_rows.append(FrozenMap({"role_ref": role_ref,
                "binding_refs": tuple(binding.id for binding in matches),
                "current_member_rows": tuple(binding.attributes["current_member_entity_refs"] for binding in matches),
                "unresolved_context_rows": tuple(binding.attributes.get("unresolved_context_fields", ()) for binding in matches),
                "control_criterion_binding_refs": tuple(binding.id for binding in criterion_matches),
                "control_criterion_member_rows": tuple(binding.attributes["current_member_entity_refs"] for binding in criterion_matches),
                "control_criterion_context_rows": tuple(binding.attributes.get("unresolved_context_fields", ()) for binding in criterion_matches)}))
        requirement_rows = []
        expected_rows = tuple(members for item in role_rows
            if item["role_ref"] in control_role_refs
            for members in item["control_criterion_member_rows"])
        control_roles_complete = bool(control_role_refs) and all(
            any(item["role_ref"] == ref and item["control_criterion_binding_refs"]
                and not any(item["control_criterion_context_rows"]) for item in role_rows)
            for ref in control_role_refs)
        exact_rows, disjoint_rows = [], []
        for term, members in selection_rows:
            term_refs.add(term.id)
            term_refs.update(item["term_ref"] for item in term.attributes.get("canonical_term_dependency_revisions", ()))
            claim_refs.update(term.attributes.get("role_premise_claim_refs", ()))
            claim_refs.update(item["claim_ref"] for item in term.attributes.get("canonical_claim_dependency_digests", ()))
            equal, intersects = False, False
            for expected in expected_rows:
                joins += 1
                if joins > profile["max_join_checks"]:
                    raise ValueError("current method control-context exact-join bound exceeded")
                equal = equal or set(members) == set(expected)
                intersects = intersects or bool(set(members).intersection(expected))
            exact_rows.append(equal)
            disjoint_rows.append(not intersects)
        result_rows = []
        for predicate in dict.fromkeys(contract.requires_predicate_refs + contract.preserves_invariant_refs + contract.provides_predicate_refs):
            proofs = snapshot.claims_for_predicate(predicate)
            if len(proofs) > profile["max_requirement_claims"]:
                raise ValueError("current method requirement proof bound exceeded")
            proofs = tuple(claim for claim in proofs if admitted(claim)
                and claim.attributes.get(profile["requirement_scene_attribute"]) == tracking.frame_ref)
            claim_refs.update(claim.id for claim in proofs)
            if predicate in contract.requires_predicate_refs + contract.preserves_invariant_refs:
                requirement_rows.append(FrozenMap({"predicate_ref": predicate,
                    "current_positive_claim_refs": tuple(claim.id for claim in proofs)}))
            if predicate in contract.provides_predicate_refs:
                # Preserve exact scopes. One observed result does not prove that
                # every current bearer or terminal target has been satisfied.
                result_rows.append(FrozenMap({"predicate_ref": predicate,
                    "current_positive_claim_refs": tuple(claim.id for claim in proofs),
                    "scoped_evidence_rows": tuple(FrozenMap({"claim_ref": claim.id,
                        "argument_refs": claim.arguments}) for claim in proofs)}))
        if len(term_refs) + len(claim_refs) > profile["max_dependency_refs"]:
            raise ValueError("current method inventory dependency bound exceeded; ties retained")
        row = FrozenMap({"frame_ref": tracking.frame_ref, "method_ref": contract.method_ref,
            "activity_term_ref": activity.id, "objective_schema_ref": contract.objective_schema_ref,
            "required_mechanism_refs": contract.required_mechanism_refs,
            "falsifier_refs": contract.falsifier_refs, "role_binding_rows": tuple(role_rows),
            "observed_control_role_refs": control_role_refs,
            "current_control_member_rows": tuple(members for _, members in selection_rows),
            "control_expected_member_rows": expected_rows,
            "control_expected_roles_complete": control_roles_complete,
            "control_selection_count": len(selection_rows),
            "control_exact_match_rows": tuple(exact_rows),
            "control_disjoint_rows": tuple(disjoint_rows),
            "control_all_exact_matches": bool(exact_rows) and all(exact_rows),
            "control_all_disjoint": bool(disjoint_rows) and all(disjoint_rows),
            "requirement_rows": tuple(requirement_rows),
            "result_rows": tuple(result_rows),
            "missing_role_refs": tuple(item["role_ref"] for item in role_rows
                if item["role_ref"] in contract.required_role_refs and not item["binding_refs"]),
            "unproved_requirement_refs": tuple(item["predicate_ref"] for item in requirement_rows if not item["current_positive_claim_refs"]),
            "activity_proof_present": bool(activity_links),
            "evidence_refs": tuple(sorted(claim_refs)),
            "canonical_term_dependency_revisions": tuple(FrozenMap({"term_ref": ref,
                "revision": snapshot.term(ref).last_changed_state_revision}) for ref in sorted(term_refs)),
            "canonical_claim_dependency_digests": tuple(FrozenMap({"claim_ref": ref,
                "content_digest": stable_digest(snapshot.claim(ref))}) for ref in sorted(claim_refs))})
        token = stable_digest(tuple((field, row[field]) for field in profile["binding_token_fields"]))
        content_digest = stable_digest(tuple((field, row[field]) for field in profile["inventory_content_fields"]))
        term_ref = profile["inventory_term_id_pattern"].format(binding_token=token)
        existing = snapshot.term(term_ref)
        previous_receipt = existing.attributes.get("consultation_claim_ref") if existing else None
        unchanged = (existing is not None
            and existing.attributes.get("inventory_content_digest") == content_digest
            and previous_receipt in snapshot.current_claim_ids)
        # A repeated consultation is a no-op; returning to a previous content
        # after an actual change requires a fresh identity, not an old receipt.
        claim_ref = previous_receipt if unchanged else profile["inventory_claim_id_pattern"].format(
            binding_token=token, inventory_content_digest=content_digest,
            inventory_slot_revision=existing.last_changed_state_revision if existing else 0)
        revision_families = {field: pattern.format(binding_token=token)
            for field, pattern in sorted(profile["claim_revision_family_patterns"].items())}
        family_activity = {field: any(claim.disposition.value == profile["retirement_source_disposition"]
            for claim in snapshot.current_claims_by_revision_family.get(revision_families[family_field], ()))
            for field, family_field in sorted(profile["claim_revision_active_queries"].items())}
        rows.append(FrozenMap.overlay(row, FrozenMap({**revision_families, **family_activity, "binding_token": token,
            "inventory_content_digest": content_digest, "inventory_claim_ref": claim_ref,
            "inventory_pair_absent": not unchanged})))
    emitted_rows = tuple(row for row in rows if row["inventory_pair_absent"]) if profile.get("emit_only_changed_inventory_rows", False) else tuple(rows)
    evidence = {ref for row in emitted_rows for ref in row["evidence_refs"]}
    return FrozenMap({"inventory_rows": emitted_rows, "has_inventory_delta": any(
        row["inventory_pair_absent"] for row in rows), "evidence_refs": tuple(sorted(evidence)),
        "state_revision": snapshot.revision})
