"""Bounded exact relocation of observed effect slots, not a causal verdict.

DRM supplies proof admission, ordinal comparison, unknown-context handling,
projection fields and identities. All appearance ties are retained per slot;
no Cartesian family of actor assignments or action plans is materialized.
"""

from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest
from agents.yf_arc3_v5.state.appearance_index import exact_context_is_compatible


def measure_current_effect_inventory(*, snapshot, frame_ref, appearances, bearer_contexts, profile):
    samples = snapshot.terms_for_projection_contract(profile["sample_contract"])
    links = snapshot.claims_for_predicate(profile["sample_link_predicate"])
    if len(samples) > profile["maximum_samples"] or len(links) > profile["maximum_links"]:
        raise ValueError("shared effect inventory sample/proof bound exceeded")
    if len(appearances) > profile["maximum_bearers"]:
        raise ValueError("shared effect inventory bearer bound exceeded")
    if profile["comparison"] != "declared_maximal_ordinal_after_context":
        raise ValueError("unsupported shared effect ordinal comparison")
    by_key, by_bearer = {}, {}
    for appearance in appearances:
        if appearance.bearer_ref in by_bearer:
            raise ValueError("duplicate current appearance bearer")
        by_bearer[appearance.bearer_ref] = appearance
        if len(appearance.keys) > profile["maximum_signature_keys"]:
            raise ValueError("current appearance key bound exceeded")
        for key in appearance.keys:
            by_key.setdefault((appearance.contract_source_hash, key.code, key.ordinal, key.digest), []).append(appearance.bearer_ref)

    admitted_refs = {}

    def admitted(claim):
        if claim is None:
            return False
        if claim.predicate not in admitted_refs:
            claims = snapshot.claims_for_predicate(claim.predicate)
            if len(claims) > profile["maximum_links"]:
                raise ValueError("effect inventory premise bound exceeded")
            admitted_refs[claim.predicate] = {item.id for item in claims}
        return (claim.id in admitted_refs[claim.predicate]
            and claim.epistemic_status.value in profile["allowed_proof_statuses"]
            and claim.disposition.value == profile["proof_disposition"]
            and claim.polarity.value == profile["proof_polarity"]
            and not claim.active_contradictions and not claim.defeated_by)

    links_by_sample = {}
    for link in links:
        if (link.proof_rule == profile["sample_link_rule"] and admitted(link)
            and all(admitted(snapshot.claim(ref)) if snapshot.claim(ref) is not None
                else ref.startswith(tuple(profile["observation_leaf_prefixes"])) for ref in link.grounds)):
            for ref in link.arguments:
                links_by_sample.setdefault(ref, []).append(link)
    rows, checks = [], 0
    for sample in sorted(samples, key=lambda term: term.id):
        term_dependencies = {sample.id: sample.last_changed_state_revision}
        proofs = links_by_sample.get(sample.id, ())
        proof_refs = {link.id for link in proofs}
        proof_refs.update(ref for link in proofs for ref in link.grounds if snapshot.claim(ref) is not None)
        slots = sample.attributes.get("effect_slots", ())
        if len(slots) > profile["maximum_effect_slots"]:
            raise ValueError("effect inventory slot bound exceeded")
        slot_rows = []
        for slot in slots:
            keys = slot["signature_keys"]
            if len(keys) > profile["maximum_signature_keys"]:
                raise ValueError("historical effect key bound exceeded")
            candidates = []
            for key in keys:
                address = (slot["appearance_contract_source_hash"], key["code"], key["ordinal"], key["digest"])
                for bearer in by_key.get(address, ()):
                    checks += 1
                    if checks > profile["maximum_join_checks"]:
                        raise ValueError("effect appearance join bound exceeded; no ties truncated")
                    current_contexts = bearer_contexts.get(bearer, ())
                    historical = slot.get("context_alternatives", ()) or (FrozenMap(),)
                    compatible = tuple((requirements, item) for requirements in historical
                        for item in current_contexts
                        if exact_context_is_compatible(requirements, item["values"]))
                    if current_contexts and not compatible:
                        continue
                    unknown = set()
                    if not current_contexts:
                        unknown.update(field for requirements in historical for field in requirements)
                    else:
                        for requirements, item in compatible:
                            unknown.update(set(requirements) - set(item["values"]))
                    candidates.append((key["ordinal"], FrozenMap({"entity_ref": bearer,
                        "match_code": key["code"], "match_ordinal": key["ordinal"],
                        "unresolved_context_fields": tuple(sorted(unknown))}), compatible))
            maximum = max((ordinal for ordinal, _, _ in candidates), default=None)
            matches = {}
            for ordinal, match, compatible in candidates:
                if ordinal != maximum:
                    continue
                matches[match["entity_ref"]] = match
                for _, item in compatible:
                    dependency = item["term_dependency"]
                    term_dependencies[dependency["term_ref"]] = dependency["revision"]
                    proof_refs.update(dep["claim_ref"] for dep in item["claim_dependencies"])
            if len(matches) > profile["maximum_matches_per_slot"]:
                raise ValueError("effect appearance tie bound exceeded; nothing dropped")
            slot_rows.append(FrozenMap({"effect": slot["effect"],
                "current_bearer_matches": tuple(match for _, match in sorted(matches.items())),
                "maximal_match_ordinal": maximum}))
        if len(proof_refs) + len(term_dependencies) > profile["maximum_dependency_refs"]:
            raise ValueError("effect inventory dependency bound exceeded")
        row = FrozenMap({"frame_ref": frame_ref, "sample_term_ref": sample.id,
            "pattern_token": sample.attributes["pattern_token"],
            "action_ref": sample.attributes["action_ref"],
            "observed_confidence_state": sample.attributes["observed_confidence_state"],
            "slot_binding_rows": tuple(slot_rows), "signature_inputs_present": bool(slots),
            "all_slots_have_matches": bool(slots) and all(slot["current_bearer_matches"] for slot in slot_rows),
            "distinct_slot_bearers_possible": bool(slots) and all(slot["current_bearer_matches"] for slot in slot_rows)
                and len({match["entity_ref"] for slot in slot_rows for match in slot["current_bearer_matches"]}) >= len(slots),
            "sample_proof_present": bool(proofs), "evidence_refs": (sample.id, *sorted(proof_refs)),
            "canonical_term_dependency_revisions": tuple(FrozenMap({"term_ref": ref, "revision": revision})
                for ref, revision in sorted(term_dependencies.items())),
            "canonical_claim_dependency_digests": tuple(FrozenMap({"claim_ref": ref,
                "content_digest": stable_digest(snapshot.claim(ref))}) for ref in sorted(proof_refs))})
        token = stable_digest(tuple((field, row[field]) for field in profile["binding_token_fields"]))
        digest = stable_digest(row)
        term_ref = profile["inventory_term_id_pattern"].format(binding_token=token)
        claim_ref = profile["inventory_claim_id_pattern"].format(binding_token=token, content_digest=digest)
        existing = snapshot.term(term_ref)
        revision_families = {field: pattern.format(binding_token=token)
            for field, pattern in sorted(profile["claim_revision_family_patterns"].items())}
        rows.append(FrozenMap.overlay(FrozenMap({**revision_families,
            "binding_token": token, "content_digest": digest,
            "inventory_claim_ref": claim_ref, "inventory_pair_absent": existing is None
                or existing.attributes.get("content_digest") != digest or snapshot.claim(claim_ref) is None}), row))
    emitted_rows = tuple(row for row in rows if row["inventory_pair_absent"]) if profile.get(
        "emit_only_changed_inventory_rows", False) else tuple(rows)
    evidence = {ref for row in emitted_rows for ref in row["evidence_refs"]}
    return FrozenMap({"effect_inventory_rows": emitted_rows, "has_inventory_delta": any(
        row["inventory_pair_absent"] for row in rows), "evidence_refs": tuple(sorted(evidence)),
        "state_revision": snapshot.revision})
