"""Exact historical translation/current lattice joins; no action selection.

The declaring DRM supplies admitted source rules, roles and proof statuses.
An unchanged raster silhouette is used only to calibrate the observed actor
at acquisition time. Retained quantities are historical evidence, not a
fixed action count for a new scene or bearer. The declaring policy controls
whether these measurements may ground a revisable command-step supposition.
"""

from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest
from agents.yf_arc3_v5.state.reference_bindings import canonical_reference_bindings_current


def measure_logical_step_records(*, snapshot, tracking, profile):
    if tracking is None:
        return FrozenMap({"step_rows": (), "has_new_steps": False,
            "evidence_refs": (), "state_revision": snapshot.revision})
    grids = tuple(term for term in snapshot.terms_for_literal_attribute("operational_scene_ref", tracking.frame_ref)
        if term.attributes.get("projection_contract") == profile["grid_contract"]
        and canonical_reference_bindings_current(snapshot, term, frame_ref=tracking.frame_ref))
    if len(grids) > 3:
        raise ValueError("logical command step lattice alternatives exceed bound")
    rows, evidence = {}, set()
    for grid in grids:
        for step in measure_command_step_rows(snapshot=snapshot, tracking=tracking,
                geometry=grid.attributes["grid_geometry"], supports=grid.attributes["cell_support_rows"], profile=profile):
            donor = snapshot.term(step["donor_term_ref"])
            # Capture while the *observed* actor still has this temporal identity;
            # another similar object is not a historical calibration witness.
            if donor.attributes.get("entity_ref") not in step["current_same_shape_entity_refs"]:
                continue
            pitch = tuple(grid.attributes["grid_geometry"][5:7])
            token = stable_digest((step["proof_claim_ref"], step["transition_ref"], pitch,
                step["delta_row_cells"], step["delta_col_cells"]))
            memory_ref = profile["memory_term_pattern"].format(step_token=token)
            if snapshot.term(memory_ref) is not None:
                continue
            refs = tuple(sorted({step["proof_claim_ref"], *step["proof_dependency_refs"],
                *grid.attributes.get("role_premise_claim_refs", ())}))
            rows[token] = FrozenMap({"step_token": token, "memory_ref": memory_ref,
                "step_claim_ref": profile["memory_claim_pattern"].format(step_token=token),
                "action_ref": step["action_ref"], "delta_row_cells": step["delta_row_cells"],
                "delta_col_cells": step["delta_col_cells"], "source_cell_pitch": pitch,
                "transition_ref": step["transition_ref"], "source_frame_ref": tracking.frame_ref,
                "row_evidence_refs": refs})
            evidence.update(refs)
            if len(rows) > profile["maximum_rows"]:
                raise ValueError("logical command step record bound exceeded")
    return FrozenMap({"step_rows": tuple(row for _, row in sorted(rows.items())),
        "has_new_steps": bool(rows), "evidence_refs": tuple(sorted(evidence)), "state_revision": snapshot.revision})


def read_logical_step_rows(*, snapshot, profile, for_action_grounding=False):
    if for_action_grounding and not profile["action_grounding_enabled"]:
        return ()
    terms = snapshot.terms_for_projection_contract(profile["memory_contract"])
    proofs = snapshot.claims_for_predicate(profile["memory_predicate"])
    if max(len(terms), len(proofs)) > profile["maximum_records"]:
        raise ValueError("logical command step memory bound exceeded")

    def admitted(claim):
        return (claim is not None and claim.epistemic_status.value in profile["proof_statuses"]
            and any(current.id == claim.id for current in snapshot.claims_for_predicate(claim.predicate))
            and claim.disposition.value == profile["proof_disposition"]
            and claim.polarity.value == profile["proof_polarity"]
            and not claim.active_contradictions and not claim.defeated_by)

    by_term = {}
    for proof in proofs:
        if (proof.proof_rule == profile["memory_rule"] and admitted(proof)
                and all(snapshot.claim(ref) is None or admitted(snapshot.claim(ref)) for ref in proof.grounds)):
            for ref in proof.arguments:
                by_term.setdefault(ref, []).append(proof)
    rows = []
    for term in terms:
        for proof in by_term.get(term.id, ()):
            attrs = term.attributes
            rows.append(FrozenMap({"donor_term_ref": term.id, "proof_claim_ref": proof.id,
                "proof_dependency_refs": tuple(sorted(ref for ref in proof.grounds if snapshot.claim(ref) is not None)),
                "transition_ref": attrs["transition_ref"], "action_ref": attrs["action_ref"],
                "delta_row_cells": attrs["delta_row_cells"], "delta_col_cells": attrs["delta_col_cells"]}))
            if len(rows) > profile["maximum_rows"]:
                raise ValueError("logical command step recalled row bound exceeded")
    return tuple(rows)


def measure_command_step_rows(*, snapshot, tracking, geometry, supports, profile):
    proofs = snapshot.claims_for_predicate(profile["proof_predicate"])
    entities = tuple(entity for entity in tracking.entities if entity.current_frame_ref == tracking.frame_ref)
    if any(len(rows) > profile["maximum_records"] for rows in (proofs, entities)):
        raise ValueError("command step evidence bound exceeded")
    if len(supports) > 128 or sum(len(entity.component.pixels) for entity in entities) > 8192:
        raise ValueError("command step support bound exceeded")
    _, ro, co, height, width, rp, cp, rows, cols = geometry
    if min(height, width, rp, cp, rows, cols) <= 0 or height > rp or width > cp or rows * cols > 4096:
        raise ValueError("invalid command step lattice calibration")
    occupied = tuple(frozenset((support["top"]+r, support["left"]+c)
        for r, c in support["normalized_cells"]) for support in supports)
    components = {}
    for entity in entities:
        component = entity.component
        logical = set()
        for r, c in component.pixels:
            lr, ir = divmod(r-ro, rp)
            lc, ic = divmod(c-co, cp)
            if not (0 <= lr < rows and 0 <= lc < cols and ir < height and ic < width):
                break
            logical.add((lr, lc))
        else:
            if logical and frozenset(logical) in occupied:
                key = (stable_digest(component.relative_pixels)[:profile["shape_digest_length"]], component.area,
                       component.bbox.height, component.bbox.width)
                components.setdefault(key, []).append(entity.entity_id)
    # This join has always required a currently supported matching shape.
    # Query that input domain before enforcing its bound, retaining unrelated
    # historical suppositions in canonical memory for future observations.
    current_donors = []
    for key in sorted(components):
        for donor in snapshot.terms_for_literal_attribute("shape_digest", key[0]):
            attrs = donor.attributes
            if (attrs.get("role") == profile["donor_role"]
                and tuple(attrs.get(field) for field in
                    ("shape_digest", "area", "bbox_height", "bbox_width")) == key):
                current_donors.append(donor)
    donors = tuple(sorted(current_donors, key=lambda donor: donor.id))
    if len(donors) > profile["maximum_records"]:
        raise ValueError("command step evidence bound exceeded")
    linked = {}
    for claim in proofs:
        if (claim.proof_rule in profile["proof_rules"]
                and claim.epistemic_status.value in profile["proof_statuses"]
                and claim.disposition.value == profile["proof_disposition"]
                and claim.polarity.value == profile["proof_polarity"]
                and not claim.active_contradictions and not claim.defeated_by
                and all((premise := snapshot.claim(ref)) is None or (
                    premise.epistemic_status.value in profile["proof_statuses"]
                    and
                    premise.disposition.value == profile["proof_disposition"]
                    and premise.polarity.value == profile["proof_polarity"]
                    and not premise.active_contradictions and not premise.defeated_by)
                    for ref in claim.grounds)):
            for ref in claim.arguments:
                linked.setdefault(ref, []).append(claim)
    result = []
    for donor in donors:
        attrs = donor.attributes
        key = tuple(attrs.get(field) for field in ("shape_digest", "area", "bbox_height", "bbox_width"))
        matches = tuple(sorted(components.get(key, ())))
        if not matches or attrs.get("member_bound_exceeded"):
            continue
        dr, dc = attrs.get("delta_row"), attrs.get("delta_col")
        if type(dr) is not int or type(dc) is not int or (dr == 0) == (dc == 0):
            continue
        if dr % rp or dc % cp:
            continue
        for proof in linked.get(donor.id, ()):
            if (proof.attributes.get("delta_row") != dr or proof.attributes.get("delta_col") != dc
                    or proof.attributes.get("shape_digest") != attrs.get("shape_digest")
                    or not attrs.get("transition_ref")
                    or proof.attributes.get("transition_ref") != attrs["transition_ref"]
                    or not proof.attributes.get("action_ref")):
                continue
            result.append(FrozenMap({"donor_term_ref": donor.id, "proof_claim_ref": proof.id,
                "proof_dependency_refs": tuple(sorted(ref for ref in proof.grounds if snapshot.claim(ref) is not None)),
                "transition_ref": attrs["transition_ref"], "action_ref": proof.attributes["action_ref"],
                "delta_row_cells": dr // rp, "delta_col_cells": dc // cp,
                "current_same_shape_entity_refs": matches}))
            if profile.get("retrospective_common_translation"):
                result.extend(_measure_prior_common_steps(snapshot=snapshot, donor=donor,
                    control_proof=proof, matches=matches, pitch=(rp, cp), profile=profile))
            if len(result) > profile["maximum_rows"]:
                raise ValueError("command step row bound exceeded; no alternatives truncated")
    return tuple(result)


def _measure_prior_common_steps(*, snapshot, donor, control_proof, matches, pitch, profile):
    """Join retained observations to a later role proof, never replay directions.

    Only the same tracked actor in an observed uniform translation is admitted.
    The control-witness transition supplies the comparable context epoch. A
    later role does not retroactively prove direct control: retained steps are
    still revisable and the normal first-effect checkpoint remains mandatory.
    """
    contract = profile["retrospective_common_translation"]
    translations = snapshot.claims_for_predicate(contract["translation_predicate"])
    groups = snapshot.claims_for_predicate(contract["cotranslation_predicate"])
    if max(len(translations), len(groups)) > profile["maximum_records"]:
        raise ValueError("retrospective command observation bound exceeded")

    def admitted(claim):
        return (claim is not None and claim.proof_rule == contract["proof_rule"]
            and claim.epistemic_status.value in profile["proof_statuses"]
            and claim.disposition.value == profile["proof_disposition"]
            and claim.polarity.value == profile["proof_polarity"]
            and not claim.active_contradictions and not claim.defeated_by
            and all((premise := snapshot.claim(ref)) is None or (
                premise.epistemic_status.value in profile["proof_statuses"]
                and premise.disposition.value == profile["proof_disposition"]
                and premise.polarity.value == profile["proof_polarity"]
                and not premise.active_contradictions and not premise.defeated_by)
                for ref in claim.grounds))

    witnesses = tuple(claim for claim in translations if admitted(claim)
        and claim.arguments == (control_proof.attributes["action_ref"], control_proof.attributes["transition_ref"]))
    if len(witnesses) != 1 or donor.attributes.get("entity_ref") not in matches:
        return ()
    witness, = witnesses
    epoch = witness.attributes.get("context_epoch")
    if epoch is None:
        return ()
    result, checks = [], 0
    for effect in translations:
        if (not admitted(effect) or len(effect.arguments) != 2
                or effect.attributes.get("context_epoch") != epoch
                or effect.attributes.get("distinct_translation_delta_count") != 1
                or effect.arguments[1] == control_proof.attributes["transition_ref"]):
            continue
        dr, dc = effect.attributes.get("delta_row"), effect.attributes.get("delta_col")
        if (type(dr) is not int or type(dc) is not int or (dr == 0) == (dc == 0)
                or dr % pitch[0] or dc % pitch[1]):
            continue
        for group in groups:
            checks += 1
            if checks > contract["maximum_join_checks"]:
                raise ValueError("retrospective command join bound exceeded")
            if (not admitted(group) or not group.arguments or group.arguments[0] != effect.arguments[0]
                    or group.attributes.get("transition_ref") != effect.arguments[1]
                    or donor.attributes["entity_ref"] not in group.attributes.get("entity_refs", ())):
                continue
            dependencies = {control_proof.id, witness.id, group.id}
            dependencies.update(ref for claim in (control_proof, witness, group, effect)
                for ref in claim.grounds if snapshot.claim(ref) is not None)
            result.append(FrozenMap({"donor_term_ref": donor.id, "proof_claim_ref": effect.id,
                "proof_dependency_refs": tuple(sorted(dependencies)),
                "transition_ref": effect.arguments[1], "action_ref": effect.arguments[0],
                "delta_row_cells": dr // pitch[0], "delta_col_cells": dc // pitch[1],
                "current_same_shape_entity_refs": matches}))
            if len(result) > profile["maximum_rows"]:
                raise ValueError("retrospective command row bound exceeded")
    return tuple(result)
