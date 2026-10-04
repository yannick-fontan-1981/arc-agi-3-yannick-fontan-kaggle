"""Exact canonical provenance joins for Source-requested retrospective geometry."""

from agents.yf_arc3_v5.capabilities.terminal_support_coverage import measure_terminal_support_coverage
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest
from agents.yf_arc3_v5.state.reference_bindings import canonical_reference_bindings_current


def measure_terminal_outcome_evidence(*, snapshot, before_frame, after_frame,
        completion_facts, transition_ref, profile, repository=None, before_ref=None, after_ref=None,
        store=None):
    """Retain a bounded observation pair, not an inferred winning predicate.

    The returned final observation may already depict the next scene. It must
    never be substituted for an observed terminal configuration. Source owns
    that interpretation and the status of every predicted relation.
    """
    official = snapshot.claims_for_predicate(profile["official_predicate"])
    if len(official) > profile["maximum_records"]:
        raise ValueError("terminal outcome proof bound exceeded")
    proofs = tuple(claim.id for claim in official
        if claim.epistemic_status.value == profile["official_status"]
        and claim.proof_rule in profile["official_rules"]
        and claim.disposition.value == "active" and claim.polarity.value == "positive"
        and not claim.active_contradictions and not claim.defeated_by
        and len(claim.arguments) == 2 and claim.arguments[1] == transition_ref)
    if len(proofs) != 1:
        raise ValueError("terminal outcome requires one exact official completion proof")

    def bounded_frame(rows):
        if rows is None:
            return None
        if not isinstance(rows, (tuple, list)) or not rows:
            raise ValueError("terminal outcome frame must be a nonempty rectangle")
        if len(rows) > profile["maximum_pixels"]:
            raise ValueError("terminal outcome frame bound exceeded")
        width = len(rows[0]) if isinstance(rows[0], (tuple, list)) else 0
        if not width or len(rows) * width > profile["maximum_pixels"]:
            raise ValueError("terminal outcome frame bound exceeded")
        if any(not isinstance(row, (tuple, list)) or len(row) != width
            or any(type(pixel) is not int for pixel in row) for row in rows):
            raise ValueError("terminal outcome frame must contain rectangular integer pixels")
        return tuple(tuple(row) for row in rows)

    from agents.yf_arc3_v5.state.terminal_executed_evidence import (
        bind_executed_before_observation, measure_executed_action_contract,
    )
    executed = measure_executed_action_contract(store=store, after_ref=after_ref, profile=profile)
    start_ref = before_ref
    executed_before = bind_executed_before_observation(repository=repository,
        executed=executed, before_ref=before_ref, after_ref=after_ref, profile=profile)
    if executed_before is not None:
        before_ref, before_frame = executed_before.raw_input_ref, executed_before.frame
    before, after = bounded_frame(before_frame), bounded_frame(after_frame)
    from agents.yf_arc3_v5.state.terminal_observation_history import measure_terminal_observation_history
    history = measure_terminal_observation_history(repository=repository,
        before_ref=before_ref, after_ref=after_ref, profile=profile)
    # This is a whitelist projection of existing Source facts, never an
    # alternative goal authored here or a copy of the complete action context.
    fields = profile["retained_fields"]
    if len(fields) > profile["maximum_fields"] or len(set(fields)) != len(fields):
        raise ValueError("terminal outcome fact projection bound exceeded")
    retained = FrozenMap({field: completion_facts.get(field) for field in fields})
    for field in fields:
        value = retained[field]
        if value is not None and (type(value) not in (str, int, bool)
                or (isinstance(value, str) and len(value) > profile["maximum_reference_length"])):
            raise ValueError("terminal outcome retained scalar bound exceeded")
    facts = FrozenMap({
        "transition_ref": transition_ref,
        "before_frame_rows": before, "after_frame_rows": after,
        "before_frame_present": before is not None, "after_frame_present": after is not None,
        "before_frame_digest": stable_digest(before), "after_frame_digest": stable_digest(after),
        "retained_completion_facts": retained,
        "retained_completion_digest": stable_digest(retained),
        "official_boundary_present": True,
        "observation_history": history,
        "executed_action_contract": executed,
        "executed_before_raw_input_ref": executed_before.raw_input_ref if executed_before is not None else None,
        "retrospective_start_raw_input_ref": start_ref,
    })
    from agents.yf_arc3_v5.state.terminal_entry_geometry import measure_terminal_entry_geometry
    entry = measure_terminal_entry_geometry(snapshot=snapshot, repository=repository, store=store,
        before_ref=before_ref, before_frame=before, executed=executed, history=history, profile=profile)
    return FrozenMap({"descriptive_facts": FrozenMap.overlay(facts, entry), "evidence_refs": proofs,
                      "entry_evidence_refs": tuple(dict.fromkeys((*proofs, *entry.get("entry_geometry_premise_refs", ())))),
                      "state_revision": snapshot.revision})


def measure_terminal_coverage_evidence(*, snapshot, scene, action_ref, context_epoch, transition_ref,
        profile, frame_rows=None, repository=None, store=None):
    empty = FrozenMap({"coverage_rows": (), "has_coverage_measurements": False,
                      "evidence_refs": (), "state_revision": snapshot.revision})
    if (scene is None and frame_rows is None) or context_epoch is None:
        return empty
    terms = snapshot.terms_for_projection_contract(profile["layer_contract"])
    links = snapshot.claims_for_predicate(profile["layer_predicate"])
    associations = snapshot.claims_for_predicate(profile["association_predicate"])
    if any(len(items) > profile["maximum_records"] for items in (terms, links, associations)):
        raise ValueError("terminal coverage canonical record bound exceeded")

    def admitted(claim):
        return (claim is not None and claim.epistemic_status.value in profile["proof_statuses"]
            and claim.disposition.value == profile["proof_disposition"]
            and claim.polarity.value == profile["proof_polarity"]
            and not claim.active_contradictions and not claim.defeated_by)

    official = snapshot.claims_for_predicate(profile["official_predicate"])
    if len(official) > profile["maximum_records"]:
        raise ValueError("terminal coverage completion proof bound exceeded")
    official = tuple(claim for claim in official if admitted(claim)
        and claim.epistemic_status.value == profile["official_status"]
        and claim.proof_rule in profile["official_rules"] and len(claim.arguments) == 2
        and claim.arguments[1] == transition_ref)
    if len(official) != 1:
        return empty

    if scene is None:
        # A committed series may retain the observation without materializing a
        # scene. Measure only the geometry requested by Source, not a new goal
        # or a complete perception/decision loop.
        from agents.yf_arc3_v5.capabilities import ComponentExtractionInput, FrameGrid, detect_components
        if sum(len(row) for row in frame_rows) > profile["maximum_pixels"]:
            raise ValueError("terminal coverage frame bound exceeded")
        frame = FrameGrid(rows=frame_rows)
        blocks = detect_components(ComponentExtractionInput(frame=frame,
            connectivity=profile["component_connectivity"]))
        scene = {"frame": frame, "blocks": blocks}

    association_index = {}
    for link in associations:
        if not admitted(link) or link.proof_rule != profile["association_rule"] or len(link.arguments) != 1:
            continue
        term = snapshot.term(link.arguments[0])
        if term is None or term.attributes.get("active") is not True:
            continue
        for ref in term.attributes.get("source_claim_refs", ()):
            source = snapshot.claim(ref)
            if not admitted(source) or not source.arguments or source.predicate != profile["role_predicate"]:
                continue
            association_index.setdefault((source.arguments[0], term.attributes.get("role_ref")), []).append((term.id, link.id, ref))
    rows, evidence, geometry_records = [], set(), []
    for term in terms:
        attrs = term.attributes
        if attrs.get("context_epoch") != context_epoch or attrs.get("action_ref") != action_ref:
            continue
        proofs = tuple(link for link in links if admitted(link)
            and link.proof_rule == profile["layer_rule"] and link.arguments[0] == term.id)
        if not proofs or not all(attrs.get(name) == expected for name, expected in sorted(profile["layer_equalities"].items())):
            continue
        geometry_records.append(FrozenMap({"geometry_evidence_ref": term.id,
            "layer_rows": attrs["scene_layer_rows"], "proof_refs": tuple(link.id for link in proofs)}))
    from agents.yf_arc3_v5.state.terminal_executed_evidence import measure_executed_support_records
    geometry_records.extend(measure_executed_support_records(snapshot=snapshot, store=store,
        repository=repository, action_ref=action_ref, context_epoch=context_epoch,
        transition_ref=transition_ref, profile=profile))
    if len(geometry_records) > profile["maximum_rows"]:
        raise ValueError("terminal coverage geometry source bound exceeded")
    for geometry in geometry_records:
        measurements = measure_terminal_support_coverage(layer_rows=geometry["layer_rows"], current_scene=scene,
            maximum_layers=profile["maximum_layers"], maximum_pixels=profile["maximum_pixels"],
            maximum_components=profile["maximum_components"], maximum_rows=profile["maximum_rows"])
        for measured in measurements:
            moving = association_index.get((measured["moving_entity_ref"], profile["moving_role_ref"]), ())
            fixed = association_index.get((measured["fixed_entity_ref"], profile["fixed_role_ref"]), ())
            if not moving or not fixed:
                continue
            moving_refs = tuple(sorted({item[0] for item in moving}))
            fixed_refs = tuple(sorted({item[0] for item in fixed}))
            proof_refs = tuple(sorted({official[0].id} | set(geometry["proof_refs"])
                | {ref for item in (*moving, *fixed) for ref in item[1:]}))
            row = {name: measured[name] for name in profile["measurement_fields"]}
            row.update({"moving_association_refs": moving_refs, "fixed_association_refs": fixed_refs,
                "relation_token": stable_digest((moving_refs, fixed_refs)), "row_evidence_refs": proof_refs,
                "geometry_evidence_ref": geometry["geometry_evidence_ref"]})
            row.update(measure_support_coverage_history(layer_rows=geometry["layer_rows"],
                moving_ref=measured["moving_entity_ref"], fixed_ref=measured["fixed_entity_ref"],
                history=geometry.get("observation_history"), repository=repository, profile=profile))
            rows.append(FrozenMap(row))
            evidence.update(proof_refs)
            if len(rows) > profile["maximum_rows"]:
                raise ValueError("terminal coverage evidence row bound exceeded; nothing dropped")
    return FrozenMap({"coverage_rows": tuple(rows), "has_coverage_measurements": bool(rows),
                      "official_boundary_present": True, "transition_ref": transition_ref,
                      "completion_event_ref": official[0].arguments[0], "evidence_refs": tuple(sorted(evidence)),
                      "state_revision": snapshot.revision})


def measure_support_coverage_history(*, layer_rows, moving_ref, fixed_ref, history, repository, profile):
    """Evaluate one identical bound support relation on every indexed old raster.

    Multiple compatible placements with disagreeing values remain unknown. This
    is a geometric counterexample measurement, never a conclusion of sufficiency.
    """
    observations = history.get("prior_observation_rows", ()) if history else ()
    excluded_initial = int(bool(observations) and history.get("prior_observation_index_complete") is True
        and profile["history_skip_partition_initial"])
    counts = {"history_true_count": 0, "history_false_count": 0,
        "history_unknown_count": len(observations) - excluded_initial, "history_evaluation_complete": False,
        "history_first_counterexample_ref": None, "history_excluded_initial_count": excluded_initial}
    if repository is None or not observations or len(observations) > profile["maximum_history_evaluations"]:
        return counts
    from agents.yf_arc3_v5.capabilities import ComponentExtractionInput, FrameGrid, detect_components
    fixed = tuple(layer for layer in layer_rows if layer.get("prior_entity_ref") == fixed_ref)
    if len(fixed) != 1:
        return counts
    geometry = fixed[0]["component_geometry"]
    forward_support_witness = False
    for ordinal, row in enumerate(observations):
        try:
            observed = repository.get(row["raw_input_ref"])
        except KeyError:
            forward_support_witness = False
            continue
        if len(observed.frame) * len(observed.frame[0]) > profile["maximum_pixels"]:
            forward_support_witness = False
            continue
        fully_visible = all(0 <= r < len(observed.frame) and 0 <= c < len(observed.frame[0])
            and observed.frame[r][c] == geometry["value"] for r, c in geometry["pixels"])
        if fully_visible:
            forward_support_witness = True
        if ordinal < excluded_initial:
            continue
        frame = FrameGrid(rows=observed.frame)
        blocks = detect_components(ComponentExtractionInput(frame=frame, connectivity=profile["component_connectivity"]))
        if len(blocks.components) > profile["maximum_components"]:
            continue
        results = measure_terminal_support_coverage(layer_rows=layer_rows,
            current_scene={"frame": frame, "blocks": blocks},
            maximum_layers=profile["maximum_layers"], maximum_pixels=profile["maximum_pixels"],
            maximum_components=profile["maximum_components"], maximum_rows=profile["maximum_rows"])
        if not results and not fully_visible:
            forward_support_witness = False
        if profile["history_require_forward_support_witness"] and not forward_support_witness:
            continue
        values = {result[profile["history_measure_field"]] == profile["history_comparison_value"]
            for result in results if result["moving_entity_ref"] == moving_ref and result["fixed_entity_ref"] == fixed_ref}
        if len(values) != 1:
            continue
        counts["history_unknown_count"] -= 1
        if values == {True}:
            counts["history_true_count"] += 1
            if counts["history_first_counterexample_ref"] is None:
                counts["history_first_counterexample_ref"] = row["raw_input_ref"]
        else:
            counts["history_false_count"] += 1
    counts["history_evaluation_complete"] = (history.get("prior_observation_index_complete") is True
        and counts["history_unknown_count"] == 0)
    return counts


def measure_current_terminal_inventory(*, snapshot, tracking, profile, drm):
    from agents.yf_arc3_v5.capabilities.member_cell_geometry import measure_member_cell_geometry
    from agents.yf_arc3_v5.state.command_step_measurements import read_logical_step_rows
    step_policy, _ = drm.resolve(profile["command_step_policy_ref"])
    recalled_steps = read_logical_step_rows(snapshot=snapshot, profile=step_policy.configuration["measurement_contract"], for_action_grounding=True)
    memories = snapshot.terms_for_projection_contract(profile["memory_contract"])
    proofs = snapshot.claims_for_predicate(profile["memory_predicate"])
    counterexamples = snapshot.claims_for_predicate(profile["counterexample_predicate"])
    current = snapshot.terms_for_literal_attribute("operational_scene_ref", tracking.frame_ref)
    bindings = tuple(term for term in current if term.attributes.get("projection_contract") == profile["binding_contract"])
    grids = tuple(term for term in current if term.attributes.get("projection_contract") == profile["cell_lattice_contract"]
        and canonical_reference_bindings_current(snapshot, term, frame_ref=tracking.frame_ref))
    if len(grids) > 3:
        raise ValueError("current terminal inventory lattice bound exceeded")
    if any(len(items) > profile["maximum_records"] for items in (memories, proofs, bindings, counterexamples)):
        raise ValueError("current terminal inventory bound exceeded")
    entities = frozenset(item.entity_id for item in tracking.entities if item.current_frame_ref == tracking.frame_ref)

    def admitted(claim):
        return (claim is not None and claim.epistemic_status.value in profile["proof_statuses"]
            and claim.disposition.value == profile["proof_disposition"]
            and claim.polarity.value == profile["proof_polarity"]
            and not claim.active_contradictions and not claim.defeated_by)

    rows, all_evidence = [], set()
    for memory in memories:
        links = tuple(claim for claim in proofs if claim.arguments[0] == memory.id
            and claim.proof_rule == profile["memory_rule"] and admitted(claim)
            and all(admitted(snapshot.claim(ref)) for ref in claim.grounds))
        term_refs, claim_refs = {memory.id}, {claim.id for claim in links}
        contrary = tuple(claim for claim in counterexamples if admitted(claim)
            and claim.proof_rule == profile["counterexample_rule"]
            and all(memory.attributes.get(field) and claim.attributes.get(field) == memory.attributes[field]
                for field in profile["association_fields"])
            and all(admitted(snapshot.claim(ref)) for ref in claim.grounds))
        claim_refs.update(claim.id for claim in contrary)
        claim_refs.update(ref for claim in links for ref in claim.grounds)
        groups = []
        unresolved = []
        for field in profile["association_fields"]:
            refs = memory.attributes.get(field, ())
            active = tuple(ref for ref in refs if snapshot.term(ref) is not None
                and snapshot.term(ref).attributes.get("active") is True)
            term_refs.update(ref for ref in refs if snapshot.term(ref) is not None)
            matches = tuple(term for term in bindings if term.attributes.get("association_ref") in active
                and term.attributes.get("current_member_entity_refs")
                and all(ref in entities for ref in term.attributes["current_member_entity_refs"])
                and canonical_reference_bindings_current(snapshot, term, frame_ref=tracking.frame_ref))
            groups.append(tuple(term.attributes["current_member_entity_refs"] for term in matches))
            unresolved.append(tuple(term.attributes.get("unresolved_context_fields", ()) for term in matches))
            for binding in matches:
                term_refs.add(binding.id)
                term_refs.update(item["term_ref"] for item in binding.attributes.get("canonical_term_dependency_revisions", ()))
                claim_refs.update(binding.attributes.get("role_premise_claim_refs", ()))
                claim_refs.update(item["claim_ref"] for item in binding.attributes.get("canonical_claim_dependency_digests", ()))
        cell_rows = []
        for grid in grids:
            steps = recalled_steps
            measured = measure_member_cell_geometry(groups=groups, tracking=tracking,
                geometry=grid.attributes["grid_geometry"], supports=grid.attributes["cell_support_rows"],
                maximum_pairs=profile["maximum_cell_pairs"],
                maximum_reflection_tests=profile["maximum_cell_reflection_tests"])
            declared_geometry = project_cell_goal_geometry(measured=measured, drm=drm, profile=profile, command_steps=steps)
            cell_rows.append(FrozenMap.overlay(measured, FrozenMap({"grid_term_ref": grid.id,
                "grid_ref": grid.attributes["grid_geometry"][0], "declared_geometry": declared_geometry,
                "grid_geometry": grid.attributes["grid_geometry"],
                "logical_extent": grid.attributes["grid_geometry"][-2:], "command_step_rows": steps})))
            term_refs.update(step["donor_term_ref"] for step in steps)
            claim_refs.update(step["proof_claim_ref"] for step in steps)
            claim_refs.update(ref for step in steps for ref in step["proof_dependency_refs"])
            term_refs.add(grid.id)
            claim_refs.update(grid.attributes.get("role_premise_claim_refs", ()))
        if len(term_refs) + len(claim_refs) > profile["maximum_dependencies"]:
            raise ValueError("current terminal inventory dependency bound exceeded")
        row = FrozenMap({"memory_ref": memory.id, "frame_ref": tracking.frame_ref,
            "moving_member_rows": groups[0], "fixed_member_rows": groups[1],
            "cell_geometry_rows": tuple(cell_rows),
            "unresolved_context_rows": tuple(unresolved), "historical_proof_present": bool(links),
            "historical_counterexample_count": len(contrary),
            "current_bearers_present": bool(groups[0]) and bool(groups[1]),
            "current_contexts_resolved": not any(fields for group in unresolved for fields in group),
            "row_evidence_refs": (memory.id, *sorted(claim_refs)),
            "canonical_term_dependency_revisions": tuple(FrozenMap({"term_ref": ref,
                "revision": snapshot.term(ref).last_changed_state_revision}) for ref in sorted(term_refs)),
            "canonical_claim_dependency_digests": tuple(FrozenMap({"claim_ref": ref,
                "content_digest": stable_digest(snapshot.claim(ref))}) for ref in sorted(claim_refs))})
        token = stable_digest(memory.id)
        digest = stable_digest(row)
        term_ref = profile["inventory_term_pattern"].format(binding_token=token)
        claim_ref = profile["inventory_claim_pattern"].format(binding_token=token, content_digest=digest)
        existing = snapshot.term(term_ref)
        rows.append(FrozenMap.overlay(row, FrozenMap({"binding_token": token, "content_digest": digest,
            "current_goal_ref": term_ref, "current_goal_chain_refs": (term_ref,),
            "consultation_claim_ref": claim_ref, "inventory_delta": existing is None
                or existing.attributes.get("content_digest") != digest or snapshot.claim(claim_ref) is None})))
        all_evidence.update(claim_refs)
        all_evidence.add(memory.id)
    return FrozenMap({"inventory_rows": tuple(rows), "has_inventory_delta": any(row["inventory_delta"] for row in rows),
        "evidence_refs": tuple(sorted(all_evidence)), "state_revision": snapshot.revision})


def project_cell_goal_geometry(*, measured, drm, profile, command_steps=()):
    """Execute only the DRM-named affine law on the exact cell correspondences.

The zero-motion projection supplies the inverse required axis position.
Explicit donor steps additionally supply counterfactual axis-motion inputs,
never an actuator choice or an authorization to move.
"""
    from agents.yf_arc3_v5.drm.meaning import project_declared_alternatives
    pairs = measured["pair_rows"]
    reflections = measured["reflection_measurements"]
    if not measured["pair_enumeration_complete"] or not reflections["complete"]:
        return FrozenMap({"complete": False, "rows": ()})
    joins = tuple((i,j) for i,pair in enumerate(pairs) for j,relation in enumerate(reflections["rows"])
                  if pair["first_support_ref"] == relation["first_support_ref"])
    if len(joins) * (1 + 2*len(command_steps)) > profile["maximum_geometry_projections"]:
        return FrozenMap({"complete": False, "rows": ()})
    policy_ref = profile["geometry_policy_ref"]
    policy, source = drm.resolve(policy_ref)
    unique_geometry_ref, = policy.declared_alternative_refs
    results = []
    for pair_index, reflection_index in joins:
        pair, relation = pairs[pair_index], reflections["rows"][reflection_index]
        residual = (pair["offset_row_cells"], pair["offset_col_cells"])[relation["dimension_index"]]
        facts = FrozenMap({"current_pair_exact_reflection": True,
            "normal_tangent_basis_measured": True, "coordinate_scale": 2,
            "axis_normal_coordinate": relation["axis_coordinate_twice"],
            "source_normal_coordinate": 2*relation["second_normal_max"],
            "source_tangent_coordinate": 2*relation["tangent_min"],
            "desired_reflected_normal_coordinate": 2*(relation["first_normal_min"]+residual),
            "desired_reflected_tangent_coordinate": 2*(relation["tangent_min"]+
                (pair["offset_row_cells"], pair["offset_col_cells"])[1-relation["dimension_index"]]),
            "axis_normal_delta": 0, "source_normal_delta": 0, "source_tangent_delta": 0})
        projected = project_declared_alternatives(drm=drm, selection_policy_ref=policy_ref,
            alternative_refs=(), descriptive_facts=facts)
        command_rows, source_command_rows = [], []
        for step in command_steps:
            normal = (step["delta_row_cells"], step["delta_col_cells"])[relation["dimension_index"]]
            tangent = (step["delta_col_cells"], step["delta_row_cells"])[relation["dimension_index"]]
            counterfactual = FrozenMap.overlay(FrozenMap({"axis_normal_delta": 2*normal}), facts)
            prediction = project_declared_alternatives(drm=drm, selection_policy_ref=policy_ref,
                alternative_refs=(), descriptive_facts=counterfactual)
            command_rows.append(FrozenMap({"step": step, "axis_tangent_delta_cells": tangent,
                "facts": prediction.alternative_facts[unique_geometry_ref]}))
            source_counterfactual = FrozenMap.overlay(FrozenMap({"source_normal_delta": 2*normal,
                "source_tangent_delta": 2*tangent}), facts)
            source_prediction = project_declared_alternatives(drm=drm, selection_policy_ref=policy_ref,
                alternative_refs=(), descriptive_facts=source_counterfactual)
            source_command_rows.append(FrozenMap({"step": step,
                "facts": source_prediction.alternative_facts[unique_geometry_ref]}))
        results.append(FrozenMap({"pair_index": pair_index, "reflection_index": reflection_index,
            "policy_ref": policy_ref, "policy_source_hash": source.source_hash,
            "facts": projected.alternative_facts[unique_geometry_ref], "command_rows": tuple(command_rows),
            "source_command_rows": tuple(source_command_rows)}))
    return FrozenMap({"complete": True, "rows": tuple(results)})
