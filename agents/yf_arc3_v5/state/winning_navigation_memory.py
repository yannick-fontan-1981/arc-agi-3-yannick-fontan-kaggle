"""Bounded joins and geometry for a Source-requested winning cursor history.

DRM supplies roles, admissibility, semantic templates and bounds. This module
returns measurements; it cannot choose a goal, route, command or next action.
"""
from collections import defaultdict

from agents.yf_arc3_v5.capabilities import ComponentExtractionInput, FrameGrid, detect_components
from agents.yf_arc3_v5.capabilities.appearance import BearerAppearanceInput, measure_bearer_appearance
from agents.yf_arc3_v5.capabilities.navigation_transition_measurements import measure_remote_navigation_transition
from agents.yf_arc3_v5.logos.operations import ActionIntent, EnvironmentActionDispatchedRecord
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest


def measure_winning_navigation(*, snapshot, store, repository, transition_ref, profile, key_contract):
    empty = FrozenMap({"measurement_rows": (), "has_measurements": False,
        "evidence_refs": (), "state_revision": snapshot.revision})
    if repository is None or store is None:
        return empty

    def admitted(claim):
        return (claim is not None and claim.epistemic_status.value in profile["proof_statuses"]
            and claim.disposition.value == "active" and claim.polarity.value == "positive"
            and not claim.active_contradictions and not claim.defeated_by)

    outcomes = tuple(t for t in snapshot.terms_for_projection_contract(profile["outcome_contract"])
        if t.attributes.get("transition_ref") == transition_ref)
    associations = tuple(t for t in snapshot.terms_for_literal_attribute("role_ref", profile["cursor_role_ref"])
        if t.attributes.get("active") is True and t.attributes.get("normalized_valued_pixels"))
    links = snapshot.claims_for_predicate(profile["association_predicate"])
    outcome_links = snapshot.claims_for_predicate(profile["outcome_predicate"])
    records = store.artifacts_of_type(EnvironmentActionDispatchedRecord)
    if max(len(outcomes), len(associations), len(links), len(outcome_links)) > profile["maximum_records"]:
        raise ValueError("winning navigation canonical bound exceeded")
    if len(records) > profile["maximum_history_inputs"] or len(outcomes) != 1:
        return empty
    outcome, = outcomes
    proof = tuple(c for c in outcome_links if admitted(c) and c.arguments[0] == outcome.id
        and c.proof_rule == profile["outcome_rule"])
    official = tuple(c for c in snapshot.claims_for_predicate(profile["official_predicate"])
        if admitted(c) and len(c.arguments) == 2 and c.arguments[1] == transition_ref)
    history = outcome.attributes.get("observation_history") or {}
    indexed = tuple(history.get("prior_observation_rows") or ())
    if len(proof) != 1 or len(official) != 1 or not history.get("prior_observation_index_complete") or not indexed:
        return empty
    if len(indexed) > profile["maximum_observations"]:
        raise ValueError("winning navigation observation bound exceeded")
    observations = tuple(repository.get(row["raw_input_ref"]) for row in indexed)
    components = {}
    for observation in observations:
        if len(observation.frame) * len(observation.frame[0]) > profile["maximum_pixels"]:
            raise ValueError("winning navigation pixel bound exceeded")
        measured = detect_components(ComponentExtractionInput(
            frame=FrameGrid(rows=observation.frame), connectivity=profile["connectivity"])).components
        if len(measured) > profile["maximum_components"]:
            # No truncated component history may support navigation memory.
            # Source already branches on absent measurements and retains the
            # independent official outcome and other winning-memory families.
            return FrozenMap({**dict(empty), "measurement_count": 0,
                "component_measurement_complete": False,
                "observed_component_count": len(measured),
                "maximum_components": profile["maximum_components"],
                "component_bound_raw_input_ref": observation.raw_input_ref})
        components[observation.raw_input_ref] = measured
    dispatch_by_input = defaultdict(list)
    for record in records:
        dispatch_by_input[record.raw_input_ref].append(record)

    def sprite(component):
        return tuple(sorted((r-component.bbox.top, c-component.bbox.left, component.value)
            for r, c in component.pixels))

    def learning_row(component, endpoint, evidence_refs):
        role_ref = profile["roles"][endpoint]
        pixels = tuple((r, c, component.value) for r, c in component.pixels)
        token = stable_digest((role_ref, sprite(component), key_contract.source_hash))
        entity_ref = profile["entity_pattern"].format(association_token=token)
        claim_ref = profile["role_claim_pattern"].format(association_token=token)
        appearance = measure_bearer_appearance(BearerAppearanceInput(bearer_ref=entity_ref,
            valued_pixels=pixels, measurement_basis_ref=profile["measurement_basis_ref"],
            key_contract=key_contract))
        return FrozenMap({**dict(profile["role_capture_neutral_facts"]),
            "association_token": token, "causal_role_ref": role_ref,
            "entity_ref": entity_ref, "role_claim_ref": claim_ref,
            "source_role_term_ref": entity_ref,
            "source_claim_refs": (claim_ref,), "row_evidence_refs": evidence_refs,
            "signature_keys": tuple(FrozenMap(k.model_dump(mode="json")) for k in appearance.keys),
            "normalized_valued_pixels": sprite(component),
            "orthogonal_band_measurements": appearance.descriptive_delta["orthogonal_band_measurements"],
            "contract_source_hash": key_contract.source_hash,
            "mechanism_schema_refs": tuple(profile["mechanism_schema_refs"]),
            "falsifier_refs": tuple(profile["falsifier_refs"]),
            "role_context_requirements": FrozenMap()})

    rows, all_evidence = [], set()
    for association in associations:
        association_links = tuple(c for c in links if admitted(c) and c.arguments == (association.id,))
        if len(association_links) != 1:
            continue
        signature = tuple(tuple(pixel) for pixel in association.attributes["normalized_valued_pixels"])
        matched = [tuple(c for c in components[o.raw_input_ref] if sprite(c) == signature)
            for o in observations]
        if any(len(group) != 1 for group in matched):
            continue
        actors = tuple(group[0] for group in matched)
        deltas, remote_rows, motion_count = defaultdict(set), [], 0
        movement_records = defaultdict(list)
        entered_patterns = defaultdict(list)
        for index in range(1, len(observations)):
            before, after = observations[index-1:index+1]
            dispatched = dispatch_by_input[after.raw_input_ref]
            if len(dispatched) != 1 or dispatched[0].frame_id != before.frame_id:
                continue
            record, = dispatched
            delta = (actors[index].bbox.top-actors[index-1].bbox.top,
                actors[index].bbox.left-actors[index-1].bbox.left)
            if delta != (0, 0):
                deltas[record.action_ref].add(delta)
                movement_records[record.action_ref].append(record.id)
                entered_patterns[(actors[index].bbox.height, actors[index].bbox.width,
                    tuple(before.frame[r][c] for r, c in sorted(actors[index].pixels)))].append(record.id)
                motion_count += 1
            elif record.action_ref == profile["point_action_ref"]:
                intent = store.artifact(record.action_intent_id)
                if not isinstance(intent, ActionIntent):
                    continue
                x, y = intent.action_data.get("x"), intent.action_data.get("y")
                if type(x) is not int or type(y) is not int:
                    continue
                controls = tuple(c for c in components[before.raw_input_ref] if (y, x) in c.pixels)
                if len(controls) != 1:
                    continue
                changed = frozenset((r, c) for r in range(len(before.frame))
                    for c in range(len(before.frame[r])) if before.frame[r][c] != after.frame[r][c])
                outside = changed - frozenset(controls[0].pixels) - frozenset(actors[index].pixels)
                later_occupied = frozenset(pixel for later in actors[index+1:] for pixel in later.pixels)
                if outside & later_occupied:
                    top = min(r for r, c in outside)
                    left = min(c for r, c in outside)
                    remote_rows.append((controls[0], FrozenMap({
                        "typed_transition_measurements": measure_remote_navigation_transition(
                            before=before.frame, after=after.frame,
                            before_components=components[before.raw_input_ref],
                            after_components=components[after.raw_input_ref],
                            control=controls[0], actor=actors[index],
                            maximum_pixels=profile["maximum_pixels"]),
                        "dispatch_record_ref": record.id,
                        "actor_unchanged": True,
                        "affected_pixel_count": len(outside),
                        "subsequently_occupied_pixel_count": len(outside & later_occupied),
                        "before_relative_valued_pixels": tuple(sorted(
                            (r-top, c-left, before.frame[r][c]) for r, c in outside)),
                        "after_relative_valued_pixels": tuple(sorted(
                            (r-top, c-left, after.frame[r][c]) for r, c in outside)),
                    })))
        contract = outcome.attributes.get("executed_action_contract") or {}
        vectors = deltas.get(contract.get("action_ref"), set())
        if len(vectors) != 1 or contract.get("frame_id") != observations[-1].frame_id:
            continue
        dr, dc = next(iter(vectors))
        actor = actors[-1]
        projected = frozenset((r+dr, c+dc) for r,c in actor.pixels)
        targets = tuple(c for c in components[observations[-1].raw_input_ref]
            if frozenset(c.pixels) == projected and c.value != actor.value)
        if len(targets) != 1:
            continue
        target, = targets
        true_count = sum(frozenset(c.pixels) == projected for c in actors)
        unknown_count = sum(not all(o.frame[r][c] == target.value for r,c in target.pixels)
            and frozenset(actor_at.pixels) != projected for o, actor_at in zip(observations, actors))
        evidence_refs = tuple(sorted({proof[0].id, official[0].id, association_links[0].id}))
        learning = [learning_row(actor, "moving", evidence_refs), learning_row(target, "fixed", evidence_refs)]
        controls = {}
        for control, effect in remote_rows:
            controls.setdefault(sprite(control), control)
        learning.extend(learning_row(controls[key], "control", evidence_refs) for key in sorted(controls))
        control_refs = {key: profile["association_pattern"].format(
            association_token=row["association_token"])
            for key, row in zip(sorted(controls), learning[2:])}
        remote_effects = tuple(FrozenMap.overlay(effect, FrozenMap({
            "control_association_ref": control_refs[sprite(control)]}))
            for control, effect in remote_rows)
        command_translations = tuple(FrozenMap({"action_ref": action,
            "observed_delta_rows": tuple(sorted(vectors)),
            "observation_count": len(movement_records[action]),
            "dispatch_record_refs": tuple(movement_records[action]),
            "source_actor_extent": (actor.bbox.height, actor.bbox.width),
            "translation_is_unique": len(vectors) == 1})
            for action, vectors in sorted(deltas.items()))
        calibrations = tuple(sorted({(c.bbox.height, c.bbox.width,
            c.bbox.top % c.bbox.height, c.bbox.left % c.bbox.width)
            for c in actors if c.area == c.bbox.height * c.bbox.width}))
        homogeneous_count = sum(c.area == c.bbox.height * c.bbox.width for c in actors)
        calibration_rows = tuple(FrozenMap({
            "cell_height": h, "cell_width": w, "row_pitch": h, "col_pitch": w,
            "row_phase": ro, "col_phase": co,
            "observed_pose_count": len(actors),
            "homogeneous_pose_count": homogeneous_count,
            "compatible_translation_count": sum(dr % h == 0 and dc % w == 0
                for action in sorted(deltas) for dr, dc in sorted(deltas[action])),
            "translation_count": sum(len(deltas[action]) for action in sorted(deltas)),
        }) for h, w, ro, co in calibrations)
        moving_ref = profile["association_pattern"].format(association_token=learning[0]["association_token"])
        fixed_ref = profile["association_pattern"].format(association_token=learning[1]["association_token"])
        token = stable_digest(((moving_ref,), (fixed_ref,)))
        coverage = FrozenMap({"relation_token": token, "moving_association_refs": (moving_ref,),
            "fixed_association_refs": (fixed_ref,), "predicted_supports_equal": True,
            "uncovered_pixel_count_before": len(projected - frozenset(actor.pixels)),
            "row_gap_before": abs(target.bbox.top-actor.bbox.top),
            "column_gap_before": abs(target.bbox.left-actor.bbox.left),
            "row_gap_after": 0, "column_gap_after": 0,
            "predicted_outside_pixel_count": 0, "history_true_count": true_count,
            "history_false_count": len(observations)-true_count-unknown_count,
            "history_unknown_count": unknown_count, "history_evaluation_complete": unknown_count == 0,
            "history_first_counterexample_ref": next((o.raw_input_ref for o, c in zip(observations, actors)
                if frozenset(c.pixels) == projected), None),
            "geometry_evidence_ref": outcome.id, "row_evidence_refs": evidence_refs})
        rows.append(FrozenMap({"relation_token": token, "learning_rows": tuple(learning),
            "coverage_rows": (coverage,), "motion_observation_count": motion_count,
            "remote_change_count": len(remote_rows), "unique_remote_control_count": len(controls),
            "history_true_count": true_count, "history_unknown_count": unknown_count,
            "history_evaluation_complete": unknown_count == 0,
            "predicted_supports_equal": True, "evidence_refs": evidence_refs,
            "transition_ref": transition_ref,
            "completion_event_ref": official[0].arguments[0], "official_boundary_present": True,
            "method_support_claim_ref": profile["method_support_claim_pattern"].format(transition_ref=transition_ref),
            "workflow_memory_claim_ref": profile["workflow_claim_pattern"].format(relation_token=token, transition_ref=transition_ref),
            "moving_association_refs": (moving_ref,), "fixed_association_refs": (fixed_ref,),
            "observed_command_translation_rows": command_translations,
            "observed_remote_effect_rows": remote_effects,
            "observed_entered_surface_rows": tuple(FrozenMap({
                "cell_height": h, "cell_width": w, "before_pattern": pattern,
                "dispatch_record_refs": tuple(entered_patterns[h, w, pattern]),
                "observation_count": len(entered_patterns[h, w, pattern])})
                for h, w, pattern in sorted(entered_patterns)),
            "observed_lattice_calibration_rows": calibration_rows,
            "control_association_refs": tuple(profile["association_pattern"].format(association_token=r["association_token"]) for r in learning[2:])}))
        all_evidence.update(evidence_refs)
        if len(rows) > profile["maximum_rows"]:
            raise ValueError("winning navigation candidate bound exceeded")
    # Ambiguity is reported to Source, never resolved by Python ordering.
    return FrozenMap({**(dict(rows[0]) if len(rows) == 1 else {}),
        "measurement_rows": tuple(rows), "has_measurements": bool(rows),
        "measurement_count": len(rows), "evidence_refs": tuple(sorted(all_evidence)),
        "state_revision": snapshot.revision})
