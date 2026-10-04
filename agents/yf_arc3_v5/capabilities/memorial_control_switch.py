"""Bounded current-support joins for a declared control-switch experiment.

This module measures coincidences and residuals. It neither chooses an action
nor treats a recalled method as executable authority.
"""

from agents.yf_arc3_v5.logos.types import FrozenMap


def measure_memorial_control_switches(terminals, methods, unselected):
    if any(len(rows) > 64 for rows in (terminals, methods, unselected)):
        raise ValueError("memorial control input bound exceeded")
    rows, comparisons = [], 0
    for terminal in terminals:
        if len(terminal.get("cell_geometry_rows", ())) > 64:
            raise ValueError("memorial control grid bound exceeded")
        for grid in terminal.get("cell_geometry_rows", ()):
            geometry = grid["declared_geometry"]
            if not geometry["complete"]:
                continue
            if len(geometry["rows"]) > 256 or len(grid.get("support_member_rows", ())) > 128:
                raise ValueError("memorial control geometry bound exceeded")
            supports = {row["support_ref"]: row for row in grid.get("support_member_rows", ())}
            for projection in geometry["rows"]:
                relation = grid["reflection_measurements"]["rows"][projection["reflection_index"]]
                pair = grid["pair_rows"][projection["pair_index"]]
                source = supports.get(relation["second_support_ref"])
                if source is None or not source["complete_cell_coverage"]:
                    continue
                source_members = frozenset(source["member_refs"])
                carriers = frozenset(relation["spanning_coincident_entity_refs"])
                for method in methods:
                    comparisons += 1
                    if comparisons > 64:
                        raise ValueError("memorial control join bound exceeded; no partial result")
                    if any(len(method.get(field, ())) > 64 or any(len(group) > 128 for group in method.get(field, ()))
                        for field in ("control_expected_member_rows", "current_control_member_rows")):
                        raise ValueError("memorial control membership bound exceeded")
                    expected = tuple(frozenset(group) for group in method.get("control_expected_member_rows", ()))
                    matched = tuple(group for group in expected if group and group <= source_members)
                    # An unrelated activity has no joined source bearer. Its
                    # presence must not turn one exact support join into two.
                    # Every actual match remains for DRM to assess, including
                    # ambiguous matches and rows with missing causal proofs.
                    if not matched:
                        continue
                    current = tuple(frozenset(group) for group in method.get("current_control_member_rows", ()))
                    operators = tuple(row for row in unselected
                        if frozenset(row.get("member_refs") or (row.get("entity_ref"),)) in matched)
                    refs = tuple(sorted({terminal["consultation_claim_ref"], method["consultation_claim_ref"],
                        *(ref for row in operators for ref in row.get("role_premise_claim_refs", ()))}))
                    residual = (pair["offset_row_cells"], pair["offset_col_cells"])
                    rows.append(FrozenMap({
                        "memorial_switch_goal_ref": terminal["term_ref"],
                        "memorial_switch_source_support_ref": source["support_ref"],
                        "memorial_switch_source_match_count": len(matched),
                        "memorial_switch_current_axis_match": bool(current) and all(bool(group & carriers) for group in current),
                        "memorial_switch_source_disjoint": bool(current) and all(not group & source_members for group in current),
                        "memorial_switch_discrete_operator_count": sum("discrete_focus_switch" in row.get("possible_selection_operator_kinds", ()) for row in operators),
                        "memorial_switch_normal_residual": residual[relation["dimension_index"]],
                        "memorial_switch_tangent_residual_abs": abs(residual[1-relation["dimension_index"]]),
                        "memorial_switch_pair_same_shape": pair["same_cell_shape"],
                        "memorial_switch_historical_goal_proof": terminal["historical_proof_present"],
                        "memorial_switch_contexts_resolved": terminal["current_contexts_resolved"],
                        "memorial_switch_counterexamples": terminal["historical_counterexample_count"],
                        "memorial_switch_method_proof": method["activity_proof_present"],
                        "memorial_switch_control_roles_complete": method["control_expected_roles_complete"],
                        "memorial_switch_premise_refs": refs,
                        "memorial_switch_premise_count": len(refs),
                    }))
    return tuple(rows)


def annotate_memorial_control_switches(value, agenda):
    rows = measure_memorial_control_switches(value.canonical_terminal_inventory_facts,
        value.canonical_method_inventory_facts, value.canonical_unselected_control_facts)
    nondirectional_count = sum(facts.get("interaction_measurement_kind") == "discrete"
        and facts.get("is_directional_interface_action") is False
        for _, facts in sorted(agenda.alternative_facts.items()))
    delta = FrozenMap({"memorial_switch_join_count": len(rows),
        "memorial_switch_nondirectional_count": nondirectional_count})
    if len(rows) == 1:
        unique_row, = rows
        delta = FrozenMap.overlay(delta, unique_row)
    agenda = agenda.model_copy(update={
        "alternative_facts": FrozenMap({ref: FrozenMap.overlay(delta, facts)
            for ref, facts in sorted(agenda.alternative_facts.items())}),
        "descriptive_delta_facts": FrozenMap({ref: FrozenMap.overlay(delta, facts)
            for ref, facts in sorted(agenda.descriptive_delta_facts.items())}),
    })
    return annotate_unbound_source_control_requirement(value, agenda)


def annotate_unbound_source_control_requirement(value, agenda):
    """Transport exact current geometry and receiver membership, never a switch law."""
    inventories = value.canonical_terminal_inventory_facts
    controls = value.canonical_provisional_control_term_facts
    if len(inventories) > 64 or len(controls) > 64:
        raise ValueError("source receiver requirement input bound exceeded")
    descriptions = {}
    visits = 0
    current = tuple(frozenset(row.get("member_refs") or (row.get("entity_ref"),))
        for row in controls)
    if any(len(group) > 128 for group in current):
        raise ValueError("source receiver membership bound exceeded")
    for inventory in inventories:
        grids = inventory.get("cell_geometry_rows", ())
        if len(grids) > 64:
            raise ValueError("source receiver grid bound exceeded")
        for grid in grids:
            geometry = grid["declared_geometry"]
            if not geometry["complete"]:
                continue
            supports = {row["support_ref"]:row for row in grid.get("support_member_rows", ())}
            for projection in geometry["rows"]:
                visits += 1
                if visits > 64:
                    raise ValueError("source receiver join bound exceeded; no partial result")
                relation = grid["reflection_measurements"]["rows"][projection["reflection_index"]]
                pair = grid["pair_rows"][projection["pair_index"]]
                source = supports.get(relation["second_support_ref"])
                if source is None:
                    continue
                members = frozenset(source["member_refs"])
                carriers = frozenset(relation["spanning_coincident_entity_refs"])
                if len(members) > 128 or len(carriers) > 128:
                    raise ValueError("source receiver support bound exceeded")
                residual = (pair["offset_row_cells"], pair["offset_col_cells"])
                dimension = relation["dimension_index"]
                refs = tuple(sorted({inventory["consultation_claim_ref"],
                    *(ref for row in controls for ref in row.get("role_premise_claim_refs", ()))}))
                key = (inventory["term_ref"], source["support_ref"], tuple(sorted(carriers)), residual)
                descriptions[key] = FrozenMap({
                    "source_receiver_goal_ref":inventory["term_ref"],
                    "source_receiver_current_axis_match":bool(current) and all(bool(group & carriers) for group in current),
                    "source_receiver_source_disjoint":bool(current) and all(not group & members for group in current),
                    "source_receiver_source_complete":source["complete_cell_coverage"],
                    "source_receiver_pair_same_shape":pair["same_cell_shape"],
                    "source_receiver_normal_residual":residual[dimension],
                    "source_receiver_tangent_residual_abs":abs(residual[1-dimension]),
                    "source_receiver_historical_goal_proof":inventory["historical_proof_present"],
                    "source_receiver_contexts_resolved":inventory["current_contexts_resolved"],
                    "source_receiver_counterexamples":inventory["historical_counterexample_count"],
                    "source_receiver_premise_refs":refs, "source_receiver_premise_count":len(refs)})
    modes = tuple(c for c in agenda.candidates if c.point is None
        and c.action_ref != value.level_start_checkpoint_action_ref
        and agenda.alternative_facts[c.candidate_ref].get("interaction_measurement_kind") == "discrete"
        and agenda.alternative_facts[c.candidate_ref].get("is_directional_interface_action") is False
        and int(dict(value.action_execution_counts).get(c.action_ref, 0)) == 0)
    delta = FrozenMap({"source_receiver_description_count":len(descriptions),
        "source_receiver_unmeasured_control_count":len(modes)})
    if len(descriptions) == 1:
        delta = FrozenMap.overlay(next(iter(descriptions.values())), delta)
    mode_refs = frozenset(c.candidate_ref for c in modes)
    def enrich(ref, facts):
        return FrozenMap.overlay(FrozenMap({"source_receiver_unmeasured_control_candidate":ref in mode_refs,
            "source_receiver_interface_ordinal":value.available_action_refs.index(facts["action_ref"])
                if facts.get("action_ref") in value.available_action_refs else len(value.available_action_refs)}),
            FrozenMap.overlay(delta, facts))
    return agenda.model_copy(update={
        "context_facts":FrozenMap.overlay(delta, agenda.context_facts),
        "alternative_facts":FrozenMap({ref:enrich(ref,facts) for ref,facts in sorted(agenda.alternative_facts.items())}),
        "descriptive_delta_facts":FrozenMap({ref:enrich(ref,facts) for ref,facts in sorted(agenda.descriptive_delta_facts.items())})})
