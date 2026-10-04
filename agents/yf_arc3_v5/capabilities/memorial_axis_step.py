"""Bounded equality joins and arithmetic on source-projected cell geometry.

No candidate is selected here. Ambiguous joins remain counted, rather than
being silently collapsed into one action contract.
"""

from agents.yf_arc3_v5.logos.types import FrozenMap


def measure_memorial_axis_steps(inventories, control_rows):
    if len(inventories) > 64 or len(control_rows) > 64:
        raise ValueError("memorial axis input bound exceeded")
    result = {}
    comparisons = 0
    for inventory in inventories:
        for grid in inventory.get("cell_geometry_rows", ()):
            geometry = grid["declared_geometry"]
            if not geometry["complete"]:
                continue
            for projection in geometry["rows"]:
                relation = grid["reflection_measurements"]["rows"][projection["reflection_index"]]
                carriers = relation["spanning_coincident_entity_refs"]
                focus = tuple(row for row in control_rows if row.get("entity_ref") in carriers)
                focus_refs = tuple(sorted({row["entity_ref"] for row in focus}))
                for command in projection.get("command_rows", ()):
                    comparisons += 1
                    if comparisons > 64:
                        raise ValueError("memorial axis join bound exceeded; no partial candidates")
                    step, facts = command["step"], command["facts"]
                    residual = facts["reflected_normal_residual"]
                    delta = facts["expected_reflected_normal_delta"]
                    scale = facts["coordinate_scale"]
                    normal_limit = grid["logical_extent"][relation["dimension_index"]]
                    axis_next = facts["axis_normal_coordinate"] + facts["axis_normal_delta"]
                    in_bounds = (0 <= axis_next <= scale*(normal_limit-1)
                        and 0 <= scale*relation["first_normal_min"] + delta
                        and scale*relation["first_normal_max"] + delta < scale*normal_limit)
                    refs = tuple(sorted({inventory["consultation_claim_ref"], step["proof_claim_ref"],
                        *(ref for row in focus for ref in row.get("role_premise_claim_refs", ()))}))
                    row = FrozenMap({
                        "memorial_axis_step_goal_ref": inventory["term_ref"],
                        "memorial_axis_step_premise_refs": refs,
                        "memorial_axis_step_premise_count": len(refs),
                        "memorial_axis_step_focus_count": len(focus_refs),
                        "memorial_axis_step_focus_proof_count": len({ref for row in focus for ref in row.get("role_premise_claim_refs", ())}),
                        "memorial_axis_step_tangent_delta_cells": command["axis_tangent_delta_cells"],
                        "memorial_axis_step_residual_reduction": abs(residual)-abs(residual-delta),
                        "memorial_axis_step_residual_sign_preserved": residual*(residual-delta) >= 0,
                        "memorial_axis_step_destination_in_grid": in_bounds,
                        "memorial_axis_step_historical_goal_proof": inventory["historical_proof_present"],
                        "memorial_axis_step_contexts_resolved": inventory["current_contexts_resolved"],
                        "memorial_axis_step_goal_counterexamples": inventory["historical_counterexample_count"],
                        "memorial_axis_step_axis_delta_scaled": facts["axis_normal_delta"],
                        "memorial_axis_step_reflected_delta_scaled": delta,
                        "memorial_axis_step_coordinate_scale": scale,
                    })
                    result.setdefault(step["action_ref"], []).append(row)
    return FrozenMap({action: tuple(rows) for action, rows in sorted(result.items())})


def annotate_memorial_axis_steps(value, agenda):
    if not value.canonical_terminal_inventory_facts:
        return agenda
    rows = measure_memorial_axis_steps(value.canonical_terminal_inventory_facts,
        value.canonical_provisional_control_term_facts)
    by_ref = {}
    for candidate in agenda.candidates:
        matches = rows.get(candidate.action_ref, ()) if candidate.point is None else ()
        delta = FrozenMap({"memorial_axis_step_join_count": len(matches)})
        if len(matches) == 1:
            unique_row, = matches
            delta = FrozenMap.overlay(delta, unique_row)
        by_ref[candidate.candidate_ref] = delta
    return agenda.model_copy(update={
        "alternative_facts": FrozenMap({ref: FrozenMap.overlay(by_ref.get(ref, FrozenMap()), facts)
            for ref, facts in sorted(agenda.alternative_facts.items())}),
        "descriptive_delta_facts": FrozenMap({ref: FrozenMap.overlay(by_ref.get(ref, FrozenMap()), facts)
            for ref, facts in sorted(agenda.descriptive_delta_facts.items())}),
    })
