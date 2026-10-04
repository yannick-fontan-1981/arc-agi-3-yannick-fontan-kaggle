"""Exact application measurements of the DRM-projected reflected source step."""

from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest


def measure_inverse_source_path(*, geometry, dimension, command_deltas, max_depth):
    """One bounded existing-BFS witness to DRM's scene-local inverse target.

    This measurement neither chooses a role nor releases an action. The caller
    must separately bind current control, supports, constraints and evidence.
    As with the reused BFS, paths requiring temporary residual growth are not
    represented by this monotone search.
    """
    from agents.yf_arc3_v5.capabilities.interaction import _bfs_residual_path

    empty = FrozenMap({"complete": False, "action_refs": ()})
    fields = ("required_source_normal_delta", "required_source_tangent_delta")
    if dimension not in (0, 1) or not all(field in geometry for field in fields):
        return empty
    scale = geometry.get("coordinate_scale", 0)
    if type(scale) is not int or scale <= 0:
        return empty
    deltas = tuple(geometry[field] for field in fields)
    if any(type(delta) is not int or delta % scale for delta in deltas):
        return empty
    if type(max_depth) is not int or not 1 <= max_depth <= 64 or len(command_deltas) > 32:
        raise ValueError("inverse source path search bound exceeded")
    if any(len(delta) != 2 or any(type(v) is not int for v in delta)
            or (delta[0] == 0) == (delta[1] == 0)
            for action in sorted(command_deltas) for delta in (command_deltas[action],)):
        raise ValueError("inverse source path requires exact cardinal cell deltas")
    normal, tangent = (delta // scale for delta in deltas)
    residual = (normal, tangent) if dimension == 0 else (tangent, normal)
    path = _bfs_residual_path(residual_row=2*residual[0], residual_col=2*residual[1],
        action_deltas=command_deltas, couple=(1, 1), max_depth=max_depth)
    return FrozenMap({"complete": bool(path) or residual == (0, 0),
        "action_refs": path, "residual_cells": residual})


def measure_memorial_source_steps(inventories, control_rows):
    if len(inventories) > 64 or len(control_rows) > 64:
        raise ValueError("memorial source input bound exceeded")
    result, comparisons = {}, 0
    for inventory in inventories:
        if len(inventory.get("cell_geometry_rows", ())) > 64:
            raise ValueError("memorial source grid bound exceeded")
        for grid in inventory.get("cell_geometry_rows", ()):
            geometry = grid["declared_geometry"]
            if not geometry["complete"]:
                continue
            if len(geometry["rows"]) > 64 or len(grid.get("support_member_rows", ())) > 128:
                raise ValueError("memorial source geometry bound exceeded")
            supports = {row["support_ref"]: row for row in grid.get("support_member_rows", ())}
            for projection in geometry["rows"]:
                relation = grid["reflection_measurements"]["rows"][projection["reflection_index"]]
                pair = grid["pair_rows"][projection["pair_index"]]
                source, reflected = supports.get(relation["second_support_ref"]), supports.get(relation["first_support_ref"])
                if not source or not reflected or not source["complete_cell_coverage"] or not reflected["complete_cell_coverage"]:
                    continue
                members = frozenset(source["member_refs"])
                focus_sets = tuple(frozenset(row.get("member_refs") or (row.get("entity_ref"),)) for row in control_rows)
                if len(members) > 128 or any(len(group) > 128 for group in focus_sets):
                    raise ValueError("memorial source membership bound exceeded")
                focus_matches = bool(focus_sets) and all(group and group <= members for group in focus_sets)
                focus_proofs = tuple(sorted({ref for row in control_rows for ref in row.get("role_premise_claim_refs", ())}))
                residual = (pair["offset_row_cells"], pair["offset_col_cells"])
                dimension = relation["dimension_index"]
                for command in projection.get("source_command_rows", ()):
                    comparisons += 1
                    if comparisons > 64:
                        raise ValueError("memorial source join bound exceeded; no partial candidates")
                    step, facts = command["step"], command["facts"]
                    scale = facts["coordinate_scale"]
                    normal, tangent = facts["expected_reflected_normal_delta"], facts["expected_reflected_tangent_delta"]
                    deltas = (normal, tangent) if dimension == 0 else (tangent, normal)
                    in_bounds = True
                    for support, shifts in ((reflected, deltas), (source, (scale*step["delta_row_cells"], scale*step["delta_col_cells"]))):
                        box = support.get("cell_bbox", ())
                        in_bounds = in_bounds and len(box) == 4 and all(
                            0 <= scale*box[i] + shifts[i] and scale*box[i+2] + shifts[i] < scale*grid["logical_extent"][i]
                            for i in (0,1))
                    refs = tuple(sorted({inventory["consultation_claim_ref"], step["proof_claim_ref"], *focus_proofs}))
                    remainder = scale*residual[1-dimension] - tangent
                    result.setdefault(step["action_ref"], []).append(FrozenMap({
                        "memorial_source_goal_ref": inventory["term_ref"],
                        "memorial_source_focus_matches": focus_matches,
                        "memorial_source_focus_proof_count": len(focus_proofs),
                        "memorial_source_normal_residual": residual[dimension],
                        "memorial_source_normal_delta": normal,
                        "memorial_source_residual_reduction": abs(scale*residual[1-dimension])-abs(remainder),
                        "memorial_source_sign_preserved": residual[1-dimension]*remainder >= 0,
                        "memorial_source_destination_in_grid": in_bounds,
                        "memorial_source_pair_same_shape": pair["same_cell_shape"],
                        "memorial_source_historical_goal_proof": inventory["historical_proof_present"],
                        "memorial_source_contexts_resolved": inventory["current_contexts_resolved"],
                        "memorial_source_counterexamples": inventory["historical_counterexample_count"],
                        "memorial_source_premise_refs": refs, "memorial_source_premise_count": len(refs),
                    }))
    return FrozenMap({ref: tuple(rows) for ref, rows in sorted(result.items())})


def enumerate_memorial_inverse_source_paths(value):
    """Measure at most three geometric witnesses; DRM decides applicability."""
    from agents.yf_arc3_v5.capabilities.contracts import InteractionProbeCandidate

    inventories = value.canonical_terminal_inventory_facts
    controls = value.canonical_provisional_control_term_facts
    # This shared measurement enforces the existing inventory/member bounds.
    step_rows = measure_memorial_source_steps(inventories, controls)
    result = []
    for inventory in inventories:
        for grid in inventory.get("cell_geometry_rows", ()):
            supports = {row["support_ref"]: row for row in grid.get("support_member_rows", ())}
            for projection in grid["declared_geometry"]["rows"]:
                relation = grid["reflection_measurements"]["rows"][projection["reflection_index"]]
                source = supports.get(relation["second_support_ref"])
                reflected = supports.get(relation["first_support_ref"])
                if not source or not reflected:
                    continue
                commands = {}
                for row in projection.get("source_command_rows", ()):
                    action = row["step"]["action_ref"]
                    if action not in value.available_action_refs:
                        continue
                    delta = (row["step"]["delta_row_cells"], row["step"]["delta_col_cells"])
                    commands.setdefault(action, {}).setdefault(delta, []).append(row)
                if any(len(commands[action]) != 1 for action in sorted(commands)):
                    continue
                deltas = {action: next(iter(commands[action])) for action in sorted(commands)}
                geometry = projection["facts"]
                witness = measure_inverse_source_path(geometry=geometry,
                    dimension=relation["dimension_index"], command_deltas=deltas,
                    max_depth=geometry["maximum_inverse_source_path_depth"])
                path = witness["action_refs"]
                if not witness["complete"] or not path:
                    continue
                first_rows = tuple(row for row in step_rows.get(path[0], ())
                    if row["memorial_source_goal_ref"] == inventory["term_ref"])
                if len(first_rows) != 1:
                    continue
                base, = first_rows
                from agents.yf_arc3_v5.capabilities.route_cell_layers import capture_cell_layer
                pair = grid['pair_rows'][projection['pair_index']]
                target = supports[pair['second_support_ref']]
                layers = FrozenMap()
                if hasattr(value, 'tracking'):
                    layers = FrozenMap({
                        'fixed': capture_cell_layer(value.tracking, grid['grid_geometry'], target),
                        'moving': capture_cell_layer(value.tracking, grid['grid_geometry'], reflected)})
                source_boxes = [tuple(source.get("cell_bbox", ()))]
                reflected_boxes = [tuple(reflected.get("cell_bbox", ()))]
                if len(source_boxes[0]) != 4 or len(reflected_boxes[0]) != 4:
                    continue
                refs = set(base["memorial_source_premise_refs"])
                in_bounds = True
                for action in path:
                    rows = commands[action][deltas[action]]
                    prediction = rows[0]["facts"]
                    scale = prediction["coordinate_scale"]
                    normal, tangent = (prediction["expected_reflected_normal_delta"], prediction["expected_reflected_tangent_delta"])
                    if normal % scale or tangent % scale:
                        in_bounds = False
                        break
                    reflected_delta = ((normal//scale, tangent//scale) if relation["dimension_index"] == 0
                        else (tangent//scale, normal//scale))
                    for boxes, delta in ((source_boxes, deltas[action]), (reflected_boxes, reflected_delta)):
                        box = tuple(coordinate + delta[i % 2] for i, coordinate in enumerate(boxes[-1]))
                        in_bounds = in_bounds and all(0 <= box[i] <= box[i+2] < grid["logical_extent"][i] for i in (0, 1))
                        boxes.append(box)
                    refs.update(row["step"]["proof_claim_ref"] for row in rows)
                token = stable_digest((inventory["term_ref"], source_boxes[0], reflected_boxes[0], path))[:24]
                candidate_ref = f"measurement.inverse_source_route.{token}"
                candidate = InteractionProbeCandidate(candidate_ref=candidate_ref, action_ref=path[0],
                    action_data=FrozenMap(), basis_refs=("measurement.residual_control_path",),
                    expectation_refs=())
                facts = FrozenMap.overlay(FrozenMap({
                    "candidate_ref": candidate_ref, "action_ref": path[0], "action_data": FrozenMap(),
                    "memorial_route_measured": True, "memorial_route_action_refs": path,
                    "memorial_route_initial": True,
                    "memorial_route_cell_layers": layers,
                    "memorial_route_step_count": len(path), "memorial_route_in_bounds": in_bounds,
                    "memorial_route_grid_geometry": grid["grid_geometry"],
                    "memorial_route_source_member_refs": tuple(source["member_refs"]),
                    "memorial_route_reflected_member_refs": tuple(reflected["member_refs"]),
                    "memorial_route_source_cells": source["normalized_cells"],
                    "memorial_route_reflected_cells": reflected["normalized_cells"],
                    "memorial_route_source_boxes": tuple(source_boxes),
                    "memorial_route_reflected_boxes": tuple(reflected_boxes),
                    "memorial_route_axis_coordinate_twice": relation["axis_coordinate_twice"],
                    "memorial_route_dimension": relation["dimension_index"],
                    "memorial_route_normal_change_count": sum(
                        left[relation["dimension_index"]] != right[relation["dimension_index"]]
                        for left, right in zip(reflected_boxes, reflected_boxes[1:])),
                    "memorial_route_premise_refs": tuple(sorted(refs)),
                    "memorial_route_candidate_previously_executed": candidate_ref in value.current_configuration_executed_candidate_refs,
                }), base)
                result.append((candidate, facts))
                if len(result) > 3:
                    raise ValueError("inverse source route witness bound exceeded")
    return tuple(result)


def measure_inverse_source_cursor(value, facts):
    """Compare the next observed poses with the committed prefix; no BFS."""
    if not facts.get("memorial_route_commitment_present"):
        return FrozenMap({"memorial_route_cursor_present": False})
    index = int(facts.get("executed_verified_step_count", 0)) + 1
    geometry = facts.get("memorial_route_grid_geometry", ())
    matches = len(geometry) == 9
    if not matches:
        return FrozenMap({"memorial_route_cursor_present": True, "memorial_route_cursor_matches": False})
    from agents.yf_arc3_v5.capabilities.member_cell_geometry import measure_member_cell_geometry
    supports = []
    for prefix in ("source", "reflected"):
        refs = facts.get(f"memorial_route_{prefix}_member_refs", ())
        boxes = facts.get(f"memorial_route_{prefix}_boxes", ())
        cells = facts.get(f"memorial_route_{prefix}_cells", ())
        if not refs or len(refs) > 128 or len(boxes) > 65 or index >= len(boxes) or not cells:
            return FrozenMap({"memorial_route_cursor_present": True, "memorial_route_cursor_matches": False})
        supports.append(FrozenMap({"support_ref": prefix, "top": boxes[index][0],
            "left": boxes[index][1], "normalized_cells": cells}))
    # Reuse exact cell-support joining: decorations can receive new tracking IDs,
    # but complete support coverage and a surviving member are both required.
    measured = measure_member_cell_geometry(groups=((), ()), tracking=value.tracking,
        geometry=geometry, supports=tuple(supports))
    members = {row["support_ref"]: frozenset(row["member_refs"]) for row in measured["support_member_rows"]}
    pose_matches = {row['support_ref']: row["complete_cell_coverage"] and
        bool(members[row["support_ref"]] & frozenset(facts[f'memorial_route_{row["support_ref"]}_member_refs']))
        for row in measured["support_member_rows"]}
    source_refs = members["source"]
    controls = value.canonical_provisional_control_term_facts
    focus_matches = bool(controls) and all(
        bool(group := frozenset(row.get("member_refs") or (row.get("entity_ref"),)))
        and group <= source_refs and bool(row.get("role_premise_claim_refs")) for row in controls)
    path = facts.get("verified_repetition_action_refs", ())
    rows = measure_memorial_source_steps(value.canonical_terminal_inventory_facts, controls)
    current_rows = tuple(row for row in rows.get(path[1], ())
        if row["memorial_source_goal_ref"] == facts.get("active_terminal_goal_ref")) if len(path) > 1 else ()
    current_refs = current_rows[0]["memorial_source_premise_refs"] if len(current_rows) == 1 else ()
    from agents.yf_arc3_v5.capabilities.route_cell_layers import measure_route_cell_layers
    layers = facts.get('memorial_route_cell_layers', FrozenMap())
    layer_measurement = measure_route_cell_layers(value.tracking, geometry,
        layers.get('fixed'), layers.get('moving'), facts['memorial_route_reflected_boxes'][index])
    # A rendered overlap can split the reflected component into new tracking
    # IDs. Exact frozen-layer sample correspondence is then a second geometric
    # witness; the controlled source and its current selection still match.
    matches = pose_matches['source'] and focus_matches and (
        pose_matches['reflected'] or layer_measurement['matches'])
    inventory_matches = tuple(row for row in value.canonical_terminal_inventory_facts
        if row.get('term_ref') == facts.get('active_terminal_goal_ref'))
    layer_refs = ()
    layer_history, layer_context, layer_counterexamples = False, False, None
    if len(inventory_matches) == 1 and len(path)>1:
        inventory, = inventory_matches
        steps = tuple(step for grid in inventory.get('cell_geometry_rows', ())
            if tuple(grid['grid_geometry']) == tuple(geometry)
            for step in grid.get('command_step_rows', ()) if step['action_ref'] == path[1])
        if steps:
            layer_refs = tuple(sorted({inventory['consultation_claim_ref'],
                *(step['proof_claim_ref'] for step in steps),
                *(ref for row in controls for ref in row.get('role_premise_claim_refs',()))}))
        layer_history = inventory.get('historical_proof_present')
        layer_context = inventory.get('current_contexts_resolved')
        layer_counterexamples = inventory.get('historical_counterexample_count')
    return FrozenMap({"memorial_route_cursor_present": True, "memorial_route_cursor_matches": matches,
        'memorial_route_layer_matches': layer_measurement['matches'],
        'memorial_route_layer_premise_refs': layer_refs,
        'memorial_route_layer_premise_count': len(layer_refs),
        'memorial_route_layer_revalidation': FrozenMap({
            'prior_premise_refs': tuple(facts.get('canonical_premise_claim_refs', ())),
            'current_premise_refs': layer_refs, 'terminal_goal_ref': facts.get('active_terminal_goal_ref'),
            'frame_ref': value.tracking.frame_ref}),
        'memorial_route_layer_history_present': layer_history,
        'memorial_route_layer_context_resolved': layer_context,
        'memorial_route_layer_counterexample_count': layer_counterexamples,
        "memorial_route_current_premises_complete": len(current_rows) == 1 and len(current_refs) > 2,
        "memorial_route_current_premise_refs": current_refs,
        "memorial_route_premise_revalidation": FrozenMap({
            "prior_premise_refs": tuple(facts.get("canonical_premise_claim_refs", ())),
            "current_premise_refs": current_refs,
            "terminal_goal_ref": facts.get("active_terminal_goal_ref"),
            "frame_ref": value.tracking.frame_ref,
        })})


def annotate_memorial_source_steps(value, agenda):
    if not value.canonical_terminal_inventory_facts:
        return agenda
    rows = measure_memorial_source_steps(value.canonical_terminal_inventory_facts,
        value.canonical_provisional_control_term_facts)
    deltas = {}
    for candidate in agenda.candidates:
        matches = rows.get(candidate.action_ref, ()) if candidate.point is None else ()
        delta = FrozenMap({"memorial_source_join_count": len(matches)})
        if len(matches) == 1:
            unique_row, = matches
            delta = FrozenMap.overlay(delta, unique_row)
        deltas[candidate.candidate_ref] = delta
    return agenda.model_copy(update={
        "alternative_facts": FrozenMap({ref: FrozenMap.overlay(deltas.get(ref, FrozenMap()), facts)
            for ref, facts in sorted(agenda.alternative_facts.items())}),
        "descriptive_delta_facts": FrozenMap({ref: FrozenMap.overlay(deltas.get(ref, FrozenMap()), facts)
            for ref, facts in sorted(agenda.descriptive_delta_facts.items())}),
    })
