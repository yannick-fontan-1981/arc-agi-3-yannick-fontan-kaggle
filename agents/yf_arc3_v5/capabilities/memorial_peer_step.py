"""Bounded geometric joins for both assignments of a remembered pair.

This module measures hypotheses; DRM owns their roles, goal and release.
Historical pixel coordinates and action sequences are never inputs.
"""

from agents.yf_arc3_v5.logos.types import FrozenMap


def order_peers_by_measured_interface_alignment(cells, positions, aligned_bboxes):
    """Join a unique current measured body; neither row nor side assigns it."""
    aligned = tuple(position for position in positions
        if cells[position].bbox in aligned_bboxes)
    if len(aligned) != 1:
        return None
    return (aligned[0], *(position for position in positions if position != aligned[0]))


def measure_unmatched_floor_mixture_pattern_refs(cells, known_palette_rows):
    """Return mixtures sharing known floor colors but not its exact palette."""
    rows = frozenset(tuple(row) for row in known_palette_rows)
    values = frozenset(value for row in rows for value in row)
    return frozenset(cell.pattern_ref for cell in cells
        if tuple(cell.palette_values) not in rows
        and len(cell.palette_values) > 1
        and any(value in values for value in cell.palette_values))


def measured_peer_route_packet_matches(frame, facts):
    """Exact visible footprint comparison with the declared next pair state."""
    states = facts.get("peer_route_expected_pixel_boxes", ())
    index = int(facts.get("executed_verified_step_count", 0))
    palette = tuple(facts.get("peer_palette_values", ()))
    if len(palette) != 1 or not 0 <= index < len(states):
        return False
    expected = set()
    for top, left, height, width in states[index]:
        if top < 0 or left < 0 or top+height > len(frame) or left+width > len(frame[0]):
            return False
        expected.update((r, c) for r in range(top, top+height) for c in range(left, left+width))
    actual = {(r, c) for r, row in enumerate(frame) for c, value in enumerate(row)
        if value == palette[0]}
    return actual == expected


def measure_memorial_peer_steps(memories, grids, available_actions):
    if len(memories) > 16 or len(grids) > 16:
        raise ValueError("peer memory join bound exceeded")
    result = {}
    for memory in memories:
        palette = tuple(memory["peer_palette_values"])
        passable = {tuple(row) for row in memory["peer_passable_palette_rows"]}
        for grid in grids:
            peers = tuple(c for c in grid.cells if tuple(c.palette_values) == palette)
            if len(peers) != 2:
                continue
            cells = {(c.row, c.col): c for c in grid.cells}
            positions = tuple((c.row, c.col) for c in peers)
            before = sum(abs(a-b) for a, b in zip(*positions))
            for action, dr, dc in memory["peer_interface_delta_rows"]:
                if action not in available_actions:
                    continue
                outcomes = []
                unknown_count = 0
                for assignment in ((0, 1), (1, 0)):
                    destination = list(positions)
                    for member, position in enumerate(positions):
                        factor_r = 1 if assignment[member] == 0 else memory["peer_coupling_row"]
                        factor_c = 1 if assignment[member] == 0 else memory["peer_coupling_column"]
                        target = (position[0] + dr * factor_r, position[1] + dc * factor_c)
                        cell = cells.get(target)
                        if cell is None:
                            unknown_count += 1
                        elif tuple(cell.palette_values) in passable or target in positions:
                            destination[member] = target
                        else:
                            # A newly encountered texture is not a proved wall.
                            unknown_count += 1
                    outcomes.append(tuple(destination))
                distances = [sum(abs(a-b) for a,b in zip(*row)) for row in outcomes]
                facts = FrozenMap({
                    "memorial_peer_goal_ref": memory["term_ref"],
                    "memorial_peer_premise_refs": memory["role_premise_claim_refs"],
                    "memorial_peer_assignment_count": len(outcomes),
                    "memorial_peer_distinct_outcome_count": len(set(outcomes)),
                    "memorial_peer_unknown_destination_count": unknown_count,
                    "memorial_peer_best_distance_reduction": before - min(distances),
                    "memorial_peer_max_distance_increase": max(distances) - before,
                    "memorial_peer_current_extent": (grid.cell_height, grid.cell_width),
                    "memorial_peer_current_positions": positions,
                    "memorial_peer_predicted_assignment_outcomes": tuple(outcomes),
                    "memorial_peer_step_has_horizontal_component": dc != 0,
                })
                result.setdefault(action, []).append(facts)
    return result


def annotate_memorial_peer_steps(value, agenda):
    memories = value.canonical_peer_method_facts
    if not memories:
        return agenda
    rows = measure_memorial_peer_steps(memories, value.periodic_cell_grids.candidates,
        value.available_action_refs)
    additions = {}
    for candidate in agenda.candidates:
        matches = rows.get(candidate.action_ref, ()) if candidate.point is None else ()
        delta = FrozenMap({"memorial_peer_join_count": len(matches)})
        if len(matches) == 1:
            delta = FrozenMap.overlay(delta, matches[0])
        additions[candidate.candidate_ref] = delta
    return agenda.model_copy(update={
        key: FrozenMap({ref: FrozenMap.overlay(additions.get(ref, FrozenMap()), facts)
            for ref, facts in sorted(getattr(agenda, key).items())})
        for key in ("alternative_facts", "descriptive_delta_facts")
    })


def measure_observed_peer_blocker_palettes(before_frame, after_frame, facts, action_ref):
    """Measure contacted palettes only after an exact observed one-member stop."""
    if not measured_peer_route_packet_matches(after_frame, facts):
        return ()
    index = int(facts.get("executed_verified_step_count", 0))
    states = facts.get("peer_route_expected_pixel_boxes", ())
    prior = states[index-1] if index else facts.get("peer_route_initial_pixel_boxes", ())
    if len(prior) != 2:
        return ()
    # The expected source footprints must themselves be observed.
    source_facts = {"peer_palette_values": facts.get("peer_palette_values", ()),
        "peer_route_expected_pixel_boxes": (prior,), "executed_verified_step_count": 0}
    if not measured_peer_route_packet_matches(before_frame, source_facts):
        return ()
    after = states[index]
    unchanged = tuple(i for i in (0,1) if tuple(after[i]) == tuple(prior[i]))
    if len(unchanged) != 1:
        return ()
    deltas = {ref:(dr,dc) for ref,dr,dc in facts.get("peer_interface_delta_rows", ())}
    if action_ref not in deltas:
        return ()
    member = unchanged[0]
    dr,dc = deltas[action_ref]
    if member:
        dr *= int(facts.get("peer_coupling_row", 1))
        dc *= int(facts.get("peer_coupling_column", 1))
    top,left,height,width = prior[member]
    row_pitch, col_pitch = facts.get("peer_current_step_pixels", (height, width))
    top += dr*row_pitch
    left += dc*col_pitch
    if top < 0 or left < 0 or top+height > len(before_frame) or left+width > len(before_frame[0]):
        return ()
    palette = tuple(sorted({before_frame[r][c] for r in range(top,top+height)
        for c in range(left,left+width)}))
    return (palette,)


def measure_peer_route_causal_step(facts, index=0):
    """Expose the current simulated delta and its measured blocker support."""
    rows = facts.get("peer_route_causal_steps", ())
    if not 0 <= index < len(rows):
        return FrozenMap()
    row = dict(rows[index])
    observed = frozenset(tuple(p) for p in facts.get("peer_observed_blocker_palette_rows", ()))
    contact = tuple(tuple(p) for p in row.get("peer_step_contact_palette_rows", ()))
    row["peer_step_blocker_observation_supported"] = bool(contact and all(p in observed for p in contact))
    next_stop = next((entry for entry in rows[index+1:] if entry["peer_step_one_member_stopped"]), None)
    row["peer_step_later_independent_stop_present"] = next_stop is not None
    row["peer_step_next_stop_member_ref"] = next_stop["peer_step_stationary_member_ref"] if next_stop else "member.none"
    row["peer_step_next_stop_row_delta_after"] = next_stop["peer_step_relative_row_delta_after"] if next_stop else 0
    row["peer_step_next_stop_column_delta_after"] = next_stop["peer_step_relative_column_delta_after"] if next_stop else 0
    return FrozenMap(row)
