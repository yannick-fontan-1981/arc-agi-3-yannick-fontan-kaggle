"""Bounded translations under an explicit box-step / uniform-surface model.

This enumerates geometric witnesses. It neither assigns a target role nor
authorizes the model or its repeated execution; those decisions belong to DRM.
"""

from collections import deque
from collections.abc import Mapping

from agents.yf_arc3_v5.capabilities.contracts import InteractionProbeInput
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest


def measure_palette_support_routes(value: InteractionProbeInput) -> tuple[FrozenMap, ...]:
    contract = value.route_generation_policy.get("palette_support_route_measurement", {})
    if not isinstance(contract, Mapping) or not contract:
        return ()
    boxes = value.canonical_provisional_control_geometry_bboxes
    if len(boxes) != 1 or len(value.canonical_provisional_control_term_facts) != 1:
        return ()
    body = boxes[0]
    if body.height != body.width or body.height < 2:
        return ()
    if body.height * body.width > int(contract["max_support_pixels"]):
        return ()
    frame = value.frame
    if body.top < 0 or body.left < 0 or body.bottom >= frame.height or body.right >= frame.width:
        return ()
    occupied = frozenset((r, c) for r in range(body.top, body.bottom + 1)
                         for c in range(body.left, body.right + 1))
    normalized = tuple((r - body.top, c - body.left, frame.rows[r][c]) for r, c in sorted(occupied))
    # The passed support is already bound by DRM. A hollow box is not its union.
    if stable_digest(normalized) != value.canonical_provisional_control_term_facts[0].get("support_digest"):
        return ()
    palette = frozenset(frame.rows[r][c] for r, c in occupied)
    if not 2 <= len(palette) <= int(contract["max_palette_values"]):
        return ()
    if frame.height * frame.width > int(contract["max_frame_pixels"]):
        return ()
    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    # A currently bound body may touch its counterpart. Its known support must
    # not merge the two into one colour-connected component at the last step.
    remaining = {(r, c) for r, row in enumerate(frame.rows) for c, pixel in enumerate(row)
                 if pixel in palette and (r, c) not in occupied}
    regions = []
    while remaining:
        pending = [min(remaining)]
        region = set()
        while pending:
            r, c = pending.pop()
            if (r, c) not in remaining:
                continue
            remaining.remove((r, c))
            region.add((r, c))
            pending.extend((r + dr, c + dc) for dr, dc in offsets if (r + dr, c + dc) in remaining)
        if region.intersection(occupied) or len(region) <= len(occupied):
            continue
        if frozenset(frame.rows[r][c] for r, c in region) != palette:
            continue
        top, left = min(r for r, c in region), min(c for r, c in region)
        bottom, right = max(r for r, c in region), max(c for r, c in region)
        if (bottom - top + 1) * (right - left + 1) != len(region):
            continue
        if bottom - top + 1 < body.height or right - left + 1 < body.width:
            continue
        # An independently changed region is not a stationary counterpart.
        if any(not (box.bottom < top or box.top > bottom or box.right < left or box.left > right)
               for box in value.current_transition_change_bboxes):
            continue
        regions.append((frozenset(region), (top, left, bottom, right)))
        if len(regions) > int(contract["max_witnesses"]):
            return ()
    boundary_values = {frame.rows[r + dr][c + dc] for r, c in occupied for dr, dc in offsets
                       if 0 <= r + dr < frame.height and 0 <= c + dc < frame.width
                       and (r + dr, c + dc) not in occupied} - palette
    observed_underlay = set(value.current_input_aligned_underlay_values)
    # Current raster under a previously occupied, complete declared support.
    # This is an occupancy constraint on surface models, not an atomic-piece
    # translation: a body can rotate while moving and lose piecewise lineage.
    visits = tuple(value.visited_input_aligned_entity_bboxes)
    vacated_support_values = set()
    if contract.get("vacated_current_control_support_constrains_surface") is True and len(visits) >= 2 and visits[-1] == body:
        prior = visits[-2]
        if prior.height == body.height and prior.width == body.width:
            vacated = {(r, c) for r in range(prior.top, prior.bottom + 1)
                       for c in range(prior.left, prior.right + 1)} - occupied
            if vacated and all(0 <= r < frame.height and 0 <= c < frame.width for r, c in vacated):
                values = {frame.rows[r][c] for r, c in vacated}
                if len(values) == 1 and not values.intersection(palette):
                    vacated_support_values = values
                    observed_underlay.update(values)
    if observed_underlay:
        boundary_values.intersection_update(observed_underlay)
    deltas = tuple((str(action), int(dr) * body.height, int(dc) * body.width)
                   for action, dr, dc in value.interface_action_translation_deltas
                   if action in value.available_action_refs and abs(int(dr)) + abs(int(dc)) == 1)
    changed_inside_support = bool(value.current_transition_change_bboxes) and all(
        body.top <= box.top <= box.bottom <= body.bottom
        and body.left <= box.left <= box.right <= body.right
        for box in value.current_transition_change_bboxes)
    blocked_delta = next(((dr, dc) for action, dr, dc in deltas
                          if action == value.last_action_ref and changed_inside_support), None)
    models = []
    for region, region_box in regions:
        for surface in sorted(boundary_values):
            def fits(position, surface=surface, region=region):
                r, c = position
                return all(0 <= rr < frame.height and 0 <= cc < frame.width
                           and (frame.rows[rr][cc] == surface or (rr, cc) in occupied or (rr, cc) in region)
                           for rr in range(r, r + body.height) for cc in range(c, c + body.width))
            if blocked_delta is not None and fits((body.top + blocked_delta[0], body.left + blocked_delta[1])):
                continue
            models.append((region_box, surface, fits))
    if len(models) > int(contract["max_witnesses"]):
        return ()
    results = []
    memory_join = contract.get("relation_memory_join", {})
    matching_refs = tuple(sorted({str(row["claim_ref"])
        for row in getattr(value, "canonical_entry_relation_facts", ())
        if memory_join and row.get("claim_ref")
        and tuple(row.get(memory_join["palette_attribute"], ())) == tuple(sorted(palette))
        and row.get("epistemic_status") in memory_join["proof_statuses"]
        and row.get("disposition") == memory_join["proof_disposition"]}))
    start = (body.top, body.left)
    for region_box, surface, fits in models:
        queue = deque([start])
        parent = {start: None}
        endpoint = None
        while queue:
            current = queue.popleft()
            r, c = current
            top, left, bottom, right = region_box
            if top <= r and left <= c and r + body.height - 1 <= bottom and c + body.width - 1 <= right:
                endpoint = current
                break
            for action, dr, dc in deltas:
                next_position = (r + dr, c + dc)
                if next_position not in parent and fits(next_position):
                    if len(parent) >= int(contract["max_expanded_states"]):
                        return ()
                    parent[next_position] = (current, action)
                    queue.append(next_position)
        if endpoint is None or endpoint == start:
            continue
        path = []
        current = endpoint
        while parent[current] is not None:
            previous, action = parent[current]
            path.append(action)
            current = previous
        path.reverse()
        if len(path) > int(contract["max_path_actions"]):
            continue
        results.append(FrozenMap({
            "region_token": stable_digest(region_box), "region_bbox": region_box,
            "surface_value": surface, "step_extent": body.width,
            "action_refs": tuple(path), "first_action_ref": path[0],
            "expanded_state_count": len(parent),
            "stationary_directional_change_constrains_surface": blocked_delta is not None,
            "vacated_support_uniform_value_count": len(vacated_support_values),
            "matching_relation_claim_refs": matching_refs,
            "joined_premise_refs": tuple(dict.fromkeys((
                *value.canonical_provisional_control_term_facts[0].get("role_premise_claim_refs", ()), *matching_refs))),
        }))
    # Count executable witnesses separately from the surface models supporting
    # them. Exact route/endpoint/extent equality is a measurement, not a choice
    # between models. Keep every surface alternative in the returned witness.
    grouped = {}
    for result in results:
        key = (result["region_bbox"], result["step_extent"], result["action_refs"])
        grouped.setdefault(key, []).append(result)
    witnesses = []
    for key in sorted(grouped):
        members = grouped[key]
        surfaces = tuple(member["surface_value"] for member in members)
        facts = dict(members[0])
        facts.update(
            surface_value=surfaces[0] if len(surfaces) == 1 else None,
            surface_values=surfaces,
            model_count=len(members),
            expanded_state_count=sum(member["expanded_state_count"] for member in members),
        )
        witnesses.append(FrozenMap(facts))
    return tuple(witnesses)
