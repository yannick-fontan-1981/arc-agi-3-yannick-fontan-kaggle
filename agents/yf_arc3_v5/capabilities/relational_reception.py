"""Bounded raster witnesses under a declared, uncommitted support supposition.

No palette assigns a role. A rectangular perimeter and a thin connected band
are measurements, not evidence that the body receives material or obeys arrows.
Only the next primitive is exposed; DRM/SRC owns its meaning and release.
"""
from collections import deque

from agents.yf_arc3_v5.logos.types import FrozenMap


def measure_reception_band_witnesses(*, frame, components, requirement, lattice, contract):
    """One shortest witness per body/band/absent-landing description.

    The declared bounds apply to the whole inventory, including disconnected
    alternatives. Overflow returns no executable partial inventory.
    """
    absent = FrozenMap({"complete": False, "witnesses": ()})
    geometry = lattice.get("current_lattice_geometry")
    if not geometry or len(geometry) != 9 or not contract:
        return absent
    _, ro, co, height, width, rp, cp, nr, nc = geometry
    if min(height, width, rp, cp, nr, nc) <= 0 or nr * nc > contract["max_cells"]:
        return absent
    if height != rp or width != cp:
        return absent
    rows = frame.rows
    cells = tuple((ro + i * rp + (height - 1) // 2,
                   co + j * cp + (width - 1) // 2)
                  for i in range(nr) for j in range(nc))
    if any(not (0 <= r < len(rows) and 0 <= c < len(rows[r])) for r, c in cells):
        return absent
    cell_set = frozenset(cells)
    preserved = frozenset(tuple(p) for p in requirement.get("material_positions_to_preserve", ()))
    overlap_required = bool(requirement.get("body_center_material_overlap_required", False))
    endpoint_field = requirement.get("body_endpoint_field", "landing_position")
    if endpoint_field not in ("source_position", "landing_position"):
        return absent
    targets = requirement.get("future_triple_alternatives", ())
    if not 0 < len(targets) <= contract["max_descriptions"]:
        return absent
    if len(components) > contract["max_components"]:
        return absent
    bodies = []
    for component in components:
        box = component.bbox
        center = ((box.top + box.bottom) // 2, (box.left + box.right) // 2)
        if center not in cell_set or (center in preserved) != overlap_required:
            continue
        if box.bottom - box.top + 1 != height or box.right - box.left + 1 != width:
            continue
        perimeter = frozenset((r, c) for r in range(box.top, box.bottom + 1)
            for c in range(box.left, box.right + 1)
            if r in (box.top, box.bottom) or c in (box.left, box.right))
        if frozenset(component.pixels) == perimeter:
            bodies.append((component, center))
    if len(bodies) > contract["max_descriptions"]:
        return absent
    directions = tuple(tuple(delta) for delta in contract["traversal_deltas"])
    witnesses = []
    for body, start in bodies:
        # The body can occupy the required endpoint while masking its band.
        # Exact containment is a current measurement, not proof of reception.
        for target in targets:
            if requirement.get("middle_must_remain_outside_current_body", False) and tuple(target.get("middle_position", ())) == start:
                continue
            if tuple(target.get(endpoint_field, ())) != start:
                continue
            if len(witnesses) == contract["max_descriptions"]:
                return absent
            witnesses.append(FrozenMap({
                "body_ref": body.component_id, "body_value": body.value,
                "body_bbox": (body.bbox.top, body.bbox.left, body.bbox.bottom, body.bbox.right),
                "body_position": start, "band_value": None,
                "landing_position": target["landing_position"], "source_position": target["source_position"],
                "middle_position": target["middle_position"],
                "fixed_prefix_triples": target.get("fixed_prefix_triples", ()),
                "path_length": 0, "first_step_delta": None,
                "material_positions_to_preserve": tuple(sorted(preserved)),
            }))
        # A band must connect the observed perimeter to the adjacent cell
        # center, with a bounded section on a declared axis along that segment.
        # At an elbow, length along the incoming perpendicular is not its width.
        # A broad background or a merely nearby detached stripe fails this test.
        domains = set()
        for dr, dc in directions:
            neighbor = (start[0] + dr * rp, start[1] + dc * cp)
            if neighbor not in cell_set or neighbor in preserved:
                continue
            edge = (body.bbox.bottom + 1, start[1]) if dr > 0 else (
                (body.bbox.top - 1, start[1]) if dr < 0 else (
                (start[0], body.bbox.right + 1) if dc > 0 else (start[0], body.bbox.left - 1)))
            if not (0 <= edge[0] < len(rows) and 0 <= edge[1] < len(rows[edge[0]])):
                continue
            shade = rows[neighbor[0]][neighbor[1]]
            ray = tuple((edge[0] + k * dr, edge[1] + k * dc)
                        for k in range(abs(neighbor[0] - edge[0]) + abs(neighbor[1] - edge[1]) + 1))
            if any(rows[r][c] != shade for r, c in ray):
                continue
            section_present = False
            for point in ray:
                for ar, ac in contract["band_section_axes"]:
                    limit = (height if ar else width) - contract["band_section_cell_margin"]
                    extent, closed_ends = 1, 0
                    for sign in (-1, 1):
                        for k in range(1, limit + 2):
                            r, c = point[0] + sign * k * ar, point[1] + sign * k * ac
                            if not (0 <= r < len(rows) and 0 <= c < len(rows[r])):
                                break
                            if rows[r][c] != shade:
                                closed_ends += 1
                                break
                            extent += 1
                    if closed_ends == 2 and extent <= limit:
                        section_present = True
                        break
                if section_present:
                    break
            if section_present:
                domains.add(shade)
        for shade in sorted(domains):
            nodes = frozenset(p for p in cells if p == start or (
                p not in preserved and rows[p[0]][p[1]] == shade))
            parents = {start: None}
            queue = deque((start,))
            while queue:
                point = queue.popleft()
                for dr, dc in directions:
                    following = (point[0] + dr * rp, point[1] + dc * cp)
                    if following not in nodes or following in parents:
                        continue
                    if point == start:
                        first = (body.bbox.bottom + 1, point[1]) if dr > 0 else (
                            (body.bbox.top - 1, point[1]) if dr < 0 else (
                            (point[0], body.bbox.right + 1) if dc > 0 else (point[0], body.bbox.left - 1)))
                    else:
                        first = point
                    distance = abs(following[0] - first[0]) + abs(following[1] - first[1])
                    if all(rows[first[0] + k * dr][first[1] + k * dc] == shade
                           for k in range(distance + 1)):
                        parents[following] = point
                        queue.append(following)
            for target in targets:
                if requirement.get("middle_must_remain_outside_current_body", False) and tuple(target.get("middle_position", ())) == start:
                    continue
                landing = tuple(target.get(endpoint_field, ()))
                if landing == start or landing not in parents:
                    continue
                path = [landing]
                while parents[path[-1]] is not None:
                    path.append(parents[path[-1]])
                path.reverse()
                if len(path) - 1 > contract["max_path_depth"]:
                    return absent
                if len(witnesses) == contract["max_descriptions"]:
                    return absent
                witnesses.append(FrozenMap({
                    "body_ref": body.component_id, "body_value": body.value,
                    "body_bbox": (body.bbox.top, body.bbox.left, body.bbox.bottom, body.bbox.right),
                    "body_position": start, "band_value": shade,
                    "landing_position": target["landing_position"], "source_position": target["source_position"],
                    "middle_position": target["middle_position"],
                    "fixed_prefix_triples": target.get("fixed_prefix_triples", ()),
                    "path_length": len(path) - 1,
                    "first_step_delta": ((path[1][0] - start[0], path[1][1] - start[1]) if len(path) > 1 else None),
                    "material_positions_to_preserve": tuple(sorted(preserved)),
                }))
    return FrozenMap({"complete": True, "witnesses": tuple(witnesses), "body_count": len(bodies)})


def annotate_relational_reception(value, agenda):
    agenda = agenda.model_copy(update={"context_facts": FrozenMap.overlay(
        FrozenMap({"relational_reception_single_band_step_present": False,
            "relational_reception_action_present": False,
            "relational_reception_zero_step_witness_present": False,
            "relational_reception_body_material_overlap": False}), agenda.context_facts)})
    requirements = value.canonical_relational_reception_facts + value.canonical_relational_transport_facts
    lattices = value.canonical_relational_lattice_facts
    if len(requirements) != 1 or len(lattices) != 1:
        return agenda
    requirement, lattice = requirements[0], lattices[0]
    if requirement.get("current_frame_ref") != lattice.get("current_frame_ref"):
        return agenda
    from agents.yf_arc3_v5.capabilities import ComponentExtractionInput, detect_components
    frame = value.current_observed_frame or value.frame
    components = value.components.components if frame == value.frame else detect_components(
        ComponentExtractionInput(frame=frame, connectivity=4)).components
    measured = measure_reception_band_witnesses(frame=frame, components=components,
        requirement=requirement, lattice=lattice, contract=value.relational_reception_measurement_contract)
    witnesses = measured["witnesses"]
    # More than one structural explanation remains an explicit ontology gap;
    # do not manufacture a body/path selection through Python ordering.
    unique = witnesses[0] if measured["complete"] and len(witnesses) == 1 else None
    deltas = {}
    for candidate in agenda.candidates:
        matches = False
        prefix_source = False
        prefix_landing = False
        receiving_source = False
        receiving_landing = False
        if unique and unique["first_step_delta"] is not None and candidate.point is None:
            step = unique["first_step_delta"]
            matches = any(ref == candidate.action_ref and dr * step[0] + dc * step[1] > 0
                and dr * step[1] == dc * step[0]
                for ref, dr, dc in value.interface_action_translation_deltas)
        if unique and unique["path_length"] == 0 and not requirement.get("body_center_material_overlap_required", False):
            fixed_prefix = unique["fixed_prefix_triples"]
            source, middle, landing = fixed_prefix[0] if fixed_prefix else (
                unique["source_position"], unique["middle_position"], unique["landing_position"])
            preserved = frozenset(unique["material_positions_to_preserve"])
            from .interaction import POINT_ACTION_REF
            geometry = lattice["current_lattice_geometry"]
            landing_changed_after_source = bool(value.last_action_ref == POINT_ACTION_REF
                and value.last_point_position == tuple(source)
                and any(box.top <= landing[0] <= box.bottom
                    and box.left <= landing[1] <= box.right
                    and box.height <= geometry[3] and box.width <= geometry[4]
                    for box in value.current_transition_change_bboxes))
            current_triple = tuple(source) in preserved and tuple(middle) in preserved and tuple(landing) not in preserved
            source_point = bool(current_triple and not landing_changed_after_source
                and candidate.point == tuple(source))
            landing_point = bool(current_triple and landing_changed_after_source
                and candidate.point == tuple(landing))
            prefix_source = bool(fixed_prefix and source_point)
            prefix_landing = bool(fixed_prefix and landing_point)
            receiving_source = bool(not fixed_prefix and source_point)
            receiving_landing = bool(not fixed_prefix and landing_point)
        delta = {"relational_reception_band_step_candidate": matches,
            "relational_reception_body_material_overlap": bool(requirement.get("body_center_material_overlap_required", False)),
            "relational_reception_prefix_source_candidate": prefix_source,
            "relational_reception_prefix_landing_candidate": prefix_landing,
            "relational_reception_prefix_candidate": prefix_source or prefix_landing,
            "relational_reception_body_source_candidate": receiving_source,
            "relational_reception_body_landing_candidate": receiving_landing,
            "relational_reception_body_point_candidate": receiving_source or receiving_landing,
            "relational_reception_point_candidate": prefix_source or prefix_landing or receiving_source or receiving_landing}
        if matches or delta["relational_reception_point_candidate"]:
            delta.update({"relational_reception_measurement": unique,
                "relational_reception_frame_ref": requirement["current_frame_ref"],
                "relational_reception_requirement_ref": requirement["term_ref"],
                "relational_reception_terminal_ref": requirement["terminal_condition_ref"],
                "relational_reception_premise_refs": requirement["role_premise_claim_refs"],
                "relational_reception_complete": measured["complete"],
                "relational_reception_body_count": measured["body_count"],
                "relational_reception_attachment_status": requirement.get("attachment_status", "unknown")})
        deltas[candidate.candidate_ref] = FrozenMap(delta)
    return agenda.model_copy(update={
        "alternative_facts": FrozenMap({ref: FrozenMap.overlay(deltas.get(ref, FrozenMap()), facts)
            for ref, facts in sorted(agenda.alternative_facts.items())}),
        "descriptive_delta_facts": FrozenMap({ref: FrozenMap.overlay(deltas.get(ref, FrozenMap()), facts)
            for ref, facts in sorted(agenda.descriptive_delta_facts.items())}),
        "context_facts": FrozenMap.overlay(FrozenMap({
            "relational_reception_single_band_step_present": bool(unique
                and unique["first_step_delta"] is not None and measured.get("body_count") == 1),
            "relational_reception_action_present": bool(measured.get("body_count") == 1
                and any(row["relational_reception_band_step_candidate"]
                    or row["relational_reception_point_candidate"]
                    for _, row in sorted(deltas.items()))),
            "relational_reception_band_inventory_complete": measured["complete"],
            "relational_reception_zero_step_witness_present": bool(unique
                and unique["path_length"] == 0 and measured.get("body_count") == 1),
            "relational_reception_band_witness_count": len(witnesses)}), agenda.context_facts)})
