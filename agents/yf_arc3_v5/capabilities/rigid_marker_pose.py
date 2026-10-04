"""Bounded translations of visible raster support onto repeated marker loci.

Values propose partitions only. No body, control, goal or action is selected;
Source interprets the exact correspondences and their incompleteness.
"""
from collections import defaultdict

from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest


def measure_rigid_marker_poses(*, frame, components, quantum, contract, association_rows=(),
        prior_target_rows=(), prior_body_rows=()):
    empty = FrozenMap({"rigid_marker_pose_rows": (),
        "rigid_marker_pose_enumeration_complete": True,
        "rigid_marker_pose_group_count": 0})
    if not contract or quantum is None or quantum <= 0:
        return empty
    if not frame or not frame[0] or len({len(row) for row in frame}) != 1:
        return FrozenMap({**dict(empty), "rigid_marker_pose_enumeration_complete": False})
    height, width = len(frame), len(frame[0])
    if height * width > contract["maximum_pixels"]:
        return FrozenMap({**dict(empty), "rigid_marker_pose_enumeration_complete": False})
    def bounded_point(point):
        return (len(point) == 2 and all(isinstance(x, int) for x in point)
            and 0 <= point[0] < height and 0 <= point[1] < width)
    if (len(prior_target_rows) > contract["maximum_groups"]
            or len(prior_body_rows) > contract["maximum_groups"]
            or len({row["partition_value"] for row in prior_target_rows}) != len(prior_target_rows)
            or len({row["partition_value"] for row in prior_body_rows}) != len(prior_body_rows)):
        return FrozenMap({**dict(empty), "rigid_marker_pose_enumeration_complete": False})
    for row in prior_target_rows:
        centres, rings = row["marker_centres"], row["marker_ring_rows"]
        if (not contract["minimum_markers"] <= len(centres) <= contract["maximum_markers"]
                or len(rings) != len(centres) or any(not bounded_point(p) for p in centres)
                or set(map(tuple, centres)) != {tuple(ring["centre"]) for ring in rings}
                or any(ring["ring_width"] not in contract["ring_widths"] for ring in rings)):
            return FrozenMap({**dict(empty), "rigid_marker_pose_enumeration_complete": False})
    for row in prior_body_rows:
        offsets = row["relative_support_pixels"]
        if (not bounded_point(row["indicator_position"])
                or not contract["minimum_support_pixels"] <= len(offsets) <= contract["maximum_support_pixels"]
                or any(len(p) != 2 or any(not isinstance(x, int) or abs(x) >= max(height, width)
                    for x in p) for p in offsets)):
            return FrozenMap({**dict(empty), "rigid_marker_pose_enumeration_complete": False})
    groups = defaultdict(list)
    ring_rows = defaultdict(list)
    excluded = set()
    for component in components:
        box = component.bbox
        if component.touches_frame_boundary:
            excluded.update(component.pixels)
        if (box.height != box.width or box.height not in contract["ring_widths"]
                or component.area != box.height * box.width - 1):
            continue
        middle = box.height // 2
        centre = (box.top + middle, box.left + middle)
        expected = tuple((r, c) for r in range(box.height) for c in range(box.width)
            if (r, c) != (middle, middle))
        if tuple(component.relative_pixels) != expected or frame[centre[0]][centre[1]] == component.value:
            continue
        centre_value = frame[centre[0]][centre[1]]
        groups[centre_value].append(centre)
        ring_rows[centre_value].append(FrozenMap({"centre": centre,
            "ring_width": box.height, "ring_value": component.value}))
    contradictions = 0
    for row in prior_target_rows:
        value = row["partition_value"]
        centres = tuple(map(tuple, row["marker_centres"]))
        contradictions += len(set(groups.get(value, ())) - set(centres))
        groups[value] = list(centres)
        ring_rows[value] = tuple(row["marker_ring_rows"])
    if len(groups) > contract["maximum_groups"]:
        return FrozenMap({**dict(empty), "rigid_marker_pose_enumeration_complete": False})
    singleton_positions = tuple(component.pixels[0] for component in components
        if component.area == 1 and not component.touches_frame_boundary)
    glyph_bindings = tuple((point, association) for point in singleton_positions
        for association in association_rows if association.get("active") is True
        and association.get("role_ref") == contract["indicator_role_ref"]
        and tuple(map(tuple, association.get("normalized_valued_pixels") or ()))
            == ((0, 0, frame[point[0]][point[1]]),))
    glyph_positions = tuple(sorted({point for point, association in glyph_bindings}))
    marker_pixels = frozenset((r + dr, c + dc)
        for value in sorted(ring_rows) for ring in ring_rows[value]
        for r, c in (ring["centre"],)
        for dr in range(-(ring["ring_width"]//2), ring["ring_width"]//2 + 1)
        for dc in range(-(ring["ring_width"]//2), ring["ring_width"]//2 + 1))
    templates = {row["partition_value"]: row for row in prior_body_rows}
    rows = []
    for value, centres in sorted(groups.items()):
        if not contract["minimum_markers"] <= len(centres) <= contract["maximum_markers"]:
            continue
        centre_set = frozenset(centres)
        visible_pixels = frozenset((r, c) for r in range(height) for c in range(width)
            if frame[r][c] == value and (r, c) not in centre_set and (r, c) not in excluded)
        pixels = visible_pixels
        template = templates.get(value)
        template_anchor_candidates = ()
        template_anchor = None
        incompatible = 0
        if template is not None:
            offsets = tuple(map(tuple, template["relative_support_pixels"]))
            def placed(anchor):
                return frozenset((anchor[0] + dr, anchor[1] + dc) for dr, dc in offsets)
            def incompatible_pixels(predicted):
                return sum(not (0 <= r < height and 0 <= c < width)
                    or (frame[r][c] != value and (r,c) not in marker_pixels
                        and frame[r][c] not in groups)
                    for r,c in predicted)
            template_anchor_candidates = tuple(point for point in glyph_positions
                if sum(p in visible_pixels for p in placed(point)) >= contract["minimum_support_pixels"]
                and incompatible_pixels(placed(point)) == 0)
            if len(template_anchor_candidates) <= 1:
                template_anchor = (template_anchor_candidates[0] if template_anchor_candidates
                    else tuple(template["indicator_position"]))
                predicted = placed(template_anchor)
                incompatible = incompatible_pixels(predicted)
                if incompatible == 0:
                    pixels = visible_pixels | predicted
        if not contract["minimum_support_pixels"] <= len(pixels) <= contract["maximum_support_pixels"]:
            continue
        top, bottom = min(r for r, c in pixels), max(r for r, c in pixels)
        left, right = min(c for r, c in pixels), max(c for r, c in pixels)
        adjacent = {(r, c): sum((r + dr, c + dc) in pixels
            for dr, dc in contract["neighbour_offsets"])
            for r, c in singleton_positions}
        internal = tuple((r, c) for r, c in singleton_positions
            if top < r < bottom and left < c < right and frame[r][c] != value
            and (adjacent[r, c] >= contract["minimum_internal_neighbours"]
                or (abs(2 * r - top - bottom) <= contract["maximum_midpoint_twice_offset"]
                    and abs(2 * c - left - right) <= contract["maximum_midpoint_twice_offset"])))
        if template is not None:
            internal = template_anchor_candidates
        # Connectivity is a measured partition, not an identity assignment.
        support = pixels | frozenset(point for point in internal
            if adjacent[point] >= contract["minimum_internal_neighbours"])
        remaining = set(support)
        partitions = 0
        while remaining:
            partitions += 1
            pending = [remaining.pop()]
            while pending:
                r, c = pending.pop()
                for dr, dc in contract["neighbour_offsets"]:
                    neighbour = (r + dr, c + dc)
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        pending.append(neighbour)
        if partitions != 1 and template is None:
            continue
        translations = None
        for r, c in centres:
            supported = {(r - br, c - bc) for br, bc in pixels}
            translations = supported if translations is None else translations & supported
        translated_rows = []
        for dr, dc in sorted(translations or ()):
            if dr % quantum or dc % quantum:
                continue
            if not (0 <= top + dr <= bottom + dr < height
                    and 0 <= left + dc <= right + dc < width):
                continue
            translated_rows.append(FrozenMap({"delta_row": dr, "delta_col": dc,
                "primitive_distance": (abs(dr) + abs(dc)) // quantum}))
        if len(translated_rows) > contract["maximum_poses_per_group"]:
            return FrozenMap({**dict(empty), "rigid_marker_pose_enumeration_complete": False})
        token = stable_digest((value, tuple(sorted(pixels)), tuple(sorted(centres))))[:16]
        rows.append(FrozenMap({"measurement_ref": f"measurement.rigid_marker_pose.{token}",
            "partition_value": value, "marker_centres": tuple(sorted(centres)),
            "marker_ring_rows": tuple(ring_rows[value]),
            "visible_support_pixels": tuple(sorted(pixels)),
            "support_bbox": (top, left, bottom, right),
            "support_partition_count": partitions,
            "internal_singleton_positions": internal,
            "internal_singleton_values": tuple(frame[r][c] for r, c in internal),
            "internal_singleton_association_rows": tuple(FrozenMap({
                "position": point,
                "association_ref": association["term_ref"],
                "source_claim_refs": tuple(association.get("source_claim_refs") or ())})
                for point in internal for association in association_rows
                if association.get("active") is True
                and association.get("role_ref") == contract["indicator_role_ref"]
                and tuple(map(tuple, association.get("normalized_valued_pixels") or ()))
                    == ((0, 0, frame[point[0]][point[1]]),)),
            "pose_rows": tuple(translated_rows), "pose_count": len(translated_rows),
            "uses_scoped_support_template": template is not None,
            "template_pixel_incompatibility_count": incompatible,
            "template_anchor_match_count": len(template_anchor_candidates),
            "template_anchor": template_anchor,
            "relative_support_pixels": tuple((r-internal[0][0],c-internal[0][1]) for r,c in sorted(pixels))
                if len(internal)==1 else (),
            "translation_quantum": quantum,
            "uses_visible_support_lower_bound": True,
            "unobserved_support_is_not_certified": True}))
    return FrozenMap({"rigid_marker_pose_rows": tuple(rows),
        "rigid_marker_pose_enumeration_complete": True,
        "rigid_marker_pose_group_count": len(rows),
        "rigid_marker_pose_all_groups_admitted": len(rows) == len(groups),
        "rigid_marker_visible_target_contradiction_count": contradictions})


def measure_rigid_marker_action_delta(measurements, delta):
    """Exact distance deltas; interpretation and admission belong to Source."""
    rows = measurements.get("rigid_marker_pose_rows", ())
    bound_rows = tuple(row for row in rows if len(row["internal_singleton_association_rows"]) == 1)
    row = bound_rows[0] if len(bound_rows) == 1 else None
    distances = tuple(p["primitive_distance"] for p in row["pose_rows"]) if row else ()
    minimum = min(distances) if distances else -1
    quantum = row["translation_quantum"] if row else 0
    after = tuple((abs(p["delta_row"]-delta[0])+abs(p["delta_col"]-delta[1]))//quantum
        for p in row["pose_rows"]) if row and delta is not None and quantum > 0 else ()
    return FrozenMap({
        "rigid_marker_control_binding_count": len(bound_rows),
        "rigid_marker_bound_pose_count": len(distances),
        "rigid_marker_bound_minimum_distance": minimum,
        "rigid_marker_candidate_minimum_distance_delta": min(after)-minimum if after else 0,
        "rigid_marker_candidate_has_measured_delta": delta is not None,
        "rigid_marker_other_nonzero_distance_count": sum(
            bool(r["pose_rows"]) and min(p["primitive_distance"] for p in r["pose_rows"]) > 0
            for r in rows if r not in bound_rows),
        "rigid_marker_template_contradiction_count": sum(r["template_pixel_incompatibility_count"]
            + int(r["template_anchor_match_count"] > 1) for r in rows),
        "rigid_marker_bound_body_rows": (FrozenMap({"partition_value":row["partition_value"],
            "relative_support_pixels":row["relative_support_pixels"],
            "indicator_position":row["internal_singleton_association_rows"][0]["position"]}),)
            if row and row["relative_support_pixels"] else (),
        "rigid_marker_bound_association_refs": tuple(a["association_ref"]
            for r in bound_rows for a in r["internal_singleton_association_rows"]),
        "rigid_marker_bound_source_claim_refs": tuple(dict.fromkeys(c
            for r in bound_rows for a in r["internal_singleton_association_rows"]
            for c in a["source_claim_refs"])),
        "rigid_marker_bound_source_claim_count": len({c for r in bound_rows
            for a in r["internal_singleton_association_rows"] for c in a["source_claim_refs"]}),
    })
