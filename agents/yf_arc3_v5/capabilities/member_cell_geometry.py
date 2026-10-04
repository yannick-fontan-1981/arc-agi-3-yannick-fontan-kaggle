"""Bounded joins of tracked component samples and complete typed-cell supports.

These are geometric measurements, not role arbitration or action selection.
Missing correspondences remain explicit and do not erase any supplied member.
"""

from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest


def measure_member_cell_geometry(*, groups, tracking, geometry, supports,
                                 maximum_pairs=16, maximum_reflection_tests=256):
    if len(groups) != 2 or any(len(group) > 64 for group in groups):
        raise ValueError("member cell geometry group bound exceeded")
    if len(supports) > 128 or not 1 <= maximum_pairs <= 64:
        raise ValueError("member cell geometry support/pair bound exceeded")
    _, row_offset, col_offset, height, width, row_pitch, col_pitch, rows, cols = geometry
    if min(height, width, row_pitch, col_pitch, rows, cols) <= 0:
        raise ValueError("member cell geometry requires positive calibration")
    if height > row_pitch or width > col_pitch or rows * cols > 4096:
        raise ValueError("member cell geometry calibration bound exceeded")
    entities = {item.entity_id: item for item in tracking.entities
                if item.current_frame_ref == tracking.frame_ref}
    if len(entities) > 1024:
        raise ValueError("member cell geometry entity bound exceeded")
    support_cells = {}
    for support in supports:
        cells = frozenset((support["top"] + r, support["left"] + c)
                          for r, c in support["normalized_cells"])
        if len(cells) > 4096 or any(not (0 <= r < rows and 0 <= c < cols) for r, c in cells):
            raise ValueError("member cell geometry support lies outside lattice")
        if support["support_ref"] in support_cells:
            raise ValueError("member cell geometry requires unique support references")
        support_cells[support["support_ref"]] = cells
    # One bounded cell projection per current component, shared by all support
    # joins. Decorations remain members; a component crossing the support is
    # never silently clipped into a different object.
    entity_cells = {}
    for ref, entity in sorted(entities.items()):
        cells, outside = set(), False
        for r, c in entity.component.pixels:
            lr, ir = divmod(r - row_offset, row_pitch)
            lc, ic = divmod(c - col_offset, col_pitch)
            if not (0 <= lr < rows and 0 <= lc < cols and ir < height and ic < width):
                outside = True
            cells.add((lr, lc))
        if cells and not outside:
            entity_cells[ref] = frozenset(cells)
    # A background-hole support can merge when two already-bound bodies touch.
    # Recover only an exact disjoint partition witnessed by the supplied member
    # groups. Every cell perimeter must belong to its group: an isolated inner
    # indicator cannot create a body. No role is inferred by this partition.
    partitions = {}
    for group in groups:
        for refs in group:
            if not refs or any(ref not in entity_cells for ref in refs):
                continue
            cells = frozenset(cell for ref in refs for cell in entity_cells[ref])
            pixels = frozenset(point for ref in refs for point in entities[ref].component.pixels)
            if all((row_offset+r*row_pitch+dr, col_offset+c*col_pitch+dc) in pixels
                   for r,c in cells for dr in range(height) for dc in range(width)
                   if dr in (0,height-1) or dc in (0,width-1)):
                partitions[cells] = tuple(sorted(refs))
    additions = {}
    for parent in tuple(support_cells.values()):
        parts = tuple(cells for cells in partitions if cells < parent)
        if (len(parts) >= 2 and sum(map(len, parts)) == len(parent)
                and frozenset(cell for part in parts for cell in part) == parent):
            for cells in parts:
                ref = "measurement.member_cell_partition." + stable_digest(tuple(sorted(cells)))[:24]
                additions[ref] = cells
    if len(support_cells) + len(additions) > 128:
        raise ValueError("member cell partition bound exceeded")
    support_cells.update(additions)
    support_members = []
    for ref, cells in sorted(support_cells.items()):
        members = tuple(member for member, occupied in sorted(entity_cells.items())
            if occupied <= cells)
        if len(members) > 128:
            raise ValueError("cell support member bound exceeded; no partial support")
        union = frozenset(cell for member in members for cell in entity_cells[member])
        top = min((r for r, _ in cells), default=0)
        left = min((c for _, c in cells), default=0)
        support_members.append(FrozenMap({"support_ref": ref, "member_refs": members,
            "normalized_cells": tuple(sorted((r-top, c-left)
                for r, c in cells)) if cells else (),
            "cell_bbox": (min(r for r,c in cells), min(c for r,c in cells),
                max(r for r,c in cells), max(c for r,c in cells)) if cells else (),
            "complete_cell_coverage": bool(cells) and union == cells}))
    results = []
    for group in groups:
        entries = []
        for members in group:
            if len(members) > 128 or len(set(members)) != len(members):
                raise ValueError("member cell geometry member bound exceeded")
            samples = tuple(entities[ref].component.pixels for ref in members if ref in entities)
            if sum(len(item) for item in samples) > 8192:
                raise ValueError("member cell geometry sample bound exceeded")
            cells, outside = set(), False
            for pixels in samples:
                for r, c in pixels:
                    lr, ir = divmod(r - row_offset, row_pitch)
                    lc, ic = divmod(c - col_offset, col_pitch)
                    if not (0 <= lr < rows and 0 <= lc < cols and ir < height and ic < width):
                        outside = True
                    else:
                        cells.add((lr, lc))
            present = bool(members) and len(samples) == len(members)
            refs = tuple(ref for ref, occupied in sorted(support_cells.items())
                         if present and not outside and cells == occupied)
            entries.append(FrozenMap({"member_refs": tuple(members),
                "all_members_present": present, "samples_outside_grid": outside,
                "sampled_cell_count": len(cells), "complete_support_refs": refs}))
        results.append(tuple(entries))
    first_refs = tuple(sorted({ref for row in results[0] for ref in row["complete_support_refs"]}))
    second_refs = tuple(sorted({ref for row in results[1] for ref in row["complete_support_refs"]}))
    pair_count = len(first_refs) * len(second_refs)
    pairs = []
    if pair_count <= maximum_pairs:
        for first in first_refs:
            for second in second_refs:
                a, b = support_cells[first], support_cells[second]
                ar, ac = min(r for r, _ in a), min(c for _, c in a)
                br, bc = min(r for r, _ in b), min(c for _, c in b)
                shape_a = frozenset((r-ar, c-ac) for r, c in a)
                shape_b = frozenset((r-br, c-bc) for r, c in b)
                pairs.append(FrozenMap({"first_support_ref": first, "second_support_ref": second,
                    "distinct_supports": first != second, "same_cell_shape": shape_a == shape_b,
                    "first_cell_count": len(a), "second_cell_count": len(b),
                    "offset_row_cells": br-ar, "offset_col_cells": bc-ac,
                    "current_intersection_cell_count": len(a & b),
                    "second_uncovered_cell_count": len(b-a)}))
    return FrozenMap({"member_group_rows": tuple(results), "pair_rows": tuple(pairs),
        "support_member_rows": tuple(support_members),
        "pair_count": pair_count, "pair_enumeration_complete": pair_count <= maximum_pairs,
        "reflection_measurements": measure_cell_reflections(
            first_refs=first_refs, support_cells=support_cells, tracking=tracking,
            geometry=geometry, maximum_tests=maximum_reflection_tests)})


def measure_cell_reflections(*, first_refs, support_cells, tracking, geometry, maximum_tests=256):
    """Exact cell-set reflection and axis-coincident spanning component boxes.

No whole-scene symmetry, palette equality, axis role or control is asserted.
The doubled axis coordinate permits both cell-centred and inter-cell axes.
"""
    if not 1 <= maximum_tests <= 256:
        raise ValueError("cell reflection comparison bound exceeded")
    if len(first_refs) > 64 or len(support_cells) > 128:
        raise ValueError("cell reflection input bound exceeded")
    tests = 2 * sum(other != first for first in first_refs for other in support_cells)
    if tests > maximum_tests:
        return FrozenMap({"rows": (), "comparison_count": tests, "complete": False})
    _, ro, co, height, width, rp, cp, _, _ = geometry
    result = []
    for first in first_refs:
        a = support_cells[first]
        if not a:
            continue
        for second, b in sorted(support_cells.items()):
            if first == second or len(a) != len(b) or not b:
                continue
            for dimension in (0, 1):
                coordinate_twice = min(cell[dimension] for cell in a) + max(cell[dimension] for cell in b)
                reflected = frozenset((coordinate_twice-r, c) if dimension == 0
                                      else (r, coordinate_twice-c) for r,c in a)
                if reflected != b:
                    continue
                perpendicular = 1-dimension
                low = min(cell[perpendicular] for cell in a | b)
                high = max(cell[perpendicular] for cell in a | b)
                carrier_refs = []
                for entity in tracking.entities:
                    if entity.current_frame_ref != tracking.frame_ref:
                        continue
                    box = entity.component.bbox
                    if dimension == 0:
                        numerator, pitch = box.top + box.bottom - 2*ro - (height-1), rp
                        lower, upper, offset, perp_pitch = box.left, box.right, co, cp
                    else:
                        numerator, pitch = box.left + box.right - 2*co - (width-1), cp
                        lower, upper, offset, perp_pitch = box.top, box.bottom, ro, rp
                    if numerator == coordinate_twice * pitch and (lower-offset)//perp_pitch <= low and (upper-offset)//perp_pitch >= high:
                        carrier_refs.append(entity.entity_id)
                result.append(FrozenMap({"first_support_ref": first, "second_support_ref": second,
                    "dimension_index": dimension, "axis_coordinate_twice": coordinate_twice,
                    "first_normal_min": min(cell[dimension] for cell in a),
                    "first_normal_max": max(cell[dimension] for cell in a),
                    "second_normal_max": max(cell[dimension] for cell in b),
                    "tangent_min": low,
                    "spanning_coincident_entity_refs": tuple(sorted(carrier_refs)),
                    "support_cell_count": len(a)}))
    return FrozenMap({"rows": tuple(result), "comparison_count": tests, "complete": True})
