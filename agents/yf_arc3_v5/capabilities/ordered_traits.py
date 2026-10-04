"""Bounded geometry of repeated line columns; no instruction or role selection."""

from __future__ import annotations

from collections import defaultdict

from .contracts import ComponentDescription
from ..logos.types import stable_digest

MAX_TRAITS = 128
MAX_ARRAYS = 4
MAX_COLUMNS = 16
MAX_ROWS = 16
MAX_CONTROLS = 16


def measure_ordered_trait_arrays(components: tuple[ComponentDescription, ...]) -> dict:
    """Describe complete regular arrays, retaining whole column value vectors.

    Every trait is an observed connected line.  Separate gaps split arrays;
    colours never identify a role.  Bounds reject, rather than truncate, a family.
    """
    lines = tuple(c for c in components if 2 <= c.area <= 8
                  and (c.bbox.height == 1 or c.bbox.width == 1)
                  and c.bbox.height * c.bbox.width == c.area
                  and not c.touches_frame_boundary)
    if not lines or len(lines) > MAX_TRAITS:
        return {}
    columns = defaultdict(list)
    for c in lines:
        columns[(c.bbox.left + c.bbox.right, c.area)].append(c)
    families = defaultdict(list)
    for (col2, area), members in sorted(columns.items()):
        members.sort(key=lambda c: c.bbox.top + c.bbox.bottom)
        runs = [[]]
        for c in members:
            row2 = c.bbox.top + c.bbox.bottom
            if runs[-1] and row2 - sum((runs[-1][-1].bbox.top, runs[-1][-1].bbox.bottom)) > 2 * area:
                runs.append([])
            runs[-1].append(c)
        for run in runs:
            if not 2 <= len(run) <= MAX_ROWS:
                continue
            rows2 = tuple(c.bbox.top + c.bbox.bottom for c in run)
            steps = tuple(b - a for a, b in zip(rows2, rows2[1:]))
            shapes = tuple((c.bbox.height, c.bbox.width) for c in run)
            if len(set(steps)) != 1 or steps[0] <= 0 or len(set(shapes)) != 2:
                continue
            if any(a == b for a, b in zip(shapes, shapes[1:])):
                continue
            families[(rows2, shapes, area)].append((col2, tuple(run)))
    arrays = []
    for (rows2, shapes, area), columns in sorted(families.items()):
        columns.sort(key=lambda row: row[0])
        runs = [[]]
        for column in columns:
            if runs[-1] and column[0] - runs[-1][-1][0] > 4 * area:
                runs.append([])
            runs[-1].append(column)
        for run in runs:
            if not 3 <= len(run) <= MAX_COLUMNS:
                continue
            cols2 = tuple(c[0] for c in run)
            steps = tuple(b - a for a, b in zip(cols2, cols2[1:]))
            if len(set(steps)) != 1 or steps[0] <= 0:
                continue
            flat = tuple(c for _, cells in run for c in cells)
            bbox = (min(c.bbox.top for c in flat), min(c.bbox.left for c in flat),
                    max(c.bbox.bottom for c in flat), max(c.bbox.right for c in flat))
            extent = max(bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1)
            controls = defaultdict(list)
            for c in components:
                if (c.area < area * area or c.touches_frame_boundary
                        or not 2 <= c.bbox.height <= 3 * area
                        or not 2 <= c.bbox.width <= 3 * area
                        or not bbox[2] < c.bbox.top <= bbox[2] + extent
                        or c.bbox.left < bbox[1] - area or c.bbox.right > bbox[3] + area):
                    continue
                center = ((c.bbox.top + c.bbox.bottom) // 2,
                          (c.bbox.left + c.bbox.right) // 2)
                controls[center].append(c.component_id)
            if len(controls) > MAX_CONTROLS:
                return {}
            peers = defaultdict(list)
            for c in components:
                if (4 <= c.area <= 64 and c.bbox.width > 1 and c.bbox.height > 1
                        and not c.touches_frame_boundary and c.bbox.bottom < bbox[0]
                        and 2 * bbox[1] <= c.bbox.left + c.bbox.right <= 2 * bbox[3]):
                    peers[(c.relative_pixels, c.area)].append(c)
            pairs = []
            for (pixels, size), members in sorted(peers.items()):
                if len(members) != 2:
                    continue
                first, second = sorted(members, key=lambda c: (c.bbox.top, c.bbox.left))
                pairs.append({"relative_pixels": pixels, "area": size,
                    "bboxes": tuple((c.bbox.top, c.bbox.left, c.bbox.bottom, c.bbox.right)
                                    for c in (first, second))})
            if len(pairs) > 3:
                return {}
            arrays.append({
                "bbox": bbox, "row_centers_twice": rows2,
                "column_centers_twice": cols2, "trait_shapes": shapes,
                "values_by_column": tuple(tuple(c.value for c in cells) for _, cells in run),
                "component_refs_by_column": tuple(tuple(c.component_id for c in cells) for _, cells in run),
                "same_shape_pairs_above": tuple(pairs),
                "controls": tuple({"point": p, "component_refs": tuple(sorted(refs))}
                                  for p, refs in sorted(controls.items())),
            })
    if not arrays or len(arrays) > MAX_ARRAYS:
        return {}
    arrays.sort(key=lambda a: a["bbox"])
    structure = tuple({k: a[k] for k in ("bbox", "row_centers_twice", "column_centers_twice", "trait_shapes", "same_shape_pairs_above")}
                      for a in arrays)
    return {"geometry_ref": "measurement.ordered_traits." + stable_digest(structure),
            "layout_ref": "measurement.ordered_trait_layout." + stable_digest(tuple(
                {k: a[k] for k in ("bbox", "row_centers_twice", "column_centers_twice", "trait_shapes")}
                for a in arrays)),
            "arrays": tuple(arrays), "structure": structure}


def measure_focused_word_translations(frames) -> dict:
    """Fit exact whole-vector/translation witnesses in an ordered frame packet.

    A transient enclosing component identifies one column geometrically.
    Only adjacent frames with that enclosure contribute a witness; a reset
    before the first focus is never divided by the instruction count.
    """
    from .components import detect_components
    from .contracts import ComponentExtractionInput

    if not 2 <= len(frames) <= 64:
        return {}
    snapshots = tuple(detect_components(ComponentExtractionInput(frame=f)).components for f in frames)
    layouts = tuple(measure_ordered_trait_arrays(cs) for cs in snapshots)
    if not all(layouts) or len({m["layout_ref"] for m in layouts}) != 1:
        return {}
    witnesses = []
    for previous, current, layout in zip(snapshots, snapshots[1:], layouts[1:]):
        old = defaultdict(list)
        new = defaultdict(list)
        for items, table in ((previous, old), (current, new)):
            for c in items:
                if (4 <= c.area <= 64 and c.bbox.height > 1 and c.bbox.width > 1
                        and not c.touches_frame_boundary):
                    table[(c.value, c.relative_pixels, c.area)].append(c)
        for ordinal, array in enumerate(layout["arrays"]):
            top, left, bottom, right = array["bbox"]
            focuses = set()
            for c in current:
                if c.bbox.top > top or c.bbox.bottom < bottom:
                    continue
                contained = []
                for index, col2 in enumerate(array["column_centers_twice"]):
                    width = max(s[1] for s in array["trait_shapes"])
                    if c.bbox.left <= (col2 - width + 1) // 2 and c.bbox.right >= (col2 + width - 1) // 2:
                        contained.append(index)
                if len(contained) == 1:
                    focuses.add(contained[0])
            if len(focuses) != 1:
                continue
            movers = []
            for signature, earlier in sorted(old.items()):
                later = new.get(signature, ())
                if len(earlier) != 1 or len(later) != 1:
                    continue
                a, b = earlier[0], later[0]
                if a.bbox.bottom >= top or b.bbox.bottom >= top:
                    continue
                if not (2 * left <= a.bbox.left + a.bbox.right <= 2 * right
                        and 2 * left <= b.bbox.left + b.bbox.right <= 2 * right):
                    continue
                delta = (b.bbox.top - a.bbox.top, b.bbox.left - a.bbox.left)
                if delta != (0, 0):
                    movers.append((signature, delta, b))
            if len(movers) != 1:
                continue
            index = next(iter(focuses))
            signature, delta, body = movers[0]
            witnesses.append((ordinal, index, array["values_by_column"][index],
                              signature, delta, array, body))
    # More than one relation remains an explicit unknown, not a ranked guess.
    identities = {(w[0], w[2], w[3], w[4]) for w in witnesses}
    if len(identities) != 1 or len(witnesses) < 2:
        return {}
    if len({w[1] for w in witnesses}) != len(witnesses):
        return {}
    _, _, vector, signature, delta, array, body = witnesses[-1]
    return {
        "focused_word_translation_unique": True,
        "focused_word_value_vector": vector,
        "focused_word_trait_shapes": array["trait_shapes"],
        "focused_word_delta": delta,
        "focused_word_body_relative_pixels": signature[1],
        "focused_word_body_area": signature[2],
        "focused_word_array_bbox": array["bbox"],
        "focused_word_body_after_bbox": (body.bbox.top, body.bbox.left, body.bbox.bottom, body.bbox.right),
        "focused_word_support_count": len(witnesses),
        "focused_word_geometry_ref": layouts[-1]["geometry_ref"],
    }


def enumerate_uniform_word_fits(layout, relations, *, traversal_order: str) -> tuple:
    """Exact repeated-vector endpoint fits, with one edit prefix per witness.

    The declarative caller chooses the traversal and interprets coincidence.
    This does not search a code space or infer individual trait semantics.
    """
    if traversal_order != "column_then_trait_ascending":
        return ()
    mappings = defaultdict(list)
    for relation in relations:
        vector = tuple(relation.get("word_values", ()))
        delta = tuple(relation.get("delta", ()))
        shapes = tuple(tuple(s) for s in relation.get("trait_shapes", ()))
        pixels = tuple(tuple(p) for p in relation.get("body_relative_pixels", ()))
        if not vector or len(delta) != 2 or delta == (0, 0) or len(shapes) != len(vector):
            continue
        mappings[(vector, delta, shapes, pixels, int(relation.get("body_area", 0)))].append(str(relation["claim_ref"]))
    witnesses = []
    for array in layout.get("arrays", ()):
        for (vector, delta, shapes, pixels, size), refs in sorted(mappings.items()):
            if shapes != tuple(array["trait_shapes"]):
                continue
            for pair in array["same_shape_pairs_above"]:
                if tuple(pair["relative_pixels"]) != pixels or pair["area"] != size:
                    continue
                for first, second in (pair["bboxes"], tuple(reversed(pair["bboxes"]))):
                    count = len(array["values_by_column"])
                    if (second[0] - first[0], second[1] - first[1]) != (count * delta[0], count * delta[1]):
                        continue
                    differences = tuple((column, row) for column, values in enumerate(array["values_by_column"])
                                        for row, (old, new) in enumerate(zip(values, vector)) if old != new)
                    for control in array["controls"]:
                        point = tuple(control["point"])
                        component_refs = tuple(control["component_refs"])
                        if differences:
                            column, row = differences[0]
                            point = (array["row_centers_twice"][row] // 2, array["column_centers_twice"][column] // 2)
                            component_refs = (array["component_refs_by_column"][column][row],)
                        if len(witnesses) >= 3:
                            raise ValueError("complete-word witness bound exceeded before materializing a fourth route")
                        witnesses.append({
                            "array_bbox": array["bbox"], "array_control_count": len(array["controls"]),
                            "point": point, "component_refs": component_refs,
                            "difference_count": len(differences), "word_values": vector,
                            "word_repeat_count": count, "delta": delta,
                            "body_before_bbox": first, "body_endpoint_bbox": second,
                            "relation_claim_refs": tuple(sorted(set(refs))),
                            "witness_ref": "measurement.uniform_word_fit." + stable_digest((array["bbox"], vector, delta, first, second, control["point"])),
                        })
    return tuple(witnesses)
