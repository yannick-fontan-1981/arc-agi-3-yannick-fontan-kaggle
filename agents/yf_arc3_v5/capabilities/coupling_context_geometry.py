"""Bounded raster geometry under an explicitly supplied paired translation.

This reports intersections, not obstacle roles, safety or test eligibility.
Unknown/unsupported operators never become a zero-intersection measurement.
"""

from collections.abc import Mapping

from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest


def _field(value, name):
    # SRC transports immutable interned mappings. Do not reconstruct complete
    # scene/tracking models just to read a few descriptive fields.
    return value[name] if isinstance(value, Mapping) else getattr(value, name)


def measure_translation_context(*, scene, tracking, entity_refs, transformations,
                                maximum_steps, maximum_support_pixels, entity_member_refs=None,
                                include_support_digests=False):
    if not 1 <= maximum_steps <= 64 or not 1 <= maximum_support_pixels <= 8192:
        raise ValueError("invalid coupling translation geometry bound")
    unknown = FrozenMap({
        "translation_geometry_available": False,
        "translation_geometry_within_bound": True,
        "translated_support_outside_frame_counts": None,
        "swept_support_other_component_refs": None,
        "translated_pair_intersection_count": None,
        "translation_geometry_frame_ref": _field(tracking, "frame_ref"),
        **({"current_support_digests": None} if include_support_digests else {}),
    })
    if len(entity_refs) != 2 or len(set(entity_refs)) != 2:
        return unknown
    entities = {_field(item, "entity_id"): item for item in _field(tracking, "entities")}
    components = _field(_field(scene, "blocks"), "components")
    if len(entities) > 128 or len(components) > 128:
        return FrozenMap({**unknown, "translation_geometry_within_bound": False})
    members = (tuple((ref,) for ref in entity_refs)
               if entity_member_refs is None else entity_member_refs)
    if (len(members) != 2 or any(not group or len(group) > 64 for group in members)
            or len({ref for group in members for ref in group}) != sum(map(len, members))):
        raise ValueError("invalid disjoint coupling support membership")
    supports = []
    for ref, group in zip(entity_refs, members):
        if (len(group) == 1 and group[0] != ref) or any(member not in entities for member in group):
            return unknown
        parts = tuple(_field(entities[member], "component") for member in group)
        if sum(_field(part, "area") for part in parts) > maximum_support_pixels:
            return FrozenMap({**unknown, "translation_geometry_within_bound": False})
        if len(parts) == 1:
            # A primitive already has canonical geometry. Do not sort and
            # reconstruct it merely because compounds are also supported.
            part = parts[0]
            supports.append({"pixels": _field(part, "pixels"), "area": _field(part, "area"),
                "value": _field(part, "value"), "relative_pixels": _field(part, "relative_pixels"),
                "component_refs": (_field(part, "component_id"),)})
            continue
        pixels = tuple(sorted(pixel for part in parts for pixel in _field(part, "pixels")))
        if not pixels or len(set(pixels)) != len(pixels):
            return unknown
        top, left = min(row for row, col in pixels), min(col for row, col in pixels)
        supports.append({"pixels": pixels, "area": len(pixels),
            "value": tuple(sorted({_field(part, "value") for part in parts})),
            "relative_pixels": tuple((row - top, col - left) for row, col in pixels),
            "component_refs": tuple(_field(part, "component_id") for part in parts)})
    support_digests = (tuple(stable_digest(support["pixels"]) for support in supports)
                       if include_support_digests else None)
    if include_support_digests:
        unknown = FrozenMap({**unknown, "current_support_digests": support_digests})
    if len(transformations) != 2:
        return unknown
    required = ("delta_row", "delta_col", "delta_height", "delta_width", "delta_area",
                "value_before", "value_after", "shape_before", "shape_after")
    translated, swept, own_component_refs = [], set(), set()
    for support, descriptor in zip(supports, transformations):
        if any(key not in descriptor for key in required):
            return unknown
        dr, dc = descriptor["delta_row"], descriptor["delta_col"]
        if type(dr) is not int or type(dc) is not int or (dr and dc):
            return unknown  # The supplied raster path contract is orthogonal.
        if (any(descriptor[key] != 0 for key in ("delta_height", "delta_width", "delta_area"))
                or descriptor["value_before"] != descriptor["value_after"]
                or descriptor["value_before"] != support["value"]
                or descriptor["shape_before"] != descriptor["shape_after"]
                or descriptor["shape_before"] != stable_digest(support["relative_pixels"])):
            return unknown
        steps = abs(dr) + abs(dc)
        if steps > maximum_steps or support["area"] * (steps + 1) > maximum_support_pixels:
            return FrozenMap({**unknown, "translation_geometry_within_bound": False})
        pixels = frozenset(support["pixels"])
        own_component_refs.update(support["component_refs"])
        translated.append(frozenset((row + dr, col + dc) for row, col in pixels))
        unit_row, unit_col = (0 if not dr else dr // abs(dr)), (0 if not dc else dc // abs(dc))
        for step in range(steps + 1):
            swept.update((row + step * unit_row, col + step * unit_col) for row, col in pixels)
            if len(swept) > maximum_support_pixels:
                return FrozenMap({**unknown, "translation_geometry_within_bound": False})
    other_refs = []
    measured_pixels = 0
    for component in components:
        if _field(component, "component_id") in own_component_refs:
            continue
        measured_pixels += _field(component, "area")
        if measured_pixels > 65536:
            return FrozenMap({**unknown, "translation_geometry_within_bound": False})
        if any(pixel in swept for pixel in _field(component, "pixels")):
            other_refs.append(_field(component, "component_id"))
    frame_rows = _field(_field(scene, "frame"), "rows")
    height, width = len(frame_rows), len(frame_rows[0])
    return FrozenMap({
        "translation_geometry_available": True,
        "translation_geometry_within_bound": True,
        "translated_support_outside_frame_counts": tuple(
            sum(not (0 <= row < height and 0 <= col < width)
                for row, col in support) for support in translated),
        "swept_support_other_component_refs": tuple(sorted(other_refs)),
        "translated_pair_intersection_count": len(translated[0] & translated[1]),
        "translation_geometry_frame_ref": _field(tracking, "frame_ref"),
        **({"current_support_digests": support_digests} if include_support_digests else {}),
    })
