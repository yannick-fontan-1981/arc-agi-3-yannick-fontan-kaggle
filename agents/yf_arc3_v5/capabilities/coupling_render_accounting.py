"""Bounded rendered-pixel accounting under supplied rigid translations.

Every overlapping component is reported. This neither identifies an occluder
nor replaces a body, proves passage free, or decides a coupling's status.
"""

from collections.abc import Mapping

from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest


def _field(value, name):
    return value[name] if isinstance(value, Mapping) else getattr(value, name)


def measure_translation_render(*, before_scene, before_tracking, after_scene,
                               entity_refs, transformations,
                               maximum_steps, maximum_support_pixels):
    if not 1 <= maximum_steps <= 64 or not 1 <= maximum_support_pixels <= 8192:
        raise ValueError("invalid coupling render accounting bound")
    unknown = FrozenMap({
        "translation_render_available": False,
        "translation_render_within_bound": True,
        "translation_render_before_frame_ref": _field(before_tracking, "frame_ref"),
        "predicted_render_values_equal": None,
        "predicted_render_representation_equal": None,
        "predicted_support_counts": None,
        "predicted_support_digests": None,
        "predicted_same_value_pixel_counts": None,
        "predicted_replaced_pixel_counts": None,
        "predicted_outside_pixel_counts": None,
        "replaced_pixel_prior_component_rows": None,
        "predicted_same_value_component_rows": None,
    })

    def bounded_unknown():
        return FrozenMap.from_frozen_items(dict((*unknown.items(),
            ("translation_render_within_bound", False))))

    if len(entity_refs) != 2 or len(set(entity_refs)) != 2 or len(transformations) != 2:
        return unknown
    entities = _field(before_tracking, "entities")
    prior_components = _field(_field(before_scene, "blocks"), "components")
    current_components = _field(_field(after_scene, "blocks"), "components")
    if any(len(items) > 128 for items in (entities, prior_components, current_components)):
        return bounded_unknown()
    if any(sum(_field(item, "area") for item in items) > 65536
           for items in (prior_components, current_components)):
        return bounded_unknown()
    rows = _field(_field(after_scene, "frame"), "rows")
    prior_rows = _field(_field(before_scene, "frame"), "rows")
    height, width = len(rows), len(rows[0])
    if height != len(prior_rows) or width != len(prior_rows[0]):
        return unknown
    if height * width > 65536:
        return bounded_unknown()
    by_ref = {_field(item, "entity_id"): item for item in entities}
    if len(by_ref) != len(entities) or any(ref not in by_ref for ref in entity_refs):
        return unknown
    required = ("delta_row", "delta_col", "delta_height", "delta_width", "delta_area",
                "value_before", "value_after", "shape_before", "shape_after")
    supports, values, own_refs = [], [], set()
    for ref, descriptor in zip(entity_refs, transformations):
        component = _field(by_ref[ref], "component")
        if any(key not in descriptor for key in required):
            return unknown
        dr, dc = descriptor["delta_row"], descriptor["delta_col"]
        if type(dr) is not int or type(dc) is not int or (dr and dc):
            return unknown
        if (any(descriptor[key] != 0 for key in ("delta_height", "delta_width", "delta_area"))
                or descriptor["value_before"] != descriptor["value_after"]
                or descriptor["value_before"] != _field(component, "value")
                or descriptor["shape_before"] != descriptor["shape_after"]
                or descriptor["shape_before"] != stable_digest(_field(component, "relative_pixels"))):
            return unknown
        if abs(dr) + abs(dc) > maximum_steps:
            return bounded_unknown()
        supports.append(frozenset((row + dr, col + dc)
                                 for row, col in _field(component, "pixels")))
        if sum(map(len, supports)) > maximum_support_pixels:
            return bounded_unknown()
        values.append(_field(component, "value"))
        own_refs.add(_field(component, "component_id"))
    enclosing_refs = frozenset(_field(before_scene, "frame_enclosing_component_refs"))
    visible_counts, replaced_counts, outside_counts, coverage_rows, component_rows = [], [], [], [], []
    for support, value in zip(supports, values):
        inside = frozenset((row, col) for row, col in support
                           if 0 <= row < height and 0 <= col < width)
        replaced = frozenset((row, col) for row, col in inside if rows[row][col] != value)
        outside_counts.append(len(support) - len(inside))
        visible_counts.append(len(inside) - len(replaced))
        replaced_counts.append(len(replaced))
        coverage = []
        for component in prior_components:
            component_ref = _field(component, "component_id")
            if component_ref in own_refs:
                continue
            intersection = replaced.intersection(_field(component, "pixels"))
            if intersection:
                component_value = _field(component, "value")
                coverage.append(FrozenMap({
                    "component_ref": component_ref, "value": component_value,
                    "intersection_count": len(intersection),
                    "current_equal_value_count": sum(rows[row][col] == component_value
                                                     for row, col in intersection),
                    "frame_enclosing": component_ref in enclosing_refs,
                }))
        coverage_rows.append(tuple(coverage))
        intersections = []
        for component in current_components:
            if _field(component, "value") != value:
                continue
            pixels = frozenset(_field(component, "pixels"))
            intersection_count = len(support & pixels)
            if intersection_count:
                intersections.append(FrozenMap({
                    "component_ref": _field(component, "component_id"),
                    "intersection_count": intersection_count,
                    "outside_prediction_count": len(pixels - support),
                }))
        component_rows.append(tuple(intersections))
    return FrozenMap({
        "translation_render_available": True,
        "translation_render_within_bound": True,
        "translation_render_before_frame_ref": _field(before_tracking, "frame_ref"),
        "predicted_render_values_equal": not any((*replaced_counts, *outside_counts)),
        "predicted_render_representation_equal": not any((*replaced_counts, *outside_counts)) and all(
            len(items) == 1 and items[0]["intersection_count"] == len(support)
            and items[0]["outside_prediction_count"] == 0
            for support, items in zip(supports, component_rows)),
        "predicted_support_counts": tuple(map(len, supports)),
        "predicted_support_digests": tuple(stable_digest((value, tuple(sorted(support))))
                                          for value, support in zip(values, supports)),
        "predicted_same_value_pixel_counts": tuple(visible_counts),
        "predicted_replaced_pixel_counts": tuple(replaced_counts),
        "predicted_outside_pixel_counts": tuple(outside_counts),
        "replaced_pixel_prior_component_rows": tuple(coverage_rows),
        "predicted_same_value_component_rows": tuple(component_rows),
    })
