"""Bounded retrospective geometry, including independently retained masks.

An occluded raster is not substituted for its complete support. The caller
supplies a recorded layer description and its translations; every compatible
current placement is retained. These are conditional measurements, not a win,
an identity decision, an objective or an action authorization.
"""

from collections.abc import Mapping

from agents.yf_arc3_v5.logos.types import FrozenMap


def _field(value, name):
    return value[name] if isinstance(value, Mapping) else getattr(value, name)


def measure_terminal_support_coverage(*, layer_rows, current_scene,
                                      maximum_layers, maximum_pixels,
                                      maximum_components, maximum_rows):
    if len(layer_rows) > maximum_layers:
        raise ValueError("terminal coverage layer bound exceeded")
    if current_scene is None:
        return ()
    components = _field(_field(current_scene, "blocks"), "components")
    raster = _field(_field(current_scene, "frame"), "rows")
    height, width = len(raster), len(raster[0])
    if len(components) > maximum_components or height * width > maximum_pixels:
        raise ValueError("terminal coverage scene bound exceeded")
    supports = tuple(frozenset(map(tuple, row["component_geometry"]["pixels"]))
                     for row in layer_rows)
    if sum(map(len, supports)) > maximum_pixels:
        raise ValueError("terminal coverage support bound exceeded")
    stationary = tuple(i for i, row in enumerate(layer_rows)
        if row["prediction_delta_row"] == row["prediction_delta_col"] == 0
        and row["last_observed_delta_row"] == row["last_observed_delta_col"] == 0)
    stationary_values = {}
    for i in stationary:
        for pixel in supports[i]:
            stationary_values.setdefault(pixel, set()).add(layer_rows[i]["component_geometry"]["value"])
    result = []
    for i, layer in enumerate(layer_rows):
        dr, dc = layer["prediction_delta_row"], layer["prediction_delta_col"]
        if (dr == dc == 0) or not layer.get("prior_entity_ref"):
            continue
        geometry = layer["component_geometry"]
        box = geometry["bbox"]
        placements = set()
        for component in components:
            current_box = _field(component, "bbox")
            if (_field(component, "value") != geometry["value"]
                    or _field(current_box, "bottom") - _field(current_box, "top") != box["bottom"] - box["top"]
                    or _field(current_box, "right") - _field(current_box, "left") != box["right"] - box["left"]):
                continue
            offset = (_field(current_box, "top") - box["top"], _field(current_box, "left") - box["left"])
            support = frozenset((r + offset[0], c + offset[1]) for r, c in supports[i])
            visible = frozenset(map(tuple, _field(component, "pixels")))
            if not visible or not visible <= support or offset in placements:
                continue
            # Missing pixels must have a value of an independently retained
            # stationary layer at that very position. No hole is filled merely
            # because doing so would satisfy a target.
            if any(not (0 <= r < height and 0 <= c < width)
                   or (raster[r][c] != geometry["value"]
                       and raster[r][c] not in stationary_values.get((r, c), ()))
                   for r, c in support):
                continue
            placements.add(offset)
            following = frozenset((r + dr, c + dc) for r, c in support)
            for j in stationary:
                fixed = layer_rows[j]
                if not fixed.get("prior_entity_ref"):
                    continue
                target = supports[j]
                fixed_value = fixed["component_geometry"]["value"]
                if fixed_value == geometry["value"]:
                    continue
                if any(not (0 <= r < height and 0 <= c < width)
                       or raster[r][c] not in ({fixed_value, geometry["value"]}
                                              if (r, c) in support else {fixed_value})
                       for r, c in target):
                    continue
                result.append(FrozenMap({
                    "moving_entity_ref": layer["prior_entity_ref"],
                    "fixed_entity_ref": fixed["prior_entity_ref"],
                    "moving_support_pixel_count": len(support),
                    "fixed_support_pixel_count": len(target),
                    "intersection_pixel_count_before": len(support & target),
                    "intersection_pixel_count_after": len(following & target),
                    "uncovered_pixel_count_before": len(target - support),
                    "uncovered_pixel_count_after": len(target - following),
                    "predicted_supports_equal": following == target,
                    "predicted_outside_pixel_count": sum(not (0 <= r < height and 0 <= c < width) for r, c in following),
                    "visible_moving_pixel_count": len(visible),
                    "row_gap_before": min(r for r, _ in target) - min(r for r, _ in support),
                    "column_gap_before": min(c for _, c in target) - min(c for _, c in support),
                    "row_gap_after": min(r for r, _ in target) - min(r for r, _ in following),
                    "column_gap_after": min(c for _, c in target) - min(c for _, c in following),
                }))
                if len(result) > maximum_rows:
                    raise ValueError("terminal coverage placement bound exceeded; no ties dropped")
    return tuple(result)
