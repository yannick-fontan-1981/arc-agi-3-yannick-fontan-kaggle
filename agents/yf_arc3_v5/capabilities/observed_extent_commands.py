"""Exact scoped extent, edge-origin and equal-translation observations only."""
from fractions import Fraction

from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest


def _point(row, col):
    return tuple((v.numerator, v.denominator) for v in (Fraction(row), Fraction(col)))


def _edges(component, extent):
    box = component.bbox
    if extent.axis == "column":
        row = Fraction(box.top + box.bottom, 2)
        origin = box.left if extent.stable_edge == "left" else box.right + 1
        direction = 1 if extent.stable_edge == "left" else -1
        return _point(row, origin), _point(row, origin + direction * box.width), (0, direction)
    col = Fraction(box.left + box.right, 2)
    origin = box.top if extent.stable_edge == "top" else box.bottom + 1
    direction = 1 if extent.stable_edge == "top" else -1
    return _point(origin, col), _point(origin + direction * box.height, col), (direction, 0)


def measure_observed_extent_commands(*, analysis, before_tracking, after_tracking,
                                    operator_scope_ref, operator_action_data,
                                    source_palette_value, action_ref, context_epoch):
    if operator_scope_ref is None or operator_action_data is None or source_palette_value is None:
        return ()
    if max(len(before_tracking.entities), len(after_tracking.entities)) > 128:
        raise ValueError("extent observation tracking bound exceeded")
    if max(len(analysis.extent_transitions), len(analysis.translations)) > 64:
        raise ValueError("extent observation transition bound exceeded")
    x, y = operator_action_data.get("x"), operator_action_data.get("y")
    if type(x) is not int or type(y) is not int:
        return ()
    before = {item.entity_id: item for item in before_tracking.entities}
    after = {item.entity_id: item for item in after_tracking.entities}
    if len(before) != len(before_tracking.entities) or len(after) != len(after_tracking.entities):
        raise ValueError("duplicate extent observation identity")
    extents = tuple(e for e in analysis.extent_transitions
                    if e.entity_ref in before and e.entity_ref in after)
    palette_count = sum(before[e.entity_ref].component.value == source_palette_value
                        and after[e.entity_ref].component.value == source_palette_value for e in extents)
    stationary = tuple(sorted(ref for ref, item in after.items() if ref in before
        and item.component.value == before[ref].component.value
        and item.component.pixels == before[ref].component.pixels))
    rows = []
    for extent in extents:
        first, second = before[extent.entity_ref].component, after[extent.entity_ref].component
        origin_before, end_before, axis = _edges(first, extent)
        origin_after, end_after, _ = _edges(second, extent)
        delta = tuple(extent.delta_extent * v for v in axis)
        a, b = first.bbox, second.bbox
        before_length, after_length = ((a.width, b.width) if extent.axis == "column" else (a.height, b.height))
        extent_consistent = (before_length == extent.before_extent and after_length == extent.after_extent
            and after_length - before_length == extent.delta_extent)
        orthogonal_bounds_equal = ((a.top, a.bottom) == (b.top, b.bottom)
            if extent.axis == "column" else (a.left, a.right) == (b.left, b.right))
        co_movers = tuple(sorted(t.entity_ref for t in analysis.translations
            if (t.delta_row, t.delta_col) == delta
            and t.entity_ref in before and t.entity_ref in after
            and after[t.entity_ref].match_kind == "translated_exact"
            and (after[t.entity_ref].component.bbox.top - before[t.entity_ref].component.bbox.top,
                 after[t.entity_ref].component.bbox.left - before[t.entity_ref].component.bbox.left) == delta))
        if len(set(co_movers)) != len(co_movers):
            raise ValueError("duplicate extent co-translation identity")
        outside = not any(box.top <= y <= box.bottom and box.left <= x <= box.right for box in (a, b))
        rows.append(FrozenMap({
            "observation_token": stable_digest((analysis.transition_ref, operator_scope_ref, extent.entity_ref)),
            "source_transition_ref": analysis.transition_ref,
            "before_frame_ref": before_tracking.frame_ref,
            "after_frame_ref": after_tracking.frame_ref,
            "context_epoch": context_epoch,
            "operator_scope_ref": operator_scope_ref,
            "observed_action_ref": action_ref,
            "observed_action_data": FrozenMap(operator_action_data),
            "extent_entity_ref": extent.entity_ref,
            "extent_before": extent.before_extent,
            "extent_after": extent.after_extent,
            "delta_extent": extent.delta_extent,
            "axis_vector": axis,
            "origin_before": origin_before, "origin_after": origin_after,
            "free_edge_before": end_before, "free_edge_after": end_after,
            "origin_stable": origin_before == origin_after and orthogonal_bounds_equal,
            "extent_measurements_consistent": extent_consistent,
            "point_outside_extent_boxes": outside,
            "extent_palette_matches_source": first.value == second.value == source_palette_value,
            "same_palette_extent_occurrence_count": palette_count,
            "co_translated_entity_refs": co_movers,
            "co_translation_delta": delta,
            "stationary_entity_refs": stationary,
            "nonzero_extent_delta": extent.delta_extent != 0,
        }))
    return tuple(rows)
