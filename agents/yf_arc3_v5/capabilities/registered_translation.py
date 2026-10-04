"""Exact coordinate projection under a frame-bound perceptual registration.

No camera/terrain verdict: both interpretations share this relative frame.
Raw screen observations stay in the transition analysis, unmodified.
"""
from agents.yf_arc3_v5.logos.types import FrozenMap


def register_scene_grid_pair(*, before_rows, after_rows, geometry,
                             before_frame_ref, after_frame_ref, edited_prior_cell_indexes=()):
    """Recount the exact requested frame pair, not an intermediate fast-path pair."""
    from agents.yf_arc3_v5.capabilities.contracts import FrameGrid, KnownGridCellRecountInput
    from agents.yf_arc3_v5.capabilities.cell_recount import (
        recount_known_grid_cells, measure_retained_grid_viewport_shift_frontier,
    )
    grid = dict(grid_ref=str(geometry[0]), row_offset=int(geometry[1]), col_offset=int(geometry[2]),
        cell_height=int(geometry[3]), cell_width=int(geometry[4]), row_pitch=int(geometry[5]),
        col_pitch=int(geometry[6]), logical_rows=int(geometry[7]), logical_columns=int(geometry[8]))
    prior = recount_known_grid_cells(KnownGridCellRecountInput(frame=FrameGrid(rows=before_rows), **grid))
    current = recount_known_grid_cells(KnownGridCellRecountInput(frame=FrameGrid(rows=after_rows),
        prior_type_patterns=prior.type_patterns, prior_cell_type_ids=prior.cell_type_ids,
        prior_separator_values=prior.separator_values, **grid))
    measured = measure_retained_grid_viewport_shift_frontier(prior, current,
        logical_rows=grid["logical_rows"], logical_columns=grid["logical_columns"],
        executed_action_delta=None, prior_effectful_point_source_pattern_refs=(),
        maximum_shift=max(3, min(max(grid["logical_rows"], grid["logical_columns"]), 8)),
        ignored_prior_cell_indexes=tuple(dict.fromkeys((*prior.unique_multivalue_cell_indexes,
            *edited_prior_cell_indexes))), ignored_current_cell_indexes=current.unique_multivalue_cell_indexes)
    if measured.shift_delta is None:
        return None
    return (before_frame_ref, after_frame_ref, measured.shift_delta[0] * grid["row_pitch"],
        measured.shift_delta[1] * grid["col_pitch"])


def registration_delta(registration, *, before_frame_ref, after_frame_ref):
    if registration is None:
        return (0, 0)
    if (not isinstance(registration, (tuple, list)) or len(registration) != 4
            or tuple(registration[:2]) != (before_frame_ref, after_frame_ref)
            or any(type(value) is not int for value in registration[2:])):
        raise ValueError("scene registration is not bound to this frame pair")
    return tuple(registration[2:])


def offset_transformations(descriptors, delta):
    if delta == (0, 0):
        return descriptors
    return tuple(FrozenMap.overlay(FrozenMap({
        "delta_row": item["delta_row"] + delta[0],
        "delta_col": item["delta_col"] + delta[1],
    }), item) for item in descriptors)


def project_relative_entity_rows(rows, delta):
    if delta == (0, 0):
        return rows
    if len(rows) > 128:
        raise ValueError("registered entity projection bound exceeded")
    result = []
    for row in rows:
        descriptor, = offset_transformations((row["descriptor"],), (-delta[0], -delta[1]))
        changed = bool(descriptor["delta_row"] or descriptor["delta_col"]
            or descriptor["delta_height"] or descriptor["delta_width"] or descriptor["delta_area"]
            or descriptor["value_before"] != descriptor["value_after"]
            or descriptor["shape_before"] != descriptor["shape_after"])
        result.append(FrozenMap.overlay(FrozenMap({"descriptor": descriptor, "changed": changed}), row))
    return tuple(result)
