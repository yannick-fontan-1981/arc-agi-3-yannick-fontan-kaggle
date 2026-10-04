"""Exact cell recount for a source-declared, still-valid grid geometry."""

from __future__ import annotations

from dataclasses import dataclass


from agents.yf_arc3_v5.capabilities.contracts import (
    BoundingBox,
    KnownGridCellRecountInput,
    KnownGridCellRecountDelta,
    KnownGridCellRecountResult,
    PeriodicCellDescription,
    PeriodicCellGridCandidate,
)
from agents.yf_arc3_v5.capabilities.frame import _cell_pattern_ref


@dataclass(frozen=True, slots=True)
class RetainedGridCellMeasurements:
    """Exact cell-local projection; makes no claim about grid eligibility."""

    cells: tuple[PeriodicCellDescription, ...]
    repeated_pattern_count: int
    distinct_pattern_count: int


@dataclass(frozen=True, slots=True)
class GridFootprintTransport:
    """Cell containment before/after movement, without assigning a role."""

    before_cells: tuple[int | None, ...]
    after_cells: tuple[int | None, ...]
    same_moving_cell: bool
    lattice_displacement: bool


@dataclass(frozen=True, slots=True)
class RetainedGridNeighborWitness:
    """One exact neighboring cell under a declared directional action."""

    action_ref: str
    destination_cell_index: int
    destination_type_id: int
    destination_pattern_occurrence_count: int
    destination_previously_visited: bool


@dataclass(frozen=True, slots=True)
class RetainedGridMotionFrontier:
    """Three structural branches around one transported unique cell pattern."""

    before_cell_index: int | None
    after_cell_index: int | None
    motion_action_refs: tuple[str, ...]
    transported_pattern_changed: bool
    continuation: tuple[RetainedGridNeighborWitness, ...]
    orthogonal: tuple[RetainedGridNeighborWitness, ...]
    reverse: tuple[RetainedGridNeighborWitness, ...]
    # Bounded route witness toward the spatial frontier farthest from the
    # cursor's starting cell. DRM may supersede it with a knowledge frontier.
    exploratory_frontier: tuple[RetainedGridNeighborWitness, ...] = ()


@dataclass(frozen=True, slots=True)
class RetainedGridAdjacentPatternWitness:
    """One newly adjacent repeated multivalue pattern; no role is assigned."""

    cell_index: int
    type_id: int
    pattern_occurrence_count: int
    distinct_value_count: int


@dataclass(frozen=True, slots=True)
class RetainedGridInteractionFrontier:
    """Exact local structural change around one transported unique pattern."""

    before_cell_index: int | None
    after_cell_index: int | None
    nonlocal_changed_cell_count: int
    new_adjacent_repeated_multivalue: tuple[
        RetainedGridAdjacentPatternWitness, ...
    ]


@dataclass(frozen=True, slots=True)
class RetainedGridPointDeltaFrontier:
    """Exact local delta around one previously released grid-cell point."""

    point_cell_index: int
    changed_cell_indexes: tuple[int, ...]
    point_cell_pattern_changed: bool
    prior_pattern_ref: str
    current_pattern_ref: str
    point_cell_current_distinct_value_count: int
    orthogonal_cells_preserving_prior_pattern: tuple[int, ...]
    cells_preserving_prior_pattern: tuple[int, ...]
    current_unique_multivalue_cell_indexes: tuple[int, ...]
    point_adjacent_unique_multivalue_cell_indexes: tuple[int, ...]
    unique_multivalue_adjacent_cells_preserving_prior_pattern: tuple[int, ...]
    nearest_prior_pattern_cell_indexes: tuple[int, ...]
    nearest_prior_pattern_cell_distance: int | None
    axis_successor_prior_pattern_cell_indexes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RetainedGridViewportShiftFrontier:
    """Exact viewport translation and one bounded revealed-edge cell demand."""

    shift_delta: tuple[int, int] | None
    aligned_cell_count: int
    compared_cell_count: int
    action_transported_cell_index: int | None
    revealed_edge_delta: tuple[int, int] | None
    revealed_effectful_cell_indexes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RetainedGridTerrainRouteMeasurements:
    """Bounded route closure after one retained-grid terrain edit.

    These are mechanical measurements only.  A blocked cell is not labelled
    support or hazard here, and a known contact pattern is only counted.  DRM
    owns every interpretation and SRC owns whether the route is committed.
    """

    route_derived: bool
    requested_horizontal_step_count: int
    automatic_settlement_simulated: bool
    automatic_transition_count: int
    settled_successor_compared: bool
    first_blocked_after_settle_position_count: int
    known_contact_position_count: int


def measure_retained_grid_terrain_route(
    current: KnownGridCellRecountResult,
    *,
    logical_rows: int,
    logical_columns: int,
    mover_cell_index: int | None,
    opened_cell_index: int | None,
    known_viewport_grid_shift_deltas: tuple[tuple[str, int, int], ...],
    traversable_pattern_refs: tuple[str, ...],
    contact_pattern_refs: tuple[str, ...],
) -> RetainedGridTerrainRouteMeasurements:
    """Measure one horizontal entry followed by the learned automatic drift."""

    empty = RetainedGridTerrainRouteMeasurements(
        route_derived=False,
        requested_horizontal_step_count=0,
        automatic_settlement_simulated=False,
        automatic_transition_count=0,
        settled_successor_compared=False,
        first_blocked_after_settle_position_count=0,
        known_contact_position_count=0,
    )
    cell_count = logical_rows * logical_columns
    if (
        mover_cell_index is None
        or opened_cell_index is None
        or len(current.cell_type_ids) != cell_count
        or not 0 <= mover_cell_index < cell_count
        or not 0 <= opened_cell_index < cell_count
    ):
        return empty
    drift_rows = {
        -1 if int(delta_row) > 0 else 1
        for _action_ref, delta_row, _delta_col
        in known_viewport_grid_shift_deltas
        if int(delta_row) != 0
    }
    if len(drift_rows) != 1:
        return empty
    drift_row = next(iter(drift_rows))
    traversable = frozenset(traversable_pattern_refs)
    contacts = frozenset(contact_pattern_refs)

    def pattern_ref(cell_index: int) -> str:
        return _cell_pattern_ref(
            current.type_patterns[current.cell_type_ids[cell_index]]
        )

    mover_row, mover_column = divmod(mover_cell_index, logical_columns)
    _opened_row, opened_column = divmod(opened_cell_index, logical_columns)
    step_column = 0 if mover_column == opened_column else (
        -1 if opened_column < mover_column else 1
    )
    requested_steps = abs(opened_column - mover_column)
    horizontal_indexes = tuple(
        mover_row * logical_columns + column
        for column in range(
            mover_column + step_column,
            opened_column + step_column,
            step_column or 1,
        )
    )
    allowed = traversable | contacts
    if any(pattern_ref(index) not in allowed for index in horizontal_indexes):
        return empty

    current_row = mover_row
    automatic_transition_count = 0
    known_contact_indexes = {
        index for index in horizontal_indexes if pattern_ref(index) in contacts
    }
    first_blocked_count = 0
    for _ in range(logical_rows):
        advanced_row = current_row + drift_row
        if not 0 <= advanced_row < logical_rows:
            break
        advanced_index = advanced_row * logical_columns + opened_column
        advanced_pattern_ref = pattern_ref(advanced_index)
        if advanced_pattern_ref not in allowed:
            first_blocked_count = 1
            break
        automatic_transition_count += 1
        current_row = advanced_row
        if advanced_pattern_ref in contacts:
            known_contact_indexes.add(advanced_index)

    return RetainedGridTerrainRouteMeasurements(
        route_derived=True,
        requested_horizontal_step_count=requested_steps,
        automatic_settlement_simulated=True,
        automatic_transition_count=automatic_transition_count,
        settled_successor_compared=bool(
            automatic_transition_count or first_blocked_count
        ),
        first_blocked_after_settle_position_count=first_blocked_count,
        known_contact_position_count=len(known_contact_indexes),
    )


def _orthogonal_cell_indexes(
    cell_index: int,
    *,
    logical_rows: int,
    logical_columns: int,
) -> tuple[int, ...]:
    row, column = divmod(cell_index, logical_columns)
    return tuple(
        candidate_row * logical_columns + candidate_column
        for candidate_row, candidate_column in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        )
        if 0 <= candidate_row < logical_rows
        and 0 <= candidate_column < logical_columns
    )


def measure_retained_grid_interaction_frontier(
    prior: KnownGridCellRecountResult,
    current: KnownGridCellRecountResult,
    *,
    logical_rows: int,
    logical_columns: int,
) -> RetainedGridInteractionFrontier:
    """Measure newly adjacent repeated patterns after one cell translation.

    The result says only what became adjacent and how broadly the grid changed.
    It does not call a cell a cursor, obstacle, opening, target, or click locus.
    """

    empty = RetainedGridInteractionFrontier(
        before_cell_index=None,
        after_cell_index=None,
        nonlocal_changed_cell_count=0,
        new_adjacent_repeated_multivalue=(),
    )
    if (
        prior.grid_ref != current.grid_ref
        or len(prior.unique_multivalue_cell_indexes) != 1
        or len(current.unique_multivalue_cell_indexes) != 1
        or len(prior.cell_type_ids) != logical_rows * logical_columns
        or len(current.cell_type_ids) != logical_rows * logical_columns
    ):
        return empty
    before_index = prior.unique_multivalue_cell_indexes[0]
    after_index = current.unique_multivalue_cell_indexes[0]
    before_pattern = prior.type_patterns[prior.cell_type_ids[before_index]]
    after_pattern = current.type_patterns[current.cell_type_ids[after_index]]
    before_row, before_column = divmod(before_index, logical_columns)
    after_row, after_column = divmod(after_index, logical_columns)
    if (
        before_pattern != after_pattern
        or abs(after_row - before_row) + abs(after_column - before_column) != 1
    ):
        return empty

    prior_adjacent_patterns = {
        prior.type_patterns[prior.cell_type_ids[index]]
        for index in _orthogonal_cell_indexes(
            before_index,
            logical_rows=logical_rows,
            logical_columns=logical_columns,
        )
    }
    witnesses: list[RetainedGridAdjacentPatternWitness] = []
    for index in _orthogonal_cell_indexes(
        after_index,
        logical_rows=logical_rows,
        logical_columns=logical_columns,
    ):
        type_id = current.cell_type_ids[index]
        pattern = current.type_patterns[type_id]
        distinct_value_count = len({value for row in pattern for value in row})
        occurrence_count = current.type_occurrences[type_id]
        if (
            pattern != after_pattern
            and pattern not in prior_adjacent_patterns
            and distinct_value_count > 1
            and occurrence_count > 1
        ):
            witnesses.append(RetainedGridAdjacentPatternWitness(
                cell_index=index,
                type_id=type_id,
                pattern_occurrence_count=occurrence_count,
                distinct_value_count=distinct_value_count,
            ))
    return RetainedGridInteractionFrontier(
        before_cell_index=before_index,
        after_cell_index=after_index,
        nonlocal_changed_cell_count=sum(
            index not in {before_index, after_index}
            for index in current.changed_cell_indexes
        ),
        new_adjacent_repeated_multivalue=tuple(witnesses[:4]),
    )


def measure_retained_grid_point_delta_frontier(
    prior: KnownGridCellRecountResult,
    current: KnownGridCellRecountResult,
    *,
    point_cell_index: int,
    logical_rows: int,
    logical_columns: int,
    reference_axis_delta: tuple[int, int] | None = None,
) -> RetainedGridPointDeltaFrontier:
    """Measure a bounded point-conditioned cell delta without assigning roles."""

    cell_count = logical_rows * logical_columns
    if (
        prior.grid_ref != current.grid_ref
        or len(prior.cell_type_ids) != cell_count
        or len(current.cell_type_ids) != cell_count
        or not 0 <= point_cell_index < cell_count
    ):
        return RetainedGridPointDeltaFrontier(
            point_cell_index=point_cell_index,
            changed_cell_indexes=(),
            point_cell_pattern_changed=False,
            prior_pattern_ref="",
            current_pattern_ref="",
            point_cell_current_distinct_value_count=0,
            orthogonal_cells_preserving_prior_pattern=(),
            cells_preserving_prior_pattern=(),
            current_unique_multivalue_cell_indexes=(),
            point_adjacent_unique_multivalue_cell_indexes=(),
            unique_multivalue_adjacent_cells_preserving_prior_pattern=(),
            nearest_prior_pattern_cell_indexes=(),
            nearest_prior_pattern_cell_distance=None,
            axis_successor_prior_pattern_cell_indexes=(),
        )
    prior_pattern = prior.type_patterns[prior.cell_type_ids[point_cell_index]]
    current_pattern = current.type_patterns[current.cell_type_ids[point_cell_index]]
    preserving_neighbors = tuple(
        index
        for index in _orthogonal_cell_indexes(
            point_cell_index,
            logical_rows=logical_rows,
            logical_columns=logical_columns,
        )
        if current.type_patterns[current.cell_type_ids[index]] == prior_pattern
    )
    preserving_cells = tuple(
        index
        for index, type_id in enumerate(current.cell_type_ids)
        if current.type_patterns[type_id] == prior_pattern
    )
    point_adjacent_unique_multivalue_cells = tuple(
        index
        for index in _orthogonal_cell_indexes(
            point_cell_index,
            logical_rows=logical_rows,
            logical_columns=logical_columns,
        )
        if index in current.unique_multivalue_cell_indexes
    )
    mover_anchor_indexes = (
        tuple(current.unique_multivalue_cell_indexes)
        if len(current.unique_multivalue_cell_indexes) == 1
        else point_adjacent_unique_multivalue_cells
        if len(point_adjacent_unique_multivalue_cells) == 1
        else ()
    )
    unique_multivalue_neighbors = (
        tuple(
            index
            for index in _orthogonal_cell_indexes(
                mover_anchor_indexes[0],
                logical_rows=logical_rows,
                logical_columns=logical_columns,
            )
            if current.type_patterns[current.cell_type_ids[index]] == prior_pattern
        )
        if len(mover_anchor_indexes) == 1
        else ()
    )
    nearest_cells: tuple[int, ...] = ()
    nearest_distance: int | None = None
    if len(mover_anchor_indexes) == 1 and preserving_cells:
        unique_index = mover_anchor_indexes[0]
        unique_row, unique_column = divmod(unique_index, logical_columns)
        distances = tuple(
            (
                abs(row - unique_row) + abs(column - unique_column),
                index,
            )
            for index in preserving_cells
            for row, column in (divmod(index, logical_columns),)
        )
        nearest_distance = min(distance for distance, _index in distances)
        nearest_cells = tuple(
            index for distance, index in distances if distance == nearest_distance
        )
    axis_successors: tuple[int, ...] = ()
    if reference_axis_delta in {(-1, 0), (1, 0), (0, -1), (0, 1)}:
        point_row, point_column = divmod(point_cell_index, logical_columns)
        successor_row = point_row + int(reference_axis_delta[0])
        successor_column = point_column + int(reference_axis_delta[1])
        if 0 <= successor_row < logical_rows and 0 <= successor_column < logical_columns:
            successor_index = successor_row * logical_columns + successor_column
            if successor_index in preserving_cells:
                axis_successors = (successor_index,)
    return RetainedGridPointDeltaFrontier(
        point_cell_index=point_cell_index,
        changed_cell_indexes=tuple(current.changed_cell_indexes[:128]),
        point_cell_pattern_changed=prior_pattern != current_pattern,
        prior_pattern_ref=_cell_pattern_ref(prior_pattern),
        current_pattern_ref=_cell_pattern_ref(current_pattern),
        point_cell_current_distinct_value_count=len({
            value for row in current_pattern for value in row
        }),
        orthogonal_cells_preserving_prior_pattern=preserving_neighbors[:4],
        cells_preserving_prior_pattern=preserving_cells[:128],
        current_unique_multivalue_cell_indexes=tuple(
            current.unique_multivalue_cell_indexes[:3]
        ),
        point_adjacent_unique_multivalue_cell_indexes=(
            point_adjacent_unique_multivalue_cells[:4]
        ),
        unique_multivalue_adjacent_cells_preserving_prior_pattern=(
            unique_multivalue_neighbors[:4]
        ),
        nearest_prior_pattern_cell_indexes=nearest_cells[:3],
        nearest_prior_pattern_cell_distance=nearest_distance,
        axis_successor_prior_pattern_cell_indexes=axis_successors,
    )


def measure_retained_grid_viewport_shift_frontier(
    prior: KnownGridCellRecountResult,
    current: KnownGridCellRecountResult,
    *,
    logical_rows: int,
    logical_columns: int,
    executed_action_delta: tuple[int, int] | None,
    prior_effectful_point_source_pattern_refs: tuple[str, ...],
    maximum_shift: int = 3,
    ignored_prior_cell_indexes: tuple[int, ...] = (),
    ignored_current_cell_indexes: tuple[int, ...] = (),
) -> RetainedGridViewportShiftFrontier:
    """Measure one exact camera shift and the matching revealed-edge neighbor.

    The measurement assigns no cursor, gravity, barrier, or objective role.  It
    only aligns unchanged cell patterns, transports the prior unique cell by
    the executed action delta, and checks whether the cell toward the newly
    revealed edge has a pattern previously changed by a point action.
    """

    cell_count = logical_rows * logical_columns
    ignored_prior = frozenset(ignored_prior_cell_indexes)
    ignored_current = frozenset(ignored_current_cell_indexes)
    if any(type(index) is not int or not 0 <= index < cell_count
           for index in (*ignored_prior, *ignored_current)):
        raise ValueError("viewport registration mask is outside the retained grid")
    empty = RetainedGridViewportShiftFrontier(
        shift_delta=None,
        aligned_cell_count=0,
        compared_cell_count=0,
        action_transported_cell_index=None,
        revealed_edge_delta=None,
        revealed_effectful_cell_indexes=(),
    )
    if (
        cell_count <= 0
        or len(prior.cell_type_ids) != cell_count
        or len(current.cell_type_ids) != cell_count
    ):
        return empty
    prior_index: int | None = None
    transported_index: int | None = None
    transported_row: int | None = None
    transported_column: int | None = None
    if executed_action_delta is not None:
        if len(prior.unique_multivalue_cell_indexes) != 1:
            return empty
        prior_index = int(prior.unique_multivalue_cell_indexes[0])
        prior_row, prior_column = divmod(prior_index, logical_columns)
        transported_row = prior_row + int(executed_action_delta[0])
        transported_column = prior_column + int(executed_action_delta[1])
        if not (
            0 <= transported_row < logical_rows
            and 0 <= transported_column < logical_columns
        ):
            return empty
        transported_index = transported_row * logical_columns + transported_column
        if transported_index not in current.unique_multivalue_cell_indexes:
            return empty

    stationary_pairs = tuple(index for index in range(cell_count)
        if index not in ignored_prior and index not in ignored_current
        and index != prior_index and index != transported_index)
    if stationary_pairs and all(
        prior.type_patterns[prior.cell_type_ids[index]] == current.type_patterns[current.cell_type_ids[index]]
        for index in stationary_pairs
    ):
        return empty  # Zero shift is compatible; do not invent a periodic scroll.
    candidates: list[tuple[int, int, int, int]] = []
    # The caller supplies a geometry-derived mechanical bound.  Preserve a
    # hard safety ceiling, but do not silently clamp a measured multi-cell
    # transition back to the old three-cell assumption.
    shift_bound = max(1, min(int(maximum_shift), 16))
    for shift_row, shift_column in (
        *((delta, 0) for delta in range(-shift_bound, shift_bound + 1) if delta),
        *((0, delta) for delta in range(-shift_bound, shift_bound + 1) if delta),
    ):
        compared = 0
        aligned = 0
        for row in range(logical_rows):
            for column in range(logical_columns):
                current_row = row + shift_row
                current_column = column + shift_column
                if not (
                    0 <= current_row < logical_rows
                    and 0 <= current_column < logical_columns
                ):
                    continue
                old_index = row * logical_columns + column
                new_index = current_row * logical_columns + current_column
                # Edited cells and independently tracked foreground are not
                # registration anchors. They remain in the original recounts;
                # this mask excludes them only from exact background alignment.
                if old_index in ignored_prior or new_index in ignored_current:
                    continue
                if (
                    prior_index is not None
                    and old_index == prior_index
                ) or (
                    transported_index is not None
                    and new_index == transported_index
                ):
                    continue
                compared += 1
                old_pattern = prior.type_patterns[prior.cell_type_ids[old_index]]
                new_pattern = current.type_patterns[current.cell_type_ids[new_index]]
                aligned += int(old_pattern == new_pattern)
        if compared and aligned == compared:
            candidates.append((shift_row, shift_column, aligned, compared))
    if len(candidates) != 1:
        return empty
    shift_row, shift_column, aligned, compared = candidates[0]
    revealed_delta = (
        -1 if shift_row > 0 else 1 if shift_row < 0 else 0,
        -1 if shift_column > 0 else 1 if shift_column < 0 else 0,
    )
    target_row = (
        transported_row + revealed_delta[0]
        if transported_row is not None
        else None
    )
    target_column = (
        transported_column + revealed_delta[1]
        if transported_column is not None
        else None
    )
    demanded: tuple[int, ...] = ()
    if (
        target_row is not None
        and target_column is not None
        and 0 <= target_row < logical_rows
        and 0 <= target_column < logical_columns
    ):
        target_index = target_row * logical_columns + target_column
        target_pattern = current.type_patterns[current.cell_type_ids[target_index]]
        if _cell_pattern_ref(target_pattern) in prior_effectful_point_source_pattern_refs:
            demanded = (target_index,)
    return RetainedGridViewportShiftFrontier(
        shift_delta=(shift_row, shift_column),
        aligned_cell_count=aligned,
        compared_cell_count=compared,
        action_transported_cell_index=transported_index,
        revealed_edge_delta=revealed_delta,
        revealed_effectful_cell_indexes=demanded,
    )


def measure_retained_grid_viewport_shift_sequence(
    recounts: tuple[KnownGridCellRecountResult, ...],
    *,
    logical_rows: int,
    logical_columns: int,
    prior_effectful_point_source_pattern_refs: tuple[str, ...],
    maximum_shift: int = 3,
) -> RetainedGridViewportShiftFrontier:
    """Compose exact cardinal shifts measured across transition frames.

    Intermediate animation frames are treated as additional observations, not
    as semantic decisions.  Each consecutive pair must yield one unique
    cardinal alignment; mixed or ambiguous directions fail closed.  The
    resulting displacement is the bounded sum of those measured steps.
    """

    if len(recounts) < 2:
        return RetainedGridViewportShiftFrontier(
            shift_delta=None,
            aligned_cell_count=0,
            compared_cell_count=0,
            action_transported_cell_index=None,
            revealed_edge_delta=None,
            revealed_effectful_cell_indexes=(),
        )
    measured = tuple(
        measure_retained_grid_viewport_shift_frontier(
            prior,
            current,
            logical_rows=logical_rows,
            logical_columns=logical_columns,
            executed_action_delta=None,
            prior_effectful_point_source_pattern_refs=(
                prior_effectful_point_source_pattern_refs
            ),
            maximum_shift=maximum_shift,
        )
        for prior, current in zip(recounts, recounts[1:])
    )
    valid = tuple(item for item in measured if item.shift_delta is not None)
    if not valid:
        return RetainedGridViewportShiftFrontier(
            shift_delta=None,
            aligned_cell_count=0,
            compared_cell_count=0,
            action_transported_cell_index=None,
            revealed_edge_delta=None,
            revealed_effectful_cell_indexes=(),
        )
    directions = {
        (
            0 if item.shift_delta[0] == 0 else (1 if item.shift_delta[0] > 0 else -1),
            0 if item.shift_delta[1] == 0 else (1 if item.shift_delta[1] > 0 else -1),
        )
        for item in valid
    }
    if len(valid) != len(measured) or len(directions) != 1:
        return RetainedGridViewportShiftFrontier(
            shift_delta=None,
            aligned_cell_count=0,
            compared_cell_count=0,
            action_transported_cell_index=None,
            revealed_edge_delta=None,
            revealed_effectful_cell_indexes=(),
        )
    shift_row = sum(int(item.shift_delta[0]) for item in valid)
    shift_column = sum(int(item.shift_delta[1]) for item in valid)
    revealed_edge_delta = (
        -1 if shift_row > 0 else 1 if shift_row < 0 else 0,
        -1 if shift_column > 0 else 1 if shift_column < 0 else 0,
    )
    demanded = tuple(
        sorted(
            {
                int(index)
                for item in valid
                for index in item.revealed_effectful_cell_indexes
            }
        )
    )[:3]
    return RetainedGridViewportShiftFrontier(
        shift_delta=(shift_row, shift_column),
        aligned_cell_count=sum(item.aligned_cell_count for item in valid),
        compared_cell_count=sum(item.compared_cell_count for item in valid),
        action_transported_cell_index=None,
        revealed_edge_delta=revealed_edge_delta,
        revealed_effectful_cell_indexes=demanded,
    )


def measure_retained_grid_motion_frontier(
    prior: KnownGridCellRecountResult,
    current: KnownGridCellRecountResult,
    *,
    logical_rows: int,
    logical_columns: int,
    interface_action_translation_deltas: tuple[tuple[str, int, int], ...],
    visited_cell_indexes: tuple[int, ...] = (),
    frontier_origin_cell_index: int | None = None,
) -> RetainedGridMotionFrontier:
    """Group exact neighbors as continue, orthogonal pair, and reverse.

    This is structural measurement only.  It neither names the transported
    pattern as a cursor nor chooses one branch or action.
    """

    empty = RetainedGridMotionFrontier(
        before_cell_index=None,
        after_cell_index=None,
        motion_action_refs=(),
        transported_pattern_changed=False,
        continuation=(),
        orthogonal=(),
        reverse=(),
        exploratory_frontier=(),
    )
    if (
        prior.grid_ref != current.grid_ref
        or len(prior.unique_multivalue_cell_indexes) != 1
        or len(current.unique_multivalue_cell_indexes) != 1
        or len(current.cell_type_ids) != logical_rows * logical_columns
    ):
        return empty
    before_index = prior.unique_multivalue_cell_indexes[0]
    after_index = current.unique_multivalue_cell_indexes[0]
    before_pattern = prior.type_patterns[prior.cell_type_ids[before_index]]
    after_pattern = current.type_patterns[current.cell_type_ids[after_index]]
    if before_index == after_index:
        return empty
    before_row, before_col = divmod(before_index, logical_columns)
    after_row, after_col = divmod(after_index, logical_columns)
    movement = (after_row - before_row, after_col - before_col)
    if abs(movement[0]) + abs(movement[1]) != 1:
        return empty
    transported_pattern_changed = before_pattern != after_pattern
    if transported_pattern_changed and frozenset(current.changed_cell_indexes) != {
        before_index,
        after_index,
    }:
        # A pose-changing transport is admissible only when the complete grid
        # delta is exactly the vacated and entered cells.
        return empty

    normalized_actions = tuple(
        (
            str(action_ref),
            0 if int(delta_row) == 0 else (1 if int(delta_row) > 0 else -1),
            0 if int(delta_col) == 0 else (1 if int(delta_col) > 0 else -1),
        )
        for action_ref, delta_row, delta_col
        in interface_action_translation_deltas
        if (int(delta_row) == 0) ^ (int(delta_col) == 0)
    )
    motion_action_refs = tuple(
        action_ref
        for action_ref, delta_row, delta_col in normalized_actions
        if (delta_row, delta_col) == movement
    )
    if not motion_action_refs:
        return empty
    visited = frozenset((*visited_cell_indexes, before_index, after_index))
    grouped: dict[str, list[RetainedGridNeighborWitness]] = {
        "continuation": [],
        "orthogonal": [],
        "reverse": [],
    }
    for action_ref, delta_row, delta_col in sorted(normalized_actions):
        destination_row = after_row + delta_row
        destination_col = after_col + delta_col
        if not (
            0 <= destination_row < logical_rows
            and 0 <= destination_col < logical_columns
        ):
            continue
        destination_index = destination_row * logical_columns + destination_col
        destination_type_id = current.cell_type_ids[destination_index]
        witness = RetainedGridNeighborWitness(
            action_ref=action_ref,
            destination_cell_index=destination_index,
            destination_type_id=destination_type_id,
            destination_pattern_occurrence_count=(
                current.type_occurrences[destination_type_id]
            ),
            destination_previously_visited=destination_index in visited,
        )
        direction = (delta_row, delta_col)
        branch = (
            "continuation"
            if direction == movement
            else "reverse"
            if direction == (-movement[0], -movement[1])
            else "orthogonal"
        )
        grouped[branch].append(witness)
    exploratory_frontier: tuple[RetainedGridNeighborWitness, ...] = ()
    if frontier_origin_cell_index is not None and 0 <= int(frontier_origin_cell_index) < logical_rows * logical_columns:
        origin_row, origin_col = divmod(int(frontier_origin_cell_index), logical_columns)
        directions = {
            action_ref: (delta_row, delta_col)
            for action_ref, delta_row, delta_col in normalized_actions
        }
        frontier_candidates: list[tuple[int, int, str, int]] = []
        for action_ref, (delta_row, delta_col) in sorted(directions.items()):
            is_reverse = (delta_row, delta_col) == (-movement[0], -movement[1])
            row, col = after_row + delta_row, after_col + delta_col
            if not (0 <= row < logical_rows and 0 <= col < logical_columns):
                continue
            first_index = row * logical_columns + col
            corridor_type_id = current.cell_type_ids[first_index]
            corridor_pattern = current.type_patterns[corridor_type_id]
            reference_pattern = current.type_patterns[
                current.cell_type_ids[before_index]
            ]
            first_is_corridor = corridor_pattern == reference_pattern
            steps = 1
            last_reachable = first_index
            first_distinct_repeated: int | None = None
            while True:
                row += delta_row
                col += delta_col
                if not (0 <= row < logical_rows and 0 <= col < logical_columns):
                    break
                index = row * logical_columns + col
                type_id = current.cell_type_ids[index]
                pattern = current.type_patterns[type_id]
                if pattern == corridor_pattern:
                    steps += 1
                    last_reachable = index
                    continue
                # A singleton transition cell is a passage, not an "after".
                if current.type_occurrences[type_id] == 1:
                    steps += 1
                    last_reachable = index
                    continue
                # A less-complex repeated pattern is another reachable region.
                # Keep the first such boundary as the historical witness;
                # when no distinct region exists, the corridor endpoint itself
                # remains the useful spatial frontier before the wall.
                if (
                    pattern != corridor_pattern
                    and current.type_occurrences[type_id] > 1
                    and len({value for row_values in pattern for value in row_values})
                    <= len({value for row_values in corridor_pattern for value in row_values})
                ):
                    first_distinct_repeated = index
                    break
                break
            if first_distinct_repeated is not None:
                target_index = first_distinct_repeated
            elif steps >= 2 or first_is_corridor:
                target_index = last_reachable
            else:
                continue
            if target_index in visited:
                # A reverse ray that only returns to the already traversed
                # corridor is not a new spatial frontier.
                continue
            if is_reverse and first_distinct_repeated is None:
                continue
            target_row, target_col = divmod(target_index, logical_columns)
            distance_from_origin = abs(target_row - origin_row) + abs(target_col - origin_col)
            frontier_candidates.append((distance_from_origin, steps, action_ref, target_index))
        if frontier_candidates:
            _distance, _steps, action_ref, target = max(
                frontier_candidates,
                key=lambda item: (item[0], -item[1], item[2]),
            )
            target_type_id = current.cell_type_ids[target]
            exploratory_frontier = (
                RetainedGridNeighborWitness(
                    action_ref=action_ref,
                    destination_cell_index=target,
                    destination_type_id=target_type_id,
                    destination_pattern_occurrence_count=current.type_occurrences[target_type_id],
                    destination_previously_visited=target in visited,
                ),
            )
    return RetainedGridMotionFrontier(
        before_cell_index=before_index,
        after_cell_index=after_index,
        motion_action_refs=motion_action_refs,
        transported_pattern_changed=transported_pattern_changed,
        continuation=tuple(grouped["continuation"][:1]),
        orthogonal=tuple(grouped["orthogonal"][:2]),
        reverse=tuple(grouped["reverse"][:1]),
        exploratory_frontier=exploratory_frontier,
    )


def measure_retained_grid_initial_exploratory_frontier(
    current: KnownGridCellRecountResult,
    *,
    logical_rows: int,
    logical_columns: int,
    interface_action_translation_deltas: tuple[tuple[str, int, int], ...],
    frontier_origin_cell_index: int | None,
) -> tuple[RetainedGridNeighborWitness, ...]:
    """Measure one bounded opening that continues beyond the initial cell.

    This is a morphology-only extension of the initial cursor cell.  A ray is
    useful only when it crosses a locally uniform corridor (or singleton
    transition cells), reaches a different repeated pattern, and still has a
    further cell in the same direction.  A repeated boundary with no such
    continuation is therefore not an exploratory frontier.  No palette value
    or semantic role is assigned here; DRM decides whether the witness should
    precede a point probe.
    """

    if (
        frontier_origin_cell_index is None
        or not 0 <= int(frontier_origin_cell_index) < logical_rows * logical_columns
        or len(current.cell_type_ids) != logical_rows * logical_columns
    ):
        return ()
    origin_row, origin_col = divmod(int(frontier_origin_cell_index), logical_columns)
    normalized_actions = tuple(
        (
            str(action_ref),
            0 if int(delta_row) == 0 else (1 if int(delta_row) > 0 else -1),
            0 if int(delta_col) == 0 else (1 if int(delta_col) > 0 else -1),
        )
        for action_ref, delta_row, delta_col in interface_action_translation_deltas
        if (int(delta_row) == 0) ^ (int(delta_col) == 0)
    )
    candidates: list[tuple[int, int, str, int]] = []
    for action_ref, delta_row, delta_col in sorted(normalized_actions):
        row, col = origin_row + delta_row, origin_col + delta_col
        if not (0 <= row < logical_rows and 0 <= col < logical_columns):
            continue
        first_type_id = current.cell_type_ids[row * logical_columns + col]
        corridor_pattern = current.type_patterns[first_type_id]
        steps = 1
        last_reachable_index = row * logical_columns + col
        while True:
            row += delta_row
            col += delta_col
            if not (0 <= row < logical_rows and 0 <= col < logical_columns):
                break
            cell_index = row * logical_columns + col
            type_id = current.cell_type_ids[cell_index]
            pattern = current.type_patterns[type_id]
            occurrence_count = current.type_occurrences[type_id]
            if pattern == corridor_pattern or occurrence_count == 1:
                steps += 1
                last_reachable_index = cell_index
                continue
            if occurrence_count <= 1:
                break
            if len({value for row_values in pattern for value in row_values}) > len(
                {value for row_values in corridor_pattern for value in row_values}
            ):
                # A more internally varied repeated pattern is a bounded
                # obstacle/marker candidate, not evidence of an open space.
                break
            # A repeated pattern with no greater internal complexity is an
            # accessible continuation.  Continue the ray through it and keep
            # its last cell as the spatial frontier; a later more varied
            # pattern or frame boundary closes the route.
            corridor_pattern = pattern
            steps += 1
            last_reachable_index = cell_index
        if steps >= 2:
            candidates.append(
                (
                    steps,
                    abs((last_reachable_index // logical_columns) - origin_row)
                    + abs((last_reachable_index % logical_columns) - origin_col),
                    action_ref,
                    last_reachable_index,
                )
            )
    if not candidates:
        return ()
    _steps, _distance, action_ref, destination_index = max(
        candidates,
        key=lambda item: (item[1], -item[0], item[2]),
    )
    destination_type_id = current.cell_type_ids[destination_index]
    return (
        RetainedGridNeighborWitness(
            action_ref=action_ref,
            destination_cell_index=destination_index,
            destination_type_id=destination_type_id,
            destination_pattern_occurrence_count=(
                current.type_occurrences[destination_type_id]
            ),
            destination_previously_visited=False,
        ),
    )


def cell_containing_footprint(
    grid: PeriodicCellGridCandidate,
    footprint: BoundingBox,
) -> int | None:
    """Return the cell containing the whole bbox; reject clipped footprints."""

    row = (footprint.top - grid.row_offset) // grid.row_pitch
    col = (footprint.left - grid.col_offset) // grid.col_pitch
    if not (0 <= row < grid.logical_rows and 0 <= col < grid.logical_columns):
        return None
    top = grid.row_offset + row * grid.row_pitch
    left = grid.col_offset + col * grid.col_pitch
    if (
        footprint.top < top
        or footprint.bottom >= top + grid.cell_height
        or footprint.left < left
        or footprint.right >= left + grid.cell_width
    ):
        return None
    return row * grid.logical_columns + col


def measure_grid_footprint_transport(
    grid: PeriodicCellGridCandidate,
    footprints: tuple[tuple[BoundingBox, BoundingBox], ...],
) -> GridFootprintTransport:
    """Measure whether all supplied footprints share one moving grid cell."""

    before_cells = tuple(
        cell_containing_footprint(grid, before) for before, _ in footprints
    )
    after_cells = tuple(
        cell_containing_footprint(grid, after) for _, after in footprints
    )
    lattice_displacement = bool(footprints) and all(
        before.height == after.height
        and before.width == after.width
        and (after.top - before.top) % grid.row_pitch == 0
        and (after.left - before.left) % grid.col_pitch == 0
        for before, after in footprints
    )
    return GridFootprintTransport(
        before_cells=before_cells,
        after_cells=after_cells,
        same_moving_cell=(
            bool(before_cells)
            and None not in before_cells
            and None not in after_cells
            and len(set(before_cells)) == 1
            and len(set(after_cells)) == 1
            and before_cells[0] != after_cells[0]
        ),
        lattice_displacement=lattice_displacement,
    )


def project_retained_grid_cell_measurements(
    prior_grid: PeriodicCellGridCandidate,
    current: KnownGridCellRecountResult,
) -> RetainedGridCellMeasurements:
    """Refresh cells and pattern counts, never stale global grid metrics."""

    return RetainedGridCellMeasurements(
        cells=project_retained_grid_cell_descriptions(prior_grid, current),
        repeated_pattern_count=sum(
            occurrence > 1 for occurrence in current.type_occurrences
        ),
        distinct_pattern_count=len(current.type_patterns),
    )


def _internal_separator_values(value: KnownGridCellRecountInput) -> tuple[int, ...]:
    """Read exact internal gaps; outside margins may change independently."""

    row_gap = value.row_pitch - value.cell_height
    col_gap = value.col_pitch - value.cell_width
    if row_gap <= 0 and col_gap <= 0:
        return ()
    bottom = value.row_offset + (value.logical_rows - 1) * value.row_pitch + value.cell_height
    right = value.col_offset + (value.logical_columns - 1) * value.col_pitch + value.cell_width
    separator_pixels: list[int] = []
    for row in range(value.row_offset, bottom):
        source_row = value.frame.rows[row]
        if (row - value.row_offset) % value.row_pitch >= value.cell_height:
            separator_pixels.extend(source_row[value.col_offset:right])
        else:
            for logical_col in range(value.logical_columns - 1):
                left = value.col_offset + logical_col * value.col_pitch + value.cell_width
                separator_pixels.extend(source_row[left:left + col_gap])
    return tuple(separator_pixels)


def recount_known_grid_cells(
    value: KnownGridCellRecountInput,
) -> KnownGridCellRecountResult:
    """Measure types and deltas without rediscovering or selecting a grid."""

    patterns: list[tuple[tuple[int, ...], ...]] = []
    type_ids: dict[tuple[tuple[int, ...], ...], int] = {}
    cell_type_ids: list[int] = []
    occurrences: list[int] = []
    changed: list[int] = []
    previous = value.prior_cell_type_ids
    for logical_row in range(value.logical_rows):
        top = value.row_offset + logical_row * value.row_pitch
        for logical_col in range(value.logical_columns):
            left = value.col_offset + logical_col * value.col_pitch
            pattern = tuple(
                tuple(row[left : left + value.cell_width])
                for row in value.frame.rows[top : top + value.cell_height]
            )
            type_id = type_ids.get(pattern)
            if type_id is None:
                type_id = len(patterns)
                type_ids[pattern] = type_id
                patterns.append(pattern)
                occurrences.append(0)
            cell_type_ids.append(type_id)
            occurrences[type_id] += 1
            cell_index = len(cell_type_ids) - 1
            if (
                not previous
                or pattern
                != value.prior_type_patterns[previous[cell_index]]
            ):
                changed.append(cell_index)
    separator_values = _internal_separator_values(value)
    unique_multivalue_type_ids = {
        type_id
        for type_id, pattern in enumerate(patterns)
        if occurrences[type_id] == 1
        and len({pixel for row in pattern for pixel in row}) > 1
    }
    return KnownGridCellRecountResult(
        grid_ref=value.grid_ref,
        type_patterns=tuple(patterns),
        cell_type_ids=tuple(cell_type_ids),
        type_occurrences=tuple(occurrences),
        changed_cell_indexes=tuple(changed),
        separator_values=separator_values,
        separator_values_changed=(
            bool(value.prior_separator_values)
            and value.prior_separator_values != separator_values
        ),
        unique_multivalue_cell_indexes=tuple(
            index
            for index, type_id in enumerate(cell_type_ids)
            if type_id in unique_multivalue_type_ids
        ),
    )


def measure_retained_grid_new_singletons(
    prior: KnownGridCellRecountResult,
    current: KnownGridCellRecountResult,
    *,
    logical_columns: int,
    known_mover_pattern_refs: tuple[str, ...],
    executed_axis_delta: tuple[int, int] | None,
) -> dict[str, object]:
    """Measure unique raster continuations and remaining new cell motifs.

    Pixel multisets are orientation-invariant, but a match is retained only
    when one known mover and one new motif form a unique action-consistent pair.
    No semantic role is assigned here.
    """

    if (
        prior.grid_ref != current.grid_ref
        or len(prior.cell_type_ids) != len(current.cell_type_ids)
        or logical_columns < 2
        or executed_axis_delta is None
    ):
        return {"retained_grid_novel_unique_pattern_count": 0}

    def pattern(recount: KnownGridCellRecountResult, index: int):
        return recount.type_patterns[recount.cell_type_ids[index]]

    def pixels(recount: KnownGridCellRecountResult, index: int) -> tuple[int, ...]:
        return tuple(sorted(pixel for row in pattern(recount, index) for pixel in row))

    prior_pattern_refs = {
        _cell_pattern_ref(item) for item in prior.type_patterns
    }
    known = tuple(
        index for index in prior.unique_multivalue_cell_indexes
        if _cell_pattern_ref(pattern(prior, index)) in known_mover_pattern_refs
    )
    novel = tuple(
        index for index in current.unique_multivalue_cell_indexes
        if _cell_pattern_ref(pattern(current, index)) not in prior_pattern_refs
    )
    axis_row, axis_col = executed_axis_delta
    continuation_pairs = tuple(
        (before_index, after_index)
        for before_index in known
        for after_index in novel
        for before_row, before_col in (divmod(before_index, logical_columns),)
        for after_row, after_col in (divmod(after_index, logical_columns),)
        if pixels(prior, before_index) == pixels(current, after_index)
        and (
            (axis_row == 0 and axis_col != 0 and before_row == after_row
             and 0 <= (after_col - before_col) * axis_col < logical_columns)
            or
            (axis_col == 0 and axis_row != 0 and before_col == after_col
             and 0 <= (after_row - before_row) * axis_row
             < len(prior.cell_type_ids) // logical_columns)
        )
    )
    facts: dict[str, object] = {}
    continued_index: int | None = None
    if len(continuation_pairs) == 1:
        before_index, continued_index = continuation_pairs[0]
        facts["prior_reoriented_cell_pattern_ref"] = _cell_pattern_ref(
            pattern(prior, before_index)
        )
        facts["unique_reoriented_cell_pattern_ref"] = _cell_pattern_ref(
            pattern(current, continued_index)
        )
        # A pure local translation with an appearance change still exposes
        # the substrate vacated by this same measured bearer. A changed
        # background elsewhere requires general transition reconciliation.
        changed_indexes = {
            index for index in range(len(prior.cell_type_ids))
            if pattern(prior, index) != pattern(current, index)
        }
        if before_index != continued_index and changed_indexes == {before_index, continued_index}:
            facts["reoriented_moving_pattern_ref"] = facts["unique_reoriented_cell_pattern_ref"]
            facts["reoriented_vacated_pattern_ref"] = _cell_pattern_ref(pattern(current, before_index))
    other_novel = tuple(index for index in novel if index != continued_index)
    facts["retained_grid_novel_unique_pattern_count"] = len(other_novel)
    if len(other_novel) == 1 and continued_index is not None:
        index = other_novel[0]
        facts["unique_new_cell_pattern_ref"] = _cell_pattern_ref(pattern(current, index))
        facts["unique_new_cell_row"], facts["unique_new_cell_col"] = divmod(
            index, logical_columns
        )
    return facts


def delta_between_known_grid_recounts(
    prior: KnownGridCellRecountResult,
    current: KnownGridCellRecountResult,
) -> KnownGridCellRecountDelta:
    """Reduce one exact recount to only changed immutable measurements."""

    if prior.grid_ref != current.grid_ref:
        raise ValueError("grid identity changed across recount delta")
    if len(prior.cell_type_ids) != len(current.cell_type_ids):
        raise ValueError("grid cell count changed across recount delta")
    if len(prior.separator_values) != len(current.separator_values):
        raise ValueError("grid separator count changed across recount delta")
    return KnownGridCellRecountDelta(
        grid_ref=current.grid_ref,
        cell_count=len(current.cell_type_ids),
        separator_count=len(current.separator_values),
        changed_cells=tuple(
            (index, current.type_patterns[current_type_id])
            for index, current_type_id in enumerate(current.cell_type_ids)
            if prior.type_patterns[prior.cell_type_ids[index]]
            != current.type_patterns[current_type_id]
        ),
        changed_separators=tuple(
            (index, pixel)
            for index, pixel in enumerate(current.separator_values)
            if pixel != prior.separator_values[index]
        ),
    )


def apply_known_grid_recount_delta(
    prior: KnownGridCellRecountResult,
    delta: KnownGridCellRecountDelta,
) -> KnownGridCellRecountResult:
    """Reconstruct the exact current result without resending unchanged cells."""

    if prior.grid_ref != delta.grid_ref:
        raise ValueError("grid identity differs from recount delta")
    if len(prior.cell_type_ids) != delta.cell_count:
        raise ValueError("grid cell count differs from recount delta")
    if len(prior.separator_values) != delta.separator_count:
        raise ValueError("grid separator count differs from recount delta")
    patterns_by_cell = [
        prior.type_patterns[type_id] for type_id in prior.cell_type_ids
    ]
    separators = list(prior.separator_values)
    changed_cell_indexes: list[int] = []
    last_index = -1
    for index, pattern in delta.changed_cells:
        if index <= last_index or index >= delta.cell_count:
            raise ValueError("changed cell indexes must be unique, ordered and in bounds")
        if patterns_by_cell[index] == pattern:
            raise ValueError("recount delta includes an unchanged cell")
        patterns_by_cell[index] = pattern
        changed_cell_indexes.append(index)
        last_index = index
    last_index = -1
    for index, pixel in delta.changed_separators:
        if index <= last_index or index >= delta.separator_count:
            raise ValueError("changed separator indexes must be unique, ordered and in bounds")
        if separators[index] == pixel:
            raise ValueError("recount delta includes an unchanged separator")
        separators[index] = pixel
        last_index = index
    type_ids: dict[tuple[tuple[int, ...], ...], int] = {}
    type_occurrences: list[int] = []
    cell_type_ids: list[int] = []
    for pattern in patterns_by_cell:
        type_id = type_ids.get(pattern)
        if type_id is None:
            type_id = len(type_ids)
            type_ids[pattern] = type_id
            type_occurrences.append(0)
        cell_type_ids.append(type_id)
        type_occurrences[type_id] += 1
    unique_multivalue_type_ids = {
        type_id
        for pattern, type_id in sorted(type_ids.items())
        if type_occurrences[type_id] == 1
        and len({pixel for row in pattern for pixel in row}) > 1
    }
    return KnownGridCellRecountResult(
        grid_ref=delta.grid_ref,
        type_patterns=tuple(type_ids),
        cell_type_ids=tuple(cell_type_ids),
        type_occurrences=tuple(type_occurrences),
        changed_cell_indexes=tuple(changed_cell_indexes),
        separator_values=tuple(separators),
        separator_values_changed=bool(delta.changed_separators),
        unique_multivalue_cell_indexes=tuple(
            index for index, type_id in enumerate(cell_type_ids)
            if type_id in unique_multivalue_type_ids
        ),
    )


def project_retained_grid_cell_descriptions(
    prior_grid: PeriodicCellGridCandidate,
    current: KnownGridCellRecountResult,
) -> tuple[PeriodicCellDescription, ...]:
    """Refresh exact cells only; leave grid-wide metrics to their own demand."""

    if prior_grid.candidate_ref != current.grid_ref:
        raise ValueError("retained grid identity differs from cell recount")
    if len(prior_grid.cells) != len(current.cell_type_ids):
        raise ValueError("retained grid cell count differs from cell recount")
    refreshed: list[PeriodicCellDescription] = []
    for index, old_cell in enumerate(prior_grid.cells):
        type_id = current.cell_type_ids[index]
        pattern = current.type_patterns[type_id]
        pattern_ref = _cell_pattern_ref(pattern)
        occurrence_count = current.type_occurrences[type_id]
        if (
            old_cell.pattern_ref == pattern_ref
            and old_cell.pattern_occurrence_count == occurrence_count
        ):
            refreshed.append(old_cell)
            continue
        palette_values = tuple(sorted({pixel for row in pattern for pixel in row}))
        refreshed.append(
            old_cell.model_copy(
                update={
                    "pattern_ref": pattern_ref,
                    "palette_value_count": len(palette_values),
                    "palette_values": palette_values,
                    "pattern_occurrence_count": occurrence_count,
                }
            )
        )
    return tuple(refreshed)
