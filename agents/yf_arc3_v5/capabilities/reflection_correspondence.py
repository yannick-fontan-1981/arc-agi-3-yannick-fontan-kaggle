"""Exact finite cell-reflection measurements without semantic selection.

The measurement enumerates row/column reflections, their joint closure and a
half-turn.  DRM decides whether an action-conditioned description means that a
distinctive value is repairable; this module only reports exact finite deltas.
"""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json

from agents.yf_arc3_v5.capabilities.components import detect_components
from agents.yf_arc3_v5.capabilities.contracts import (
    ComponentExtractionInput,
    FrameGrid,
)
from agents.yf_arc3_v5.capabilities.periodic_correspondence import (
    UniformLattice,
    measure_uniform_lattices,
)


MAX_REFLECTION_DESCRIPTIONS = 64
TRANSFORM_REFS = (
    "measure.reflect_rows",
    "measure.reflect_columns",
    "measure.reflect_rows_and_columns",
    "measure.rotate_half_turn",
)


class ReflectionMeasurementBoundExceeded(ValueError):
    pass


@dataclass(frozen=True)
class ReflectionDescription:
    transform_ref: str
    omitted_value: int
    projected_values: tuple[tuple[int, ...], ...]
    concordant_relation_count: int
    exception_relation_count: int
    count_difference: int


@dataclass(frozen=True)
class ReflectionDelta:
    field: UniformLattice
    projected_values: tuple[tuple[int, ...], ...] | None
    after_frame_hash: str
    point: tuple[int, int]
    before_value: int
    after_value: int
    other_cell_changes: int
    residual_delta: int
    exact_description_count: int
    distinct_projection_count: int
    minimum_concordance: int
    minimum_count_difference: int
    transform_refs: tuple[str, ...]
    boundary_rectangular_decreasing_value_count: int
    boundary_rectangular_quantity_decrease_magnitude: int
    boundary_rectangular_decreasing_values: tuple[int, ...]


def _digest(values: object) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("ascii")).hexdigest()


def _orbit(
    height: int,
    width: int,
    row: int,
    column: int,
    transform_ref: str,
) -> frozenset[tuple[int, int]]:
    if transform_ref == "measure.reflect_rows":
        return frozenset(((row, column), (height - 1 - row, column)))
    if transform_ref == "measure.reflect_columns":
        return frozenset(((row, column), (row, width - 1 - column)))
    if transform_ref == "measure.rotate_half_turn":
        return frozenset(
            ((row, column), (height - 1 - row, width - 1 - column))
        )
    if transform_ref == "measure.reflect_rows_and_columns":
        return frozenset(
            (
                (row, column),
                (height - 1 - row, column),
                (row, width - 1 - column),
                (height - 1 - row, width - 1 - column),
            )
        )
    raise ValueError(f"unsupported finite transform: {transform_ref}")


@lru_cache(maxsize=16)
def measure_reflection_descriptions(
    field: UniformLattice,
) -> tuple[ReflectionDescription, ...]:
    height = len(field.rows)
    width = len(field.columns)
    values = tuple(sorted({value for row in field.values for value in row}))
    descriptions: list[ReflectionDescription] = []
    attempts = 0
    for transform_ref in TRANSFORM_REFS:
        for omitted_value in values:
            attempts += 1
            if attempts > MAX_REFLECTION_DESCRIPTIONS:
                raise ReflectionMeasurementBoundExceeded(
                    "reflection description bound"
                )
            projected = [list(row) for row in field.values]
            visited: set[frozenset[tuple[int, int]]] = set()
            concordant = 0
            exceptions = 0
            complete = True
            for row in range(height):
                for column in range(width):
                    orbit = _orbit(
                        height, width, row, column, transform_ref
                    )
                    if orbit in visited:
                        continue
                    visited.add(orbit)
                    if len(orbit) == 1:
                        only_row, only_column = next(iter(orbit))
                        if field.values[only_row][only_column] == omitted_value:
                            complete = False
                            break
                        continue
                    retained_values = tuple(
                        field.values[other_row][other_column]
                        for other_row, other_column in sorted(orbit)
                        if field.values[other_row][other_column] != omitted_value
                    )
                    omitted_count = sum(
                        field.values[other_row][other_column] == omitted_value
                        for other_row, other_column in orbit
                    )
                    if not retained_values or len(set(retained_values)) != 1:
                        complete = False
                        break
                    measured_value = retained_values[0]
                    for other_row, other_column in orbit:
                        projected[other_row][other_column] = measured_value
                    concordant += len(retained_values) - 1
                    exceptions += omitted_count
                if not complete:
                    break
            if complete and exceptions > 0:
                descriptions.append(
                    ReflectionDescription(
                        transform_ref=transform_ref,
                        omitted_value=omitted_value,
                        projected_values=tuple(tuple(row) for row in projected),
                        concordant_relation_count=concordant,
                        exception_relation_count=exceptions,
                        count_difference=concordant - exceptions,
                    )
                )
    return tuple(descriptions)


def _logical_index(
    field: UniformLattice, point: tuple[int, int]
) -> tuple[int, int] | None:
    matches = tuple(
        (row, column)
        for row in range(len(field.rows))
        for column in range(len(field.columns))
        if field.point(row, column) == point
    )
    return matches[0] if len(matches) == 1 else None


def _outside_rectangular_quantities(
    frame: FrameGrid, field: UniformLattice
) -> Counter[int]:
    """Count compact filled rectangles outside one lattice, by observed value."""

    quantities: Counter[int] = Counter()
    for component in detect_components(
        ComponentExtractionInput(frame=frame)
    ).components:
        box = component.bbox
        if (
            not field.contains(box)
            and component.area == box.height * box.width
            and min(box.height, box.width) <= field.extent
        ):
            quantities[component.value] += component.area
    return quantities


def measure_reflection_delta(
    before: FrameGrid,
    after: FrameGrid,
    point: tuple[int, int] | None,
    prior: ReflectionDelta | None = None,
) -> ReflectionDelta | None:
    if point is None:
        return None
    # An unchanged raster and a disjoint point do not replace an existing
    # field measurement. Return the original immutable witness, including its
    # original point/hash, rather than attribute the field change to this probe.
    if (
        prior is not None
        and before.rows == after.rows
        and prior.after_frame_hash == _digest(after.rows)
        and not (
            prior.field.rows[0] <= point[0] < prior.field.rows[-1] + prior.field.extent
            and prior.field.columns[0] <= point[1] < prior.field.columns[-1] + prior.field.extent
        )
    ):
        return prior
    before_fields = measure_uniform_lattices(before)
    after_fields = measure_uniform_lattices(after)
    field_pairs = tuple(
        (before_field, after_field)
        for before_field in before_fields
        for after_field in after_fields
        if (
            before_field.rows,
            before_field.columns,
            before_field.extent,
        )
        == (
            after_field.rows,
            after_field.columns,
            after_field.extent,
        )
        and _logical_index(before_field, point) is not None
    )
    if len(field_pairs) != 1:
        return None
    before_field, after_field = field_pairs[0]
    logical = _logical_index(before_field, point)
    if logical is None:
        return None
    row, column = logical
    before_value = before_field.values[row][column]
    after_value = after_field.values[row][column]
    descriptions = tuple(
        description
        for description in measure_reflection_descriptions(after_field)
        if description.omitted_value == before_value
        and before_value != after_value
        and description.projected_values[row][column] == after_value
    )
    projections = {
        description.projected_values for description in descriptions
    }
    unique_projection = next(iter(projections)) if len(projections) == 1 else None
    changes = tuple(
        (other_row, other_column)
        for other_row in range(len(before_field.rows))
        for other_column in range(len(before_field.columns))
        if before_field.values[other_row][other_column]
        != after_field.values[other_row][other_column]
    )
    if unique_projection is None:
        residual_delta = 0
    else:
        before_residual = sum(
            before_field.values[other_row][other_column]
            != unique_projection[other_row][other_column]
            for other_row in range(len(before_field.rows))
            for other_column in range(len(before_field.columns))
        )
        after_residual = sum(
            after_field.values[other_row][other_column]
            != unique_projection[other_row][other_column]
            for other_row in range(len(after_field.rows))
            for other_column in range(len(after_field.columns))
        )
        residual_delta = after_residual - before_residual
    before_boundary = _outside_rectangular_quantities(before, before_field)
    after_boundary = _outside_rectangular_quantities(after, after_field)
    boundary_decreases = {
        value: before_boundary[value] - after_boundary[value]
        for value in sorted(before_boundary)
        if before_boundary[value] > after_boundary[value]
    }
    return ReflectionDelta(
        field=after_field,
        projected_values=unique_projection,
        after_frame_hash=_digest(after.rows),
        point=point,
        before_value=before_value,
        after_value=after_value,
        other_cell_changes=sum(change != logical for change in changes),
        residual_delta=residual_delta,
        exact_description_count=len(descriptions),
        distinct_projection_count=len(projections),
        minimum_concordance=min(
            (
                description.concordant_relation_count
                for description in descriptions
            ),
            default=0,
        ),
        minimum_count_difference=min(
            (description.count_difference for description in descriptions),
            default=0,
        ),
        transform_refs=tuple(
            sorted({description.transform_ref for description in descriptions})
        ),
        boundary_rectangular_decreasing_value_count=len(boundary_decreases),
        boundary_rectangular_quantity_decrease_magnitude=sum(
            boundary_decreases.values()
        ),
        boundary_rectangular_decreasing_values=tuple(boundary_decreases),
    )


def measure_reflection_context(
    frame: FrameGrid,
    fields: tuple[UniformLattice, ...],
    prior: ReflectionDelta | None,
) -> dict[str, object]:
    current_prior = bool(
        prior is not None and prior.after_frame_hash == _digest(frame.rows)
    )
    return {
        "reflection_field_count": len(fields),
        "reflection_inventory_complete": True,
        "reflection_transition_present": current_prior,
        "reflection_exact_description_count": (
            prior.exact_description_count if current_prior and prior else 0
        ),
        "reflection_distinct_projection_count": (
            prior.distinct_projection_count if current_prior and prior else 0
        ),
        "reflection_minimum_concordance": (
            prior.minimum_concordance if current_prior and prior else 0
        ),
        "reflection_minimum_count_difference": (
            prior.minimum_count_difference if current_prior and prior else 0
        ),
        "reflection_transform_measurement_refs": (
            prior.transform_refs if current_prior and prior else ()
        ),
        "reflection_prior_point_reached_projection": bool(
            current_prior
            and prior
            and prior.projected_values is not None
            and prior.after_value
            == prior.projected_values[
                _logical_index(prior.field, prior.point)[0]
            ][_logical_index(prior.field, prior.point)[1]]
        ),
        "reflection_prior_other_cell_changes": (
            prior.other_cell_changes if current_prior and prior else 0
        ),
        "reflection_prior_residual_delta": (
            prior.residual_delta if current_prior and prior else 0
        ),
        "reflection_boundary_rectangular_decreasing_value_count": (
            prior.boundary_rectangular_decreasing_value_count
            if current_prior and prior
            else 0
        ),
        "reflection_boundary_rectangular_quantity_decrease_magnitude": (
            prior.boundary_rectangular_quantity_decrease_magnitude
            if current_prior and prior
            else 0
        ),
        "reflection_boundary_rectangular_decreasing_values": (
            prior.boundary_rectangular_decreasing_values
            if current_prior and prior
            else ()
        ),
    }


def measure_reflection_candidate(
    fields: tuple[UniformLattice, ...],
    point: tuple[int, int] | None,
    prior: ReflectionDelta | None,
) -> dict[str, object]:
    facts: dict[str, object] = {
        "reflection_point_in_field": False,
        "reflection_field_distinct_value_count": 0,
        "reflection_cell_value_occurrence_count": 0,
        "reflection_field_minimum_value_occurrence_count": 0,
        "reflection_field_minimum_value_kind_count": 0,
        "reflection_cell_has_unique_strict_minimum_value_frequency": False,
        "reflection_point_matches_prior_distinct_value": False,
        "reflection_point_differs_from_projection": False,
        "reflection_point_projected_value": -1,
    }
    if point is None:
        return facts
    matching = tuple(
        (field, logical)
        for field in fields
        if (logical := _logical_index(field, point)) is not None
    )
    if len(matching) != 1:
        return facts
    field, (row, column) = matching[0]
    counts = Counter(value for values in field.values for value in values)
    occurrence_counts = tuple(counts[value] for value in sorted(counts))
    minimum = min(occurrence_counts, default=0)
    minimum_kind_count = sum(
        count == minimum for count in occurrence_counts
    )
    current_value = field.values[row][column]
    facts.update(
        {
            "reflection_point_in_field": True,
            "reflection_field_distinct_value_count": len(counts),
            "reflection_cell_value_occurrence_count": counts[current_value],
            "reflection_field_minimum_value_occurrence_count": minimum,
            "reflection_field_minimum_value_kind_count": minimum_kind_count,
            "reflection_cell_has_unique_strict_minimum_value_frequency": bool(
                len(counts) > 1
                and minimum_kind_count == 1
                and counts[current_value] == minimum
            ),
        }
    )
    if (
        prior is None
        or prior.projected_values is None
    ):
        return facts
    mapped_value = prior.projected_values[row][column]
    facts.update(
        {
            "reflection_point_matches_prior_distinct_value": (
                current_value == prior.before_value
            ),
            "reflection_point_differs_from_projection": (
                current_value != mapped_value
            ),
            "reflection_point_projected_value": mapped_value,
        }
    )
    return facts
