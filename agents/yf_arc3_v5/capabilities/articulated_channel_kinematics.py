"""Exact bounded measurements for articulated local frames and joint deltas."""

from __future__ import annotations

from fractions import Fraction

from agents.yf_arc3_v5.capabilities.contracts import (
    ArticulatedBodyGeometry,
    ArticulatedBodyState,
    ArticulatedKinematicsInput,
    ArticulatedKinematicsMeasurements,
    ArticulatedLocalOperation,
    ArticulatedSharedSnapshotInput,
    ArticulatedSharedSnapshotMeasurements,
    ArticulatedSuccessorInput,
    ArticulatedSuccessorMeasurements,
    Rational,
    RationalMatrix2,
    RationalPoint,
)


FractionPoint = tuple[Fraction, Fraction]
FractionMatrix2 = tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]


def _fraction(value: Rational) -> Fraction:
    return Fraction(value[0], value[1])


def _rational(value: Fraction) -> Rational:
    return (value.numerator, value.denominator)


def _point(value: RationalPoint) -> FractionPoint:
    return (_fraction(value[0]), _fraction(value[1]))


def _matrix(value: RationalMatrix2) -> FractionMatrix2:
    return (
        (_fraction(value[0][0]), _fraction(value[0][1])),
        (_fraction(value[1][0]), _fraction(value[1][1])),
    )


def _rational_point(value: FractionPoint) -> RationalPoint:
    return (_rational(value[0]), _rational(value[1]))


def _rational_matrix(value: FractionMatrix2) -> RationalMatrix2:
    return (
        (_rational(value[0][0]), _rational(value[0][1])),
        (_rational(value[1][0]), _rational(value[1][1])),
    )


def _matrix_product(left: FractionMatrix2, right: FractionMatrix2) -> FractionMatrix2:
    return (
        (
            left[0][0] * right[0][0] + left[0][1] * right[1][0],
            left[0][0] * right[0][1] + left[0][1] * right[1][1],
        ),
        (
            left[1][0] * right[0][0] + left[1][1] * right[1][0],
            left[1][0] * right[0][1] + left[1][1] * right[1][1],
        ),
    )


def _matrix_point(matrix: FractionMatrix2, point: FractionPoint) -> FractionPoint:
    return (
        matrix[0][0] * point[0] + matrix[0][1] * point[1],
        matrix[1][0] * point[0] + matrix[1][1] * point[1],
    )


def _point_add(left: FractionPoint, right: FractionPoint) -> FractionPoint:
    return (left[0] + right[0], left[1] + right[1])


def _compose(
    outer: tuple[FractionMatrix2, FractionPoint],
    inner: tuple[FractionMatrix2, FractionPoint],
) -> tuple[FractionMatrix2, FractionPoint]:
    outer_matrix, outer_translation = outer
    inner_matrix, inner_translation = inner
    return (
        _matrix_product(outer_matrix, inner_matrix),
        _point_add(_matrix_point(outer_matrix, inner_translation), outer_translation),
    )


def _world_frames(
    bodies: tuple[ArticulatedBodyState, ...],
) -> dict[str, tuple[FractionMatrix2, FractionPoint]]:
    by_ref = {body.body_ref: body for body in bodies}
    frames: dict[str, tuple[FractionMatrix2, FractionPoint]] = {}

    def resolve(body_ref: str) -> tuple[FractionMatrix2, FractionPoint]:
        cached = frames.get(body_ref)
        if cached is not None:
            return cached
        body = by_ref[body_ref]
        local = (_matrix(body.local_linear), _point(body.local_translation))
        frame = local if body.parent_ref is None else _compose(resolve(body.parent_ref), local)
        frames[body_ref] = frame
        return frame

    for body in bodies:
        resolve(body.body_ref)
    return frames


def measure_articulated_kinematics(
    value: ArticulatedKinematicsInput,
) -> ArticulatedKinematicsMeasurements:
    """Compose exact local frames and expose roots, free ends and markers."""

    frames = _world_frames(value.bodies)
    geometries: list[ArticulatedBodyGeometry] = []
    for body in value.bodies:
        linear, translation = frames[body.body_ref]
        world_axis = _matrix_point(linear, _point(body.local_axis))
        extent = _fraction(body.local_extent)
        free_end = _point_add(
            translation, (world_axis[0] * extent, world_axis[1] * extent)
        )
        markers = tuple(
            (
                marker_ref,
                _rational_point(
                    _point_add(translation, _matrix_point(linear, _point(offset)))
                ),
            )
            for marker_ref, offset in body.marker_offsets
        )
        geometries.append(
            ArticulatedBodyGeometry(
                body_ref=body.body_ref,
                root_position=_rational_point(translation),
                free_end_position=_rational_point(free_end),
                world_axis=_rational_point(world_axis),
                local_extent=body.local_extent,
                marker_positions=markers,
            )
        )
    return ArticulatedKinematicsMeasurements(
        body_geometries=tuple(geometries), body_count=len(geometries)
    )


def _apply_local_operation(
    body: ArticulatedBodyState, operation: ArticulatedLocalOperation
) -> ArticulatedBodyState:
    current = (_matrix(body.local_linear), _point(body.local_translation))
    delta = (_matrix(operation.linear_delta), _point(operation.translation_delta))
    updated = (
        _compose(delta, current)
        if operation.composition_side == "before_local"
        else _compose(current, delta)
    )
    extent = _fraction(body.local_extent) + _fraction(operation.extent_delta)
    return body.model_copy(
        update={
            "local_linear": _rational_matrix(updated[0]),
            "local_translation": _rational_point(updated[1]),
            "local_extent": _rational(extent),
        }
    )


def measure_articulated_shared_snapshot(
    value: ArticulatedSharedSnapshotInput,
) -> ArticulatedSharedSnapshotMeasurements:
    """Compose distinct supplied causes once, without selecting their scope or order."""

    unique: list[ArticulatedLocalOperation] = []
    duplicate_refs: list[str] = []
    seen: set[tuple[str, str]] = set()
    for operation in value.operations:
        key = (operation.cause_ref, operation.receiver_ref)
        if key in seen:
            duplicate_refs.append(operation.operation_ref)
            continue
        seen.add(key)
        unique.append(operation)

    if len(unique) > 1 and any(item.order_index is None for item in unique):
        return ArticulatedSharedSnapshotMeasurements(
            status="ambiguous_order",
            resulting_bodies=value.bodies,
            deduplicated_operation_refs=tuple(duplicate_refs),
            unresolved_order_refs=tuple(
                item.operation_ref for item in unique if item.order_index is None
            ),
            applied_operation_count=0,
        )

    ordered = sorted(
        unique,
        key=lambda item: (
            item.order_index if item.order_index is not None else 0,
            item.operation_ref,
        ),
    )
    body_by_ref = {body.body_ref: body for body in value.bodies}
    for operation in ordered:
        body_by_ref[operation.receiver_ref] = _apply_local_operation(
            body_by_ref[operation.receiver_ref], operation
        )
    resulting = tuple(body_by_ref[item.body_ref] for item in value.bodies)
    return ArticulatedSharedSnapshotMeasurements(
        status="applied",
        resulting_bodies=resulting,
        deduplicated_operation_refs=tuple(duplicate_refs),
        unresolved_order_refs=(),
        applied_operation_count=len(ordered),
    )


def measure_articulated_successor(
    value: ArticulatedSuccessorInput,
) -> ArticulatedSuccessorMeasurements:
    """Return one complete joint delta for the source-declared transaction result."""

    resulting = (
        value.before_bodies
        if value.committed_packet == "before"
        else value.supplied_after_bodies
    )
    before_geometry = measure_articulated_kinematics(
        ArticulatedKinematicsInput(bodies=value.before_bodies)
    )
    after_geometry = measure_articulated_kinematics(
        ArticulatedKinematicsInput(bodies=resulting)
    )
    before_by_ref = {item.body_ref: item for item in value.before_bodies}
    before_geometry_by_ref = {
        item.body_ref: item for item in before_geometry.body_geometries
    }
    changed_extents: list[str] = []
    changed_poses: list[str] = []
    changed_markers: list[str] = []
    for body, geometry in zip(resulting, after_geometry.body_geometries, strict=True):
        before_body = before_by_ref[body.body_ref]
        before_shape = before_geometry_by_ref[body.body_ref]
        if body.local_extent != before_body.local_extent:
            changed_extents.append(body.body_ref)
        if (
            geometry.root_position != before_shape.root_position
            or geometry.world_axis != before_shape.world_axis
        ):
            changed_poses.append(body.body_ref)
        before_markers = dict(before_shape.marker_positions)
        changed_markers.extend(
            marker_ref
            for marker_ref, position in geometry.marker_positions
            if before_markers.get(marker_ref) != position
        )
    return ArticulatedSuccessorMeasurements(
        resulting_bodies=resulting,
        changed_extent_refs=tuple(changed_extents),
        changed_pose_refs=tuple(changed_poses),
        changed_marker_refs=tuple(changed_markers),
        transaction_outcome=value.transaction_outcome,
        atomic_model=value.atomic_model,
    )
