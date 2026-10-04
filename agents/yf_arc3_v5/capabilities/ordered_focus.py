"""Bounded geometry measurements for outlined horizontal member rows.

This module assigns no semantic role.  It only measures rectangular supports,
compact members, one locally enclosing hollow contour, and exact ordinal change.
DRM decides whether those facts describe a reading order or an action mapping.
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.yf_arc3_v5.capabilities.contracts import (
    BoundingBox,
    ComponentDescription,
    ComponentExtractionInput,
    FrameGrid,
)
from agents.yf_arc3_v5.capabilities.components import detect_components
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest


MAX_ROW_DESCRIPTIONS = 3
MAX_ROW_MEMBERS = 8


@dataclass(frozen=True)
class HorizontalContouredMemberRow:
    support_bbox: tuple[int, int, int, int]
    contour_bbox: tuple[int, int, int, int]
    member_bboxes: tuple[tuple[int, int, int, int], ...]
    member_shape_digests: tuple[str, ...]
    member_values: tuple[int, ...]
    outlined_member_ordinal: int


def _member_centers_twice(
    row: HorizontalContouredMemberRow,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (top + bottom, left + right)
        for top, left, bottom, right in row.member_bboxes
    )


def horizontal_member_order_digest(row: HorizontalContouredMemberRow) -> str:
    """Hash the stable support and ordered slot centers, not mutable decoration."""

    return stable_digest((row.support_bbox, _member_centers_twice(row)))[:16]


def _bbox_tuple(bbox: BoundingBox) -> tuple[int, int, int, int]:
    return bbox.top, bbox.left, bbox.bottom, bbox.right


def _strictly_contains(outer: BoundingBox, inner: BoundingBox) -> bool:
    return (
        outer.top < inner.top
        and outer.left < inner.left
        and outer.bottom > inner.bottom
        and outer.right > inner.right
    )


def _is_hollow_rectangle_contour(component: ComponentDescription) -> bool:
    height = component.bbox.height
    width = component.bbox.width
    if height < 3 or width < 3:
        return False
    perimeter = frozenset(
        (row, col)
        for row in range(height)
        for col in range(width)
        if row in {0, height - 1} or col in {0, width - 1}
    )
    pixels = frozenset(component.relative_pixels)
    if pixels == perimeter:
        return True
    rounded_perimeter = frozenset(
        (row, col)
        for row in range(height)
        for col in range(width)
        if (
            row in {0, height - 1} and 0 < col < width - 1
        )
        or (
            col in {0, width - 1} and 0 < row < height - 1
        )
    )
    return pixels == rounded_perimeter


def _compact_member(component: ComponentDescription) -> bool:
    return (
        3 <= component.area <= 32
        and 2 <= component.bbox.height <= 7
        and 2 <= component.bbox.width <= 7
        and not component.touches_frame_boundary
    )


def measure_horizontal_contoured_member_rows(
    frame: FrameGrid,
) -> tuple[HorizontalContouredMemberRow, ...]:
    """Enumerate at most three exact horizontally ordered row descriptions."""

    descriptions: list[HorizontalContouredMemberRow] = []
    all_components = tuple(
        detect_components(
            ComponentExtractionInput(frame=frame, connectivity=8)
        ).components
    )
    supports = tuple(
        component
        for component in all_components
        if _is_hollow_rectangle_contour(component)
        and component.bbox.height >= 7
        and component.bbox.width >= component.bbox.height * 2
        and not component.touches_frame_boundary
    )
    for support in sorted(supports, key=lambda item: _bbox_tuple(item.bbox)):
        enclosed = tuple(
            component
            for component in all_components
            if component.component_id != support.component_id
            and _strictly_contains(support.bbox, component.bbox)
        )
        compact = tuple(component for component in enclosed if _compact_member(component))
        members = tuple(
            sorted(
                (
                    component
                    for component in compact
                    if not any(
                        other.component_id != component.component_id
                        and other.area > component.area
                        and _strictly_contains(other.bbox, component.bbox)
                        for other in compact
                    )
                ),
                key=lambda item: (
                    item.bbox.left,
                    item.bbox.top,
                    item.component_id,
                ),
            )
        )
        if not 2 <= len(members) <= MAX_ROW_MEMBERS:
            continue
        center_rows_twice = tuple(
            member.bbox.top + member.bbox.bottom for member in members
        )
        if max(center_rows_twice) - min(center_rows_twice) > 2:
            continue
        if any(
            left.bbox.right >= right.bbox.left
            for left, right in zip(members, members[1:])
        ):
            continue
        contours = tuple(
            component
            for component in enclosed
            if component.component_id not in {
                member.component_id for member in members
            }
            and _is_hollow_rectangle_contour(component)
            and 5 <= component.bbox.height <= 11
            and 5 <= component.bbox.width <= 11
        )
        outlined_pairs = tuple(
            (contour, ordinal)
            for contour in contours
            for ordinal, member in enumerate(members)
            if _strictly_contains(contour.bbox, member.bbox)
            and sum(
                _strictly_contains(contour.bbox, other.bbox)
                for other in members
            )
            == 1
        )
        if len(outlined_pairs) != 1:
            continue
        contour, outlined_ordinal = outlined_pairs[0]
        descriptions.append(
            HorizontalContouredMemberRow(
                support_bbox=_bbox_tuple(support.bbox),
                contour_bbox=_bbox_tuple(contour.bbox),
                member_bboxes=tuple(_bbox_tuple(member.bbox) for member in members),
                member_shape_digests=tuple(
                    stable_digest(tuple(member.relative_pixels))[:16]
                    for member in members
                ),
                member_values=tuple(int(member.value) for member in members),
                outlined_member_ordinal=outlined_ordinal,
            )
        )
        if len(descriptions) >= MAX_ROW_DESCRIPTIONS:
            break
    return tuple(descriptions)


def measure_left_to_right_outlined_successor(
    before_frame: FrameGrid,
    after_frame: FrameGrid,
) -> FrozenMap:
    """Measure an exact one-member rightward outline advance, if unique."""

    before_rows = measure_horizontal_contoured_member_rows(before_frame)
    after_rows = measure_horizontal_contoured_member_rows(after_frame)
    matches = tuple(
        (before, after)
        for before in before_rows
        for after in after_rows
        if before.support_bbox == after.support_bbox
        and _member_centers_twice(before) == _member_centers_twice(after)
        and all(
            before_shape == after_shape
            for ordinal, (before_shape, after_shape) in enumerate(
                zip(before.member_shape_digests, after.member_shape_digests)
            )
            if ordinal != before.outlined_member_ordinal
        )
    )
    facts: dict[str, object] = {
        "horizontal_contoured_member_row_before_count": len(before_rows),
        "horizontal_contoured_member_row_after_count": len(after_rows),
        "horizontal_contoured_member_row_match_count": len(matches),
        "outlined_member_exact_right_successor_observed": False,
    }
    if len(matches) != 1:
        return FrozenMap(facts)
    before, after = matches[0]
    ordinal_delta = after.outlined_member_ordinal - before.outlined_member_ordinal
    contour_column_delta = after.contour_bbox[1] - before.contour_bbox[1]
    facts.update(
        {
            "horizontal_contoured_member_count": len(before.member_bboxes),
            "horizontal_contoured_member_distinct_shape_count": len(
                frozenset(before.member_shape_digests)
            ),
            "outlined_member_before_ordinal": before.outlined_member_ordinal,
            "outlined_member_after_ordinal": after.outlined_member_ordinal,
            "outlined_member_ordinal_delta": ordinal_delta,
            "outlined_contour_column_delta": contour_column_delta,
            "outlined_member_before_shape_digest": before.member_shape_digests[
                before.outlined_member_ordinal
            ],
            "outlined_member_exact_right_successor_observed": bool(
                ordinal_delta == 1 and contour_column_delta > 0
            ),
            "horizontal_member_order_digest": horizontal_member_order_digest(before),
        }
    )
    return FrozenMap(facts)
