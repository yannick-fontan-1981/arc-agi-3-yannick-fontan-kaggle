"""Bounded affine-lattice, marked-carrier, and ordered-collection measurements.

Every result in this module is descriptive.  Repetition does not make a grid,
a marked carrier does not make an operator, and an aligned collection does not
make a program.  DRM owns those meanings after action-conditioned evidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations
from typing import Iterable

from agents.yf_arc3_v5.capabilities.contracts import (
    ComponentDescription,
    ComponentExtractionResult,
    FrameGrid,
    VisualSpriteDescription,
)
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest


Position = tuple[int, int]
Vector = tuple[int, int]

MAX_VECTOR_EXTENT = 16
MAX_VECTOR_CANDIDATES = 12
MAX_AFFINE_BASES = 3
MAX_GLYPH_FAMILIES = 3
MAX_COLLECTION_FAMILIES = 3
MAX_ACTIVATION_EPISODES = 12


def _canonical_half_plane(vector: Vector) -> bool:
    row, column = vector
    return row > 0 or (row == 0 and column > 0)


@lru_cache(maxsize=32)
def measure_affine_repetition_bases(frame: FrameGrid) -> FrozenMap:
    """Enumerate at most three rank-two repeated-component-centre bases.

    Only translation-invariant component morphology groups with at least nine
    occurrences participate.  This excludes flat interiors and display bands:
    a vector must transport one complete measured component occurrence to an
    exact morphology/value peer, rather than merely connect adjacent pixels.
    """

    from agents.yf_arc3_v5.capabilities.components import detect_components
    from agents.yf_arc3_v5.capabilities.contracts import ComponentExtractionInput

    components = detect_components(
        ComponentExtractionInput(frame=frame, connectivity=4)
    ).components
    groups: dict[tuple[object, ...], set[Position]] = defaultdict(set)
    for component in components:
        if component.touches_frame_boundary:
            continue
        groups[
            (component.value, component.area, component.relative_pixels)
        ].add(
            (
                component.bbox.top + component.bbox.bottom,
                component.bbox.left + component.bbox.right,
            )
        )
    repeated_groups = tuple(
        frozenset(centres)
        for _signature, centres in sorted(groups.items(), key=lambda item: str(item[0]))
        if len(centres) >= 9
    )
    if not repeated_groups:
        return FrozenMap(
            {
                "affine_repetition_basis_count": 0,
                "affine_repetition_bases": (),
                "affine_repetition_has_oblique_rank_two_basis": False,
            }
        )
    measured_component_count = sum(len(items) for items in repeated_groups)
    minimum_support = max(8, min(64, measured_component_count // 20))
    vectors: list[tuple[int, int, int]] = []
    for delta_row in range(-MAX_VECTOR_EXTENT, MAX_VECTOR_EXTENT + 1):
        for delta_column in range(-MAX_VECTOR_EXTENT, MAX_VECTOR_EXTENT + 1):
            vector = (delta_row, delta_column)
            if vector == (0, 0) or not _canonical_half_plane(vector):
                continue
            support = sum(
                1
                for centres in repeated_groups
                for row_twice, column_twice in centres
                if (
                    row_twice + 2 * delta_row,
                    column_twice + 2 * delta_column,
                )
                in centres
            )
            if support >= minimum_support:
                vectors.append((support, delta_row, delta_column))
    vectors.sort(
        key=lambda item: (
            -item[0],
            item[1] * item[1] + item[2] * item[2],
            item[1],
            item[2],
        )
    )
    retained_vectors = tuple(vectors[:MAX_VECTOR_CANDIDATES])
    pairs: list[tuple[tuple[int, ...], FrozenMap]] = []
    for first, second in combinations(retained_vectors, 2):
        first_support, first_row, first_column = first
        second_support, second_row, second_column = second
        determinant = first_row * second_column - first_column * second_row
        if determinant == 0:
            continue
        first_vector = (first_row, first_column)
        second_vector = (second_row, second_column)
        record = FrozenMap(
            {
                "affine_repetition_basis_ref": (
                    "measurement.affine_repetition_basis."
                    + stable_digest(
                        (
                            first_vector,
                            second_vector,
                            first_support,
                            second_support,
                        )
                    )[:16]
                ),
                "first_basis_vector": first_vector,
                "second_basis_vector": second_vector,
                "first_vector_same_value_support_count": first_support,
                "second_vector_same_value_support_count": second_support,
                "basis_determinant": determinant,
                "basis_vectors_are_both_oblique": bool(
                    first_row
                    and first_column
                    and second_row
                    and second_column
                ),
                "measured_repeated_component_occurrence_count": (
                    measured_component_count
                ),
            }
        )
        order = (
            -min(first_support, second_support),
            -(first_support + second_support),
            first_row * first_row
            + first_column * first_column
            + second_row * second_row
            + second_column * second_column,
            abs(determinant),
            first_row,
            first_column,
            second_row,
            second_column,
        )
        pairs.append((order, record))
    pairs.sort(key=lambda item: item[0])
    bases = tuple(record for _order, record in pairs[:MAX_AFFINE_BASES])
    return FrozenMap(
        {
            "affine_repetition_basis_count": len(bases),
            "affine_repetition_bases": bases,
            "affine_repetition_has_oblique_rank_two_basis": any(
                item["basis_vectors_are_both_oblique"] for item in bases
            ),
            "affine_repetition_vector_candidate_count": len(retained_vectors),
            "affine_repetition_measurement_bound_reached": bool(
                len(vectors) > MAX_VECTOR_CANDIDATES
            ),
        }
    )


def _translation_coefficients(
    translation: Vector,
    first: Vector,
    second: Vector,
) -> tuple[int, int] | None:
    determinant = first[0] * second[1] - first[1] * second[0]
    if determinant == 0:
        return None
    first_numerator = translation[0] * second[1] - translation[1] * second[0]
    second_numerator = first[0] * translation[1] - first[1] * translation[0]
    if first_numerator % determinant or second_numerator % determinant:
        return None
    return first_numerator // determinant, second_numerator // determinant


def measure_translations_against_affine_bases(
    affine_measurement: FrozenMap,
    translations: Iterable[tuple[str, int, int]],
) -> FrozenMap:
    """Measure exact integral coordinates of observed translations."""

    matches: list[FrozenMap] = []
    for action_ref, delta_row, delta_column in translations:
        if not delta_row and not delta_column:
            continue
        for basis in affine_measurement.get("affine_repetition_bases", ()):
            first = tuple(basis["first_basis_vector"])
            second = tuple(basis["second_basis_vector"])
            coefficients = _translation_coefficients(
                (int(delta_row), int(delta_column)),
                (int(first[0]), int(first[1])),
                (int(second[0]), int(second[1])),
            )
            if coefficients is None:
                continue
            matches.append(
                FrozenMap(
                    {
                        "action_ref": str(action_ref),
                        "translation_delta": (int(delta_row), int(delta_column)),
                        "affine_repetition_basis_ref": basis[
                            "affine_repetition_basis_ref"
                        ],
                        "integral_basis_coefficients": coefficients,
                        "single_basis_quantum": (
                            abs(coefficients[0]) + abs(coefficients[1]) == 1
                        ),
                    }
                )
            )
    return FrozenMap(
        {
            "affine_integral_translation_match_count": len(matches),
            "affine_integral_translation_matches": tuple(matches[:32]),
            "affine_single_basis_quantum_match_present": any(
                item["single_basis_quantum"] for item in matches
            ),
        }
    )


def _regular_axis_members(
    members: Iterable[VisualSpriteDescription],
    *,
    axis: str,
) -> tuple[VisualSpriteDescription, ...]:
    ordered = tuple(
        sorted(
            members,
            key=lambda item: (
                item.bbox.left if axis == "horizontal" else item.bbox.top,
                item.sprite_id,
            ),
        )
    )
    if len(ordered) < 3 or len(ordered) > 8:
        return ()
    coordinates = tuple(
        item.bbox.left + item.bbox.right
        if axis == "horizontal"
        else item.bbox.top + item.bbox.bottom
        for item in ordered
    )
    pitches = tuple(second - first for first, second in zip(coordinates, coordinates[1:]))
    if not pitches or len(set(pitches)) != 1 or pitches[0] <= 0:
        return ()
    if any(
        (
            first.bbox.right >= second.bbox.left
            if axis == "horizontal"
            else first.bbox.bottom >= second.bbox.top
        )
        for first, second in zip(ordered, ordered[1:])
    ):
        return ()
    return ordered


def measure_affine_glyph_carriers(
    frame: FrameGrid,
    components: ComponentExtractionResult,
    sprites: tuple[VisualSpriteDescription, ...],
) -> tuple[FrozenMap, FrozenMap]:
    """Factor aligned carriers into common internal mark positions."""

    affine = measure_affine_repetition_bases(frame)
    components_by_ref = {
        component.component_id: component for component in components.components
    }
    horizontal_groups: dict[
        tuple[int, int, int, int], list[VisualSpriteDescription]
    ] = defaultdict(list)
    vertical_groups: dict[
        tuple[int, int, int, int], list[VisualSpriteDescription]
    ] = defaultdict(list)
    for sprite in sprites:
        anchor = components_by_ref.get(sprite.anchor_component_ref)
        if (
            anchor is None
            or sprite.touches_frame_boundary
            or sprite.internal_component_count < 1
            or not (3 <= sprite.bbox.height <= 16)
            or not (3 <= sprite.bbox.width <= 16)
            or anchor.bbox != sprite.bbox
        ):
            continue
        horizontal_groups[
            (
                sprite.bbox.top,
                sprite.bbox.bottom,
                sprite.bbox.height,
                sprite.bbox.width,
            )
        ].append(sprite)
        vertical_groups[
            (
                sprite.bbox.left,
                sprite.bbox.right,
                sprite.bbox.height,
                sprite.bbox.width,
            )
        ].append(sprite)

    raw_families: list[tuple[str, tuple[VisualSpriteDescription, ...]]] = []
    for _group_key, members in sorted(
        horizontal_groups.items(), key=lambda item: str(item[0])
    ):
        regular = _regular_axis_members(members, axis="horizontal")
        if regular:
            raw_families.append(("horizontal", regular))
    for _group_key, members in sorted(
        vertical_groups.items(), key=lambda item: str(item[0])
    ):
        regular = _regular_axis_members(members, axis="vertical")
        if regular:
            raw_families.append(("vertical", regular))
    raw_families.sort(
        key=lambda item: (
            -len(item[1]),
            -(item[1][0].bbox.height * item[1][0].bbox.width),
            item[0],
            tuple(member.sprite_id for member in item[1]),
        )
    )

    family_records: list[FrozenMap] = []
    component_facts: dict[str, FrozenMap] = {}
    for axis, members in raw_families[:MAX_GLYPH_FAMILIES]:
        carrier_values = tuple(
            components_by_ref[member.anchor_component_ref].value for member in members
        )
        if len(set(carrier_values)) != len(carrier_values):
            continue
        marker_positions_by_member: list[frozenset[Position]] = []
        for member, carrier_value in zip(members, carrier_values):
            marker_positions_by_member.append(
                frozenset(
                    (row, column)
                    for row, pattern_row in enumerate(member.pattern)
                    for column, value in enumerate(pattern_row)
                    if value != carrier_value
                )
            )
        common_positions = frozenset.intersection(*marker_positions_by_member)
        if len(common_positions) < 3:
            continue
        family_ref = (
            "measurement.aligned_glyph_carrier_family."
            + stable_digest(
                (
                    axis,
                    members[0].bbox.height,
                    members[0].bbox.width,
                    tuple(sorted(common_positions)),
                    tuple(member.sprite_id for member in members),
                )
            )[:16]
        )
        member_records: list[FrozenMap] = []
        for ordinal, (member, carrier_value, marker_positions) in enumerate(
            zip(members, carrier_values, marker_positions_by_member)
        ):
            common_value_counts = Counter(
                member.pattern[row][column]
                for row, column in sorted(common_positions)
            )
            singleton_position: Position | None = None
            if len(common_value_counts) == 2:
                singleton_values = tuple(
                    value
                    for value, count in sorted(common_value_counts.items())
                    if count == 1
                )
                if len(singleton_values) == 1:
                    singleton_position = next(
                        position
                        for position in sorted(common_positions)
                        if member.pattern[position[0]][position[1]]
                        == singleton_values[0]
                    )
            singleton_vector_twice = (
                (
                    2 * singleton_position[0] - (member.bbox.height - 1),
                    2 * singleton_position[1] - (member.bbox.width - 1),
                )
                if singleton_position is not None
                else (0, 0)
            )
            basis_vectors = tuple(
                tuple(vector)
                for basis in affine.get("affine_repetition_bases", ())
                for vector in (
                    basis["first_basis_vector"],
                    basis["second_basis_vector"],
                )
            )
            singleton_matches_basis = bool(
                singleton_position is not None
                and any(
                    singleton_vector_twice == vector
                    or singleton_vector_twice == (-vector[0], -vector[1])
                    for vector in basis_vectors
                )
            )
            extra_positions = tuple(sorted(marker_positions - common_positions))
            record = FrozenMap(
                {
                    "component_ref": member.anchor_component_ref,
                    "sprite_ref": member.sprite_id,
                    "carrier_palette_value": carrier_value,
                    "glyph_family_ordinal": ordinal,
                    "common_mark_count": len(common_positions),
                    "common_mark_positions": tuple(sorted(common_positions)),
                    "common_mark_distinct_value_count": len(common_value_counts),
                    "common_marks_have_uniform_value": len(common_value_counts) == 1,
                    "singleton_common_mark_present": singleton_position is not None,
                    "singleton_common_mark_position": singleton_position or (-1, -1),
                    "singleton_common_mark_vector_twice": singleton_vector_twice,
                    "singleton_common_mark_matches_affine_basis_vector": (
                        singleton_matches_basis
                    ),
                    "extra_mark_count": len(extra_positions),
                    "extra_mark_positions": extra_positions,
                }
            )
            member_records.append(record)
            component_facts[member.anchor_component_ref] = FrozenMap(
                {
                    "candidate_in_aligned_glyph_carrier_family": True,
                    "candidate_glyph_family_ref": family_ref,
                    "candidate_glyph_family_ordinal": ordinal,
                    "candidate_glyph_common_mark_count": len(common_positions),
                    "candidate_glyph_common_marks_have_uniform_value": (
                        len(common_value_counts) == 1
                    ),
                    "candidate_glyph_singleton_common_mark_present": (
                        singleton_position is not None
                    ),
                    "candidate_glyph_singleton_common_mark_vector_twice": (
                        singleton_vector_twice
                    ),
                    "candidate_glyph_singleton_mark_matches_affine_basis_vector": (
                        singleton_matches_basis
                    ),
                    "candidate_glyph_extra_mark_count": len(extra_positions),
                }
            )
        family_records.append(
            FrozenMap(
                {
                    "glyph_family_ref": family_ref,
                    "alignment_axis": axis,
                    "member_count": len(member_records),
                    "common_mark_count": len(common_positions),
                    "member_measurements": tuple(member_records),
                }
            )
        )
    context = FrozenMap.overlay(
        affine,
        FrozenMap(
            {
                "aligned_glyph_carrier_family_count": len(family_records),
                "aligned_glyph_carrier_families": tuple(family_records),
                "aligned_glyph_carrier_member_count_max": max(
                    (int(item["member_count"]) for item in family_records),
                    default=0,
                ),
                "aligned_glyph_common_mark_count_max": max(
                    (int(item["common_mark_count"]) for item in family_records),
                    default=0,
                ),
                "glyph_singleton_affine_marker_member_count": sum(
                    bool(member["singleton_common_mark_matches_affine_basis_vector"])
                    for family in family_records
                    for member in family["member_measurements"]
                ),
                "glyph_uniform_common_mark_member_count": sum(
                    bool(member["common_marks_have_uniform_value"])
                    for family in family_records
                    for member in family["member_measurements"]
                ),
            }
        ),
    )
    return context, FrozenMap(component_facts)


def measure_ordered_rectangular_collections(
    frame: FrameGrid,
    components: ComponentExtractionResult,
) -> tuple[FrozenMap, ...]:
    """Measure regularly aligned filled rectangles without assigning a role.

    A component may carry a one-pixel attached marker of the same value.  The
    bounded two-pixel trim below recovers the largest exact rectangular body
    while retaining the attachment as separate, still-uninterpreted evidence.
    """

    rectangles: list[tuple[str, int, tuple[int, int, int, int], int]] = []
    for component in components.components:
        if (
            component.area < 4
            or component.bbox.height > 20
            or component.bbox.width > 36
        ):
            continue
        best: tuple[int, int, int, int] | None = None
        best_area = 0
        for trim_top in range(3):
            for trim_bottom in range(3):
                top = component.bbox.top + trim_top
                bottom = component.bbox.bottom - trim_bottom
                if bottom - top + 1 < 2:
                    continue
                for trim_left in range(3):
                    for trim_right in range(3):
                        left = component.bbox.left + trim_left
                        right = component.bbox.right - trim_right
                        height = bottom - top + 1
                        width = right - left + 1
                        if width < 2 or height > 16 or width > 32:
                            continue
                        area = height * width
                        if area <= best_area:
                            continue
                        if all(
                            frame.rows[row][column] == component.value
                            for row in range(top, bottom + 1)
                            for column in range(left, right + 1)
                        ):
                            best = (top, left, bottom, right)
                            best_area = area
        if best is not None:
            rectangles.append(
                (component.component_id, component.value, best, best_area)
            )
    unique_rectangles = {
        (value, box): (component_ref, value, box, area)
        for component_ref, value, box, area in rectangles
    }
    groups: dict[
        tuple[str, int, int, int, int],
        list[tuple[str, int, tuple[int, int, int, int], int]],
    ] = defaultdict(list)
    for rectangle in sorted(
        unique_rectangles.values(),
        key=lambda item: (item[2], item[1], item[0]),
    ):
        _component_ref, _value, box, _area = rectangle
        top, left, bottom, right = box
        height = bottom - top + 1
        width = right - left + 1
        groups[
            (
                "vertical",
                left,
                right,
                height,
                width,
            )
        ].append(rectangle)
        groups[
            (
                "horizontal",
                top,
                bottom,
                height,
                width,
            )
        ].append(rectangle)
    records: list[FrozenMap] = []
    for key, raw_members in sorted(groups.items(), key=lambda item: str(item[0])):
        axis = key[0]
        ordered = tuple(
            sorted(
                raw_members,
                key=lambda item: (
                    item[2][0] if axis == "vertical" else item[2][1],
                    item[0],
                ),
            )
        )
        if not (2 <= len(ordered) <= 12):
            continue
        coordinates = tuple(
            item[2][0] + item[2][2]
            if axis == "vertical"
            else item[2][1] + item[2][3]
            for item in ordered
        )
        pitches = tuple(
            second - first for first, second in zip(coordinates, coordinates[1:])
        )
        if len(set(pitches)) != 1 or pitches[0] <= 0:
            continue
        if any(
            (
                first[2][2] >= second[2][0]
                if axis == "vertical"
                else first[2][3] >= second[2][1]
            )
            for first, second in zip(ordered, ordered[1:])
        ):
            continue
        signature = (
            axis,
            key[1:],
            ordered[0][2][2] - ordered[0][2][0] + 1,
            ordered[0][2][3] - ordered[0][2][1] + 1,
            pitches[0],
        )
        records.append(
            FrozenMap(
                {
                    "collection_measurement_ref": (
                        "measurement.ordered_rectangular_collection."
                        + stable_digest(
                            (signature, tuple(item[0] for item in ordered))
                        )[:16]
                    ),
                    "alignment_axis": axis,
                    "geometry_signature": signature,
                    "member_count": len(ordered),
                    "member_area": ordered[0][3],
                    "member_pitch_twice": pitches[0],
                    "member_component_refs": tuple(
                        item[0] for item in ordered
                    ),
                    "member_palette_values": tuple(item[1] for item in ordered),
                    "member_bounding_boxes": tuple(
                        item[2]
                        for item in ordered
                    ),
                }
            )
        )
    unique = {
        tuple(item["geometry_signature"]): item
        for item in records
    }
    ordered_records = sorted(
        unique.values(),
        key=lambda item: (
            -int(item["member_area"]),
            -int(item["member_count"]),
            str(item["collection_measurement_ref"]),
        ),
    )
    return tuple(ordered_records[:MAX_COLLECTION_FAMILIES])


def measure_point_ordered_collection_delta(
    *,
    before: FrameGrid,
    after: FrameGrid,
    point: Position | None,
) -> FrozenMap:
    """Compare the largest aligned collection around one point transition."""

    from agents.yf_arc3_v5.capabilities.components import detect_components
    from agents.yf_arc3_v5.capabilities.contracts import ComponentExtractionInput
    from agents.yf_arc3_v5.capabilities.sprites import describe_visual_glyphs
    from agents.yf_arc3_v5.capabilities.contracts import VisualSceneInput

    before_components = detect_components(
        ComponentExtractionInput(frame=before, connectivity=4)
    )
    after_components = detect_components(
        ComponentExtractionInput(frame=after, connectivity=4)
    )
    before_collections = measure_ordered_rectangular_collections(
        before, before_components
    )
    after_collections = measure_ordered_rectangular_collections(after, after_components)
    before_collection = before_collections[0] if before_collections else None
    after_by_signature = {
        tuple(item["geometry_signature"]): item for item in after_collections
    }
    after_collection = (
        after_by_signature.get(tuple(before_collection["geometry_signature"]))
        if before_collection is not None
        else None
    )
    before_count = int(before_collection["member_count"]) if before_collection else 0
    after_count = int(after_collection["member_count"]) if after_collection else 0
    if before_collection is not None and after_collection is None:
        after_count = 0

    clicked_collection_ordinal = -1
    if point is not None and before_collection is not None:
        for ordinal, box in enumerate(before_collection["member_bounding_boxes"]):
            top, left, bottom, right = box
            if top <= point[0] <= bottom and left <= point[1] <= right:
                clicked_collection_ordinal = ordinal
                break

    before_blocks, before_sprites = describe_visual_glyphs(
        VisualSceneInput(frame=before)
    )
    _glyph_context, glyph_facts = measure_affine_glyph_carriers(
        before,
        before_blocks,
        before_sprites,
    )
    clicked_glyph_component_ref = ""
    clicked_glyph_carrier_value = -1
    if point is not None:
        components_by_ref = {
            component.component_id: component
            for component in before_blocks.components
        }
        for component_ref in glyph_facts:
            component = components_by_ref.get(component_ref)
            if component is None:
                continue
            if (
                component.bbox.top <= point[0] <= component.bbox.bottom
                and component.bbox.left <= point[1] <= component.bbox.right
            ):
                clicked_glyph_component_ref = component_ref
                clicked_glyph_carrier_value = component.value
                break

    return FrozenMap(
        {
            "ordered_collection_measurement_present": before_collection is not None,
            "ordered_collection_before_count": before_count,
            "ordered_collection_after_count": after_count,
            "ordered_collection_count_delta": after_count - before_count,
            "ordered_collection_before_palette_values": (
                tuple(before_collection["member_palette_values"])
                if before_collection is not None
                else ()
            ),
            "ordered_collection_after_palette_values": (
                tuple(after_collection["member_palette_values"])
                if after_collection is not None
                else ()
            ),
            "point_intersects_ordered_collection_member": (
                clicked_collection_ordinal >= 0
            ),
            "point_ordered_collection_ordinal": clicked_collection_ordinal,
            "point_intersects_aligned_glyph_carrier": bool(
                clicked_glyph_component_ref
            ),
            "point_glyph_carrier_component_ref": clicked_glyph_component_ref,
            "point_glyph_carrier_palette_value": clicked_glyph_carrier_value,
        }
    )


def measure_ordered_glyph_activation_episodes(
    *,
    before: FrameGrid,
    intermediate_frames: tuple[FrameGrid, ...],
    after: FrameGrid,
) -> FrozenMap:
    """Describe ordered carrier highlights and their coincident scene deltas.

    A source episode is admitted only when exactly one member of one measured
    rectangular collection differs from the pre-action frame while every peer
    member still matches it.  Candidate lattice translations are then counted
    outside the complete source collection.  Python does not call an episode a
    movement, attack, push, or program step; DRM may do so from these exact
    relations and their later falsifiers.
    """

    from agents.yf_arc3_v5.capabilities.components import detect_components
    from agents.yf_arc3_v5.capabilities.contracts import ComponentExtractionInput
    from agents.yf_arc3_v5.capabilities.sprites import describe_visual_scene
    from agents.yf_arc3_v5.capabilities.contracts import VisualSceneInput

    frames = (before, *intermediate_frames, after)
    components = detect_components(ComponentExtractionInput(frame=before))
    collections = measure_ordered_rectangular_collections(before, components)
    if not collections:
        return FrozenMap(
            {
                "ordered_glyph_activation_episode_count": 0,
                "ordered_glyph_activation_episodes": (),
                "ordered_glyph_activation_measurement_complete": True,
            }
        )

    source = collections[0]
    boxes = tuple(tuple(box) for box in source["member_bounding_boxes"])
    palettes = tuple(int(item) for item in source["member_palette_values"])

    def expanded_difference_count(frame: FrameGrid, box: tuple[int, ...]) -> int:
        top, left, bottom, right = box
        return sum(
            frame.rows[row][column] != before.rows[row][column]
            for row in range(max(0, top - 1), min(before.height, bottom + 2))
            for column in range(max(0, left - 1), min(before.width, right + 2))
        )

    labels: list[int | None] = []
    for frame in frames:
        scores = tuple(expanded_difference_count(frame, box) for box in boxes)
        positive = tuple(index for index, score in enumerate(scores) if score > 0)
        labels.append(positive[0] if len(positive) == 1 else None)

    runs: list[tuple[int, int, int]] = []
    index = 1
    while index < len(labels) and len(runs) < MAX_ACTIVATION_EPISODES:
        label = labels[index]
        if label is None:
            index += 1
            continue
        end = index
        while end + 1 < len(labels) and labels[end + 1] == label:
            end += 1
        runs.append((label, index, end))
        index = end + 1

    # The exact episode detector needs no glyph/scene decomposition when no
    # collection member was uniquely highlighted in the observed frame series.
    if not runs:
        return FrozenMap(
            {
                "ordered_glyph_activation_episode_count": 0,
                "ordered_glyph_activation_episodes": (),
                "ordered_glyph_activation_measurement_complete": True,
                "ordered_glyph_activation_all_members_observed_once": False,
                "ordered_glyph_activation_direction_match_count": 0,
            }
        )

    scene = describe_visual_scene(VisualSceneInput(frame=before))
    glyph_context, _component_facts = measure_affine_glyph_carriers(
        before,
        components,
        scene.sprites,
    )

    glyph_by_palette: dict[int, FrozenMap] = {}
    for family in glyph_context.get("aligned_glyph_carrier_families", ()):
        for member in family["member_measurements"]:
            palette = int(member["carrier_palette_value"])
            if palette not in glyph_by_palette:
                glyph_by_palette[palette] = member

    basis_vectors: tuple[Vector, ...] = ()
    affine_bases = tuple(glyph_context.get("affine_repetition_bases", ()))
    if affine_bases:
        first = tuple(int(item) for item in affine_bases[0]["first_basis_vector"])
        second = tuple(int(item) for item in affine_bases[0]["second_basis_vector"])
        basis_vectors = (first, second, (-first[0], -first[1]), (-second[0], -second[1]))

    excluded = frozenset(
        (row, column)
        for top, left, bottom, right in boxes
        for row in range(max(0, top - 1), min(before.height, bottom + 2))
        for column in range(max(0, left - 1), min(before.width, right + 2))
    )
    background_value = Counter(
        value for row in before.rows for value in row
    ).most_common(1)[0][0]

    episodes: list[FrozenMap] = []
    for ordinal, start, end in runs:
        prior = frames[start - 1]
        settled = frames[end]
        changed = tuple(
            (row, column)
            for row in range(before.height)
            for column in range(before.width)
            if (row, column) not in excluded
            and prior.rows[row][column] != settled.rows[row][column]
        )
        translation_supports: list[FrozenMap] = []
        for delta_row, delta_column in basis_vectors:
            support = 0
            for row in range(before.height):
                for column in range(before.width):
                    destination = (row + delta_row, column + delta_column)
                    if (
                        (row, column) in excluded
                        or destination in excluded
                        or not (0 <= destination[0] < before.height)
                        or not (0 <= destination[1] < before.width)
                    ):
                        continue
                    value = prior.rows[row][column]
                    if value == background_value:
                        continue
                    if value != settled.rows[destination[0]][destination[1]]:
                        continue
                    if (
                        prior.rows[row][column] == settled.rows[row][column]
                        and prior.rows[destination[0]][destination[1]]
                        == settled.rows[destination[0]][destination[1]]
                    ):
                        continue
                    support += 1
            if support:
                translation_supports.append(
                    FrozenMap(
                        {
                            "translation_delta": (delta_row, delta_column),
                            "translated_changed_pixel_support": support,
                        }
                    )
                )
        translation_supports.sort(
            key=lambda item: (
                -int(item["translated_changed_pixel_support"]),
                tuple(item["translation_delta"]),
            )
        )
        leading_translation = (
            translation_supports[0]["translation_delta"]
            if translation_supports
            else (0, 0)
        )
        glyph = glyph_by_palette.get(palettes[ordinal], FrozenMap())
        singleton_vector = tuple(
            glyph.get("singleton_common_mark_vector_twice") or (0, 0)
        )
        episodes.append(
            FrozenMap(
                {
                    "source_collection_ref": source["collection_measurement_ref"],
                    "source_member_ordinal": ordinal,
                    "source_palette_value": palettes[ordinal],
                    "activation_first_frame_index": start,
                    "activation_last_frame_index": end,
                    "activation_frame_count": end - start + 1,
                    "outside_source_changed_pixel_count": len(changed),
                    "outside_source_change_bounding_box": (
                        (
                            min(row for row, _column in changed),
                            min(column for _row, column in changed),
                            max(row for row, _column in changed),
                            max(column for _row, column in changed),
                        )
                        if changed
                        else (-1, -1, -1, -1)
                    ),
                    "equal_palette_glyph_member_present": bool(glyph),
                    "glyph_common_mark_count": int(glyph.get("common_mark_count") or 0),
                    "glyph_singleton_common_mark_present": bool(
                        glyph.get("singleton_common_mark_present")
                    ),
                    "glyph_singleton_common_mark_vector_twice": singleton_vector,
                    "affine_translation_candidates": tuple(translation_supports),
                    "leading_affine_translation_delta": leading_translation,
                    "leading_affine_translation_matches_singleton_glyph_vector": bool(
                        singleton_vector != (0, 0)
                        and singleton_vector == tuple(leading_translation)
                    ),
                }
            )
        )

    return FrozenMap(
        {
            "ordered_glyph_activation_episode_count": len(episodes),
            "ordered_glyph_activation_episodes": tuple(episodes),
            "ordered_glyph_activation_measurement_complete": len(runs) < MAX_ACTIVATION_EPISODES,
            "ordered_glyph_activation_all_members_observed_once": bool(
                len(episodes) == len(boxes)
                and sorted(int(item["source_member_ordinal"]) for item in episodes)
                == list(range(len(boxes)))
            ),
            "ordered_glyph_activation_direction_match_count": sum(
                bool(item["leading_affine_translation_matches_singleton_glyph_vector"])
                for item in episodes
            ),
        }
    )
