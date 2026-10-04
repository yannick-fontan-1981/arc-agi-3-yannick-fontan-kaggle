"""Bounded multi-resolution measurements over an already decomposed scene."""

from __future__ import annotations

from functools import lru_cache

from agents.yf_arc3_v5.capabilities.contracts import (
    BoundingBox,
    ComponentDescription,
    MultiResolutionViewInput,
    MultiResolutionViewMeasurements,
    VisualSceneDescription,
)
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest


def measure_multi_resolution_views(
    value: MultiResolutionViewInput,
) -> MultiResolutionViewMeasurements:
    """Summarize existing geometry without rescanning pixels or assigning roles."""

    scene = value.scene
    limit = value.maximum_descriptions_per_family
    distinct_visual_element_limit = value.maximum_distinct_visual_elements
    height = scene.frame.height
    width = scene.frame.width
    frame_area = height * width
    block_by_ref = {
        component.component_id: component for component in scene.blocks.components
    }

    enclosure_all = tuple(
        _frame_enclosure_tuple(block_by_ref[component_ref], height, width, frame_area)
        for component_ref in scene.frame_enclosing_component_refs
    )
    lattice_all = tuple(
        (
            candidate.candidate_ref,
            candidate.row_offset,
            candidate.col_offset,
            candidate.cell_height,
            candidate.cell_width,
            candidate.row_gap,
            candidate.col_gap,
            candidate.row_pitch,
            candidate.col_pitch,
            candidate.logical_rows,
            candidate.logical_columns,
            candidate.separator_consistency_ppm,
        )
        for candidate in scene.periodic_cell_grids.candidates
    )
    zero_gap_lattice_all = tuple(
        measurement
        for measurement in lattice_all
        if measurement[5] == 0 and measurement[6] == 0
    )
    scale_all = tuple(
        (sprite.sprite_id, sprite.exact_dezoom_factors)
        for sprite in scene.sprites
        if sprite.exact_dezoom_factors
    )
    padding_all = tuple(
        (
            sprite.sprite_id,
            sprite.bbox.top,
            sprite.bbox.left,
            height - 1 - sprite.bbox.bottom,
            width - 1 - sprite.bbox.right,
        )
        for sprite in scene.sprites
    )
    thin_all = tuple(
        (
            component.component_id,
            component.bbox.height,
            component.bbox.width,
            _boundary_contact_count(component, height, width),
        )
        for component in scene.blocks.components
        if _has_thin_extent(component)
    )
    composite_all = tuple(
        (sprite.sprite_id, sprite.internal_component_count + 1)
        for sprite in scene.sprites
        if sprite.internal_component_count > 0
    )
    hole_all = tuple(
        (
            item.hole_sprite_id,
            item.enclosing_component_ref,
            item.bbox.height,
            item.bbox.width,
            len(item.pixels),
            item.value_count,
        )
        for item in scene.hole_sprites
    )
    repeated_all = tuple(
        (sprite.sprite_id, sprite.pattern_occurrence_count)
        for sprite in scene.sprites
        if sprite.pattern_occurrence_count > 1
    )
    (
        distinct_visual_element_measurements,
        distinct_visual_element_measurement_count,
        distinct_visual_element_supports,
        distinct_visual_element_value_supports,
    ) = _distinct_visual_element_measurements(
        scene,
        maximum_measurements=distinct_visual_element_limit,
    )
    (
        distinct_visual_element_pair_relation_measurements,
        distinct_visual_element_pair_relation_measurement_count,
    ) = _distinct_visual_element_pair_relations(
        distinct_visual_element_measurements,
        distinct_visual_element_supports,
        distinct_visual_element_value_supports,
        frame_height=height,
        frame_width=width,
        maximum_measurements=(
            value.maximum_distinct_visual_element_pair_relations
        ),
    )
    exact_mask_transform_all = tuple(
        FrozenMap(
            {
                "transform_ref": (
                    "measurement.exact_mask_transform:"
                    f"{stable_digest((relation['relation_ref'], index, candidate))[:20]}"
                ),
                "relation_ref": relation["relation_ref"],
                "first_element_ref": relation["first_element_ref"],
                "second_element_ref": relation["second_element_ref"],
                "first_bbox_top": relation["first_bbox_top"],
                "first_bbox_left": relation["first_bbox_left"],
                "first_bbox_bottom": relation["first_bbox_bottom"],
                "first_bbox_right": relation["first_bbox_right"],
                "second_bbox_top": relation["second_bbox_top"],
                "second_bbox_left": relation["second_bbox_left"],
                "second_bbox_bottom": relation["second_bbox_bottom"],
                "second_bbox_right": relation["second_bbox_right"],
                **dict(candidate),
                "normalized_cell_count": (
                    int(candidate["normalized_height"])
                    * int(candidate["normalized_width"])
                ),
                "scale_relation_non_identity": (
                    candidate["first_isotropic_factor"]
                    != candidate["second_isotropic_factor"]
                ),
                "rotation_relation_non_identity": (
                    int(candidate["quarter_turns_clockwise"]) != 0
                ),
            }
        )
        for relation in distinct_visual_element_pair_relation_measurements
        for index, candidate in enumerate(
            relation["exact_mask_transform_candidates"]
        )
    )
    exact_mask_transform_measurements = exact_mask_transform_all[
        : value.maximum_distinct_visual_element_pair_relations
    ]
    exact_mask_transform_enumeration_truncated = (
        len(exact_mask_transform_all)
        > value.maximum_distinct_visual_element_pair_relations
    )
    (
        distinct_visual_element_repeated_pair_thin_relation_measurements,
        distinct_visual_element_repeated_pair_thin_relation_measurement_count,
    ) = _distinct_visual_element_repeated_pair_thin_relations(
        distinct_visual_element_measurements,
        distinct_visual_element_supports,
        maximum_measurements=(
            value.maximum_distinct_visual_element_repeated_pair_thin_relations
        ),
    )
    families = (
        ("frame_enclosure", enclosure_all),
        ("periodic_lattice", lattice_all),
        ("zero_gap_lattice", zero_gap_lattice_all),
        ("exact_scale", scale_all),
        ("frame_padding", padding_all),
        ("thin_extent", thin_all),
        ("composite_crop", composite_all),
        ("hole_sprite", hole_all),
        ("repeated_crop", repeated_all),
    )
    distinct_visual_element_enumeration_truncated = (
        scene.hole_sprite_enumeration_truncated
        or distinct_visual_element_measurement_count > distinct_visual_element_limit
    )
    distinct_visual_element_pair_relation_enumeration_truncated = (
        distinct_visual_element_enumeration_truncated
        or distinct_visual_element_pair_relation_measurement_count
        > value.maximum_distinct_visual_element_pair_relations
    )
    distinct_visual_element_repeated_pair_thin_relation_enumeration_truncated = (
        distinct_visual_element_enumeration_truncated
        or distinct_visual_element_repeated_pair_thin_relation_measurement_count
        > value.maximum_distinct_visual_element_repeated_pair_thin_relations
    )
    description_truncated = (
        any(len(items) > limit for _, items in families)
        or distinct_visual_element_enumeration_truncated
        or exact_mask_transform_enumeration_truncated
    )
    observation_ref = value.observation_ref
    measurement_refs = (
        f"measurement.multi_resolution:{observation_ref}",
        *(
            f"measurement.{family}:{observation_ref}"
            for family, items in families
            if items
        ),
        *(
            item["element_ref"]
            for item in distinct_visual_element_measurements
        ),
        *(
            item["relation_ref"]
            for item in distinct_visual_element_pair_relation_measurements
        ),
        *(
            item["relation_ref"]
            for item in (
                distinct_visual_element_repeated_pair_thin_relation_measurements
            )
        ),
        *(item["transform_ref"] for item in exact_mask_transform_measurements),
    )
    facts = FrozenMap(
        {
            "observation_ref": observation_ref,
            "frame_ref": value.frame_ref,
            "is_initial_observation": value.is_initial_observation,
            "available_action_refs": value.available_action_refs,
            "available_action_count": len(value.available_action_refs),
            "frame_height": height,
            "frame_width": width,
            "connectivity_4_component_count": len(scene.blocks.components),
            "connectivity_8_component_count": len(scene.zones.components),
            "diagonal_connectivity_merge_count": max(
                0, len(scene.blocks.components) - len(scene.zones.components)
            ),
            "frame_enclosure_measurement_count": len(enclosure_all),
            "frame_enclosure_measurements": enclosure_all[:limit],
            "periodic_lattice_measurement_count": len(lattice_all),
            "periodic_lattice_measurements": lattice_all[:limit],
            "zero_gap_lattice_measurement_count": len(zero_gap_lattice_all),
            "zero_gap_lattice_measurements": zero_gap_lattice_all[:limit],
            "exact_scale_measurement_count": len(scale_all),
            "exact_scale_measurements": scale_all[:limit],
            "frame_padding_measurement_count": len(padding_all),
            "frame_padding_measurements": padding_all[:limit],
            "thin_extent_measurement_count": len(thin_all),
            "thin_extent_measurements": thin_all[:limit],
            "composite_crop_measurement_count": len(composite_all),
            "composite_crop_measurements": composite_all[:limit],
            "hole_sprite_measurement_count": scene.hole_sprite_measurement_count,
            "hole_sprite_measurements": hole_all[:limit],
            "hole_sprite_enumeration_truncated": (
                scene.hole_sprite_enumeration_truncated
            ),
            "repeated_crop_measurement_count": len(repeated_all),
            "repeated_crop_measurements": repeated_all[:limit],
            "distinct_visual_element_measurement_count": (
                distinct_visual_element_measurement_count
            ),
            "distinct_visual_element_measurements": (
                distinct_visual_element_measurements
            ),
            "distinct_visual_element_enumeration_truncated": (
                distinct_visual_element_enumeration_truncated
            ),
            "distinct_visual_element_nesting_relation_enumeration_complete": (
                not distinct_visual_element_enumeration_truncated
            ),
            "distinct_visual_element_pair_relation_measurement_count": (
                distinct_visual_element_pair_relation_measurement_count
            ),
            "distinct_visual_element_pair_relation_measurements": (
                distinct_visual_element_pair_relation_measurements
            ),
            "distinct_visual_element_pair_relation_enumeration_truncated": (
                distinct_visual_element_pair_relation_enumeration_truncated
            ),
            "distinct_visual_element_pair_relation_source_enumeration_complete": (
                not distinct_visual_element_enumeration_truncated
            ),
            "exact_mask_transform_measurement_count": len(
                exact_mask_transform_all
            ),
            "exact_mask_transform_measurements": exact_mask_transform_measurements,
            "exact_mask_transform_enumeration_truncated": (
                exact_mask_transform_enumeration_truncated
            ),
            "distinct_visual_element_repeated_pair_thin_relation_measurement_count": (
                distinct_visual_element_repeated_pair_thin_relation_measurement_count
            ),
            "distinct_visual_element_repeated_pair_thin_relation_measurements": (
                distinct_visual_element_repeated_pair_thin_relation_measurements
            ),
            "distinct_visual_element_repeated_pair_thin_relation_enumeration_truncated": (
                distinct_visual_element_repeated_pair_thin_relation_enumeration_truncated
            ),
            "distinct_visual_element_repeated_pair_thin_relation_source_enumeration_complete": (
                not distinct_visual_element_enumeration_truncated
            ),
            "descriptive_family_count": sum(bool(items) for _, items in families),
            "structural_descriptive_family_count": sum(
                bool(items)
                for items in (
                    enclosure_all,
                    lattice_all,
                    thin_all,
                    composite_all,
                    hole_all,
                )
            ),
            "enclosure_lattice_thin_or_composite_family_count": sum(
                bool(items)
                for items in (enclosure_all, lattice_all, thin_all, composite_all)
            ),
            "description_truncated": description_truncated,
        }
    )
    return MultiResolutionViewMeasurements(
        descriptive_facts=facts,
        measurement_refs=measurement_refs,
        description_truncated=description_truncated,
    )


def _distinct_visual_element_measurements(
    scene: VisualSceneDescription,
    *,
    maximum_measurements: int,
) -> tuple[
    tuple[FrozenMap, ...],
    int,
    tuple[frozenset[tuple[int, int]], ...],
    tuple[tuple[tuple[int, int, int], ...], ...],
]:
    """Quotient directly observed element descriptors by exact valued support.

    Periodic cells remain bounded alternatives in their dedicated measurement
    family.  Competing grid partitions do not become hundreds of canonical
    first-frame elements before DRM has committed a segmentation.
    """

    descriptors: list[dict[str, object]] = []
    dominant_grid_cells = tuple(
        cell.bbox
        for candidate in scene.periodic_cell_grids.candidates
        # A separator or a strongly repeated zero-gap tile can justify a
        # bounded quotient. Weak zero-gap partitions remain parallel only.
        if (
            (candidate.row_gap > 0 and candidate.col_gap > 0)
            or (
                candidate.row_gap == 0
                and candidate.col_gap == 0
                and candidate.cells
                and max(cell.pattern_occurrence_count for cell in candidate.cells) * 2
                >= len(candidate.cells)
            )
        )
        if len(candidate.cells) < len(scene.blocks.components)
        for cell in candidate.cells
    )

    def lies_inside_dominant_grid_cell(bbox: object) -> bool:
        return any(
            cell_bbox.top <= bbox.top
            and cell_bbox.left <= bbox.left
            and cell_bbox.bottom >= bbox.bottom
            and cell_bbox.right >= bbox.right
            for cell_bbox in dominant_grid_cells
        )

    def append_descriptor(
        family: str,
        family_order: int,
        descriptor_ref: str,
        value_support: tuple[tuple[int, int, int], ...],
        *,
        component_ref: str | None = None,
        anchor_component_ref: str | None = None,
    ) -> None:
        descriptors.append(
            {
                "family": family,
                "family_order": family_order,
                "descriptor_ref": descriptor_ref,
                "value_support": value_support,
                "component_ref": component_ref,
                "anchor_component_ref": anchor_component_ref,
            }
        )

    for component in scene.blocks.components:
        if dominant_grid_cells and lies_inside_dominant_grid_cell(component.bbox):
            continue
        append_descriptor(
            "component_4",
            0,
            component.component_id,
            tuple((row, col, component.value) for row, col in component.pixels),
            component_ref=component.component_id,
        )
    for component in scene.zones.components:
        if dominant_grid_cells and lies_inside_dominant_grid_cell(component.bbox):
            continue
        append_descriptor(
            "component_8",
            1,
            component.component_id,
            tuple((row, col, component.value) for row, col in component.pixels),
            component_ref=component.component_id,
        )
    for hole_sprite in scene.hole_sprites:
        append_descriptor(
            "hole_sprite",
            2,
            hole_sprite.hole_sprite_id,
            tuple(
                (row, col, scene.frame.rows[row][col])
                for row, col in hole_sprite.pixels
            ),
            anchor_component_ref=hole_sprite.enclosing_component_ref,
        )
    for sprite in scene.sprites:
        if dominant_grid_cells and lies_inside_dominant_grid_cell(sprite.bbox):
            continue
        value_support = _bbox_value_support(scene, sprite.bbox)
        append_descriptor(
            "sprite_crop",
            3,
            sprite.sprite_id,
            value_support,
            anchor_component_ref=sprite.anchor_component_ref,
        )
        if sprite.internal_component_count > 0:
            append_descriptor(
                "composite_crop",
                4,
                f"measurement.composite_crop:{sprite.sprite_id}",
                value_support,
                anchor_component_ref=sprite.anchor_component_ref,
            )

    groups: dict[
        tuple[tuple[int, int, int], ...], list[dict[str, object]]
    ] = {}
    for descriptor in descriptors:
        signature = descriptor["value_support"]
        assert isinstance(signature, tuple)
        groups.setdefault(signature, []).append(descriptor)

    grouped: list[dict[str, object]] = []
    for value_support, members in sorted(groups.items()):
        ordered_members = tuple(sorted(members, key=_descriptor_order_key))
        canonical = ordered_members[0]
        positions = frozenset((row, col) for row, col, _ in value_support)
        top = min(row for row, _ in positions)
        left = min(col for _, col in positions)
        bottom = max(row for row, _ in positions)
        right = max(col for _, col in positions)
        grouped.append(
            {
                "canonical": canonical,
                "members": ordered_members,
                "value_support": value_support,
                "positions": positions,
                "bbox": (top, left, bottom, right),
                "element_ref": (
                    "measurement.distinct_visual_element:"
                    f"{stable_digest(value_support)[:20]}"
                ),
            }
        )
    grouped.sort(key=lambda item: _descriptor_order_key(item["canonical"]))
    total_measurement_count = len(grouped)
    grouped = grouped[:maximum_measurements]

    supports = tuple(item["positions"] for item in grouped)
    containing_indices = tuple(
        tuple(
            other_index
            for other_index, other_support in enumerate(supports)
            if index != other_index and support < other_support
        )
        for index, support in enumerate(supports)
    )
    parent_indices = tuple(
        tuple(
            candidate_index
            for candidate_index in containers
            if not any(
                support < supports[between_index] < supports[candidate_index]
                for between_index in containers
                if between_index != candidate_index
            )
        )
        for support, containers in zip(supports, containing_indices)
    )
    child_indices = tuple(
        tuple(
            child_index
            for child_index, parents in enumerate(parent_indices)
            if index in parents
        )
        for index in range(len(grouped))
    )
    contained_indices = tuple(
        tuple(
            other_index
            for other_index, other_support in enumerate(supports)
            if index != other_index and other_support < support
        )
        for index, support in enumerate(supports)
    )

    measurements: list[FrozenMap] = []
    for index, item in enumerate(grouped):
        members = item["members"]
        value_support = item["value_support"]
        top, left, bottom, right = item["bbox"]
        element_ref = item["element_ref"]
        assert isinstance(members, tuple) and isinstance(value_support, tuple)
        assert isinstance(element_ref, str)
        values = tuple(sorted({value for _, _, value in value_support}))
        relative_positions = tuple(
            (row - top, col - left) for row, col, _ in value_support
        )
        relative_value_pattern = tuple(
            (row - top, col - left, value) for row, col, value in value_support
        )
        descriptor_refs = tuple(str(member["descriptor_ref"]) for member in members)
        family_refs = tuple(
            dict.fromkeys(
                f"measurement_family.{member['family']}" for member in members
            )
        )
        component_refs = tuple(
            str(member["component_ref"])
            for member in members
            if member["component_ref"] is not None
        )
        anchor_refs = tuple(
            sorted(
                {
                    str(member["anchor_component_ref"])
                    for member in members
                    if member["anchor_component_ref"] is not None
                }
            )
        )
        bbox_area = (bottom - top + 1) * (right - left + 1)
        measurements.append(
            FrozenMap(
                {
                    "element_ref": element_ref,
                    "canonical_component_ref": component_refs[0] if component_refs else None,
                    "alias_refs": descriptor_refs[1:],
                    "descriptor_refs": descriptor_refs,
                    "descriptor_family_refs": family_refs,
                    "anchor_component_refs": anchor_refs,
                    "value": values[0] if len(values) == 1 else None,
                    "values": values,
                    "value_count": len(values),
                    "bbox_top": top,
                    "bbox_left": left,
                    "bbox_bottom": bottom,
                    "bbox_right": right,
                    "height": bottom - top + 1,
                    "width": right - left + 1,
                    "area": len(value_support),
                    "bbox_area": bbox_area,
                    "fill_ppm": len(value_support) * 1_000_000 // bbox_area,
                    "boundary_contact_count": sum(
                        (
                            top == 0,
                            left == 0,
                            bottom == scene.frame.height - 1,
                            right == scene.frame.width - 1,
                        )
                    ),
                    "relative_shape_ref": (
                        "measurement.relative_shape:"
                        f"{stable_digest(relative_positions)[:20]}"
                    ),
                    "relative_value_pattern_ref": (
                        "measurement.relative_value_pattern:"
                        f"{stable_digest(relative_value_pattern)[:20]}"
                    ),
                    "parent_element_refs": tuple(
                        grouped[parent_index]["element_ref"]
                        for parent_index in parent_indices[index]
                    ),
                    "child_element_refs": tuple(
                        grouped[child_index]["element_ref"]
                        for child_index in child_indices[index]
                    ),
                    "containing_element_refs": tuple(
                        grouped[parent_index]["element_ref"]
                        for parent_index in containing_indices[index]
                    ),
                    "contained_element_refs": tuple(
                        grouped[child_index]["element_ref"]
                        for child_index in contained_indices[index]
                    ),
                }
            )
        )
    value_supports = tuple(
        tuple(item["value_support"])
        for item in grouped
    )
    return tuple(measurements), total_measurement_count, supports, value_supports


def _distinct_visual_element_pair_relations(
    elements: tuple[FrozenMap, ...],
    supports: tuple[frozenset[tuple[int, int]], ...],
    value_supports: tuple[tuple[tuple[int, int, int], ...], ...],
    *,
    frame_height: int,
    frame_width: int,
    maximum_measurements: int,
) -> tuple[tuple[FrozenMap, ...], int]:
    measurements: list[FrozenMap] = []
    hole_extent_counts: dict[tuple[int, int], int] = {}
    for element in elements:
        if "measurement_family.hole_sprite" not in element["descriptor_family_refs"]:
            continue
        extent = (int(element["height"]), int(element["width"]))
        hole_extent_counts[extent] = hole_extent_counts.get(extent, 0) + 1
    nearest_aligned_pairs = _nearest_aligned_side_pairs(elements)
    pair_indices = tuple(
        (first_index, second_index)
        for first_index in range(len(elements))
        for second_index in range(first_index + 1, len(elements))
    )
    exact_mask_candidates_by_pair: dict[
        tuple[int, int], tuple[FrozenMap, ...]
    ] = {}
    exact_multiscale_rotated_pairs: set[tuple[int, int]] = set()
    for first_index, second_index in pair_indices:
        first = elements[first_index]
        second = elements[second_index]
        first_extent = (int(first["height"]), int(first["width"]))
        second_extent = (int(second["height"]), int(second["width"]))
        if first_extent == second_extent:
            continue
        if not (set(first["values"]) & set(second["values"])):
            continue
        candidates = _exact_mask_transform_candidates(
            value_supports[first_index],
            value_supports[second_index],
        )
        if not candidates:
            continue
        exact_mask_candidates_by_pair[(first_index, second_index)] = candidates
        if any(
            int(candidate["first_isotropic_factor"])
            != int(candidate["second_isotropic_factor"])
            and int(candidate["quarter_turns_clockwise"]) % 4 != 0
            for candidate in candidates
        ):
            exact_multiscale_rotated_pairs.add((first_index, second_index))

    def pair_priority(indices: tuple[int, int]) -> tuple[object, ...]:
        first_index, second_index = indices
        first = elements[first_index]
        second = elements[second_index]
        first_extent = (int(first["height"]), int(first["width"]))
        second_extent = (int(second["height"]), int(second["width"]))
        first_hole = (
            "measurement_family.hole_sprite"
            in first["descriptor_family_refs"]
        )
        second_hole = (
            "measurement_family.hole_sprite"
            in second["descriptor_family_refs"]
        )
        unique_hole_pair = (
            first_hole
            and second_hole
            and first_extent == second_extent
            and hole_extent_counts.get(first_extent, 0) == 2
        )
        nearest_aligned = (
            (first_index, second_index) in nearest_aligned_pairs
            or (second_index, first_index) in nearest_aligned_pairs
        )
        same_value_pattern = (
            first["relative_value_pattern_ref"]
            == second["relative_value_pattern_ref"]
        )
        same_shape_and_extent = (
            first_extent == second_extent
            and first["relative_shape_ref"] == second["relative_shape_ref"]
        )
        return (
            indices not in exact_multiscale_rotated_pairs,
            not unique_hole_pair,
            not nearest_aligned,
            not same_value_pattern,
            not same_shape_and_extent,
            first_index,
            second_index,
        )

    for first_index, second_index in sorted(pair_indices, key=pair_priority)[
        :maximum_measurements
    ]:
            first = elements[first_index]
            second = elements[second_index]
            first_support = supports[first_index]
            second_support = supports[second_index]
            exact_mask_transform_candidates = exact_mask_candidates_by_pair.get(
                (first_index, second_index)
            ) or _exact_mask_transform_candidates(
                value_supports[first_index], value_supports[second_index]
            )
            first_ref = str(first["element_ref"])
            second_ref = str(second["element_ref"])
            first_height = int(first["height"])
            first_width = int(first["width"])
            second_height = int(second["height"])
            second_width = int(second["width"])
            first_bbox_area = int(first["bbox_area"])
            second_bbox_area = int(second["bbox_area"])
            first_edge_distance = _frame_edge_distance(first, frame_height, frame_width)
            second_edge_distance = _frame_edge_distance(
                second, frame_height, frame_width
            )
            first_center_distance = _frame_center_distance_x2(
                first, frame_height, frame_width
            )
            second_center_distance = _frame_center_distance_x2(
                second, frame_height, frame_width
            )
            first_hole_sprite = (
                "measurement_family.hole_sprite"
                in first["descriptor_family_refs"]
            )
            second_hole_sprite = (
                "measurement_family.hole_sprite"
                in second["descriptor_family_refs"]
            )
            same_extent_hole_count = hole_extent_counts.get(
                (first_height, first_width), 0
            ) if (first_height, first_width) == (second_height, second_width) else 0
            first_side_ref, first_reflected_side_ref = _bbox_side_refs(first, second)
            second_side_ref, second_reflected_side_ref = _bbox_side_refs(second, first)
            first_nearest_aligned = (
                first_index, second_index
            ) in nearest_aligned_pairs
            second_nearest_aligned = (
                second_index, first_index
            ) in nearest_aligned_pairs
            measurements.append(
                FrozenMap(
                    {
                        "relation_ref": (
                            "measurement.distinct_visual_element_pair_relation:"
                            f"{stable_digest((first_ref, second_ref))[:20]}"
                        ),
                        "first_element_ref": first_ref,
                        "second_element_ref": second_ref,
                        "first_bbox_top": int(first["bbox_top"]),
                        "first_bbox_left": int(first["bbox_left"]),
                        "first_bbox_bottom": int(first["bbox_bottom"]),
                        "first_bbox_right": int(first["bbox_right"]),
                        "second_bbox_top": int(second["bbox_top"]),
                        "second_bbox_left": int(second["bbox_left"]),
                        "second_bbox_bottom": int(second["bbox_bottom"]),
                        "second_bbox_right": int(second["bbox_right"]),
                        "first_height": first_height,
                        "first_width": first_width,
                        "first_area": int(first["area"]),
                        "first_bbox_area": first_bbox_area,
                        "first_fill_ppm": int(first["fill_ppm"]),
                        "second_height": second_height,
                        "second_width": second_width,
                        "second_area": int(second["area"]),
                        "second_bbox_area": second_bbox_area,
                        "second_fill_ppm": int(second["fill_ppm"]),
                        "first_oriented_extent_present": first_height != first_width,
                        "second_oriented_extent_present": second_height != second_width,
                        "first_larger_bbox_area": first_bbox_area > second_bbox_area,
                        "second_larger_bbox_area": second_bbox_area > first_bbox_area,
                        "same_height": first_height == second_height,
                        "same_width": first_width == second_width,
                        "same_extent": (
                            first_height == second_height
                            and first_width == second_width
                        ),
                        "same_area": int(first["area"]) == int(second["area"]),
                        "same_relative_shape": (
                            first["relative_shape_ref"]
                            == second["relative_shape_ref"]
                        ),
                        "same_relative_value_pattern": (
                            first["relative_value_pattern_ref"]
                            == second["relative_value_pattern_ref"]
                        ),
                        "exact_mask_transform_candidate_count": len(
                            exact_mask_transform_candidates
                        ),
                        "exact_mask_transform_candidates": (
                            exact_mask_transform_candidates
                        ),
                        "first_hole_sprite_descriptor_present": first_hole_sprite,
                        "second_hole_sprite_descriptor_present": second_hole_sprite,
                        "same_extent_hole_sprite_element_count": same_extent_hole_count,
                        "unique_same_extent_hole_sprite_pair": (
                            first_hole_sprite
                            and second_hole_sprite
                            and same_extent_hole_count == 2
                        ),
                        "first_relative_side_ref": first_side_ref,
                        "first_reflected_side_ref": first_reflected_side_ref,
                        "second_relative_side_ref": second_side_ref,
                        "second_reflected_side_ref": second_reflected_side_ref,
                        "first_nearest_aligned_side_candidate": first_nearest_aligned,
                        "second_nearest_aligned_side_candidate": second_nearest_aligned,
                        "first_contains_second": second_support < first_support,
                        "second_contains_first": first_support < second_support,
                        "support_overlap_present": bool(
                            first_support & second_support
                        ),
                        "orthogonal_adjacency_present": _supports_are_adjacent(
                            first_support, second_support, diagonal=False
                        ),
                        "diagonal_adjacency_present": _supports_are_adjacent(
                            first_support, second_support, diagonal=True
                        ),
                        "bbox_row_offset": (
                            int(second["bbox_top"]) - int(first["bbox_top"])
                        ),
                        "bbox_col_offset": (
                            int(second["bbox_left"]) - int(first["bbox_left"])
                        ),
                        "center_row_offset_x2": (
                            int(second["bbox_top"])
                            + int(second["bbox_bottom"])
                            - int(first["bbox_top"])
                            - int(first["bbox_bottom"])
                        ),
                        "center_col_offset_x2": (
                            int(second["bbox_left"])
                            + int(second["bbox_right"])
                            - int(first["bbox_left"])
                            - int(first["bbox_right"])
                        ),
                        "first_frame_edge_distance": first_edge_distance,
                        "second_frame_edge_distance": second_edge_distance,
                        "first_frame_center_distance_x2": first_center_distance,
                        "second_frame_center_distance_x2": second_center_distance,
                        "first_closer_to_frame_edge": (
                            first_edge_distance < second_edge_distance
                        ),
                        "second_closer_to_frame_edge": (
                            second_edge_distance < first_edge_distance
                        ),
                        "first_closer_to_frame_center": (
                            first_center_distance < second_center_distance
                        ),
                        "second_closer_to_frame_center": (
                            second_center_distance < first_center_distance
                        ),
                    }
                )
            )
    return tuple(measurements), len(pair_indices)


@lru_cache(maxsize=128)
def _exact_mask_transform_candidates(
    first_value_support: tuple[tuple[int, int, int], ...],
    second_value_support: tuple[tuple[int, int, int], ...],
) -> tuple[FrozenMap, ...]:
    """Enumerate exact finite-mask scale/quarter-turn relations without meaning."""

    first_values = {value for _, _, value in first_value_support}
    second_values = {value for _, _, value in second_value_support}
    measured: list[FrozenMap] = []
    for value in sorted(first_values & second_values):
        first_mask = _value_mask(first_value_support, value)
        second_mask = _value_mask(second_value_support, value)
        for first_factor, first_reduced in _exact_isotropic_reductions(first_mask):
            for second_factor, second_reduced in _exact_isotropic_reductions(
                second_mask
            ):
                for quarter_turns in range(4):
                    if _rotate_quarter_turns(first_reduced, quarter_turns) != second_reduced:
                        continue
                    measured.append(
                        FrozenMap(
                            {
                                "shared_value": value,
                                "first_isotropic_factor": first_factor,
                                "second_isotropic_factor": second_factor,
                                "quarter_turns_clockwise": quarter_turns,
                                "normalized_height": len(second_reduced),
                                "normalized_width": len(second_reduced[0]),
                                "first_normalized_mask_ref": (
                                    "measurement.normalized_mask:"
                                    f"{stable_digest(first_reduced)[:20]}"
                                ),
                                "second_normalized_mask_ref": (
                                    "measurement.normalized_mask:"
                                    f"{stable_digest(second_reduced)[:20]}"
                                ),
                            }
                        )
                    )
    return tuple(measured)


def _value_mask(
    value_support: tuple[tuple[int, int, int], ...],
    value: int,
) -> tuple[tuple[bool, ...], ...]:
    positions = {(row, col) for row, col, item_value in value_support if item_value == value}
    top = min(row for row, _ in positions)
    left = min(col for _, col in positions)
    bottom = max(row for row, _ in positions)
    right = max(col for _, col in positions)
    return tuple(
        tuple((row, col) in positions for col in range(left, right + 1))
        for row in range(top, bottom + 1)
    )


@lru_cache(maxsize=128)
def _exact_isotropic_reductions(
    mask: tuple[tuple[bool, ...], ...],
) -> tuple[tuple[int, tuple[tuple[bool, ...], ...]], ...]:
    height = len(mask)
    width = len(mask[0])
    reductions: list[tuple[int, tuple[tuple[bool, ...], ...]]] = []
    for factor in range(1, min(height, width) + 1):
        if height % factor or width % factor:
            continue
        reduced_rows: list[tuple[bool, ...]] = []
        exact = True
        for top in range(0, height, factor):
            reduced_row: list[bool] = []
            for left in range(0, width, factor):
                block = {
                    mask[row][col]
                    for row in range(top, top + factor)
                    for col in range(left, left + factor)
                }
                if len(block) != 1:
                    exact = False
                    break
                reduced_row.append(next(iter(block)))
            if not exact:
                break
            reduced_rows.append(tuple(reduced_row))
        if exact:
            reductions.append((factor, tuple(reduced_rows)))
    return tuple(reductions)


def _rotate_quarter_turns(
    mask: tuple[tuple[bool, ...], ...],
    quarter_turns: int,
) -> tuple[tuple[bool, ...], ...]:
    rotated = mask
    for _ in range(quarter_turns % 4):
        rotated = tuple(
            tuple(rotated[row][col] for row in range(len(rotated) - 1, -1, -1))
            for col in range(len(rotated[0]))
        )
    return rotated


def _bbox_side_refs(
    candidate: FrozenMap, target: FrozenMap
) -> tuple[str, str]:
    column_overlap = not (
        int(candidate["bbox_right"]) < int(target["bbox_left"])
        or int(candidate["bbox_left"]) > int(target["bbox_right"])
    )
    row_overlap = not (
        int(candidate["bbox_bottom"]) < int(target["bbox_top"])
        or int(candidate["bbox_top"]) > int(target["bbox_bottom"])
    )
    if column_overlap and int(candidate["bbox_bottom"]) < int(target["bbox_top"]):
        return "spatial_side.above", "spatial_side.below"
    if column_overlap and int(candidate["bbox_top"]) > int(target["bbox_bottom"]):
        return "spatial_side.below", "spatial_side.above"
    if row_overlap and int(candidate["bbox_right"]) < int(target["bbox_left"]):
        return "spatial_side.left", "spatial_side.right"
    if row_overlap and int(candidate["bbox_left"]) > int(target["bbox_right"]):
        return "spatial_side.right", "spatial_side.left"
    return "spatial_side.unseparated", "spatial_side.unseparated"


def _nearest_aligned_side_pairs(
    elements: tuple[FrozenMap, ...],
) -> frozenset[tuple[int, int]]:
    """Return every tied nearest oriented-small-element/large-surface pair."""

    measured: set[tuple[int, int]] = set()
    for target_index, target in enumerate(elements):
        by_side: dict[str, tuple[int, list[int]]] = {}
        for candidate_index, candidate in enumerate(elements):
            if candidate_index == target_index:
                continue
            side_ref, _ = _bbox_side_refs(candidate, target)
            if (
                side_ref == "spatial_side.unseparated"
                or int(candidate["height"]) == int(candidate["width"])
                or int(candidate["bbox_area"]) >= int(target["bbox_area"])
            ):
                continue
            if side_ref == "spatial_side.above":
                gap = int(target["bbox_top"]) - int(candidate["bbox_bottom"]) - 1
            elif side_ref == "spatial_side.below":
                gap = int(candidate["bbox_top"]) - int(target["bbox_bottom"]) - 1
            elif side_ref == "spatial_side.left":
                gap = int(target["bbox_left"]) - int(candidate["bbox_right"]) - 1
            else:
                gap = int(candidate["bbox_left"]) - int(target["bbox_right"]) - 1
            prior = by_side.get(side_ref)
            if prior is None or gap < prior[0]:
                by_side[side_ref] = (gap, [candidate_index])
            elif gap == prior[0]:
                prior[1].append(candidate_index)
        for _gap, candidate_indices in (by_side[__yf_order_key] for __yf_order_key in sorted(by_side)):
            measured.update(
                (candidate_index, target_index)
                for candidate_index in candidate_indices
            )
    return frozenset(measured)


def _is_nearest_aligned_side_candidate(
    candidate: FrozenMap,
    target: FrozenMap,
    elements: tuple[FrozenMap, ...],
) -> bool:
    side_ref, _ = _bbox_side_refs(candidate, target)
    if (
        side_ref == "spatial_side.unseparated"
        or int(candidate["height"]) == int(candidate["width"])
        or int(candidate["bbox_area"]) >= int(target["bbox_area"])
    ):
        return False

    def gap(element: FrozenMap) -> int | None:
        element_side_ref, _ = _bbox_side_refs(element, target)
        if (
            element_side_ref != side_ref
            or int(element["height"]) == int(element["width"])
            or int(element["bbox_area"]) >= int(target["bbox_area"])
        ):
            return None
        if side_ref == "spatial_side.above":
            return int(target["bbox_top"]) - int(element["bbox_bottom"]) - 1
        if side_ref == "spatial_side.below":
            return int(element["bbox_top"]) - int(target["bbox_bottom"]) - 1
        if side_ref == "spatial_side.left":
            return int(target["bbox_left"]) - int(element["bbox_right"]) - 1
        return int(element["bbox_left"]) - int(target["bbox_right"]) - 1

    candidate_gap = gap(candidate)
    eligible_gaps = tuple(
        measured_gap
        for element in elements
        if (measured_gap := gap(element)) is not None
    )
    return candidate_gap is not None and candidate_gap == min(eligible_gaps)


def _distinct_visual_element_repeated_pair_thin_relations(
    elements: tuple[FrozenMap, ...],
    supports: tuple[frozenset[tuple[int, int]], ...],
    *,
    maximum_measurements: int,
) -> tuple[tuple[FrozenMap, ...], int]:
    measurements: list[FrozenMap] = []
    relation_count = 0
    for first_index, first in enumerate(elements):
        for second_index in range(first_index + 1, len(elements)):
            second = elements[second_index]
            if (
                first["relative_shape_ref"] != second["relative_shape_ref"]
                or first["height"] != second["height"]
                or first["width"] != second["width"]
            ):
                continue
            for thin_index, thin in enumerate(elements):
                if thin_index in {first_index, second_index}:
                    continue
                thin_height = int(thin["height"])
                thin_width = int(thin["width"])
                if min(thin_height, thin_width) * 4 > max(thin_height, thin_width):
                    continue
                first_support = supports[first_index]
                second_support = supports[second_index]
                thin_support = supports[thin_index]
                orthogonal_first = _supports_are_adjacent(
                    thin_support, first_support, diagonal=False
                )
                orthogonal_second = _supports_are_adjacent(
                    thin_support, second_support, diagonal=False
                )
                diagonal_first = _supports_are_adjacent(
                    thin_support, first_support, diagonal=True
                )
                diagonal_second = _supports_are_adjacent(
                    thin_support, second_support, diagonal=True
                )
                overlap_first = bool(thin_support & first_support)
                overlap_second = bool(thin_support & second_support)
                if not (
                    orthogonal_first
                    or orthogonal_second
                    or diagonal_first
                    or diagonal_second
                    or overlap_first
                    or overlap_second
                    or thin_support < first_support
                    or thin_support < second_support
                    or first_support < thin_support
                    or second_support < thin_support
                ):
                    continue
                relation_count += 1
                if len(measurements) >= maximum_measurements:
                    continue
                first_ref = str(first["element_ref"])
                second_ref = str(second["element_ref"])
                thin_ref = str(thin["element_ref"])
                measurements.append(
                    FrozenMap(
                        {
                            "relation_ref": (
                                "measurement.repeated_pair_thin_local_relation:"
                                f"{stable_digest((first_ref, second_ref, thin_ref))[:20]}"
                            ),
                            "first_repeated_element_ref": first_ref,
                            "second_repeated_element_ref": second_ref,
                            "local_thin_element_ref": thin_ref,
                            "repeated_same_extent": True,
                            "repeated_same_relative_shape": True,
                            "local_thin_extent_present": True,
                            "local_orthogonally_adjacent_to_first": orthogonal_first,
                            "local_orthogonally_adjacent_to_second": orthogonal_second,
                            "local_diagonally_adjacent_to_first": diagonal_first,
                            "local_diagonally_adjacent_to_second": diagonal_second,
                            "local_overlaps_first": overlap_first,
                            "local_overlaps_second": overlap_second,
                            "local_contains_first": first_support < thin_support,
                            "local_contains_second": second_support < thin_support,
                            "first_contains_local": thin_support < first_support,
                            "second_contains_local": thin_support < second_support,
                            "local_to_first_row_offset": (
                                int(thin["bbox_top"]) - int(first["bbox_top"])
                            ),
                            "local_to_first_col_offset": (
                                int(thin["bbox_left"]) - int(first["bbox_left"])
                            ),
                            "local_to_second_row_offset": (
                                int(thin["bbox_top"]) - int(second["bbox_top"])
                            ),
                            "local_to_second_col_offset": (
                                int(thin["bbox_left"]) - int(second["bbox_left"])
                            ),
                        }
                    )
                )
    return tuple(measurements), relation_count


@lru_cache(maxsize=8192)
def _supports_are_adjacent(
    first: frozenset[tuple[int, int]],
    second: frozenset[tuple[int, int]],
    *,
    diagonal: bool,
) -> bool:
    # Adjacency is symmetric. Probe the smaller immutable support; a thin
    # enclosing region can contain thousands of pixels while its peer has four.
    # This preserves the exact relation, including diagonal-only contact.
    if len(first) > len(second):
        first, second = second, first
    offsets = ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else (
        (-1, 0),
        (0, -1),
        (0, 1),
        (1, 0),
    )
    return any(
        (row + delta_row, col + delta_col) in second
        for row, col in first
        for delta_row, delta_col in offsets
    )


def _frame_edge_distance(
    element: FrozenMap,
    frame_height: int,
    frame_width: int,
) -> int:
    return min(
        int(element["bbox_top"]),
        int(element["bbox_left"]),
        frame_height - 1 - int(element["bbox_bottom"]),
        frame_width - 1 - int(element["bbox_right"]),
    )


def _frame_center_distance_x2(
    element: FrozenMap,
    frame_height: int,
    frame_width: int,
) -> int:
    return abs(
        int(element["bbox_top"])
        + int(element["bbox_bottom"])
        - (frame_height - 1)
    ) + abs(
        int(element["bbox_left"])
        + int(element["bbox_right"])
        - (frame_width - 1)
    )


def _bbox_value_support(
    scene: VisualSceneDescription,
    bbox: BoundingBox,
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (row, col, scene.frame.rows[row][col])
        for row in range(bbox.top, bbox.bottom + 1)
        for col in range(bbox.left, bbox.right + 1)
    )


def _descriptor_order_key(descriptor: object) -> tuple[object, ...]:
    assert isinstance(descriptor, dict)
    value_support = descriptor["value_support"]
    assert isinstance(value_support, tuple)
    rows = tuple(item[0] for item in value_support)
    cols = tuple(item[1] for item in value_support)
    return (
        descriptor["family_order"],
        min(rows),
        min(cols),
        max(rows),
        max(cols),
        descriptor["descriptor_ref"],
    )


def _frame_enclosure_tuple(
    component: ComponentDescription,
    frame_height: int,
    frame_width: int,
    frame_area: int,
) -> tuple[str, int, int, int]:
    bbox_area = component.bbox.height * component.bbox.width
    return (
        component.component_id,
        _boundary_contact_count(component, frame_height, frame_width),
        bbox_area * 1_000_000 // frame_area,
        component.area * 1_000_000 // frame_area,
    )


def _boundary_contact_count(
    component: ComponentDescription,
    frame_height: int,
    frame_width: int,
) -> int:
    return sum(
        (
            component.bbox.top == 0,
            component.bbox.left == 0,
            component.bbox.bottom == frame_height - 1,
            component.bbox.right == frame_width - 1,
        )
    )


def _has_thin_extent(component: ComponentDescription) -> bool:
    minor = min(component.bbox.height, component.bbox.width)
    major = max(component.bbox.height, component.bbox.width)
    return minor * 4 <= major
