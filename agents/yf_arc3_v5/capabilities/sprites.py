"""Exact block/zone/sprite decomposition without semantic role assignment."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache

from agents.yf_arc3_v5.capabilities.components import detect_components
from agents.yf_arc3_v5.capabilities.frame import (
    enumerate_periodic_cell_grids,
    materialize_retained_periodic_cell_grid,
)
from agents.yf_arc3_v5.capabilities.tile_lattice import exact_bordered_tile_lattices
from agents.yf_arc3_v5.capabilities.contracts import (
    BoundingBox,
    BlockGridSceneDescription,
    ComponentDescription,
    ComponentExtractionInput,
    ComponentExtractionResult,
    PeriodicCellGridInput,
    PeriodicCellGridResult,
    VisualSceneDescription,
    VisualSceneInput,
    VisualSpriteDescription,
    HoleSpriteDescription,
)


_MAXIMUM_HOLE_SPRITES = 256


def describe_block_grid_scene(value: VisualSceneInput) -> BlockGridSceneDescription:
    """Measure only 4-connected blocks and candidate grids on explicit demand.

    This is a descriptive capability, not an eligibility decision.  SRC/DRM
    must request this scope; omitted families are not fields of this model and
    therefore cannot be mistaken for evidence that such objects do not exist.
    """

    blocks = detect_components(
        ComponentExtractionInput(frame=value.frame, connectivity=4)
    )
    if value.retained_grid_geometry is not None:
        periodic_cell_grids = materialize_retained_periodic_cell_grid(
            value.frame,
            value.retained_grid_geometry,
            separator_consistency_ppm=value.retained_grid_separator_consistency_ppm,
        )
    else:
        periodic_cell_grids = enumerate_periodic_cell_grids(
            PeriodicCellGridInput(
                frame=value.frame,
                action_conditioned_grid_evidence=value.action_conditioned_grid_evidence,
            )
        )
    frame_enclosing_components = tuple(
        component.component_id
        for component in blocks.components
        if _meets_frame_enclosure_measurement(component, value)
    )
    return BlockGridSceneDescription(
        frame=value.frame,
        blocks=blocks,
        periodic_cell_grids=periodic_cell_grids,
        frame_enclosing_component_refs=frame_enclosing_components,
        hole_sprite_enumeration_pruned_by_periodic_partition=(
            _periodic_partition_prunes_hole_sprite_enumeration(
                blocks, periodic_cell_grids
            )
        ),
    )


def _describe_glyph_sprites(
    value: VisualSceneInput,
    blocks: ComponentExtractionResult,
    zones: ComponentExtractionResult | None = None,
) -> tuple[tuple[str, ...], tuple[VisualSpriteDescription, ...]]:
    """Build the exact block/sprite projection without unrelated grid families."""

    frame_enclosing_components = tuple(
        anchor.component_id
        for anchor in blocks.components
        if _meets_frame_enclosure_measurement(anchor, value)
    )
    frame_enclosing_component_set = frozenset(frame_enclosing_components)
    block_position_index = _component_position_index(blocks.components)
    # Sprite membership is defined against the same 4-connected block index
    # as the full visual scene.  No role, candidate, or action is selected.
    block_order = {
        component.component_id: index
        for index, component in enumerate(blocks.components)
    }
    zone_position_index = (
        _component_position_index(zones.components)
        if zones is not None
        else None
    )
    zone_order = (
        {
            component.component_id: index
            for index, component in enumerate(zones.components)
        }
        if zones is not None
        else None
    )
    provisional: list[dict[str, object]] = []
    for anchor in blocks.components:
        if anchor.component_id in frame_enclosing_component_set:
            continue
        pattern = _crop(value, anchor.bbox)
        members = _component_refs_in_bbox(
            block_position_index,
            anchor.bbox,
            order=block_order,
        )
        zone_members = (
            _component_refs_in_bbox(
                zone_position_index,
                anchor.bbox,
                order=zone_order,
            )
            if zone_position_index is not None and zone_order is not None
            else ()
        )
        provisional.append(
            {
                "anchor": anchor,
                "pattern": pattern,
                "members": members,
                "zone_members": zone_members,
            }
        )
    pattern_counts = Counter(item["pattern"] for item in provisional)
    sprites = tuple(
        _sprite_description(item, pattern_counts[item["pattern"]])
        for item in provisional
    )
    return frame_enclosing_components, sprites


@lru_cache(maxsize=32)
def describe_visual_glyphs(
    value: VisualSceneInput,
) -> tuple[ComponentExtractionResult, tuple[VisualSpriteDescription, ...]]:
    """Cache only the glyph facts needed by point-conditioned delta measurement."""

    blocks = detect_components(
        ComponentExtractionInput(frame=value.frame, connectivity=4)
    )
    zones = detect_components(
        ComponentExtractionInput(frame=value.frame, connectivity=8)
    )
    _enclosing, sprites = _describe_glyph_sprites(value, blocks, zones)
    return blocks, sprites


@lru_cache(maxsize=32)
def describe_visual_scene(value: VisualSceneInput) -> VisualSceneDescription:
    """Keep exhaustive blocks/zones and object-scale exact sprite crops.

    Components meeting the exact frame-enclosure measurement remain available
    as blocks and zones.  They are not duplicated into the bounded object-scale
    crop projection; this exclusion assigns no semantic role.
    """

    blocks = detect_components(
        ComponentExtractionInput(frame=value.frame, connectivity=4)
    )
    zones = detect_components(
        ComponentExtractionInput(frame=value.frame, connectivity=8)
    )
    frame_enclosing_components, sprites = _describe_glyph_sprites(
        value,
        blocks,
        zones,
    )
    frame_enclosing_component_set = frozenset(frame_enclosing_components)
    periodic_cell_grids = (
        materialize_retained_periodic_cell_grid(
            value.frame,
            value.retained_grid_geometry,
            separator_consistency_ppm=value.retained_grid_separator_consistency_ppm,
        )
        if value.retained_grid_geometry is not None
        else enumerate_periodic_cell_grids(
            PeriodicCellGridInput(
                frame=value.frame,
                action_conditioned_grid_evidence=value.action_conditioned_grid_evidence,
            )
        )
    )
    # A periodic alternative dominates the component perspective only when its
    # exact cell partition is strictly smaller.  Mere raster periodicity is not
    # grid recognition: many ordinary scenes license bounded grid candidates.
    # Once the partition is structurally smaller, hole complements over every
    # component duplicate that perspective and grow quadratically.
    periodic_grid_dominates_components = (
        _periodic_partition_prunes_hole_sprite_enumeration(
            blocks, periodic_cell_grids
        )
    )
    all_hole_sprites = (
        ()
        if periodic_grid_dominates_components
        else _enumerate_hole_sprites(value, blocks.components)
    )
    # Full exact tiles already carry every internal pixel and component. A
    # complement wholly inside one such tile is a duplicate decomposition,
    # not another independent sprite candidate. Preserve regions spanning
    # cells and every ambiguous/non-exact lattice perspective.
    bordered_lattices = (
        exact_bordered_tile_lattices(value.frame, maximum_pitch=16)
        if all_hole_sprites and len(periodic_cell_grids.candidates) == 1
        else ()
    )
    if len(bordered_lattices) == 1 and len(periodic_cell_grids.candidates) == 1:
        grid = periodic_cell_grids.candidates[0]
        if (grid.row_offset, grid.col_offset, grid.row_pitch, grid.col_pitch) == bordered_lattices[0]:
            all_hole_sprites = tuple(
                hole for hole in all_hole_sprites
                if ((hole.bbox.top - grid.row_offset) // grid.row_pitch
                    != (hole.bbox.bottom - grid.row_offset) // grid.row_pitch)
                or ((hole.bbox.left - grid.col_offset) // grid.col_pitch
                    != (hole.bbox.right - grid.col_offset) // grid.col_pitch)
            )
    hole_sprites = all_hole_sprites[:_MAXIMUM_HOLE_SPRITES]
    return VisualSceneDescription(
        frame=value.frame,
        blocks=blocks,
        zones=zones,
        sprites=sprites,
        hole_sprites=hole_sprites,
        hole_sprite_measurement_count=len(all_hole_sprites),
        hole_sprite_enumeration_truncated=(
            len(all_hole_sprites) > _MAXIMUM_HOLE_SPRITES
        ),
        periodic_cell_grids=periodic_cell_grids,
        frame_enclosing_component_refs=frame_enclosing_components,
    )


def _periodic_partition_prunes_hole_sprite_enumeration(
    blocks: ComponentExtractionResult,
    periodic_cell_grids: PeriodicCellGridResult,
) -> bool:
    """Reuse the full-scene exact cardinality gate without asserting no holes."""

    # Competing partitions are alternatives, not an attested replacement of
    # the support perspective. One coarse alternative must not erase objects
    # still needed to compare the other lattices or recall composed bearers.
    return len(periodic_cell_grids.candidates) == 1 and any(
        len(candidate.cells) < len(blocks.components)
        for candidate in periodic_cell_grids.candidates
    )


def _enumerate_hole_sprites(
    value: VisualSceneInput,
    components: tuple[ComponentDescription, ...],
) -> tuple[HoleSpriteDescription, ...]:
    """Enumerate exact 4-connected component complements enclosed by support."""

    measured: list[HoleSpriteDescription] = []
    for component in components:
        box = component.bbox
        enclosing_positions = {
            (row, col)
            for row in range(box.top, box.bottom + 1)
            for col in range(box.left, box.right + 1)
        }
        remaining = enclosing_positions.difference(component.pixels)
        regions: list[tuple[tuple[int, int], ...]] = []
        while remaining:
            seed = min(remaining)
            remaining.remove(seed)
            frontier = [seed]
            region: list[tuple[int, int]] = []
            touches_boundary = False
            while frontier:
                row, col = frontier.pop()
                region.append((row, col))
                touches_boundary = touches_boundary or (
                    row == box.top
                    or col == box.left
                    or row == box.bottom
                    or col == box.right
                )
                for neighbor in (
                    (row - 1, col),
                    (row, col - 1),
                    (row, col + 1),
                    (row + 1, col),
                ):
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        frontier.append(neighbor)
            if not touches_boundary:
                regions.append(tuple(sorted(region)))
        regions.sort(key=lambda region: (region[0], len(region), region))
        for region_index, pixels in enumerate(regions):
            rows = tuple(row for row, _ in pixels)
            cols = tuple(col for _, col in pixels)
            values = {value.frame.rows[row][col] for row, col in pixels}
            measured.append(
                HoleSpriteDescription(
                    hole_sprite_id=(
                        f"hole_sprite.enclosed:{component.component_id}:{region_index}"
                    ),
                    enclosing_component_ref=component.component_id,
                    bbox=BoundingBox(
                        top=min(rows),
                        left=min(cols),
                        bottom=max(rows),
                        right=max(cols),
                    ),
                    pixels=pixels,
                    value_count=len(values),
                )
            )
    return tuple(measured)


def _sprite_description(
    item: dict[str, object],
    occurrence_count: int,
) -> VisualSpriteDescription:
    anchor = item["anchor"]
    assert isinstance(anchor, ComponentDescription)
    pattern = item["pattern"]
    assert isinstance(pattern, tuple)
    members = item["members"]
    zone_members = item["zone_members"]
    assert isinstance(members, tuple) and isinstance(zone_members, tuple)
    colors = Counter(pixel for row in pattern for pixel in row)
    return VisualSpriteDescription(
        sprite_id=f"sprite.crop.{anchor.component_id}",
        anchor_component_ref=anchor.component_id,
        bbox=anchor.bbox,
        pattern=pattern,
        member_component_refs=members,
        member_zone_refs=zone_members,
        color_histogram=tuple(sorted(colors.items())),
        internal_component_count=len(members) - 1,
        pattern_occurrence_count=occurrence_count,
        exact_dezoom_factors=_exact_dezoom_factors(pattern),
        touches_frame_boundary=anchor.touches_frame_boundary,
    )


def _crop(value: VisualSceneInput, bbox: BoundingBox) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(value.frame.rows[row][bbox.left : bbox.right + 1])
        for row in range(bbox.top, bbox.bottom + 1)
    )


def _component_position_index(
    components: tuple[ComponentDescription, ...],
) -> dict[tuple[int, int], str]:
    return {
        position: component.component_id
        for component in components
        for position in component.pixels
    }


def _component_refs_in_bbox(
    position_index: dict[tuple[int, int], str],
    bbox: BoundingBox,
    *,
    order: dict[str, int],
) -> tuple[str, ...]:
    refs = {
        position_index[(row, col)]
        for row in range(bbox.top, bbox.bottom + 1)
        for col in range(bbox.left, bbox.right + 1)
    }
    return tuple(
        sorted(refs, key=order.__getitem__)
    )


def _meets_frame_enclosure_measurement(
    component: ComponentDescription,
    value: VisualSceneInput,
) -> bool:
    frame_area = value.frame.height * value.frame.width
    bbox_area = component.bbox.height * component.bbox.width
    boundary_contacts = sum(
        (
            component.bbox.top == 0,
            component.bbox.left == 0,
            component.bbox.bottom == value.frame.height - 1,
            component.bbox.right == value.frame.width - 1,
        )
    )
    return bool(
        frame_area >= 64
        and component.touches_frame_boundary
        and boundary_contacts >= 2
        and bbox_area * 2 >= frame_area
        and component.area * 2 >= frame_area
    )


def _exact_dezoom_factors(pattern: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    height = len(pattern)
    width = len(pattern[0])
    factors: list[int] = []
    for factor in range(2, min(height, width) + 1):
        if height % factor or width % factor:
            continue
        if all(
            len(
                {
                    pattern[top + delta_row][left + delta_col]
                    for delta_row in range(factor)
                    for delta_col in range(factor)
                }
            )
            == 1
            for top in range(0, height, factor)
            for left in range(0, width, factor)
        ):
            factors.append(factor)
    return tuple(factors)


__all__ = ("describe_visual_scene",)
