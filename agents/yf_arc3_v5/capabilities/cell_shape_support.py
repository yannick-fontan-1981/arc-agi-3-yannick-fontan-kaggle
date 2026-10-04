"""Bounded raster facts about repeated same-value shapes and grid cells.

This measures geometry only. It neither names a color nor commits a grid.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from agents.yf_arc3_v5.capabilities.contracts import (
    BoundingBox,
    FrameGrid,
    PeriodicCellGridResult,
)


@dataclass(frozen=True, slots=True)
class RepeatedShapeFamily:
    value: int
    height: int
    width: int
    area: int
    occurrence_count: int
    origins: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class GridShapeFit:
    grid_ref: str
    exact_family_counts: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class RepeatedShapeCellSupport:
    families: tuple[RepeatedShapeFamily, ...]
    grid_fits: tuple[GridShapeFit, ...]
    ignored_large_component_count: int
    enumeration_truncated: bool


@dataclass(frozen=True, slots=True)
class ShapeLatticeGeometry:
    family_index: int
    row_offset: int
    col_offset: int
    row_pitch: int
    col_pitch: int
    row_gap: int
    col_gap: int
    separator_value_counts: tuple[tuple[int, int], ...]
    embedded_full_footprint_windows: int
    repeated_multivalue_windows: int


@dataclass(frozen=True, slots=True)
class ShapeLatticeGeometries:
    geometries: tuple[ShapeLatticeGeometry, ...]
    enumeration_truncated: bool


def measure_repeated_shape_cell_support(
    frame: FrameGrid,
    grids: PeriodicCellGridResult,
    *,
    max_frame_pixels: int = 4096,
    max_component_area: int = 64,
    max_components: int = 64,
    max_families: int = 16,
) -> RepeatedShapeCellSupport:
    """Count exact cell-bbox fits for every repeated small color/shape family.

    Four-neighbor components are built only on demand. Pixels and masks are
    transient; the returned facts have no raw pixel coordinates or role labels.
    Bounds fail closed rather than silently choosing a subset of components.
    """

    if not (
        1 <= max_frame_pixels <= 4096
        and 1 <= max_component_area <= 64
        and 1 <= max_components <= 64
        and 1 <= max_families <= 16
    ):
        raise ValueError("shape support bounds must stay within hard limits")
    empty = RepeatedShapeCellSupport((), (), 0, True)
    height, width = frame.height, frame.width
    if height * width > max_frame_pixels:
        return empty

    seen = bytearray(height * width)
    components: list[tuple[tuple[int, tuple[tuple[int, int], ...]], BoundingBox, int]] = []
    ignored_large = 0
    for row in range(height):
        for col in range(width):
            origin = row * width + col
            if seen[origin]:
                continue
            value = frame.rows[row][col]
            seen[origin] = 1
            pending = [origin]
            positions: list[tuple[int, int]] = []
            area = 0
            top = bottom = row
            left = right = col
            while pending:
                current_row, current_col = divmod(pending.pop(), width)
                area += 1
                if area <= max_component_area:
                    positions.append((current_row, current_col))
                top = min(top, current_row)
                bottom = max(bottom, current_row)
                left = min(left, current_col)
                right = max(right, current_col)
                for neighbor_row, neighbor_col in (
                    (current_row - 1, current_col),
                    (current_row + 1, current_col),
                    (current_row, current_col - 1),
                    (current_row, current_col + 1),
                ):
                    if not (0 <= neighbor_row < height and 0 <= neighbor_col < width):
                        continue
                    neighbor = neighbor_row * width + neighbor_col
                    if seen[neighbor] or frame.rows[neighbor_row][neighbor_col] != value:
                        continue
                    seen[neighbor] = 1
                    pending.append(neighbor)
            if area > max_component_area:
                ignored_large += 1
                continue
            if area == 1:
                continue
            signature = (value, tuple(sorted(
                (pixel_row - top, pixel_col - left)
                for pixel_row, pixel_col in positions
            )))
            components.append((
                signature,
                BoundingBox(top=top, left=left, bottom=bottom, right=right),
                area,
            ))
            if len(components) > max_components:
                return RepeatedShapeCellSupport((), (), ignored_large, True)

    counts = Counter(signature for signature, _, _ in components)
    families: list[RepeatedShapeFamily] = []
    family_signatures: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    for signature, bbox, area in components:
        if counts[signature] < 2 or signature in family_signatures:
            continue
        family_signatures.append(signature)
        families.append(RepeatedShapeFamily(
            value=signature[0], height=bbox.height, width=bbox.width,
            area=area, occurrence_count=counts[signature],
            origins=tuple(
                (component_bbox.top, component_bbox.left)
                for component_signature, component_bbox, _ in components
                if component_signature == signature
            ),
        ))
        if len(families) > max_families:
            return RepeatedShapeCellSupport((), (), ignored_large, True)

    family_indexes = {
        signature: index for index, signature in enumerate(family_signatures)
    }
    grid_fits = []
    for grid in grids.candidates:
        cell_boxes = {cell.bbox for cell in grid.cells}
        exact = Counter(
            family_indexes[signature]
            for signature, bbox, _ in components
            if signature in family_indexes and bbox in cell_boxes
        )
        grid_fits.append(GridShapeFit(
            grid_ref=grid.candidate_ref,
            exact_family_counts=tuple(sorted(exact.items())),
        ))
    return RepeatedShapeCellSupport(
        families=tuple(families), grid_fits=tuple(grid_fits),
        ignored_large_component_count=ignored_large,
        enumeration_truncated=grids.enumeration_truncated,
    )


def measure_shape_lattice_geometries(
    frame: FrameGrid,
    families: tuple[RepeatedShapeFamily, ...],
    *,
    max_pitch: int = 16,
    max_geometries: int = 16,
) -> ShapeLatticeGeometries:
    """Enumerate every bounded lattice consistent with repeated shape origins.

    Separator content and full-sized repeated motifs inside a proposed gap are
    measurements, not an ordering or a commitment to any lattice.
    """

    if not (1 <= max_pitch <= 32 and 1 <= max_geometries <= 32):
        raise ValueError("lattice bounds must stay within hard limits")
    geometries: list[ShapeLatticeGeometry] = []
    for family_index, family in enumerate(families):
        if len(family.origins) < 2:
            continue
        row_origins = {origin[0] for origin in family.origins}
        col_origins = {origin[1] for origin in family.origins}
        for row_pitch in range(family.height, max_pitch + 1):
            row_residues = {row % row_pitch for row in row_origins}
            if len(row_residues) != 1:
                continue
            for col_pitch in range(family.width, max_pitch + 1):
                col_residues = {col % col_pitch for col in col_origins}
                if len(col_residues) != 1:
                    continue
                row_offset = next(iter(row_residues))
                col_offset = next(iter(col_residues))
                if (
                    row_offset + family.height > frame.height
                    or col_offset + family.width > frame.width
                ):
                    continue
                span_bottom = (
                    row_offset
                    + ((frame.height - row_offset - family.height) // row_pitch)
                    * row_pitch
                    + family.height
                )
                span_right = (
                    col_offset
                    + ((frame.width - col_offset - family.width) // col_pitch)
                    * col_pitch
                    + family.width
                )
                separator_values = Counter(
                    frame.rows[row][col]
                    for row in range(row_offset, span_bottom)
                    for col in range(col_offset, span_right)
                    if (
                        (row - row_offset) % row_pitch >= family.height
                        or (col - col_offset) % col_pitch >= family.width
                    )
                )
                embedded_patterns = Counter(
                    tuple(
                        tuple(source_row[col:col + family.width])
                        for source_row in frame.rows[
                            inner_top:inner_top + family.height
                        ]
                    )
                    for row_start in range(row_offset, span_bottom, row_pitch)
                    for inner_top in range(
                        row_start + family.height,
                        min(
                            row_start + row_pitch - family.height + 1,
                            span_bottom - family.height + 1,
                        ),
                    )
                    for col in range(
                        col_offset,
                        span_right - family.width + 1,
                        col_pitch,
                    )
                )
                geometries.append(ShapeLatticeGeometry(
                    family_index=family_index,
                    row_offset=row_offset,
                    col_offset=col_offset,
                    row_pitch=row_pitch,
                    col_pitch=col_pitch,
                    row_gap=row_pitch - family.height,
                    col_gap=col_pitch - family.width,
                    separator_value_counts=tuple(sorted(separator_values.items())),
                    embedded_full_footprint_windows=sum(embedded_patterns.values()),
                    repeated_multivalue_windows=sum(
                        count for pattern, count in sorted(embedded_patterns.items())
                        if count > 1
                        and len({pixel for line in pattern for pixel in line}) > 1
                    ),
                ))
                if len(geometries) > max_geometries:
                    return ShapeLatticeGeometries((), True)
    return ShapeLatticeGeometries(tuple(geometries), False)
