"""Pure frame, difference, scale, and grid evidence calculations."""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache, reduce
import hashlib
import heapq
from threading import RLock

from agents.yf_arc3_v5.logos.types import FrozenMap, canonical_json, stable_digest
from agents.yf_arc3_v5.capabilities.tile_lattice import (
    exact_bordered_tile_lattices, exact_joint_tile_lattices,
)

from agents.yf_arc3_v5.capabilities.contracts import (
    BoundingBox,
    FrameDifferenceInput,
    FrameDifferenceResult,
    FrameGrid,
    FrameNormalizationInput,
    FrameNormalizationResult,
    GridPartitionCandidate,
    GridPartitionInput,
    GridPartitionResult,
    PeriodicCellDescription,
    PeriodicCellGridCandidate,
    PeriodicCellGridInput,
    PeriodicCellGridResult,
    PaletteCanonicalizationInput,
    PaletteCanonicalizationResult,
    PixelChange,
    SceneRebindingMeasurementInput,
    SceneRebindingMeasurements,
    UniformBlockReductionInput,
    UniformBlockReductionResult,
)


def _largest_two_counts(values: Iterable[int]) -> tuple[int, int]:
    """Order-independent top two, including two occurrences of the maximum."""

    def retain(pair: tuple[int, int], count: int) -> tuple[int, int]:
        first, second = pair
        if count >= first:
            return count, first
        if count > second:
            return first, count
        return pair

    return reduce(retain, values, (0, 0))


def normalize_frame(value: FrameNormalizationInput) -> FrameNormalizationResult:
    return FrameNormalizationResult(
        frame=value.frames[-1],
        transition_frames=value.frames[:-1],
        source_frame_count=len(value.frames),
    )


def difference_frames(value: FrameDifferenceInput) -> FrameDifferenceResult:
    if (value.before.height, value.before.width) != (
        value.after.height,
        value.after.width,
    ):
        raise ValueError("frame difference requires equal dimensions")
    changes = tuple(
        PixelChange(row=row, col=col, before=before, after=after)
        for row, (before_row, after_row) in enumerate(
            zip(value.before.rows, value.after.rows, strict=True)
        )
        for col, (before, after) in enumerate(zip(before_row, after_row, strict=True))
        if before != after
    )
    changed_bbox = None
    if changes:
        changed_bbox = BoundingBox(
            top=min(change.row for change in changes),
            left=min(change.col for change in changes),
            bottom=max(change.row for change in changes),
            right=max(change.col for change in changes),
        )
    changed_count = len(changes)
    return FrameDifferenceResult(
        changes=changes,
        changed_count=changed_count,
        unchanged_count=value.before.height * value.before.width - changed_count,
        changed_bbox=changed_bbox,
        action_ref=value.action_ref,
    )


def canonicalize_palette(
    value: PaletteCanonicalizationInput,
) -> PaletteCanonicalizationResult:
    source_to_canonical: dict[int, int] = {}
    next_value = 0
    canonical_rows: list[tuple[int, ...]] = []
    for row in value.frame.rows:
        canonical_row: list[int] = []
        for source in row:
            if source not in source_to_canonical:
                source_to_canonical[source] = next_value
                next_value += 1
            canonical_row.append(source_to_canonical[source])
        canonical_rows.append(tuple(canonical_row))
    return PaletteCanonicalizationResult(
        canonical_frame=FrameGrid(rows=tuple(canonical_rows)),
        source_to_canonical=tuple(source_to_canonical.items()),
    )


def measure_scene_rebinding(
    value: SceneRebindingMeasurementInput,
) -> SceneRebindingMeasurements:
    """Project bounded cross-scene structure and tracking measurements only."""

    established_tracking_count = sum(
        entity.identity_status == "established"
        for entity in value.current_tracking.entities
    )
    ambiguous_component_refs = {
        entity.component.component_id
        for entity in value.current_tracking.entities
        if entity.identity_status == "special" or entity.match_kind == "ambiguous"
    }
    ambiguous_component_refs.update(
        value.current_tracking.ambiguous_current_component_refs
    )
    ambiguous_tracking_count = len(ambiguous_component_refs)
    new_tracking_count = sum(
        entity.identity_status == "new"
        for entity in value.current_tracking.entities
    )
    prior_frame = value.prior_structure.canonical_frame
    current_frame = value.current_structure.canonical_frame
    return SceneRebindingMeasurements(
        descriptive_facts=FrozenMap(
            {
                "measurement_ref": (
                    "measurement.scene_rebinding."
                    + stable_digest(
                        (
                            prior_frame.rows,
                            current_frame.rows,
                            value.current_tracking.frame_ref,
                            value.current_frame_ref,
                            value.handoff_claim_ref,
                            value.scope_ref,
                            value.transported_reference_refs,
                        )
                    )[:16]
                ),
                "handoff_claim_ref": value.handoff_claim_ref,
                "scope_ref": value.scope_ref,
                "tracking_frame_ref": value.current_tracking.frame_ref,
                "tracking_frame_matches_current": (
                    value.current_tracking.frame_ref == value.current_frame_ref
                ),
                "canonical_dimensions_match": (
                    prior_frame.height == current_frame.height
                    and prior_frame.width == current_frame.width
                ),
                "canonical_frame_equality": prior_frame.rows == current_frame.rows,
                "prior_palette_partition_cardinality": len(
                    value.prior_structure.source_to_canonical
                ),
                "current_palette_partition_cardinality": len(
                    value.current_structure.source_to_canonical
                ),
                "tracked_entity_count": len(value.current_tracking.entities),
                "established_tracking_count": established_tracking_count,
                "ambiguous_tracking_count": ambiguous_tracking_count,
                "new_tracking_count": new_tracking_count,
                "disappeared_tracking_count": len(
                    value.current_tracking.disappeared_entity_refs
                ),
                "transported_reference_count": len(
                    value.transported_reference_refs
                ),
                "transported_reference_refs": value.transported_reference_refs,
            }
        )
    )


def reduce_uniform_blocks(
    value: UniformBlockReductionInput,
) -> UniformBlockReductionResult:
    frame = value.frame
    available_height = frame.height - value.offset_row
    available_width = frame.width - value.offset_col
    if available_height <= 0 or available_width <= 0:
        return UniformBlockReductionResult(
            reduced_frame=None,
            non_uniform_blocks=(),
            covered_bbox=None,
        )
    if (
        available_height % value.block_height != 0
        or available_width % value.block_width != 0
    ):
        raise ValueError("block dimensions must exactly partition the covered frame")

    reduced: list[tuple[int, ...]] = []
    non_uniform: list[BoundingBox] = []
    for top in range(value.offset_row, frame.height, value.block_height):
        reduced_row: list[int] = []
        for left in range(value.offset_col, frame.width, value.block_width):
            bbox = BoundingBox(
                top=top,
                left=left,
                bottom=top + value.block_height - 1,
                right=left + value.block_width - 1,
            )
            values = {
                frame.rows[row][col]
                for row in range(bbox.top, bbox.bottom + 1)
                for col in range(bbox.left, bbox.right + 1)
            }
            if len(values) != 1:
                non_uniform.append(bbox)
                reduced_row.append(0)
            else:
                reduced_row.append(next(iter(values)))
        reduced.append(tuple(reduced_row))

    covered_bbox = BoundingBox(
        top=value.offset_row,
        left=value.offset_col,
        bottom=frame.height - 1,
        right=frame.width - 1,
    )
    reduced_frame = None if non_uniform else FrameGrid(rows=tuple(reduced))
    return UniformBlockReductionResult(
        reduced_frame=reduced_frame,
        non_uniform_blocks=tuple(non_uniform),
        covered_bbox=covered_bbox,
    )


def enumerate_grid_partitions(value: GridPartitionInput) -> GridPartitionResult:
    frame = value.frame
    candidates: list[GridPartitionCandidate] = []
    for cell_height in _divisors(frame.height):
        grid_rows = frame.height // cell_height
        if grid_rows < value.minimum_grid_rows:
            continue
        for cell_width in _divisors(frame.width):
            grid_columns = frame.width // cell_width
            if grid_columns < value.minimum_grid_columns:
                continue
            patterns = tuple(
                tuple(
                    tuple(frame.rows[row][left : left + cell_width])
                    for row in range(top, top + cell_height)
                )
                for top in range(0, frame.height, cell_height)
                for left in range(0, frame.width, cell_width)
            )
            counts = Counter(patterns)
            uniform_count = sum(
                1
                for pattern in patterns
                if len({item for row in pattern for item in row}) == 1
            )
            candidates.append(
                GridPartitionCandidate(
                    candidate_id=f"partition.{cell_height}x{cell_width}",
                    cell_height=cell_height,
                    cell_width=cell_width,
                    grid_rows=grid_rows,
                    grid_columns=grid_columns,
                    distinct_pattern_count=len(counts),
                    repeated_pattern_count=sum(
                        1 for count in (counts[__yf_order_key] for __yf_order_key in sorted(counts)) if count > 1
                    ),
                    uniform_cell_count=uniform_count,
                )
            )
    return GridPartitionResult(candidates=tuple(candidates))


@dataclass(frozen=True)
class _PeriodicAxis:
    offset: int
    cell_extent: int
    gap_extent: int
    pitch: int
    starts: tuple[int, ...]
    separator_consistent: int
    separator_total: int
    boundary_contrast_ppm: int
    boundary_count: int


_POSITIVE_GRID_REUSE_LIMIT = 8
_positive_grid_reuse: OrderedDict[
    PeriodicCellGridInput, PeriodicCellGridResult
] = OrderedDict()
_positive_grid_reuse_lock = RLock()


def clear_periodic_cell_grid_caches() -> None:
    """Clear bounded exact-result caches for cold measurement benches."""

    _cached_periodic_cell_grids.cache_clear()
    _cached_periodic_axes.cache_clear()
    _count_encoded_cell_patterns.cache_clear()
    with _positive_grid_reuse_lock:
        _positive_grid_reuse.clear()


def enumerate_periodic_cell_grids(
    value: PeriodicCellGridInput,
) -> PeriodicCellGridResult:
    """Enumerate bounded raster-to-cell descriptions without assigning meaning."""

    # Exact immutable measurement reuse only. Large rasters are computed
    # normally without being retained. With zero extra translation quantum,
    # removing mixed/zero-gap descriptions cannot change a complete retained
    # frontier whose candidates all have positive gaps on both axes. The
    # unconditioned request may therefore reuse that same already measured
    # raster result; no grid meaning or role is selected here.
    if value.frame.height * value.frame.width <= 4096:
        if (
            not value.action_conditioned_grid_evidence
            and value.observed_translation_quantum == 0
        ):
            conditioned_key = value.model_copy(
                update={"action_conditioned_grid_evidence": True}
            )
            with _positive_grid_reuse_lock:
                equivalent = _positive_grid_reuse.get(conditioned_key)
                if equivalent is not None:
                    _positive_grid_reuse.move_to_end(conditioned_key)
                    return equivalent
        result = _cached_periodic_cell_grids(value)
        if (
            value.action_conditioned_grid_evidence
            and value.observed_translation_quantum == 0
            and result.candidates
            and all(
                candidate.row_gap > 0 and candidate.col_gap > 0
                for candidate in result.candidates
            )
        ):
            with _positive_grid_reuse_lock:
                _positive_grid_reuse[value] = result
                _positive_grid_reuse.move_to_end(value)
                if len(_positive_grid_reuse) > _POSITIVE_GRID_REUSE_LIMIT:
                    oldest_key = next(iter(_positive_grid_reuse))
                    del _positive_grid_reuse[oldest_key]
        return result
    return _enumerate_periodic_cell_grids(value)


def _with_visible_grid_edges(frame: FrameGrid, grid: PeriodicCellGridCandidate) -> PeriodicCellGridCandidate:
    """Measure clipped cell cores; gaps stay outside the pattern comparison."""
    from .contracts import ClippedPeriodicCell

    first_row = -1 if grid.row_offset - grid.row_pitch + grid.cell_height > 0 else 0
    first_col = -1 if grid.col_offset - grid.col_pitch + grid.cell_width > 0 else 0
    stop_row = (frame.height - 1 - grid.row_offset) // grid.row_pitch + 1
    stop_col = (frame.width - 1 - grid.col_offset) // grid.col_pitch + 1
    edge_rows = (*range(first_row, 0), *range(grid.logical_rows, stop_row))
    edge_cols = (*range(first_col, 0), *range(grid.logical_columns, stop_col))
    cropped_positions = (
        tuple((row, col) for row in edge_rows for col in range(first_col, stop_col))
        + tuple((row, col) for row in range(grid.logical_rows) for col in edge_cols)
    )
    patterns = {}
    pattern_rows = []
    if cropped_positions:
        for cell in grid.cells:
            if cell.pattern_ref not in patterns:
                patterns[cell.pattern_ref] = tuple(
                    line[cell.bbox.left:cell.bbox.right + 1]
                    for line in frame.rows[cell.bbox.top:cell.bbox.bottom + 1]
                )
                pattern_rows.append((cell.pattern_ref, patterns[cell.pattern_ref]))
    clipped = []
    for row, col in cropped_positions:
        top = grid.row_offset + row * grid.row_pitch
        left = grid.col_offset + col * grid.col_pitch
        box = BoundingBox(
            top=max(0, top), left=max(0, left),
            bottom=min(frame.height - 1, top + grid.cell_height - 1),
            right=min(frame.width - 1, left + grid.cell_width - 1),
        )
        dr, dc = box.top - top, box.left - left
        crop = tuple(line[box.left:box.right + 1] for line in frame.rows[box.top:box.bottom + 1])
        matches = tuple(
            ref for ref, pattern in pattern_rows
            if tuple(line[dc:dc + box.width] for line in pattern[dr:dr + box.height]) == crop
        )
        clipped.append(ClippedPeriodicCell(
            row=row, col=col, bbox=box, crop_offset=(dr, dc), compatible_pattern_refs=matches,
        ))
    return grid.model_copy(update={
        "clipped_cells": tuple(clipped),
        "visible_row_range": (first_row, stop_row),
        "visible_column_range": (first_col, stop_col),
    })


def restrict_to_retained_periodic_cell_grid(
    frame: FrameGrid,
    measured: PeriodicCellGridResult,
    geometry: tuple[object, ...] | None,
    *,
    separator_consistency_ppm: int = 0,
) -> PeriodicCellGridResult:
    """Transport the SRC-bound lattice exclusively, without ranking alternatives."""

    if geometry is None:
        return measured
    matches = tuple(
        grid for grid in measured.candidates
        if (
            grid.candidate_ref, grid.row_offset, grid.col_offset,
            grid.cell_height, grid.cell_width, grid.row_pitch, grid.col_pitch,
            grid.logical_rows, grid.logical_columns,
        ) == geometry
    )
    if len(matches) == 1:
        if len(measured.candidates) == 1:
            return measured
        return PeriodicCellGridResult(candidates=matches)
    # A cold top-N may omit the bound lattice. Recount that exact geometry;
    # its omission cannot authorize a switch to another candidate.
    return materialize_retained_periodic_cell_grid(
        frame, geometry, separator_consistency_ppm=separator_consistency_ppm,
    )


def allocate_periodic_cell_extent(
    core_origin: int, core_size: int, pitch: int,
) -> tuple[int, int, int]:
    """Partition the measured interval; odd remainders go to the trailing side.

    The returned origin is not reduced modulo pitch, so a clipped leading
    extent remains explicit. The core and its original pixels are unchanged.
    """
    if not 0 < core_size <= pitch:
        raise ValueError("cell core must fit within its measured pitch")
    gap = pitch - core_size
    leading = gap // 2
    return core_origin - leading, leading, gap - leading


def materialize_retained_periodic_cell_grid(
    frame: FrameGrid,
    geometry: tuple[int, ...],
    *,
    separator_consistency_ppm: int = 0,
) -> PeriodicCellGridResult:
    """Recount one retained lattice without searching alternative lattices."""

    if len(geometry) != 9:
        return PeriodicCellGridResult(candidates=())
    (
        candidate_ref,
        row_offset,
        col_offset,
        cell_height,
        cell_width,
        row_pitch,
        col_pitch,
        logical_rows,
        logical_columns,
    ) = geometry
    if any(item < 0 for item in (row_offset, col_offset)):
        return PeriodicCellGridResult(candidates=())
    if any(item < 1 for item in (cell_height, cell_width, logical_rows, logical_columns)):
        return PeriodicCellGridResult(candidates=())
    if any(item < 2 for item in (row_pitch, col_pitch)):
        return PeriodicCellGridResult(candidates=())
    span_height = (logical_rows - 1) * row_pitch + cell_height
    span_width = (logical_columns - 1) * col_pitch + cell_width
    if row_offset + span_height > frame.height or col_offset + span_width > frame.width:
        return PeriodicCellGridResult(candidates=())

    patterns: list[tuple[tuple[int, ...], ...]] = []
    cells: list[PeriodicCellDescription] = []
    appearance_by_pattern: dict[tuple[tuple[int, ...], ...], tuple[str, tuple[int, ...]]] = {}
    for logical_row in range(logical_rows):
        top = row_offset + logical_row * row_pitch
        for logical_col in range(logical_columns):
            left = col_offset + logical_col * col_pitch
            pattern = tuple(
                tuple(frame.rows[row][left : left + cell_width])
                for row in range(top, top + cell_height)
            )
            patterns.append(pattern)
            appearance = appearance_by_pattern.get(pattern)
            if appearance is None:
                palette_values = tuple(sorted({pixel for row in pattern for pixel in row}))
                appearance = (_cell_pattern_ref(pattern), palette_values)
                appearance_by_pattern[pattern] = appearance
            cells.append(
                PeriodicCellDescription(
                    row=logical_row,
                    col=logical_col,
                    bbox=BoundingBox(
                        top=top,
                        left=left,
                        bottom=top + cell_height - 1,
                        right=left + cell_width - 1,
                    ),
                    point=(top + cell_height // 2, left + cell_width // 2),
                    pattern_ref=appearance[0],
                    palette_value_count=len(appearance[1]),
                    palette_values=appearance[1],
                    pattern_occurrence_count=1,
                )
            )
    counts = Counter(patterns)
    cells = [
        cell.model_copy(update={"pattern_occurrence_count": counts[patterns[index]]})
        for index, cell in enumerate(cells)
    ]
    candidate = PeriodicCellGridCandidate(
        candidate_ref=str(candidate_ref),
        row_offset=row_offset,
        col_offset=col_offset,
        cell_height=cell_height,
        cell_width=cell_width,
        row_gap=row_pitch - cell_height,
        col_gap=col_pitch - cell_width,
        row_pitch=row_pitch,
        col_pitch=col_pitch,
        logical_rows=logical_rows,
        logical_columns=logical_columns,
        separator_consistency_ppm=int(separator_consistency_ppm),
        repeated_pattern_count=sum(
            counts[key] for key in sorted(counts) if counts[key] > 1
        ),
        distinct_pattern_count=len(counts),
        cells=tuple(cells),
    )
    return PeriodicCellGridResult(candidates=(_with_visible_grid_edges(frame, candidate),))


@lru_cache(maxsize=8)
def _cached_periodic_cell_grids(value: PeriodicCellGridInput) -> PeriodicCellGridResult:
    return _enumerate_periodic_cell_grids(value)


def _enumerate_periodic_cell_grids(
    value: PeriodicCellGridInput,
    *,
    prune_dominated_positive_gap: bool = True,
) -> PeriodicCellGridResult:

    frame = value.frame
    exact_lattices = tuple(sorted(set(
        exact_bordered_tile_lattices(frame, maximum_pitch=value.maximum_pitch)
        + exact_joint_tile_lattices(frame, maximum_pitch=value.maximum_pitch)
    )))
    row_axes = _periodic_axes(
        frame,
        axis="row",
        maximum_pitch=min(value.maximum_pitch, frame.height),
        exact_zero_gap_axes=tuple((height, top) for top, _, height, _ in exact_lattices),
    )
    column_axes = _periodic_axes(
        frame,
        axis="column",
        maximum_pitch=min(value.maximum_pitch, frame.width),
        exact_zero_gap_axes=tuple((width, left) for _, left, _, width in exact_lattices),
    )
    palette = tuple(sorted({pixel for row in frame.rows for pixel in row}))
    palette_index = {pixel: index for index, pixel in enumerate(palette)}
    encoded_rows = (
        tuple(bytes(palette_index[pixel] for pixel in row) for row in frame.rows)
        if len(palette) <= 256
        else ()
    )
    described: list[tuple[object, ...]] = []
    encoded_column_slices: dict[tuple[int, int], tuple[int, ...]] = {}
    encoded_slice_ids: dict[bytes, int] = {}
    for column_axis in column_axes:
        if encoded_rows:
            for left in column_axis.starts:
                column_slice_key = (left, column_axis.cell_extent)
                column_slices = encoded_column_slices.get(column_slice_key)
                if column_slices is None:
                    column_slices = tuple(
                        encoded_slice_ids.setdefault(
                            row[left : left + column_axis.cell_extent],
                            len(encoded_slice_ids),
                        )
                        for row in encoded_rows
                    )
                    encoded_column_slices[column_slice_key] = column_slices
    encoded_id_width = max(
        1, ((len(encoded_slice_ids) - 1).bit_length() + 7) // 8
    )
    encoded_column_slice_bytes = {
        key: b"".join(
            slice_id.to_bytes(encoded_id_width, "little")
            for slice_id in slice_ids
        )
        for key, slice_ids in sorted(encoded_column_slices.items())
    }
    column_axis_encoded_byte_slices = tuple(
        tuple(
            encoded_column_slice_bytes[(left, column_axis.cell_extent)]
            for left in column_axis.starts
        )
        for column_axis in column_axes
    )
    mechanically_pruned = False
    positive_gap_top_orders: list[tuple[int, ...]] = []

    def proper_divisor_boundary_gain_line_ppm(
        axis: _PeriodicAxis,
        alternatives: tuple[_PeriodicAxis, ...],
    ) -> int:
        if axis.gap_extent != 0:
            return 0
        aliases = tuple(
            item
            for item in alternatives
            if item.gap_extent == 0
            and item.pitch < axis.pitch
            and axis.pitch % item.pitch == 0
            and item.offset == axis.offset % item.pitch
        )
        if not aliases:
            return 0
        alias_contrast = max(item.boundary_contrast_ppm for item in aliases)
        return max(
            0,
            (axis.boundary_contrast_ppm - alias_contrast)
            * axis.boundary_count,
        )

    # These exact axis-only facts used to be recomputed for every row/column
    # pair.  On a 64x64 frame that repeats the same divisor scan thousands of
    # times, although neither fact depends on the paired orthogonal axis.
    row_divisor_gains = tuple(
        proper_divisor_boundary_gain_line_ppm(axis, row_axes)
        for axis in row_axes
    )
    column_divisor_gains = tuple(
        proper_divisor_boundary_gain_line_ppm(axis, column_axes)
        for axis in column_axes
    )

    def count_pair_patterns(
        row_axis: _PeriodicAxis,
        column_axis: _PeriodicAxis,
        column_axis_index: int,
    ) -> tuple[Counter[object], int]:
        if encoded_rows:
            arguments = (
                row_axis.starts, row_axis.cell_extent, encoded_id_width,
                column_axis_encoded_byte_slices[column_axis_index],
            )
            if frame.height * frame.width <= 4096:
                return _count_encoded_cell_patterns(*arguments)
            return _count_encoded_cell_patterns.__wrapped__(*arguments)
        else:
            patterns = [
                tuple(
                    tuple(
                        frame.rows[row][left : left + column_axis.cell_extent]
                    )
                    for row in range(top, top + row_axis.cell_extent)
                )
                for top in row_axis.starts
                for left in column_axis.starts
            ]
        return Counter(patterns), len(patterns)

    # The special zero-gap retention paths inspect isotropic descriptions
    # only.  If at least one separated isotropic alternative exists and no
    # zero-gap isotropic description can open those paths, mixed/non-isotropic
    # zero-gap pairs cannot reach the final bounded result.  Cache every
    # premeasured isotropic count so this gate never repeats its own work.
    reusable_zero_isotropic_counts: dict[
        tuple[int, int], tuple[Counter[object], int]
    ] = {}
    skip_nonisotropic_zero_gap = False
    if prune_dominated_positive_gap and value.action_conditioned_grid_evidence:
        separated_isotropic_exists = any(
            row.gap_extent > 0
            and col.gap_extent == row.gap_extent
            and col.cell_extent == row.cell_extent
            for row in row_axes
            for col in column_axes
        )
        if separated_isotropic_exists:
            zero_isotropic_special_possible = False
            for row_index, row in enumerate(row_axes):
                if row.gap_extent != 0:
                    continue
                for col_index, col in enumerate(column_axes):
                    if col.gap_extent != 0 or col.cell_extent != row.cell_extent:
                        continue
                    counts, pattern_count = count_pair_patterns(
                        row, col, col_index
                    )
                    reusable_zero_isotropic_counts[(row_index, col_index)] = (
                        counts, pattern_count
                    )
                    if not counts:
                        continue
                    repeated_coverage = sum(
                        filter(lambda count: count > 1, counts.values())
                    )
                    primitive_possible = (
                        row_divisor_gains[row_index] >= 1_000_000
                        and column_divisor_gains[col_index] >= 1_000_000
                        and repeated_coverage * 2 >= pattern_count
                    )
                    modal, second = _largest_two_counts(counts.values())
                    strong_possible = (
                        len(counts) > 1
                        and modal >= 2 * (pattern_count - modal)
                        and 3 * second >= modal
                    )
                    if primitive_possible or strong_possible:
                        zero_isotropic_special_possible = True
            skip_nonisotropic_zero_gap = not zero_isotropic_special_possible
    exact_lattice_pairs = set(exact_lattices)

    for row_axis_index, row_axis in enumerate(row_axes):
        for column_axis_index, column_axis in enumerate(column_axes):
            if (
                not value.action_conditioned_grid_evidence
                and (row_axis.gap_extent == 0 or column_axis.gap_extent == 0)
                and not (
                    row_axis.gap_extent == 0
                    and column_axis.gap_extent == 0
                    and row_axis.cell_extent == column_axis.cell_extent
                )
            ):
                # Without action-conditioned evidence, the retention contract
                # below can expose only separated descriptions or the exact
                # isotropic zero-gap alternative.  Do not materialize cross-
                # product entries that are mechanically unable to reach it.
                mechanically_pruned = True
                continue
            positive_gap_pair = (
                row_axis.gap_extent > 0 and column_axis.gap_extent > 0
            )
            isotropic_pair = (
                row_axis.cell_extent == column_axis.cell_extent
                and row_axis.gap_extent == column_axis.gap_extent
            )
            if (
                skip_nonisotropic_zero_gap
                and not isotropic_pair
                and (row_axis.gap_extent == 0 or column_axis.gap_extent == 0)
                and (
                    row_axis.offset,
                    column_axis.offset,
                    row_axis.cell_extent,
                    column_axis.cell_extent,
                ) not in exact_lattice_pairs
            ):
                mechanically_pruned = True
                continue
            if (
                prune_dominated_positive_gap
                and positive_gap_pair
                and not isotropic_pair
                and len(positive_gap_top_orders) == value.maximum_candidates
            ):
                # Non-isotropic positive-gap pairs can enter only the final
                # structural top-N fill, never a special isotropic/zero-gap
                # retention branch.  This is an optimistic bound on that
                # pair's exact raster order; strict inferiority to N already
                # measured positive-gap pairs proves it cannot be retained.
                cell_count = len(row_axis.starts) * len(column_axis.starts)
                consistent = (
                    row_axis.separator_consistent
                    + column_axis.separator_consistent
                )
                total = row_axis.separator_total + column_axis.separator_total
                upper_order = (
                    int(total > 0 and consistent == total),
                    0,
                    cell_count * row_axis.cell_extent * column_axis.cell_extent,
                    cell_count,
                    cell_count // 2,
                    (consistent * 1_000_000) // total if total else 0,
                    cell_count,
                    -row_axis.pitch,
                    -column_axis.pitch,
                    -row_axis.offset,
                    -column_axis.offset,
                )
                if upper_order < positive_gap_top_orders[0]:
                    mechanically_pruned = True
                    continue
            precomputed = reusable_zero_isotropic_counts.get(
                (row_axis_index, column_axis_index)
            )
            counts, pattern_count = (
                precomputed
                if precomputed is not None
                else count_pair_patterns(
                    row_axis, column_axis, column_axis_index
                )
            )
            # These commutative aggregates never consume an iteration order.
            # Materialize canonical order only for emitted candidates below.
            repeated_cell_coverage = sum(
                filter(lambda count: count > 1, counts.values())
            )
            repeated_pattern_count = sum(
                map(lambda count: count > 1, counts.values())
            )
            consistent = (
                row_axis.separator_consistent
                + column_axis.separator_consistent
            )
            total = row_axis.separator_total + column_axis.separator_total
            consistency_ppm = (consistent * 1_000_000) // total if total else 0
            zero_gap_dominance_ppm = (
                (max(counts.values()) * 1_000_000) // pattern_count
                if (
                    row_axis.gap_extent == 0
                    and column_axis.gap_extent == 0
                    and pattern_count
                )
                else 0
            )
            candidate_ref = (
                "measurement.periodic_grid."
                f"r{row_axis.offset}.{row_axis.cell_extent}.{row_axis.gap_extent}."
                f"c{column_axis.offset}.{column_axis.cell_extent}."
                f"{column_axis.gap_extent}"
            )
            # This ordering only bounds a pure measurement set; the DRM still
            # decides whether any retained description has grid meaning.
            order = (
                int(total > 0 and consistency_ppm == 1_000_000),
                zero_gap_dominance_ppm,
                repeated_cell_coverage
                * row_axis.cell_extent
                * column_axis.cell_extent,
                repeated_cell_coverage,
                repeated_pattern_count,
                consistency_ppm,
                pattern_count,
                -row_axis.pitch,
                -column_axis.pitch,
                -row_axis.offset,
                -column_axis.offset,
            )
            if positive_gap_pair:
                # Only the Nth-best order is consulted for the optimistic
                # bound.  Keep a bounded min-heap instead of sorting the
                # entire shortlist after every measured pair.
                if len(positive_gap_top_orders) < value.maximum_candidates:
                    heapq.heappush(positive_gap_top_orders, order)
                elif order > positive_gap_top_orders[0]:
                    heapq.heapreplace(positive_gap_top_orders, order)
            # Keep exhaustive measurements lightweight.  Full typed cell
            # descriptions are materialized only for the bounded retained set.
            described.append(
                (
                    order,
                    candidate_ref,
                    row_axis,
                    column_axis,
                    pattern_count,
                    counts,
                    consistency_ppm,
                    row_divisor_gains[row_axis_index],
                    column_divisor_gains[column_axis_index],
                )
            )
    described.sort(key=lambda item: (item[0], item[1]), reverse=True)
    # Preserve the two previously established positive-gap isotropic
    # explanations. When no perfect separator exists and the leading zero-gap
    # description's modal pattern occurs at least twice as often as all other
    # patterns combined, retain the first two isotropic
    # descriptions by the common structural order instead. Weak contiguous
    # partitions cannot perturb the validated separated-grid frontier.
    bounded_measurements: list[tuple[object, ...]] = []
    separated_isotropic = tuple(
        item
        for item in described
        if item[2].cell_extent == item[3].cell_extent
        and item[2].gap_extent == item[3].gap_extent
        and item[2].gap_extent > 0
    )[: min(2, value.maximum_candidates)]
    # A fine periodic texture can alias a larger zero-gap cell carrier.  Keep
    # the smallest isotropic macro period whose boundary contrast gains more
    # than one complete orthogonal line over every aligned proper divisor on
    # both axes.  This is a pure raster description; DRM still decides whether
    # the retained carrier has grid meaning.
    primitive_zero_gap_isotropic = next(
        iter(
            sorted(
                (
                    item
                    for item in described
                    if item[2].cell_extent == item[3].cell_extent
                    and item[2].gap_extent == 0
                    and item[3].gap_extent == 0
                    and int(item[7]) >= 1_000_000
                    and int(item[8]) >= 1_000_000
                    and all(
                        int(existing[6]) < 1_000_000
                        for existing in separated_isotropic
                    )
                    and sum(
                        count for count in item[5].values() if count > 1
                    )
                    * 2
                    >= int(item[4])
                ),
                key=lambda item: (item[2].pitch, str(item[1])),
            )
        ),
        None,
    )
    if primitive_zero_gap_isotropic is not None:
        bounded_measurements.append(primitive_zero_gap_isotropic)
    bounded_measurements.extend(
        item
        for item in separated_isotropic
        if str(item[1])
        not in {str(existing[1]) for existing in bounded_measurements}
    )
    bounded_measurements = bounded_measurements[: value.maximum_candidates]
    retained_refs = {str(item[1]) for item in bounded_measurements}
    action_quantized_zero_gap_isotropic = next(
        (
            item
            for item in described
            if value.action_conditioned_grid_evidence
            and value.observed_translation_quantum > 0
            and item[2].cell_extent == value.observed_translation_quantum
            and item[3].cell_extent == value.observed_translation_quantum
            and item[2].gap_extent == 0
            and item[3].gap_extent == 0
        ),
        None,
    )
    if (
        action_quantized_zero_gap_isotropic is not None
        and str(action_quantized_zero_gap_isotropic[1]) not in retained_refs
    ):
        if len(bounded_measurements) >= value.maximum_candidates:
            bounded_measurements = bounded_measurements[
                : max(0, value.maximum_candidates - 1)
            ]
        bounded_measurements.append(action_quantized_zero_gap_isotropic)
        retained_refs = {str(item[1]) for item in bounded_measurements}
    strong_zero_gap_isotropic = next(
        (
            item
            for item in described
            if item[2].cell_extent == item[3].cell_extent
            and item[2].gap_extent == 0
            and item[3].gap_extent == 0
            and item[4]
            and max(item[5].values())
            >= 2 * (int(item[4]) - max(item[5].values()))
            and len(item[5]) > 1
            and 3 * _largest_two_counts(item[5].values())[1]
            >= max(item[5].values())
            and str(item[1]) not in retained_refs
        ),
        None,
    )
    if (
        strong_zero_gap_isotropic is not None
        and all(
            int(item[6]) < 1_000_000
            for item in separated_isotropic
        )
    ):
        bounded_measurements = list(
            tuple(
                item
                for item in described
                if item[2].cell_extent == item[3].cell_extent
                and item[2].gap_extent == item[3].gap_extent
            )[: min(2, value.maximum_candidates)]
        )
        retained_refs = {str(item[1]) for item in bounded_measurements}
    # Retain zero-gap candidates only when no separated explanation exists;
    # otherwise a weak contiguous partition must not displace the separated
    # structural frontier.
    allow_zero_gap = (
        strong_zero_gap_isotropic is not None
        or primitive_zero_gap_isotropic is not None
        or (
            value.action_conditioned_grid_evidence
            and not separated_isotropic
        )
    )
    for measurement in described:
        if len(bounded_measurements) >= value.maximum_candidates:
            break
        if (
            not allow_zero_gap
            and (
                measurement[2].gap_extent == 0
                or measurement[3].gap_extent == 0
            )
        ):
            continue
        if str(measurement[1]) not in retained_refs:
            bounded_measurements.append(measurement)
            retained_refs.add(str(measurement[1]))

    bounded_items: list[PeriodicCellGridCandidate] = []
    # An exact repeated full-tile partition falsifies offset/period aliases of
    # that partition. Do not let a partially consistent separator cut those
    # same tiles into different logical cells. Multiple geometric lattices or
    # a distinct measured macro-carrier remain ambiguous: retain the existing
    # alternatives in that case. This is raster compatibility, never a role or
    # a claim that a tile can be crossed.
    if len(exact_lattices) == 1:
        row_offset, col_offset, row_pitch, col_pitch = exact_lattices[0]
        macro_compatible = primitive_zero_gap_isotropic is None or (
            primitive_zero_gap_isotropic[2].pitch == row_pitch
            and primitive_zero_gap_isotropic[3].pitch == col_pitch
        )
        exact_measurements = [
            item for item in described
            if item[2].offset == row_offset and item[3].offset == col_offset
            and item[2].cell_extent == row_pitch and item[3].cell_extent == col_pitch
            and item[2].gap_extent == 0 and item[3].gap_extent == 0
        ]
        # A fully observed separator partition is also exact. In particular,
        # edge cells can be complete while their half-open full-pitch boxes
        # extend beyond the frame. Preserve this established description.
        exact_separator_present = any(
            int(item[6]) == 1_000_000
            and item[2].pitch == row_pitch and item[3].pitch == col_pitch
            for item in separated_isotropic
        )
        if macro_compatible and not exact_separator_present and len(exact_measurements) == 1:
            bounded_measurements = exact_measurements
    for (
        _,
        candidate_ref,
        row_axis,
        column_axis,
        _pattern_count,
        counts,
        consistency_ppm,
        row_boundary_gain_line_ppm,
        col_boundary_gain_line_ppm,
    ) in bounded_measurements:
        cells: list[PeriodicCellDescription] = []
        # A lattice can contain many copies of one logical cell type.  Its
        # appearance is immutable within this candidate, so measure it once.
        appearance_by_pattern: dict[object, tuple[str, tuple[int, ...]]] = {}
        for logical_row, top in enumerate(row_axis.starts):
            for logical_col, left in enumerate(column_axis.starts):
                pattern = tuple(
                    tuple(
                        frame.rows[row][left : left + column_axis.cell_extent]
                    )
                    for row in range(top, top + row_axis.cell_extent)
                )
                if encoded_rows:
                    encoded_column = encoded_column_slice_bytes[
                        (left, column_axis.cell_extent)
                    ]
                    pattern_key = encoded_column[
                        top * encoded_id_width :
                        (top + row_axis.cell_extent) * encoded_id_width
                    ]
                else:
                    pattern_key = pattern
                appearance = appearance_by_pattern.get(pattern_key)
                if appearance is None:
                    palette_values = tuple(sorted({pixel for row in pattern for pixel in row}))
                    appearance = (_cell_pattern_ref(pattern), palette_values)
                    appearance_by_pattern[pattern_key] = appearance
                cells.append(
                    PeriodicCellDescription(
                        row=logical_row,
                        col=logical_col,
                        bbox=BoundingBox(
                            top=top,
                            left=left,
                            bottom=top + row_axis.cell_extent - 1,
                            right=left + column_axis.cell_extent - 1,
                        ),
                        point=(
                            top + row_axis.cell_extent // 2,
                            left + column_axis.cell_extent // 2,
                        ),
                        pattern_ref=appearance[0],
                        palette_value_count=len(appearance[1]),
                        palette_values=appearance[1],
                        pattern_occurrence_count=counts[pattern_key],
                    )
                )
        bounded_items.append(
            PeriodicCellGridCandidate(
                candidate_ref=candidate_ref,
                row_offset=row_axis.offset,
                col_offset=column_axis.offset,
                cell_height=row_axis.cell_extent,
                cell_width=column_axis.cell_extent,
                row_gap=row_axis.gap_extent,
                col_gap=column_axis.gap_extent,
                row_pitch=row_axis.pitch,
                col_pitch=column_axis.pitch,
                logical_rows=len(row_axis.starts),
                logical_columns=len(column_axis.starts),
                separator_consistency_ppm=consistency_ppm,
                row_boundary_contrast_ppm=row_axis.boundary_contrast_ppm,
                col_boundary_contrast_ppm=column_axis.boundary_contrast_ppm,
                row_proper_divisor_boundary_gain_line_ppm=(
                    row_boundary_gain_line_ppm
                ),
                col_proper_divisor_boundary_gain_line_ppm=(
                    col_boundary_gain_line_ppm
                ),
                repeated_pattern_count=sum(
                    map(lambda count: count > 1, counts.values())
                ),
                distinct_pattern_count=len(counts),
                cells=tuple(cells),
            )
        )
    bounded = tuple(_with_visible_grid_edges(frame, grid) for grid in bounded_items)
    return PeriodicCellGridResult(
        candidates=bounded,
        enumeration_truncated=(
            mechanically_pruned or len(described) > len(bounded)
        ),
    )


@lru_cache(maxsize=8192)
def _count_encoded_cell_patterns(
    row_starts: tuple[int, ...], cell_extent: int, encoded_id_width: int,
    column_slices: tuple[bytes, ...],
) -> tuple[Counter[object], int]:
    """Exact byte-pattern counts shared by evidence modes; private read-only result.

    Keys contain the encoded raster slices and every slicing parameter, never
    roles or a chosen grid. Only <=4096-pixel rasters use the bounded cache.
    """
    patterns = [
        column[top * encoded_id_width : (top + cell_extent) * encoded_id_width]
        for top in row_starts for column in column_slices
    ]
    return Counter(patterns), len(patterns)


def _periodic_axes(
    frame: FrameGrid,
    *,
    axis: str,
    maximum_pitch: int,
    exact_zero_gap_axes: tuple[tuple[int, int], ...] = (),
) -> tuple[_PeriodicAxis, ...]:
    if frame.height * frame.width <= 4096:
        return _cached_periodic_axes(frame, axis, maximum_pitch, exact_zero_gap_axes)
    return _measure_periodic_axes(
        frame, axis=axis, maximum_pitch=maximum_pitch,
        exact_zero_gap_axes=exact_zero_gap_axes,
    )


@lru_cache(maxsize=16)
def _cached_periodic_axes(
    frame: FrameGrid, axis: str, maximum_pitch: int,
    exact_zero_gap_axes: tuple[tuple[int, int], ...],
) -> tuple[_PeriodicAxis, ...]:
    # The action-conditioned flag affects grid composition, never these exact
    # raster-axis measurements. Keep every actual measurement input in the key.
    return _measure_periodic_axes(
        frame, axis=axis, maximum_pitch=maximum_pitch,
        exact_zero_gap_axes=exact_zero_gap_axes,
    )


def _measure_periodic_axes(
    frame: FrameGrid, *, axis: str, maximum_pitch: int,
    exact_zero_gap_axes: tuple[tuple[int, int], ...] = (),
) -> tuple[_PeriodicAxis, ...]:
    extent = frame.height if axis == "row" else frame.width
    orthogonal_extent = frame.width if axis == "row" else frame.height
    palette = tuple(sorted({pixel for row in frame.rows for pixel in row}))
    # Every axis candidate asks about the same adjacent raster lines.
    # Count each boundary once; sum the exact integers before PPM rounding.
    boundary_changes = (0,) + tuple(
        sum(frame.rows[boundary - 1][i] != frame.rows[boundary][i]
            if axis == "row" else
            frame.rows[i][boundary - 1] != frame.rows[i][boundary]
            for i in range(orthogonal_extent))
        for boundary in range(1, extent)
    )

    def boundary_contrast(boundaries: tuple[int, ...]) -> int:
        return (sum(boundary_changes[index] for index in boundaries) * 1_000_000
                // (len(boundaries) * orthogonal_extent)) if boundaries else 0

    palette_index = {pixel: index for index, pixel in enumerate(palette)}
    line_histograms: list[tuple[int, ...]] = []
    for index in range(extent):
        histogram = [0] * len(palette)
        pixels = (
            frame.rows[index]
            if axis == "row"
            else tuple(frame.rows[row][index] for row in range(frame.height))
        )
        for pixel in pixels:
            histogram[palette_index[pixel]] += 1
        line_histograms.append(tuple(histogram))
    candidates: list[_PeriodicAxis] = []
    for pitch in range(2, maximum_pitch + 1):
        # A two-pixel pitch is retained only for a genuine zero-gap raster.
        # A one-pixel cell plus one-pixel separator is too small to establish
        # useful periodic structure and aliases ordinary component centers.
        gaps = (0,) if pitch == 2 else range(0, min(3, pitch - 1) + 1)
        for gap in gaps:
            cell_extent = pitch - gap
            for offset in range(pitch):
                starts = tuple(
                    start
                    for start in range(offset, extent - cell_extent + 1, pitch)
                )
                if len(starts) < 3:
                    continue
                separator_indices = tuple(
                    index
                    for start in starts
                    for index in range(
                        start + cell_extent,
                        min(start + pitch, extent),
                    )
                )
                if gap > 0 and len(separator_indices) < len(starts) - 1:
                    continue
                aggregate_histogram = [0] * len(palette)
                for separator_index in separator_indices:
                    line_histogram = line_histograms[separator_index]
                    for value_index, count in enumerate(line_histogram):
                        aggregate_histogram[value_index] += count
                separator_total = len(separator_indices) * orthogonal_extent
                candidates.append(
                    _PeriodicAxis(
                        offset=offset,
                        cell_extent=cell_extent,
                        gap_extent=gap,
                        pitch=pitch,
                        starts=starts,
                        separator_consistent=(
                            max(aggregate_histogram)
                            if separator_total
                            else 0
                        ),
                        separator_total=separator_total,
                        boundary_contrast_ppm=boundary_contrast(
                            tuple(
                                start + cell_extent
                                for start in starts
                                if 0 < start + cell_extent < extent
                            ),
                        ),
                        boundary_count=sum(
                            1
                            for start in starts
                            if 0 < start + cell_extent < extent
                        ),
                    )
                )
    # Bound the mechanical cross-product before row/column composition.  The
    # retained axis descriptions are still alternatives; no grid meaning is
    # assigned here.  Preserve bounded pitch diversity so a visually dominant
    # long separator cannot erase a shorter repeated-cell explanation before
    # row/column composition supplies the actual repetition evidence.
    retained: list[_PeriodicAxis] = []
    for pitch in range(2, maximum_pitch + 1):
        same_pitch_with_separator = [
            item
            for item in candidates
            if item.pitch == pitch and item.gap_extent > 0
        ]
        same_pitch_with_separator.sort(
            key=lambda item: (
                item.separator_consistent / item.separator_total,
                len(item.starts),
                -item.gap_extent,
                -item.offset,
            ),
            reverse=True,
        )
        retained.extend(same_pitch_with_separator[:2])
        same_pitch_without_separator = [
            item
            for item in candidates
            if item.pitch == pitch and item.gap_extent == 0
        ]
        same_pitch_without_separator.sort(
            key=lambda item: (
                int(item.offset == 0)
                + int(item.starts[-1] + item.cell_extent == extent),
                len(item.starts),
                -item.offset,
            ),
            reverse=True,
        )
        retained.extend(same_pitch_without_separator[:2])
        strongest_boundary_axis = max(
            same_pitch_without_separator,
            key=lambda item: (
                item.boundary_contrast_ppm,
                item.boundary_count,
                -item.offset,
            ),
            default=None,
        )
        if strongest_boundary_axis is not None and strongest_boundary_axis not in retained:
            retained.append(strongest_boundary_axis)
    # Exact repeated tile origins must survive the preliminary axis bound.
    # Otherwise the correct phase can be discarded before cell comparison.
    for item in candidates:
        if (item.gap_extent == 0 and (item.pitch, item.offset) in exact_zero_gap_axes
                and item not in retained):
            retained.append(item)
    return tuple(retained)


def _axis_boundary_contrast_ppm(
    frame: FrameGrid,
    *,
    axis: str,
    boundaries: tuple[int, ...],
) -> int:
    """Measure exact pixel changes across repeated candidate boundaries."""

    if not boundaries:
        return 0
    orthogonal_extent = frame.width if axis == "row" else frame.height
    changed = 0
    for boundary in boundaries:
        for orthogonal in range(orthogonal_extent):
            before = (
                frame.rows[boundary - 1][orthogonal]
                if axis == "row"
                else frame.rows[orthogonal][boundary - 1]
            )
            after = (
                frame.rows[boundary][orthogonal]
                if axis == "row"
                else frame.rows[orthogonal][boundary]
            )
            changed += int(before != after)
    return changed * 1_000_000 // (len(boundaries) * orthogonal_extent)


@lru_cache(maxsize=256)
def _cell_pattern_ref(pattern: tuple[tuple[int, ...], ...]) -> str:
    # Retain the exact patterns from several lattice alternatives. Sixteen
    # entries evict recurring patterns during one bounded frame measurement.
    digest = hashlib.sha256(canonical_json(pattern).encode("utf-8")).hexdigest()[:16]
    return f"measurement.cell_pattern.{digest}"


def _divisors(value: int) -> tuple[int, ...]:
    return tuple(
        candidate for candidate in range(1, value + 1) if value % candidate == 0
    )
