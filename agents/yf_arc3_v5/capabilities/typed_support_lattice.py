"""Exact cell-lattice alternatives exposed by already measured closed supports.

Raster coordinates occur only at this perception boundary. Returned objects
are sets of typed cells. No lattice, object role, control or action is selected.
"""

from collections import Counter
from dataclasses import dataclass

from .cell_recount import recount_known_grid_cells
from .contracts import FrameGrid, HoleSpriteDescription, KnownGridCellRecountInput, KnownGridCellRecountResult
from .sprites import _MAXIMUM_HOLE_SPRITES
from agents.yf_arc3_v5.logos.types import stable_digest


@dataclass(frozen=True, slots=True)
class TypedCellSupport:
    support_ref: str
    top: int
    left: int
    height: int
    width: int
    normalized_cells: tuple[tuple[int, int], ...]
    cell_type_ids: tuple[int, ...]
    cell_type_counts: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class TypedSupportLattice:
    # Raster calibration, not symbolic object size or displacement.
    row_offset: int
    col_offset: int
    cell_extent: int
    logical_rows: int
    logical_columns: int
    uncovered_bottom_extent: int
    uncovered_right_extent: int
    recount: KnownGridCellRecountResult
    supports: tuple[TypedCellSupport, ...]
    repeated_multivalue_support_count: int
    repeated_monochrome_support_count: int


@dataclass(frozen=True, slots=True)
class TypedSupportLatticeMeasurements:
    candidates: tuple[TypedSupportLattice, ...]
    enumeration_truncated: bool


def measure_typed_support_lattices(
    frame: FrameGrid,
    supports: tuple[HoleSpriteDescription, ...],
    *,
    maximum_cell_extent: int = 16,
    maximum_candidates: int = 3,
) -> TypedSupportLatticeMeasurements:
    """Enumerate exact square tilings, then recount their complete cell motifs.

Each support must be an exact union of whole cells on a candidate lattice.
Other supports may disagree and remain unbound; the caller receives their
identities, never a silently completed silhouette. Coarse compatible lattices
are retained as alternatives. Single-sample cells are outside this detector
of internally patterned cells; other declared grid detectors may expose them.
"""
    if not (2 <= maximum_cell_extent <= 32 and 1 <= maximum_candidates <= 3):
        raise ValueError("typed support lattice bounds exceed the declared hard limits")
    # Accept the complete bounded support inventory emitted by perception.
    # A second, smaller transport ceiling must not reject a valid scene.
    if frame.height * frame.width > 4096 or len(supports) > _MAXIMUM_HOLE_SPRITES:
        raise ValueError("typed support lattice input bound exceeded")
    if sum(len(item.pixels) for item in supports) > 8192:
        raise ValueError("typed support lattice support bound exceeded")
    if len({item.hole_sprite_id for item in supports}) != len(supports):
        raise ValueError("typed support lattice requires unique support identities")
    tiled = {}
    for support in supports:
        box = support.bbox
        if box.bottom >= frame.height or box.right >= frame.width:
            raise ValueError("typed support lies outside the observed frame")
        for extent in range(2, min(maximum_cell_extent, box.height, box.width) + 1):
            if box.height % extent or box.width % extent:
                continue
            occupancy = Counter(((r - box.top) // extent, (c - box.left) // extent)
                                for r, c in support.pixels)
            if any(count != extent * extent for count in sorted(occupancy.values())):
                continue
            key = (box.top % extent, box.left % extent, extent)
            tiled.setdefault(key, []).append((support, tuple(sorted(occupancy))))
            if len(tiled) > maximum_candidates:
                return TypedSupportLatticeMeasurements((), True)
    results = []
    for (row_offset, col_offset, extent), rows in sorted(tiled.items()):
        logical_rows = (frame.height - row_offset) // extent
        logical_columns = (frame.width - col_offset) // extent
        ref = "measurement.typed_support_lattice." + stable_digest((row_offset, col_offset, extent))[:20]
        recount = recount_known_grid_cells(KnownGridCellRecountInput(
            frame=frame, grid_ref=ref, row_offset=row_offset, col_offset=col_offset,
            cell_height=extent, cell_width=extent, row_pitch=extent, col_pitch=extent,
            logical_rows=logical_rows, logical_columns=logical_columns,
        ))
        multivalue = frozenset(index for index, pattern in enumerate(recount.type_patterns)
                              if len({value for line in pattern for value in line}) > 1)
        cell_supports = []
        repeated_multivalue_support_count = 0
        repeated_monochrome_support_count = 0
        for support, normalized_cells in rows:
            top = (support.bbox.top - row_offset) // extent
            left = (support.bbox.left - col_offset) // extent
            cell_type_ids = tuple(recount.cell_type_ids[(top + r) * logical_columns + left + c]
                                  for r, c in normalized_cells)
            counts = Counter(cell_type_ids)
            repeated_multivalue_support_count += int(any(
                cell_type in multivalue and count > 1 for cell_type, count in sorted(counts.items())))
            repeated_monochrome_support_count += int(any(
                cell_type not in multivalue and count > 1 for cell_type, count in sorted(counts.items())))
            cell_supports.append(TypedCellSupport(
                support.hole_sprite_id, top, left,
                support.bbox.height // extent, support.bbox.width // extent,
                normalized_cells, cell_type_ids, tuple(sorted(counts.items())),
            ))
        results.append(TypedSupportLattice(
            row_offset, col_offset, extent, logical_rows, logical_columns,
            (frame.height - row_offset) % extent, (frame.width - col_offset) % extent,
            recount, tuple(cell_supports), repeated_multivalue_support_count,
            repeated_monochrome_support_count,
        ))
    return TypedSupportLatticeMeasurements(tuple(results), False)
