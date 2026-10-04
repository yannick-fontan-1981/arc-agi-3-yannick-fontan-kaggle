"""Exact cross-grid support for a distinctive cell; no role assignment."""

from __future__ import annotations

from collections.abc import Mapping

from agents.yf_arc3_v5.capabilities.contracts import (
    PeriodicCellGridCandidate,
    PeriodicCellGridResult,
)
from agents.yf_arc3_v5.logos.types import FrozenMap


def measure_singleton_multivalue_grid_support(
    periodic_grids: Mapping[str, object] | PeriodicCellGridResult,
) -> FrozenMap:
    """Measure singleton cells retained across every bounded grid alternative.

    The result says only what the candidate partitions contain. DRM decides
    whether this morphology warrants a revisable control supposition.
    """

    grids = tuple(
        periodic_grids.candidates
        if isinstance(periodic_grids, PeriodicCellGridResult)
        else periodic_grids.get("candidates", ())
    )
    if not grids or len(grids) > 3:
        return FrozenMap(
            {
                "grid_candidate_count": len(grids),
                "singleton_grid_candidate_count": 0,
                "cross_grid_unique_pixel_count": 0,
            }
        )
    unique_pixel_sets: list[set[tuple[int, int]]] = []
    singleton_count = 0
    for grid in grids:
        if isinstance(grid, PeriodicCellGridCandidate):
            unique_cells = tuple(
                cell for cell in grid.cells
                if cell.pattern_occurrence_count == 1
                and cell.palette_value_count > 1
            )
        elif isinstance(grid, Mapping):
            unique_cells = tuple(
                cell for cell in grid.get("cells", ())
                if cell.get("pattern_occurrence_count") == 1
                and cell.get("palette_value_count", 0) > 1
            )
        else:
            raise TypeError("periodic grid candidate is not a mapping")
        singleton_count += len(unique_cells) == 1
        pixels: set[tuple[int, int]] = set()
        for cell in unique_cells:
            if isinstance(grid, PeriodicCellGridCandidate):
                box = cell.bbox
                top, left, bottom, right = box.top, box.left, box.bottom, box.right
            else:
                box = cell["bbox"]
                top, left, bottom, right = (
                    int(box[name]) for name in ("top", "left", "bottom", "right")
                )
            pixels.update(
                (row, col)
                for row in range(top, bottom + 1)
                for col in range(left, right + 1)
            )
        unique_pixel_sets.append(pixels)
    return FrozenMap(
        {
            "grid_candidate_count": len(grids),
            "singleton_grid_candidate_count": singleton_count,
            "cross_grid_unique_pixel_count": len(
                set.intersection(*unique_pixel_sets)
            ),
        }
    )
