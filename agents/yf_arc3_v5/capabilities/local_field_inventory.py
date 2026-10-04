"""Neutral measurements of an equally spaced array and adjacent components.

The caller declares size and enumeration bounds. This module assigns no HUD,
resource, request, mechanism or terminal role to any component.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from pydantic import Field

from agents.yf_arc3_v5.capabilities.components import detect_components
from agents.yf_arc3_v5.capabilities.contracts import ComponentExtractionInput, FrameGrid
from agents.yf_arc3_v5.logos.types import FrozenMap, FrozenModel, Ref


class ArrayRegionMeasurementRequest(FrozenModel):
    schema_version: Ref = "yf_arc3_v5.array_region_measurement_request.v1"
    frame: FrameGrid
    component_entity_rows: tuple[tuple[Ref, Ref], ...] = ()
    maximum_square_side: int = Field(ge=1, le=32)
    minimum_array_size: int = Field(ge=2, le=16)
    maximum_array_size: int = Field(ge=2, le=16)
    maximum_band_height: int = Field(ge=1, le=64)
    maximum_pattern_rows: int = Field(ge=1, le=64)


class ArrayRegionMeasurements(FrozenModel):
    schema_version: Ref = "yf_arc3_v5.array_region_measurements.v1"
    enumeration_complete: bool
    pattern_count: int = Field(ge=0)
    pattern_rows: tuple[FrozenMap, ...]


def measure_array_adjacent_regions(request: ArrayRegionMeasurementRequest) -> ArrayRegionMeasurements:
    components = detect_components(ComponentExtractionInput(frame=request.frame, connectivity=4)).components
    entities = dict(request.component_entity_rows)
    squares = tuple(c for c in components if c.bbox.width == c.bbox.height
        and c.area == c.bbox.width * c.bbox.height
        and c.bbox.width <= request.maximum_square_side and not c.touches_frame_boundary)

    def inside(c, region):
        return (region.bbox.top <= c.bbox.top <= c.bbox.bottom <= region.bbox.bottom
            and region.bbox.left <= c.bbox.left <= c.bbox.right <= region.bbox.right)

    def row(c):
        return FrozenMap({"component_ref": c.component_id,
            "item_ref": entities.get(c.component_id, c.component_id),
            "signature_ref": f"appearance.uniform.value.{c.value}",
            "value": c.value, "side": c.bbox.width,
            "bbox": (c.bbox.top, c.bbox.left, c.bbox.bottom, c.bbox.right)})

    bands = tuple(c for c in components if c.touches_frame_boundary
        and c.bbox.top == 0 and c.bbox.width == request.frame.width
        and c.bbox.height <= request.maximum_band_height)
    patterns = []
    for band in bands:
        groups = defaultdict(list)
        for c in squares:
            if inside(c, band):
                groups[(c.bbox.top, c.bbox.width)].append(c)
        adjacent = tuple(c for c in components if c.touches_frame_boundary
            and c.bbox.top == band.bbox.bottom + 1
            and c.bbox.width == request.frame.width and c.area > band.area)
        for key in sorted(groups):
            members = sorted(groups[key], key=lambda c: c.bbox.left)
            by_col = {c.bbox.left: c for c in members}
            for a, b in combinations(members, 2):
                spacing = b.bbox.left - a.bbox.left
                if spacing <= a.bbox.width or a.bbox.left - spacing in by_col:
                    continue
                chain = []; col = a.bbox.left
                while col in by_col:
                    chain.append(by_col[col]); col += spacing
                if not request.minimum_array_size <= len(chain) <= request.maximum_array_size:
                    continue
                values = tuple(c.value for c in chain)
                if len(set(values)) != len(values):
                    continue
                enlarged = tuple(c for c in squares if inside(c, band)
                    and c.value in values and c.bbox.width > a.bbox.width)
                if not enlarged or not adjacent:
                    continue
                for region in adjacent:
                    matching = tuple(c for c in squares if inside(c, region) and c.value in values)
                    inner = tuple(c for c in components if inside(c, region) and not c.touches_frame_boundary)
                    patterns.append(FrozenMap({
                        "band_component_ref": band.component_id,
                        "band_bbox": (band.bbox.top, band.bbox.left, band.bbox.bottom, band.bbox.right),
                        "adjacent_component_ref": region.component_id,
                        "adjacent_bbox": (region.bbox.top, region.bbox.left, region.bbox.bottom, region.bbox.right),
                        "array_count": len(chain), "array_rows": tuple(row(c) for c in chain),
                        "neighbor_signature_pairs": tuple((row(a)["signature_ref"], row(b)["signature_ref"])
                            for a, b in zip(chain, chain[1:])),
                        "enlarged_matching_count": len(enlarged),
                        "enlarged_matching_rows": tuple(row(c) for c in enlarged),
                        "unique_enlarged_matching_signature_ref": row(enlarged[0])["signature_ref"] if len(enlarged) == 1 else None,
                        "adjacent_matching_count": len(matching),
                        "adjacent_inner_component_count": len(inner),
                        "adjacent_matching_rows": tuple(row(c) for c in matching),
                        "adjacent_other_rows": tuple(FrozenMap({**dict(row(c)),
                            "signature_ref": f"appearance.component.{c.component_id}"})
                            for c in inner if c not in matching),
                    }))
                    if len(patterns) > request.maximum_pattern_rows:
                        return ArrayRegionMeasurements(enumeration_complete=False,
                            pattern_count=len(patterns), pattern_rows=())
    return ArrayRegionMeasurements(enumeration_complete=True,
        pattern_count=len(patterns), pattern_rows=tuple(patterns))
