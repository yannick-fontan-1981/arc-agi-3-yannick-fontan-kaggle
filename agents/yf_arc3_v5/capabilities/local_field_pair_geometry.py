"""Finite point/footprint geometry; no fusion law or action is selected here."""
from __future__ import annotations

from itertools import combinations
from pydantic import Field, model_validator

from agents.yf_arc3_v5.logos.types import FrozenMap, FrozenModel, Ref


class FootprintRow(FrozenModel):
    item_ref: Ref
    signature_ref: Ref
    bbox: tuple[int, int, int, int]


class PairPointGeometryRequest(FrozenModel):
    schema_version: Ref = "yf_arc3_v5.pair_point_geometry_request.v1"
    rows: tuple[FootprintRow, ...] = Field(max_length=64)
    input_signature_refs: tuple[Ref, Ref]
    region_bbox: tuple[int, int, int, int]
    radius: int = Field(ge=1, le=64)
    maximum_point_checks: int = Field(ge=1, le=1048576)
    representative_order: tuple[str, ...]

    @model_validator(mode="after")
    def unique_rows_and_order(self):
        if len({row.item_ref for row in self.rows}) != len(self.rows):
            raise ValueError("current footprints require distinct identities")
        if self.representative_order != ("point_residual_squared", "midpoint_offset_squared", "row", "col"):
            raise ValueError("unsupported geometric representative order")
        for top, left, bottom, right in (self.region_bbox, *(row.bbox for row in self.rows)):
            if bottom < top or right < left:
                raise ValueError("invalid closed rectangle")
        return self


class PairPointGeometryMeasurements(FrozenModel):
    schema_version: Ref = "yf_arc3_v5.pair_point_geometry_measurements.v1"
    enumeration_complete: bool
    point_check_count: int
    pair_count: int
    candidate_rows: tuple[FrozenMap, ...]


def footprint_distance_squared(bbox, point):
    top, left, bottom, right = bbox
    row, col = point
    dr = max(top - row, 0, row - bottom)
    dc = max(left - col, 0, col - right)
    return dr * dr + dc * dc


def measure_pair_point_geometry(request: PairPointGeometryRequest) -> PairPointGeometryMeasurements:
    """One canonical geometric representative per pair and affected subset.

    The caller declares the finite region, radius hypothesis and mathematical
    representative order. All unlisted bodies are excluded from each domain;
    tangent points remain ambiguous and never enter the strict witnesses.
    """
    radius2 = request.radius * request.radius
    top, left, bottom, right = request.region_bbox
    rows = request.rows
    checks = pairs = 0
    candidates = []
    wanted = tuple(sorted(request.input_signature_refs))
    for a, b in combinations(rows, 2):
        if tuple(sorted((a.signature_ref, b.signature_ref))) != wanted:
            continue
        pairs += 1
        ca = ((a.bbox[0] + a.bbox[2]) // 2, (a.bbox[1] + a.bbox[3]) // 2)
        cb = ((b.bbox[0] + b.bbox[2]) // 2, (b.bbox[1] + b.bbox[3]) // 2)
        separation = sum((x-y)**2 for x,y in zip(ca, cb))
        representatives = {}
        for row in range(top, bottom + 1):
            for col in range(left, right + 1):
                checks += 1
                if checks > request.maximum_point_checks:
                    return PairPointGeometryMeasurements(enumeration_complete=False,
                        point_check_count=checks, pair_count=pairs, candidate_rows=())
                point = (row, col)
                distances = tuple(footprint_distance_squared(item.bbox, point) for item in rows)
                if any(distance == radius2 for distance in distances):
                    continue
                affected = tuple(item.item_ref for item, distance in zip(rows, distances) if distance < radius2)
                if not affected or not set(affected) <= {a.item_ref, b.item_ref}:
                    continue
                if len(affected) == 2:
                    residual = 0
                else:
                    source, peer = (ca, cb) if affected[0] == a.item_ref else (cb, ca)
                    residual = sum((x-y)**2 for x,y in zip(point, peer))
                    # Exact directional geometry, not a prediction of motion quantum.
                    if residual >= separation or point == source:
                        continue
                midpoint = (4*row-a.bbox[0]-a.bbox[2]-b.bbox[0]-b.bbox[2])**2 + (
                    4*col-a.bbox[1]-a.bbox[3]-b.bbox[1]-b.bbox[3])**2
                metrics = {"point_residual_squared": residual, "midpoint_offset_squared": midpoint,
                    "row": row, "col": col}
                key = tuple(metrics[field] for field in request.representative_order)
                previous = representatives.get(affected)
                if previous is None or key < previous[0]:
                    representatives[affected] = (key, FrozenMap({
                        "candidate_ref": f"point-pair:{a.item_ref}:{b.item_ref}:{':'.join(affected)}",
                        "participant_refs": (a.item_ref, b.item_ref), "affected_refs": affected,
                        "participant_bboxes": (a.bbox, b.bbox),
                        "participant_a_top": a.bbox[0], "participant_a_left": a.bbox[1],
                        "participant_b_top": b.bbox[0], "participant_b_left": b.bbox[1],
                        "affected_count": len(affected), "point": point,
                        "pair_separation_squared": separation, **metrics,
                        "strict_domain_under_supplied_radius": True,
                        "excluded_body_count": len(rows)-len(affected)}))
        candidates.extend(representatives[ref][1] for ref in sorted(representatives))
    return PairPointGeometryMeasurements(enumeration_complete=True, point_check_count=checks,
        pair_count=pairs, candidate_rows=tuple(candidates))
