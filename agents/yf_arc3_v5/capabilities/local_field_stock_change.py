"""Exact stock deltas for Source-supplied participants and output signature.

This measures a finite before/after comparison. It never assigns a recipe,
authorizes an action or interprets an animation as persistent stock.
"""
from __future__ import annotations

from collections import Counter
from pydantic import Field, model_validator

from agents.yf_arc3_v5.capabilities.contracts import (
    LocalFieldTransactionInput, LocalFieldTypedBody, LocalFieldTypedSnapshot,
)
from agents.yf_arc3_v5.capabilities.local_field_pair_geometry import FootprintRow, footprint_distance_squared
from agents.yf_arc3_v5.capabilities.local_field_type_production import measure_local_field_transaction
from agents.yf_arc3_v5.logos.types import FrozenMap, FrozenModel, Ref


class StockChangeRequest(FrozenModel):
    schema_version: Ref = "yf_arc3_v5.stock_change_request.v1"
    before_rows: tuple[FootprintRow, ...] = Field(max_length=64)
    after_rows: tuple[FootprintRow, ...] = Field(max_length=64)
    before_other_rows: tuple[FootprintRow, ...] = Field(max_length=64)
    after_other_rows: tuple[FootprintRow, ...] = Field(max_length=64)
    participant_refs: tuple[Ref, ...] = Field(min_length=2, max_length=8)
    output_signature_ref: Ref
    point: tuple[int, int]
    before_observation_ref: Ref
    after_observation_ref: Ref

    @model_validator(mode="after")
    def validate_identities(self):
        for rows in (self.before_rows, self.after_rows, self.before_other_rows, self.after_other_rows):
            if len({row.item_ref for row in rows}) != len(rows):
                raise ValueError("stock snapshots require distinct physical identities")
        if len(set(self.participant_refs)) != len(self.participant_refs):
            raise ValueError("one participant cannot supply two consumed inputs")
        if not set(self.participant_refs) <= {row.item_ref for row in self.before_rows}:
            raise ValueError("participants must belong to the supplied before stock")
        for row in (*self.before_rows, *self.after_rows, *self.before_other_rows, *self.after_other_rows):
            if row.bbox[2] < row.bbox[0] or row.bbox[3] < row.bbox[1]:
                raise ValueError("invalid stock rectangle")
        return self


class StockChangeMeasurements(FrozenModel):
    schema_version: Ref = "yf_arc3_v5.stock_change_measurements.v1"
    descriptive_facts: FrozenMap


def measure_stock_change(request: StockChangeRequest) -> StockChangeMeasurements:
    before = {row.item_ref: row for row in request.before_rows}
    after = {row.item_ref: row for row in request.after_rows}
    participants = set(request.participant_refs)
    outside = {ref: row for ref, row in sorted(before.items()) if ref not in participants}
    residual = tuple(row for ref, row in sorted(after.items()) if ref not in outside)
    # Reuse the transaction capability for persistent identity/type comparisons.
    # Stock rows are measured solid rectangles, unlike the unassigned other rows.
    def snapshot(ref, rows):
        return LocalFieldTypedSnapshot(snapshot_ref=ref, bodies=tuple(
            LocalFieldTypedBody(body_ref=row.item_ref, type_ref=row.signature_ref,
                positions=tuple((r, c) for r in range(row.bbox[0], row.bbox[2]+1)
                    for c in range(row.bbox[1], row.bbox[3]+1))) for row in rows))
    old = snapshot(request.before_observation_ref, request.before_rows)
    new = snapshot(request.after_observation_ref, request.after_rows)
    transaction = measure_local_field_transaction(LocalFieldTransactionInput(
        before=old, transient=old, after_packet=(new,)))
    expected_counts = Counter(row.signature_ref for row in request.before_rows)
    expected_counts.subtract(before[ref].signature_ref for ref in request.participant_refs)
    expected_counts[request.output_signature_ref] += 1
    actual_counts = Counter(row.signature_ref for row in request.after_rows)
    retained = all(ref in after and after[ref].signature_ref == before[ref].signature_ref
        for ref in request.participant_refs)
    def separation(rows):
        centers = [((row.bbox[0]+row.bbox[2])/2, (row.bbox[1]+row.bbox[3])/2) for row in rows]
        return sum(sum((a-b)**2 for a,b in zip(centers[i], centers[j]))
            for i in range(len(centers)) for j in range(i+1,len(centers)))
    before_gap = separation([before[ref] for ref in request.participant_refs])
    after_gap = separation([after[ref] for ref in request.participant_refs]) if retained else before_gap
    return StockChangeMeasurements(descriptive_facts=FrozenMap({
        "outside_rows_unchanged": all(after.get(ref) == row for ref,row in sorted(outside.items())),
        "other_rows_unchanged": {row.item_ref: row for row in request.before_other_rows}
            == {row.item_ref: row for row in request.after_other_rows},
        "signature_counts_match_supplied_rewrite": actual_counts == +expected_counts,
        "residual_body_count": len(residual),
        "residual_output_signature_matches": len(residual) == 1 and residual[0].signature_ref == request.output_signature_ref,
        "residual_contains_point": len(residual) == 1 and footprint_distance_squared(residual[0].bbox, request.point) == 0,
        "participant_identities_and_signatures_retained": retained,
        "stock_count_unchanged": len(before) == len(after),
        "participant_separation_decreased": after_gap < before_gap,
        "before_stock_count": len(before), "after_stock_count": len(after),
        "input_signature_refs": tuple(before[ref].signature_ref for ref in request.participant_refs),
        "residual_rows": tuple(row.model_dump(mode="python") for row in residual),
        "transaction_measurements": transaction.model_dump(mode="python"),
    }))
