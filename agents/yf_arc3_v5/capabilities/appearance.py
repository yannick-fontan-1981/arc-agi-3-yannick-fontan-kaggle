"""Exact appearance measurements on an explicitly supplied bearer support.

No palette value has a meaning here. Order and key composition are supplied
by the source-hashed declarative contract, not authored by this capability.
"""

from __future__ import annotations

from collections import Counter
from enum import IntEnum
from functools import reduce
from math import gcd

from pydantic import Field, model_validator

from agents.yf_arc3_v5.capabilities.digital_topology import (
    DigitalTopologyContract, DigitalTopologyInput, measure_digital_topology,
)
from agents.yf_arc3_v5.logos.types import FrozenMap, FrozenModel, Ref, stable_digest


class AppearanceKeyTier(FrozenModel):
    code: Ref
    ordinal: int = Field(ge=1)
    fields: tuple[Ref, ...] = Field(min_length=1)
    pixel_scale_sensitive: bool


class AppearanceKeyContract(FrozenModel):
    tiers: tuple[AppearanceKeyTier, ...] = Field(min_length=1)
    rotation_quarter_turns: tuple[int, ...] = Field(min_length=1, max_length=4)
    source_hash: Ref
    digital_topology_contract: DigitalTopologyContract | None = None

    @model_validator(mode="after")
    def validate_unique_contract(self) -> "AppearanceKeyContract":
        if len({tier.code for tier in self.tiers}) != len(self.tiers):
            raise ValueError("duplicate appearance tier code")
        if len({tier.ordinal for tier in self.tiers}) != len(self.tiers):
            raise ValueError("duplicate appearance tier ordinal")
        rotations = self.rotation_quarter_turns
        if set(rotations) != {0, 1, 2, 3} or len(rotations) != 4:
            raise ValueError("rotation contract must declare the complete quarter-turn group")
        return self

    def ordinal_enum(self) -> type[IntEnum]:
        """Mechanical representation of the declared symbolic order."""
        return IntEnum("AppearanceRank", {tier.code: tier.ordinal for tier in self.tiers})


class BearerAppearanceInput(FrozenModel):
    bearer_ref: Ref
    # Values only on the actual bearer: no inferred background/crop padding.
    valued_pixels: tuple[tuple[int, int, int], ...] = Field(min_length=1, max_length=4096)
    measurement_basis_ref: Ref
    cell_quantum: int | None = Field(default=None, ge=1)
    key_contract: AppearanceKeyContract
    # Produced by a separately attested structural recognizer, never guessed.
    structural_descriptor: FrozenMap | None = None
    structural_source_hash: Ref | None = None
    max_bbox_area: int = Field(default=4096, ge=1, le=4096)
    schema_version: Ref = "yf_arc3_v5.bearer_appearance_input.v1"

    @model_validator(mode="after")
    def validate_support(self) -> "BearerAppearanceInput":
        points = {(r, c) for r, c, _value in self.valued_pixels}
        if len(points) != len(self.valued_pixels):
            raise ValueError("bearer support repeats a pixel")
        height = max(r for r, _ in points) - min(r for r, _ in points) + 1
        width = max(c for _, c in points) - min(c for _, c in points) + 1
        if height * width > self.max_bbox_area:
            raise ValueError("appearance bbox measurement bound exceeded")
        if (self.structural_descriptor is None) != (self.structural_source_hash is None):
            raise ValueError("structural descriptor requires its source attestation")
        if self.structural_descriptor is not None and not self.structural_descriptor:
            raise ValueError("empty structural descriptor is unknown, not a match")
        return self


class AppearanceSignatureKey(FrozenModel):
    code: Ref
    ordinal: int = Field(ge=1)
    digest: Ref


class BearerAppearanceMeasurement(FrozenModel):
    bearer_ref: Ref
    descriptor_digest: Ref
    descriptors: FrozenMap
    orientation_quarter_turns: tuple[int, ...]
    keys: tuple[AppearanceSignatureKey, ...]
    key_count: int = Field(ge=0)
    descriptive_delta: FrozenMap
    contract_source_hash: Ref
    schema_version: Ref = "yf_arc3_v5.bearer_appearance_measurement.v1"


def _normalize(points: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    top = min(r for r, _ in points)
    left = min(c for _, c in points)
    return tuple(sorted((r - top, c - left) for r, c in points))


def measure_orthogonal_bands(points: tuple[tuple[int, int], ...]) -> FrozenMap:
    """Exact silhouette modulo repeated adjacent rows/columns, not role identity.

    Keeps orientation, gaps and band order. Independent band thicknesses may
    differ; the original dimensions remain available to distinguish scales.
    Called only after BearerAppearanceInput has validated the finite bbox.
    """
    support = set(_normalize(points))
    height = max(r for r, _ in support) + 1
    width = max(c for _, c in support) + 1
    rows = tuple(tuple((r, c) in support for c in range(width)) for r in range(height))
    row_starts = tuple(i for i in range(height) if i == 0 or rows[i] != rows[i - 1])
    columns = tuple(tuple(rows[r][c] for r in row_starts) for c in range(width))
    col_starts = tuple(i for i in range(width) if i == 0 or columns[i] != columns[i - 1])
    reduced = tuple(tuple(rows[r][c] for c in col_starts) for r in row_starts)
    return FrozenMap({
        "measurement_contract": "yf_arc3_v5.orthogonal_silhouette_bands.v1",
        "oriented_signature": stable_digest(reduced),
        "band_rows": len(row_starts), "band_columns": len(col_starts),
        "row_thicknesses": tuple(b - a for a, b in zip(row_starts, row_starts[1:] + (height,))),
        "column_thicknesses": tuple(b - a for a, b in zip(col_starts, col_starts[1:] + (width,))),
        "height": height, "width": width,
    })


def measure_bearer_appearance(value: BearerAppearanceInput) -> BearerAppearanceMeasurement:
    points = tuple((r, c) for r, c, _pixel in value.valued_pixels)
    oriented = _normalize(points)
    rotations: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    for quarter_turns in value.key_contract.rotation_quarter_turns:
        rotated = points
        for _ in range(quarter_turns):
            rotated = tuple((c, -r) for r, c in rotated)
        rotations.append((quarter_turns, _normalize(rotated)))
    canonical = min(item[1] for item in rotations)
    top, left = min(r for r, _ in points), min(c for _, c in points)
    sprite = tuple(sorted((r - top, c - left, pixel) for r, c, pixel in value.valued_pixels))
    histogram = tuple(sorted(Counter(pixel for _, _, pixel in value.valued_pixels).items()))
    divisor = reduce(gcd, (count for _pixel, count in histogram))
    descriptors = FrozenMap({
        "sprite": stable_digest(sprite),
        "oriented_shape": stable_digest(oriented),
        "shape": stable_digest(canonical),
        "colors": tuple(pixel for pixel, _count in histogram),
        "histogram": histogram,
        "proportions": tuple((pixel, count // divisor) for pixel, count in histogram),
        "structure": (
            stable_digest((value.structural_source_hash, value.structural_descriptor))
            if value.structural_descriptor is not None else None
        ),
    })
    keys: list[AppearanceSignatureKey] = []
    topology_measurements = (
        measure_digital_topology(DigitalTopologyInput(support=points, contract=value.key_contract.digital_topology_contract))
        if value.key_contract.digital_topology_contract is not None else None
    )
    for tier in value.key_contract.tiers:
        if any(field not in descriptors for field in tier.fields):
            raise ValueError("appearance contract names an unmeasured descriptor")
        parts = tuple(descriptors[field] for field in tier.fields)
        if any(part is None for part in parts):
            continue
        basis = (
            (value.measurement_basis_ref, value.cell_quantum)
            if tier.pixel_scale_sensitive else ()
        )
        keys.append(AppearanceSignatureKey(
            code=tier.code,
            ordinal=tier.ordinal,
            digest=stable_digest((value.key_contract.source_hash, tier.code, basis, parts)),
        ))
    return BearerAppearanceMeasurement(
        bearer_ref=value.bearer_ref,
        descriptor_digest=stable_digest(descriptors),
        descriptors=descriptors,
        orientation_quarter_turns=tuple(turns for turns, mask in rotations if mask == canonical),
        keys=tuple(keys),
        key_count=len(keys),
        descriptive_delta=FrozenMap({
            "normalized_valued_pixels": sprite,
            "signature_keys": tuple(FrozenMap(key.model_dump()) for key in keys),
            "signature_key_count": len(keys),
            "contract_source_hash": value.key_contract.source_hash,
            "digital_topology_measurements": topology_measurements,
            "orthogonal_band_measurements": measure_orthogonal_bands(points),
        }),
        contract_source_hash=value.key_contract.source_hash,
    )
