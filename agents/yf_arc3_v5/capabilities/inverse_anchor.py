"""Pure bounded arithmetic for source-supplied dependent anchor relations."""

from __future__ import annotations

from fractions import Fraction
from math import ceil, floor, trunc

from agents.yf_arc3_v5.capabilities.contracts import (
    InverseMeanAnchorDomainInput,
    InverseMeanAnchorDomainMeasurements,
    RationalAnchorDependenceInput,
    RationalAnchorDependenceMeasurements,
)


def measure_rational_anchor_dependence(
    value: RationalAnchorDependenceInput,
) -> RationalAnchorDependenceMeasurements:
    """Evaluate only the supplied affine relation and supplied quantizer."""

    exact_coordinates = tuple(
        Fraction(value.bias_numerators[axis], value.bias_denominator)
        + sum(
            Fraction(numerator, denominator) * position[axis]
            for position, numerator, denominator in zip(
                value.anchor_positions,
                value.coefficient_numerators,
                value.coefficient_denominators,
                strict=True,
            )
        )
        for axis in range(2)
    )
    return RationalAnchorDependenceMeasurements(
        exact_coordinates=tuple(
            (coordinate.numerator, coordinate.denominator)
            for coordinate in exact_coordinates
        ),
        quantized_position=tuple(
            _quantize(coordinate, value.quantizer)
            for coordinate in exact_coordinates
        ),
        quantizer=value.quantizer,
    )


def enumerate_inverse_mean_anchor_domain(
    value: InverseMeanAnchorDomainInput,
) -> InverseMeanAnchorDomainMeasurements:
    """Enumerate every bounded integer position compatible with a supplied mean."""

    fixed_sums = tuple(
        sum(position[axis] for position in value.fixed_anchor_positions)
        for axis in range(2)
    )
    top, left = value.candidate_top_left
    bottom, right = value.candidate_bottom_right
    compatible: list[tuple[int, int]] = []
    tested = 0
    for row in range(top, bottom + 1):
        for column in range(left, right + 1):
            tested += 1
            candidate = (row, column)
            produced = tuple(
                _quantize(
                    Fraction(fixed_sums[axis] + candidate[axis], value.anchor_count),
                    value.quantizer,
                )
                for axis in range(2)
            )
            if produced == value.desired_quantized_position:
                compatible.append(candidate)
    return InverseMeanAnchorDomainMeasurements(
        compatible_positions=tuple(compatible),
        tested_position_count=tested,
        domain_exhaustive=True,
        quantizer=value.quantizer,
    )


def _quantize(value: Fraction, quantizer: str) -> int:
    if quantizer == "floor":
        return floor(value)
    if quantizer == "ceil":
        return ceil(value)
    if quantizer == "truncate_toward_zero":
        return trunc(value)
    if quantizer == "nearest_ties_to_even":
        quotient, remainder = divmod(value.numerator, value.denominator)
        doubled = remainder * 2
        if doubled < value.denominator:
            return quotient
        if doubled > value.denominator:
            return quotient + 1
        return quotient if quotient % 2 == 0 else quotient + 1
    raise ValueError(f"unsupported quantizer: {quantizer}")
