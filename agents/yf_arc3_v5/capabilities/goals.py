"""Bounded exact measurements for dimensions declared by open goals."""

from __future__ import annotations

from agents.yf_arc3_v5.capabilities.contracts import (
    DeclaredGoalGapInput,
    DeclaredGoalGapMeasurement,
    DeclaredGoalGapRequest,
    DeclaredGoalGapResult,
)


def measure_declared_goal_gaps(value: DeclaredGoalGapInput) -> DeclaredGoalGapResult:
    """Compare only requested dimensions; never open, rank, or select a goal."""

    measurements = tuple(_measure(item) for item in value.requests)
    return DeclaredGoalGapResult(
        measurements=measurements,
        all_satisfied=all(item.satisfied for item in measurements),
    )


def _measure(value: DeclaredGoalGapRequest) -> DeclaredGoalGapMeasurement:
    observed = value.observed_value
    target = value.target_value
    integer_pair = (
        isinstance(observed, int)
        and not isinstance(observed, bool)
        and isinstance(target, int)
        and not isinstance(target, bool)
    )
    return DeclaredGoalGapMeasurement(
        goal_ref=value.goal_ref,
        goal_kind=value.goal_kind,
        dimension=value.dimension,
        observed_value=observed,
        target_value=target,
        satisfied=observed == target,
        signed_integer_delta=(target - observed) if integer_pair else None,
    )
