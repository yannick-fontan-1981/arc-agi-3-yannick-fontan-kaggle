"""Bounded set filtering for declaratively authored abductive explanations."""

from __future__ import annotations

from agents.yf_arc3_v5.capabilities.contracts import (
    AbductiveDominanceObservation,
    AbductiveExplanationRejection,
    AbductiveVersionSpaceInput,
    AbductiveVersionSpaceMeasurements,
)
from agents.yf_arc3_v5.logos.types import FrozenMap


def measure_abductive_version_space(
    value: AbductiveVersionSpaceInput,
) -> AbductiveVersionSpaceMeasurements:
    """Filter exact violations, then compute unweighted set dominance."""

    required = set(value.required_obligation_refs)
    admissible = []
    rejections = []
    for item in value.explanations:
        missing = tuple(sorted(required - set(item.covered_obligation_refs)))
        reasons = []
        if missing:
            reasons.append("measurement.incomplete_obligation_coverage")
        if item.contradiction_refs:
            reasons.append("measurement.observed_contradiction")
        if item.forbidden_instance_fact_refs:
            reasons.append("measurement.forbidden_instance_dependency")
        if not item.falsifier_refs:
            reasons.append("measurement.missing_explicit_falsifier")
        if reasons:
            rejections.append(
                AbductiveExplanationRejection(
                    explanation_ref=item.explanation_ref,
                    reason_refs=tuple(reasons),
                    missing_obligation_refs=missing,
                )
            )
        else:
            admissible.append(item)

    dominance = []
    dominated_refs: set[str] = set()
    for left in admissible:
        left_coverage = set(left.covered_obligation_refs)
        left_dependencies = set(left.dependency_refs)
        for right in admissible:
            if left.explanation_ref == right.explanation_ref:
                continue
            right_coverage = set(right.covered_obligation_refs)
            right_dependencies = set(right.dependency_refs)
            if not (
                left_coverage.issuperset(right_coverage)
                and left_dependencies.issubset(right_dependencies)
            ):
                continue
            strict = []
            if left_coverage != right_coverage:
                strict.append("measurement.coverage_superset")
            if left_dependencies != right_dependencies:
                strict.append("measurement.dependency_subset")
            if not strict:
                continue
            dominance.append(
                AbductiveDominanceObservation(
                    dominant_explanation_ref=left.explanation_ref,
                    dominated_explanation_ref=right.explanation_ref,
                    strict_dimension_refs=tuple(strict),
                )
            )
            dominated_refs.add(right.explanation_ref)

    undominated = tuple(
        item.explanation_ref
        for item in admissible
        if item.explanation_ref not in dominated_refs
    )
    tests = FrozenMap(
        {
            item.explanation_ref: item.discriminating_test_refs
            for item in admissible
            if item.explanation_ref in undominated
        }
    )
    status = (
        "empty" if not undominated else "unique" if len(undominated) == 1 else "ambiguous"
    )
    return AbductiveVersionSpaceMeasurements(
        admissible_explanation_refs=tuple(item.explanation_ref for item in admissible),
        rejections=tuple(rejections),
        dominance_observations=tuple(dominance),
        undominated_explanation_refs=undominated,
        surviving_discriminating_test_refs=tests,
        descriptive_facts=FrozenMap(
            {
                "isolated_reasoning_scope_ref": value.isolated_reasoning_scope_ref,
                "measurement_context_ref": value.measurement_context_ref,
                "declared_explanation_count": len(value.explanations),
                "admissible_explanation_count": len(admissible),
                "rejected_explanation_count": len(rejections),
                "undominated_explanation_count": len(undominated),
                "version_space_status": status,
                "arbitrary_tie_break_required": len(undominated) > 1,
                "arbitrary_tie_break_basis_ref": (
                    "basis.canonical_structural_reference_only"
                    if len(undominated) > 1
                    else None
                ),
                "admissible_explanation_refs": tuple(
                    item.explanation_ref for item in admissible
                ),
                "undominated_explanation_refs": undominated,
                "rejection_reason_refs": FrozenMap(
                    {item.explanation_ref: item.reason_refs for item in rejections}
                ),
                "surviving_discriminating_test_refs": tests,
            }
        ),
    )
