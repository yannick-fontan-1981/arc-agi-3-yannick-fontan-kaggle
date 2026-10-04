"""Pure bounded compaction and retrodiction over declared transition outputs."""

from __future__ import annotations

import hashlib

from agents.yf_arc3_v5.capabilities.contracts import (
    TransitionDivergence,
    TransitionEvidenceClass,
    TransitionEvidenceLedger,
    TransitionEvidenceLedgerInput,
    TransitionRetrodictionInput,
    TransitionRetrodictionMeasurements,
)
from agents.yf_arc3_v5.logos.types import FrozenMap


def compact_transition_evidence(
    value: TransitionEvidenceLedgerInput,
) -> TransitionEvidenceLedger:
    """Group exact observed signatures and retain at most three witness refs."""

    grouped: dict[tuple[object, ...], list[str]] = {}
    for item in sorted(value.transitions, key=lambda row: row.sequence_index):
        signature = (
            item.action_ref,
            item.action_payload_digest,
            item.observed_delta_digest,
            item.observed_changed_count,
            item.observed_terminal_ref,
        )
        grouped.setdefault(signature, []).append(item.transition_ref)

    classes = []
    for signature, refs in sorted(grouped.items()):
        encoded = "\x1f".join("" if part is None else str(part) for part in signature)
        classes.append(
            TransitionEvidenceClass(
                signature_digest=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
                action_ref=str(signature[0]),
                action_payload_digest=str(signature[1]),
                observed_delta_digest=str(signature[2]),
                observed_changed_count=int(signature[3]),
                observed_terminal_ref=(
                    None if signature[4] is None else str(signature[4])
                ),
                occurrence_count=len(refs),
                witness_transition_refs=tuple(refs[:3]),
            )
        )
    return TransitionEvidenceLedger(
        observed_transition_count=len(value.transitions),
        evidence_classes=tuple(classes),
    )


def measure_transition_retrodiction(
    value: TransitionRetrodictionInput,
) -> TransitionRetrodictionMeasurements:
    """Compare declared predictions exactly; never select or revise their meaning."""

    predictions = {item.transition_ref: item for item in value.predictions}
    uncovered: list[str] = []
    divergent: list[str] = []
    first: TransitionDivergence | None = None
    exact_count = 0

    for observed in sorted(value.transitions, key=lambda row: row.sequence_index):
        predicted = predictions[observed.transition_ref]
        if not predicted.prediction_applies:
            uncovered.append(observed.transition_ref)
            continue

        mismatches: list[str] = []
        if predicted.predicted_after_digest != observed.after_digest:
            mismatches.append("measurement.after_digest")
        if predicted.predicted_delta_digest != observed.observed_delta_digest:
            mismatches.append("measurement.delta_digest")
        if predicted.predicted_changed_count != observed.observed_changed_count:
            mismatches.append("measurement.changed_count")
        if predicted.predicted_terminal_ref != observed.observed_terminal_ref:
            mismatches.append("measurement.terminal_ref")
        if not mismatches:
            exact_count += 1
            continue

        divergent.append(observed.transition_ref)
        if first is None:
            first = TransitionDivergence(
                transition_ref=observed.transition_ref,
                sequence_index=observed.sequence_index,
                observed_after_digest=observed.after_digest,
                predicted_after_digest=predicted.predicted_after_digest,
                observed_delta_digest=observed.observed_delta_digest,
                predicted_delta_digest=predicted.predicted_delta_digest,
                observed_changed_count=observed.observed_changed_count,
                predicted_changed_count=predicted.predicted_changed_count,
                observed_terminal_ref=observed.observed_terminal_ref,
                predicted_terminal_ref=predicted.predicted_terminal_ref,
                mismatch_dimensions=tuple(mismatches),
            )

    covered_count = len(value.transitions) - len(uncovered)
    return TransitionRetrodictionMeasurements(
        model_ref=value.model_ref,
        observed_count=len(value.transitions),
        covered_count=covered_count,
        exact_count=exact_count,
        uncovered_transition_refs=tuple(uncovered),
        divergent_transition_refs=tuple(divergent),
        first_divergence=first,
        descriptive_facts=FrozenMap(
            {
                "model_ref": value.model_ref,
                "observed_count": len(value.transitions),
                "covered_count": covered_count,
                "exact_count": exact_count,
                "uncovered_transition_refs": tuple(uncovered),
                "divergent_transition_refs": tuple(divergent),
                "first_divergence_transition_ref": (
                    None if first is None else first.transition_ref
                ),
                "has_divergence": bool(divergent),
                "has_coverage": covered_count > 0,
                "complete_coverage": covered_count == len(value.transitions),
                "exact_on_covered": exact_count == covered_count,
            }
        ),
    )
