"""Exact mechanical projection of a DRM SELECT result into Logos output."""

from agents.yf_arc3_v5.logos.operations import AlternativeSet
from agents.yf_arc3_v5.logos.types import Cardinality
from agents.yf_arc3_v5.operators.contracts import SelectionResult, SelectOutput


def project_selection_output(
    alternative_refs: tuple[str, ...], decision: SelectionResult
) -> SelectOutput:
    """Preserve the operator's canonical alternative order and explicit reasons."""

    canonical_alternatives = tuple(sorted(alternative_refs))
    if decision.selected_ref is None:
        raise ValueError("SELECT preserved alternatives without commitment")
    if decision.selected_ref not in canonical_alternatives:
        raise ValueError("selection policy chose an undeclared alternative")
    preserved = (
        tuple(ref for ref in canonical_alternatives if ref != decision.selected_ref)
        if decision.preserves_non_selected else ()
    )
    live_alternatives = (
        canonical_alternatives if decision.preserves_non_selected
        else (decision.selected_ref,)
    )
    return SelectOutput(
        alternatives=AlternativeSet(
            selected=decision.selected_ref,
            live_alternatives=live_alternatives,
            cardinality=(
                Cardinality.SPECIAL if len(live_alternatives) > 1
                else Cardinality.SINGULAR
            ),
        ),
        preserved_non_selected=preserved,
        reason_refs=decision.reason_refs,
        comparison_trace=decision.comparison_trace,
    )
