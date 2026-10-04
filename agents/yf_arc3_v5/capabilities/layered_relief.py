"""Bounded perception of truncated reliefs and occlusion-order alternatives.

The input ``reconstructed_cells`` is a measured or previously established
shape completion (for example, from an exact peer), never an oracle answer.
This module only compares masks.  It reports which visible layer can explain
which missing cells and enumerates the compatible front-to-back orders.  It
does not assign palette roles, choose an explanation, or release an action.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from agents.yf_arc3_v5.logos.types import FrozenModel


Cell = tuple[int, int]

_MAXIMUM_RELIEF_LAYERS = 8
_MAXIMUM_RELIEF_RELATIONS = 32
_MAXIMUM_ORDER_EXPANSIONS = 2048


class ReliefLayerInput(FrozenModel):
    """One visible mask and its mechanically reconstructed full mask."""

    layer_ref: str
    visible_cells: frozenset[tuple[int, int]] = Field(min_length=1)
    reconstructed_cells: frozenset[tuple[int, int]] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_masks(self) -> "ReliefLayerInput":
        if not self.visible_cells.issubset(self.reconstructed_cells):
            raise ValueError("visible cells must belong to the reconstructed mask")
        return self


class TruncatedReliefMeasurementInput(FrozenModel):
    """Registry request containing one bounded visible-layer collection."""

    layers: tuple[ReliefLayerInput, ...] = Field(min_length=1, max_length=8)
    maximum_relations: int = Field(default=_MAXIMUM_RELIEF_RELATIONS, ge=1, le=32)


class TruncatedReliefEvidence(FrozenModel):
    """One directed, still-revisable occlusion relation."""

    covering_layer_ref: str
    covered_layer_ref: str
    hidden_cell_count: int = Field(ge=1)
    explained_hidden_cell_count: int = Field(ge=1)
    support_kind: str


class LayerOrderCandidate(FrozenModel):
    """One counterfactual order, from front to back."""

    front_to_back: tuple[str, ...] = Field(min_length=1)


class LayeredReliefMeasurement(FrozenModel):
    """Compact measurement result; no candidate is committed."""

    relations: tuple[TruncatedReliefEvidence, ...]
    order_candidates: tuple[LayerOrderCandidate, ...]
    order_enumeration_truncated: bool
    cycle_detected: bool

    @property
    def unique_order(self) -> bool:
        return len(self.order_candidates) == 1 and not self.order_enumeration_truncated

    def descriptive_facts(self) -> dict[str, object]:
        """Expose bounded quantities for a declarative projection layer."""

        exact_count = sum(
            evidence.support_kind == "exact_truncation"
            for evidence in self.relations
        )
        return {
            "truncated_relief_relation_count": len(self.relations),
            "exact_truncated_relief_relation_count": exact_count,
            "truncated_relief_order_candidate_count": len(self.order_candidates),
            "truncated_relief_unique_order_present": self.unique_order,
            "truncated_relief_order_enumeration_truncated": (
                self.order_enumeration_truncated
            ),
            "truncated_relief_order_cycle_detected": self.cycle_detected,
        }


def _orthogonal_neighbors(cell: Cell) -> tuple[Cell, ...]:
    row, column = cell
    return (
        (row - 1, column),
        (row + 1, column),
        (row, column - 1),
        (row, column + 1),
    )


def _validate_layers(layers: tuple[ReliefLayerInput, ...]) -> tuple[str, ...]:
    if not layers:
        raise ValueError("truncated-relief analysis needs at least one layer")
    refs = tuple(layer.layer_ref for layer in layers)
    if len(layers) > _MAXIMUM_RELIEF_LAYERS:
        raise ValueError("truncated-relief layer count exceeds the bounded limit")
    if len(set(refs)) != len(refs):
        raise ValueError("truncated-relief layer refs must be unique")
    for layer in layers:
        if not layer.visible_cells:
            raise ValueError("a truncated-relief layer needs visible cells")
        if not layer.visible_cells.issubset(layer.reconstructed_cells):
            raise ValueError("visible cells must belong to the reconstructed mask")
    return refs


def enumerate_layer_orders(
    layer_refs: tuple[str, ...],
    precedence_edges: tuple[tuple[str, str], ...],
    *,
    maximum_orders: int = 3,
    maximum_expansions: int = _MAXIMUM_ORDER_EXPANSIONS,
) -> tuple[tuple[LayerOrderCandidate, ...], bool, bool]:
    """Enumerate at most three topological front-to-back order witnesses.

    An edge ``(front, back)`` is a measured constraint.  The function keeps
    alternatives when the evidence is insufficient and reports a bound rather
    than silently picking an order.  A fourth witness is probed only to tell
    whether the returned set is incomplete.
    """

    if not 1 <= maximum_orders <= 3:
        raise ValueError("maximum_orders must be between one and three")
    if maximum_expansions < 1:
        raise ValueError("maximum_expansions must be positive")
    refs = tuple(layer_refs)
    if len(set(refs)) != len(refs):
        raise ValueError("layer refs must be unique")
    ref_set = frozenset(refs)
    edges = tuple(dict.fromkeys(precedence_edges))
    if any(left == right or left not in ref_set or right not in ref_set for left, right in edges):
        raise ValueError("precedence edges must reference two distinct known layers")

    predecessors = {ref: set() for ref in refs}
    for front, back in edges:
        predecessors[back].add(front)

    witnesses: list[LayerOrderCandidate] = []
    expansions = 0
    overflow = False
    cycle_detected = False

    def visit(prefix: tuple[str, ...], remaining: frozenset[str]) -> None:
        nonlocal expansions, overflow, cycle_detected
        if expansions >= maximum_expansions:
            overflow = True
            return
        expansions += 1
        if not remaining:
            if len(witnesses) < maximum_orders:
                witnesses.append(LayerOrderCandidate(front_to_back=prefix))
            else:
                overflow = True
            return
        available = tuple(
            ref
            for ref in sorted(remaining)
            if predecessors[ref].isdisjoint(remaining)
        )
        if not available:
            cycle_detected = True
            return
        for ref in available:
            visit(prefix + (ref,), remaining - {ref})
            if overflow and len(witnesses) >= maximum_orders:
                return

    visit((), frozenset(refs))
    return tuple(witnesses), overflow, cycle_detected


def measure_truncated_relief(
    layers: tuple[ReliefLayerInput, ...],
    *,
    maximum_relations: int = _MAXIMUM_RELIEF_RELATIONS,
) -> LayeredReliefMeasurement:
    """Measure cover explanations and derive compatible layer orders.

    A relation is emitted only when a reconstructed mask has a hidden cell
    touching its visible contour and another visible mask occupies that hidden
    region.  Exact coverage is stronger evidence than partial coverage, but
    both remain evidence rather than a committed semantic interpretation.
    """

    if maximum_relations < 1 or maximum_relations > _MAXIMUM_RELIEF_RELATIONS:
        raise ValueError("maximum_relations is outside the bounded range")
    refs = _validate_layers(layers)
    relations: list[TruncatedReliefEvidence] = []
    for covered in layers:
        hidden = covered.reconstructed_cells - covered.visible_cells
        if not hidden:
            continue
        truncated_boundary = frozenset(
            cell
            for cell in hidden
            if any(neighbor in covered.visible_cells for neighbor in _orthogonal_neighbors(cell))
        )
        if not truncated_boundary:
            continue
        for covering in layers:
            if covering.layer_ref == covered.layer_ref:
                continue
            explained = hidden.intersection(covering.visible_cells)
            if not explained:
                continue
            relation = TruncatedReliefEvidence(
                covering_layer_ref=covering.layer_ref,
                covered_layer_ref=covered.layer_ref,
                hidden_cell_count=len(hidden),
                explained_hidden_cell_count=len(explained),
                support_kind=(
                    "exact_truncation"
                    if explained == hidden
                    else "partial_truncation"
                ),
            )
            relations.append(relation)
            if len(relations) >= maximum_relations:
                break
        if len(relations) >= maximum_relations:
            break

    edges = tuple(
        (evidence.covering_layer_ref, evidence.covered_layer_ref)
        for evidence in relations
    )
    orders, truncated, cycle = enumerate_layer_orders(refs, edges)
    return LayeredReliefMeasurement(
        relations=tuple(relations),
        order_candidates=orders,
        order_enumeration_truncated=truncated,
        cycle_detected=cycle,
    )


def measure_truncated_relief_request(
    value: TruncatedReliefMeasurementInput,
) -> LayeredReliefMeasurement:
    """Registry adapter that keeps the pure tuple-based function convenient."""

    return measure_truncated_relief(
        value.layers,
        maximum_relations=value.maximum_relations,
    )
