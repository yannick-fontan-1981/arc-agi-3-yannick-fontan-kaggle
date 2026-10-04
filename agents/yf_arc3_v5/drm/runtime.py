"""Generic evaluators for explicitly declared DRM authority modes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import cmp_to_key
import re

from agents.yf_arc3_v5.drm.contracts import (
    DrmDeclaration,
    DrmDeclarationKind,
    DrmEvaluationKind,
    DrmFactCondition,
    DrmFactOperator,
    DrmPreferredOrder,
    DrmSelectionTier,
)
from agents.yf_arc3_v5.drm.registry import DrmRegistry
from agents.yf_arc3_v5.logos.terms import Term
from agents.yf_arc3_v5.logos.types import FrozenModel, TermKind
from agents.yf_arc3_v5.observability.decision_capture import authority_observer
from agents.yf_arc3_v5.operators.authority import (
    AuthorityDefinition,
    OperatorAuthorityRegistry,
    RuntimeAuthorityKind,
)
from agents.yf_arc3_v5.operators.contracts import (
    ComparisonInput,
    ComparisonKind,
    ComparisonResult,
    DerivationInput,
    DerivationResult,
    DistinctionInput,
    DistinctionResult,
    ExactOrder,
    PredicateInput,
    PredicateResult,
    PreferredOrder,
    SelectionInput,
    SelectionComparisonTrace,
    SelectionCriterionTrace,
    SelectionResult,
    UpdatePolicyInput,
    UpdatePolicyResult,
)


_TIER_PARTITION_CACHE_LIMIT = 512
_TierPartitionKey = tuple[type[object], object]
_TierEqualityFilter = tuple[
    str,
    dict[_TierPartitionKey, tuple[int, ...]],
    tuple[int, ...],
]
_TierPartitionPlan = tuple[
    _TierEqualityFilter,
    dict[_TierPartitionKey, _TierEqualityFilter | None],
    _TierEqualityFilter | None,
]
_TIER_PARTITION_CACHE: dict[
    int,
    tuple[tuple[DrmSelectionTier, ...], _TierPartitionPlan | None],
] = {}
_TIER_FILTER_CACHE_LIMIT = 4096
_TIER_FILTER_CACHE: dict[
    tuple[int, tuple[int, ...], tuple[str, ...]],
    tuple[tuple[DrmSelectionTier, ...], _TierEqualityFilter | None],
] = {}


_MISSING_FACT = object()


class _SharedSelectionFacts:
    """Two-map read view used by the pure DRM evaluator without copying facts."""

    __slots__ = ("_delta", "_context")

    def __init__(
        self,
        delta: Mapping[str, object],
        context: Mapping[str, object],
    ) -> None:
        self._delta = delta
        self._context = context

    def get(self, key: object, default: object = None) -> object:
        value = self._delta.get(key, _MISSING_FACT)
        if value is not _MISSING_FACT:
            return value
        return self._context.get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self._delta or key in self._context


def _selection_facts(
    value: SelectionInput,
    alternative_ref: str,
) -> Mapping[str, object] | _SharedSelectionFacts:
    facts = value.alternative_facts[alternative_ref]
    if not value.context_facts:
        return facts
    return _SharedSelectionFacts(facts, value.context_facts)


def build_drm_authority_registry(drm: DrmRegistry) -> OperatorAuthorityRegistry:
    authorities = OperatorAuthorityRegistry()
    static_symbols = {item.id: item for item in drm.symbol_definitions()}
    for declaration_id in drm.declaration_ids:
        declaration, loaded = drm.resolve(declaration_id)
        runtime_kind = _runtime_kind(declaration.kind)
        if runtime_kind is None:
            continue
        authorities.register(
            AuthorityDefinition(
                id=declaration.id,
                kind=runtime_kind,
                supported_operators=declaration.supported_operators,
                input_contract=declaration.input_contract or "",
                output_contract=declaration.output_contract or "",
                effect_class=declaration.effect_class,
                source_unit=loaded.source_unit,
                source_hash=loaded.source_hash,
            ),
            _implementation(
                declaration,
                source_unit=loaded.source_unit,
                source_hash=loaded.source_hash,
            ),
            static_symbol=static_symbols[declaration.id],
        )
    return authorities


def _runtime_kind(kind: DrmDeclarationKind) -> RuntimeAuthorityKind | None:
    return {
        DrmDeclarationKind.SCHEMA: None,
        DrmDeclarationKind.PRINCIPLE: None,
        DrmDeclarationKind.MODEL: RuntimeAuthorityKind.MODEL,
        DrmDeclarationKind.PREDICATE: RuntimeAuthorityKind.PREDICATE,
        DrmDeclarationKind.RULE: RuntimeAuthorityKind.RULE,
        DrmDeclarationKind.CRITERION: RuntimeAuthorityKind.CRITERION,
        DrmDeclarationKind.UPDATE_POLICY: RuntimeAuthorityKind.UPDATE_POLICY,
        DrmDeclarationKind.SELECTION_POLICY: RuntimeAuthorityKind.SELECTION_POLICY,
    }[kind]


def _implementation(
    declaration: DrmDeclaration,
    *,
    source_unit: str = "",
    source_hash: str = "",
) -> Callable[[FrozenModel], FrozenModel]:
    evaluation = declaration.evaluation

    def evaluate(value: FrozenModel) -> FrozenModel:
        if evaluation is DrmEvaluationKind.DISTINGUISH_REFERENCES:
            assert isinstance(value, DistinctionInput)
            return DistinctionResult(
                distinctions=tuple(
                    Term(
                        id=f"distinction:{source_ref}",
                        kind=TermKind.ENTITY,
                        label="observed distinct referent",
                        provenance=(source_ref,),
                    )
                    for source_ref in value.source_refs
                ),
                reason_refs=declaration.reason_refs,
            )
        if evaluation is DrmEvaluationKind.BASIS_PRESENT:
            assert isinstance(value, PredicateInput)
            return PredicateResult(
                holds=bool(value.arguments and value.basis_refs),
                reason_refs=declaration.reason_refs,
            )
        if evaluation is DrmEvaluationKind.PROJECT_REQUESTED_CONCLUSIONS:
            assert isinstance(value, DerivationInput)
            return DerivationResult(
                conclusions=value.requested_conclusions,
                reason_refs=declaration.reason_refs,
            )
        if evaluation is DrmEvaluationKind.EXACT_SET_EQUALITY:
            assert isinstance(value, ComparisonInput)
            comparison = (
                ComparisonKind.MATCH
                if tuple(sorted(value.left_refs)) == tuple(sorted(value.right_refs))
                else ComparisonKind.MISMATCH
            )
            return ComparisonResult(
                comparison=comparison,
                reason_refs=declaration.reason_refs,
            )
        if evaluation is DrmEvaluationKind.EXACT_REVISION:
            assert isinstance(value, UpdatePolicyInput)
            return UpdatePolicyResult(
                allowed=value.proposal.state_revision == value.current_state_revision,
                reason_refs=declaration.reason_refs,
            )
        if evaluation is DrmEvaluationKind.SINGLE_ADMISSIBLE:
            assert isinstance(value, SelectionInput)
            selected = (
                value.alternative_refs[0] if len(value.alternative_refs) == 1 else None
            )
            return SelectionResult(
                selected_ref=selected,
                preserves_non_selected=declaration.preserves_alternatives,
                reason_refs=declaration.reason_refs,
            )
        if evaluation is DrmEvaluationKind.DECLARED_ELIGIBILITY_SINGLE:
            assert isinstance(value, SelectionInput)
            eligible_refs = _declared_eligible_alternatives(value, declaration)
            selected = eligible_refs[0] if len(eligible_refs) == 1 else None
            ordered = tuple((0, item, None) for item in eligible_refs)
            return SelectionResult(
                selected_ref=selected,
                preserves_non_selected=declaration.preserves_alternatives,
                reason_refs=declaration.reason_refs,
                comparison_trace=_selection_comparison_trace(
                    value,
                    declaration,
                    ordered,
                    selected,
                ),
            )
        if evaluation is DrmEvaluationKind.SYMBOLIC_PARTIAL_ORDER:
            assert isinstance(value, SelectionInput)
            return SelectionResult(
                selected_ref=_unique_dominant_alternative(value),
                preserves_non_selected=declaration.preserves_alternatives,
                reason_refs=declaration.reason_refs,
            )
        if evaluation is DrmEvaluationKind.DECLARED_LEXICOGRAPHIC_ORDER:
            assert isinstance(value, SelectionInput)
            ordered = declared_lexicographic_order(value, declaration)
            selected, tier_reason = _declared_lexicographic_alternative(
                value, declaration, ordered=ordered
            )
            debt_reason = (
                _canonical_equivalence_debt_ref(
                    ordered[0],
                    ordered[1] if len(ordered) > 1 else None,
                    value,
                    declaration,
                )
                if selected is not None and ordered
                else None
            )
            return SelectionResult(
                selected_ref=selected,
                preserves_non_selected=declaration.preserves_alternatives,
                reason_refs=tuple(
                    dict.fromkeys(
                        (
                            *declaration.reason_refs,
                            *((tier_reason,) if tier_reason else ()),
                            *((debt_reason,) if debt_reason else ()),
                        )
                    )
                ),
                comparison_trace=_selection_comparison_trace(
                    value,
                    declaration,
                    ordered,
                    selected,
                ),
            )
        raise ValueError(f"declaration has no runtime evaluator: {declaration.id}")

    def observed_evaluate(value: FrozenModel) -> FrozenModel:
        result = evaluate(value)
        observer = authority_observer()
        if observer is not None:
            observer(declaration, value, result, source_unit, source_hash)
        return result

    return observed_evaluate


def _unique_dominant_alternative(value: SelectionInput) -> str | None:
    candidates = tuple(
        alternative
        for alternative in value.alternative_refs
        if all(
            alternative == other or _dominates(alternative, other, value)
            for other in value.alternative_refs
        )
    )
    return candidates[0] if len(candidates) == 1 else None


def declared_eligibility_matches(
    facts: Mapping[str, object], declaration: DrmDeclaration,
) -> bool:
    """Measure the exact source-declared gate, without choosing a candidate."""
    return all(
        _condition_holds(condition, facts)
        for condition in declaration.selection_eligibility_conditions
    ) and all(
        field in facts and facts.get(field) is not None
        for field in declaration.selection_required_fact_fields
    )


def _declared_eligible_alternatives(
    value: SelectionInput,
    declaration: DrmDeclaration,
) -> tuple[str, ...]:
    """Filter only with DRM-authored exact conditions, without ranking."""

    eligible: list[str] = []
    for alternative_ref in value.alternative_refs:
        facts = _selection_facts(value, alternative_ref)
        if not declared_eligibility_matches(facts, declaration):
            continue
        eligible.append(alternative_ref)
    return tuple(eligible)


def _dominates(left: str, right: str, value: SelectionInput) -> bool:
    comparisons = tuple(
        item
        for item in value.dominance_witnesses
        if {item.left_ref, item.right_ref} == {left, right}
    )
    if not comparisons:
        return False
    strict = False
    for item in comparisons:
        order = item.observed_order
        if item.left_ref != left:
            order = {
                ExactOrder.LESS_THAN: ExactOrder.GREATER_THAN,
                ExactOrder.EQUAL: ExactOrder.EQUAL,
                ExactOrder.GREATER_THAN: ExactOrder.LESS_THAN,
            }[order]
        favorable = (
            ExactOrder.LESS_THAN
            if item.preferred_order is PreferredOrder.MINIMIZE
            else ExactOrder.GREATER_THAN
        )
        unfavorable = (
            ExactOrder.GREATER_THAN
            if favorable is ExactOrder.LESS_THAN
            else ExactOrder.LESS_THAN
        )
        if order is unfavorable:
            return False
        strict = strict or order is favorable
    return strict


def _declared_lexicographic_alternative(
    value: SelectionInput,
    declaration: DrmDeclaration,
    *,
    ordered: tuple[tuple[int, str, DrmSelectionTier | None], ...] | None = None,
) -> tuple[str | None, str | None]:
    """Apply only the exact tier and tie-break table contained in the DRM."""

    if not value.alternative_facts:
        return None, None
    if ordered is None:
        ordered = declared_lexicographic_order(value, declaration)
    if not ordered:
        return None, None
    if len(ordered) > 1 and _compare_declared_items(
        ordered[0], ordered[1], value, declaration
    ) == 0:
        return None, None
    tier = ordered[0][2]
    if len(ordered) > 1:
        left = _selection_facts(value, ordered[0][1])
        right = _selection_facts(value, ordered[1][1])
        for criterion in declaration.selection_precedence_criteria:
            if _compare_tie_breaker_values(criterion, left.get(criterion.field), right.get(criterion.field)):
                return ordered[0][1], criterion.reason_ref
    return ordered[0][1], tier.reason_ref if tier is not None else None


def _canonical_equivalence_debt_ref(
    selected: tuple[int, str, DrmSelectionTier | None],
    runner_up: tuple[int, str, DrmSelectionTier | None] | None,
    value: SelectionInput,
    declaration: DrmDeclaration,
) -> str | None:
    """Project the DRM-authored debt only when canonical order decides a tie."""

    debt_ref = declaration.canonical_equivalence_tie_break_debt_ref
    if runner_up is None or not debt_ref or selected[0] != runner_up[0]:
        return None
    selected_facts = _selection_facts(value, selected[1])
    runner_up_facts = _selection_facts(value, runner_up[1])
    if any(_compare_tie_breaker_values(item, selected_facts.get(item.field), runner_up_facts.get(item.field))
           for item in declaration.selection_precedence_criteria):
        return None
    for tie_breaker in declaration.selection_tie_breakers:
        if _compare_tie_breaker_values(
            tie_breaker,
            selected_facts.get(tie_breaker.field),
            runner_up_facts.get(tie_breaker.field),
        ):
            return (
                debt_ref
                if tie_breaker.field in {"candidate_ref", "alternative_ref"}
                else None
            )
    return None


def declared_lexicographic_order(
    value: SelectionInput,
    declaration: DrmDeclaration,
) -> tuple[tuple[int, str, DrmSelectionTier | None], ...]:
    """Materialize the complete order encoded by one DRM selection policy.

    This is the same exact evaluator used by ``SELECT``.  It adds no tier,
    tie-breaker, eligibility rule, or fallback; callers may only preserve the
    ordering already present in the source-hashed DRM declaration.
    """

    ranked: list[tuple[int, str, DrmSelectionTier | None]] = []
    for alternative_ref in value.alternative_refs:
        facts = _selection_facts(value, alternative_ref)
        if not all(
            _condition_holds(condition, facts)
            for condition in declaration.selection_eligibility_conditions
        ):
            continue
        if any(
            field not in facts or facts.get(field) is None
            for field in declaration.selection_required_fact_fields
        ):
            continue
        tier_index, tier = _first_matching_tier(facts, declaration.selection_tiers)
        ranked.append((tier_index, alternative_ref, tier))
    if not ranked:
        return ()

    return tuple(
        sorted(
            ranked,
            key=cmp_to_key(
                lambda left, right: _compare_declared_items(
                    left, right, value, declaration
                )
            ),
        )
    )


def _compare_declared_items(
    left: tuple[int, str, DrmSelectionTier | None],
    right: tuple[int, str, DrmSelectionTier | None],
    value: SelectionInput,
    declaration: DrmDeclaration,
) -> int:
    left_facts = _selection_facts(value, left[1])
    right_facts = _selection_facts(value, right[1])
    for criterion in declaration.selection_precedence_criteria:
        order = _compare_tie_breaker_values(criterion, left_facts.get(criterion.field), right_facts.get(criterion.field))
        if order:
            return order if criterion.preferred_order is DrmPreferredOrder.MINIMIZE else -order
    if left[0] != right[0]:
        return -1 if left[0] < right[0] else 1
    for tie_breaker in declaration.selection_tie_breakers:
        order = _compare_tie_breaker_values(
            tie_breaker,
            left_facts.get(tie_breaker.field),
            right_facts.get(tie_breaker.field),
        )
        if order:
            return (
                order
                if tie_breaker.preferred_order is DrmPreferredOrder.MINIMIZE
                else -order
            )
    return 0


def _selection_comparison_trace(
    value: SelectionInput,
    declaration: DrmDeclaration,
    ordered: tuple[tuple[int, str, DrmSelectionTier | None], ...],
    selected_ref: str | None,
) -> SelectionComparisonTrace:
    eligible_refs = tuple(item[1] for item in ordered)
    eligible_set = frozenset(eligible_refs)
    ineligible_refs = tuple(
        ref for ref in value.alternative_refs if ref not in eligible_set
    )
    runner_up_ref = ordered[1][1] if len(ordered) > 1 else None
    criterion_traces: list[SelectionCriterionTrace] = []
    first_discriminating: str | None = None
    if ordered and runner_up_ref is not None:
        left = ordered[0]
        right = ordered[1]
        left_facts = _selection_facts(value, left[1])
        right_facts = _selection_facts(value, right[1])
        for criterion in declaration.selection_precedence_criteria:
            raw_order = _compare_tie_breaker_values(criterion, left_facts.get(criterion.field), right_facts.get(criterion.field))
            order = raw_order if criterion.preferred_order is DrmPreferredOrder.MINIMIZE else -raw_order
            criterion_traces.append(SelectionCriterionTrace(
                criterion_ref=criterion.criterion_ref,
                field=criterion.field, preferred_order=criterion.preferred_order.value,
                left_ref=left[1], right_ref=right[1],
                left_value=left_facts.get(criterion.field), right_value=right_facts.get(criterion.field),
                outcome="left_preferred" if order < 0 else "right_preferred" if order > 0 else "equal",
                reason_refs=(criterion.reason_ref,),
            ))
            if first_discriminating is None and order:
                first_discriminating = criterion.criterion_ref
        tier_order = -1 if left[0] < right[0] else (1 if left[0] > right[0] else 0)
        tier_reason_refs = tuple(
            dict.fromkeys(
                (
                    *((left[2].reason_ref,) if left[2] is not None else ()),
                    *((right[2].reason_ref,) if right[2] is not None else ()),
                    *declaration.reason_refs,
                )
            )
        )
        tier_criterion_ref = tier_reason_refs[0]
        criterion_traces.append(
            SelectionCriterionTrace(
                criterion_ref=tier_criterion_ref,
                field="selection_tier_index",
                preferred_order="minimize",
                left_ref=left[1],
                right_ref=right[1],
                left_value=left[0],
                right_value=right[0],
                outcome=("left_preferred" if tier_order < 0 else "right_preferred" if tier_order > 0 else "equal"),
                reason_refs=tier_reason_refs,
            )
        )
        if first_discriminating is None and tier_order:
            first_discriminating = tier_criterion_ref
        left_facts = _selection_facts(value, left[1])
        right_facts = _selection_facts(value, right[1])
        for tie_breaker in declaration.selection_tie_breakers:
            raw_order = _compare_tie_breaker_values(
                tie_breaker,
                left_facts.get(tie_breaker.field),
                right_facts.get(tie_breaker.field),
            )
            preferred_order = tie_breaker.preferred_order.value
            effective_order = (
                raw_order
                if tie_breaker.preferred_order is DrmPreferredOrder.MINIMIZE
                else -raw_order
            )
            outcome = (
                "left_preferred"
                if effective_order < 0
                else "right_preferred"
                if effective_order > 0
                else "equal"
            )
            criterion_ref = tie_breaker.criterion_ref or tie_breaker.field
            criterion_traces.append(
                SelectionCriterionTrace(
                    criterion_ref=criterion_ref,
                    field=tie_breaker.field,
                    preferred_order=preferred_order,
                    left_ref=left[1],
                    right_ref=right[1],
                    left_value=left_facts.get(tie_breaker.field),
                    right_value=right_facts.get(tie_breaker.field),
                    outcome=outcome,
                    reason_refs=(
                        (tie_breaker.reason_ref,)
                        if tie_breaker.reason_ref
                        else declaration.reason_refs
                    ),
                )
            )
            if first_discriminating is None and effective_order:
                first_discriminating = criterion_ref
    return SelectionComparisonTrace(
        selection_policy_ref=declaration.id,
        eligible_refs=eligible_refs,
        ineligible_refs=ineligible_refs,
        ordered_refs=eligible_refs,
        selected_ref=selected_ref,
        runner_up_ref=runner_up_ref,
        first_discriminating_criterion_ref=first_discriminating,
        criterion_traces=tuple(criterion_traces),
        unresolved_equality=bool(ordered and runner_up_ref and selected_ref is None),
    )


def _first_matching_tier(
    facts: object,
    tiers: tuple[DrmSelectionTier, ...],
) -> tuple[int, DrmSelectionTier | None]:
    partition = _tier_equality_partition(tiers)
    if partition is None:
        candidate_indexes = tuple(range(len(tiers)))
        excluded_fields: tuple[str, ...] = ()
    else:
        primary, refinements_by_value, default_refinement = partition
        field, candidates_by_value, unconstrained = primary
        excluded_fields = (field,)
        value = _tier_partition_fact_value(facts, field)
        partition_key = (type(value), value)
        candidate_indexes = candidates_by_value.get(
            partition_key, unconstrained
        )
        refinement = refinements_by_value.get(partition_key, default_refinement)
        if refinement is not None:
            field, candidates_by_value, unconstrained = refinement
            excluded_fields = (*excluded_fields, field)
            value = _tier_partition_fact_value(facts, field)
            candidate_indexes = candidates_by_value.get(
                (type(value), value), unconstrained
            )
    # Continue the same exact equality filtering lazily.  Two fields are often
    # insufficient for large generic policies; additional fields only remove
    # tiers whose declared equality premise is already false.
    for _depth in range(6):
        refinement = _cached_best_tier_equality_filter(
            tiers,
            candidate_indexes,
            excluded_fields=excluded_fields,
        )
        if refinement is None:
            break
        field, candidates_by_value, unconstrained = refinement
        excluded_fields = (*excluded_fields, field)
        value = _tier_partition_fact_value(facts, field)
        candidate_indexes = candidates_by_value.get(
            (type(value), value), unconstrained
        )
    for index in candidate_indexes:
        tier = tiers[index]
        all_hold = all(
            _condition_holds(condition, facts) for condition in tier.all_conditions
        )
        any_hold = not tier.any_conditions or any(
            _condition_holds(condition, facts) for condition in tier.any_conditions
        )
        if all_hold and any_hold:
            return index, tier
    return len(tiers), None


def _tier_equality_partition(
    tiers: tuple[DrmSelectionTier, ...],
) -> _TierPartitionPlan | None:
    """Compile a bounded two-step index without changing declared tier order."""

    identity = id(tiers)
    cached = _TIER_PARTITION_CACHE.get(identity)
    if cached is not None and cached[0] is tiers:
        return cached[1]
    primary = _best_tier_equality_filter(
        tiers,
        tuple(range(len(tiers))),
        excluded_fields=(),
    )
    partition = None
    if primary is not None:
        field, candidates_by_value, unconstrained = primary
        refinements_by_value = {
            key: _best_tier_equality_filter(
                tiers,
                candidates_by_value[key],
                excluded_fields=(field,),
            )
            for key in sorted(
                candidates_by_value,
                key=lambda item: (item[0].__name__, item[1]),
            )
        }
        default_refinement = _best_tier_equality_filter(
            tiers,
            unconstrained,
            excluded_fields=(field,),
        )
        partition = primary, refinements_by_value, default_refinement
    if len(_TIER_PARTITION_CACHE) >= _TIER_PARTITION_CACHE_LIMIT:
        _TIER_PARTITION_CACHE.pop(next(iter(_TIER_PARTITION_CACHE)))
    _TIER_PARTITION_CACHE[identity] = tiers, partition
    return partition


def _best_tier_equality_filter(
    tiers: tuple[DrmSelectionTier, ...],
    candidate_indexes: tuple[int, ...],
    *,
    excluded_fields: tuple[str, ...],
) -> _TierEqualityFilter | None:
    buckets_by_field: dict[
        str, dict[_TierPartitionKey, list[int]]
    ] = {}
    for index in candidate_indexes:
        tier = tiers[index]
        seen_fields: set[str] = set()
        for condition in tier.all_conditions:
            if condition.operator is not DrmFactOperator.EQUALS:
                continue
            field = condition.field
            if field in excluded_fields or field in seen_fields:
                continue
            seen_fields.add(field)
            expected = condition.expected
            field_buckets = buckets_by_field.setdefault(field, {})
            field_buckets.setdefault((type(expected), expected), []).append(index)
    useful_fields = tuple(
        field
        for field in sorted(buckets_by_field)
        if len(buckets_by_field[field]) > 1
        and len(candidate_indexes)
        - sum(
            len(buckets_by_field[field][key])
            for key in sorted(
                buckets_by_field[field],
                key=lambda item: (item[0].__name__, item[1]),
            )
        )
        + max(
            len(buckets_by_field[field][key])
            for key in sorted(
                buckets_by_field[field],
                key=lambda item: (item[0].__name__, item[1]),
            )
        )
        < len(candidate_indexes)
    )
    if not useful_fields:
        return None
    selected_field = min(
        useful_fields,
        key=lambda field: (
            len(candidate_indexes)
            - sum(
                len(buckets_by_field[field][key])
                for key in sorted(
                    buckets_by_field[field],
                    key=lambda item: (item[0].__name__, item[1]),
                )
            )
            + max(
                len(buckets_by_field[field][key])
                for key in sorted(
                    buckets_by_field[field],
                    key=lambda item: (item[0].__name__, item[1]),
                )
            ),
            -len(buckets_by_field[field]),
            field,
        ),
    )
    selected_buckets = buckets_by_field[selected_field]
    constrained = {
        index
        for key in sorted(
            selected_buckets,
            key=lambda item: (item[0].__name__, item[1]),
        )
        for index in selected_buckets[key]
    }
    unconstrained = tuple(
        index for index in candidate_indexes if index not in constrained
    )
    candidates_by_value = {
        key: tuple(
            index
            for index in candidate_indexes
            if index in unconstrained or index in selected_buckets[key]
        )
        for key in sorted(
            selected_buckets,
            key=lambda item: (item[0].__name__, item[1]),
        )
    }
    return selected_field, candidates_by_value, unconstrained


def _cached_best_tier_equality_filter(
    tiers: tuple[DrmSelectionTier, ...],
    candidate_indexes: tuple[int, ...],
    *,
    excluded_fields: tuple[str, ...],
) -> _TierEqualityFilter | None:
    key = (id(tiers), candidate_indexes, excluded_fields)
    cached = _TIER_FILTER_CACHE.get(key)
    if cached is not None and cached[0] is tiers:
        return cached[1]
    result = _best_tier_equality_filter(
        tiers,
        candidate_indexes,
        excluded_fields=excluded_fields,
    )
    if len(_TIER_FILTER_CACHE) >= _TIER_FILTER_CACHE_LIMIT:
        _TIER_FILTER_CACHE.pop(next(iter(_TIER_FILTER_CACHE)))
    _TIER_FILTER_CACHE[key] = tiers, result
    return result


def _tier_partition_fact_value(facts: object, field: str) -> object:
    if not hasattr(facts, "get"):
        return None
    value = facts.get(field)
    if value is None and field == "interaction_kind":
        return facts.get("interaction_measurement_kind")
    if value is None and field == "last_action_is_directional_interface_action":
        return bool(
            facts.get("last_action_ref")
            and not facts.get("last_action_was_nondirectional_discrete")
            and not facts.get("last_action_is_point_interface_action")
        )
    if (
        value is None
        and field == "is_boundary_collinear_component"
        and facts.get("interaction_measurement_kind") == "discrete"
    ):
        return False
    if value is None and field == "dynamic_goal_binding_satisfied":
        return facts.get("dynamic_goal_selection_present") is not True
    return value


def _condition_holds(condition: DrmFactCondition, facts: object) -> bool:
    if not hasattr(facts, "get"):
        return False
    value = facts.get(condition.field)
    # Match the declared presence semantics used by DRM fact projections.
    # False, zero and empty collections are values; only absent/null is absent.
    # Check the recorded field before legacy value aliases below can supply one.
    if condition.operator is DrmFactOperator.IS_PRESENT:
        return value is not None
    if condition.operator is DrmFactOperator.IS_ABSENT:
        return value is None
    # `interaction_kind` is a DRM projection vocabulary.  Capabilities only
    # measure the neutral `interaction_measurement_kind`; when a projected
    # selection delta is inspected directly, preserve the same declared
    # semantics by reading that measured source instead of requiring Python to
    # author a cognitive field.
    if value is None and condition.field == "interaction_kind":
        value = facts.get("interaction_measurement_kind")
    if value is None and condition.field == "last_action_is_directional_interface_action":
        value = bool(
            facts.get("last_action_ref")
            and not facts.get("last_action_was_nondirectional_discrete")
            and not facts.get("last_action_is_point_interface_action")
        )
    if (
        value is None
        and condition.field == "is_boundary_collinear_component"
        and facts.get("interaction_measurement_kind") == "discrete"
    ):
        value = False
    if value is None and condition.field == "dynamic_goal_binding_satisfied":
        # Legacy direct policy witnesses have no dynamic-frontier transport.
        # The absence-only case leaves the binding unconstrained; an explicit
        # frontier still requires the DRM-projected fact.
        value = facts.get("dynamic_goal_selection_present") is not True
    if value is None and condition.field == "terminal_access_candidate_ref_present":
        value = bool(facts.get("terminal_access_candidate_ref"))
    if condition.operator is DrmFactOperator.EQUALS:
        return value == condition.expected
    if condition.operator is DrmFactOperator.NOT_EQUALS:
        return value is not None and value != condition.expected
    if condition.operator is DrmFactOperator.EQUALS_REFERENCE_FIELD:
        return value == facts.get(condition.reference_field)
    if condition.operator is DrmFactOperator.NOT_EQUALS_REFERENCE_FIELD:
        reference = facts.get(condition.reference_field)
        return value is not None and reference is not None and value != reference
    if condition.operator is DrmFactOperator.GREATER_THAN_REFERENCE_FIELD:
        reference = facts.get(condition.reference_field)
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and isinstance(reference, int)
            and not isinstance(reference, bool)
            and value > reference
        )
    if condition.operator is DrmFactOperator.MEMBER_OF_REFERENCE_FIELD:
        reference = facts.get(condition.reference_field)
        return isinstance(reference, (tuple, list, set, frozenset)) and value in reference
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    if condition.operator is DrmFactOperator.GREATER_THAN:
        return value > condition.expected
    if condition.operator is DrmFactOperator.BETWEEN_INCLUSIVE:
        return condition.lower <= value <= condition.upper
    reference = facts.get(condition.reference_field)
    return (
        isinstance(reference, int)
        and not isinstance(reference, bool)
        and value <= condition.factor * reference
    )


def _compare_fact_values(left: object, right: object) -> int:
    """Compare declared JSON scalars exactly and fail closed otherwise."""

    if left is None and right is None:
        return 0
    if left is None:
        return 1
    if right is None:
        return -1
    if type(left) is not type(right) or not isinstance(left, (bool, int, str)):
        raise ValueError(
            "declared tie-break values must be same-type JSON scalars; "
            f"got {type(left).__name__} and {type(right).__name__}"
        )
    return -1 if left < right else 1 if left > right else 0


_HASH_REFERENCE_TOKEN = re.compile(
    r"(?:^|[.:_-])[0-9a-f]{8,}(?=$|[.:_-])",
    re.IGNORECASE,
)


def _compare_tie_breaker_values(
    tie_breaker: object,
    left: object,
    right: object,
) -> int:
    """Reject hash-shaped canonical references as decision discriminants."""

    field = str(getattr(tie_breaker, "field", ""))
    if field in {"alternative_ref", "candidate_ref"} and any(
        isinstance(value, str) and _HASH_REFERENCE_TOKEN.search(value)
        for value in (left, right)
    ):
        return 0
    return _compare_fact_values(left, right)
