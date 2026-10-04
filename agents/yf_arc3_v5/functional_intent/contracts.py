"""Finite transport contracts for DRM-authored functional obligations."""

from __future__ import annotations

from pydantic import Field, model_validator

from agents.yf_arc3_v5.logos.types import (
    FrozenModel,
    NonNegativeRevision,
    Ref,
    require_unique,
)


class FunctionalUsefulnessProjection(FrozenModel):
    source_state_revision: NonNegativeRevision
    declared_disposition_ref: Ref
    active_function_obligation_ref: Ref | None = None
    eligible_affordance_ref: Ref | None = None
    pending_affordance_must_match: bool = False
    lifecycle_ref: Ref
    alternative_function_refs: tuple[Ref, ...] = Field(default=(), max_length=16)
    blocking_predicate_refs: tuple[Ref, ...] = Field(default=(), max_length=16)
    beneficiary_goal_refs: tuple[Ref, ...] = Field(default=(), max_length=16)
    opportunity_window_ref: Ref | None = None
    recommended_activity_ref: Ref
    expected_effect_ref: Ref | None = None
    falsifier_refs: tuple[Ref, ...] = Field(default=(), max_length=16)
    wake_condition_ref: Ref | None = None
    graph_semantic_hash: Ref
    schema_version: Ref = "yf_arc3_v5.functional_usefulness_projection.v1"

    @model_validator(mode="after")
    def validate_projection(self) -> "FunctionalUsefulnessProjection":
        for refs, label in (
            (self.alternative_function_refs, "functional alternatives"),
            (self.blocking_predicate_refs, "functional blockers"),
            (self.beneficiary_goal_refs, "functional beneficiaries"),
            (self.falsifier_refs, "functional falsifiers"),
        ):
            require_unique(refs, label)
        if self.active_function_obligation_ref is not None:
            if self.eligible_affordance_ref is None:
                raise ValueError("a function obligation requires one affordance")
            if not self.alternative_function_refs:
                raise ValueError("a function obligation requires live alternatives")
            if self.expected_effect_ref is None or not self.falsifier_refs:
                raise ValueError("a function obligation requires effect and falsifier")
        elif self.eligible_affordance_ref is not None:
            raise ValueError("an affordance cannot be active without an obligation")
        if self.pending_affordance_must_match and self.eligible_affordance_ref is None:
            raise ValueError("a current function test requires one eligible affordance")
        return self


class FunctionalResearchSubjectProjection(FrozenModel):
    subject_ref: Ref
    subject_kind: Ref
    declared_disposition_ref: Ref
    active_function_obligation_ref: Ref | None = None
    alternative_function_refs: tuple[Ref, ...] = Field(min_length=1, max_length=16)
    falsifier_refs: tuple[Ref, ...] = Field(min_length=1, max_length=16)
    wake_condition_ref: Ref
    execution_count: NonNegativeRevision | None = None
    interaction_evidence_present: bool | None = None
    current_declared_test: bool = False
    schema_version: Ref = "yf_arc3_v5.functional_research_subject_projection.v1"

    @model_validator(mode="after")
    def validate_subject_disposition(self) -> "FunctionalResearchSubjectProjection":
        require_unique(self.alternative_function_refs, "function alternatives")
        require_unique(self.falsifier_refs, "function falsifiers")
        if self.subject_kind == "available_action":
            if self.execution_count is None:
                raise ValueError("an available action requires its execution count")
            if self.interaction_evidence_present is not None:
                raise ValueError("an action cannot carry visual interaction evidence")
        elif self.subject_kind == "visible_subject":
            if self.interaction_evidence_present is None:
                raise ValueError("a visible subject requires interaction evidence status")
            if self.execution_count is not None:
                raise ValueError("a visible subject cannot carry an action count")
        else:
            raise ValueError("functional research subject kind is unsupported")
        if self.current_declared_test and self.active_function_obligation_ref is None:
            raise ValueError("a current test requires its declared function obligation")
        return self


class FunctionalResearchCoverageProjection(FrozenModel):
    source_state_revision: NonNegativeRevision
    frame_ref: Ref
    subject_dispositions: tuple[FunctionalResearchSubjectProjection, ...] = Field(
        min_length=1,
        max_length=80,
    )
    graph_semantic_hash: Ref
    schema_version: Ref = "yf_arc3_v5.functional_research_coverage_projection.v1"

    @model_validator(mode="after")
    def validate_complete_coverage(self) -> "FunctionalResearchCoverageProjection":
        require_unique(
            tuple(item.subject_ref for item in self.subject_dispositions),
            "functional research subjects",
        )
        return self


class MechanismUseDispositionProjection(FrozenModel):
    """Finite carrier for one DRM-authored use disposition.

    Every semantic reference is supplied by DRM or transported from one
    canonical mechanism-evidence term.  Validation is deliberately limited to
    completeness, identity, and finite bounds.
    """

    mechanism_ref: Ref
    evidence_contract_ref: Ref
    mechanism_family_ref: Ref
    declared_disposition_ref: Ref
    evidence_refs: tuple[Ref, ...] = Field(min_length=1, max_length=16)
    candidate_goal_refs: tuple[Ref, ...] = Field(default=(), max_length=16)
    candidate_action_refs: tuple[Ref, ...] = Field(default=(), max_length=16)
    beneficiary_goal_refs: tuple[Ref, ...] = Field(default=(), max_length=16)
    precondition_refs: tuple[Ref, ...] = Field(default=(), max_length=16)
    resource_bound_refs: tuple[Ref, ...] = Field(min_length=1, max_length=16)
    observed_resource_bounds: tuple[tuple[Ref, int], ...] = Field(
        default=(), max_length=16
    )
    compatible_mechanism_refs: tuple[Ref, ...] = Field(default=(), max_length=16)
    combination_disposition_ref: Ref
    active_plan_or_experiment_ref: Ref | None = None
    application_stage_refs: tuple[Ref, ...] = Field(default=(), max_length=16)
    application_test_ref: Ref
    utility_test_success_evidence_refs: tuple[Ref, ...] = Field(
        min_length=1, max_length=16
    )
    utility_test_failure_evidence_refs: tuple[Ref, ...] = Field(
        min_length=1, max_length=16
    )
    utility_test_stop_condition_refs: tuple[Ref, ...] = Field(
        min_length=1, max_length=16
    )
    utility_test_reconciliation_ref: Ref
    expected_effect_ref: Ref
    alternative_use_refs: tuple[Ref, ...] = Field(min_length=1, max_length=16)
    falsifier_refs: tuple[Ref, ...] = Field(min_length=1, max_length=16)
    blocking_predicate_refs: tuple[Ref, ...] = Field(default=(), max_length=16)
    wake_condition_ref: Ref
    last_reconciled_contribution_ref: Ref | None = None
    last_reconciled_contribution_reason_ref: Ref | None = None
    last_reconciled_contribution_is_useful: bool | None = None
    last_reconciled_application_test_refs: tuple[Ref, ...] = Field(
        default=(), max_length=16
    )
    last_reconciled_utility_success_evidence_refs: tuple[Ref, ...] = Field(
        default=(), max_length=16
    )
    last_reconciled_utility_failure_evidence_refs: tuple[Ref, ...] = Field(
        default=(), max_length=16
    )
    last_reconciled_utility_stop_condition_refs: tuple[Ref, ...] = Field(
        default=(), max_length=16
    )
    last_reconciled_utility_reconciliation_refs: tuple[Ref, ...] = Field(
        default=(), max_length=16
    )
    last_reconciled_completed_level_boundary: bool | None = None
    last_reconciled_expected_route_delta_observed: bool | None = None
    last_reconciled_exact_context_observation_count: (
        NonNegativeRevision | None
    ) = None
    last_reconciled_objective_stop_configuration_revisited: bool | None = None
    last_reconciled_active_safety_gate_ref: Ref | None = None
    last_reconciled_declared_causal_class_ref: Ref | None = None
    last_reconciled_effect_signature_novelty_ref: Ref | None = None
    last_reconciled_observable_change_scope_ref: Ref | None = None
    application_replay_eligible: bool
    application_replay_disposition_ref: Ref
    application_replay_blocking_predicate_refs: tuple[Ref, ...] = Field(
        default=(), max_length=16
    )
    application_replay_wake_condition_ref: Ref
    application_replay_falsifier_refs: tuple[Ref, ...] = Field(
        min_length=1, max_length=16
    )
    exact_failed_application_reconciled: bool = False
    active_in_current_plan: bool = False
    goal_plan_gate_ref: Ref | None = None
    goal_plan_goal_kind_ref: Ref | None = None
    goal_plan_gap_dimension_ref: Ref | None = None
    goal_plan_subgoal_ref: Ref | None = None
    goal_plan_observed_value_ref: Ref | None = None
    goal_plan_target_value_ref: Ref | None = None
    quantity_gate_required: bool = False
    schema_version: Ref = "yf_arc3_v5.mechanism_use_disposition_projection.v5"

    @model_validator(mode="after")
    def validate_use_disposition(self) -> "MechanismUseDispositionProjection":
        for refs, label in (
            (self.evidence_refs, "mechanism evidence"),
            (self.candidate_goal_refs, "mechanism candidate goals"),
            (self.candidate_action_refs, "mechanism candidate actions"),
            (self.beneficiary_goal_refs, "mechanism beneficiary goals"),
            (self.precondition_refs, "mechanism preconditions"),
            (self.resource_bound_refs, "mechanism resource bounds"),
            (self.compatible_mechanism_refs, "compatible mechanisms"),
            (self.application_stage_refs, "mechanism application stages"),
            (
                self.utility_test_success_evidence_refs,
                "mechanism utility-test success evidence",
            ),
            (
                self.utility_test_failure_evidence_refs,
                "mechanism utility-test failure evidence",
            ),
            (
                self.utility_test_stop_condition_refs,
                "mechanism utility-test stop conditions",
            ),
            (self.alternative_use_refs, "mechanism use alternatives"),
            (self.falsifier_refs, "mechanism use falsifiers"),
            (self.blocking_predicate_refs, "mechanism use blockers"),
            (
                self.last_reconciled_application_test_refs,
                "last reconciled mechanism application tests",
            ),
            (
                self.last_reconciled_utility_success_evidence_refs,
                "last reconciled mechanism utility success evidence",
            ),
            (
                self.last_reconciled_utility_failure_evidence_refs,
                "last reconciled mechanism utility failure evidence",
            ),
            (
                self.last_reconciled_utility_stop_condition_refs,
                "last reconciled mechanism utility stop conditions",
            ),
            (
                self.last_reconciled_utility_reconciliation_refs,
                "last reconciled mechanism utility reconciliations",
            ),
            (
                self.application_replay_blocking_predicate_refs,
                "mechanism application replay blockers",
            ),
            (
                self.application_replay_falsifier_refs,
                "mechanism application replay falsifiers",
            ),
        ):
            require_unique(refs, label)
        require_unique(
            tuple(ref for ref, _value in self.observed_resource_bounds),
            "observed mechanism resource bounds",
        )
        if self.mechanism_ref in self.compatible_mechanism_refs:
            raise ValueError("a mechanism cannot be listed as compatible with itself")
        last_reconciled_fields_present = (
            self.last_reconciled_contribution_ref is not None,
            self.last_reconciled_contribution_reason_ref is not None,
            self.last_reconciled_contribution_is_useful is not None,
            bool(self.last_reconciled_application_test_refs),
            bool(self.last_reconciled_utility_success_evidence_refs),
            bool(self.last_reconciled_utility_failure_evidence_refs),
            bool(self.last_reconciled_utility_stop_condition_refs),
            bool(self.last_reconciled_utility_reconciliation_refs),
        )
        if any(last_reconciled_fields_present) and not all(
            last_reconciled_fields_present
        ):
            raise ValueError(
                "last reconciled mechanism application context must be complete or absent"
            )
        last_observation_fields_present = (
            self.last_reconciled_completed_level_boundary is not None,
            self.last_reconciled_expected_route_delta_observed is not None,
            self.last_reconciled_exact_context_observation_count is not None,
            self.last_reconciled_objective_stop_configuration_revisited
            is not None,
            self.last_reconciled_active_safety_gate_ref is not None,
        )
        if any(last_observation_fields_present) and not all(
            last_observation_fields_present
        ):
            raise ValueError(
                "last reconciled mechanism observation must be complete or absent"
            )
        if all(last_observation_fields_present) and not all(
            last_reconciled_fields_present
        ):
            raise ValueError(
                "last reconciled mechanism observation requires its application context"
            )
        if self.exact_failed_application_reconciled:
            if self.application_replay_eligible:
                raise ValueError(
                    "an exact failed mechanism application cannot remain replay eligible"
                )
            if self.application_test_ref not in (
                self.last_reconciled_application_test_refs
            ):
                raise ValueError(
                    "an exact failed mechanism application requires the same test identity"
                )
            if not (
                self.last_reconciled_completed_level_boundary is False
                and self.last_reconciled_expected_route_delta_observed is False
                and self.last_reconciled_objective_stop_configuration_revisited
                is True
            ):
                raise ValueError(
                    "an exact failed mechanism application requires reconciled failure evidence"
                )
        if self.active_in_current_plan:
            if self.active_plan_or_experiment_ref is None:
                raise ValueError("an active mechanism use requires one engaged plan")
            if not self.beneficiary_goal_refs or not self.application_stage_refs:
                raise ValueError(
                    "an active mechanism use requires beneficiary and application stage"
                )
            if not all(
                (
                    self.goal_plan_gate_ref,
                    self.goal_plan_goal_kind_ref,
                    self.goal_plan_gap_dimension_ref,
                    self.goal_plan_subgoal_ref,
                    self.goal_plan_observed_value_ref,
                    self.goal_plan_target_value_ref,
                )
            ):
                raise ValueError(
                    "an active mechanism use requires its DRM-authored goal-plan parent"
                )
        elif any(
            (
                self.goal_plan_gate_ref,
                self.goal_plan_goal_kind_ref,
                self.goal_plan_gap_dimension_ref,
                self.goal_plan_subgoal_ref,
                self.goal_plan_observed_value_ref,
                self.goal_plan_target_value_ref,
            )
        ):
            raise ValueError("an inactive mechanism use cannot expose an active goal-plan parent")
        return self


class MechanismUtilizationPlanProjection(FrozenModel):
    """Complete bounded plan over every canonically understood mechanism."""

    source_state_revision: NonNegativeRevision
    frame_ref: Ref
    active_goal_ref: Ref | None = None
    active_plan_or_experiment_ref: Ref | None = None
    understood_mechanism_refs: tuple[Ref, ...] = Field(default=(), max_length=64)
    use_dispositions: tuple[MechanismUseDispositionProjection, ...] = Field(
        default=(), max_length=64
    )
    coverage_status_ref: Ref
    non_blocking_contract_ref: Ref
    action_release_authority_unchanged: bool = True
    graph_semantic_hash: Ref
    schema_version: Ref = "yf_arc3_v5.mechanism_utilization_plan_projection.v1"

    @model_validator(mode="after")
    def validate_complete_mechanism_plan(self) -> "MechanismUtilizationPlanProjection":
        require_unique(self.understood_mechanism_refs, "understood mechanisms")
        item_refs = tuple(item.mechanism_ref for item in self.use_dispositions)
        require_unique(item_refs, "mechanism use dispositions")
        if item_refs != self.understood_mechanism_refs:
            raise ValueError(
                "every understood mechanism requires exactly one ordered use disposition"
            )
        if not self.action_release_authority_unchanged:
            raise ValueError("a mechanism-use projection cannot release an action")
        return self
