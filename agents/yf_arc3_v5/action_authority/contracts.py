"""Finite validation of the joint declarative priority/usefulness decision.

DRM and SRC supply every semantic reference.  This module only checks identity,
presence, bounds, and the exact pending affordance when the declared functional
disposition says that the utility test itself must be released now.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from agents.yf_arc3_v5.logos.types import FrozenModel, NonNegativeRevision, Ref


class ActionLifecycleAuthorityProjection(FrozenModel):
    priority_source_state_revision: NonNegativeRevision
    usefulness_source_state_revision: NonNegativeRevision
    frame_ref: Ref
    action_intent_id: Ref
    action_ref: Ref
    candidate_ref: Ref
    pending_activity_ref: Ref
    declared_priority_activity_ref: Ref
    recommended_usefulness_activity_ref: Ref
    active_obligation_ref: Ref
    active_goal_ref: Ref
    committed_plan_ref: Ref | None = None
    discriminating_experiment_ref: Ref | None = None
    intent_lifecycle_engagement_ref: Ref | None = None
    active_terminal_goal_ref: Ref | None = None
    subgoal_ref: Ref | None = None
    goal_dependency_chain_refs: tuple[Ref, ...] = Field(default=(), max_length=16)
    declared_plan_ref: Ref | None = None
    declared_plan_stage_refs: tuple[Ref, ...] = Field(default=(), max_length=16)
    declared_plan_ordered_member_refs: tuple[Ref, ...] = Field(
        default=(), max_length=16
    )
    declared_plan_ordered_member_values: tuple[int, ...] = Field(
        default=(), max_length=16
    )
    declared_plan_current_stage_ref: Ref | None = None
    declared_plan_cursor_ordinal: NonNegativeRevision | None = None
    declared_plan_member_count: NonNegativeRevision | None = None
    reconciliation_checkpoint_ref: Ref
    goal_route_stop_lifecycle_ref: Ref
    pursued_goal_ref: Ref
    engaged_plan_or_experiment_ref: Ref
    stop_condition_ref: Ref
    premise_claim_refs: tuple[Ref, ...] = Field(min_length=1, max_length=32)
    expected_effect_class: Ref
    expected_configuration_or_goal_delta_ref: Ref
    intent_falsifier_refs: tuple[Ref, ...] = Field(min_length=1, max_length=16)
    remaining_action_bound: NonNegativeRevision | None = None
    priority_graph_semantic_hash: Ref
    usefulness_graph_semantic_hash: Ref
    functional_disposition_ref: Ref
    functional_lifecycle_ref: Ref
    active_function_obligation_ref: Ref | None = None
    eligible_affordance_ref: Ref | None = None
    pending_affordance_must_match: bool = False
    functional_expected_effect_ref: Ref | None = None
    functional_falsifier_refs: tuple[Ref, ...] = Field(default=(), max_length=16)
    functional_wake_condition_ref: Ref | None = None
    authorized: bool = True
    schema_version: Ref = "yf_arc3_v5.action_lifecycle_authority_projection.v1"

    @model_validator(mode="after")
    def validate_joint_authority(self) -> "ActionLifecycleAuthorityProjection":
        if not self.authorized:
            raise ValueError("an action authority projection must authorize its exact intent")
        if self.pending_activity_ref != self.declared_priority_activity_ref:
            raise ValueError(
                "pending activity differs from declared priority activity: "
                f"{self.pending_activity_ref} != {self.declared_priority_activity_ref}"
            )
        if self.pending_activity_ref != self.recommended_usefulness_activity_ref:
            raise ValueError("pending activity differs from declared usefulness activity")
        if self.committed_plan_ref is None and self.discriminating_experiment_ref is None:
            raise ValueError("an authorized action requires a committed route or experiment")
        engaged_ref = self.committed_plan_ref or self.discriminating_experiment_ref
        exact_declared_engagement_refs = {
            engaged_ref,
            *(
                (self.intent_lifecycle_engagement_ref,)
                if self.intent_lifecycle_engagement_ref is not None
                else ()
            ),
        }
        if self.engaged_plan_or_experiment_ref not in exact_declared_engagement_refs:
            raise ValueError("declared lifecycle differs from the exact route or experiment")
        if self.pursued_goal_ref != self.active_goal_ref:
            raise ValueError("declared lifecycle differs from the pursued goal")
        if self.declared_plan_ref:
            if not self.active_terminal_goal_ref or not self.subgoal_ref:
                raise ValueError("declared complete plan omits its active goal hierarchy")
            if not {
                self.active_terminal_goal_ref,
                self.active_goal_ref,
                self.subgoal_ref,
            }.issubset(set(self.goal_dependency_chain_refs)):
                raise ValueError("declared complete plan differs from the active goal chain")
            if self.declared_plan_current_stage_ref not in self.declared_plan_stage_refs:
                raise ValueError("declared complete plan omits its current stage")
            if bool(self.declared_plan_ordered_member_refs) != bool(
                self.declared_plan_ordered_member_values
            ):
                raise ValueError(
                    "declared ordered plan members require refs and observed values"
                )
            if self.declared_plan_ordered_member_refs and (
                len(self.declared_plan_ordered_member_refs)
                != self.declared_plan_member_count
                or len(self.declared_plan_ordered_member_values)
                != self.declared_plan_member_count
            ):
                raise ValueError(
                    "declared ordered plan members must match the declared member count"
                )
        if self.stop_condition_ref != self.reconciliation_checkpoint_ref:
            raise ValueError("declared lifecycle differs from the stop condition")
        if self.pending_affordance_must_match:
            if self.eligible_affordance_ref is None:
                raise ValueError("a current utility test requires one eligible affordance")
            if self.eligible_affordance_ref not in {self.action_ref, self.candidate_ref}:
                raise ValueError("pending action differs from the required utility affordance")
        return self
