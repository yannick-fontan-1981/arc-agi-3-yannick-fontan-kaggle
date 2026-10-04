"""Typed canonical records surrounding the nine Logos operators."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from agents.yf_arc3_v5.logos.types import (
    Cardinality,
    EffectClass,
    FrozenMap,
    FrozenModel,
    NonNegativeRevision,
    OperatorName,
    PositiveSequence,
    Ref,
    require_unique,
    stable_digest,
)


class OperationStatus(str, Enum):
    COMPLETED = "completed"
    BLOCKED = "blocked"
    UNRESOLVED = "unresolved"
    FAILED = "failed"


class WorkflowStatus(str, Enum):
    READY = "ready"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEFERRED = "deferred"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


class ViabilityStatus(str, Enum):
    VIABLE = "viable"
    CONDITIONAL = "conditional"
    BLOCKED = "blocked"
    UNDETERMINED = "undetermined"


class Reversibility(str, Enum):
    REVERSIBLE = "reversible"
    RECOVERABLE = "recoverable"
    IRREVERSIBLE = "irreversible"
    UNKNOWN = "unknown"


class CommitmentKind(str, Enum):
    COMMIT = "commit"
    REVISE = "revise"
    DEFER = "defer"
    REJECT = "reject"


class AuthorityKind(str, Enum):
    RULE = "rule"
    MODEL = "model"
    CRITERION = "criterion"
    POLICY = "policy"


class AppliedPrincipleStatus(str, Enum):
    PROPOSED = "proposed"
    SUPPORTED = "supported"
    ESTABLISHED = "established"
    DEFEATED = "defeated"


class WorkflowLifecycleKind(str, Enum):
    STARTED = "started"
    STEP_CHANGED = "step_changed"
    SUSPENDED = "suspended"
    RESUMED = "resumed"
    COMPLETED = "completed"
    DEFERRED = "deferred"
    BLOCKED = "blocked"
    FAILED = "failed"


class WorkflowBlockKind(str, Enum):
    STEP = "step"
    BODY = "body"
    ELSE = "else"
    MATCH_ARM = "match_arm"
    FOR_EACH = "for_each"
    WHILE = "while"
    TRANSACTION = "transaction"


class LineagedRecord(FrozenModel):
    record_kind: Ref
    id: Ref
    source_unit: Ref
    source_hash: Ref
    state_revision: NonNegativeRevision
    workflow_definition_id: Ref | None = None
    workflow_instance_id: Ref | None = None
    workflow_step_id: Ref | None = None
    timeline_id: Ref | None = None
    frame_id: Ref | None = None
    provenance: tuple[Ref, ...] = ()
    schema_version: Ref

    @model_validator(mode="after")
    def validate_lineage(self) -> "LineagedRecord":
        require_unique(self.provenance, "provenance")
        if self.workflow_step_id and not self.workflow_instance_id:
            raise ValueError("workflow_step_id requires workflow_instance_id")
        return self


class AlternativeSet(FrozenModel):
    selected: Ref | None = None
    live_alternatives: tuple[Ref, ...] = Field(min_length=1)
    cardinality: Cardinality = Cardinality.SINGULAR
    schema_version: Ref = "yf_arc3_v5.alternative_set.v1"

    @model_validator(mode="after")
    def validate_alternatives(self) -> "AlternativeSet":
        require_unique(self.live_alternatives, "live_alternatives")
        if self.selected and self.selected not in self.live_alternatives:
            raise ValueError("selected must be one of the live alternatives")
        if self.cardinality is Cardinality.SPECIAL and len(self.live_alternatives) < 2:
            raise ValueError("special cardinality requires multiple live alternatives")
        return self

    def revise_falsified_scope(
        self,
        falsified_alternative_refs: tuple[Ref, ...],
    ) -> "AlternativeSet":
        """Mechanically remove only explicitly named falsified alternatives."""

        require_unique(falsified_alternative_refs, "falsified_alternative_refs")
        unknown = tuple(
            ref
            for ref in falsified_alternative_refs
            if ref not in self.live_alternatives
        )
        if unknown:
            raise ValueError(
                "falsified alternatives must belong to the current live scope: "
                f"{unknown!r}"
            )
        retained = tuple(
            ref
            for ref in self.live_alternatives
            if ref not in falsified_alternative_refs
        )
        if not retained:
            raise ValueError("falsification cannot remove every live alternative")
        return AlternativeSet(
            selected=self.selected if self.selected in retained else None,
            live_alternatives=retained,
            cardinality=(
                Cardinality.SPECIAL if len(retained) > 1 else Cardinality.SINGULAR
            ),
        )


class UnresolvedQuestion(FrozenModel):
    id: Ref
    statement: Ref
    scope: FrozenMap = Field(default_factory=FrozenMap)
    missing_evidence: tuple[Ref, ...] = ()
    falsifiers: tuple[Ref, ...] = ()
    alternatives: AlternativeSet | None = None
    schema_version: Ref = "yf_arc3_v5.unresolved_question.v1"


class OperatorInvocation(LineagedRecord):
    record_kind: Literal["operator_invocation"] = "operator_invocation"
    operator: OperatorName
    input_refs: tuple[Ref, ...] = ()
    authority_ref: Ref | None = None
    authority_source_unit: Ref | None = None
    authority_source_hash: Ref | None = None
    attributes: FrozenMap = Field(default_factory=FrozenMap)
    schema_version: Ref = "yf_arc3_v5.operator_invocation.v1"

    @model_validator(mode="after")
    def validate_authority_lineage(self) -> "OperatorInvocation":
        if bool(self.authority_source_unit) != bool(self.authority_source_hash):
            raise ValueError("authority source unit and hash must appear together")
        if self.authority_source_hash and not self.authority_ref:
            raise ValueError("authority source lineage requires authority_ref")
        return self


class OperatorResult(LineagedRecord):
    record_kind: Literal["operator_result"] = "operator_result"
    invocation_id: Ref
    operator: OperatorName
    status: OperationStatus
    output_refs: tuple[Ref, ...] = ()
    reason_refs: tuple[Ref, ...] = ()
    attributes: FrozenMap = Field(default_factory=FrozenMap)
    schema_version: Ref = "yf_arc3_v5.operator_result.v1"


class RuleOrModel(FrozenModel):
    id: Ref
    kind: AuthorityKind
    input_schema: Ref
    output_schema: Ref
    reads: tuple[Ref, ...] = ()
    effect_class: EffectClass = EffectClass.PURE
    applicability: tuple[Ref, ...] = ()
    invariants: tuple[Ref, ...] = ()
    falsifiers: tuple[Ref, ...] = ()
    pure: bool = True
    source_unit: Ref
    source_hash: Ref
    generalization_layer: Ref
    schema_version: Ref = "yf_arc3_v5.rule_or_model.v1"

    @model_validator(mode="after")
    def validate_purity(self) -> "RuleOrModel":
        effect_is_pure = self.effect_class is EffectClass.PURE
        if self.pure != effect_is_pure:
            raise ValueError("purity flag and effect class must agree")
        return self


class CapabilityDefinition(FrozenModel):
    id: Ref
    input_schema: Ref
    output_schema: Ref
    reads: tuple[Ref, ...] = ()
    effect_class: EffectClass
    applicability: tuple[Ref, ...] = ()
    invariants: tuple[Ref, ...] = ()
    falsifiers: tuple[Ref, ...] = ()
    source_unit: Ref
    source_hash: Ref
    generalization_layer: Ref
    schema_version: Ref = "yf_arc3_v5.capability_definition.v1"


class WorkflowControlFrame(FrozenModel):
    """One durable interpreter frame; executable statements stay in compiled SRC."""

    block_kind: WorkflowBlockKind
    owner_id: Ref
    next_statement_index: NonNegativeRevision = 0
    iteration_index: NonNegativeRevision = 0
    control_data: FrozenMap = Field(default_factory=FrozenMap)
    schema_version: Ref = "yf_arc3_v5.workflow_control_frame.v1"


class WorkflowInstanceRecord(LineagedRecord):
    record_kind: Literal["workflow_instance"] = "workflow_instance"
    checkpoint_sequence: PositiveSequence
    status: WorkflowStatus
    current_step_id: Ref
    current_statement_id: Ref | None = None
    control_stack: tuple[WorkflowControlFrame, ...] = ()
    local_bindings: FrozenMap = Field(default_factory=FrozenMap)
    local_binding_updates: FrozenMap = Field(default_factory=FrozenMap)
    local_binding_removals: tuple[Ref, ...] = ()
    local_bindings_complete: bool = True
    pending_expectation_refs: tuple[Ref, ...] = ()
    module_source_hash: Ref
    awaiting_frame_binding: Ref | None = None
    pending_action_intent_id: Ref | None = None
    pending_action_release_permit_id: Ref | None = None
    pending_environment_adapter_ref: Ref | None = None
    pending_world_context: FrozenMap | None = None
    pre_action_snapshot_digest: Ref | None = None
    suspension_frame_id: Ref | None = None
    consumed_observation_refs: tuple[Ref, ...] = ()
    terminal_result: Ref | None = None
    schema_version: Ref = "yf_arc3_v5.workflow_instance.v1"

    @model_validator(mode="after")
    def validate_checkpoint(self) -> "WorkflowInstanceRecord":
        require_unique(self.local_binding_removals, "local binding removals")
        if self.local_bindings_complete:
            if self.local_binding_updates or self.local_binding_removals:
                raise ValueError("complete checkpoint cannot also carry binding deltas")
        elif self.local_bindings:
            raise ValueError("delta checkpoint cannot repeat complete local bindings")
        if set(self.local_binding_updates) & set(self.local_binding_removals):
            raise ValueError("one binding cannot be both updated and removed")
        require_unique(self.pending_expectation_refs, "pending expectations")
        require_unique(self.consumed_observation_refs, "consumed observations")
        for description, source_hash in (
            ("workflow transitive source hash", self.source_hash),
            ("workflow module source hash", self.module_source_hash),
        ):
            if len(source_hash) != 64 or any(
                character not in "0123456789abcdef" for character in source_hash
            ):
                raise ValueError(f"{description} must be a lowercase SHA-256 digest")
        terminal = self.status in {
            WorkflowStatus.DEFERRED,
            WorkflowStatus.BLOCKED,
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
        }
        if terminal:
            if self.control_stack:
                raise ValueError("terminal workflow cannot retain a control stack")
            if not self.terminal_result:
                raise ValueError("terminal workflow requires a typed result")
        elif not self.control_stack:
            raise ValueError("non-terminal workflow requires a control stack")
        pending_values = (
            self.pending_action_intent_id,
            self.pending_action_release_permit_id,
            self.pending_environment_adapter_ref,
            self.pending_world_context,
            self.pre_action_snapshot_digest,
        )
        if any(pending_values) and not all(pending_values):
            raise ValueError("pending action lineage must be complete")
        if self.status is WorkflowStatus.SUSPENDED:
            if not all(pending_values):
                raise ValueError("suspended workflow requires pending action lineage")
            if not self.awaiting_frame_binding or not self.suspension_frame_id:
                raise ValueError(
                    "suspended workflow requires an exact frame continuation"
                )
        elif self.awaiting_frame_binding or self.suspension_frame_id:
            raise ValueError("only a suspended workflow may await a frame")
        return self


class WorkflowBindingValueRecord(LineagedRecord):
    """One content-addressed immutable value shared by checkpoint deltas."""

    record_kind: Literal["workflow_binding_value"] = "workflow_binding_value"
    value_digest: Ref
    payload: FrozenMap
    schema_version: Ref = "yf_arc3_v5.workflow_binding_value.v1"

    @model_validator(mode="after")
    def validate_value_digest(self) -> "WorkflowBindingValueRecord":
        if set(self.payload) != {"value"}:
            raise ValueError("workflow binding value payload must contain only value")
        if self.value_digest != stable_digest(self.payload["value"]):
            raise ValueError("workflow binding value digest mismatch")
        return self


class CompiledSelectionDecisionRecord(LineagedRecord):
    """Canonical output of one source-bound compiled SRC/DRM selection path."""

    record_kind: Literal["compiled_selection_decision"] = "compiled_selection_decision"
    workflow_definition_id: Ref
    workflow_instance_id: Ref
    module_source_hash: Ref
    drm_source_hashes: tuple[Ref, ...] = Field(min_length=1, max_length=8)
    path: tuple[tuple[Ref, Ref], ...] = Field(min_length=1, max_length=8)
    terminal_ref: Ref
    assessment: FrozenMap
    descriptive_facts: FrozenMap
    schema_version: Ref = "yf_arc3_v5.compiled_selection_decision.v1"

    @model_validator(mode="after")
    def validate_compiled_selection(self) -> "CompiledSelectionDecisionRecord":
        for source_hash in (self.source_hash, self.module_source_hash, *self.drm_source_hashes):
            if len(source_hash) != 64 or any(char not in "0123456789abcdef" for char in source_hash):
                raise ValueError("compiled selection requires exact source hashes")
        if self.workflow_step_id != self.path[-1][0]:
            raise ValueError("compiled selection terminal step must end its path")
        alternatives = self.assessment.get("alternatives")
        if not isinstance(alternatives, FrozenMap) or alternatives.get("selected") != self.terminal_ref:
            raise ValueError("compiled selection assessment and terminal disagree")
        if not self.assessment.get("reason_refs"):
            raise ValueError("compiled selection requires inspectable reasons")
        return self


class CompiledSceneAssimilationRecord(LineagedRecord):
    """Small source-bound witness for a mechanically measured scene."""

    record_kind: Literal["compiled_scene_assimilation"] = "compiled_scene_assimilation"
    workflow_definition_id: Ref
    workflow_instance_id: Ref
    workflow_step_id: Ref
    frame_id: Ref
    module_source_hash: Ref
    raw_input_ref: Ref
    scene_scope_ref: Literal["scene_scope.full", "scene_scope.block_grid"] = (
        "scene_scope.full"
    )
    action_conditioned_grid_evidence: bool
    retained_grid_geometry: tuple[str | int, ...] | None = None
    retained_grid_separator_consistency_ppm: int = 0
    path: tuple[Ref, Ref, Ref]
    scene_digest: Ref
    schema_version: Ref = "yf_arc3_v5.compiled_scene_assimilation.v1"

    @model_validator(mode="after")
    def validate_compiled_scene(self) -> "CompiledSceneAssimilationRecord":
        for source_hash in (self.source_hash, self.module_source_hash, self.scene_digest):
            if len(source_hash) != 64 or any(char not in "0123456789abcdef" for char in source_hash):
                raise ValueError("compiled scene requires exact source and scene hashes")
        if self.workflow_step_id != self.path[-1]:
            raise ValueError("compiled scene terminal step must end its path")
        return self


class CompiledTemporalTrackingRecord(LineagedRecord):
    """Small witness for source-declared mechanical temporal identity matching."""

    record_kind: Literal["compiled_temporal_tracking"] = "compiled_temporal_tracking"
    workflow_definition_id: Ref
    workflow_instance_id: Ref
    workflow_step_id: Ref
    frame_id: Ref
    module_source_hash: Ref
    previous_frame_ref: Ref | None = None
    action_ref: Ref
    reset_correspondence: bool
    tracking_scope_ref: Literal[
        "tracking_scope.full_blocks",
        "tracking_scope.retained_grid_unique_multivalue_cells",
        "tracking_scope.retained_grid_informative_cells",
        "tracking_scope.retained_grid_delta_cells",
    ] = "tracking_scope.full_blocks"
    scoped_component_refs: tuple[Ref, ...] = Field(default=(), max_length=64)
    scene_digest: Ref
    tracking_input_digest: Ref
    tracking_digest: Ref
    schema_version: Ref = "yf_arc3_v5.compiled_temporal_tracking.v2"

    @model_validator(mode="after")
    def validate_compiled_tracking(self) -> "CompiledTemporalTrackingRecord":
        for source_hash in (
            self.source_hash, self.module_source_hash,
            self.scene_digest, self.tracking_input_digest, self.tracking_digest,
        ):
            if len(source_hash) != 64 or any(char not in "0123456789abcdef" for char in source_hash):
                raise ValueError("compiled tracking requires exact source and value hashes")
        if self.previous_frame_ref == self.frame_id:
            raise ValueError("compiled tracking predecessor must be an earlier frame")
        if (
            self.tracking_scope_ref == "tracking_scope.full_blocks"
            and self.scoped_component_refs
        ):
            raise ValueError("full tracking scope must not duplicate component refs")
        if (
            self.tracking_scope_ref != "tracking_scope.full_blocks"
            and not self.scoped_component_refs
        ):
            raise ValueError("retained-grid tracking scope requires exact component refs")
        return self


class CompiledInteractionEffectRecord(LineagedRecord):
    """Compact witness for one source-declared interaction classification."""

    record_kind: Literal["compiled_interaction_effect"] = (
        "compiled_interaction_effect"
    )
    workflow_definition_id: Ref
    workflow_instance_id: Ref
    workflow_step_id: Ref
    frame_id: Ref
    module_source_hash: Ref
    drm_source_hashes: tuple[Ref, ...] = Field(min_length=1, max_length=8)
    path: tuple[Ref, ...] = Field(min_length=4, max_length=5)
    effect_ref: Ref
    transient_evidence_ref: Ref
    measurement_digest: Ref
    contextual_claim_refs: tuple[Ref, ...] = Field(min_length=1)
    transient_claim_refs: tuple[Ref, ...] = ()
    state_revision_before: NonNegativeRevision
    schema_version: Ref = "yf_arc3_v5.compiled_interaction_effect.v1"

    @model_validator(mode="after")
    def validate_compiled_interaction_effect(
        self,
    ) -> "CompiledInteractionEffectRecord":
        for source_hash in (
            self.source_hash,
            self.module_source_hash,
            *self.drm_source_hashes,
            self.measurement_digest,
        ):
            if len(source_hash) != 64 or any(
                character not in "0123456789abcdef" for character in source_hash
            ):
                raise ValueError(
                    "compiled interaction effect requires exact source hashes"
                )
        if self.workflow_step_id != self.path[-1]:
            raise ValueError("compiled interaction path must end at its recorded step")
        if self.state_revision_before > self.state_revision:
            raise ValueError("compiled interaction revisions are reversed")
        require_unique(self.contextual_claim_refs, "contextual interaction claims")
        require_unique(self.transient_claim_refs, "transient interaction claims")
        return self


class StateChange(FrozenModel):
    target_ref: Ref
    before_digest: Ref | None = None
    after_digest: Ref
    update_policy_ref: Ref
    invalidates: tuple[Ref, ...] = ()
    recalculates: tuple[Ref, ...] = ()
    schema_version: Ref = "yf_arc3_v5.state_change.v1"


class StateChangeProposal(LineagedRecord):
    record_kind: Literal["state_change_proposal"] = "state_change_proposal"
    source_change_id: Ref
    changes: tuple[StateChange, ...] = Field(min_length=1)
    evidence_refs: tuple[Ref, ...] = Field(min_length=1)
    schema_version: Ref = "yf_arc3_v5.state_change_proposal.v1"


class ViabilityAssessment(LineagedRecord):
    record_kind: Literal["viability_assessment"] = "viability_assessment"
    change_ref: Ref
    status: ViabilityStatus
    principles_preserved: tuple[Ref, ...] = ()
    principles_violated: tuple[Ref, ...] = ()
    unresolved_conditions: tuple[Ref, ...] = ()
    missing_evidence: tuple[Ref, ...] = ()
    resource_consequences: FrozenMap = Field(default_factory=FrozenMap)
    reversibility: Reversibility = Reversibility.UNKNOWN
    alternatives_preserved: tuple[Ref, ...] = ()
    schema_version: Ref = "yf_arc3_v5.viability_assessment.v1"


class CommitmentDecision(LineagedRecord):
    record_kind: Literal["commitment_decision"] = "commitment_decision"
    change_ref: Ref
    decision: CommitmentKind
    viability_assessment_id: Ref
    policy_ref: Ref
    reason_refs: tuple[Ref, ...] = Field(min_length=1)
    alternatives_preserved: tuple[Ref, ...] = ()
    schema_version: Ref = "yf_arc3_v5.commitment_decision.v1"


class PropagationReport(LineagedRecord):
    record_kind: Literal["propagation_report"] = "propagation_report"
    source_change_id: Ref
    affected_refs: tuple[Ref, ...] = ()
    invalidated_refs: tuple[Ref, ...] = ()
    recomputed_refs: tuple[Ref, ...] = ()
    fixpoint_reached: bool
    blocking_contradiction_ref: Ref | None = None
    schema_version: Ref = "yf_arc3_v5.propagation_report.v1"

    @model_validator(mode="after")
    def validate_stop_result(self) -> "PropagationReport":
        if self.fixpoint_reached and self.blocking_contradiction_ref:
            raise ValueError(
                "fixpoint and blocking contradiction are mutually exclusive"
            )
        if not self.fixpoint_reached and not self.blocking_contradiction_ref:
            raise ValueError(
                "non-fixpoint propagation requires a blocking contradiction"
            )
        return self


class ActionIntent(LineagedRecord):
    record_kind: Literal["action_intent"] = "action_intent"
    action_ref: Ref
    available_action_ref: Ref
    # Optional immutable payload for a precommitted action series.  It keeps
    # action-specific coordinates/data with the intent so CONTINUE does not
    # rebuild a WorldContext between triggers.
    action_data: FrozenMap = Field(default_factory=FrozenMap)
    # A series identity and ordinal are transport lineage, not a new semantic
    # choice.  They let the action boundary verify a predeclared burst while
    # keeping every primitive intent independently auditable.
    declared_series_ref: Ref | None = None
    declared_series_index: NonNegativeRevision | None = None
    world_context_ref: Ref
    commitment_decision_id: Ref
    expectation_refs: tuple[Ref, ...] = ()
    # Declarative grounding payload.  Legacy callers may leave it empty while
    # the migration progressively closes the anti-errance firewall.
    active_goal_ref: Ref | None = None
    active_terminal_goal_ref: Ref | None = None
    declared_terminal_goal_ref: Ref | None = None
    subgoal_ref: Ref | None = None
    goal_graph_edge_refs: tuple[Ref, ...] = ()
    goal_dependency_chain_refs: tuple[Ref, ...] = ()
    declared_dependency_chain_refs: tuple[Ref, ...] = ()
    advances_active_goal_chain: bool | None = None
    inherited_priority_from_terminal_goal: bool = False
    priority_class_ref: Ref | None = None
    selection_principle_ref: Ref | None = None
    premise_claim_refs: tuple[Ref, ...] = ()
    # Source may require a subset to resolve to live canonical proofs rather
    # than static symbolic declarations. The obligation travels with intent.
    canonical_premise_claim_refs: tuple[Ref, ...] = Field(default=(), max_length=64)
    expected_effect_class: Ref | None = None
    expected_configuration_or_goal_delta_ref: Ref | None = None
    falsifier_refs: tuple[Ref, ...] = ()
    committed_plan_ref: Ref | None = None
    discriminating_experiment_ref: Ref | None = None
    lifecycle_engagement_ref: Ref | None = None
    declared_plan_ref: Ref | None = None
    declared_plan_stage_refs: tuple[Ref, ...] = ()
    declared_plan_ordered_member_refs: tuple[Ref, ...] = ()
    declared_plan_ordered_member_values: tuple[int, ...] = ()
    declared_plan_current_stage_ref: Ref | None = None
    declared_plan_cursor_ordinal: NonNegativeRevision | None = None
    declared_plan_member_count: NonNegativeRevision | None = None
    remaining_action_bound: NonNegativeRevision | None = None
    reconciliation_checkpoint_ref: Ref | None = None
    schema_version: Ref = "yf_arc3_v5.action_intent.v1"


class ConcentrationLeaseState(str, Enum):
    """SRC-owned lifecycle states for the unique action-releasing focus."""

    ACQUIRED = "acquired"
    VALIDATED = "validated"
    RELEASED = "released"
    OBSERVED = "observed"
    RECONCILED = "reconciled"
    SUSPENDED = "suspended"
    INTERRUPTED = "interrupted"
    INVALIDATED = "invalidated"
    EXPIRED = "expired"


class ConcentrationOperation(str, Enum):
    """SRC-owned focus operation; vocabulary is separate from lease state."""

    ATTRIBUTE = "attribute"
    MAINTAIN = "maintain"
    ENGAGE = "engage"
    ACTUALIZE = "actualize"
    DELEGATE = "delegate"
    RETURN = "return"
    ESCALATE = "escalate"
    RETROCEDE = "retrocede"
    HANDOFF = "handoff"
    SUSPEND = "suspend"
    WAIT = "wait"
    YIELD = "yield"
    PREEMPT = "preempt"
    RESUME = "resume"
    REORIENT = "reorient"
    RETRY = "retry"
    CLOSE = "close"
    REOPEN = "reopen"
    REVOKE = "revoke"
    RESET = "reset"


class ConcentrationTransformation(str, Enum):
    """Change to the problem or knowledge, never an implicit ownership change."""

    DECOMPOSE = "decompose"
    MUTUALIZE = "mutualize"
    SPLIT = "split"
    REFRAME = "reframe"
    REATTACH = "reattach"
    REVISE_INVALIDATE = "revise_invalidate"
    CAPITALIZE = "capitalize"


class ConcentrationOutcome(str, Enum):
    """Result of the committed path, distinct from mechanism enrichment."""

    SATISFIED = "satisfied"
    REFUTED = "refuted"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    INCONCLUSIVE_BUDGET_EXHAUSTED = "inconclusive_budget_exhausted"
    CANCELLED_ABANDONED = "cancelled_abandoned"
    SUPERSEDED_IRRELEVANT = "superseded_irrelevant"
    TECHNICAL_ERROR = "technical_error"


class ConcentrationInvariant(str, Enum):
    UNIQUE_AUTHORITY = "unique_authority"
    ATOMIC_TRANSFER = "atomic_transfer"
    WHY_PRESERVED = "why_preserved"
    OBLIGATIONS_PRESERVED = "obligations_preserved"
    RESULTS_COUNTED_ONCE = "results_counted_once"
    RESUMPTION_REVALIDATED = "resumption_revalidated"
    BUDGET_INHERITED = "budget_inherited"
    RECONCILE_AT_COMMITTED_PATH_END = "reconcile_at_committed_path_end"
    WAIT_HAS_WAKE_CONDITION = "wait_has_wake_condition"
    HISTORY_PRESERVED = "history_preserved"


class ConcentrationReviewMotive(str, Enum):
    DEADLINE_EXCEEDED = "deadline_exceeded"
    REPETITION_WITHOUT_GAIN = "repetition_without_gain"
    CONTRADICTION_IGNORED = "contradiction_ignored"
    LOOP = "loop"
    SUBPROBLEM_UNNECESSARY = "subproblem_unnecessary"
    LOCAL_PROGRESS_NO_PARENT_CONTRIBUTION = "local_progress_no_parent_contribution"
    BUDGET_EXCEEDED = "budget_exceeded"
    OBLIGATION_NEGLECTED = "obligation_neglected"
    SERIOUS_ALTERNATIVE_IGNORED = "serious_alternative_ignored"


class ConcentrationMeasure(str, Enum):
    PERSISTENCE = "persistence"
    STAGNATION = "stagnation"
    DISPERSION = "dispersion"
    DEPRIVATION = "deprivation"
    OBSTINATION = "obstination"


class ConcentrationSignal(str, Enum):
    REQUEST_LEASE = "request_lease"
    PROPOSE_PRIORITY = "propose_priority"
    NOTIFY_RESULT = "notify_result"
    REFUSE_REQUEST = "refuse_request"


class ConcentrationAim(str, Enum):
    MEANS = "means"
    SUBGOAL = "subgoal"
    FINALITY = "finality"


class ConcentrationMeasures(FrozenModel):
    """Separate counters; unknown measures are never silently treated as zero."""

    persistence_actions: NonNegativeRevision | None = None
    stagnation_actions: NonNegativeRevision | None = None
    dispersion_events: NonNegativeRevision | None = None
    deprivation_actions: NonNegativeRevision | None = None
    obstination_actions: NonNegativeRevision = 0
    schema_version: Ref = "yf_arc3_v5.concentration_measures.v1"


class ConcentrationGovernanceRecord(LineagedRecord):
    """Orthogonal source-selected facets of one focus decision or notification.

    The record does not execute a vocabulary entry. Only SRC may give an
    operation control effect; DRM supplies any priority or relevance judgment.
    """

    record_kind: Literal["concentration_governance"] = "concentration_governance"
    concentration_lease_id: Ref
    why_contract_ref: Ref
    reason_ref: Ref
    operation: ConcentrationOperation | None = None
    transformations: tuple[ConcentrationTransformation, ...] = ()
    outcome: ConcentrationOutcome | None = None
    review_motives: tuple[ConcentrationReviewMotive, ...] = ()
    signal: ConcentrationSignal | None = None
    aim: ConcentrationAim | None = None
    invariant_refs: tuple[ConcentrationInvariant, ...] = ()
    carried_obligation_refs: tuple[Ref, ...] = ()
    committed_path_ref: Ref | None = None
    result_ref: Ref | None = None
    wake_condition_ref: Ref | None = None
    target_ref: Ref | None = None
    inherited_action_budget: NonNegativeRevision | None = None
    measures: ConcentrationMeasures | None = None
    schema_version: Ref = "yf_arc3_v5.concentration_governance.v1"

    @model_validator(mode="after")
    def validate_governance(self) -> "ConcentrationGovernanceRecord":
        if not any((self.operation, self.transformations, self.outcome,
                    self.review_motives, self.signal, self.measures)):
            raise ValueError("concentration governance needs a declared facet")
        require_unique(self.transformations, "concentration transformations")
        require_unique(self.review_motives, "concentration review motives")
        require_unique(self.invariant_refs, "concentration invariants")
        require_unique(self.carried_obligation_refs, "concentration obligations")
        if self.operation in {ConcentrationOperation.WAIT, ConcentrationOperation.SUSPEND}:
            if not self.wake_condition_ref:
                raise ValueError("waiting or suspended focus needs a wake condition")
        if self.operation is ConcentrationOperation.REORIENT and self.aim is None:
            raise ValueError("focus reorientation must name means, subgoal or finality")
        if self.operation in {ConcentrationOperation.HANDOFF, ConcentrationOperation.PREEMPT}:
            if not self.target_ref:
                raise ValueError("focus transfer must name its target")
        return self


class ConcentrationActionAccountingRecord(LineagedRecord):
    """One DRM-assessed action; Python checks only exact integer accounting."""

    record_kind: Literal["concentration_action_accounting"] = (
        "concentration_action_accounting"
    )
    concentration_lease_id: Ref
    pursuit_ref: Ref
    action_intent_id: Ref
    attempt_variant_ref: Ref | None = None
    relevance_decision_ref: Ref
    relevant_to_pursuit: bool
    review_due_unhandled: bool
    obstination_count: NonNegativeRevision
    schema_version: Ref = "yf_arc3_v5.concentration_action_accounting.v1"


class ConcentrationLease(LineagedRecord):
    """One bounded operational focus; never a second reasoning graph.

    Every semantic field is copied from the already selected DRM/SRC action
    contract.  The record grants no authority by itself: only a VALIDATED lease
    attached to an ActionReleasePermit can cross the ACT boundary.
    """

    record_kind: Literal["concentration_lease"] = "concentration_lease"
    game_ref: Ref
    level_ref: Ref
    session_ref: Ref
    frame_revision: NonNegativeRevision
    contract_revision: NonNegativeRevision
    active_objective_ref: Ref | None = None
    information_step_ref: Ref | None = None
    committed_path_ref: Ref | None = None
    remaining_action_bound: NonNegativeRevision | None = None
    drm_decision_ref: Ref
    src_decision_ref: Ref
    allowed_action_ref: Ref
    action_intent_id: Ref
    premise_claim_refs: tuple[Ref, ...] = Field(min_length=1)
    generic_principle_ref: Ref
    expected_effect_ref: Ref
    falsifier_refs: tuple[Ref, ...] = Field(min_length=1)
    preserve_constraint_refs: tuple[Ref, ...] = ()
    overshoot_constraint_refs: tuple[Ref, ...] = ()
    state: ConcentrationLeaseState
    acquisition_reason_ref: Ref
    release_or_invalidation_reason_ref: Ref | None = None
    expires_after_state_revision: NonNegativeRevision
    extension_intent_ids: tuple[Ref, ...] = ()
    requires_post_action_full_revalidation: bool = False
    schema_version: Ref = "yf_arc3_v5.concentration_lease.v1"

    @model_validator(mode="after")
    def validate_focus_contract(self) -> "ConcentrationLease":
        if not self.active_objective_ref and not self.information_step_ref:
            raise ValueError(
                "concentration lease requires an active objective or information step"
            )
        if self.contract_revision != self.state_revision:
            raise ValueError("concentration lease contract revision must be current")
        if self.expires_after_state_revision < self.contract_revision:
            raise ValueError("concentration lease is expired at creation")
        require_unique(self.premise_claim_refs, "concentration lease premises")
        require_unique(self.falsifier_refs, "concentration lease falsifiers")
        require_unique(self.preserve_constraint_refs, "concentration lease preserves")
        require_unique(self.overshoot_constraint_refs, "concentration lease overshoots")
        require_unique(self.extension_intent_ids, "concentration lease extensions")
        if self.action_intent_id not in self.extension_intent_ids:
            raise ValueError("concentration lease extensions must include its first intent")
        if (
            self.requires_post_action_full_revalidation
            and len(self.extension_intent_ids) > 1
        ):
            raise ValueError(
                "full post-action revalidation forbids a multi-action lease extension"
            )
        if self.state not in {
            ConcentrationLeaseState.ACQUIRED,
            ConcentrationLeaseState.VALIDATED,
        }:
            raise ValueError("a new concentration lease must be acquired or validated")
        return self


class ConcentrationLeaseTransition(LineagedRecord):
    """Append-only lifecycle evidence for one concentration lease."""

    record_kind: Literal["concentration_lease_transition"] = (
        "concentration_lease_transition"
    )
    concentration_lease_id: Ref
    from_state: ConcentrationLeaseState
    to_state: ConcentrationLeaseState
    reason_ref: Ref
    action_intent_id: Ref
    schema_version: Ref = "yf_arc3_v5.concentration_lease_transition.v1"


def project_concentration_operation(
    *,
    lease: ConcentrationLease,
    lineage: LineagedRecord,
    operation: ConcentrationOperation,
    reason_ref: Ref,
) -> ConcentrationGovernanceRecord:
    """Copy one SRC-selected phase into the canonical focus vocabulary."""

    return ConcentrationGovernanceRecord(
        id=f"{lineage.id}:focus:{operation.value}",
        source_unit=lineage.source_unit,
        source_hash=lineage.source_hash,
        state_revision=lineage.state_revision,
        workflow_definition_id=lineage.workflow_definition_id,
        workflow_instance_id=lineage.workflow_instance_id,
        workflow_step_id=lineage.workflow_step_id,
        timeline_id=lineage.timeline_id,
        frame_id=lineage.frame_id,
        provenance=(lease.id, lineage.id) if lease.id != lineage.id else (lease.id,),
        concentration_lease_id=lease.id,
        why_contract_ref=lease.id,
        reason_ref=reason_ref,
        operation=operation,
        invariant_refs=(
            ConcentrationInvariant.UNIQUE_AUTHORITY,
            ConcentrationInvariant.WHY_PRESERVED,
            ConcentrationInvariant.OBLIGATIONS_PRESERVED,
        ),
        carried_obligation_refs=tuple(dict.fromkeys(
            (*lease.preserve_constraint_refs, *lease.overshoot_constraint_refs)
        )),
        committed_path_ref=lease.committed_path_ref,
        inherited_action_budget=lease.remaining_action_bound,
    )


class ActionReleasePermit(LineagedRecord):
    record_kind: Literal["action_release_permit"] = "action_release_permit"
    action_intent_id: Ref
    viability_assessment_id: Ref
    commitment_decision_id: Ref
    available_action_ref: Ref
    pre_action_snapshot_digest: Ref
    valid_through_state_revision: NonNegativeRevision
    # When present, this permit authorizes exactly the listed predeclared
    # intents in order.  Ordinary one-action permits leave these fields empty.
    declared_series_ref: Ref | None = None
    declared_series_intent_ids: tuple[Ref, ...] = ()
    declared_series_action_refs: tuple[Ref, ...] = ()
    declared_series_target_workflow_instance_id: Ref | None = None
    # SRC may prepare a shadow permit without owning focus. Its first ACT
    # acquires, validates and engages the lease in one journal transaction.
    concentration_acquisition_deferred: bool = False
    # Optional only so historical journals remain readable.  ACT is fail-closed
    # and rejects every permit that does not carry the validated current lease.
    concentration_lease: ConcentrationLease | None = None
    schema_version: Ref = "yf_arc3_v5.action_release_permit.v1"

    @model_validator(mode="after")
    def validate_revision_window(self) -> "ActionReleasePermit":
        if self.valid_through_state_revision < self.state_revision:
            raise ValueError("release permit is stale at creation")
        if self.concentration_lease is not None:
            lease = self.concentration_lease
            if lease.state not in {
                ConcentrationLeaseState.ACQUIRED,
                ConcentrationLeaseState.VALIDATED,
            }:
                raise ValueError("release permit requires an acquired concentration lease")
            if lease.action_intent_id != self.action_intent_id:
                raise ValueError("release permit and concentration lease intent mismatch")
            if lease.timeline_id != self.timeline_id:
                raise ValueError("release permit and concentration lease timeline mismatch")
            if lease.contract_revision != self.state_revision:
                raise ValueError("release permit and concentration lease revision mismatch")
        return self


class EnvironmentActionDispatchedRecord(LineagedRecord):
    """Durable proof that the sole environment boundary dispatched one intent."""

    record_kind: Literal["environment_action_dispatched"] = (
        "environment_action_dispatched"
    )
    dispatch_id: Ref
    environment_adapter_ref: Ref
    action_intent_id: Ref
    action_release_permit_id: Ref
    action_ref: Ref
    request_digest: Ref
    response_digest: Ref
    raw_input_ref: Ref
    schema_version: Ref = "yf_arc3_v5.environment_action_dispatched.v1"


class EnvironmentActionAttemptRecord(LineagedRecord):
    """Durable pre-transport fact that never claims the effect was sent."""

    record_kind: Literal["environment_action_attempt"] = (
        "environment_action_attempt"
    )
    dispatch_id: Ref
    environment_adapter_ref: Ref
    action_intent_id: Ref
    action_release_permit_id: Ref
    action_ref: Ref
    request_digest: Ref
    unconfirmed_outcome: Literal["unknown"] = "unknown"
    redelivery_policy: Literal["forbid_without_reconciliation"] = (
        "forbid_without_reconciliation"
    )
    schema_version: Ref = "yf_arc3_v5.environment_action_attempt.v1"


class AppliedPrincipleRecord(LineagedRecord):
    record_kind: Literal["applied_principle"] = "applied_principle"
    statement: Ref
    scope: FrozenMap = Field(default_factory=FrozenMap)
    conditions: tuple[Ref, ...] = ()
    source_claims: tuple[Ref, ...] = Field(min_length=1)
    supporting_evidence: tuple[Ref, ...] = ()
    contradicting_evidence: tuple[Ref, ...] = ()
    successful_predictions: tuple[Ref, ...] = ()
    failed_predictions: tuple[Ref, ...] = ()
    known_falsifiers: tuple[Ref, ...] = ()
    originating_workflows: tuple[Ref, ...] = ()
    status: AppliedPrincipleStatus = AppliedPrincipleStatus.PROPOSED
    human_review_required: bool = True
    schema_version: Ref = "yf_arc3_v5.applied_principle.v1"

    @model_validator(mode="after")
    def prevent_automatic_promotion(self) -> "AppliedPrincipleRecord":
        if not self.human_review_required:
            raise ValueError("applied principles require human review before promotion")
        return self


class ObservationAcquiredRecord(LineagedRecord):
    """Immutable provenance for one acquired world input."""

    record_kind: Literal["observation_acquired"] = "observation_acquired"
    acquisition_model_ref: Ref
    raw_input_ref: Ref
    observation_ref: Ref
    schema_version: Ref = "yf_arc3_v5.observation_acquired.v1"


class WorkflowLifecycleRecord(LineagedRecord):
    """A typed workflow step, suspension, or resume transition."""

    record_kind: Literal["workflow_lifecycle"] = "workflow_lifecycle"
    transition: WorkflowLifecycleKind
    from_step_id: Ref | None = None
    to_step_id: Ref | None = None
    from_status: WorkflowStatus
    to_status: WorkflowStatus
    cause_ref: Ref
    schema_version: Ref = "yf_arc3_v5.workflow_lifecycle.v1"

    @model_validator(mode="after")
    def validate_transition(self) -> "WorkflowLifecycleRecord":
        expected = {
            WorkflowLifecycleKind.STARTED: (
                WorkflowStatus.READY,
                WorkflowStatus.ACTIVE,
            ),
            WorkflowLifecycleKind.STEP_CHANGED: (
                WorkflowStatus.ACTIVE,
                WorkflowStatus.ACTIVE,
            ),
            WorkflowLifecycleKind.SUSPENDED: (
                WorkflowStatus.ACTIVE,
                WorkflowStatus.SUSPENDED,
            ),
            WorkflowLifecycleKind.RESUMED: (
                WorkflowStatus.SUSPENDED,
                WorkflowStatus.ACTIVE,
            ),
            WorkflowLifecycleKind.COMPLETED: (
                WorkflowStatus.ACTIVE,
                WorkflowStatus.COMPLETED,
            ),
            WorkflowLifecycleKind.DEFERRED: (
                WorkflowStatus.ACTIVE,
                WorkflowStatus.DEFERRED,
            ),
            WorkflowLifecycleKind.BLOCKED: (
                WorkflowStatus.ACTIVE,
                WorkflowStatus.BLOCKED,
            ),
            WorkflowLifecycleKind.FAILED: (
                WorkflowStatus.ACTIVE,
                WorkflowStatus.FAILED,
            ),
        }[self.transition]
        if (self.from_status, self.to_status) != expected:
            raise ValueError(
                f"invalid {self.transition.value} lifecycle transition: "
                f"{self.from_status.value}->{self.to_status.value}"
            )
        return self


class StateChangeCommittedRecord(LineagedRecord):
    """Audit identity for a reducer-accepted state-change proposal."""

    record_kind: Literal["state_change_committed"] = "state_change_committed"
    proposal_id: Ref
    commitment_decision_id: Ref | None = None
    transaction_id: Ref
    state_revision_before: NonNegativeRevision
    state_revision_after: NonNegativeRevision
    affected_refs: tuple[Ref, ...] = Field(min_length=1)
    schema_version: Ref = "yf_arc3_v5.state_change_committed.v1"

    @model_validator(mode="after")
    def validate_committed_revision(self) -> "StateChangeCommittedRecord":
        if self.state_revision_after != self.state_revision_before + 1:
            raise ValueError("a committed state change advances exactly one revision")
        if self.state_revision != self.state_revision_after:
            raise ValueError(
                "record state revision must equal committed after-revision"
            )
        require_unique(self.affected_refs, "affected_refs")
        return self


class ActionObservedRecord(LineagedRecord):
    """Causal link between a released action and its later observation."""

    record_kind: Literal["action_observed"] = "action_observed"
    action_intent_id: Ref
    action_release_permit_id: Ref
    environment_action_event_ref: Ref
    observation_record_id: Ref
    schema_version: Ref = "yf_arc3_v5.action_observed.v1"
