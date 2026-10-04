"""Bounded immutable contracts for M8 projections and diagnostics."""

from __future__ import annotations

from pydantic import Field, model_validator

from agents.yf_arc3_v5.logos.types import FrozenMap, FrozenModel, Ref


class CompactSrlLine(FrozenModel):
    evidence_ref: Ref
    event_sequence: int = Field(ge=1)
    keyword: Ref
    statement: Ref
    record_ref: Ref
    record_kind: Ref
    workflow_definition_id: Ref | None = None
    workflow_instance_id: Ref | None = None
    workflow_step_id: Ref | None = None
    frame_id: Ref | None = None
    state_revision: int = Field(ge=0)
    source_hash: Ref
    supporting_refs: tuple[Ref, ...] = ()
    schema_version: Ref = "yf_arc3_v5.compact_srl_line.v1"


class TermProjection(FrozenModel):
    id: Ref
    kind: Ref
    label: Ref
    provenance: tuple[Ref, ...] = ()
    last_changed_state_revision: int = Field(ge=0)


class ClaimProjection(FrozenModel):
    id: Ref
    predicate: Ref
    arguments: tuple[Ref, ...]
    epistemic_status: Ref
    disposition: Ref
    cardinality: Ref
    grounds: tuple[Ref, ...] = ()
    contradictions: tuple[Ref, ...] = ()
    defeated_by: tuple[Ref, ...] = ()
    proof_rule: Ref | None = None


class UnresolvedProjection(FrozenModel):
    id: Ref
    statement: Ref
    reason: Ref
    missing_evidence: tuple[Ref, ...] = ()
    alternatives: tuple[Ref, ...] = ()
    falsifiers: tuple[Ref, ...] = ()


class WisdomProjection(FrozenModel):
    decision_id: Ref
    decision: Ref
    policy_ref: Ref
    viability_assessment_id: Ref
    viability_status: Ref | None = None
    reasons: tuple[Ref, ...]
    alternatives_preserved: tuple[Ref, ...] = ()
    missing_evidence: tuple[Ref, ...] = ()
    contradictions: tuple[Ref, ...] = ()


class SourceProvenanceProjection(FrozenModel):
    record_ref: Ref
    record_kind: Ref
    source_unit: Ref
    source_hash: Ref
    workflow_definition_id: Ref | None = None
    workflow_instance_id: Ref | None = None
    workflow_step_id: Ref | None = None


class DecisionReferenceProjection(FrozenModel):
    reference_ref: Ref
    relation_kind: Ref
    source_kind: Ref


class PrincipleProvenanceProjection(FrozenModel):
    principle_ref: Ref
    relation_kind: Ref
    owner_layer: Ref = "drm"
    declaration_source_unit: Ref | None = None
    source_hash: Ref | None = None
    activation_status: Ref


class DecisionEnvelope(FrozenModel):
    terminal_goal_ref: Ref
    active_goal_ref: Ref
    instrumental_subgoal_ref: Ref
    dependency_chain_refs: tuple[Ref, ...] = Field(min_length=1)
    priority_or_dominance_ref: Ref
    committed_plan_or_experiment_ref: Ref
    primitive_action_ref: Ref
    available_action_ref: Ref
    expected_effect_class: Ref
    expected_configuration_or_goal_delta_ref: Ref
    explicit_falsifier_refs: tuple[Ref, ...] = Field(min_length=1)
    remaining_bound: int = Field(ge=0)
    reconciliation_checkpoint_ref: Ref
    workflow_ref: Ref
    workflow_instance_ref: Ref
    decision_ref: Ref
    action_intent_ref: Ref
    release_permit_ref: Ref
    premise_claim_refs: tuple[Ref, ...] = Field(min_length=1)
    element_bindings: FrozenMap = Field(default_factory=FrozenMap)
    source_provenance: tuple[SourceProvenanceProjection, ...] = Field(min_length=4)
    principle_provenance: tuple[PrincipleProvenanceProjection, ...] = ()
    reason_links: tuple[DecisionReferenceProjection, ...] = ()
    trace_line: Ref
    schema_version: Ref = "yf_arc3_v5.decision_envelope.v1"


class ActiveStateProjection(FrozenModel):
    run_id: Ref
    state_revision: int = Field(ge=0)
    prediction_revision: int = Field(ge=0)
    state_hash: Ref
    source_event_count: int = Field(ge=0)
    terms: tuple[TermProjection, ...] = ()
    claims: tuple[ClaimProjection, ...] = ()
    goals: tuple[Ref, ...] = ()
    valid_alternatives: tuple[Ref, ...] = ()
    wisdom: WisdomProjection | None = None
    decision_envelope: DecisionEnvelope | None = None
    unresolved_questions: tuple[UnresolvedProjection, ...] = ()
    contradictions: tuple[Ref, ...] = ()
    counts: FrozenMap = Field(default_factory=FrozenMap)
    bounded: bool = True
    schema_version: Ref = "yf_arc3_v5.active_state_projection.v1"


class AttestationQuery(FrozenModel):
    reference: Ref | None = None
    workflow_definition_id: Ref | None = None
    workflow_instance_id: Ref | None = None
    workflow_step_id: Ref | None = None
    operator: Ref | None = None
    record_kind: Ref | None = None
    limit: int = Field(default=40, ge=1, le=100)
    schema_version: Ref = "yf_arc3_v5.attestation_query.v1"

    @model_validator(mode="after")
    def require_selector(self) -> "AttestationQuery":
        if not any(
            (
                self.reference,
                self.workflow_definition_id,
                self.workflow_instance_id,
                self.workflow_step_id,
                self.operator,
                self.record_kind,
            )
        ):
            raise ValueError("attestation query requires at least one selector")
        return self


class AttestedRecord(FrozenModel):
    id: Ref
    record_kind: Ref
    event_ref: Ref
    event_sequence: int = Field(ge=1)
    state_revision: int = Field(ge=0)
    source_unit: Ref | None = None
    source_hash: Ref
    workflow_definition_id: Ref | None = None
    workflow_instance_id: Ref | None = None
    workflow_step_id: Ref | None = None
    frame_id: Ref | None = None
    operator: Ref | None = None
    authority_ref: Ref | None = None
    premise_refs: tuple[Ref, ...] = ()
    policy_ref: Ref | None = None
    change_ref: Ref | None = None
    action_lineage: FrozenMap = Field(default_factory=FrozenMap)
    provenance: tuple[Ref, ...] = ()
    source_location: FrozenMap = Field(default_factory=FrozenMap)


class AttestationResult(FrozenModel):
    query: AttestationQuery
    records: tuple[AttestedRecord, ...] = ()
    matched_count: int = Field(ge=0)
    truncated: bool = False
    schema_version: Ref = "yf_arc3_v5.attestation_result.v1"


class FirstDivergence(FrozenModel):
    index: int = Field(ge=0)
    classification: Ref
    expected: FrozenMap | None = None
    actual: FrozenMap | None = None
    previous_match: FrozenMap | None = None
    schema_version: Ref = "yf_arc3_v5.first_divergence.v1"


class TraceDiffResult(FrozenModel):
    equivalent: bool
    expected_count: int = Field(ge=0)
    actual_count: int = Field(ge=0)
    matched_prefix_count: int = Field(ge=0)
    first_divergence: FirstDivergence | None = None
    schema_version: Ref = "yf_arc3_v5.trace_diff.v1"


class RunReport(FrozenModel):
    identity: FrozenMap
    result: FrozenMap
    source_hashes: tuple[FrozenMap, ...]
    first_divergence: FrozenMap
    resource_outcome: FrozenMap
    performance: FrozenMap
    action_lineage: FrozenMap
    state: FrozenMap
    schema_version: Ref = "yf_arc3_v5.run_report.v1"


class ArchitectureAuditFinding(FrozenModel):
    check: Ref
    status: Ref
    details: tuple[Ref, ...] = ()


class ArchitectureAuditReport(FrozenModel):
    passed: bool
    findings: tuple[ArchitectureAuditFinding, ...]
    scanned_files: int = Field(ge=0)
    schema_version: Ref = "yf_arc3_v5.architecture_audit.v1"
