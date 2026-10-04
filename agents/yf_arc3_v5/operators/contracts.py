"""Typed requests, authority payloads, and outputs for the nine Logos operators."""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from agents.yf_arc3_v5.logos.claims import Claim
from agents.yf_arc3_v5.logos.operations import (
    ActionIntent,
    ActionReleasePermit,
    AlternativeSet,
    OperatorInvocation,
    OperatorResult,
    StateChangeCommittedRecord,
    StateChangeProposal,
)
from agents.yf_arc3_v5.logos.terms import Term
from agents.yf_arc3_v5.logos.types import (
    Cardinality,
    Disposition,
    FrozenMap,
    FrozenModel,
    NonNegativeRevision,
    OperatorName,
    Polarity,
    Ref,
    RelationFamily,
    require_unique,
)
from agents.yf_arc3_v5.state.events import PutClaim, PutTerm


def _require_sha256(value: str, description: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")


class OperatorRequestBase(FrozenModel):
    request_id: Ref
    operator: OperatorName
    run_id: Ref
    timeline_id: Ref
    frame_id: Ref | None = None
    expected_state_revision: NonNegativeRevision
    source_unit: Ref
    source_hash: Ref
    workflow_definition_id: Ref
    workflow_instance_id: Ref
    workflow_step_id: Ref

    @model_validator(mode="after")
    def validate_request_lineage(self) -> "OperatorRequestBase":
        _require_sha256(self.source_hash, "operator request source_hash")
        return self


class ObserveRequest(OperatorRequestBase):
    operator: Literal[OperatorName.OBSERVE] = OperatorName.OBSERVE
    raw_input_ref: Ref
    observation_ref: Ref
    acquisition_model_ref: Ref


class DistinguishRequest(OperatorRequestBase):
    operator: Literal[OperatorName.DISTINGUISH] = OperatorName.DISTINGUISH
    source_refs: tuple[Ref, ...] = Field(min_length=1)
    model_ref: Ref


class ClaimSeed(FrozenModel):
    id: Ref
    predicate: Ref
    family: RelationFamily
    arguments: tuple[Ref, ...] = Field(min_length=1)
    polarity: Polarity = Polarity.POSITIVE
    disposition: Disposition = Disposition.ACTIVE
    cardinality: Cardinality = Cardinality.SINGULAR
    scope: FrozenMap = Field(default_factory=FrozenMap)
    conditions: tuple[Ref, ...] = ()
    attributes: FrozenMap = Field(default_factory=FrozenMap)
    derivation_premise_refs: tuple[Ref, ...] | None = None

    @model_validator(mode="after")
    def validate_seed(self) -> "ClaimSeed":
        require_unique(self.arguments, "claim seed arguments")
        require_unique(self.conditions, "claim seed conditions")
        if self.derivation_premise_refs is not None:
            if not self.derivation_premise_refs:
                raise ValueError("declared conclusion premises must not be empty")
            require_unique(self.derivation_premise_refs, "declared conclusion premises")
        return self


class RelateRequest(OperatorRequestBase):
    operator: Literal[OperatorName.RELATE] = OperatorName.RELATE
    relation: ClaimSeed
    predicate_ref: Ref
    basis_refs: tuple[Ref, ...] = Field(min_length=1)


class ProposeRequest(OperatorRequestBase):
    operator: Literal[OperatorName.PROPOSE] = OperatorName.PROPOSE
    proposals: tuple[ClaimSeed, ...] = Field(min_length=1)
    basis_refs: tuple[Ref, ...] = Field(min_length=1)
    live_alternatives: tuple[Ref, ...] = Field(min_length=1)
    missing_evidence: tuple[Ref, ...] = ()
    falsifier_refs: tuple[Ref, ...] = ()

    @model_validator(mode="after")
    def validate_proposal_request(self) -> "ProposeRequest":
        require_unique(tuple(item.id for item in self.proposals), "proposal identities")
        require_unique(self.live_alternatives, "proposal alternatives")
        require_unique(self.missing_evidence, "proposal missing evidence")
        require_unique(self.falsifier_refs, "proposal falsifiers")
        return self


class DeriveRequest(OperatorRequestBase):
    operator: Literal[OperatorName.DERIVE] = OperatorName.DERIVE
    conclusions: tuple[ClaimSeed, ...] = Field(min_length=1)
    premise_refs: tuple[Ref, ...] = Field(min_length=1)
    rule_or_model_ref: Ref
    context_refs: tuple[Ref, ...] = ()

    @model_validator(mode="after")
    def validate_derivation_request(self) -> "DeriveRequest":
        require_unique(
            tuple(item.id for item in self.conclusions), "conclusion identities"
        )
        require_unique(self.premise_refs, "derivation premises")
        return self


class CompareRequest(OperatorRequestBase):
    operator: Literal[OperatorName.COMPARE] = OperatorName.COMPARE
    left_refs: tuple[Ref, ...] = Field(min_length=1)
    right_refs: tuple[Ref, ...] = Field(min_length=1)
    criterion_ref: Ref


StateWrite = Annotated[PutTerm | PutClaim, Field(discriminator="kind")]


class UpdateItem(FrozenModel):
    mutation: StateWrite
    invalidates: tuple[Ref, ...] = ()
    recalculates: tuple[Ref, ...] = ()

    @model_validator(mode="after")
    def validate_dependencies(self) -> "UpdateItem":
        require_unique(self.invalidates, "update invalidations")
        require_unique(self.recalculates, "update recalculations")
        return self


class UpdateRequest(OperatorRequestBase):
    operator: Literal[OperatorName.UPDATE] = OperatorName.UPDATE
    source_change_id: Ref
    updates: tuple[UpdateItem, ...] = Field(min_length=1)
    evidence_refs: tuple[Ref, ...] = Field(min_length=1)
    update_policy_ref: Ref
    commitment_decision_id: Ref | None = None

    @model_validator(mode="after")
    def validate_update_targets(self) -> "UpdateRequest":
        target_ids = tuple(item.mutation.after.id for item in self.updates)
        require_unique(target_ids, "UPDATE targets")
        require_unique(self.evidence_refs, "UPDATE evidence")
        return self


class SelectRequest(OperatorRequestBase):
    operator: Literal[OperatorName.SELECT] = OperatorName.SELECT
    goal_ref: Ref
    alternative_refs: tuple[Ref, ...] = Field(min_length=1)
    selection_policy_ref: Ref
    requirement_refs: tuple[Ref, ...] = ()
    preserve_refs: tuple[Ref, ...] = ()
    dominance_witnesses: tuple["DominanceWitness", ...] = ()
    alternative_facts: FrozenMap = Field(default_factory=FrozenMap)
    context_facts: FrozenMap = Field(default_factory=FrozenMap)

    @model_validator(mode="after")
    def validate_selection_request(self) -> "SelectRequest":
        require_unique(self.alternative_refs, "selection alternatives")
        require_unique(self.requirement_refs, "selection requirements")
        require_unique(self.preserve_refs, "selection preservation rules")
        alternative_set = set(self.alternative_refs)
        if any(
            witness.left_ref not in alternative_set
            or witness.right_ref not in alternative_set
            for witness in self.dominance_witnesses
        ):
            raise ValueError("dominance witnesses must compare declared alternatives")
        fact_refs = set(self.alternative_facts)
        if fact_refs and fact_refs != alternative_set:
            raise ValueError("selection facts must index every declared alternative")
        if any(
            not isinstance(self.alternative_facts[item], FrozenMap)
            for item in fact_refs
        ):
            raise ValueError("selection facts for each alternative must be a mapping")
        return self


class ActRequest(OperatorRequestBase):
    operator: Literal[OperatorName.ACT] = OperatorName.ACT
    intent: ActionIntent
    release_permit: ActionReleasePermit
    world_context: FrozenMap


AnyOperatorRequest = (
    ObserveRequest
    | DistinguishRequest
    | RelateRequest
    | ProposeRequest
    | DeriveRequest
    | CompareRequest
    | UpdateRequest
    | SelectRequest
    | ActRequest
)


class AcquisitionInput(FrozenModel):
    raw_input_ref: Ref
    requested_observation_ref: Ref
    frame_id: Ref | None = None
    schema_version: Ref = "yf_arc3_v5.authority_input.acquisition.v1"


_SEMANTIC_ROLE_WORDS = frozenset(
    {"actor", "actuator", "button", "cursor", "goal", "mechanism", "target"}
)


def _contains_semantic_role(value: object) -> bool:
    if isinstance(value, str):
        words = set(filter(None, re.split(r"[^a-z0-9]+", value.lower())))
        return bool(words & _SEMANTIC_ROLE_WORDS)
    if isinstance(value, Mapping):
        return any(
            _contains_semantic_role(key) or _contains_semantic_role(item)
            for key, item in sorted(value.items())
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_semantic_role(item) for item in value)
    return False


class AcquisitionResult(FrozenModel):
    observation_ref: Ref
    component_refs: tuple[Ref, ...] = ()
    measurements: FrozenMap = Field(default_factory=FrozenMap)
    reason_refs: tuple[Ref, ...] = ()
    schema_version: Ref = "yf_arc3_v5.authority_output.acquisition.v1"

    @model_validator(mode="after")
    def validate_semantic_neutrality(self) -> "AcquisitionResult":
        require_unique(self.component_refs, "observed component refs")
        if _contains_semantic_role(self.component_refs) or _contains_semantic_role(
            self.measurements
        ):
            raise ValueError("acquisition output cannot establish semantic roles")
        return self


class DistinctionInput(FrozenModel):
    source_refs: tuple[Ref, ...] = Field(min_length=1)
    schema_version: Ref = "yf_arc3_v5.authority_input.distinction.v1"


class DistinctionResult(FrozenModel):
    distinctions: tuple[Term, ...] = Field(min_length=1)
    reason_refs: tuple[Ref, ...] = ()
    schema_version: Ref = "yf_arc3_v5.authority_output.distinction.v1"


class PredicateInput(FrozenModel):
    predicate_ref: Ref
    arguments: tuple[Ref, ...] = Field(min_length=1)
    basis_refs: tuple[Ref, ...] = Field(min_length=1)
    schema_version: Ref = "yf_arc3_v5.authority_input.predicate.v1"


class PredicateResult(FrozenModel):
    holds: bool
    reason_refs: tuple[Ref, ...] = Field(min_length=1)
    schema_version: Ref = "yf_arc3_v5.authority_output.predicate.v1"


class DerivationInput(FrozenModel):
    premise_refs: tuple[Ref, ...] = Field(min_length=1)
    context_refs: tuple[Ref, ...] = ()
    requested_conclusions: tuple[ClaimSeed, ...] = Field(min_length=1)
    schema_version: Ref = "yf_arc3_v5.authority_input.derivation.v1"


class DerivationResult(FrozenModel):
    conclusions: tuple[ClaimSeed, ...] = Field(min_length=1)
    reason_refs: tuple[Ref, ...] = Field(min_length=1)
    schema_version: Ref = "yf_arc3_v5.authority_output.derivation.v1"


class ComparisonKind(str, Enum):
    MATCH = "match"
    MISMATCH = "mismatch"
    UNDETERMINED = "undetermined"


class ComparisonInput(FrozenModel):
    left_refs: tuple[Ref, ...] = Field(min_length=1)
    right_refs: tuple[Ref, ...] = Field(min_length=1)
    schema_version: Ref = "yf_arc3_v5.authority_input.comparison.v1"


class ComparisonResult(FrozenModel):
    comparison: ComparisonKind
    reason_refs: tuple[Ref, ...] = Field(min_length=1)
    schema_version: Ref = "yf_arc3_v5.authority_output.comparison.v1"


class UpdatePolicyInput(FrozenModel):
    proposal: StateChangeProposal
    current_state_hash: Ref
    current_state_revision: NonNegativeRevision
    schema_version: Ref = "yf_arc3_v5.authority_input.update_policy.v1"


class UpdatePolicyResult(FrozenModel):
    allowed: bool
    reason_refs: tuple[Ref, ...] = Field(min_length=1)
    schema_version: Ref = "yf_arc3_v5.authority_output.update_policy.v1"


class PreferredOrder(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ExactOrder(str, Enum):
    LESS_THAN = "less_than"
    EQUAL = "equal"
    GREATER_THAN = "greater_than"


class DominanceWitness(FrozenModel):
    criterion_ref: Ref
    left_ref: Ref
    right_ref: Ref
    preferred_order: PreferredOrder
    observed_order: ExactOrder
    evidence_refs: tuple[Ref, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_comparison(self) -> "DominanceWitness":
        if self.left_ref == self.right_ref:
            raise ValueError("dominance witness requires two alternatives")
        require_unique(self.evidence_refs, "dominance evidence")
        return self


class SelectionInput(FrozenModel):
    goal_ref: Ref
    alternative_refs: tuple[Ref, ...] = Field(min_length=1)
    requirement_refs: tuple[Ref, ...] = ()
    preserve_refs: tuple[Ref, ...] = ()
    dominance_witnesses: tuple[DominanceWitness, ...] = ()
    alternative_facts: FrozenMap = Field(default_factory=FrozenMap)
    context_facts: FrozenMap = Field(default_factory=FrozenMap)
    schema_version: Ref = "yf_arc3_v5.authority_input.selection.v1"

    @model_validator(mode="after")
    def validate_alternative_facts(self) -> "SelectionInput":
        require_unique(self.alternative_refs, "selection alternatives")
        alternative_set = set(self.alternative_refs)
        fact_refs = set(self.alternative_facts)
        if fact_refs and fact_refs != alternative_set:
            raise ValueError("selection facts must index every declared alternative")
        if any(
            not isinstance(self.alternative_facts[item], FrozenMap)
            for item in fact_refs
        ):
            raise ValueError("selection facts for each alternative must be a mapping")
        return self


class SelectionCriterionTrace(FrozenModel):
    criterion_ref: Ref
    field: Ref
    preferred_order: Literal["minimize", "maximize"]
    left_ref: Ref
    right_ref: Ref
    left_value: object = None
    right_value: object = None
    outcome: Literal["left_preferred", "right_preferred", "equal"]
    reason_refs: tuple[Ref, ...] = Field(min_length=1)


class SelectionComparisonTrace(FrozenModel):
    selection_policy_ref: Ref
    eligible_refs: tuple[Ref, ...]
    ineligible_refs: tuple[Ref, ...]
    ordered_refs: tuple[Ref, ...]
    selected_ref: Ref | None = None
    runner_up_ref: Ref | None = None
    first_discriminating_criterion_ref: Ref | None = None
    criterion_traces: tuple[SelectionCriterionTrace, ...] = ()
    unresolved_equality: bool = False
    schema_version: Ref = "yf_arc3_v5.selection_comparison_trace.v1"


class SelectionResult(FrozenModel):
    selected_ref: Ref | None = None
    preserves_non_selected: bool = True
    reason_refs: tuple[Ref, ...] = Field(min_length=1)
    comparison_trace: SelectionComparisonTrace | None = None
    schema_version: Ref = "yf_arc3_v5.authority_output.selection.v1"


class ObservationOutput(FrozenModel):
    output_kind: Literal["observation"] = "observation"
    evidence: AcquisitionResult


class DistinguishOutput(FrozenModel):
    output_kind: Literal["distinction"] = "distinction"
    distinctions: tuple[Term, ...] = Field(min_length=1)


class RelateOutput(FrozenModel):
    output_kind: Literal["relation"] = "relation"
    claim: Claim


class ProposalOutput(FrozenModel):
    output_kind: Literal["proposal"] = "proposal"
    claims: tuple[Claim, ...] = Field(min_length=1)
    alternatives: AlternativeSet
    missing_evidence: tuple[Ref, ...] = ()
    falsifier_refs: tuple[Ref, ...] = ()


class DerivationOutput(FrozenModel):
    output_kind: Literal["derivation"] = "derivation"
    claims: tuple[Claim, ...] = Field(min_length=1)
    exact_premise_refs: tuple[Ref, ...] = Field(min_length=1)
    rule_or_model_ref: Ref


class CompareOutput(FrozenModel):
    output_kind: Literal["comparison"] = "comparison"
    comparison: ComparisonKind
    left_refs: tuple[Ref, ...]
    right_refs: tuple[Ref, ...]
    reason_refs: tuple[Ref, ...]


class UpdateOutput(FrozenModel):
    output_kind: Literal["update"] = "update"
    proposal: StateChangeProposal
    committed: StateChangeCommittedRecord
    invalidated_refs: tuple[Ref, ...] = ()
    recalculated_refs: tuple[Ref, ...] = ()


class SelectOutput(FrozenModel):
    output_kind: Literal["selection"] = "selection"
    alternatives: AlternativeSet
    preserved_non_selected: tuple[Ref, ...] = ()
    reason_refs: tuple[Ref, ...] = ()
    comparison_trace: SelectionComparisonTrace | None = None


class ActOutput(FrozenModel):
    output_kind: Literal["action_dispatch"] = "action_dispatch"
    action_ref: Ref
    action_intent_id: Ref
    action_release_permit_id: Ref
    dispatch_status: Literal[
        "pending_environment_boundary"
    ] = "pending_environment_boundary"


OperatorOutput = Annotated[
    ObservationOutput
    | DistinguishOutput
    | RelateOutput
    | ProposalOutput
    | DerivationOutput
    | CompareOutput
    | UpdateOutput
    | SelectOutput
    | ActOutput,
    Field(discriminator="output_kind"),
]


class OperatorExecution(FrozenModel):
    invocation: OperatorInvocation
    result: OperatorResult
    output: OperatorOutput | None = None
    state_revision_before: NonNegativeRevision
    state_revision_after: NonNegativeRevision
    state_hash_before: Ref
    state_hash_after: Ref
    committed_event_ids: tuple[Ref, ...] = ()
    schema_version: Ref = "yf_arc3_v5.operator_execution.v1"

    @model_validator(mode="after")
    def validate_envelope(self) -> "OperatorExecution":
        if self.result.invocation_id != self.invocation.id:
            raise ValueError("operator result must reference its exact invocation")
        if self.result.operator is not self.invocation.operator:
            raise ValueError("operator invocation/result kind mismatch")
        if self.result.status.value == "completed" and self.output is None:
            raise ValueError("completed operator execution requires typed output")
        if self.result.status.value != "completed" and self.output is not None:
            raise ValueError("non-completed operator execution cannot expose output")
        require_unique(self.committed_event_ids, "operator committed event ids")
        return self
