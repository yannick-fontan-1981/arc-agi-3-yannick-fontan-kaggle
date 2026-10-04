"""Immutable state-changing events for the V5 canonical reducer."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from agents.yf_arc3_v5.dynamic_workflow.shadow import DynamicWorkflowShadowRecord
from agents.yf_arc3_v5.dynamic_workflow.runtime import RootColdExecutionRecord
from agents.yf_arc3_v5.logos.claims import Claim
from agents.yf_arc3_v5.logos.operations import (
    ActionIntent,
    ActionObservedRecord,
    ActionReleasePermit,
    AppliedPrincipleRecord,
    CommitmentDecision,
    ConcentrationActionAccountingRecord,
    ConcentrationGovernanceRecord,
    ConcentrationLease,
    ConcentrationLeaseTransition,
    CompiledInteractionEffectRecord,
    CompiledSceneAssimilationRecord,
    CompiledSelectionDecisionRecord,
    CompiledTemporalTrackingRecord,
    EnvironmentActionAttemptRecord,
    EnvironmentActionDispatchedRecord,
    ObservationAcquiredRecord,
    OperatorInvocation,
    OperatorResult,
    PropagationReport,
    StateChangeCommittedRecord,
    StateChangeProposal,
    ViabilityAssessment,
    WorkflowBindingValueRecord,
    WorkflowInstanceRecord,
    WorkflowLifecycleRecord,
)
from agents.yf_arc3_v5.logos.terms import Term
from agents.yf_arc3_v5.logos.types import (
    FrozenMap,
    FrozenModel,
    NonNegativeRevision,
    PositiveSequence,
    Ref,
    stable_digest,
)


_COGNITIVE_EVENT_V1 = "yf_arc3_v5.cognitive_event.v1"
_COGNITIVE_EVENT_V2 = "yf_arc3_v5.cognitive_event.v2"
_COGNITIVE_EVENT_V3 = "yf_arc3_v5.cognitive_event.v3"
_COGNITIVE_EVENT_V4 = "yf_arc3_v5.cognitive_event.v4"


def _event_content_hash(values: dict[str, object]) -> str:
    """Hash an event envelope without rewalking validated binding payloads.

    V2 uses the already verified content digest of a large workflow binding as
    a Merkle leaf.  Imported binding records validate that digest against the
    payload before the event validator runs, so payload tampering still fails
    closed.  V1 remains readable with its original full-payload hash.
    """

    mutation = values["mutation"]
    if (
        values.get("schema_version") in (_COGNITIVE_EVENT_V2, _COGNITIVE_EVENT_V3, _COGNITIVE_EVENT_V4)
        and isinstance(mutation, RecordArtifact)
        and isinstance(mutation.artifact, WorkflowBindingValueRecord)
    ):
        artifact = mutation.artifact
        artifact_projection = artifact.model_dump(
            mode="python", exclude={"payload"}
        )
        artifact_projection["payload"] = FrozenMap(
            {"$value_digest": artifact.value_digest}
        )
        values = {
            **values,
            "mutation": FrozenMap(
                {
                    "kind": mutation.kind,
                    "artifact": artifact_projection,
                }
            ),
        }
    return stable_digest(values)


class PutTerm(FrozenModel):
    kind: Literal["put_term"] = "put_term"
    before: Term | None = None
    after: Term

    @model_validator(mode="after")
    def validate_identity(self) -> "PutTerm":
        if self.before and self.before.id != self.after.id:
            raise ValueError("term revision must preserve term identity")
        return self


class PutClaim(FrozenModel):
    kind: Literal["put_claim"] = "put_claim"
    before: Claim | None = None
    after: Claim

    @model_validator(mode="after")
    def validate_revision(self) -> "PutClaim":
        if self.before is None:
            if self.after.supersedes is not None:
                raise ValueError("new claim cannot supersede an absent claim")
            return self
        if self.after.supersedes != self.before.id:
            raise ValueError("claim revision must identify the exact superseded claim")
        if (
            self.after.predicate != self.before.predicate
            or self.after.family != self.before.family
            or self.after.arguments != self.before.arguments
        ):
            raise ValueError("claim revision cannot silently change relation identity")
        return self


CognitiveArtifact = Annotated[
    OperatorInvocation
    | OperatorResult
    | WorkflowBindingValueRecord
    | WorkflowInstanceRecord
    | WorkflowLifecycleRecord
    | ObservationAcquiredRecord
    | StateChangeProposal
    | StateChangeCommittedRecord
    | ViabilityAssessment
    | CommitmentDecision
    | ConcentrationActionAccountingRecord
    | ConcentrationGovernanceRecord
    | ConcentrationLease
    | ConcentrationLeaseTransition
    | PropagationReport
    | ActionIntent
    | ActionReleasePermit
    | EnvironmentActionAttemptRecord
    | EnvironmentActionDispatchedRecord
    | ActionObservedRecord
    | AppliedPrincipleRecord
    | DynamicWorkflowShadowRecord
    | RootColdExecutionRecord
    | CompiledInteractionEffectRecord
    | CompiledSceneAssimilationRecord
    | CompiledTemporalTrackingRecord
    | CompiledSelectionDecisionRecord,
    Field(discriminator="record_kind"),
]


class RecordArtifact(FrozenModel):
    """Append a typed cognitive lifecycle record without creating a second state writer."""

    kind: Literal["record_artifact"] = "record_artifact"
    artifact: CognitiveArtifact


CognitiveEntry = Annotated[
    PutTerm | PutClaim | RecordArtifact,
    Field(discriminator="kind"),
]

# Kept as the transaction-facing name: entries may either mutate TERM/CLAIM
# state or append a typed audit artifact to the same canonical journal.
StateMutation = CognitiveEntry


class CognitiveEvent(FrozenModel):
    event_id: Ref
    sequence: PositiveSequence
    transaction_id: Ref
    run_id: Ref
    timeline_id: Ref
    frame_id: Ref | None = None
    state_revision_before: NonNegativeRevision
    state_revision_after: NonNegativeRevision
    source_hash: Ref
    mutation: CognitiveEntry
    content_hash: Ref
    schema_version: Ref = _COGNITIVE_EVENT_V2

    @classmethod
    def create(
        cls,
        *,
        event_id: Ref,
        sequence: int,
        transaction_id: Ref,
        run_id: Ref,
        timeline_id: Ref,
        frame_id: Ref | None,
        state_revision_before: int,
        state_revision_after: int,
        source_hash: Ref,
        mutation: CognitiveEntry,
        schema_version: Ref = _COGNITIVE_EVENT_V2,
    ) -> "CognitiveEvent":
        values = {
            "event_id": event_id,
            "sequence": sequence,
            "transaction_id": transaction_id,
            "run_id": run_id,
            "timeline_id": timeline_id,
            "frame_id": frame_id,
            "state_revision_before": state_revision_before,
            "state_revision_after": state_revision_after,
            "source_hash": source_hash,
            "mutation": mutation,
            "schema_version": schema_version,
        }
        content_hash = _event_content_hash(values)
        revision_delta = state_revision_after - state_revision_before
        if revision_delta not in (0, 1):
            raise ValueError("event revision delta must be zero or one")
        # Every value above is produced from already validated local contracts.
        # Constructing the envelope directly prevents Pydantic from recursively
        # revalidating the potentially large immutable artifact a second time.
        # Imported JSON still goes through full validation and hash checking.
        return cls.model_construct(**values, content_hash=content_hash)

    @model_validator(mode="after")
    def validate_event(self) -> "CognitiveEvent":
        if self.schema_version not in (_COGNITIVE_EVENT_V1, _COGNITIVE_EVENT_V2, _COGNITIVE_EVENT_V3, _COGNITIVE_EVENT_V4):
            raise ValueError("unsupported cognitive event schema")
        revision_delta = self.state_revision_after - self.state_revision_before
        if revision_delta not in (0, 1):
            raise ValueError("event revision delta must be zero or one")
        expected = _event_content_hash(
            {
                "event_id": self.event_id,
                "sequence": self.sequence,
                "transaction_id": self.transaction_id,
                "run_id": self.run_id,
                "timeline_id": self.timeline_id,
                "frame_id": self.frame_id,
                "state_revision_before": self.state_revision_before,
                "state_revision_after": self.state_revision_after,
                "source_hash": self.source_hash,
                "mutation": self.mutation,
                "schema_version": self.schema_version,
            }
        )
        if self.content_hash != expected:
            raise ValueError("event content hash mismatch")
        return self
