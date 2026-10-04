"""Immutable contracts for durable compiled-workflow execution and resume."""

from __future__ import annotations

from pydantic import Field, model_validator

from agents.yf_arc3_v5.logos.operations import (
    ObservationAcquiredRecord,
    WorkflowInstanceRecord,
)
from agents.yf_arc3_v5.logos.types import (
    FrozenMap,
    FrozenModel,
    NonNegativeRevision,
    OperatorName,
    Ref,
)
from agents.yf_arc3_v5.src.ast import StatementKind


def require_sha256(value: str, description: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")


class WorkflowStartRequest(FrozenModel):
    workflow_definition_id: Ref
    workflow_instance_id: Ref
    run_id: Ref
    timeline_id: Ref
    frame_id: Ref | None = None
    expected_transitive_source_hash: Ref
    inputs: FrozenMap = Field(default_factory=FrozenMap)
    parent_workflow_instance_id: Ref | None = None
    declared_composition_edge_ref: Ref | None = None
    schema_version: Ref = "yf_arc3_v5.workflow_start_request.v2"

    @model_validator(mode="after")
    def validate_source(self) -> "WorkflowStartRequest":
        require_sha256(
            self.expected_transitive_source_hash,
            "workflow start transitive source hash",
        )
        if bool(self.parent_workflow_instance_id) != bool(
            self.declared_composition_edge_ref
        ):
            raise ValueError(
                "composed workflow start requires both parent and declared edge"
            )
        return self


class CausalFrameResumeRequest(FrozenModel):
    workflow_instance_id: Ref
    expected_transitive_source_hash: Ref
    timeline_id: Ref
    frame_id: Ref
    caused_by_action_intent_id: Ref
    action_release_permit_id: Ref
    environment_action_event_ref: Ref
    pre_action_snapshot_digest: Ref
    observation_record: ObservationAcquiredRecord
    schema_version: Ref = "yf_arc3_v5.causal_frame_resume_request.v1"

    @model_validator(mode="after")
    def validate_envelope(self) -> "CausalFrameResumeRequest":
        require_sha256(
            self.expected_transitive_source_hash,
            "workflow resume transitive source hash",
        )
        if self.observation_record.frame_id != self.frame_id:
            raise ValueError("resume observation frame identity mismatch")
        if self.observation_record.timeline_id != self.timeline_id:
            raise ValueError("resume observation timeline identity mismatch")
        return self


class FunctionCallContext(FrozenModel):
    call_id: Ref
    callee_ref: Ref
    workflow_definition_id: Ref
    workflow_instance_id: Ref
    workflow_step_id: Ref
    statement_id: Ref
    timeline_id: Ref
    state_revision: NonNegativeRevision
    frame_id: Ref | None = None
    arguments: FrozenMap = Field(default_factory=FrozenMap)
    local_bindings: FrozenMap = Field(default_factory=FrozenMap)
    schema_version: Ref = "yf_arc3_v5.function_call_context.v1"


class OperatorBuildContext(FrozenModel):
    request_id: Ref
    operator: OperatorName
    statement_kind: StatementKind
    statement_id: Ref
    target_binding: Ref
    run_id: Ref
    timeline_id: Ref
    frame_id: Ref | None = None
    expected_state_revision: NonNegativeRevision
    source_unit: Ref
    source_hash: Ref
    workflow_definition_id: Ref
    workflow_instance_id: Ref
    workflow_step_id: Ref
    clauses: FrozenMap
    local_bindings: FrozenMap
    schema_version: Ref = "yf_arc3_v5.operator_build_context.v1"


class SchedulerRunResult(FrozenModel):
    checkpoint: WorkflowInstanceRecord
    executed_statement_ids: tuple[Ref, ...] = ()
    instruction_limit_reached: bool = False
    schema_version: Ref = "yf_arc3_v5.scheduler_run_result.v1"
