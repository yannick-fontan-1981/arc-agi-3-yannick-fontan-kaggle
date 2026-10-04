"""Typed transport-only contracts for the single V5 environment boundary."""

from __future__ import annotations

from typing import Protocol

from pydantic import Field, model_validator

from agents.yf_arc3_v5.logos.operations import (
    ActionIntent,
    ActionReleasePermit,
    ViabilityAssessment,
)
from agents.yf_arc3_v5.logos.types import FrozenMap, FrozenModel, Ref, require_unique


class ObservedWorldInput(FrozenModel):
    raw_input_ref: Ref
    frame_id: Ref
    frame: tuple[tuple[int, ...], ...] = Field(min_length=1)
    available_action_refs: tuple[Ref, ...] = ()
    score: int = Field(default=0, ge=0, le=254)
    state: Ref = "NOT_FINISHED"
    metadata: FrozenMap = Field(default_factory=FrozenMap)
    schema_version: Ref = "yf_arc3_v5.observed_world_input.v1"

    @model_validator(mode="after")
    def validate_frame(self) -> "ObservedWorldInput":
        width = len(self.frame[0])
        if width < 1 or any(len(row) != width for row in self.frame):
            raise ValueError("observed frame must be a non-empty rectangle")
        require_unique(self.available_action_refs, "available action refs")
        return self


class EnvironmentActionRequest(FrozenModel):
    dispatch_id: Ref
    environment_adapter_ref: Ref
    workflow_definition_id: Ref
    workflow_instance_id: Ref
    timeline_id: Ref
    frame_id: Ref
    intent: ActionIntent
    permit: ActionReleasePermit
    world_context: FrozenMap
    request_digest: Ref
    schema_version: Ref = "yf_arc3_v5.environment_action_request.v1"


class EnvironmentActionResponse(FrozenModel):
    dispatch_id: Ref
    environment_action_event_ref: Ref
    action_intent_id: Ref
    action_release_permit_id: Ref
    raw_input: ObservedWorldInput
    response_digest: Ref
    transport_metadata: FrozenMap = Field(default_factory=FrozenMap)
    schema_version: Ref = "yf_arc3_v5.environment_action_response.v1"


class SessionBootstrapResponse(FrozenModel):
    raw_input: ObservedWorldInput
    session_metadata: FrozenMap = Field(default_factory=FrozenMap)
    schema_version: Ref = "yf_arc3_v5.session_bootstrap_response.v1"


class EnvironmentTransport(Protocol):
    def execute(self, request: EnvironmentActionRequest) -> EnvironmentActionResponse:
        """Execute exactly one already-authorized environment action."""


class SessionEnvironmentTransport(EnvironmentTransport, Protocol):
    def bootstrap(self) -> SessionBootstrapResponse:
        """Reset/select a session through the explicit bootstrap exception."""


class BoundaryDispatchResult(FrozenModel):
    workflow_instance_id: Ref
    dispatch_id: Ref
    environment_action_event_ref: Ref
    frame_id: Ref
    workflow_status: Ref
    state_revision: int
    reused: bool = False
    schema_version: Ref = "yf_arc3_v5.boundary_dispatch_result.v1"


class PreparedInteractionProbe(FrozenModel):
    """Mechanical envelope for one candidate already selected by SRC/DRM."""

    selected_candidate_ref: Ref
    viability: ViabilityAssessment
    intent: ActionIntent
    context: FrozenMap
    alternatives_preserved: tuple[Ref, ...] = ()
    selection_policy_ref: Ref = ""
    selection_source_ref: Ref = ""
    schema_version: Ref = "yf_arc3_v5.prepared_interaction_probe.v1"
