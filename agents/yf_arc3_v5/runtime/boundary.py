"""Exactly-once dispatch and causal workflow resume at the environment boundary."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence

from agents.yf_arc3_v5.logos.operations import (
    ActionIntent,
    ActionObservedRecord,
    ActionReleasePermit,
    EnvironmentActionAttemptRecord,
    EnvironmentActionDispatchedRecord,
    ObservationAcquiredRecord,
    WorkflowInstanceRecord,
    WorkflowStatus,
)
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest
from agents.yf_arc3_v5.runtime.acquisition import WorldInputRepository
from agents.yf_arc3_v5.runtime.contracts import (
    BoundaryDispatchResult,
    EnvironmentActionRequest,
    EnvironmentActionResponse,
    EnvironmentTransport,
)
from agents.yf_arc3_v5.runtime.environment import (
    ENVIRONMENT_ADAPTER_ID,
    environment_symbol_definition,
)
from agents.yf_arc3_v5.scheduler.contracts import CausalFrameResumeRequest
from agents.yf_arc3_v5.scheduler.runtime import WorkflowScheduler
from agents.yf_arc3_v5.state import EventStore, RecordArtifact, TransactionRequest


class ActionBoundaryError(RuntimeError):
    """The environment boundary rejected incomplete or conflicting lineage."""


class ActionBoundaryUncertainError(ActionBoundaryError):
    """A prepared dispatch has no durable proof of whether its effect occurred."""

    outcome_status = "unknown"
    redelivery_permitted = False
    reconciliation_required = True

    def __init__(self, dispatch_id: str) -> None:
        self.dispatch_id = dispatch_id
        super().__init__(
            f"dispatch outcome is unknown for {dispatch_id}; "
            "redelivery is forbidden until external reconciliation"
        )


class ActionBoundary:
    def __init__(
        self,
        *,
        scheduler: WorkflowScheduler,
        event_store: EventStore,
        world_inputs: WorldInputRepository,
        transport: EnvironmentTransport,
        after_attempt_before_transport: Callable[[EnvironmentActionRequest], None]
        | None = None,
        after_transport_before_record: Callable[
            [EnvironmentActionRequest, EnvironmentActionResponse], None
        ]
        | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._store = event_store
        self._world_inputs = world_inputs
        self._transport = transport
        self._after_attempt_before_transport = after_attempt_before_transport
        self._after_transport_before_record = after_transport_before_record
        self._responses: dict[str, EnvironmentActionResponse] = {}
        self._indexed_action_artifact_counts: dict[type[object], int] = {}
        self._dispatch_by_intent: dict[str, EnvironmentActionDispatchedRecord] = {}
        self._attempt_by_intent: dict[str, EnvironmentActionAttemptRecord] = {}
        self._observed_by_workflow_intent: dict[
            tuple[str, str], ActionObservedRecord
        ] = {}
        self._lock = threading.RLock()

    @property
    def fault_injection_enabled(self) -> bool:
        return (
            self._after_attempt_before_transport is not None
            or self._after_transport_before_record is not None
        )

    def dispatch(self, workflow_instance_id: str) -> BoundaryDispatchResult:
        """Call the transport only after exact suspended-workflow lineage checks."""

        with self._lock:
            checkpoint = self._scheduler.current(workflow_instance_id)
            observed = self._observed_for_workflow(
                workflow_instance_id,
                intent_id=checkpoint.pending_action_intent_id,
            )
            if observed is not None:
                return BoundaryDispatchResult(
                    workflow_instance_id=workflow_instance_id,
                    dispatch_id=observed.environment_action_event_ref,
                    environment_action_event_ref=(
                        observed.environment_action_event_ref
                    ),
                    frame_id=observed.frame_id or "frame:unknown",
                    workflow_status=checkpoint.status.value,
                    state_revision=self._store.snapshot.revision,
                    reused=True,
                )

            if checkpoint.status is not WorkflowStatus.SUSPENDED:
                raise ActionBoundaryError(
                    f"workflow is not awaiting an action frame: "
                    f"{checkpoint.status.value}"
                )
            if checkpoint.pending_environment_adapter_ref != ENVIRONMENT_ADAPTER_ID:
                raise ActionBoundaryError(
                    "suspension names an unknown environment adapter"
                )
            intent = ActionIntent.model_validate(
                _record_binding_from_preferred(
                    checkpoint.local_bindings,
                    preferred_binding_names=("intent", "intents", "prepared", "action_event"),
                    record_kind="action_intent",
                    record_id=checkpoint.pending_action_intent_id,
                )
            )
            permit = ActionReleasePermit.model_validate(
                _record_binding_from_preferred(
                    checkpoint.local_bindings,
                    preferred_binding_names=("release",),
                    record_kind="action_release_permit",
                    record_id=checkpoint.pending_action_release_permit_id,
                )
            )
            world_context = checkpoint.pending_world_context
            if world_context is None:
                raise ActionBoundaryError(
                    "suspended workflow has no exact ACT world context"
                )
            request = _dispatch_request(
                checkpoint=checkpoint,
                intent=intent,
                permit=permit,
                world_context=world_context,
            )
            dispatched = self._dispatched_for_intent(intent.id)
            attempted = self._attempt_for_intent(intent.id)
            response = self._responses.get(intent.id)
            if dispatched is None:
                if attempted is not None:
                    if attempted.request_digest != request.request_digest:
                        raise ActionBoundaryError(
                            "a prior dispatch attempt exists with different request lineage"
                        )
                    raise ActionBoundaryUncertainError(request.dispatch_id)
                self._record_attempt(checkpoint, request)
                try:
                    if self._after_attempt_before_transport is not None:
                        self._after_attempt_before_transport(request)
                    response = self._transport.execute(request)
                    if self._after_transport_before_record is not None:
                        self._after_transport_before_record(request, response)
                    self._validate_response(request, response)
                    self._world_inputs.put(response.raw_input)
                    dispatched = self._record_dispatch(checkpoint, request, response)
                    self._responses[intent.id] = response
                except ActionBoundaryUncertainError:
                    raise
                except Exception as error:
                    raise ActionBoundaryUncertainError(request.dispatch_id) from error
                reused = False
            else:
                if dispatched.request_digest != request.request_digest:
                    raise ActionBoundaryError(
                        "a prior dispatch exists with different request lineage"
                    )
                if response is None:
                    try:
                        raw_input = self._world_inputs.get(dispatched.raw_input_ref)
                    except KeyError as error:
                        raise ActionBoundaryError(
                            "dispatch was recorded but its causal frame is unavailable"
                        ) from error
                    response = EnvironmentActionResponse(
                        dispatch_id=dispatched.dispatch_id,
                        environment_action_event_ref=dispatched.id,
                        action_intent_id=intent.id,
                        action_release_permit_id=permit.id,
                        raw_input=raw_input,
                        response_digest=dispatched.response_digest,
                    )
                reused = True

            resumed = self._scheduler.resume(
                CausalFrameResumeRequest(
                    workflow_instance_id=workflow_instance_id,
                    expected_transitive_source_hash=checkpoint.source_hash,
                    timeline_id=checkpoint.timeline_id or "",
                    frame_id=response.raw_input.frame_id,
                    caused_by_action_intent_id=intent.id,
                    action_release_permit_id=permit.id,
                    environment_action_event_ref=dispatched.id,
                    pre_action_snapshot_digest=permit.pre_action_snapshot_digest,
                    observation_record=ObservationAcquiredRecord(
                        id=f"observation-receipt:{dispatched.dispatch_id}",
                        source_unit=dispatched.source_unit,
                        source_hash=dispatched.source_hash,
                        state_revision=self._store.snapshot.revision,
                        workflow_definition_id=checkpoint.workflow_definition_id,
                        workflow_instance_id=workflow_instance_id,
                        workflow_step_id=checkpoint.workflow_step_id,
                        timeline_id=checkpoint.timeline_id,
                        frame_id=response.raw_input.frame_id,
                        provenance=(dispatched.id,),
                        acquisition_model_ref="transport.causal_frame",
                        raw_input_ref=response.raw_input.raw_input_ref,
                        observation_ref=f"receipt:{response.raw_input.frame_id}",
                    ),
                )
            )
            action_only_series = bool(
                checkpoint.workflow_definition_id
                == "arc3.continue_declared_action.continue_declared_action"
                and checkpoint.pending_world_context is not None
                and checkpoint.pending_world_context.get("continue_series") is True
            )
            completed = self._scheduler.advance(
                workflow_instance_id,
                durable_boundaries_only=action_only_series,
            ).checkpoint
            return BoundaryDispatchResult(
                workflow_instance_id=workflow_instance_id,
                dispatch_id=dispatched.dispatch_id,
                environment_action_event_ref=dispatched.id,
                frame_id=resumed.frame_id or response.raw_input.frame_id,
                workflow_status=completed.status.value,
                state_revision=self._store.snapshot.revision,
                reused=reused,
            )

    def _record_attempt(
        self,
        checkpoint: WorkflowInstanceRecord,
        request: EnvironmentActionRequest,
    ) -> EnvironmentActionAttemptRecord:
        definition = environment_symbol_definition()
        record = EnvironmentActionAttemptRecord(
            id=f"attempt:{request.dispatch_id}",
            source_unit=definition.source_unit,
            source_hash=definition.source_hash,
            state_revision=self._store.snapshot.revision,
            workflow_definition_id=checkpoint.workflow_definition_id,
            workflow_instance_id=checkpoint.workflow_instance_id,
            workflow_step_id=checkpoint.workflow_step_id,
            timeline_id=checkpoint.timeline_id,
            frame_id=checkpoint.frame_id,
            provenance=(request.intent.id, request.permit.id),
            dispatch_id=request.dispatch_id,
            environment_adapter_ref=request.environment_adapter_ref,
            action_intent_id=request.intent.id,
            action_release_permit_id=request.permit.id,
            action_ref=request.intent.action_ref,
            request_digest=request.request_digest,
        )
        self._store.commit(
            TransactionRequest(
                transaction_id=f"{request.dispatch_id}:attempt",
                run_id=self._store.run_id,
                timeline_id=request.timeline_id,
                frame_id=request.frame_id,
                expected_revision=self._store.snapshot.revision,
                source_hash=definition.source_hash,
                mutations=(RecordArtifact(artifact=record),),
            )
        )
        return record

    def _record_dispatch(
        self,
        checkpoint: WorkflowInstanceRecord,
        request: EnvironmentActionRequest,
        response: EnvironmentActionResponse,
    ) -> EnvironmentActionDispatchedRecord:
        definition = environment_symbol_definition()
        record = EnvironmentActionDispatchedRecord(
            id=response.environment_action_event_ref,
            source_unit=definition.source_unit,
            source_hash=definition.source_hash,
            state_revision=self._store.snapshot.revision,
            workflow_definition_id=checkpoint.workflow_definition_id,
            workflow_instance_id=checkpoint.workflow_instance_id,
            workflow_step_id=checkpoint.workflow_step_id,
            timeline_id=checkpoint.timeline_id,
            frame_id=checkpoint.frame_id,
            provenance=(request.intent.id, request.permit.id),
            dispatch_id=request.dispatch_id,
            environment_adapter_ref=request.environment_adapter_ref,
            action_intent_id=request.intent.id,
            action_release_permit_id=request.permit.id,
            action_ref=request.intent.action_ref,
            request_digest=request.request_digest,
            response_digest=response.response_digest,
            raw_input_ref=response.raw_input.raw_input_ref,
        )
        self._store.commit(
            TransactionRequest(
                transaction_id=f"{request.dispatch_id}:record",
                run_id=self._store.run_id,
                timeline_id=request.timeline_id,
                frame_id=request.frame_id,
                expected_revision=self._store.snapshot.revision,
                source_hash=definition.source_hash,
                mutations=(RecordArtifact(artifact=record),),
            )
        )
        return record

    @staticmethod
    def _validate_response(
        request: EnvironmentActionRequest,
        response: EnvironmentActionResponse,
    ) -> None:
        if response.dispatch_id != request.dispatch_id:
            raise ActionBoundaryError("transport changed dispatch identity")
        if response.action_intent_id != request.intent.id:
            raise ActionBoundaryError("transport changed action-intent identity")
        if response.action_release_permit_id != request.permit.id:
            raise ActionBoundaryError("transport changed release-permit identity")

    def _dispatched_for_intent(
        self,
        intent_id: str,
    ) -> EnvironmentActionDispatchedRecord | None:
        self._refresh_artifact_indexes()
        return self._dispatch_by_intent.get(intent_id)

    def _attempt_for_intent(
        self,
        intent_id: str,
    ) -> EnvironmentActionAttemptRecord | None:
        self._refresh_artifact_indexes()
        return self._attempt_by_intent.get(intent_id)

    def _observed_for_workflow(
        self,
        workflow_instance_id: str,
        *,
        intent_id: str | None,
    ) -> ActionObservedRecord | None:
        self._refresh_artifact_indexes()
        if intent_id is not None:
            return self._observed_by_workflow_intent.get(
                (workflow_instance_id, intent_id)
            )
        return next(
            (
                artifact
                for (known_workflow_id, _), artifact
                in reversed(tuple(self._observed_by_workflow_intent.items()))
                if known_workflow_id == workflow_instance_id
            ),
            None,
        )

    def _refresh_artifact_indexes(self) -> None:
        """Index only new action records, never the unrelated audit journal."""

        for artifact_type in (
            EnvironmentActionDispatchedRecord,
            EnvironmentActionAttemptRecord,
            ActionObservedRecord,
        ):
            artifacts = self._store.artifacts_of_type(artifact_type)
            start = self._indexed_action_artifact_counts.get(artifact_type, 0)
            for index in range(start, len(artifacts)):
                artifact = artifacts[index]
                if isinstance(artifact, EnvironmentActionDispatchedRecord):
                    self._dispatch_by_intent[artifact.action_intent_id] = artifact
                elif isinstance(artifact, EnvironmentActionAttemptRecord):
                    self._attempt_by_intent[artifact.action_intent_id] = artifact
                elif isinstance(artifact, ActionObservedRecord):
                    self._observed_by_workflow_intent[
                        (artifact.workflow_instance_id or "", artifact.action_intent_id)
                    ] = artifact
            self._indexed_action_artifact_counts[artifact_type] = len(artifacts)

    def _artifacts(self) -> Sequence[object]:
        return self._store.artifacts


def _dispatch_request(
    *,
    checkpoint: WorkflowInstanceRecord,
    intent: ActionIntent,
    permit: ActionReleasePermit,
    world_context: Mapping[str, object],
) -> EnvironmentActionRequest:
    if (
        checkpoint.pending_environment_adapter_ref is None
        or checkpoint.workflow_definition_id is None
        or checkpoint.workflow_instance_id is None
        or checkpoint.timeline_id is None
        or checkpoint.frame_id is None
    ):
        raise ActionBoundaryError("suspended workflow has incomplete dispatch identity")
    frozen_context = FrozenMap(world_context)
    # Preserve every lineage component while avoiding a second recursive JSON
    # walk through the large immutable intent/permit/context payload.  Each
    # sub-object has a canonical digest; hashing the small envelope is
    # equivalent for equality and collision resistance, and is substantially
    # cheaper on the per-primitive dispatch path.
    values = {
        "environment_adapter_ref": checkpoint.pending_environment_adapter_ref,
        "workflow_definition_id": checkpoint.workflow_definition_id,
        "workflow_instance_id": checkpoint.workflow_instance_id,
        "timeline_id": checkpoint.timeline_id,
        "frame_id": checkpoint.frame_id,
        "intent_digest": stable_digest(intent),
        "permit_digest": stable_digest(permit),
        "world_context_digest": stable_digest(frozen_context),
    }
    digest = stable_digest(FrozenMap(values))
    return EnvironmentActionRequest(
        dispatch_id=f"dispatch:{intent.id}",
        environment_adapter_ref=checkpoint.pending_environment_adapter_ref,
        workflow_definition_id=checkpoint.workflow_definition_id,
        workflow_instance_id=checkpoint.workflow_instance_id,
        timeline_id=checkpoint.timeline_id,
        frame_id=checkpoint.frame_id,
        intent=intent,
        permit=permit,
        world_context=frozen_context,
        request_digest=digest,
    )


def _record_binding(
    value: object,
    *,
    record_kind: str,
    record_id: str | None,
) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        if value.get("record_kind") == record_kind and value.get("id") == record_id:
            return value
        for child in (value[__yf_order_key] for __yf_order_key in sorted(value)):
            try:
                return _record_binding(
                    child,
                    record_kind=record_kind,
                    record_id=record_id,
                )
            except LookupError:
                pass
    elif isinstance(value, (tuple, list)):
        for child in value:
            try:
                return _record_binding(
                    child,
                    record_kind=record_kind,
                    record_id=record_id,
                )
            except LookupError:
                pass
    raise LookupError(f"missing {record_kind} binding {record_id}")


def _record_binding_from_preferred(
    bindings: Mapping[str, object],
    *,
    preferred_binding_names: tuple[str, ...],
    record_kind: str,
    record_id: str | None,
) -> Mapping[str, object]:
    """Resolve a declared record from its SRC binding before bounded fallback."""

    for name in preferred_binding_names:
        if name not in bindings:
            continue
        try:
            return _record_binding(
                bindings[name],
                record_kind=record_kind,
                record_id=record_id,
            )
        except LookupError:
            pass
    return _record_binding(
        bindings,
        record_kind=record_kind,
        record_id=record_id,
    )
