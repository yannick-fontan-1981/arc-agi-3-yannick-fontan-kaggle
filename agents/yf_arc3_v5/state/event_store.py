"""Append-only event authority with atomic reducer-validated commits."""

from __future__ import annotations

from collections import ChainMap
from collections.abc import Sequence
from threading import RLock
from typing import Generic, TypeVar, overload

from agents.yf_arc3_v5.logos.types import Ref
from agents.yf_arc3_v5.logos.claims import Claim
from agents.yf_arc3_v5.logos.operations import (
    ConcentrationActionAccountingRecord, ConcentrationGovernanceRecord,
    ConcentrationLease, ConcentrationLeaseState, ConcentrationLeaseTransition,
    ConcentrationOutcome, WorkflowInstanceRecord,
)
from agents.yf_arc3_v5.performance import begin_profile, finish_profile
from agents.yf_arc3_v5.state.events import (
    CognitiveArtifact,
    CognitiveEvent,
    PutClaim,
    PutTerm,
    RecordArtifact,
    StateMutation,
)
from agents.yf_arc3_v5.state.reducer import (
    ReductionError,
    reduce_event_delta,
    reduce_events,
)
from agents.yf_arc3_v5.state.snapshot import CognitiveSnapshot
from agents.yf_arc3_v5.state.transactions import (
    DuplicateTransactionError,
    StateConflictError,
    TransactionReceipt,
    TransactionRequest,
)


_T = TypeVar("_T")
_TERMINAL_CONCENTRATION_STATES = frozenset((
    ConcentrationLeaseState.RECONCILED, ConcentrationLeaseState.INTERRUPTED,
    ConcentrationLeaseState.INVALIDATED, ConcentrationLeaseState.EXPIRED,
))


class _ReadOnlyListView(Sequence[_T], Generic[_T]):
    """Zero-copy sequence view over one append-only canonical list."""

    __slots__ = ("_values",)

    def __init__(self, values: list[_T]) -> None:
        self._values = values

    @overload
    def __getitem__(self, index: int) -> _T: ...

    @overload
    def __getitem__(self, index: slice) -> list[_T]: ...

    def __getitem__(self, index: int | slice) -> _T | list[_T]:
        return self._values[index]

    def __len__(self) -> int:
        return len(self._values)


class EventStore:
    """The only mutable container in the M1 state path.

    Mutation is append-only and occurs only after reducing the complete
    candidate event sequence succeeds.  A failed transaction leaves both the
    journal and snapshot unchanged.
    """

    def __init__(self, run_id: Ref, *, snapshot_schema_version: Ref = "yf_arc3_v5.cognitive_snapshot.v2") -> None:
        if snapshot_schema_version not in (
            "yf_arc3_v5.cognitive_snapshot.v2", "yf_arc3_v5.cognitive_snapshot.v3",
            "yf_arc3_v5.cognitive_snapshot.v4",
        ):
            raise ValueError("unsupported event-store snapshot hash version")
        self._run_id = run_id
        self._events: list[CognitiveEvent] = []
        self._event_view = _ReadOnlyListView(self._events)
        self._artifacts: list[CognitiveArtifact] = []
        self._artifact_view = _ReadOnlyListView(self._artifacts)
        self._artifact_by_id: dict[str, CognitiveArtifact] = {}
        self._latest_workflow_by_instance: dict[str, WorkflowInstanceRecord] = {}
        self._artifacts_by_type: dict[type[object], list[CognitiveArtifact]] = {}
        self._artifact_type_views: dict[
            type[object], _ReadOnlyListView[CognitiveArtifact]
        ] = {}
        self._event_ids: set[str] = set()
        self._artifact_ids: set[str] = set()
        self._transaction_ids: set[str] = set()
        # Archive identity index, deliberately absent from the operational
        # snapshot. No lookup falls back from current meaning to old evidence.
        self._claim_evidence_by_id: dict[Ref, Claim] = {}
        # Mechanical projections of accepted records, never a second decision
        # authority. Validation reads these current values plus its own delta.
        self._concentration_states: dict[Ref, ConcentrationLeaseState] = {}
        self._concentration_owner_by_timeline: dict[Ref, Ref] = {}
        self._concentration_obstination: dict[Ref, int] = {}
        self._concentration_accounted_actions: set[tuple[Ref, Ref]] = set()
        self._concentration_credited_results: set[Ref] = set()
        self._snapshot = CognitiveSnapshot.empty(run_id, schema_version=snapshot_schema_version)
        self._commit_lock = RLock()

    @property
    def run_id(self) -> Ref:
        return self._run_id

    @property
    def events(self) -> tuple[CognitiveEvent, ...]:
        return tuple(self._events)

    @property
    def event_view(self) -> Sequence[CognitiveEvent]:
        """Expose the append-only journal without copying its growing history."""

        return self._event_view

    @property
    def artifacts(self) -> Sequence[CognitiveArtifact]:
        """Expose artifacts directly, without rescanning the growing journal."""

        return self._artifact_view

    def artifact(self, artifact_id: Ref) -> CognitiveArtifact | None:
        """Return one hot canonical artefact without rescanning the cold journal."""

        return self._artifact_by_id.get(artifact_id)

    def artifacts_of_type(
        self,
        artifact_type: type[object],
    ) -> Sequence[CognitiveArtifact]:
        """Expose one append-only typed hot index without copying history."""

        view = self._artifact_type_views.get(artifact_type)
        return () if view is None else view

    def latest_artifact(
        self,
        artifact_type: type[object],
    ) -> CognitiveArtifact | None:
        """Return the latest typed artifact in constant time."""

        values = self._artifacts_by_type.get(artifact_type)
        return None if not values else values[-1]

    def latest_workflow_instance(self, instance_id: Ref) -> WorkflowInstanceRecord | None:
        """Read the latest accepted checkpoint for one instance, without a journal scan."""

        return self._latest_workflow_by_instance.get(instance_id)

    def concentration_state(self, lease_id: Ref):
        """Return the last canonical lifecycle state, or None for an unknown lease."""
        lease = self.artifact(lease_id)
        if not isinstance(lease, ConcentrationLease):
            return None
        return self._concentration_states.get(lease_id, lease.state)

    def _index_concentration_artifact(self, artifact: CognitiveArtifact) -> None:
        """Apply one accepted delta; replay calls this once per archived record."""
        if isinstance(artifact, ConcentrationLease):
            self._concentration_states[artifact.id] = artifact.state
            if artifact.state not in _TERMINAL_CONCENTRATION_STATES:
                self._concentration_owner_by_timeline[artifact.timeline_id] = artifact.id
        elif isinstance(artifact, ConcentrationLeaseTransition):
            self._concentration_states[artifact.concentration_lease_id] = artifact.to_state
            if artifact.to_state in _TERMINAL_CONCENTRATION_STATES:
                if self._concentration_owner_by_timeline.get(artifact.timeline_id) == artifact.concentration_lease_id:
                    self._concentration_owner_by_timeline.pop(artifact.timeline_id)
            else:
                self._concentration_owner_by_timeline[artifact.timeline_id] = artifact.concentration_lease_id
        elif isinstance(artifact, ConcentrationActionAccountingRecord):
            self._concentration_obstination[artifact.pursuit_ref] = artifact.obstination_count
            self._concentration_accounted_actions.add((artifact.pursuit_ref, artifact.action_intent_id))
        elif isinstance(artifact, ConcentrationGovernanceRecord):
            if artifact.outcome is ConcentrationOutcome.SATISFIED and artifact.result_ref is not None:
                self._concentration_credited_results.add(artifact.result_ref)

    @property
    def snapshot(self) -> CognitiveSnapshot:
        return self._snapshot

    def archived_claim_evidence(self, claim_id: Ref) -> Claim | None:
        """Explicit exact proof lookup, not a current-meaning query.

        Presence here establishes neither current validity nor authorization.
        Retention of this migration journal is separate from future opt-in
        diagnostics; it must not be interpreted as bounded total memory yet.
        """
        if self._snapshot.schema_version != "yf_arc3_v5.cognitive_snapshot.v4":
            return self._snapshot.claim(claim_id)
        return self._claim_evidence_by_id.get(claim_id)

    def _normalize_mutation(
        self,
        mutation: StateMutation,
        next_revision: int,
    ) -> StateMutation:
        if not isinstance(mutation, PutTerm):
            return mutation
        after = mutation.after
        if mutation.before is None:
            if after.created_at_state_revision not in (0, next_revision):
                raise StateConflictError("new term has an invalid creation revision")
            if after.last_changed_state_revision not in (0, next_revision):
                raise StateConflictError("new term has an invalid change revision")
            normalized = after.model_copy(
                update={
                    "created_at_state_revision": next_revision,
                    "last_changed_state_revision": next_revision,
                }
            )
        else:
            if (
                after.created_at_state_revision
                != mutation.before.created_at_state_revision
            ):
                raise StateConflictError(
                    "term revision cannot change creation revision"
                )
            if after.last_changed_state_revision not in (
                mutation.before.last_changed_state_revision,
                next_revision,
            ):
                raise StateConflictError("term revision has an invalid change revision")
            normalized = after.model_copy(
                update={"last_changed_state_revision": next_revision}
            )
        return PutTerm(before=mutation.before, after=normalized)

    def prepare_mutations(
        self,
        mutations: tuple[StateMutation, ...],
    ) -> tuple[StateMutation, ...]:
        """Preview the exact immutable mutations the next commit would reduce."""

        next_revision = self._snapshot.revision + (
            1
            if any(isinstance(mutation, (PutTerm, PutClaim)) for mutation in mutations)
            else 0
        )
        return tuple(
            self._normalize_mutation(mutation, next_revision) for mutation in mutations
        )

    def commit(self, request: TransactionRequest) -> TransactionReceipt:
        started = begin_profile()
        succeeded = False
        try:
            with self._commit_lock:
                before_sequence = len(self._events)
                receipt = self._commit(request)
                succeeded = True
                return receipt
        finally:
            # Profiling is opt-in.  Avoid constructing even the compact event
            # suffix when no recorder is active: commit is a hot delta path and
            # one reasoning turn can contain hundreds of tiny transactions.
            if started is not None:
                finish_profile(
                    started,
                    kind="state",
                    name="event_store.commit",
                    metadata={
                        "committed": succeeded,
                        "event_sequences": (
                            tuple(range(before_sequence + 1, len(self._events) + 1))
                            if succeeded
                            else ()
                        ),
                    },
                )

    def _commit(self, request: TransactionRequest) -> TransactionReceipt:
        if request.run_id != self._run_id:
            raise StateConflictError(
                "transaction run identity does not match event store"
            )
        if request.expected_revision != self._snapshot.revision:
            raise StateConflictError(
                f"stale transaction revision: expected {request.expected_revision}, "
                f"current {self._snapshot.revision}"
            )
        if request.transaction_id in self._transaction_ids:
            raise DuplicateTransactionError(request.transaction_id)

        has_state_mutation = any(
            isinstance(mutation, (PutTerm, PutClaim)) for mutation in request.mutations
        )
        next_revision = self._snapshot.revision + (1 if has_state_mutation else 0)
        start_sequence = len(self._events) + 1
        normalized = self.prepare_mutations(request.mutations)
        self._validate_concentration_mutations(normalized)
        new_events = tuple(
            CognitiveEvent.create(
                event_id=f"{self._run_id}:event:{start_sequence + index:08d}",
                sequence=start_sequence + index,
                transaction_id=request.transaction_id,
                run_id=self._run_id,
                timeline_id=request.timeline_id,
                frame_id=request.frame_id,
                state_revision_before=self._snapshot.revision,
                state_revision_after=next_revision,
                source_hash=request.source_hash,
                mutation=mutation,
                schema_version=("yf_arc3_v5.cognitive_event.v4"
                    if self._snapshot.schema_version == "yf_arc3_v5.cognitive_snapshot.v4"
                    else "yf_arc3_v5.cognitive_event.v3"
                    if self._snapshot.schema_version == "yf_arc3_v5.cognitive_snapshot.v3"
                    else "yf_arc3_v5.cognitive_event.v2"),
            )
            for index, mutation in enumerate(normalized)
        )

        try:
            candidate_snapshot = reduce_event_delta(
                self._snapshot,
                new_events,
                existing_event_ids=self._event_ids,
                existing_artifact_ids=self._artifact_ids,
                existing_claim_ids=self._claim_evidence_by_id.keys(),
            )
        except ReductionError as exc:
            raise StateConflictError(str(exc)) from exc

        self._events.extend(new_events)
        if candidate_snapshot.schema_version == "yf_arc3_v5.cognitive_snapshot.v4":
            self._claim_evidence_by_id.update((event.mutation.after.id, event.mutation.after)
                for event in new_events if isinstance(event.mutation, PutClaim))
        new_artifacts = tuple(
            event.mutation.artifact
            for event in new_events
            if isinstance(event.mutation, RecordArtifact)
        )
        self._artifacts.extend(new_artifacts)
        self._event_ids.update(event.event_id for event in new_events)
        self._artifact_ids.update(
            artifact.id for artifact in new_artifacts
        )
        self._artifact_by_id.update(
            (artifact.id, artifact) for artifact in new_artifacts
        )
        for artifact in new_artifacts:
            artifact_type = type(artifact)
            if isinstance(artifact, WorkflowInstanceRecord):
                self._latest_workflow_by_instance[artifact.workflow_instance_id] = artifact
            typed = self._artifacts_by_type.get(artifact_type)
            if typed is None:
                typed = []
                self._artifacts_by_type[artifact_type] = typed
                self._artifact_type_views[artifact_type] = _ReadOnlyListView(typed)
            typed.append(artifact)
            self._index_concentration_artifact(artifact)
        self._transaction_ids.add(request.transaction_id)
        self._snapshot = candidate_snapshot
        # The events and snapshot were just reducer-validated above.  Building
        # the local receipt without recursive Pydantic validation avoids
        # recomputing every event content hash once more per transaction.
        return TransactionReceipt.model_construct(
            transaction_id=request.transaction_id,
            committed_events=new_events,
            snapshot=candidate_snapshot,
            schema_version="yf_arc3_v5.transaction_receipt.v1",
        )

    def _validate_concentration_mutations(self, mutations: Sequence[StateMutation]) -> None:
        """Enforce exclusive focus at the same atomic boundary as journal append.

        Artifact-only transactions do not advance the snapshot revision, so a
        revision check alone cannot serialize competing action leases.
        """
        from agents.yf_arc3_v5.logos.operations import (
            ConcentrationActionAccountingRecord,
            ConcentrationGovernanceRecord,
            ConcentrationLease, ConcentrationLeaseState,
            ConcentrationLeaseTransition,
            ConcentrationOperation,
            ConcentrationOutcome,
        )

        if not any(
            isinstance(mutation, RecordArtifact)
            and isinstance(
                mutation.artifact,
                (
                    ConcentrationLease,
                    ConcentrationLeaseTransition,
                    ConcentrationGovernanceRecord,
                    ConcentrationActionAccountingRecord,
                ),
            )
            for mutation in mutations
        ):
            return

        terminal = _TERMINAL_CONCENTRATION_STATES
        allowed = {
            ConcentrationLeaseState.ACQUIRED: {
                ConcentrationLeaseState.VALIDATED,
                ConcentrationLeaseState.INVALIDATED,
                ConcentrationLeaseState.EXPIRED,
            },
            ConcentrationLeaseState.VALIDATED: {
                ConcentrationLeaseState.RELEASED,
                ConcentrationLeaseState.SUSPENDED,
                ConcentrationLeaseState.INTERRUPTED,
                ConcentrationLeaseState.INVALIDATED,
                ConcentrationLeaseState.EXPIRED,
            },
            ConcentrationLeaseState.RELEASED: {
                ConcentrationLeaseState.OBSERVED,
                ConcentrationLeaseState.INTERRUPTED,
            },
            ConcentrationLeaseState.OBSERVED: {
                ConcentrationLeaseState.RELEASED,
                ConcentrationLeaseState.RECONCILED,
                ConcentrationLeaseState.SUSPENDED,
                ConcentrationLeaseState.INTERRUPTED,
                ConcentrationLeaseState.INVALIDATED,
                ConcentrationLeaseState.EXPIRED,
            },
            ConcentrationLeaseState.SUSPENDED: {
                ConcentrationLeaseState.VALIDATED,
                ConcentrationLeaseState.INTERRUPTED,
                ConcentrationLeaseState.INVALIDATED,
                ConcentrationLeaseState.EXPIRED,
            },
        }
        # Overlay only this transaction. No historical scan, full index copy or
        # mutation of accepted state before the reducer validates the whole batch.
        states = ChainMap({}, self._concentration_states)
        leases = ChainMap({}, self._artifact_by_id)
        owners = ChainMap({}, self._concentration_owner_by_timeline)
        pursuit_obstination = ChainMap({}, self._concentration_obstination)
        accounted_actions: set[tuple[Ref, Ref]] = set()
        credited_results: set[Ref] = set()
        transaction_closures = {
            mutation.artifact.concentration_lease_id
            for mutation in mutations
            if isinstance(mutation, RecordArtifact)
            and isinstance(mutation.artifact, ConcentrationLeaseTransition)
            and mutation.artifact.to_state is ConcentrationLeaseState.RECONCILED
        }
        for mutation in mutations:
            if not isinstance(mutation, RecordArtifact):
                continue
            artifact = mutation.artifact
            if isinstance(artifact, ConcentrationLease):
                if owners.get(artifact.timeline_id) is not None:
                    raise StateConflictError("another concentration lease owns this timeline")
                leases[artifact.id] = artifact
                states[artifact.id] = artifact.state
                if artifact.state not in terminal:
                    owners[artifact.timeline_id] = artifact.id
            elif isinstance(artifact, ConcentrationLeaseTransition):
                lease = leases.get(artifact.concentration_lease_id)
                if not isinstance(lease, ConcentrationLease):
                    raise StateConflictError("concentration transition has no canonical lease")
                if (
                    artifact.timeline_id != lease.timeline_id
                    or artifact.workflow_instance_id != lease.workflow_instance_id
                ):
                    raise StateConflictError("concentration transition is outside its owning workflow")
                current = states[lease.id]
                if current != artifact.from_state:
                    raise StateConflictError("concentration transition is out of order")
                if artifact.to_state not in allowed.get(current, set()):
                    raise StateConflictError("concentration transition is not allowed")
                if artifact.action_intent_id not in lease.extension_intent_ids:
                    raise StateConflictError("concentration transition intent is outside lease")
                states[lease.id] = artifact.to_state
                owners[lease.timeline_id] = None if artifact.to_state in terminal else lease.id
            elif isinstance(artifact, ConcentrationGovernanceRecord):
                lease = leases.get(artifact.concentration_lease_id)
                if not isinstance(lease, ConcentrationLease):
                    raise StateConflictError("concentration governance has no canonical lease")
                if artifact.timeline_id != lease.timeline_id:
                    raise StateConflictError("concentration governance timeline mismatch")
                if artifact.why_contract_ref != lease.id:
                    raise StateConflictError("concentration governance lost its frozen why")
                if artifact.operation is not None and (
                    artifact.workflow_instance_id != lease.workflow_instance_id
                ):
                    raise StateConflictError("concentration operation is outside its owning workflow")
                if states[lease.id] in terminal and (
                    artifact.operation is not None or artifact.outcome is not None
                ) and not (
                    artifact.operation is ConcentrationOperation.CLOSE
                    and lease.id in transaction_closures
                    and states[lease.id] is ConcentrationLeaseState.RECONCILED
                ):
                    raise StateConflictError("revoked or closed focus cannot issue a late operation or outcome")
                required = set(lease.preserve_constraint_refs) | set(
                    lease.overshoot_constraint_refs
                )
                if artifact.operation is not None and not required.issubset(
                    artifact.carried_obligation_refs
                ):
                    raise StateConflictError("concentration governance dropped obligations")
                if (
                    artifact.inherited_action_budget is not None
                    and lease.remaining_action_bound is not None
                    and artifact.inherited_action_budget > lease.remaining_action_bound
                ):
                    raise StateConflictError("concentration governance enlarged inherited budget")
                if artifact.outcome is ConcentrationOutcome.SATISFIED and artifact.result_ref:
                    if artifact.result_ref in credited_results or artifact.result_ref in self._concentration_credited_results:
                        raise StateConflictError("shared result was already credited")
                    credited_results.add(artifact.result_ref)
            elif isinstance(artifact, ConcentrationActionAccountingRecord):
                lease = leases.get(artifact.concentration_lease_id)
                if not isinstance(lease, ConcentrationLease):
                    raise StateConflictError("concentration accounting has no canonical lease")
                if artifact.timeline_id != lease.timeline_id:
                    raise StateConflictError("concentration accounting timeline mismatch")
                if artifact.action_intent_id not in lease.extension_intent_ids:
                    raise StateConflictError("concentration accounting intent is outside lease")
                key = (artifact.pursuit_ref, artifact.action_intent_id)
                if key in accounted_actions or key in self._concentration_accounted_actions:
                    raise StateConflictError("concentration action was already counted")
                expected = pursuit_obstination.get(artifact.pursuit_ref, 0) + (
                    0 if artifact.relevant_to_pursuit else 1
                )
                if artifact.obstination_count != expected:
                    raise StateConflictError("obstination counter must advance by exactly one per irrelevant action")
                pursuit_obstination[artifact.pursuit_ref] = expected
                accounted_actions.add(key)

    def to_jsonl(self) -> str:
        return "\n".join(event.model_dump_json() for event in self._events)

    @classmethod
    def from_jsonl(cls, text: str) -> "EventStore":
        lines = tuple(line for line in text.splitlines() if line.strip())
        if not lines:
            raise ValueError("cannot restore an empty event store without run identity")
        events = tuple(CognitiveEvent.model_validate_json(line) for line in lines)
        snapshot = reduce_events(events)
        store = cls(snapshot.run_id, snapshot_schema_version=snapshot.schema_version)
        store._events = list(events)
        store._event_view = _ReadOnlyListView(store._events)
        store._artifacts = [
            event.mutation.artifact
            for event in events
            if isinstance(event.mutation, RecordArtifact)
        ]
        store._artifact_view = _ReadOnlyListView(store._artifacts)
        store._artifact_by_id = {
            artifact.id: artifact for artifact in store._artifacts
        }
        for artifact in store._artifacts:
            artifact_type = type(artifact)
            if isinstance(artifact, WorkflowInstanceRecord):
                store._latest_workflow_by_instance[artifact.workflow_instance_id] = artifact
            typed = store._artifacts_by_type.get(artifact_type)
            if typed is None:
                typed = []
                store._artifacts_by_type[artifact_type] = typed
                store._artifact_type_views[artifact_type] = _ReadOnlyListView(typed)
            typed.append(artifact)
            store._index_concentration_artifact(artifact)
        store._event_ids = {event.event_id for event in events}
        store._artifact_ids = {
            event.mutation.artifact.id
            for event in events
            if isinstance(event.mutation, RecordArtifact)
        }
        store._transaction_ids = {event.transaction_id for event in events}
        if snapshot.schema_version == "yf_arc3_v5.cognitive_snapshot.v4":
            store._claim_evidence_by_id = {event.mutation.after.id: event.mutation.after
                for event in events if isinstance(event.mutation, PutClaim)}
        store._snapshot = snapshot
        return store
