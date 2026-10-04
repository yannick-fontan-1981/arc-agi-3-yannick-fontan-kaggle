"""Audited, fail-closed runtime contracts for the nine Logos primitives."""

from __future__ import annotations

from dataclasses import dataclass

from agents.yf_arc3_v5.logos.claims import Claim
from agents.yf_arc3_v5.logos.operations import (
    AlternativeSet,
    ActionReleasePermit,
    ConcentrationLeaseState,
    ConcentrationLeaseTransition,
    ConcentrationOperation,
    project_concentration_operation,
    ObservationAcquiredRecord,
    OperationStatus,
    OperatorInvocation,
    OperatorResult,
    StateChange,
    StateChangeCommittedRecord,
    StateChangeProposal,
)
from agents.yf_arc3_v5.logos.types import (
    Cardinality,
    EffectClass,
    EpistemicStatus,
    FrozenMap,
    Ref,
    immutable_json_data,
    stable_digest,
)
from agents.yf_arc3_v5.operators.authority import (
    OperatorAuthorityRegistry,
    RegisteredAuthority,
    RuntimeAuthorityKind,
)
from agents.yf_arc3_v5.operators.contracts import (
    AcquisitionInput,
    AcquisitionResult,
    ActOutput,
    ActRequest,
    AnyOperatorRequest,
    ClaimSeed,
    CompareOutput,
    CompareRequest,
    ComparisonInput,
    ComparisonResult,
    DerivationInput,
    DerivationOutput,
    DerivationResult,
    DeriveRequest,
    DistinctionInput,
    DistinctionResult,
    DistinguishOutput,
    DistinguishRequest,
    ObservationOutput,
    ObserveRequest,
    OperatorExecution,
    OperatorOutput,
    PredicateInput,
    PredicateResult,
    ProposalOutput,
    ProposeRequest,
    RelateOutput,
    RelateRequest,
    SelectionInput,
    SelectionResult,
    SelectOutput,
    SelectRequest,
    UpdateOutput,
    UpdatePolicyInput,
    UpdatePolicyResult,
    UpdateRequest,
)
from agents.yf_arc3_v5.state import EventStore, PutClaim, PutTerm, RecordArtifact
from agents.yf_arc3_v5.state.events import CognitiveArtifact, StateMutation
from agents.yf_arc3_v5.state.transactions import TransactionReceipt, TransactionRequest
from agents.yf_arc3_v5.operators.selection_projection import project_selection_output


class OperatorRuntimeError(ValueError):
    """The request cannot execute under the current canonical state."""


@dataclass(frozen=True)
class _BlockedExecution(Exception):
    reason_refs: tuple[str, ...]
    message: str
    artifacts: tuple[CognitiveArtifact, ...] = ()


def _claim_from_seed(
    seed: ClaimSeed,
    *,
    epistemic_status: EpistemicStatus,
    grounds: tuple[Ref, ...],
    proof_rule: Ref | None,
    result_id: Ref,
    attributes: FrozenMap | None = None,
) -> Claim:
    return Claim(
        id=seed.id,
        predicate=seed.predicate,
        family=seed.family,
        arguments=seed.arguments,
        polarity=seed.polarity,
        epistemic_status=epistemic_status,
        disposition=seed.disposition,
        cardinality=seed.cardinality,
        scope=seed.scope,
        conditions=seed.conditions,
        grounds=grounds,
        proof_rule=proof_rule,
        source_operator_results=(result_id,),
        attributes=seed.attributes if attributes is None else attributes,
    )


class OperatorRuntime:
    """Runs operator contracts while EventStore remains the sole journal writer."""

    def __init__(
        self,
        event_store: EventStore,
        authorities: OperatorAuthorityRegistry,
        *,
        diagnostic_outputs: bool = False,
        persist_pure_operator_audit: bool = True,
    ) -> None:
        self._store = event_store
        self._authorities = authorities
        self._diagnostic_outputs = diagnostic_outputs
        self._persist_pure_operator_audit = persist_pure_operator_audit
        self._transient_select_receipts = False

    def set_durable_boundary_mode(self, enabled: bool) -> bool:
        """Return the old mode; the scheduler restores it after one advance."""
        prior = self._transient_select_receipts
        self._transient_select_receipts = enabled
        return prior

    @staticmethod
    def _pure_request(request: AnyOperatorRequest) -> bool:
        # Closed list: acquisition, state writes and action release MUST retain
        # their recovery journal. New operators are durable unless reviewed.
        return isinstance(request, (
            DistinguishRequest, RelateRequest, ProposeRequest,
            DeriveRequest, CompareRequest, SelectRequest,
        ))

    def replay_pure_output(
        self, request: AnyOperatorRequest, result: OperatorResult,
    ) -> OperatorOutput:
        """Recompute a pure interrupted step, never an UPDATE/OBSERVE/ACT.

        Normal continuation uses the scheduler's current bindings. Only recovery
        before its checkpoint needs this path; historical output copies are not
        operational memory. Legacy journals still carry their original output.
        """
        self._validate_request_state(request)
        if (not self._pure_request(request)
                or result.attributes.get("output_recovery") != "recompute_pure"
                or result.status is not OperationStatus.COMPLETED
                or result.id != f"{request.request_id}:result"
                or result.source_hash != request.source_hash
                or result.operator is not request.operator
                or result.state_revision != self._store.snapshot.revision
                or result.attributes.get("semantic_state_hash_after")
                != self._store.snapshot.state_hash):
            raise OperatorRuntimeError("pure output recovery lineage mismatch")
        authority = self._resolve_authority(request)
        invocation = self._store.artifact(result.invocation_id)
        if (not isinstance(invocation, OperatorInvocation)
                or invocation.authority_ref != self._authority_ref(request)
                or (authority is not None and
                    (invocation.authority_source_hash != authority.definition.source_hash
                     or invocation.authority_source_unit != authority.definition.source_unit))):
            raise OperatorRuntimeError("pure output recovery authority mismatch")
        output, artifacts, output_refs, reason_refs = self._dispatch(request, authority)
        if (artifacts or tuple(sorted(set(output_refs))) != result.output_refs
                or tuple(sorted(set(reason_refs))) != result.reason_refs
                or output.output_kind != result.attributes.get("output_kind")):
            raise OperatorRuntimeError("pure output recovery changed its contract")
        return output

    @property
    def event_store(self) -> EventStore:
        return self._store

    @property
    def authorities(self) -> OperatorAuthorityRegistry:
        """Expose the immutable registry view used by this runtime."""

        return self._authorities

    def execute(self, request: AnyOperatorRequest) -> OperatorExecution:
        self._validate_request_state(request)
        revision_before = self._store.snapshot.revision
        state_hash_before = self._store.snapshot.state_hash
        authority: RegisteredAuthority | None = None
        authority_error: Exception | None = None
        try:
            authority = self._resolve_authority(request)
        except Exception as error:  # recorded as a typed failed result below
            authority_error = error

        input_refs = self._input_refs(request)
        authority_ref = self._authority_ref(request)
        invocation = OperatorInvocation(
            id=f"{request.request_id}:invocation",
            source_unit=request.source_unit,
            source_hash=request.source_hash,
            state_revision=revision_before,
            workflow_definition_id=request.workflow_definition_id,
            workflow_instance_id=request.workflow_instance_id,
            workflow_step_id=request.workflow_step_id,
            timeline_id=request.timeline_id,
            frame_id=request.frame_id,
            provenance=input_refs,
            operator=request.operator,
            input_refs=input_refs,
            authority_ref=authority_ref,
            authority_source_unit=(
                None if authority is None else authority.definition.source_unit
            ),
            authority_source_hash=(
                None if authority is None else authority.definition.source_hash
            ),
            attributes=FrozenMap(
                {
                    "request_id": request.request_id,
                    "semantic_state_hash_before": state_hash_before,
                }
            ),
        )
        event_ids: tuple[Ref, ...] = ()
        if not self._pure_request(request):
            invocation_receipt = self._commit_artifacts(
                request,
                transaction_suffix="invocation",
                artifacts=(invocation,),
            )
            event_ids = tuple(
                event.event_id for event in invocation_receipt.committed_events
            )

        if authority_error is not None:
            return self._finish_non_completed(
                request,
                invocation,
                status=OperationStatus.FAILED,
                reason_refs=(f"authority_error:{type(authority_error).__name__}",),
                message=str(authority_error),
                revision_before=revision_before,
                state_hash_before=state_hash_before,
                prior_event_ids=event_ids,
            )

        try:
            if isinstance(request, UpdateRequest):
                assert authority is not None
                return self._execute_update(
                    request,
                    invocation,
                    authority,
                    revision_before=revision_before,
                    state_hash_before=state_hash_before,
                    prior_event_ids=event_ids,
                )
            output, artifacts, output_refs, reason_refs = self._dispatch(
                request,
                authority,
            )
            return self._finish_completed(
                request,
                invocation,
                output=output,
                artifacts=artifacts,
                output_refs=output_refs,
                reason_refs=reason_refs,
                revision_before=revision_before,
                state_hash_before=state_hash_before,
                prior_event_ids=event_ids,
            )
        except _BlockedExecution as blocked:
            return self._finish_non_completed(
                request,
                invocation,
                status=OperationStatus.BLOCKED,
                reason_refs=blocked.reason_refs,
                message=blocked.message,
                artifacts=blocked.artifacts,
                revision_before=revision_before,
                state_hash_before=state_hash_before,
                prior_event_ids=event_ids,
            )
        except Exception as error:
            return self._finish_non_completed(
                request,
                invocation,
                status=OperationStatus.FAILED,
                reason_refs=(f"execution_error:{type(error).__name__}",),
                message=str(error),
                revision_before=revision_before,
                state_hash_before=state_hash_before,
                prior_event_ids=event_ids,
            )

    def _validate_request_state(self, request: AnyOperatorRequest) -> None:
        if request.run_id != self._store.run_id:
            raise OperatorRuntimeError("operator request run identity mismatch")
        if request.expected_state_revision != self._store.snapshot.revision:
            raise OperatorRuntimeError(
                "operator request was built from a stale state revision"
            )

    def _authority_ref(self, request: AnyOperatorRequest) -> Ref | None:
        if isinstance(request, ObserveRequest):
            return request.acquisition_model_ref
        if isinstance(request, DistinguishRequest):
            return request.model_ref
        if isinstance(request, RelateRequest):
            return request.predicate_ref
        if isinstance(request, DeriveRequest):
            return request.rule_or_model_ref
        if isinstance(request, CompareRequest):
            return request.criterion_ref
        if isinstance(request, UpdateRequest):
            return request.update_policy_ref
        if isinstance(request, SelectRequest):
            return request.selection_policy_ref
        return None

    def _resolve_authority(
        self,
        request: AnyOperatorRequest,
    ) -> RegisteredAuthority | None:
        authority_id = self._authority_ref(request)
        if authority_id is None:
            return None
        if isinstance(request, ObserveRequest):
            return self._authorities.resolve(
                authority_id,
                operator=request.operator,
                allowed_kinds=frozenset({RuntimeAuthorityKind.ACQUISITION_MODEL}),
                required_effect=EffectClass.OBSERVATION_ADAPTER,
                input_contract="yf_arc3_v5.authority_input.acquisition.v1",
                output_contract="yf_arc3_v5.authority_output.acquisition.v1",
            )
        if isinstance(request, DistinguishRequest):
            return self._authorities.resolve(
                authority_id,
                operator=request.operator,
                allowed_kinds=frozenset(
                    {RuntimeAuthorityKind.MODEL, RuntimeAuthorityKind.CRITERION}
                ),
                required_effect=EffectClass.PURE,
                input_contract="yf_arc3_v5.authority_input.distinction.v1",
                output_contract="yf_arc3_v5.authority_output.distinction.v1",
            )
        if isinstance(request, RelateRequest):
            return self._authorities.resolve(
                authority_id,
                operator=request.operator,
                allowed_kinds=frozenset({RuntimeAuthorityKind.PREDICATE}),
                required_effect=EffectClass.PURE,
                input_contract="yf_arc3_v5.authority_input.predicate.v1",
                output_contract="yf_arc3_v5.authority_output.predicate.v1",
            )
        if isinstance(request, DeriveRequest):
            return self._authorities.resolve(
                authority_id,
                operator=request.operator,
                allowed_kinds=frozenset(
                    {RuntimeAuthorityKind.RULE, RuntimeAuthorityKind.MODEL}
                ),
                required_effect=EffectClass.PURE,
                input_contract="yf_arc3_v5.authority_input.derivation.v1",
                output_contract="yf_arc3_v5.authority_output.derivation.v1",
            )
        if isinstance(request, CompareRequest):
            return self._authorities.resolve(
                authority_id,
                operator=request.operator,
                allowed_kinds=frozenset({RuntimeAuthorityKind.CRITERION}),
                required_effect=EffectClass.PURE,
                input_contract="yf_arc3_v5.authority_input.comparison.v1",
                output_contract="yf_arc3_v5.authority_output.comparison.v1",
            )
        if isinstance(request, UpdateRequest):
            return self._authorities.resolve(
                authority_id,
                operator=request.operator,
                allowed_kinds=frozenset({RuntimeAuthorityKind.UPDATE_POLICY}),
                required_effect=EffectClass.COGNITIVE_UPDATE_ADAPTER,
                input_contract="yf_arc3_v5.authority_input.update_policy.v1",
                output_contract="yf_arc3_v5.authority_output.update_policy.v1",
            )
        if isinstance(request, SelectRequest):
            return self._authorities.resolve(
                authority_id,
                operator=request.operator,
                allowed_kinds=frozenset({RuntimeAuthorityKind.SELECTION_POLICY}),
                required_effect=EffectClass.PURE,
                input_contract="yf_arc3_v5.authority_input.selection.v1",
                output_contract="yf_arc3_v5.authority_output.selection.v1",
            )
        return None

    def _input_refs(self, request: AnyOperatorRequest) -> tuple[Ref, ...]:
        values: tuple[Ref, ...]
        if isinstance(request, ObserveRequest):
            values = (request.raw_input_ref, request.observation_ref)
        elif isinstance(request, DistinguishRequest):
            values = request.source_refs
        elif isinstance(request, RelateRequest):
            values = (*request.relation.arguments, *request.basis_refs)
        elif isinstance(request, ProposeRequest):
            values = (*request.basis_refs, *request.live_alternatives)
        elif isinstance(request, DeriveRequest):
            values = (*request.premise_refs, *request.context_refs)
        elif isinstance(request, CompareRequest):
            values = (*request.left_refs, *request.right_refs)
        elif isinstance(request, UpdateRequest):
            values = (
                *request.evidence_refs,
                *(item.mutation.after.id for item in request.updates),
            )
        elif isinstance(request, SelectRequest):
            values = (request.goal_ref, *request.alternative_refs)
        else:
            values = (
                request.intent.id,
                request.release_permit.id,
                request.intent.action_ref,
            )
        return tuple(sorted(set(values)))

    def _dispatch(
        self,
        request: AnyOperatorRequest,
        authority: RegisteredAuthority | None,
    ) -> tuple[
        OperatorOutput,
        tuple[CognitiveArtifact, ...],
        tuple[Ref, ...],
        tuple[Ref, ...],
    ]:
        if isinstance(request, ObserveRequest):
            assert authority is not None
            return self._execute_observe(request, authority)
        if isinstance(request, DistinguishRequest):
            assert authority is not None
            return self._execute_distinguish(request, authority)
        if isinstance(request, RelateRequest):
            assert authority is not None
            return self._execute_relate(request, authority)
        if isinstance(request, ProposeRequest):
            return self._execute_propose(request)
        if isinstance(request, DeriveRequest):
            assert authority is not None
            return self._execute_derive(request, authority)
        if isinstance(request, CompareRequest):
            assert authority is not None
            return self._execute_compare(request, authority)
        if isinstance(request, SelectRequest):
            assert authority is not None
            return self._execute_select(request, authority)
        if isinstance(request, ActRequest):
            return self._execute_act(request)
        raise OperatorRuntimeError(
            f"unsupported operator request: {type(request).__name__}"
        )

    def _execute_observe(
        self,
        request: ObserveRequest,
        authority: RegisteredAuthority,
    ) -> tuple[
        OperatorOutput, tuple[CognitiveArtifact, ...], tuple[Ref, ...], tuple[Ref, ...]
    ]:
        acquisition = self._authorities.invoke(
            authority,
            AcquisitionInput(
                raw_input_ref=request.raw_input_ref,
                requested_observation_ref=request.observation_ref,
                frame_id=request.frame_id,
            ),
            expected_type=AcquisitionResult,
        )
        if acquisition.observation_ref != request.observation_ref:
            raise OperatorRuntimeError(
                "acquisition changed the requested observation identity"
            )
        record = ObservationAcquiredRecord(
            id=f"{request.request_id}:observation",
            source_unit=authority.definition.source_unit,
            source_hash=authority.definition.source_hash,
            state_revision=self._store.snapshot.revision,
            workflow_definition_id=request.workflow_definition_id,
            workflow_instance_id=request.workflow_instance_id,
            workflow_step_id=request.workflow_step_id,
            timeline_id=request.timeline_id,
            frame_id=request.frame_id,
            provenance=(request.raw_input_ref,),
            acquisition_model_ref=authority.definition.id,
            raw_input_ref=request.raw_input_ref,
            observation_ref=acquisition.observation_ref,
        )
        refs = (acquisition.observation_ref, *acquisition.component_refs)
        return (
            ObservationOutput(evidence=acquisition),
            (record,),
            refs,
            acquisition.reason_refs,
        )

    def _execute_distinguish(
        self,
        request: DistinguishRequest,
        authority: RegisteredAuthority,
    ) -> tuple[
        OperatorOutput, tuple[CognitiveArtifact, ...], tuple[Ref, ...], tuple[Ref, ...]
    ]:
        result = self._authorities.invoke(
            authority,
            DistinctionInput(source_refs=request.source_refs),
            expected_type=DistinctionResult,
        )
        identities = tuple(item.id for item in result.distinctions)
        if len(identities) != len(set(identities)):
            raise OperatorRuntimeError("DISTINGUISH returned duplicate identities")
        return (
            DistinguishOutput(distinctions=result.distinctions),
            (),
            identities,
            result.reason_refs,
        )

    def _execute_relate(
        self,
        request: RelateRequest,
        authority: RegisteredAuthority,
    ) -> tuple[
        OperatorOutput, tuple[CognitiveArtifact, ...], tuple[Ref, ...], tuple[Ref, ...]
    ]:
        decision = self._authorities.invoke(
            authority,
            PredicateInput(
                predicate_ref=request.predicate_ref,
                arguments=request.relation.arguments,
                basis_refs=request.basis_refs,
            ),
            expected_type=PredicateResult,
        )
        if not decision.holds:
            raise _BlockedExecution(
                reason_refs=decision.reason_refs,
                message="RELATE predicate did not hold",
            )
        claim = _claim_from_seed(
            request.relation,
            epistemic_status=EpistemicStatus.SUPPORTED,
            grounds=request.basis_refs,
            proof_rule=None,
            result_id=f"{request.request_id}:result",
        )
        return RelateOutput(claim=claim), (), (claim.id,), decision.reason_refs

    def _execute_propose(
        self,
        request: ProposeRequest,
    ) -> tuple[
        OperatorOutput, tuple[CognitiveArtifact, ...], tuple[Ref, ...], tuple[Ref, ...]
    ]:
        proposal_ids = {item.id for item in request.proposals}
        if not proposal_ids.issubset(set(request.live_alternatives)):
            raise OperatorRuntimeError(
                "every proposed claim must remain a live alternative"
            )
        claims: list[Claim] = []
        for seed in request.proposals:
            attributes = seed.attributes.to_dict()
            attributes.update(
                {
                    "missing_evidence": request.missing_evidence,
                    "falsifier_refs": request.falsifier_refs,
                }
            )
            claims.append(
                _claim_from_seed(
                    seed,
                    epistemic_status=EpistemicStatus.PROPOSED,
                    grounds=request.basis_refs,
                    proof_rule=None,
                    result_id=f"{request.request_id}:result",
                    attributes=FrozenMap(attributes),
                )
            )
        alternatives = tuple(sorted(request.live_alternatives))
        output = ProposalOutput(
            claims=tuple(claims),
            alternatives=AlternativeSet(
                live_alternatives=alternatives,
                cardinality=(
                    Cardinality.SPECIAL
                    if len(alternatives) > 1
                    else Cardinality.SINGULAR
                ),
            ),
            missing_evidence=request.missing_evidence,
            falsifier_refs=request.falsifier_refs,
        )
        return output, (), tuple(sorted(proposal_ids)), request.basis_refs

    def _execute_derive(
        self,
        request: DeriveRequest,
        authority: RegisteredAuthority,
    ) -> tuple[
        OperatorOutput, tuple[CognitiveArtifact, ...], tuple[Ref, ...], tuple[Ref, ...]
    ]:
        derived = self._authorities.invoke(
            authority,
            DerivationInput(
                premise_refs=request.premise_refs,
                context_refs=request.context_refs,
                requested_conclusions=request.conclusions,
            ),
            expected_type=DerivationResult,
        )
        if derived.conclusions != request.conclusions:
            raise OperatorRuntimeError(
                "DERIVE authority returned conclusions outside the requested contract"
            )
        premise_set = frozenset(request.premise_refs)
        if any(seed.derivation_premise_refs is not None
               and not frozenset(seed.derivation_premise_refs).issubset(premise_set)
               for seed in derived.conclusions):
            raise OperatorRuntimeError("declared conclusion premises are outside the DERIVE contract")
        claims = tuple(
            _claim_from_seed(
                seed,
                epistemic_status=EpistemicStatus.SUPPORTED,
                grounds=(seed.derivation_premise_refs
                         if seed.derivation_premise_refs is not None else request.premise_refs),
                proof_rule=authority.definition.id,
                result_id=f"{request.request_id}:result",
            )
            for seed in derived.conclusions
        )
        return (
            DerivationOutput(
                claims=claims,
                exact_premise_refs=request.premise_refs,
                rule_or_model_ref=authority.definition.id,
            ),
            (),
            tuple(claim.id for claim in claims),
            derived.reason_refs,
        )

    def _execute_compare(
        self,
        request: CompareRequest,
        authority: RegisteredAuthority,
    ) -> tuple[
        OperatorOutput, tuple[CognitiveArtifact, ...], tuple[Ref, ...], tuple[Ref, ...]
    ]:
        compared = self._authorities.invoke(
            authority,
            ComparisonInput(left_refs=request.left_refs, right_refs=request.right_refs),
            expected_type=ComparisonResult,
        )
        return (
            CompareOutput(
                comparison=compared.comparison,
                left_refs=request.left_refs,
                right_refs=request.right_refs,
                reason_refs=compared.reason_refs,
            ),
            (),
            (),
            compared.reason_refs,
        )

    def _execute_select(
        self,
        request: SelectRequest,
        authority: RegisteredAuthority,
    ) -> tuple[
        OperatorOutput, tuple[CognitiveArtifact, ...], tuple[Ref, ...], tuple[Ref, ...]
    ]:
        canonical_alternatives = tuple(sorted(request.alternative_refs))
        decision = self._authorities.invoke(
            authority,
            SelectionInput(
                goal_ref=request.goal_ref,
                alternative_refs=canonical_alternatives,
                requirement_refs=tuple(sorted(request.requirement_refs)),
                preserve_refs=tuple(sorted(request.preserve_refs)),
                dominance_witnesses=tuple(
                    sorted(
                        request.dominance_witnesses,
                        key=lambda item: (
                            item.left_ref,
                            item.right_ref,
                            item.criterion_ref,
                        ),
                    )
                ),
                alternative_facts=request.alternative_facts,
                context_facts=request.context_facts,
            ),
            expected_type=SelectionResult,
        )
        if decision.selected_ref is None:
            raise _BlockedExecution(
                reason_refs=decision.reason_refs,
                message=(
                    "SELECT preserved alternatives without commitment; "
                    f"authority={authority.definition.id!r}; "
                    f"goal={request.goal_ref!r}; "
                    f"alternatives={canonical_alternatives!r}"
                ),
            )
        try:
            output = project_selection_output(canonical_alternatives, decision)
        except ValueError as exc:
            raise OperatorRuntimeError(str(exc)) from exc
        return output, (), (decision.selected_ref,), decision.reason_refs

    def _execute_act(
        self,
        request: ActRequest,
    ) -> tuple[
        OperatorOutput, tuple[CognitiveArtifact, ...], tuple[Ref, ...], tuple[Ref, ...]
    ]:
        intent = request.intent
        permit = request.release_permit
        current = self._store.snapshot
        world_context = request.world_context
        lease = permit.concentration_lease
        if lease is None:
            raise OperatorRuntimeError("ACT has no concentration lease")
        if lease.state not in {
            ConcentrationLeaseState.ACQUIRED,
            ConcentrationLeaseState.VALIDATED,
        }:
            raise OperatorRuntimeError("ACT concentration lease was not acquired")
        canonical_lease = self._store.artifact(lease.id)
        deferred_acquisition = (
            canonical_lease is None and permit.concentration_acquisition_deferred
        )
        if canonical_lease != lease and not deferred_acquisition:
            raise OperatorRuntimeError("ACT concentration lease is not canonical")
        series_index = intent.declared_series_index
        series_action_refs = permit.declared_series_action_refs
        series_authorized = bool(
            intent.declared_series_ref
            and permit.declared_series_ref == intent.declared_series_ref
            and world_context.get("continue_series") is True
            and permit.declared_series_target_workflow_instance_id
            == request.workflow_instance_id
            and series_index is not None
            and 0 <= series_index < len(permit.declared_series_intent_ids)
            and permit.declared_series_intent_ids[series_index] == intent.id
            and len(series_action_refs) == len(permit.declared_series_intent_ids)
            and series_action_refs[series_index] == intent.action_ref
            and tuple(permit.declared_series_intent_ids)
            == tuple(lease.extension_intent_ids)
        )
        expected_focus_state = (
            ConcentrationLeaseState.OBSERVED
            if series_authorized and series_index is not None and series_index > 0
            else ConcentrationLeaseState.VALIDATED
        )
        if deferred_acquisition and (not series_authorized or series_index != 0):
            raise OperatorRuntimeError("ACT deferred concentration requires the first declared action")
        if not deferred_acquisition and self._store.concentration_state(lease.id) is not expected_focus_state:
            raise OperatorRuntimeError("ACT concentration lease lifecycle is not releasable")

        def stop_focus(
            *, reason_ref: Ref, state: ConcentrationLeaseState, message: str,
        ) -> None:
            transition = ConcentrationLeaseTransition(
                id=f"{lease.id}:{state.value}:{request.request_id}",
                source_unit=request.source_unit,
                source_hash=request.source_hash,
                state_revision=current.revision,
                workflow_definition_id=request.workflow_definition_id,
                workflow_instance_id=request.workflow_instance_id,
                workflow_step_id=request.workflow_step_id,
                timeline_id=request.timeline_id,
                frame_id=request.frame_id,
                provenance=(lease.id, reason_ref),
                concentration_lease_id=lease.id,
                from_state=expected_focus_state,
                to_state=state,
                reason_ref=reason_ref,
                action_intent_id=intent.id,
            )
            raise _BlockedExecution(
                (reason_ref,), message, () if deferred_acquisition else (transition,)
            )

        if intent.id not in lease.extension_intent_ids:
            raise OperatorRuntimeError("ACT intent is outside the concentration lease")
        if lease.allowed_action_ref != intent.action_ref and not series_authorized:
            raise OperatorRuntimeError("ACT action is outside the concentration lease")
        if lease.timeline_id != request.timeline_id:
            raise OperatorRuntimeError("ACT concentration lease timeline mismatch")
        if lease.session_ref != self._store.run_id:
            raise OperatorRuntimeError("ACT concentration lease session mismatch")
        if lease.contract_revision != permit.state_revision:
            raise OperatorRuntimeError("ACT concentration lease contract is stale")
        if current.revision > lease.expires_after_state_revision:
            stop_focus(
                reason_ref="focus.revision_window_elapsed",
                state=ConcentrationLeaseState.EXPIRED,
                message="ACT concentration lease expired",
            )
        if not series_authorized or series_index == 0:
            if lease.frame_id != request.frame_id:
                stop_focus(
                    reason_ref="focus.frame_stale",
                    state=ConcentrationLeaseState.INVALIDATED,
                    message="ACT concentration lease frame mismatch",
                )
            if lease.frame_revision != current.revision:
                stop_focus(
                    reason_ref="focus.frame_revision_stale",
                    state=ConcentrationLeaseState.INVALIDATED,
                    message="ACT concentration lease frame revision is stale",
                )
        if (
            series_authorized
            and series_index is not None
            and series_index > 0
            and lease.requires_post_action_full_revalidation
        ):
            raise OperatorRuntimeError(
                "ACT continuation is blocked by required full DRM revalidation"
            )
        critical_interrupt_refs = tuple(
            str(ref) for ref in (world_context.get("critical_interrupt_refs") or ())
        )
        if critical_interrupt_refs:
            stop_focus(
                reason_ref=critical_interrupt_refs[0],
                state=ConcentrationLeaseState.INTERRUPTED,
                message="ACT concentration lease was critically interrupted",
            )
        closed_objective_refs = {
            str(ref) for ref in (world_context.get("closed_objective_refs") or ())
        }
        if (
            lease.active_objective_ref
            and lease.active_objective_ref in closed_objective_refs
        ):
            stop_focus(
                reason_ref=str(lease.active_objective_ref),
                state=ConcentrationLeaseState.INVALIDATED,
                message="ACT concentration objective is no longer open",
            )
        violated_constraint_refs = {
            str(ref) for ref in (world_context.get("violated_constraint_refs") or ())
        }
        violated_focus_constraints = violated_constraint_refs.intersection(
            (*lease.preserve_constraint_refs, *lease.overshoot_constraint_refs)
        )
        if violated_focus_constraints:
            stop_focus(
                reason_ref=sorted(violated_focus_constraints)[0],
                state=ConcentrationLeaseState.INVALIDATED,
                message="ACT concentration constraints are violated",
            )
        for key, expected in (
            ("game_ref", lease.game_ref),
            ("level_ref", lease.level_ref),
            ("session_ref", lease.session_ref),
        ):
            actual = world_context.get(key)
            if actual is None or str(actual) != expected:
                raise OperatorRuntimeError(f"ACT concentration lease {key} mismatch")
        if not series_authorized and (
            intent.state_revision != current.revision
            or permit.state_revision != current.revision
        ):
            raise OperatorRuntimeError("ACT intent or permit is stale")
        if not series_authorized and permit.valid_through_state_revision < current.revision:
            raise OperatorRuntimeError("ACT release permit has expired")
        if not series_authorized and permit.pre_action_snapshot_digest != current.state_hash:
            raise OperatorRuntimeError("ACT release permit snapshot mismatch")
        if not series_authorized and permit.action_intent_id != intent.id:
            raise OperatorRuntimeError("ACT release permit references another intent")
        if permit.commitment_decision_id != intent.commitment_decision_id:
            raise OperatorRuntimeError("ACT commitment lineage mismatch")
        if not series_authorized and permit.available_action_ref != intent.available_action_ref:
            raise OperatorRuntimeError("ACT availability lineage mismatch")
        if (
            intent.timeline_id != request.timeline_id
            or permit.timeline_id != request.timeline_id
        ):
            raise OperatorRuntimeError("ACT timeline lineage mismatch")
        if (
            intent.workflow_instance_id != request.workflow_instance_id
            or permit.workflow_instance_id != request.workflow_instance_id
        ):
            raise OperatorRuntimeError("ACT workflow lineage mismatch")
        output = ActOutput(
            action_ref=intent.action_ref,
            action_intent_id=intent.id,
            action_release_permit_id=permit.id,
        )
        # A pre-released CONTINUE series reuses one immutable series permit
        # across its primitive intents.  The permit is already journaled by
        # the preparation boundary (or by the first ACT); emitting it again
        # would create a duplicate artifact identity on the next trigger.
        permit_already_journaled = isinstance(
            self._store.artifact(permit.id), ActionReleasePermit
        )
        journaled_intent = (
            intent.model_copy(update={"frame_id": request.frame_id})
            if series_authorized
            else intent
        )
        transition = ConcentrationLeaseTransition(
            id=f"{lease.id}:released:{intent.id}",
            source_unit=request.source_unit,
            source_hash=request.source_hash,
            state_revision=current.revision,
            workflow_definition_id=request.workflow_definition_id,
            workflow_instance_id=request.workflow_instance_id,
            workflow_step_id=request.workflow_step_id,
            timeline_id=request.timeline_id,
            frame_id=request.frame_id,
            provenance=(lease.id, permit.id, intent.id),
            concentration_lease_id=lease.id,
            from_state=(
                ConcentrationLeaseState.VALIDATED
                if not series_authorized or series_index == 0
                else ConcentrationLeaseState.OBSERVED
            ),
            to_state=ConcentrationLeaseState.RELEASED,
            reason_ref=permit.id,
            action_intent_id=intent.id,
        )
        governance = project_concentration_operation(
            lease=lease,
            lineage=transition,
            operation=ConcentrationOperation.ENGAGE,
            reason_ref=permit.id,
        )
        acquisition_artifacts: tuple[CognitiveArtifact, ...] = ()
        if deferred_acquisition:
            validated = ConcentrationLeaseTransition(
                id=f"{lease.id}:validated",
                source_unit=lease.source_unit,
                source_hash=lease.source_hash,
                state_revision=lease.state_revision,
                workflow_definition_id=lease.workflow_definition_id,
                workflow_instance_id=lease.workflow_instance_id,
                workflow_step_id=lease.workflow_step_id,
                timeline_id=lease.timeline_id,
                frame_id=lease.frame_id,
                provenance=(lease.id, lease.src_decision_ref),
                concentration_lease_id=lease.id,
                from_state=ConcentrationLeaseState.ACQUIRED,
                to_state=ConcentrationLeaseState.VALIDATED,
                reason_ref=lease.src_decision_ref,
                action_intent_id=lease.action_intent_id,
            )
            acquisition_artifacts = (
                lease,
                validated,
                project_concentration_operation(
                    lease=lease,
                    lineage=validated,
                    operation=ConcentrationOperation.ATTRIBUTE,
                    reason_ref=lease.src_decision_ref,
                ),
            )
        artifacts: tuple[CognitiveArtifact, ...] = acquisition_artifacts + (
            (journaled_intent, transition, governance)
            if permit_already_journaled
            else (journaled_intent, permit, transition, governance)
        )
        return output, artifacts, (intent.action_ref,), (permit.id,)

    def _execute_update(
        self,
        request: UpdateRequest,
        invocation: OperatorInvocation,
        authority: RegisteredAuthority,
        *,
        revision_before: int,
        state_hash_before: Ref,
        prior_event_ids: tuple[Ref, ...],
    ) -> OperatorExecution:
        raw_mutations: tuple[StateMutation, ...] = tuple(
            item.mutation for item in request.updates
        )
        prepared = self._store.prepare_mutations(raw_mutations)
        changes: list[StateChange] = []
        for item, mutation in zip(request.updates, prepared):
            assert isinstance(mutation, (PutTerm, PutClaim))
            changes.append(
                StateChange(
                    target_ref=mutation.after.id,
                    before_digest=(
                        None
                        if mutation.before is None
                        else stable_digest(mutation.before)
                    ),
                    after_digest=stable_digest(mutation.after),
                    update_policy_ref=authority.definition.id,
                    invalidates=item.invalidates,
                    recalculates=item.recalculates,
                )
            )
        proposal = StateChangeProposal(
            id=f"{request.request_id}:proposal",
            source_unit=request.source_unit,
            source_hash=request.source_hash,
            state_revision=self._store.snapshot.revision,
            workflow_definition_id=request.workflow_definition_id,
            workflow_instance_id=request.workflow_instance_id,
            workflow_step_id=request.workflow_step_id,
            timeline_id=request.timeline_id,
            frame_id=request.frame_id,
            provenance=request.evidence_refs,
            source_change_id=request.source_change_id,
            changes=tuple(changes),
            evidence_refs=request.evidence_refs,
        )
        policy = self._authorities.invoke(
            authority,
            UpdatePolicyInput(
                proposal=proposal,
                current_state_hash=self._store.snapshot.state_hash,
                current_state_revision=self._store.snapshot.revision,
            ),
            expected_type=UpdatePolicyResult,
        )
        # The policy consumes this immutable proposal immediately.  Its exact
        # value remains in UpdateOutput for a declared checkpoint, so a second
        # proposal-only event contributes no action evidence.
        event_ids = prior_event_ids
        if not policy.allowed:
            return self._finish_non_completed(
                request,
                invocation,
                status=OperationStatus.BLOCKED,
                reason_refs=policy.reason_refs,
                message="UPDATE policy blocked the proposal",
                revision_before=revision_before,
                state_hash_before=state_hash_before,
                prior_event_ids=event_ids,
            )

        next_revision = self._store.snapshot.revision + 1
        transaction_id = f"{request.request_id}:state"
        affected_refs = tuple(sorted(change.target_ref for change in changes))
        committed = StateChangeCommittedRecord(
            id=f"{request.request_id}:committed",
            source_unit=request.source_unit,
            source_hash=request.source_hash,
            state_revision=next_revision,
            workflow_definition_id=request.workflow_definition_id,
            workflow_instance_id=request.workflow_instance_id,
            workflow_step_id=request.workflow_step_id,
            timeline_id=request.timeline_id,
            frame_id=request.frame_id,
            provenance=(proposal.id,),
            proposal_id=proposal.id,
            commitment_decision_id=request.commitment_decision_id,
            transaction_id=transaction_id,
            state_revision_before=self._store.snapshot.revision,
            state_revision_after=next_revision,
            affected_refs=affected_refs,
        )
        state_receipt = self._store.commit(
            TransactionRequest(
                transaction_id=transaction_id,
                run_id=request.run_id,
                timeline_id=request.timeline_id,
                frame_id=request.frame_id,
                source_hash=request.source_hash,
                expected_revision=self._store.snapshot.revision,
                mutations=prepared,
            )
        )
        event_ids += tuple(event.event_id for event in state_receipt.committed_events)
        invalidated = tuple(
            sorted({ref for change in changes for ref in change.invalidates})
        )
        recalculated = tuple(
            sorted({ref for change in changes for ref in change.recalculates})
        )
        output = UpdateOutput(
            proposal=proposal,
            committed=committed,
            invalidated_refs=invalidated,
            recalculated_refs=recalculated,
        )
        return self._finish_completed(
            request,
            invocation,
            output=output,
            artifacts=(),
            output_refs=affected_refs,
            reason_refs=policy.reason_refs,
            revision_before=revision_before,
            state_hash_before=state_hash_before,
            prior_event_ids=event_ids,
        )

    def _finish_completed(
        self,
        request: AnyOperatorRequest,
        invocation: OperatorInvocation,
        *,
        output: OperatorOutput,
        artifacts: tuple[CognitiveArtifact, ...],
        output_refs: tuple[Ref, ...],
        reason_refs: tuple[Ref, ...],
        revision_before: int,
        state_hash_before: Ref,
        prior_event_ids: tuple[Ref, ...],
    ) -> OperatorExecution:
        pure = self._pure_request(request)
        output_attributes: dict[str, object] = {"output_kind": output.output_kind}
        if pure and not self._diagnostic_outputs:
            output_attributes["output_recovery"] = "recompute_pure"
        else:
            output_attributes["output"] = immutable_json_data(output)
        result = self._make_result(
            request,
            invocation,
            status=OperationStatus.COMPLETED,
            output_refs=output_refs,
            reason_refs=reason_refs,
            attributes=output_attributes,
        )
        if (pure and not isinstance(request, DeriveRequest)
                and not artifacts and not self._persist_pure_operator_audit):
            # The next durable workflow checkpoint contains the pure output.
            # Before that checkpoint, retrying the deterministic operator is
            # sufficient; no claim cites this result as durable proof.
            event_ids = prior_event_ids
        elif isinstance(request, SelectRequest) and self._transient_select_receipts:
            # The exact selected assessment is in the next durable workflow
            # checkpoint. SELECT is pure and creates no canonical CLAIM; an
            # intermediate invocation/result pair is not needed to resume.
            event_ids = prior_event_ids
        else:
            receipt = self._commit_artifacts(
                request,
                transaction_suffix="result",
                artifacts=(*((invocation,) if pure else ()), *artifacts, result),
            )
            event_ids = prior_event_ids + tuple(
                event.event_id for event in receipt.committed_events
            )
        return OperatorExecution(
            invocation=invocation,
            result=result,
            output=output,
            state_revision_before=revision_before,
            state_revision_after=self._store.snapshot.revision,
            state_hash_before=state_hash_before,
            state_hash_after=self._store.snapshot.state_hash,
            committed_event_ids=event_ids,
        )

    def _finish_non_completed(
        self,
        request: AnyOperatorRequest,
        invocation: OperatorInvocation,
        *,
        status: OperationStatus,
        reason_refs: tuple[Ref, ...],
        message: str,
        revision_before: int,
        state_hash_before: Ref,
        prior_event_ids: tuple[Ref, ...],
        artifacts: tuple[CognitiveArtifact, ...] = (),
    ) -> OperatorExecution:
        result = self._make_result(
            request,
            invocation,
            status=status,
            output_refs=(),
            reason_refs=reason_refs,
            attributes={"message": message},
        )
        receipt = self._commit_artifacts(
            request,
            transaction_suffix=f"result-{status.value}",
            artifacts=(*((invocation,) if self._pure_request(request) else ()),
                       *artifacts, result),
        )
        event_ids = prior_event_ids + tuple(
            event.event_id for event in receipt.committed_events
        )
        return OperatorExecution(
            invocation=invocation,
            result=result,
            output=None,
            state_revision_before=revision_before,
            state_revision_after=self._store.snapshot.revision,
            state_hash_before=state_hash_before,
            state_hash_after=self._store.snapshot.state_hash,
            committed_event_ids=event_ids,
        )

    def _make_result(
        self,
        request: AnyOperatorRequest,
        invocation: OperatorInvocation,
        *,
        status: OperationStatus,
        output_refs: tuple[Ref, ...],
        reason_refs: tuple[Ref, ...],
        attributes: dict[str, object],
    ) -> OperatorResult:
        authority = self._authority_ref(request)
        values = dict(attributes)
        values.update(
            {
                "semantic_state_hash_after": self._store.snapshot.state_hash,
                "authority_ref": authority,
            }
        )
        return OperatorResult(
            id=f"{request.request_id}:result",
            source_unit=request.source_unit,
            source_hash=request.source_hash,
            state_revision=self._store.snapshot.revision,
            workflow_definition_id=request.workflow_definition_id,
            workflow_instance_id=request.workflow_instance_id,
            workflow_step_id=request.workflow_step_id,
            timeline_id=request.timeline_id,
            frame_id=request.frame_id,
            provenance=(invocation.id,),
            invocation_id=invocation.id,
            operator=request.operator,
            status=status,
            output_refs=tuple(sorted(set(output_refs))),
            reason_refs=tuple(sorted(set(reason_refs))),
            attributes=FrozenMap(values),
        )

    def _commit_artifacts(
        self,
        request: AnyOperatorRequest,
        *,
        transaction_suffix: str,
        artifacts: tuple[CognitiveArtifact, ...],
    ) -> TransactionReceipt:
        return self._store.commit(
            TransactionRequest(
                transaction_id=f"{request.request_id}:{transaction_suffix}",
                run_id=request.run_id,
                timeline_id=request.timeline_id,
                frame_id=request.frame_id,
                source_hash=request.source_hash,
                expected_revision=self._store.snapshot.revision,
                mutations=tuple(RecordArtifact(artifact=item) for item in artifacts),
            )
        )
