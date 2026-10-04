"""Deterministic compact SRL and active-state projections from canonical events."""

from __future__ import annotations

from collections.abc import Iterable

from agents.yf_arc3_v5.drm.registry import DrmRegistry, DrmResolutionError
from agents.yf_arc3_v5.logos.operations import (
    ActionIntent,
    ActionObservedRecord,
    ActionReleasePermit,
    AppliedPrincipleRecord,
    CommitmentDecision,
    EnvironmentActionDispatchedRecord,
    ObservationAcquiredRecord,
    OperatorInvocation,
    OperatorResult,
    PropagationReport,
    StateChangeCommittedRecord,
    StateChangeProposal,
    ViabilityAssessment,
    WorkflowInstanceRecord,
    WorkflowLifecycleRecord,
)
from agents.yf_arc3_v5.logos.types import (
    Cardinality,
    Disposition,
    EpistemicStatus,
    FrozenMap,
    TermKind,
)
from agents.yf_arc3_v5.observability.models import (
    ActiveStateProjection,
    ClaimProjection,
    CompactSrlLine,
    DecisionEnvelope,
    DecisionReferenceProjection,
    PrincipleProvenanceProjection,
    SourceProvenanceProjection,
    TermProjection,
    UnresolvedProjection,
    WisdomProjection,
)
from agents.yf_arc3_v5.state import CognitiveEvent, PutClaim, PutTerm, RecordArtifact
from agents.yf_arc3_v5.state.event_store import EventStore
from agents.yf_arc3_v5.state.snapshot import CognitiveSnapshot


class DecisionEnvelopeProjectionError(ValueError):
    """An executed action lacks its required canonical symbolic lineage."""


def project_compact_srl(
    events: Iterable[CognitiveEvent],
    *,
    limit: int = 200,
) -> tuple[CompactSrlLine, ...]:
    """Render only executed meaning/control changes, never raw checkpoint payloads."""

    if limit < 1 or limit > 2000:
        raise ValueError("compact SRL limit must be between 1 and 2000")
    event_tuple = tuple(events)
    intent_components: dict[str, str] = {}
    workflow_components: dict[str, str] = {}
    for event in event_tuple:
        artifact = _artifact(event)
        if not isinstance(artifact, WorkflowInstanceRecord):
            continue
        context = artifact.pending_world_context
        candidate_ref = (
            context.get("selected_candidate_ref") if context is not None else None
        )
        if artifact.pending_action_intent_id and candidate_ref:
            intent_components[artifact.pending_action_intent_id] = str(candidate_ref)
        workflow_candidate = artifact.local_bindings.get("candidate_ref")
        if workflow_candidate is None:
            workflow_candidate = artifact.local_binding_updates.get("candidate_ref")
        if artifact.workflow_instance_id and workflow_candidate:
            workflow_components[artifact.workflow_instance_id] = str(
                workflow_candidate
            )
    lines: list[CompactSrlLine] = []
    seen: set[tuple[object, ...]] = set()
    for event in event_tuple:
        projected = _project_event(
            event,
            intent_components=intent_components,
            workflow_components=workflow_components,
        )
        if projected is not None:
            semantic_key = (
                projected.keyword,
                projected.statement,
                projected.workflow_definition_id,
                (
                    None
                    if projected.keyword in {"CALL", "OBSERVE"}
                    or projected.record_kind == "commitment_decision"
                    or projected.record_kind == "claim"
                    else projected.frame_id
                ),
            )
            if semantic_key in seen:
                continue
            seen.add(semantic_key)
            lines.append(projected)
    return collapse_continue_frames(lines)[-limit:]


def render_compact_srl(lines: Iterable[CompactSrlLine]) -> str:
    return "\n".join(
        f"[{line.evidence_ref}] {line.keyword} {line.statement}" for line in lines
    )


def collapse_continue_frames(
    lines: Iterable[CompactSrlLine],
) -> tuple[CompactSrlLine, ...]:
    """Keep one CONTINUE line when a frame is a lightweight cursor advance.

    TERM/CLAIM and verification events remain in the canonical event store,
    but exposing them beside CONTINUE makes a committed intermediate step look
    like a second full reasoning pass and attaches misleading reflection times
    to a lightweight frame.
    """

    line_tuple = tuple(lines)
    latest_continue_by_frame: dict[str, CompactSrlLine] = {}
    for line in line_tuple:
        if _is_cursor_only_continue_line(line) and line.frame_id:
            latest_continue_by_frame[str(line.frame_id)] = line
    if not latest_continue_by_frame:
        return line_tuple
    return tuple(
        line
        for line in line_tuple
        if not line.frame_id
        or str(line.frame_id) not in latest_continue_by_frame
        or (
            _is_cursor_only_continue_line(line)
            and latest_continue_by_frame[str(line.frame_id)] is line
        )
    )


def _is_cursor_only_continue_line(line: CompactSrlLine) -> bool:
    """Recognize only continuations emitted by a committed cursor workflow.

    An information-probe workflow can carry a ``continue_verified_path``
    expectation while still taking the full metaplan path.  Its visible label
    must not hide the MEANING/EXPECT lines or imply cursor-only latency.
    """

    if line.keyword != "CONTINUE":
        return False
    return line.workflow_definition_id in {
        "arc3.choose_verified_alignment_action.choose_verified_alignment_action",
        "arc3.continue_transferred_quantized_plan.continue_transferred_quantized_plan",
    }


def project_active_state(
    snapshot: CognitiveSnapshot,
    events: Iterable[CognitiveEvent],
    *,
    max_items: int = 50,
    drm: DrmRegistry | None = None,
) -> ActiveStateProjection:
    if max_items < 1 or max_items > 200:
        raise ValueError("active-state max_items must be between 1 and 200")
    event_tuple = tuple(events)
    artifacts = tuple(_artifact(event) for event in event_tuple)
    artifacts = tuple(item for item in artifacts if item is not None)
    current_claims = snapshot.current_claims
    terms = tuple(
        TermProjection(
            id=item.id,
            kind=item.kind.value,
            label=item.label,
            provenance=item.provenance,
            last_changed_state_revision=item.last_changed_state_revision,
        )
        for item in snapshot.terms[:max_items]
    )
    claims = tuple(
        ClaimProjection(
            id=item.id,
            predicate=item.predicate,
            arguments=item.arguments,
            epistemic_status=item.epistemic_status.value,
            disposition=item.disposition.value,
            cardinality=item.cardinality.value,
            grounds=item.grounds,
            contradictions=item.active_contradictions,
            defeated_by=item.defeated_by,
            proof_rule=item.proof_rule,
        )
        for item in current_claims[:max_items]
    )
    latest_wisdom = next(
        (item for item in reversed(artifacts) if isinstance(item, CommitmentDecision)),
        None,
    )
    viability = (
        None
        if latest_wisdom is None
        else next(
            (
                item
                for item in reversed(artifacts)
                if isinstance(item, ViabilityAssessment)
                and item.id == latest_wisdom.viability_assessment_id
            ),
            None,
        )
    )
    wisdom = (
        None
        if latest_wisdom is None
        else WisdomProjection(
            decision_id=latest_wisdom.id,
            decision=latest_wisdom.decision.value,
            policy_ref=latest_wisdom.policy_ref,
            viability_assessment_id=latest_wisdom.viability_assessment_id,
            viability_status=(viability.status.value if viability else None),
            reasons=latest_wisdom.reason_refs,
            alternatives_preserved=latest_wisdom.alternatives_preserved,
            missing_evidence=(viability.missing_evidence if viability else ()),
            contradictions=(viability.principles_violated if viability else ()),
        )
    )
    alternatives = set(latest_wisdom.alternatives_preserved if latest_wisdom else ())
    alternatives.update(
        item.id
        for item in current_claims
        if item.cardinality is Cardinality.SPECIAL
        or item.epistemic_status is EpistemicStatus.PROPOSED
    )
    unresolved: list[UnresolvedProjection] = []
    contradictions: set[str] = set()
    for claim in current_claims:
        contradictions.update(claim.active_contradictions)
        if claim.epistemic_status is EpistemicStatus.PROPOSED:
            unresolved.append(
                UnresolvedProjection(
                    id=f"unresolved:{claim.id}",
                    statement=f"claim {claim.id} is not established",
                    reason="claim_not_established",
                    missing_evidence=claim.conditions,
                    alternatives=(
                        (claim.id,) if claim.cardinality is Cardinality.SPECIAL else ()
                    ),
                    falsifiers=claim.active_contradictions,
                )
            )
        elif claim.disposition is Disposition.DEFERRED:
            unresolved.append(
                UnresolvedProjection(
                    id=f"unresolved:{claim.id}",
                    statement=f"claim {claim.id} is deferred",
                    reason="claim_deferred",
                    missing_evidence=claim.conditions,
                    falsifiers=claim.active_contradictions,
                )
            )
    if viability is not None:
        contradictions.update(viability.principles_violated)
        if viability.missing_evidence or viability.unresolved_conditions:
            unresolved.append(
                UnresolvedProjection(
                    id=f"unresolved:{viability.id}",
                    statement=f"viability {viability.id} requires more evidence",
                    reason="viability_unresolved",
                    missing_evidence=(
                        *viability.missing_evidence,
                        *viability.unresolved_conditions,
                    ),
                    alternatives=viability.alternatives_preserved,
                    falsifiers=viability.principles_violated,
                )
            )
    workflow_states = tuple(
        item for item in artifacts if isinstance(item, WorkflowInstanceRecord)
    )
    decision_envelope = project_decision_envelope(
        snapshot,
        event_tuple,
        drm=drm,
    )
    return ActiveStateProjection(
        run_id=snapshot.run_id,
        state_revision=snapshot.revision,
        prediction_revision=snapshot.prediction_revision,
        state_hash=snapshot.state_hash,
        source_event_count=len(event_tuple),
        terms=terms,
        claims=claims,
        goals=tuple(item.id for item in snapshot.terms if item.kind is TermKind.GOAL)[
            :max_items
        ],
        valid_alternatives=tuple(sorted(alternatives))[:max_items],
        wisdom=wisdom,
        decision_envelope=decision_envelope,
        unresolved_questions=tuple(unresolved[:max_items]),
        contradictions=tuple(sorted(contradictions))[:max_items],
        counts=FrozenMap(
            {
                "terms_total": len(snapshot.terms),
                "claims_total": len(current_claims),
                "goals_total": sum(
                    item.kind is TermKind.GOAL for item in snapshot.terms
                ),
                "unresolved_total": len(unresolved),
                "workflow_checkpoint_total": len(workflow_states),
                "projected_term_count": len(terms),
                "projected_claim_count": len(claims),
            }
        ),
        bounded=(
            len(snapshot.terms) > max_items
            or len(current_claims) > max_items
            or len(unresolved) > max_items
        ),
    )


def project_decision_envelope(
    snapshot: CognitiveSnapshot,
    events: Iterable[CognitiveEvent] | EventStore,
    *,
    drm: DrmRegistry | None = None,
) -> DecisionEnvelope | None:
    """Project the latest released action from canonical TERM/CLAIM and records."""

    if isinstance(events, EventStore):
        # These five already-accepted canonical records are the complete
        # lineage read by this projection. No historical event is inspected.
        permit = events.latest_artifact(ActionReleasePermit)
        if permit is None:
            return None
        intent = events.artifact(permit.action_intent_id)
        decision = events.artifact(permit.commitment_decision_id)
        viability = events.artifact(permit.viability_assessment_id)
        workflow = (
            events.latest_workflow_instance(intent.workflow_instance_id)
            if isinstance(intent, ActionIntent)
            else None
        )
        if not isinstance(intent, ActionIntent):
            intent = None
        if not isinstance(decision, CommitmentDecision):
            decision = None
        if not isinstance(viability, ViabilityAssessment):
            viability = None
    else:
        artifacts = tuple(
            artifact
            for artifact in (_artifact(event) for event in events)
            if artifact is not None
        )
        permit = next(
            (item for item in reversed(artifacts) if isinstance(item, ActionReleasePermit)),
            None,
        )
        if permit is None:
            return None
        intent = next(
            (
                item
                for item in reversed(artifacts)
                if isinstance(item, ActionIntent) and item.id == permit.action_intent_id
            ),
            None,
        )
        decision = next(
            (
                item
                for item in reversed(artifacts)
                if isinstance(item, CommitmentDecision)
                and item.id == permit.commitment_decision_id
            ),
            None,
        )
        viability = next(
            (
                item
                for item in reversed(artifacts)
                if isinstance(item, ViabilityAssessment)
                and item.id == permit.viability_assessment_id
            ),
            None,
        )
        workflow = (
            next(
                (
                    item
                    for item in reversed(artifacts)
                    if isinstance(item, WorkflowInstanceRecord)
                    and item.workflow_instance_id == intent.workflow_instance_id
                ),
                None,
            )
            if intent is not None
            else None
        )
    if intent is None:
        raise DecisionEnvelopeProjectionError(
            f"release permit {permit.id} has no canonical action intent"
        )
    if decision is None or viability is None or workflow is None:
        raise DecisionEnvelopeProjectionError(
            f"action intent {intent.id} lacks decision, viability, or workflow lineage"
        )

    required = {
        "terminal_goal_ref": (
            intent.active_terminal_goal_ref or intent.declared_terminal_goal_ref
        ),
        "active_goal_ref": intent.active_goal_ref,
        "instrumental_subgoal_ref": intent.subgoal_ref,
        "priority_or_dominance_ref": (
            intent.priority_class_ref or intent.selection_principle_ref
        ),
        "committed_plan_or_experiment_ref": (
            intent.committed_plan_ref or intent.discriminating_experiment_ref
        ),
        "expected_effect_class": intent.expected_effect_class,
        "expected_configuration_or_goal_delta_ref": (
            intent.expected_configuration_or_goal_delta_ref
        ),
        "reconciliation_checkpoint_ref": intent.reconciliation_checkpoint_ref,
    }
    missing = sorted(name for name, value in sorted(required.items()) if not value)
    dependency_chain_refs = (
        intent.goal_dependency_chain_refs or intent.declared_dependency_chain_refs
    )
    if not dependency_chain_refs:
        missing.append("dependency_chain_refs")
    if not intent.premise_claim_refs:
        missing.append("premise_claim_refs")
    if not intent.falsifier_refs:
        missing.append("explicit_falsifier_refs")
    if intent.remaining_action_bound is None:
        missing.append("remaining_bound")
    if missing:
        raise DecisionEnvelopeProjectionError(
            f"released action {intent.id} lacks envelope fields: {sorted(set(missing))}"
        )

    claim_index = {claim.id: claim for claim in snapshot.current_claims}
    term_index = snapshot.term_index
    element_bindings = FrozenMap(
        {
            claim_ref: FrozenMap(
                {
                    "claim_ref": claim_ref,
                    "predicate_ref": (
                        claim_index[claim_ref].predicate
                        if claim_ref in claim_index
                        else "unresolved_claim"
                    ),
                    "bindings": tuple(
                        FrozenMap(
                            {
                                "entity_ref": entity_ref,
                                "role_ref": _canonical_binding_role(
                                    entity_ref,
                                    argument_index,
                                    claim_index[claim_ref].predicate,
                                    term_index,
                                ),
                            }
                        )
                        for argument_index, entity_ref in enumerate(
                            claim_index[claim_ref].arguments
                            if claim_ref in claim_index
                            else ()
                        )
                    ),
                }
            )
            for claim_ref in intent.premise_claim_refs
        }
    )
    reason_links = _decision_reason_links(intent, viability)
    principle_provenance = _principle_provenance(reason_links, drm)
    source_provenance = tuple(
        _source_provenance(item)
        for item in (intent, permit, decision, viability, workflow)
    )
    terminal_goal_ref = str(required["terminal_goal_ref"])
    active_goal_ref = str(required["active_goal_ref"])
    subgoal_ref = str(required["instrumental_subgoal_ref"])
    plan_ref = str(required["committed_plan_or_experiment_ref"])
    trace_line = (
        f"{terminal_goal_ref} -> {active_goal_ref} -> {subgoal_ref} -> "
        f"{plan_ref} -> {intent.action_ref} -> {intent.expected_effect_class}/"
        f"{','.join(intent.falsifier_refs)} -> {intent.reconciliation_checkpoint_ref}"
    )
    return DecisionEnvelope(
        terminal_goal_ref=terminal_goal_ref,
        active_goal_ref=active_goal_ref,
        instrumental_subgoal_ref=subgoal_ref,
        dependency_chain_refs=dependency_chain_refs,
        priority_or_dominance_ref=str(required["priority_or_dominance_ref"]),
        committed_plan_or_experiment_ref=plan_ref,
        primitive_action_ref=intent.action_ref,
        available_action_ref=intent.available_action_ref,
        expected_effect_class=str(required["expected_effect_class"]),
        expected_configuration_or_goal_delta_ref=str(
            required["expected_configuration_or_goal_delta_ref"]
        ),
        explicit_falsifier_refs=intent.falsifier_refs,
        remaining_bound=int(intent.remaining_action_bound),
        reconciliation_checkpoint_ref=str(required["reconciliation_checkpoint_ref"]),
        workflow_ref=intent.workflow_definition_id,
        workflow_instance_ref=intent.workflow_instance_id,
        decision_ref=decision.id,
        action_intent_ref=intent.id,
        release_permit_ref=permit.id,
        premise_claim_refs=intent.premise_claim_refs,
        element_bindings=element_bindings,
        source_provenance=source_provenance,
        principle_provenance=principle_provenance,
        reason_links=reason_links,
        trace_line=trace_line,
    )


def _canonical_binding_role(
    entity_ref: str,
    argument_index: int,
    predicate_ref: str,
    term_index: dict[str, object],
) -> str:
    term = term_index.get(entity_ref)
    attributes = getattr(term, "attributes", FrozenMap())
    declared = attributes.get("role_ref") or attributes.get("role")
    if declared:
        return str(declared)
    return f"{predicate_ref}.argument_{argument_index}"


def _source_provenance(record: object) -> SourceProvenanceProjection:
    return SourceProvenanceProjection(
        record_ref=str(getattr(record, "id")),
        record_kind=str(getattr(record, "record_kind")),
        source_unit=str(getattr(record, "source_unit")),
        source_hash=str(getattr(record, "source_hash")),
        workflow_definition_id=getattr(record, "workflow_definition_id", None),
        workflow_instance_id=getattr(record, "workflow_instance_id", None),
        workflow_step_id=getattr(record, "workflow_step_id", None),
    )


def _decision_reason_links(
    intent: ActionIntent,
    viability: ViabilityAssessment,
) -> tuple[DecisionReferenceProjection, ...]:
    values: list[DecisionReferenceProjection] = []

    def add(reference: str | None, relation: str, source_kind: str) -> None:
        if reference:
            values.append(
                DecisionReferenceProjection(
                    reference_ref=reference,
                    relation_kind=relation,
                    source_kind=source_kind,
                )
            )

    add(intent.priority_class_ref, "governing", "priority")
    add(intent.selection_principle_ref, "selection_reason", "principle")
    for ref in intent.premise_claim_refs:
        add(ref, "premise", "claim")
    add(
        intent.expected_configuration_or_goal_delta_ref,
        "invariant",
        "expectation",
    )
    for ref in viability.alternatives_preserved:
        add(ref, "preserved", "alternative")
    for ref in viability.principles_preserved:
        add(ref, "preserved", "principle")
    for ref in viability.principles_violated:
        add(ref, "violated", "principle")
    for ref in intent.falsifier_refs:
        add(ref, "falsifier", "falsifier")
    return tuple(values)


def _principle_provenance(
    links: tuple[DecisionReferenceProjection, ...],
    drm: DrmRegistry | None,
) -> tuple[PrincipleProvenanceProjection, ...]:
    projected: list[PrincipleProvenanceProjection] = []
    for link in links:
        if not link.reference_ref.startswith("principle."):
            continue
        source_unit: str | None = None
        source_hash: str | None = None
        status = "unresolved_activation"
        if drm is not None:
            try:
                _declaration, loaded = drm.resolve(link.reference_ref)
            except DrmResolutionError:
                pass
            else:
                source_unit = loaded.source_unit
                source_hash = loaded.source_hash
                status = "confirmed"
        projected.append(
            PrincipleProvenanceProjection(
                principle_ref=link.reference_ref,
                relation_kind=link.relation_kind,
                declaration_source_unit=source_unit,
                source_hash=source_hash,
                activation_status=status,
            )
        )
    return tuple(projected)


def _project_event(
    event: CognitiveEvent,
    *,
    intent_components: dict[str, str],
    workflow_components: dict[str, str],
) -> CompactSrlLine | None:
    mutation = event.mutation
    if isinstance(mutation, PutTerm):
        term = mutation.after
        if mutation.before is not None and not _visible_term_revision(
            mutation.before.attributes,
            term.attributes,
        ):
            return None
        return _line(
            event,
            keyword="MEANING" if mutation.before is None else "UPDATE",
            statement=f"term {term.id}: {term.kind.value} {term.label}",
            record_ref=term.id,
            record_kind="term",
            supporting_refs=term.provenance,
        )
    if isinstance(mutation, PutClaim):
        claim = mutation.after
        arguments = ",".join(
            item
            for item in claim.arguments
            if item.startswith(("entity.", "goal.", "event.", "role.", "schema."))
        )
        return _line(
            event,
            keyword="MEANING" if mutation.before is None else "UPDATE",
            statement=(
                f"{claim.predicate}{f'({arguments})' if arguments else ''} "
                f"{claim.epistemic_status.value}/{claim.disposition.value}"
            ),
            record_ref=claim.id,
            record_kind="claim",
            supporting_refs=(
                *claim.grounds,
                *claim.active_contradictions,
                *claim.source_operator_results,
            ),
        )
    artifact = _artifact(event)
    if artifact is None or isinstance(artifact, WorkflowInstanceRecord):
        return None
    if isinstance(artifact, WorkflowLifecycleRecord):
        if artifact.transition.value != "started" or any(
            internal_name in artifact.workflow_definition_id
            for internal_name in ("operational_wisdom", "release_action")
        ):
            return None
        display_name = _workflow_display_name(artifact.workflow_definition_id)
        instance_id = str(getattr(artifact, "workflow_instance_id", "") or "")
        if display_name == "continue_transferred_quantized_plan":
            keyword = "CONTINUE"
            statement = "exact quantized click path"
        elif display_name == "construct_transferred_quantized_plan":
            keyword = "SIMULATE"
            statement = "current transformed scene routes"
        elif display_name == "reconcile_transferred_quantized_plan":
            keyword = "COMPARE"
            statement = "expected and observed click-path state"
        elif display_name == "choose_verified_alignment_action":
            # Path detail is projected from ActionIntent as SIMULATE/CONTINUE.
            # Suppress the generic workflow start line to avoid SRL duplication.
            return None
        else:
            keyword = "CALL"
            statement = f"workflow {display_name}"
        return _artifact_line(event, artifact, keyword, statement, (artifact.cause_ref,))
    if isinstance(artifact, OperatorInvocation):
        return None
    if isinstance(artifact, OperatorResult):
        if artifact.operator.value == "UPDATE":
            # TERM/CLAIM mutations are the readable delta; the operator result
            # repeats the same identifiers plus an internal workflow address.
            return None
        keyword = {
            "UPDATE": "UPDATE",
            "SELECT": "DEDUCTION",
        }.get(artifact.operator.value)
        if keyword is None:
            return None
        component_ref = workflow_components.get(artifact.workflow_instance_id or "")
        selected_output = next(
            (
                output
                for output in artifact.output_refs
                if output.startswith("scene_equivalence.")
            ),
            None,
        )
        if (
            artifact.operator.value == "SELECT"
            and selected_output is None
            and any(output.startswith("probe:") for output in artifact.output_refs)
        ):
            # The following EXPECT/ACTION lines already name the selected
            # component.  Repeating the low-level SELECT completion adds no
            # readable information to the visible reasoning trace.
            return None
        if artifact.operator.value == "SELECT" and any(
            output.endswith(".unresolved") or ".unresolved_" in output
            for output in artifact.output_refs
        ):
            return None
        if selected_output and component_ref:
            keyword = "MEANING"
            statement = (
                f"component {component_ref} produced no scene effect; unresolved"
                if selected_output == "scene_equivalence.equivalent"
                else f"component {component_ref} produced a scene effect"
            )
        elif artifact.operator.value == "SELECT":
            # The selected alternative is expressed by the following visible
            # TERM/CLAIM delta.  A generic "SELECT completed" line exposes
            # method plumbing rather than additional symbolic meaning.
            return None
        else:
            statement = (
                f"{artifact.operator.value} {artifact.status.value}: "
                f"{','.join(artifact.output_refs) or 'no_output'}"
            )
        return _artifact_line(
            event,
            artifact,
            keyword,
            statement,
            (artifact.invocation_id, *artifact.reason_refs, *artifact.output_refs),
        )
    if isinstance(artifact, ViabilityAssessment):
        return None
    if isinstance(artifact, CommitmentDecision):
        return _artifact_line(
            event,
            artifact,
            "MEANING",
            (
                "wisdom permits controlled action"
                if artifact.decision.value == "commit"
                else f"wisdom {artifact.decision.value}: {artifact.change_ref}"
            ),
            (
                artifact.viability_assessment_id,
                artifact.policy_ref,
                *artifact.reason_refs,
            ),
        )
    if isinstance(artifact, ActionIntent):
        component_ref = intent_components.get(artifact.id)
        simulate_ref = next(
            (
                ref
                for ref in artifact.expectation_refs
                if str(ref).startswith("measurement.simulate_verified_path.")
            ),
            None,
        )
        continue_ref = next(
            (
                ref
                for ref in artifact.expectation_refs
                if str(ref).startswith("measurement.continue_verified_path.")
            ),
            None,
        )
        if simulate_ref is not None:
            # Visible simulated multi-step path (Family C commit).
            path = str(simulate_ref).removeprefix(
                "measurement.simulate_verified_path."
            )
            return _artifact_line(
                event,
                artifact,
                "SIMULATE",
                f"verified path {path} ({artifact.action_ref})",
                (artifact.commitment_decision_id, *artifact.expectation_refs),
            )
        if continue_ref is not None:
            # Visible cursor advance without resimulation.
            path = str(continue_ref).removeprefix(
                "measurement.continue_verified_path."
            )
            return _artifact_line(
                event,
                artifact,
                "CONTINUE",
                f"verified path {path}",
                (artifact.commitment_decision_id, *artifact.expectation_refs),
            )
        return _artifact_line(
            event,
            artifact,
            "EXPECT",
            (
                f"probe component {component_ref} with {artifact.action_ref}"
                if component_ref
                else f"action intent {artifact.action_ref}"
            ),
            (artifact.commitment_decision_id, *artifact.expectation_refs),
        )
    if isinstance(artifact, ActionReleasePermit):
        return None
    if isinstance(artifact, EnvironmentActionDispatchedRecord):
        component_ref = intent_components.get(artifact.action_intent_id)
        return _artifact_line(
            event,
            artifact,
            "ACTION",
            (
                f"probe component {component_ref} dispatched"
                if component_ref
                else f"{artifact.action_ref} dispatched"
            ),
            (
                artifact.action_intent_id,
                artifact.action_release_permit_id,
                artifact.request_digest,
            ),
        )
    if isinstance(artifact, ObservationAcquiredRecord):
        if not str(artifact.observation_ref).startswith("observation:"):
            return None
        return _artifact_line(
            event,
            artifact,
            "OBSERVE",
            "returned frame",
            (artifact.raw_input_ref,),
        )
    if isinstance(artifact, ActionObservedRecord):
        return None
    if isinstance(artifact, PropagationReport):
        return _artifact_line(
            event,
            artifact,
            "DEDUCTION",
            (
                f"propagation {artifact.source_change_id}: "
                f"{'fixpoint' if artifact.fixpoint_reached else 'contradiction'}"
            ),
            (
                *artifact.affected_refs,
                *artifact.invalidated_refs,
                *artifact.recomputed_refs,
            ),
        )
    if isinstance(artifact, StateChangeProposal):
        return None
    if isinstance(artifact, StateChangeCommittedRecord):
        return None
    if isinstance(artifact, AppliedPrincipleRecord):
        return _artifact_line(
            event,
            artifact,
            "MEANING",
            f"applied principle {artifact.id}: {artifact.status.value}",
            (*artifact.source_claims, *artifact.supporting_evidence),
        )
    return None


def _artifact(event: CognitiveEvent) -> object | None:
    return (
        event.mutation.artifact if isinstance(event.mutation, RecordArtifact) else None
    )


def _visible_term_revision(before: FrozenMap, after: FrozenMap) -> bool:
    """Project only revisions that change semantic status or role."""

    visible_fields = (
        "role",
        "status",
        "operational_status",
        "disable_reason",
        "quantity_semantics",
    )
    return any(before.get(field) != after.get(field) for field in visible_fields)


def _workflow_display_name(workflow_definition_id: str) -> str:
    """Return the declarative workflow name without repeating its module path."""

    return workflow_definition_id.rsplit(".", 1)[-1]


def _artifact_line(
    event: CognitiveEvent,
    artifact: object,
    keyword: str,
    statement: str,
    supporting_refs: tuple[str, ...],
) -> CompactSrlLine:
    return _line(
        event,
        keyword=keyword,
        statement=statement,
        record_ref=str(getattr(artifact, "id")),
        record_kind=str(getattr(artifact, "record_kind")),
        supporting_refs=tuple(dict.fromkeys(supporting_refs)),
        workflow_definition_id=getattr(artifact, "workflow_definition_id", None),
        workflow_instance_id=getattr(artifact, "workflow_instance_id", None),
        workflow_step_id=getattr(artifact, "workflow_step_id", None),
        frame_id=getattr(artifact, "frame_id", None),
        state_revision=int(getattr(artifact, "state_revision")),
        source_hash=str(getattr(artifact, "source_hash")),
    )


def _line(
    event: CognitiveEvent,
    *,
    keyword: str,
    statement: str,
    record_ref: str,
    record_kind: str,
    supporting_refs: tuple[str, ...],
    workflow_definition_id: str | None = None,
    workflow_instance_id: str | None = None,
    workflow_step_id: str | None = None,
    frame_id: str | None = None,
    state_revision: int | None = None,
    source_hash: str | None = None,
) -> CompactSrlLine:
    return CompactSrlLine(
        evidence_ref=f"#{event.sequence:08d}",
        event_sequence=event.sequence,
        keyword=keyword,
        statement=statement,
        record_ref=record_ref,
        record_kind=record_kind,
        workflow_definition_id=workflow_definition_id,
        workflow_instance_id=workflow_instance_id,
        workflow_step_id=workflow_step_id,
        frame_id=frame_id or event.frame_id,
        state_revision=(
            event.state_revision_after if state_revision is None else state_revision
        ),
        source_hash=source_hash or event.source_hash,
        supporting_refs=tuple(dict.fromkeys(item for item in supporting_refs if item)),
    )
