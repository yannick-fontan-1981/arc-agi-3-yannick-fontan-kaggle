"""Source-attested executor for interaction-effect classification.

The compiler accepts only the exact measurement -> two DRM selections ->
declared meaning transactions topology expressed by the SRC workflow.  Runtime
code does not choose an effect, transient interpretation, rule, or update
policy: every such reference is extracted from the compiled source and resolved
through the source-hashed DRM registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from agents.yf_arc3_v5.capabilities.contracts import (
    FrameGrid,
    FrameNormalizationInput,
    FrameNormalizationResult,
    RevisionedCapabilityRequest,
    SceneTransitionMeasurementInput,
    SceneTransitionMeasurements,
)
from agents.yf_arc3_v5.capabilities.registry import CapabilityRegistry
from agents.yf_arc3_v5.drm.meaning import (
    DeclaredAlternativeFacts,
    instantiate_declared_meaning,
    prepare_meaning_updates,
    project_declared_alternatives,
)
from agents.yf_arc3_v5.drm.registry import DrmRegistry
from agents.yf_arc3_v5.drm.runtime import _implementation
from agents.yf_arc3_v5.logos.operations import OperationStatus
from agents.yf_arc3_v5.logos.types import FrozenMap, FrozenModel, OperatorName
from agents.yf_arc3_v5.operators.contracts import (
    DerivationOutput,
    DeriveRequest,
    SelectionInput,
    SelectionResult,
    SelectOutput,
    UpdateOutput,
    UpdateRequest,
)
from agents.yf_arc3_v5.operators.runtime import OperatorRuntime
from agents.yf_arc3_v5.operators.selection_projection import project_selection_output
from agents.yf_arc3_v5.src.ast import ExpressionKind, StatementKind
from agents.yf_arc3_v5.src.compiler import (
    CompiledModule,
    IrExpression,
    IrStatement,
    IrStep,
)


def _literal(expression: IrExpression | None) -> object:
    if expression is None or expression.kind not in {
        ExpressionKind.REFERENCE,
        ExpressionKind.STRING,
        ExpressionKind.INTEGER,
        ExpressionKind.BOOLEAN,
    }:
        raise ValueError("compiled interaction effect expects a source literal")
    return expression.value


def _refs(expression: IrExpression) -> tuple[str, ...]:
    if expression.kind is not ExpressionKind.LIST:
        raise ValueError("compiled interaction effect expects a source list")
    return tuple(str(_literal(item)) for item in expression.items)


def _call_arguments(statement: IrStatement, callee: str) -> dict[str, IrExpression]:
    if (
        statement.kind is not StatementKind.CALL
        or statement.expression is None
        or statement.expression.value != callee
    ):
        raise ValueError(f"compiled interaction effect requires declared call {callee}")
    arguments = {str(item.name): item.value for item in statement.expression.arguments}
    if len(arguments) != len(statement.expression.arguments) or "None" in arguments:
        raise ValueError("compiled interaction effect requires named unique call arguments")
    return arguments


def _next_step(statement: IrStatement, workflow_ref: str) -> str:
    if statement.kind is not StatementKind.NEXT:
        raise ValueError("compiled interaction effect requires declared NEXT")
    return f"{workflow_ref}.{_literal(statement.expression)}"


@dataclass(frozen=True, slots=True)
class CompiledEffectSelection:
    policy_ref: str
    policy_source_hash: str
    goal_ref: str
    requirement_refs: tuple[str, ...]
    assessment_binding: str
    projection_binding: str
    evaluate_drm: Callable[[FrozenModel], FrozenModel]

    def assess(
        self,
        *,
        drm: DrmRegistry,
        alternative_refs: tuple[str, ...],
        descriptive_facts: Mapping[str, object],
    ) -> tuple[SelectOutput, DeclaredAlternativeFacts]:
        projected = project_declared_alternatives(
            drm=drm,
            selection_policy_ref=self.policy_ref,
            alternative_refs=alternative_refs,
            descriptive_facts=descriptive_facts,
        )
        canonical_refs = tuple(sorted(projected.alternative_refs))
        result = self.evaluate_drm(
            SelectionInput(
                goal_ref=self.goal_ref,
                alternative_refs=canonical_refs,
                requirement_refs=tuple(sorted(self.requirement_refs)),
                preserve_refs=canonical_refs,
                alternative_facts=projected.alternative_facts,
            )
        )
        if not isinstance(result, SelectionResult) or result.selected_ref is None:
            raise RuntimeError("declared interaction policy selected no alternative")
        return project_selection_output(canonical_refs, result), projected


@dataclass(frozen=True, slots=True)
class CompiledMeaningCommit:
    step_id: str
    transaction_id: str
    selection_policy_ref: str
    rule_ref: str
    update_policy_ref: str
    evidence_input_refs: tuple[str, ...]
    preserve_input_refs: tuple[str, ...]
    refresh_revision: bool


@dataclass(frozen=True, slots=True)
class CompiledInteractionEffectResult:
    effect: SelectOutput
    transient_evidence: SelectOutput
    measurements: SceneTransitionMeasurements
    path: tuple[str, ...]
    contextual_claim_refs: tuple[str, ...]
    transient_claim_refs: tuple[str, ...]
    drm_source_hashes: tuple[str, ...]
    state_revision_before: int
    state_revision_after: int


@dataclass(frozen=True, slots=True)
class CompiledInteractionEffect:
    workflow_ref: str
    source_unit: str
    src_source_hash: str
    classify_step_id: str
    route_step_id: str
    transient_route_step_id: str
    effect_selection: CompiledEffectSelection
    transient_selection: CompiledEffectSelection
    contextual_commit: CompiledMeaningCommit
    transient_commit: CompiledMeaningCommit
    effect_route_refs: tuple[str, ...]
    transient_commit_refs: tuple[str, ...]
    drm: DrmRegistry

    def run(
        self,
        *,
        before_frame: tuple[tuple[int, ...], ...],
        after_frames: tuple[tuple[tuple[int, ...], ...], ...],
        before_raw_input_ref: str,
        after_raw_input_ref: str,
        action_ref: str,
        action_data: Mapping[str, object],
        candidate_ref: str,
        before_score: int,
        after_score: int,
        transition_ref: str,
        before_configuration_digest: str,
        after_configuration_digest: str,
        before_available_action_set_digest: str,
        boundary_indicator_quantum_delta_measurement: int,
        established_boundary_indicator_decrement_changed_pixel_count_measurement: int,
        declared_boundary_indicator_decrease_changed_pixel_count_measurement: int,
        current_effect_signature_known: bool,
        current_effect_signature_known_cycle: bool,
        known_effect_signature_count: int,
        known_cycle_transition_signature_count: int,
        before_revision: int,
        transitive_source_hash: str,
        workflow_instance_id: str,
        timeline_id: str,
        frame_id: str,
        capabilities: CapabilityRegistry,
        operators: OperatorRuntime,
        context_epoch: int | None = None,
    ) -> CompiledInteractionEffectResult:
        if operators.event_store.snapshot.revision != before_revision:
            raise RuntimeError("compiled interaction effect received a stale revision")
        normalized_before = capabilities.invoke(
            RevisionedCapabilityRequest(
                capability_id="capability.frame_normalization",
                input_revision=before_revision,
                payload=FrameNormalizationInput.from_payload(before_frame),
            )
        ).output
        normalized_after = capabilities.invoke(
            RevisionedCapabilityRequest(
                capability_id="capability.frame_normalization",
                input_revision=before_revision,
                payload=FrameNormalizationInput.from_payload(after_frames),
            )
        ).output
        if not isinstance(normalized_before, FrameNormalizationResult) or not isinstance(
            normalized_after, FrameNormalizationResult
        ):
            raise RuntimeError("compiled interaction frame normalization type changed")
        measurements = capabilities.invoke(
            RevisionedCapabilityRequest(
                capability_id="capability.scene_transition_measurement",
                input_revision=before_revision,
                payload=SceneTransitionMeasurementInput(
                    before=normalized_before.frame,
                    intermediate_frames=normalized_after.transition_frames,
                    after=normalized_after.frame,
                    transition_ref=transition_ref,
                    action_ref=action_ref,
                    action_data=FrozenMap(action_data),
                    candidate_ref=candidate_ref,
                    context_epoch=context_epoch,
                    before_configuration_digest=before_configuration_digest,
                    after_configuration_digest=after_configuration_digest,
                    before_available_action_set_digest=before_available_action_set_digest,
                    boundary_indicator_quantum_delta_measurement=(
                        boundary_indicator_quantum_delta_measurement
                    ),
                    established_boundary_indicator_decrement_changed_pixel_count_measurement=(
                        established_boundary_indicator_decrement_changed_pixel_count_measurement
                    ),
                    declared_boundary_indicator_decrease_changed_pixel_count_measurement=(
                        declared_boundary_indicator_decrease_changed_pixel_count_measurement
                    ),
                    current_effect_signature_known=current_effect_signature_known,
                    current_effect_signature_known_cycle=(
                        current_effect_signature_known_cycle
                    ),
                    known_effect_signature_count=known_effect_signature_count,
                    known_cycle_transition_signature_count=(
                        known_cycle_transition_signature_count
                    ),
                    before_score=before_score,
                    after_score=after_score,
                ),
            )
        ).output
        if not isinstance(measurements, SceneTransitionMeasurements):
            raise RuntimeError("compiled interaction measurement type changed")
        if set(measurements.alternative_refs) != set(self.effect_route_refs):
            raise RuntimeError(
                "compiled interaction measurement alternatives changed"
            )

        effect, effect_projected = self.effect_selection.assess(
            drm=self.drm,
            alternative_refs=measurements.alternative_refs,
            descriptive_facts=measurements.alternative_facts,
        )
        transient, transient_projected = self.transient_selection.assess(
            drm=self.drm,
            alternative_refs=(),
            descriptive_facts=measurements.ordered_packet_facts,
        )
        evidence_by_input = {
            "before_raw_input_ref": before_raw_input_ref,
            "after_raw_input_ref": after_raw_input_ref,
        }
        preserve_by_input = {"transition_ref": transition_ref}
        contextual_claim_refs = self._commit_meaning(
            commit=self.contextual_commit,
            assessment=effect,
            projected=effect_projected,
            evidence_by_input=evidence_by_input,
            preserve_by_input=preserve_by_input,
            transitive_source_hash=transitive_source_hash,
            workflow_instance_id=workflow_instance_id,
            timeline_id=timeline_id,
            frame_id=frame_id,
            operators=operators,
        )
        path = [
            self.classify_step_id,
            self.contextual_commit.step_id,
            self.route_step_id,
            self.transient_route_step_id,
        ]
        transient_claim_refs: tuple[str, ...] = ()
        transient_ref = transient.alternatives.selected or ""
        if transient_ref in self.transient_commit_refs:
            transient_claim_refs = self._commit_meaning(
                commit=self.transient_commit,
                assessment=transient,
                projected=transient_projected,
                evidence_by_input=evidence_by_input,
                preserve_by_input=preserve_by_input,
                transitive_source_hash=transitive_source_hash,
                workflow_instance_id=workflow_instance_id,
                timeline_id=timeline_id,
                frame_id=frame_id,
                operators=operators,
            )
            path.append(self.transient_commit.step_id)
        return CompiledInteractionEffectResult(
            effect=effect,
            transient_evidence=transient,
            measurements=measurements,
            path=tuple(path),
            contextual_claim_refs=contextual_claim_refs,
            transient_claim_refs=transient_claim_refs,
            drm_source_hashes=tuple(
                dict.fromkeys(
                    (
                        self.effect_selection.policy_source_hash,
                        self.transient_selection.policy_source_hash,
                    )
                )
            ),
            state_revision_before=before_revision,
            state_revision_after=operators.event_store.snapshot.revision,
        )

    def _commit_meaning(
        self,
        *,
        commit: CompiledMeaningCommit,
        assessment: SelectOutput,
        projected: DeclaredAlternativeFacts,
        evidence_by_input: Mapping[str, str],
        preserve_by_input: Mapping[str, str],
        transitive_source_hash: str,
        workflow_instance_id: str,
        timeline_id: str,
        frame_id: str,
        operators: OperatorRuntime,
    ) -> tuple[str, ...]:
        selected_ref = assessment.alternatives.selected
        if selected_ref is None:
            raise RuntimeError("compiled meaning has no selected alternative")
        evidence_refs = tuple(evidence_by_input[name] for name in commit.evidence_input_refs)
        meaning = instantiate_declared_meaning(
            drm=self.drm,
            selection_policy_ref=commit.selection_policy_ref,
            selected_ref=selected_ref,
            alternative_facts=projected.alternative_facts,
            evidence_refs=evidence_refs,
            state_revision=operators.event_store.snapshot.revision,
            rule_ref_override=commit.rule_ref,
        )
        request_prefix = f"{workflow_instance_id}:compiled:{commit.transaction_id}"
        derived_execution = operators.execute(
            DeriveRequest(
                request_id=f"{request_prefix}:derive",
                operator=OperatorName.DERIVE,
                run_id=operators.event_store.run_id,
                timeline_id=timeline_id,
                frame_id=frame_id,
                expected_state_revision=operators.event_store.snapshot.revision,
                source_unit=self.source_unit,
                source_hash=transitive_source_hash,
                workflow_definition_id=self.workflow_ref,
                workflow_instance_id=workflow_instance_id,
                workflow_step_id=commit.step_id,
                conclusions=meaning.conclusions,
                premise_refs=meaning.evidence_refs,
                rule_or_model_ref=commit.rule_ref,
            )
        )
        if (
            derived_execution.result.status is not OperationStatus.COMPLETED
            or not isinstance(derived_execution.output, DerivationOutput)
        ):
            raise RuntimeError("compiled interaction DERIVE did not complete")
        prepared = prepare_meaning_updates(
            snapshot=operators.event_store.snapshot,
            terms=meaning.terms,
            claims=derived_execution.output.claims,
        )
        preserved_refs = tuple(preserve_by_input[name] for name in commit.preserve_input_refs)
        update_execution = operators.execute(
            UpdateRequest(
                request_id=f"{request_prefix}:update",
                operator=OperatorName.UPDATE,
                run_id=operators.event_store.run_id,
                timeline_id=timeline_id,
                frame_id=frame_id,
                expected_state_revision=operators.event_store.snapshot.revision,
                source_unit=self.source_unit,
                source_hash=transitive_source_hash,
                workflow_definition_id=self.workflow_ref,
                workflow_instance_id=workflow_instance_id,
                workflow_step_id=commit.step_id,
                source_change_id=f"{request_prefix}:change",
                updates=prepared.update_items,
                evidence_refs=meaning.evidence_refs,
                update_policy_ref=commit.update_policy_ref,
            )
        )
        if (
            update_execution.result.status is not OperationStatus.COMPLETED
            or not isinstance(update_execution.output, UpdateOutput)
        ):
            raise RuntimeError("compiled interaction UPDATE did not complete")
        # UPDATE has no mutation field for PRESERVE; the compiler proves that
        # SRC preserves only existing references and the compact record retains
        # those exact refs as lineage.  Touch them here to fail on missing input.
        if not preserved_refs:
            raise RuntimeError("compiled interaction transaction preserves no lineage")
        return tuple(claim.id for claim in derived_execution.output.claims)


def _compile_selection(
    projection: IrStatement,
    selection: IrStatement,
    *,
    expected_alternatives_ref: str | None,
    expected_facts_ref: str,
    drm: DrmRegistry,
) -> CompiledEffectSelection:
    arguments = _call_arguments(projection, "control.project_declared_alternatives")
    if set(arguments) != {"selection_policy_ref", "alternative_refs", "descriptive_facts"}:
        raise ValueError("compiled interaction projection arguments changed")
    policy_ref = str(_literal(arguments["selection_policy_ref"]))
    alternatives = arguments["alternative_refs"]
    if expected_alternatives_ref is None:
        if alternatives.kind is not ExpressionKind.LIST or alternatives.items:
            raise ValueError("compiled transient projection must use DRM alternatives")
    elif _literal(alternatives) != expected_alternatives_ref:
        raise ValueError("compiled interaction alternatives source changed")
    if _literal(arguments["descriptive_facts"]) != expected_facts_ref:
        raise ValueError("compiled interaction descriptive facts source changed")
    if selection.kind is not StatementKind.SELECT or not selection.target:
        raise ValueError("compiled interaction projection must be followed by SELECT")
    clauses = {item.name: item.value for item in selection.clauses}
    if set(clauses) != {"FACTS", "FOR", "FROM", "PRESERVE", "REQUIRE", "UNDER"}:
        raise ValueError("compiled interaction SELECT clauses changed")
    projection_binding = str(projection.result_binding or "")
    if (
        _literal(clauses["FACTS"]) != f"{projection_binding}.alternative_facts"
        or _literal(clauses["FROM"]) != f"{projection_binding}.alternative_refs"
        or _literal(clauses["PRESERVE"]) != f"{projection_binding}.alternative_refs"
        or _literal(clauses["UNDER"]) != policy_ref
    ):
        raise ValueError("compiled interaction projection and SELECT disagree")
    policy, loaded = drm.resolve(policy_ref)
    if len(policy.declared_alternative_refs) > 3:
        raise ValueError("compiled interaction selection exceeds three alternatives")
    return CompiledEffectSelection(
        policy_ref=policy_ref,
        policy_source_hash=loaded.source_hash,
        goal_ref=str(_literal(clauses["FOR"])),
        requirement_refs=_refs(clauses["REQUIRE"]),
        assessment_binding=str(selection.target),
        projection_binding=projection_binding,
        evaluate_drm=_implementation(
            policy, source_unit=loaded.source_unit, source_hash=loaded.source_hash
        ),
    )


def _compile_commit(
    step: IrStep,
    *,
    selection: CompiledEffectSelection,
    expected_assessment_binding: str,
    expected_next: str | None,
    expected_stop_binding: str | None,
    workflow_ref: str,
) -> CompiledMeaningCommit:
    if len(step.statements) not in {5, 6}:
        raise ValueError("compiled interaction meaning step shape changed")
    instantiate, derive, prepare = step.statements[:3]
    if len(step.statements) == 6:
        revision_call, transaction, exit_statement = step.statements[3:]
        revision_arguments = _call_arguments(
            revision_call, "control.current_state_revision"
        )
        if revision_arguments or not revision_call.result_binding:
            raise ValueError("compiled interaction revision refresh changed")
        expected_before_binding = str(revision_call.result_binding)
        refresh_revision = True
    else:
        transaction, exit_statement = step.statements[3:]
        expected_before_binding = "before_revision"
        refresh_revision = False
    arguments = _call_arguments(instantiate, "control.instantiate_declared_meaning")
    if set(arguments) != {
        "selection_policy_ref", "selected_ref", "alternative_facts", "evidence_refs",
    }:
        raise ValueError("compiled interaction meaning arguments changed")
    if (
        _literal(arguments["selection_policy_ref"]) != selection.policy_ref
        or _literal(arguments["selected_ref"])
        != f"{expected_assessment_binding}.alternatives.selected"
        or _literal(arguments["alternative_facts"])
        != f"{selection.projection_binding}.alternative_facts"
    ):
        raise ValueError("compiled interaction meaning source disagrees with SELECT")
    evidence_input_refs = _refs(arguments["evidence_refs"])
    if derive.kind is not StatementKind.DERIVE or not derive.target:
        raise ValueError("compiled interaction meaning requires DERIVE")
    derive_clauses = {item.name: item.value for item in derive.clauses}
    if set(derive_clauses) != {"CONTEXT", "FROM", "USING"} or (
        _literal(derive_clauses["CONTEXT"]) != instantiate.result_binding
        or _literal(derive_clauses["FROM"])
        != f"{instantiate.result_binding}.evidence_refs"
    ):
        raise ValueError("compiled interaction DERIVE bindings changed")
    rule_ref = str(_literal(derive_clauses["USING"]))
    prepare_arguments = _call_arguments(prepare, "control.prepare_meaning_updates")
    if {
        name: _literal(value) for name, value in sorted(prepare_arguments.items())
    } != {
        "terms": f"{instantiate.result_binding}.terms",
        "claims": f"{derive.target}.claims",
    }:
        raise ValueError("compiled interaction prepared update source changed")
    if transaction.kind is not StatementKind.TRANSACTION or len(transaction.body) != 1:
        raise ValueError("compiled interaction requires one atomic UPDATE")
    update = transaction.body[0]
    if update.kind is not StatementKind.UPDATE:
        raise ValueError("compiled interaction transaction contains no UPDATE")
    update_clauses = {item.name: item.value for item in update.clauses}
    if set(update_clauses) != {
        "BEFORE", "FROM", "INVALIDATE", "PRESERVE", "RECALCULATE", "TARGET", "USING",
    }:
        raise ValueError("compiled interaction UPDATE clauses changed")
    if (
        _literal(update_clauses["TARGET"]) != f"{prepare.result_binding}.update_items"
        or _literal(update_clauses["FROM"])
        != f"{instantiate.result_binding}.evidence_refs"
        or _literal(update_clauses["BEFORE"]) != expected_before_binding
        or _refs(update_clauses["INVALIDATE"])
        or _refs(update_clauses["RECALCULATE"])
    ):
        raise ValueError("compiled interaction UPDATE bindings changed")
    if expected_next is not None:
        if _next_step(exit_statement, workflow_ref) != expected_next:
            raise ValueError("compiled interaction meaning NEXT changed")
    elif (
        exit_statement.kind is not StatementKind.STOP
        or _literal(exit_statement.expression) != expected_stop_binding
    ):
        raise ValueError("compiled interaction meaning STOP changed")
    return CompiledMeaningCommit(
        step_id=step.id,
        transaction_id=str(transaction.target),
        selection_policy_ref=selection.policy_ref,
        rule_ref=rule_ref,
        update_policy_ref=str(_literal(update_clauses["USING"])),
        evidence_input_refs=evidence_input_refs,
        preserve_input_refs=_refs(update_clauses["PRESERVE"]),
        refresh_revision=refresh_revision,
    )


def compile_interaction_effect(
    module: CompiledModule,
    workflow_ref: str,
    drm: DrmRegistry,
) -> CompiledInteractionEffect:
    workflow = next((item for item in module.workflows if item.id == workflow_ref), None)
    if workflow is None or len(workflow.steps) != 5:
        raise ValueError("compiled interaction effect workflow shape changed")
    by_id = {step.id: step for step in workflow.steps}
    classify = by_id[workflow.entry_step_id]
    if len(classify.statements) != 8:
        raise ValueError("compiled interaction classification step shape changed")
    normalize_before, normalize_after, measure = classify.statements[:3]
    before_args = _call_arguments(normalize_before, "capability.frame_normalization")
    after_args = _call_arguments(normalize_after, "capability.frame_normalization")
    measure_args = _call_arguments(measure, "capability.scene_transition_measurement")
    if (
        {name: _literal(value) for name, value in sorted(before_args.items())}
        != {"payload": "before_frame"}
        or {name: _literal(value) for name, value in sorted(after_args.items())}
        != {"payload": "after_frames"}
        or measure.result_binding != "measurements"
        or _literal(measure_args.get("before")) != f"{normalize_before.result_binding}.frame"
        or _literal(measure_args.get("intermediate_frames"))
        != f"{normalize_after.result_binding}.transition_frames"
        or _literal(measure_args.get("after")) != f"{normalize_after.result_binding}.frame"
        or _literal(measure_args.get("context_epoch")) != "context_epoch"
    ):
        raise ValueError("compiled interaction measurement order or bindings changed")
    effect_selection = _compile_selection(
        classify.statements[3],
        classify.statements[4],
        expected_alternatives_ref="measurements.alternative_refs",
        expected_facts_ref="measurements.alternative_facts",
        drm=drm,
    )
    transient_selection = _compile_selection(
        classify.statements[5],
        classify.statements[6],
        expected_alternatives_ref=None,
        expected_facts_ref="measurements.ordered_packet_facts",
        drm=drm,
    )
    contextual_step_id = _next_step(classify.statements[7], workflow_ref)
    contextual_step = by_id.get(contextual_step_id)
    if contextual_step is None:
        raise ValueError("compiled interaction contextual step is absent")
    contextual_next = _next_step(contextual_step.statements[-1], workflow_ref)
    contextual_commit = _compile_commit(
        contextual_step,
        selection=effect_selection,
        expected_assessment_binding=effect_selection.assessment_binding,
        expected_next=contextual_next,
        expected_stop_binding=None,
        workflow_ref=workflow_ref,
    )
    route_step = by_id.get(contextual_next)
    if route_step is None or len(route_step.statements) != 1:
        raise ValueError("compiled interaction effect route is absent")
    route = route_step.statements[0]
    if (
        route.kind is not StatementKind.MATCH
        or _literal(route.expression)
        != f"{effect_selection.assessment_binding}.alternatives.selected"
    ):
        raise ValueError("compiled interaction effect route changed")
    effect_policy, _ = drm.resolve(effect_selection.policy_ref)
    route_targets = {
        str(_literal(label)): _next_step(arm.transition, workflow_ref)
        for arm in route.match_arms
        for label in arm.labels
    }
    effect_route_refs = tuple(sorted(route_targets))
    declared_effect_refs = (
        effect_policy.declared_alternative_refs or effect_route_refs
    )
    if (
        set(route_targets) != set(declared_effect_refs)
        or not 1 <= len(route_targets) <= 3
        or len(set(sorted(route_targets.values()))) != 1
    ):
        raise ValueError("compiled interaction effect route omits a declared alternative")
    transient_route_id = next(iter(sorted(route_targets.values())))
    transient_route_step = by_id.get(transient_route_id)
    if transient_route_step is None or len(transient_route_step.statements) != 1:
        raise ValueError("compiled interaction transient route is absent")
    transient_route = transient_route_step.statements[0]
    if (
        transient_route.kind is not StatementKind.MATCH
        or _literal(transient_route.expression)
        != f"{transient_selection.assessment_binding}.alternatives.selected"
    ):
        raise ValueError("compiled interaction transient route changed")
    transient_policy, _ = drm.resolve(transient_selection.policy_ref)
    transient_targets: dict[str, str | None] = {}
    for arm in transient_route.match_arms:
        if arm.transition.kind is StatementKind.NEXT:
            target = _next_step(arm.transition, workflow_ref)
        elif (
            arm.transition.kind is StatementKind.STOP
            and _literal(arm.transition.expression) == effect_selection.assessment_binding
        ):
            target = None
        else:
            raise ValueError("compiled interaction transient route exit changed")
        transient_targets.update(
            {str(_literal(label)): target for label in arm.labels}
        )
    if (
        set(transient_targets) != set(transient_policy.declared_alternative_refs)
        or sum(
            target is not None
            for target in sorted(
                transient_targets.values(), key=lambda item: "" if item is None else item
            )
        ) != 1
    ):
        raise ValueError("compiled interaction transient route topology changed")
    transient_commit_id = next(
        target
        for target in sorted(
            transient_targets.values(), key=lambda item: "" if item is None else item
        )
        if target is not None
    )
    transient_step = by_id.get(transient_commit_id)
    if transient_step is None:
        raise ValueError("compiled interaction transient meaning step is absent")
    transient_commit = _compile_commit(
        transient_step,
        selection=transient_selection,
        expected_assessment_binding=transient_selection.assessment_binding,
        expected_next=None,
        expected_stop_binding=effect_selection.assessment_binding,
        workflow_ref=workflow_ref,
    )
    reachable = {
        classify.id,
        contextual_step.id,
        route_step.id,
        transient_route_step.id,
        transient_step.id,
    }
    if reachable != set(by_id):
        raise ValueError("compiled interaction effect contains unreachable work")
    return CompiledInteractionEffect(
        workflow_ref=workflow_ref,
        source_unit=module.module_id,
        src_source_hash=module.source_hash,
        classify_step_id=classify.id,
        route_step_id=route_step.id,
        transient_route_step_id=transient_route_step.id,
        effect_selection=effect_selection,
        transient_selection=transient_selection,
        contextual_commit=contextual_commit,
        transient_commit=transient_commit,
        effect_route_refs=effect_route_refs,
        transient_commit_refs=tuple(
            ref
            for ref, target in sorted(transient_targets.items())
            if target is not None
        ),
        drm=drm,
    )
