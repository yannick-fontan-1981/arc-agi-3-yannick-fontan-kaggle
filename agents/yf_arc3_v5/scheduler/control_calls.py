"""Source-executed wisdom and fail-closed action-release authorities."""

from __future__ import annotations

from collections import ChainMap
from collections.abc import Mapping, Sequence

from agents.yf_arc3_v5.capabilities import (
    ActionGroundingContract,
    InteractionProbeAgenda,
    VerifiedMechanismEvidence,
)
from agents.yf_arc3_v5.capabilities.interaction import POINT_ACTION_REF
from agents.yf_arc3_v5.capabilities.terminal_contact_geometry import (
    measure_exact_observed_fatal_replay,
)
from agents.yf_arc3_v5.capabilities.marker_enclosure import measure_marker_enclosures, measure_marker_ownership
from agents.yf_arc3_v5.capabilities.appearance import (
    AppearanceKeyContract, BearerAppearanceInput,
)
from agents.yf_arc3_v5.capabilities.bearer_support import (
    BearerSupportContract, CurrentBearerSupportRequest, measure_current_bearer_supports,
    measure_proper_bbox_enclosing_members,
    measure_translated_enclosed_member_refs,
)
from agents.yf_arc3_v5.capabilities.bearer_relocation import (
    BearerRelocationContract, relocate_exact_bearer_templates,
)
from agents.yf_arc3_v5.capabilities.contracts import RevisionedCapabilityRequest, TemporalTrackingResult, VisualSceneDescription
from agents.yf_arc3_v5.capabilities.registry import CapabilityRegistry
from agents.yf_arc3_v5.state.canonical_appearance_index import CanonicalAppearanceIndex
from agents.yf_arc3_v5.state.reference_bindings import canonical_reference_bindings_current
from agents.yf_arc3_v5.state.game_method_coverage import measure_current_method_coverage
from agents.yf_arc3_v5.state.game_method_inventory import measure_current_method_inventory
from agents.yf_arc3_v5.drm import DrmRegistry
from agents.yf_arc3_v5.dynamic_workflow.contracts import GameMethodActivityContract
from agents.yf_arc3_v5.drm.contracts import DrmDeclaration
from agents.yf_arc3_v5.drm.runtime import declared_lexicographic_order
from agents.yf_arc3_v5.drm.meaning import (
    NoApplicableDeclaredMeaning,
    PreparedCommittedDerivation,
    PreparedCommittedProposal,
    PreparedMeaningUpdates,
    project_declared_alternative_deltas,
    instantiate_declared_meaning,
    instantiate_dynamic_goal_frontier_activation,
    project_declared_alternatives,
    project_declared_delta,
    prepare_revisable_meaning_updates as prepare_declared_revisable_meaning_updates,
)
from agents.yf_arc3_v5.drm.meaning import (
    prepare_meaning_updates as prepare_declared_meaning_updates,
)
from agents.yf_arc3_v5.logos.claims import Claim
from agents.yf_arc3_v5.logos.operations import (
    ActionIntent,
    ActionObservedRecord,
    ActionReleasePermit,
    CommitmentDecision,
    CommitmentKind,
    ConcentrationLease,
    ConcentrationLeaseState,
    ConcentrationLeaseTransition,
    ConcentrationOperation,
    project_concentration_operation,
    Reversibility,
    ViabilityAssessment,
    ViabilityStatus,
    WorkflowInstanceRecord,
    WorkflowStatus,
)
from agents.yf_arc3_v5.logos.terms import Term
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest
from agents.yf_arc3_v5.operators.contracts import (
    DerivationOutput,
    ProposalOutput,
    SelectOutput,
    SelectionInput,
    UpdateItem,
)
from agents.yf_arc3_v5.performance import profile_span
from agents.yf_arc3_v5.runtime.contracts import PreparedInteractionProbe
from agents.yf_arc3_v5.scheduler.contracts import FunctionCallContext
from agents.yf_arc3_v5.scheduler.evaluator import RuntimeModelMap, runtime_value
from agents.yf_arc3_v5.scheduler.registry import (
    PureCallBlocked,
    PureCallDeferred,
    SchedulerRuntimeRegistry,
)
from agents.yf_arc3_v5.scheduler.runtime import (
    WorkflowInstanceNotFound,
    WorkflowScheduler,
)
from agents.yf_arc3_v5.src.production import production_control_symbol_definitions
from agents.yf_arc3_v5.src.registry import RegistryEntry, WorkflowRegistry
from agents.yf_arc3_v5.state import (
    EventStore,
    PutClaim,
    PutTerm,
    RecordArtifact,
    TransactionRequest,
)

WISDOM_WORKFLOW_ID = "core.operational_wisdom.operational_wisdom"
RELEASE_WORKFLOW_ID = "core.release_action.release_action"
ACTIVATE_DYNAMIC_GOAL_FRONTIER_WORKFLOW_ID = (
    "arc3.activate_dynamic_goal_frontier.activate_dynamic_goal_frontier"
)


def _interaction_probe_agenda(value: object) -> InteractionProbeAgenda:
    """Recover the exact validated agenda retained by immutable transport."""

    if (
        isinstance(value, RuntimeModelMap)
        and isinstance(value.source_model, InteractionProbeAgenda)
    ):
        return value.source_model
    return InteractionProbeAgenda.model_validate(value)


def _declared_terminal_marker_checkable(ref: str) -> bool:
    """Measure whether the terminal checker can evaluate this declared ref."""

    if ref.startswith("terminal.state."):
        return bool(ref.removeprefix("terminal.state."))
    if ref.startswith("terminal.score."):
        try:
            return int(ref.removeprefix("terminal.score.")) >= 0
        except ValueError:
            return False
    return False


def _selection_policy_field_refs(declaration: DrmDeclaration) -> tuple[str, ...]:
    """Return the exact descriptive fields read by one declared ordering.

    A selection policy first derives its candidate meanings through its ordered
    fact projections and then evaluates eligibility, tiers, and tie breakers.
    The compact shared context therefore has to retain projection premises as
    well as the final ordering fields.  Projection conclusions remain in each
    candidate delta and are not duplicated here.
    """

    conditions = [*declaration.selection_eligibility_conditions]
    for projection in declaration.fact_projections:
        conditions.extend(projection.all_conditions)
        conditions.extend(projection.any_conditions)
    for tier in declaration.selection_tiers:
        conditions.extend(tier.all_conditions)
        conditions.extend(tier.any_conditions)
    field_refs = set(declaration.selection_required_fact_fields)
    field_refs.update(condition.field for condition in conditions)
    field_refs.update(
        condition.reference_field
        for condition in conditions
        if condition.reference_field is not None
    )
    field_refs.update(item.field for item in declaration.selection_tie_breakers)
    field_refs.update(item.field for item in declaration.selection_precedence_criteria)
    return tuple(sorted(field_refs))


class SourceControlledActionCalls:
    """Execute cognitive gates from compiled SRC, then enforce mechanical safety."""

    def __init__(
        self,
        *,
        scheduler: WorkflowScheduler,
        workflow_registry: WorkflowRegistry,
        event_store: EventStore,
        drm_registry: DrmRegistry,
        capability_registry: CapabilityRegistry | None = None,
        world_inputs=None,
    ) -> None:
        self._scheduler = scheduler
        self._appearance_indexes: dict[str, CanonicalAppearanceIndex] = {}
        self._capabilities = capability_registry
        self._world_inputs = world_inputs
        self._workflows = workflow_registry
        self._store = event_store
        self._drm = drm_registry
        self._entries = {
            workflow_id: entry
            for entry in workflow_registry.entries
            for workflow_id in entry.workflow_ids
        }

    def register(self, runtime: SchedulerRuntimeRegistry, *, control_source_hashes: dict[str, str] | None = None) -> None:
        definitions = {
            item.id: (item if item.source_unit not in (control_source_hashes or {}) else
                      item.model_copy(update={"source_hash": control_source_hashes[item.source_unit]}))
            for item in production_control_symbol_definitions()
        }
        runtime.register_call(
            definitions["workflow.operational_wisdom"],
            self.operational_wisdom,
        )
        runtime.register_call(
            definitions["workflow.release_action"],
            self.release_action,
        )

        runtime.register_call(
            definitions["workflow.activate_dynamic_goal_frontier"],
            self.activate_dynamic_goal_frontier,
        )
        for callee_ref in (
            "workflow.construct_applied_principle",
            "workflow.instantiate_general_principle",
            "workflow.extend_insufficient_ontology",
            "workflow.plan_goal",
            "workflow.evaluate_constrained_plan",
            "workflow.assimilate_frame",
            "workflow.assimilate_block_grid_frame",
            "workflow.track_scene",
            "workflow.refresh_tracked_roles",
            "workflow.bind_typed_cell_lattices",
            "workflow.recall_navigation_cell_lattice",
            "workflow.retain_observed_logical_command_steps",
            "workflow.consult_current_shared_effects",
            "workflow.consult_current_shared_method",
            "workflow.assess_orthogonal_perimeter_assembly",
            "workflow.assess_mechanism_commitment_readiness",
            "workflow.assess_elimination_viability",
        ):
            runtime.register_call(
                definitions[callee_ref],
                self.execute_source_subworkflow,
            )
        runtime.register_call(
            definitions["control.prepare_interaction_probe"],
            self.prepare_interaction_probe,
        )
        runtime.register_call(
            definitions["control.probe_action_is_point"],
            self.probe_action_is_point,
        )
        runtime.register_call(
            definitions["control.probe_fact_is_true"],
            self.probe_fact_is_true,
        )
        runtime.register_call(
            definitions["control.validate_action_grounding"],
            self.validate_action_grounding,
        )
        runtime.register_call(
            definitions["control.action_grounding_is_valid"],
            self.action_grounding_is_valid,
        )
        runtime.register_call(
            definitions["control.bind_interaction_agenda"],
            self.bind_interaction_agenda,
        )
        runtime.register_call(
            definitions[
                "control.exclude_interaction_candidate_after_grounding_rejection"
            ],
            self.exclude_interaction_candidate_after_grounding_rejection,
        )
        runtime.register_call(
            definitions["control.assessment_selection_present"],
            self.assessment_selection_present,
        )
        runtime.register_call(
            definitions["control.projected_alternative_boolean"],
            self.projected_alternative_boolean,
        )
        runtime.register_call(
            definitions["control.project_declared_alternatives"],
            self.project_declared_alternatives,
        )
        runtime.register_call(
            definitions["control.project_memorial_binding_rows"], self.project_memorial_binding_rows,
        )
        runtime.register_call(
            definitions["control.read_perceptual_memory_matches"],
            self.read_perceptual_memory_matches,
        )
        runtime.register_call(
            definitions["control.measure_memorial_selection"], self.measure_memorial_selection,
        )
        runtime.register_call(
            definitions["control.measure_declared_role_appearance"],
            self.measure_declared_role_appearance,
        )
        runtime.register_call(
            definitions["control.project_declared_delta"],
            self.project_declared_delta,
        )
        runtime.register_call(
            definitions["control.project_declared_agenda"],
            self.project_declared_agenda,
        )
        runtime.register_call(
            definitions["control.project_declared_agenda_delta"],
            self.project_declared_agenda_delta,
        )
        runtime.register_call(
            definitions["control.materialize_declared_agenda_frontier"],
            self.materialize_declared_agenda_frontier,
        )
        runtime.register_call(
            definitions["control.retain_selected_agenda"],
            self.retain_selected_agenda,
        )
        runtime.register_call(
            definitions["control.instantiate_declared_meaning"],
            self.instantiate_declared_meaning,
        )
        runtime.register_call(
            definitions["control.instantiate_dynamic_goal_frontier_activation"],
            self.instantiate_dynamic_goal_frontier_activation,
        )
        runtime.register_call(
            definitions["control.dynamic_goal_frontier_payload_present"],
            self.dynamic_goal_frontier_payload_present,
        )
        runtime.register_call(
            definitions["control.activation_preserves_selected_action_goal"],
            self.activation_preserves_selected_action_goal,
        )
        runtime.register_call(
            definitions["control.dynamic_goal_frontier_activation_needs_commit"],
            self.dynamic_goal_frontier_activation_needs_commit,
        )
        runtime.register_call(
            definitions["control.access_goal_completion_payload_present"],
            self.access_goal_completion_payload_present,
        )
        runtime.register_call(
            definitions["control.committed_access_plan_present"],
            self.committed_access_plan_present,
        )
        runtime.register_call(
            definitions["control.instantiate_dynamic_goal_frontier_meaning"],
            self.instantiate_dynamic_goal_frontier_meaning,
        )
        runtime.register_call(definitions["control.prepare_optional_meaning_updates"],
                              self.prepare_optional_meaning_updates)
        runtime.register_call(
            definitions["control.prepare_meaning_updates"],
            self.prepare_meaning_updates,
        )
        runtime.register_call(
            definitions["control.measure_canonical_activity_support"],
            self.measure_canonical_activity_support,
        )
        runtime.register_call(
            definitions["control.measure_terminal_coverage_evidence"],
            self.measure_terminal_coverage_evidence,
        )
        runtime.register_call(
            definitions["control.measure_terminal_outcome_evidence"],
            self.measure_terminal_outcome_evidence,
        )
        runtime.register_call(definitions["control.measure_local_field_type_inventory"], self.measure_local_field_type_inventory)
        runtime.register_call(definitions["control.measure_declared_stock_dependency"], self.measure_declared_stock_dependency)
        runtime.register_call(definitions["control.measure_declared_pair_point_geometry"], self.measure_declared_pair_point_geometry)
        runtime.register_call(definitions["control.bind_declared_pair_point_agenda"], self.bind_declared_pair_point_agenda)
        runtime.register_call(definitions["control.measure_current_pair_stock_change"], self.measure_current_pair_stock_change)
        runtime.register_call(definitions["control.bind_current_stock_lineage"], self.bind_current_stock_lineage)
        runtime.register_call(definitions["control.measure_current_local_field_delivery"], self.measure_current_local_field_delivery)
        runtime.register_call(
            definitions["control.measure_winning_navigation"],
            self.measure_winning_navigation,
        )
        runtime.register_call(
            definitions["control.measure_reference_write_evidence"],
            self.measure_reference_write_evidence,
        )
        runtime.register_call(
            definitions["control.measure_current_terminal_inventory"],
            self.measure_current_terminal_inventory,
        )
        runtime.register_call(
            definitions["control.measure_logical_command_steps"],
            self.measure_logical_command_steps,
        )
        runtime.register_call(
            definitions["control.measure_typed_cell_lattices"],
            self.measure_typed_cell_lattices,
        )
        runtime.register_call(
            definitions["control.measure_recalled_navigation_lattices"],
            self.measure_recalled_navigation_lattices,
        )
        runtime.register_call(
            definitions["control.measure_current_method_activity_coverage"],
            self.measure_current_method_activity_coverage,
        )
        runtime.register_call(
            definitions["control.measure_current_shared_method_inventory"],
            self.measure_current_shared_method_inventory,
        )
        runtime.register_call(
            definitions["control.measure_current_shared_effect_inventory"],
            self.measure_current_shared_effect_inventory,
        )
        runtime.register_call(
            definitions["control.project_current_method_activity_premises"],
            self.project_current_method_activity_premises,
        )
        runtime.register_call(
            definitions["control.prepare_revisable_meaning_updates"],
            self.prepare_revisable_meaning_updates,
        )
        runtime.register_call(
            definitions["control.flatten_disjoint_update_batch"],
            self.flatten_disjoint_update_batch,
        )
        runtime.register_call(
            definitions["control.link_disjoint_update_batch"],
            self.link_disjoint_update_batch,
        )
        runtime.register_call(
            definitions["control.prepare_goal_priority_context_updates"],
            self.prepare_goal_priority_context_updates,
        )
        runtime.register_call(
            definitions["control.prepare_committed_derivation"],
            self.prepare_committed_derivation,
        )
        runtime.register_call(
            definitions["control.bind_derivation_basis"],
            self.bind_derivation_basis,
        )
        runtime.register_call(
            definitions["control.prepare_committed_proposal"],
            self.prepare_committed_proposal,
        )
        runtime.register_call(
            definitions["control.bind_declared_scene_handoff"],
            self.bind_declared_scene_handoff,
        )
        runtime.register_call(
            definitions["control.refresh_world_context_snapshot"],
            self.refresh_world_context_snapshot,
        )
        runtime.register_call(
            definitions["control.current_state_revision"],
            self.current_state_revision,
        )
        runtime.register_call(
            definitions["control.verify_declared_continue_terminal"],
            self.verify_declared_continue_terminal,
        )
        runtime.register_call(
            definitions["control.current_term_attribute_values"],
            self.current_term_attribute_values,
        )
        runtime.register_call(
            definitions["control.current_term_attribute_rows"],
            self.current_term_attribute_rows,
        )
        runtime.register_call(
            definitions["control.measure_transformation_pair_recurrence"],
            self.measure_transformation_pair_recurrence,
        )
        runtime.register_call(
            definitions["control.measure_observed_extent_commands"],
            self.measure_observed_extent_commands,
        )
        runtime.register_call(
            definitions["control.measure_observed_axis_transport"],
            self.measure_observed_axis_transport,
        )
        runtime.register_call(
            definitions["control.measure_carried_reference_center"],
            self.measure_carried_reference_center,
        )
        runtime.register_call(
            definitions["control.measure_observed_axis_frontier"],
            self.measure_observed_axis_frontier,
        )
        runtime.register_call(
            definitions["control.measure_remembered_coupling_render"],
            self.measure_remembered_coupling_render,
        )
        runtime.register_call(
            definitions["control.measure_current_coupling_context"],
            self.measure_current_coupling_context,
        )
        runtime.register_call(
            definitions["control.current_context_term_attribute_rows"],
            self.current_context_term_attribute_rows,
        )
        runtime.register_call(
            definitions["control.declared_meaning_is_applicable"],
            self.declared_meaning_is_applicable,
        )
        runtime.register_call(
            definitions["control.declared_selection_has_eligible_alternative"],
            self.declared_selection_has_eligible_alternative,
        )
        runtime.register_call(
            definitions["control.prepare_optional_revisable_meaning_updates"],
            self.prepare_optional_revisable_meaning_updates,
        )
        runtime.register_call(
            definitions["control.goal_priority_context_needs_commit"],
            self.goal_priority_context_needs_commit,
        )

    def _declared_consolidation_reason_refs_for(
        self,
        consolidated_priority_ref: str,
    ) -> tuple[str, ...]:
        """Transport existing DRM reasons for one exact consolidated priority."""

        if not consolidated_priority_ref:
            return ()
        refs: set[str] = set()
        for claim in self._store.snapshot.current_claims:
            attributes = getattr(claim, "attributes", {})
            if not isinstance(attributes, Mapping):
                continue
            if str(attributes.get("consolidated_priority_ref") or "") != (
                consolidated_priority_ref
            ):
                continue
            raw = attributes.get("consolidation_reason_refs") or ()
            values = (raw,) if isinstance(raw, str) else raw
            if isinstance(values, (tuple, list, set, frozenset)):
                refs.update(str(item) for item in values if item)
        return tuple(sorted(refs))[:8]

    def measure_declared_role_appearance(self, context: FunctionCallContext) -> FrozenMap:
        """Join exact canonical source links to their current measured support.

        Admission predicates and semantic values are supplied by DRM, never
        inferred here from palette, movement heuristics or names. The bounded
        candidates are the exact terms/claims committed by the calling SRC.
        """
        if context.arguments["tracking"] is None:
            return FrozenMap({"tracking_present": False, "has_learning_rows": False, "descriptive_facts": FrozenMap({"learning_rows": ()}), "evidence_refs": ()})
        declaration, _ = self._drm.resolve(str(context.arguments["learning_policy_ref"]))
        projection = declaration.fact_projections[0].static_facts["canonical_learning_projection"]
        key_declaration, key_loaded = self._drm.resolve(str(context.arguments["key_policy_ref"]))
        key_facts = key_declaration.fact_projections[0].static_facts
        contract = AppearanceKeyContract(
            tiers=key_facts["appearance_key_contract"],
            rotation_quarter_turns=key_facts["rotation_quarter_turns"],
            source_hash=key_loaded.source_hash,
            digital_topology_contract={**key_facts["digital_topology_contract"], "source_hash": key_loaded.source_hash},
        )
        tracking = TemporalTrackingResult.model_validate(context.arguments["tracking"])
        term_candidates = context.arguments["canonical_term_candidates"]
        claim_candidates = context.arguments["canonical_claim_candidates"]
        max_items = int(context.arguments["max_items"])
        if max_items < 1 or len(term_candidates) > max_items or len(claim_candidates) > max_items:
            raise PureCallBlocked("canonical role capture bound exceeded")
        if self._capabilities is None:
            raise PureCallBlocked("role appearance capture requires the production capability registry")
        entities = {item.entity_id: item for item in tracking.entities}
        source_claims = tuple(self._store.snapshot.claim(str(item["id"])) for item in claim_candidates)
        current_link_refs = frozenset(claim.id for claim in self._store.snapshot.claims_for_predicate(projection["link_predicate"]))
        rows = []
        evidence = []
        tokens = set()
        source_rows = []
        for candidate in term_candidates:
            if projection.get("entity_reference_origin") == "claim_argument":
                for claim in source_claims:
                    index = projection["claim_entity_argument_index"]
                    if claim is not None and candidate["id"] in claim.arguments and len(claim.arguments) > index:
                        source_rows.append((candidate, claim.arguments[index]))
            else:
                source_rows.append((candidate, None))
        for candidate, entity_reference in source_rows:
            term = self._store.snapshot.term(str(candidate["id"]))
            if term is None or any(term.attributes.get(key) != value for key, value in sorted(projection["term_attributes"].items())):
                continue
            if projection.get("transition_reference_origin") != "claim" and term.attributes.get(projection["transition_attribute"]) != context.arguments["transition_ref"]:
                continue
            member_attribute = projection.get("member_entity_refs_attribute")
            member_refs = term.attributes.get(member_attribute) if member_attribute is not None else None
            compound_measurement = projection.get("compound_membership_measurement")
            translated_union_measured = False
            if compound_measurement is not None and context.arguments.get("before_tracking") is not None:
                if compound_measurement["relation"] != "translated_enclosed_support":
                    raise PureCallBlocked("unsupported declared translated support relation")
                try:
                    measured_refs = measure_translated_enclosed_member_refs(
                        before=TemporalTrackingResult.model_validate(context.arguments["before_tracking"]), after=tracking,
                        anchor_entity_ref=term.attributes[projection["entity_attribute"]],
                        max_members=compound_measurement["max_members"], max_pixels=compound_measurement["max_pixels"],
                    )
                except ValueError as error:
                    raise PureCallBlocked(str(error)) from error
                if measured_refs is not None:
                    member_refs = measured_refs
                    translated_union_measured = True
            bound_attribute = projection.get("member_bound_exceeded_attribute")
            if bound_attribute is not None and term.attributes.get(bound_attribute) is True:
                continue
            compound_present = isinstance(member_refs, (tuple, list)) and len(member_refs) >= 2
            if member_attribute is not None and not compound_present and projection.get("singleton_member_support") != "tracked_anchor":
                continue
            if compound_present:
                anchor_attribute = projection.get("compound_source_entity_attribute")
                source_entity_ref = term.attributes.get(anchor_attribute) if anchor_attribute is not None else term.id
                if anchor_attribute is not None and source_entity_ref not in member_refs:
                    continue
                if any(reference not in entities or entities[reference].current_frame_ref != tracking.frame_ref for reference in member_refs):
                    continue
                member_deltas = {(entities[reference].delta_row, entities[reference].delta_col) for reference in member_refs}
                if not translated_union_measured and projection.get("require_common_nonzero_member_translation") is True and (len(member_deltas) != 1 or (0, 0) in member_deltas):
                    continue
                try:
                    support = measure_current_bearer_supports(
                        tracking=tracking,
                        requests=(CurrentBearerSupportRequest(bearer_ref=term.id, member_entity_refs=tuple(member_refs), observation_ref=tracking.frame_ref),),
                        contract=BearerSupportContract.model_validate(key_facts["bearer_support_contract"]),
                    )[0]
                except ValueError as error:
                    raise PureCallBlocked(str(error)) from error
                bearer_ref = support.bearer_ref
                valued_pixels = support.valued_pixels
            else:
                entity_ref = entity_reference if entity_reference is not None else term.id if projection.get("entity_reference_origin") == "term_id" else term.attributes.get(projection["entity_attribute"])
                entity = entities.get(entity_ref)
                if entity is None or entity.current_frame_ref != tracking.frame_ref:
                    continue
                translation_required = projection.get("nonzero_translation_required")
                if translation_required is not None and ((entity.delta_row, entity.delta_col) != (0, 0)) != translation_required:
                    continue
                component_attribute = projection["component_attribute"]
                if component_attribute is not None and term.attributes.get(component_attribute) != entity.component.component_id:
                    continue
                source_entity_ref = entity.entity_id
                bearer_ref = entity.component.component_id
                valued_pixels = tuple((row, col, entity.component.value) for row, col in entity.component.pixels)
            links = tuple(claim for claim in source_claims if claim is not None and (
                claim.id in current_link_refs and term.id in claim.arguments
                and len(claim.arguments) > projection["claim_entity_argument_index"]
                and claim.arguments[projection["claim_entity_argument_index"]] == source_entity_ref
                and all(claim.attributes.get(key) == value for key, value in sorted(projection.get("claim_attributes", {}).items()))
                and all(len(claim.arguments) > int(index) and claim.arguments[int(index)] == reference
                        for index, reference in sorted(projection.get("claim_required_arguments", {}).items()))
                and claim.predicate == projection["link_predicate"]
                and claim.proof_rule == projection["proof_rule"]
                and claim.epistemic_status.value in projection["allowed_link_statuses"]
                and claim.disposition.value == projection["link_disposition"]
                and claim.polarity.value == projection["link_polarity"]
                and not claim.active_contradictions and not claim.defeated_by
                and claim.attributes.get(projection.get("claim_transition_attribute", projection["transition_attribute"])) == context.arguments["transition_ref"]
                and claim.attributes.get(projection["action_attribute"]) == tracking.action_ref
                and tracking.action_ref is not None
            ))
            if not links:
                continue
            measurement = self._capabilities.invoke(RevisionedCapabilityRequest(
                capability_id="capability.bearer_appearance_measurement",
                input_revision=self._store.snapshot.revision,
                payload=BearerAppearanceInput(
                    bearer_ref=bearer_ref,
                    valued_pixels=valued_pixels,
                    measurement_basis_ref=str(context.arguments["measurement_basis_ref"]),
                    cell_quantum=context.arguments["cell_quantum"], key_contract=contract,
                ),
            )).output
            facts = projection["static_learning_facts"]
            measured_members = tuple(member_refs) if compound_present else (source_entity_ref,)
            member_aspects = FrozenMap({name: sum(
                entities[reference].component.value == value
                for reference in measured_members if reference != source_entity_ref
            ) for name, value in sorted(projection.get("member_value_count_measurements", {}).items())})
            context_field = key_facts["canonical_index_projection"]["context_requirements_field"]
            context_requirements = term.attributes.get(context_field, FrozenMap())
            token = stable_digest((facts["causal_role_ref"], measurement.descriptive_delta["signature_keys"], context_requirements, contract.source_hash))
            entry_ref = projection["dedup_entry_id_pattern"].format(association_token=token)
            if token in tokens or self._store.snapshot.term(entry_ref) is not None:
                continue
            tokens.add(token)
            link_refs = tuple(claim.id for claim in links)
            rows.append(FrozenMap({
                **facts, **measurement.descriptive_delta,
                **key_facts["role_capture_neutral_measurements"],
                **member_aspects,
                "association_token": token, "source_claim_refs": link_refs,
                "source_role_term_ref": term.id,
                context_field: context_requirements,
            }))
            evidence.extend(link_refs)
        return FrozenMap({
            "tracking_present": True,
            "has_learning_rows": bool(rows),
            "descriptive_facts": FrozenMap({"learning_rows": tuple(rows)}),
            "evidence_refs": tuple(dict.fromkeys(evidence)),
        })

    def _project_current_memorial_binding_rows(self, rows, projected_contract):
        if not rows:
            return ()
        revisions = project_declared_alternatives(
            drm=self._drm,
            selection_policy_ref=projected_contract["current_binding_revision_policy_ref"],
            alternative_refs=tuple(row["binding_token"] for row in rows),
            descriptive_facts=FrozenMap({row["binding_token"]: row for row in rows}),
        )
        rows = tuple(revisions.alternative_facts[row["binding_token"]] for row in rows)
        maxima = {}
        for row in rows:
            if row["binding_rank_eligible"]:
                key = row["role_ref"]
                maxima[key] = max(maxima.get(key, -1), row["match_rank"])
        rows = tuple(FrozenMap({**row, "scene_role_max_match_rank": maxima.get(row["role_ref"], -1)}) for row in rows)
        ranked = project_declared_alternatives(
            drm=self._drm,
            selection_policy_ref=projected_contract["current_role_rank_policy_ref"],
            alternative_refs=tuple(row["binding_token"] for row in rows),
            descriptive_facts=FrozenMap({row["binding_token"]: row for row in rows}),
        )
        rows = tuple(ranked.alternative_facts[row["binding_token"]] for row in rows)
        return rows

    def project_memorial_binding_rows(self, context: FunctionCallContext) -> FrozenMap:
        declaration, _loaded = self._drm.resolve(str(context.arguments["key_policy_ref"]))
        facts = context.arguments["recognition_facts"]
        rows = self._project_current_memorial_binding_rows(
            tuple(facts["recognition_rows"]), declaration.fact_projections[0].static_facts,
        )
        return FrozenMap({**facts, "recognition_rows": rows})

    def read_perceptual_memory_matches(self, context: FunctionCallContext) -> FrozenMap:
        """Measure current supports against a source-declared canonical index.

        The capability measures; no role, query order or action is invented.
        Meaning attributes are projected by the DRM-provided binding table.
        """
        declaration, loaded = self._drm.resolve(str(context.arguments["key_policy_ref"]))
        projected_contract = declaration.fact_projections[0].static_facts
        contract = AppearanceKeyContract(
            tiers=projected_contract["appearance_key_contract"],
            rotation_quarter_turns=projected_contract["rotation_quarter_turns"],
            source_hash=loaded.source_hash,
            digital_topology_contract={**projected_contract["digital_topology_contract"], "source_hash": loaded.source_hash},
        )
        index = self._appearance_indexes.get(loaded.source_hash)
        if index is None:
            index = CanonicalAppearanceIndex(
                store=self._store, contract_source_hash=loaded.source_hash,
                projection=projected_contract["canonical_index_projection"],
            )
            self._appearance_indexes[loaded.source_hash] = index
        index.synchronize()
        if index.entry_count == 0:
            return FrozenMap({"has_matches": False, "descriptive_facts": FrozenMap({"recognition_rows": ()}), "evidence_refs": ()})
        tracking_value = context.arguments.get("tracking")
        tracking = TemporalTrackingResult.model_validate(tracking_value) if tracking_value is not None else None
        scene = VisualSceneDescription.model_validate(context.arguments["scene"]) if tracking is None else None
        components = tuple((item.entity_id, item.component) for item in tracking.entities) if tracking is not None else tuple((item.component_id, item) for item in scene.blocks.components)
        max_bearers = int(context.arguments["max_bearers"])
        raw_supports = context.arguments.get("current_support_requests") or ()
        support_contract = BearerSupportContract.model_validate(projected_contract["bearer_support_contract"])
        if len(components) + len(raw_supports) > max_bearers or len(raw_supports) > support_contract.max_supports:
            raise PureCallBlocked("current bearer enumeration bound exceeded")
        if raw_supports and tracking is None:
            raise PureCallBlocked("composed bearer support requires current tracking")
        try:
            requests = tuple(CurrentBearerSupportRequest.model_validate(request) for request in raw_supports)
            cell_dependencies = FrozenMap()
            if tracking is not None:
                from agents.yf_arc3_v5.state.cell_bearer_supports import current_cell_bearer_requests
                cell_requests, cell_dependencies = current_cell_bearer_requests(snapshot=self._store.snapshot, tracking=tracking,
                    profile=projected_contract["cell_bearer_support_contract"], support_contract=support_contract)
                supplied_memberships = {frozenset(request.member_entity_refs) for request in requests}
                requests += tuple(request for request in cell_requests
                    if frozenset(request.member_entity_refs) not in supplied_memberships)
                relocation_contract = BearerRelocationContract.model_validate(projected_contract["bearer_relocation_contract"])
                templates = index.relocation_templates(max_templates=relocation_contract.max_templates)
                from agents.yf_arc3_v5.state.observed_extent_supports import measured_extent_supports
                extent_requests, extent_dependencies = measured_extent_supports(
                    snapshot=self._store.snapshot, tracking=tracking, templates=templates,
                    profile=projected_contract["observed_extent_contract"],
                    relocation_contract=relocation_contract, support_contract=support_contract,
                )
                supplied_memberships = {frozenset(request.member_entity_refs) for request in requests}
                requests += tuple(request for request in extent_requests
                    if frozenset(request.member_entity_refs) not in supplied_memberships)
                cell_dependencies = FrozenMap({**cell_dependencies, **extent_dependencies})
                relocated = relocate_exact_bearer_templates(
                    tracking=tracking, templates=templates,
                    contract=relocation_contract, support_contract=support_contract,
                )
                supplied_memberships = {frozenset(request.member_entity_refs) for request in requests}
                requests += tuple(request for request in relocated if frozenset(request.member_entity_refs) not in supplied_memberships)
            if len(components) + len(requests) > max_bearers:
                raise PureCallBlocked("current bearer enumeration bound exceeded after relocation")
            supports = measure_current_bearer_supports(
                tracking=tracking,
                requests=requests,
                contract=support_contract,
            ) if tracking is not None else ()
        except ValueError as error:
            raise PureCallBlocked(str(error)) from error
        if self._capabilities is None:
            raise PureCallBlocked("appearance measurement requires the production capability registry")
        measurements = tuple(self._capabilities.invoke(RevisionedCapabilityRequest(
            capability_id="capability.bearer_appearance_measurement",
            input_revision=self._store.snapshot.revision,
            payload=BearerAppearanceInput(
                bearer_ref=bearer_ref,
                valued_pixels=tuple((row, column, component.value) for row, column in component.pixels),
                measurement_basis_ref=str(context.arguments["measurement_basis_ref"]),
                cell_quantum=context.arguments["cell_quantum"], key_contract=contract,
            ),
        )).output for bearer_ref, component in components)
        measurements += tuple(self._capabilities.invoke(RevisionedCapabilityRequest(
            capability_id="capability.bearer_appearance_measurement",
            input_revision=self._store.snapshot.revision,
            payload=BearerAppearanceInput(
                bearer_ref=support.bearer_ref, valued_pixels=support.valued_pixels,
                measurement_basis_ref=str(context.arguments["measurement_basis_ref"]),
                cell_quantum=context.arguments["cell_quantum"], key_contract=contract,
                max_bbox_area=support_contract.max_bbox_area,
            ),
        )).output for support in supports)
        rows = index.recognition_rows(
            measurements=measurements, observation_ref=str(context.arguments["observation_ref"]),
            max_matches_per_bearer=int(context.arguments["max_matches_per_bearer"]),
            max_rows=int(context.arguments["max_rows"]),
            bearer_contexts=self._read_declared_bearer_contexts(
                projected_contract["canonical_index_projection"]["current_context_projection"],
                frame_ref=str(context.arguments["observation_ref"]),
            ),
        )
        geometry = {bearer_ref: FrozenMap({
            "current_tracking_present": True,
            "current_entity_ref": bearer_ref,
            "current_member_entity_refs": (bearer_ref,),
            "current_support_digest": stable_digest(tuple(
                (row, column, component.value) for row, column in component.pixels
            )),
            "current_component_ref": component.component_id,
            "current_probe_candidate_ref": projected_contract["canonical_index_projection"]["current_probe_candidate_ref_pattern"].format(component_ref=component.component_id),
            "current_shape_digest": stable_digest(component.relative_pixels)[:16],
            "current_bbox_height": component.bbox.height,
            "current_bbox_width": component.bbox.width,
            "current_observed_value": component.value,
        }) for bearer_ref, component in components} if tracking is not None else {}
        geometry.update({support.bearer_ref: FrozenMap({
            "current_support_present": True,
            "current_member_count": len(support.member_entity_refs),
            "current_member_entity_refs": support.member_entity_refs,
            "current_support_digest": support.support_digest,
            "current_bbox_top": support.bbox_top,
            "current_bbox_left": support.bbox_left,
            "current_bbox_height": support.bbox_height,
            "current_bbox_width": support.bbox_width,
        }) for support in supports})
        anchor_contract = projected_contract["compound_anchor_contract"]
        if anchor_contract["relation"] != "proper_bbox_enclosure":
            raise PureCallBlocked("unsupported declared compound anchor relation")
        tracked_entities = {entity.entity_id: entity for entity in tracking.entities} if tracking is not None else {}
        for support in supports:
            enclosing_refs = measure_proper_bbox_enclosing_members(
                tracking=tracking, support=support, max_members=anchor_contract["max_members"],
            )
            if len(enclosing_refs) != 1:
                continue
            unique_anchor = tracked_entities[enclosing_refs[0]]
            component = unique_anchor.component
            geometry[support.bearer_ref] = FrozenMap.overlay(FrozenMap({
                "current_tracking_present": True,
                "current_entity_ref": unique_anchor.entity_id,
                "current_component_ref": component.component_id,
                "current_probe_candidate_ref": projected_contract["canonical_index_projection"]["current_probe_candidate_ref_pattern"].format(component_ref=component.component_id),
                "current_shape_digest": stable_digest(component.relative_pixels)[:16],
                "current_observed_value": component.value,
            }), geometry[support.bearer_ref])
        rows = tuple(FrozenMap({**row, **geometry.get(row["bearer_ref"], FrozenMap()),
            **{field: row[field] + values for field, values in sorted(cell_dependencies.get(row["bearer_ref"], FrozenMap()).items())}
        }) for row in rows)
        rows = self._project_current_memorial_binding_rows(rows, projected_contract)
        if context.arguments.get("omit_current_bindings") is True:
            projection = projected_contract["canonical_index_projection"]
            retained = []
            for row in rows:
                existing = self._store.snapshot.term(projection["binding_id_pattern"].format(binding_token=row["binding_token"]))
                if existing is None or any(existing.attributes.get(field) != row[field] for field in projection["binding_dedup_fields"]):
                    retained.append(row)
            rows = tuple(retained)
        return FrozenMap({
            "has_matches": bool(rows), "descriptive_facts": FrozenMap({"recognition_rows": rows}),
            "evidence_refs": tuple(dict.fromkeys(row["association_ref"] for row in rows)),
        })

    def _read_declared_bearer_contexts(self, projection: FrozenMap, *, frame_ref: str) -> FrozenMap:
        """Read already declared zone/condition literals; never classify a zone.

        Every live context retains its exact TERM revision and positive proof.
        Competing contexts remain separate; a missing context stays unknown.
        """
        if (projection["comparison"] != "exact_known_literals"
                or projection["unknown_context"] != "retain_revisable"
                or projection["mismatched_context"] != "exclude_before_ordinal_dominance"):
            raise PureCallBlocked("unsupported declared context comparison")
        snapshot = self._store.snapshot
        terms = snapshot.terms_for_literal_attribute(projection["query_attribute"], projection["query_value"])
        rows_by_entity = {}
        count = 0
        for term in sorted(terms, key=lambda item: item.id):
            if term.attributes.get("operational_scene_ref") != frame_ref or not canonical_reference_bindings_current(snapshot, term, frame_ref=frame_ref):
                continue
            values = term.attributes.get(projection["values_attribute"])
            references = term.attributes.get(projection["premise_attribute"], ())
            entity_ref = term.attributes.get(projection["entity_attribute"])
            if not isinstance(values, FrozenMap) or not values or not references or not entity_ref:
                continue
            if len(values) > projection["max_fields"] or len(references) > projection["max_terms"]:
                raise PureCallBlocked("current bearer context field/proof bound exceeded")
            claims = tuple(snapshot.claim(reference) for reference in references)
            if any(claim is None or claim.epistemic_status.value not in projection["allowed_statuses"]
                   or claim.disposition.value != projection["disposition"] or claim.polarity.value != projection["polarity"]
                   or claim.active_contradictions or claim.defeated_by
                   or not any(current.id == claim.id for current in snapshot.claims_for_predicate(claim.predicate))
                   for claim in claims):
                continue
            count += 1
            if count > projection["max_terms"]:
                raise PureCallBlocked("current bearer context bound exceeded; no alternatives truncated")
            rows_by_entity.setdefault(entity_ref, []).append(FrozenMap({
                "values": values,
                "term_dependency": FrozenMap({"term_ref": term.id, "revision": term.last_changed_state_revision}),
                "claim_dependencies": tuple(FrozenMap({"claim_ref": claim.id, "content_digest": stable_digest(claim)})
                                            for claim in sorted(claims, key=lambda item: item.id)),
            }))
        return FrozenMap({ref: tuple(rows) for ref, rows in sorted(rows_by_entity.items())})

    def _current_compound_marker_requests(self, *, tracking, projection):
        snapshot = self._store.snapshot
        terms = tuple(term for term in snapshot.terms_for_literal_attribute(
            projection["compound_scene_attribute"], tracking.frame_ref)
            if term.attributes.get(projection["compound_query_attribute"]) == projection["compound_query_value"])
        if len(terms) > projection["max_roles"]:
            raise PureCallBlocked("compound marker role bound exceeded; no proofs truncated")
        requests = []
        for term in sorted(terms, key=lambda item: item.id):
            if not canonical_reference_bindings_current(
                snapshot, term, frame_ref=tracking.frame_ref,
                require_projectable=projection["compound_require_operational_projection"],
            ):
                continue
            requests.append(CurrentBearerSupportRequest(bearer_ref=term.id,
                member_entity_refs=tuple(term.attributes[projection["compound_member_attribute"]]),
                observation_ref=tracking.frame_ref))
        return tuple(requests)

    def measure_memorial_selection(self, context: FunctionCallContext) -> FrozenMap:
        """Measure a remembered decoration on current bodies of any value/shape.

        Source owns the interpretation and whether an unmarked role becomes
        dormant. Every enclosure and tie is retained before Source projection.
        """
        declaration, _loaded = self._drm.resolve(str(context.arguments["selection_policy_ref"]))
        projection = declaration.fact_projections[0].static_facts["marker_projection"]
        snapshot = self._store.snapshot
        tracking = TemporalTrackingResult.model_validate(context.arguments["tracking"])
        hints = snapshot.claims_for_predicate(projection["hint_predicate"])
        requests = self._current_compound_marker_requests(tracking=tracking, projection=projection)
        rows, compound_rows, evidence = [], [], []
        for hint in sorted(hints, key=lambda item: item.id):
            if (hint.epistemic_status.value not in projection["allowed_hint_statuses"]
                    or hint.disposition.value != projection["hint_disposition"]
                    or hint.polarity.value != projection["hint_polarity"]
                    or hint.active_contradictions or hint.defeated_by):
                continue
            if len(hint.arguments) <= projection["schema_argument_index"]:
                continue
            schema = snapshot.term(hint.arguments[projection["schema_argument_index"]])
            if schema is None or projection["marker_value_attribute"] not in schema.attributes:
                continue
            measurements, compounds = measure_marker_ownership(tracking,
                marker_value=int(schema.attributes[projection["marker_value_attribute"]]), requests=requests,
                max_entities=projection["max_entities"],
                support_contract=BearerSupportContract.model_validate(projection["compound_support_contract"]))
            for measurement in (*measurements, *compounds):
                token = stable_digest((tracking.frame_ref, schema.id, schema.last_changed_state_revision,
                                       stable_digest(hint), measurement))
                source = snapshot.term(measurement["bearer_ref"]) if "bearer_ref" in measurement else None
                premise_refs = (hint.id,) + (tuple(source.attributes["role_premise_claim_refs"]) if source is not None else ())
                evidence.extend(premise_refs)
                dependencies = (FrozenMap({"term_ref": schema.id, "revision": schema.last_changed_state_revision}),)
                if source is not None:
                    dependencies += (FrozenMap({"term_ref": source.id, "revision": source.last_changed_state_revision}),) + tuple(source.attributes["canonical_term_dependency_revisions"])
                claim_dependencies = tuple(FrozenMap({"claim_ref": ref, "content_digest": stable_digest(snapshot.claim(ref))}) for ref in premise_refs)
                if source is not None:
                    claim_dependencies += tuple(source.attributes["canonical_claim_dependency_digests"])
                row = FrozenMap.overlay(FrozenMap({"selection_token": token, "observation_ref": tracking.frame_ref,
                    "row_evidence_refs": premise_refs,
                    "role_premise_claim_refs": (projection["compound_selection_claim_pattern" if source is not None else "current_selection_claim_pattern"].format(selection_token=token),),
                    "canonical_term_dependency_revisions": dependencies,
                    "canonical_claim_dependency_digests": claim_dependencies,
                }), measurement)
                (compound_rows if source is not None else rows).append(row)
            evidence.append(hint.id)
        if len(rows) + len(compound_rows) > projection["max_rows"]:
            raise PureCallBlocked("memorial marker row bound exceeded; no ties truncated")
        all_rows = (*rows, *compound_rows)
        inner_refs = {row["carrier_identity"] for row in all_rows if not row["encloses_more_specific_marker_carrier"]}
        marked_refs = {row["entity_ref"] for row in rows}
        marked_memberships = {tuple(row["member_refs"]) for row in compound_rows}
        # Without any measured enclosure the declared update cannot apply
        # (has_measurements is false). Do not enumerate unrelated role proofs
        # or exhaust their bound for an empty marker measurement.
        role_terms = (snapshot.terms_for_literal_attribute(
            projection["current_role_query_attribute"], projection["current_role_query_value"])
            if all_rows else ())
        unmarked = []
        for term in sorted(role_terms, key=lambda item: item.id):
            if canonical_reference_bindings_current(snapshot, term, frame_ref=tracking.frame_ref):
                unmarked.append(FrozenMap({"term_ref": term.id,
                    "bearer_has_enclosed_markers": term.attributes.get("entity_ref") in marked_refs or tuple(sorted(term.attributes.get(projection["compound_member_attribute"], ()))) in marked_memberships,
                    "innermost_enclosure_count": len(inner_refs), "row_evidence_refs": tuple(sorted(set(evidence)))}))
        if len(unmarked) > projection["max_roles"]:
            raise PureCallBlocked("memorial current role bound exceeded; no roles truncated")
        pending = tuple(row for row in rows if snapshot.term(projection["current_selection_term_pattern"].format(selection_token=row["selection_token"])) is None)
        compound_pending = tuple(row for row in compound_rows if snapshot.term(projection["compound_selection_term_pattern"].format(selection_token=row["selection_token"])) is None)
        pending_count = sum(all(row[field] == value for field, value in sorted(projection["pending_row_literals"].items())) for row in (*pending, *compound_pending))
        role_update_count = sum(all(row[field] == value for field, value in sorted(projection["role_update_row_literals"].items())) for row in unmarked)
        return FrozenMap({"has_measurements": bool(all_rows) and bool(pending_count or role_update_count),
            "descriptive_facts": FrozenMap({"selection_rows": pending, "compound_selection_rows": compound_pending, "unmarked_role_rows": tuple(unmarked),
                                            "innermost_enclosure_count": len(inner_refs)}),
            "evidence_refs": tuple(sorted(set(evidence)))})

    def project_declared_alternatives(
        self,
        context: FunctionCallContext,
    ) -> object:
        facts = context.arguments.get("descriptive_facts")
        if not isinstance(facts, Mapping):
            raise PureCallBlocked("declared projection requires descriptive facts")
        try:
            return project_declared_alternatives(
                drm=self._drm,
                selection_policy_ref=str(
                    context.arguments.get("selection_policy_ref") or ""
                ),
                alternative_refs=tuple(
                    str(item)
                    for item in context.arguments.get("alternative_refs", ())
                ),
                descriptive_facts=facts,
            )
        except (TypeError, ValueError, LookupError) as error:
            raise PureCallBlocked(str(error)) from error

    def verify_declared_continue_terminal(
        self,
        context: FunctionCallContext,
    ) -> str:
        """Check only the declared terminal markers after an action series.

        This is deliberately a terminal mechanical check.  It does not infer
        a goal, inspect a scene, or reopen perception.  The declared refs may
        name only observable terminal markers (currently state or score).  An
        absent or unrecognized marker fails closed into retrospection.
        """

        observation = context.arguments.get("observation")
        evidence = (
            observation.get("evidence", observation)
            if isinstance(observation, Mapping)
            else getattr(observation, "evidence", observation)
        )
        measurements = (
            evidence.get("measurements")
            if isinstance(evidence, Mapping)
            else getattr(evidence, "measurements", None)
        )
        if not isinstance(measurements, Mapping):
            return "continue.terminal_unknown_retro"
        reference_score = context.arguments.get("reference_score")
        observed_score = measurements.get("score")
        if (
            isinstance(reference_score, int)
            and not isinstance(reference_score, bool)
            and isinstance(observed_score, int)
            and not isinstance(observed_score, bool)
            and observed_score > reference_score
        ):
            return "continue.terminal_match"
        refs = tuple(
            str(item)
            for item in context.arguments.get("expected_terminal_refs", ())
        )
        recognized = False
        for ref in refs:
            if ref.startswith("terminal.state."):
                recognized = True
                expected = ref.removeprefix("terminal.state.")
                if str(measurements.get("state") or "") == expected:
                    return "continue.terminal_match"
            elif ref.startswith("terminal.score."):
                recognized = True
                try:
                    expected_score = int(ref.removeprefix("terminal.score."))
                except ValueError:
                    continue
                if measurements.get("score") == expected_score:
                    return "continue.terminal_match"
        if recognized:
            return "continue.terminal_mismatch_retro"
        return "continue.terminal_unknown_retro"

    def project_declared_agenda(
        self,
        context: FunctionCallContext,
    ) -> object:
        try:
            agenda = _interaction_probe_agenda(context.arguments["agenda"])
            canonical_facts = self._project_declared_canonical_joins(
                str(context.arguments.get("selection_policy_ref") or ""),
                agenda.alternative_refs, agenda.alternative_facts, agenda.context_facts, context.frame_id)
            descriptive_facts = (
                {
                    ref: ChainMap(
                        canonical_facts[ref], agenda.context_facts
                    )
                    for ref in agenda.alternative_refs
                }
                if agenda.context_facts
                else canonical_facts
            )
            projected = project_declared_alternatives(
                drm=self._drm,
                selection_policy_ref=str(
                    context.arguments.get("selection_policy_ref") or ""
                ),
                alternative_refs=agenda.alternative_refs,
                descriptive_facts=descriptive_facts,
            )
            return self._projected_agenda(agenda, projected, FrozenMap())
        except (TypeError, ValueError, LookupError) as error:
            raise PureCallBlocked(str(error)) from error

    def project_declared_agenda_delta(
        self,
        context: FunctionCallContext,
    ) -> object:
        try:
            agenda = _interaction_probe_agenda(context.arguments["agenda"])
            reasoning_strategy_ref = str(
                context.arguments.get("reasoning_strategy_ref") or ""
            )
            active_priority_node_ref = str(
                context.arguments.get("active_priority_node_ref") or ""
            )
            dynamic_goal_selection_present = bool(
                context.arguments.get("dynamic_goal_selection_present", False)
            )
            selected_dynamic_goal_ref = str(
                context.arguments.get("selected_dynamic_goal_ref") or ""
            )
            selected_dynamic_goal_facts: Mapping[str, object] = {}
            dynamic_goal_facts = agenda.context_facts.get("dynamic_goal_facts")
            if selected_dynamic_goal_ref and isinstance(dynamic_goal_facts, Mapping):
                selected = dynamic_goal_facts.get(selected_dynamic_goal_ref)
                if isinstance(selected, Mapping):
                    selected_dynamic_goal_facts = selected
            # The selected dynamic goal is already a DRM/SRC decision.  Carry
            # that exact reference through the shared agenda context so the
            # later action boundary cannot silently fall back to the generic
            # information-probe goal.  Python only transports an existing
            # declared ref; it does not originate a cognitive goal.
            selected_dynamic_active_goal_ref = str(
                selected_dynamic_goal_facts.get("active_goal_ref")
                or selected_dynamic_goal_facts.get("goal_ref")
                or selected_dynamic_goal_ref
                or ""
            )
            selected_dynamic_terminal_goal_ref = str(
                selected_dynamic_goal_facts.get("active_terminal_goal_ref")
                or selected_dynamic_active_goal_ref
                or ""
            )
            selected_dynamic_dependency_chain_refs = tuple(
                str(ref)
                for ref in (
                    selected_dynamic_goal_facts.get("goal_dependency_chain_refs")
                    or ()
                )
                if ref
            ) or (
                (selected_dynamic_active_goal_ref,)
                if selected_dynamic_active_goal_ref
                else ()
            )
            frozen_pre_reaction_click_frontier_exhausted = bool(
                context.arguments.get(
                    "frozen_pre_reaction_click_frontier_exhausted", False
                )
            )
            descriptive_context = (
                FrozenMap(
                    {
                        **selected_dynamic_goal_facts,
                        **(
                            {"reasoning_strategy_ref": reasoning_strategy_ref}
                            if reasoning_strategy_ref
                            else {}
                        ),
                        **(
                            {
                                "active_priority_node_ref": active_priority_node_ref,
                                "declared_priority_context_ref": active_priority_node_ref,
                            }
                            if active_priority_node_ref
                            else {}
                        ),
                        "dynamic_goal_selection_present": (
                            dynamic_goal_selection_present
                        ),
                        "selected_dynamic_goal_ref": selected_dynamic_goal_ref,
                        **(
                            {"active_goal_ref": selected_dynamic_active_goal_ref}
                            if selected_dynamic_active_goal_ref
                            else {}
                        ),
                        **(
                            {
                                "active_terminal_goal_ref": (
                                    selected_dynamic_terminal_goal_ref
                                )
                            }
                            if selected_dynamic_terminal_goal_ref
                            else {}
                        ),
                        **(
                            {
                                "goal_dependency_chain_refs": (
                                    selected_dynamic_dependency_chain_refs
                                )
                            }
                            if selected_dynamic_dependency_chain_refs
                            else {}
                        ),
                        "frozen_pre_reaction_click_frontier_exhausted": (
                            frozen_pre_reaction_click_frontier_exhausted
                        ),
                    }
                )
                if (
                    reasoning_strategy_ref
                    or active_priority_node_ref
                    or dynamic_goal_selection_present
                    or selected_dynamic_goal_ref
                    or frozen_pre_reaction_click_frontier_exhausted
                )
                else FrozenMap()
            )
            projected_context = FrozenMap(
                {
                    **agenda.context_facts.to_shallow_dict(),
                    **descriptive_context.to_shallow_dict(),
                }
            )
            selection_policy_ref = str(
                context.arguments.get("selection_policy_ref") or ""
            )
            with profile_span(
                kind="workflow-stage",
                name="declared_agenda:drm_projection",
            ):
                projected = project_declared_alternative_deltas(
                    drm=self._drm,
                    selection_policy_ref=selection_policy_ref,
                    alternative_refs=agenda.alternative_refs,
                    alternative_facts=self._project_declared_canonical_joins(
                        selection_policy_ref, agenda.alternative_refs,
                        agenda.alternative_facts, projected_context, context.frame_id),
                    context_facts=projected_context,
                )
            extension_facts = tuple(
                ChainMap(facts, projected_context)
                for facts in (projected.alternative_facts[__yf_order_key] for __yf_order_key in sorted(projected.alternative_facts))
                if isinstance(facts, Mapping)
                and ChainMap(facts, projected_context).get(
                    "requires_ontology_extension"
                ) is True
            )
            extension_contexts = tuple(
                facts.get("ontology_extension_context")
                for facts in extension_facts
                if facts.get("ontology_extension_context")
            )
            if extension_contexts and any(
                item != extension_contexts[0] for item in extension_contexts[1:]
            ):
                raise PureCallBlocked(
                    "DRM projections produced conflicting ontology extensions"
                )
            # SRC retains the measured candidates in their original binding.
            # Expose to SELECT only the fields read by its declared DRM policy;
            # retain the complete shared context once for later preparation at
            # the action boundary.  Python only reduces immutable transport.
            declaration, _loaded = self._drm.resolve(selection_policy_ref)
            policy_field_refs = _selection_policy_field_refs(declaration)
            selection_context = FrozenMap(
                {
                    field_ref: projected_context[field_ref]
                    for field_ref in policy_field_refs
                    if field_ref in projected_context
                }
            )
            return FrozenMap(
                {
                    "alternative_refs": projected.alternative_refs,
                    "alternative_facts": projected.alternative_facts,
                    "context_facts": projected_context,
                    "selection_context_facts": selection_context,
                    "has_candidates": bool(projected.alternative_refs),
                    "requires_ontology_extension": bool(extension_facts),
                    "ontology_extension_context": (
                        extension_contexts[0] if extension_contexts else FrozenMap()
                    ),
                }
            )
        except (TypeError, ValueError, LookupError) as error:
            raise PureCallBlocked(str(error)) from error

    @staticmethod
    def _projected_agenda(
        agenda: InteractionProbeAgenda,
        projected: object,
        projected_context: FrozenMap,
    ) -> InteractionProbeAgenda:
            extension_facts = tuple(
                ChainMap(facts, projected_context)
                for facts in (projected.alternative_facts[__yf_order_key] for __yf_order_key in sorted(projected.alternative_facts))
                if isinstance(facts, Mapping)
                and ChainMap(facts, projected_context).get(
                    "requires_ontology_extension"
                ) is True
            )
            extension_contexts = tuple(
                facts.get("ontology_extension_context")
                for facts in extension_facts
                if facts.get("ontology_extension_context")
            )
            if extension_contexts and any(
                item != extension_contexts[0] for item in extension_contexts[1:]
            ):
                raise PureCallBlocked(
                    "DRM projections produced conflicting ontology extensions"
                )
            projected_candidates = tuple(
                candidate.model_copy(
                    update={
                        "basis_refs": tuple(
                            str(item)
                            for item in projected.alternative_facts[
                                candidate.candidate_ref
                            ].get("declared_basis_refs", candidate.basis_refs)
                        ),
                        "expectation_refs": tuple(
                            dict.fromkeys(
                                (
                                    *(
                                        str(item)
                                        for item in projected.alternative_facts[
                                            candidate.candidate_ref
                                        ].get("declared_expectation_refs", ())
                                    ),
                                    # Preserve measurement-only path labels from
                                    # capability enumeration (e.g. simulate/continue
                                    # cursor tags for compact SRL projection).
                                    *(
                                        str(item)
                                        for item in candidate.expectation_refs
                                    ),
                                )
                            )
                        ),
                    }
                )
                for candidate in agenda.candidates
            )
            return agenda.model_copy(
                update={
                    "candidates": projected_candidates,
                    "alternative_facts": projected.alternative_facts,
                    "context_facts": projected_context,
                    "requires_ontology_extension": bool(extension_facts),
                    "ontology_extension_context": (
                        extension_contexts[0] if extension_contexts else FrozenMap()
                    ),
                }
            )

    def assessment_selection_present(
        self,
        context: FunctionCallContext,
    ) -> bool:
        """Measure whether one SRC/DRM assessment contains an exact selection."""

        assessment = context.arguments.get("assessment")
        alternatives = (
            assessment.get("alternatives")
            if isinstance(assessment, Mapping)
            else None
        )
        return bool(
            isinstance(alternatives, Mapping)
            and str(alternatives.get("selected") or "")
        )

    def projected_alternative_boolean(self, context: FunctionCallContext) -> bool:
        """Read one strict Boolean from the exact SRC-bound alternative."""

        projection = context.arguments["projection"]
        rows = (
            projection.get("alternative_facts")
            if isinstance(projection, Mapping)
            else getattr(projection, "alternative_facts", None)
        )
        ref = context.arguments["alternative_ref"]
        field = context.arguments["field"]
        row = rows.get(ref) if isinstance(rows, Mapping) else None
        value = row.get(field) if isinstance(row, Mapping) else None
        if not isinstance(value, bool):
            raise PureCallBlocked("bound projected alternative Boolean is absent or invalid")
        return value

    def retain_selected_agenda(
        self,
        context: FunctionCallContext,
    ) -> object:
        """Discard every mechanically enumerated witness except the DRM choice."""

        try:
            agenda = _interaction_probe_agenda(context.arguments["agenda"])
            selected_ref = str(context.arguments.get("selected_ref") or "")
            if selected_ref not in agenda.alternative_refs:
                raise PureCallBlocked("selected agenda ref is not an enumerated witness")
            selected_candidate = next(
                item for item in agenda.candidates if item.candidate_ref == selected_ref
            )
            return agenda.model_copy(
                update={
                    "candidates": (selected_candidate,),
                    "alternative_refs": (selected_ref,),
                    "alternative_facts": FrozenMap(
                        {selected_ref: agenda.alternative_facts[selected_ref]}
                    ),
                    "has_candidates": True,
                }
            )
        except (TypeError, ValueError, LookupError) as error:
            raise PureCallBlocked(str(error)) from error

    def materialize_declared_agenda_frontier(
        self,
        context: FunctionCallContext,
    ) -> object:
        """Freeze the complete order already encoded by one DRM policy."""

        try:
            agenda = _interaction_probe_agenda(context.arguments["agenda"])
            projected = context.arguments.get("projected_agenda")
            if not isinstance(projected, Mapping):
                raise PureCallBlocked("declared frontier requires a projected agenda")
            refs = tuple(str(item) for item in projected.get("alternative_refs", ()))
            raw_facts = projected.get("alternative_facts")
            if not refs or not isinstance(raw_facts, Mapping):
                return agenda.model_copy(
                    update={
                        "candidates": (),
                        "alternative_refs": (),
                        "alternative_facts": FrozenMap(),
                        "has_candidates": False,
                    }
                )
            policy_ref = str(context.arguments.get("selection_policy_ref") or "")
            declaration, _loaded = self._drm.resolve(policy_ref)
            # ``projected_agenda`` may have been reduced for a different
            # selection policy.  Materialization applies ``policy_ref`` and
            # must therefore reduce the complete measured context against the
            # fields declared by this policy, rather than reuse the upstream
            # policy's narrower selection view.
            raw_context = projected.get("context_facts")
            if not isinstance(raw_context, Mapping):
                raw_context = projected.get("selection_context_facts")
            if not isinstance(raw_context, Mapping):
                raw_context = agenda.context_facts
            policy_field_refs = _selection_policy_field_refs(declaration)
            selection_context = FrozenMap(
                {
                    field_ref: raw_context[field_ref]
                    for field_ref in policy_field_refs
                    if field_ref in raw_context
                }
            )
            facts = FrozenMap.from_frozen_items(
                {
                    ref: FrozenMap(
                        ChainMap(
                            raw_facts[ref],
                            selection_context,
                        )
                    )
                    for ref in refs
                }
            )
            frontier_projection = project_declared_alternatives(
                drm=self._drm,
                selection_policy_ref=policy_ref,
                alternative_refs=refs,
                descriptive_facts=facts,
            )
            facts = frontier_projection.alternative_facts
            declared = declared_lexicographic_order(
                SelectionInput(
                    goal_ref=str(context.arguments.get("goal_ref") or ""),
                    alternative_refs=refs,
                    alternative_facts=facts,
                ),
                declaration,
            )
            maximum = int(context.arguments.get("maximum_candidate_count") or 0)
            if maximum <= 0:
                raise PureCallBlocked("declared frontier requires a positive bound")
            if len(declared) > maximum:
                raise PureCallBlocked(
                    "declared frontier exceeded its SRC bound: "
                    f"{len(declared)} > {maximum}"
                )
            ordered_refs = tuple(item[1] for item in declared)
            candidates_by_ref = {
                item.candidate_ref: item for item in agenda.candidates
            }
            ordered_facts = FrozenMap.from_frozen_items(
                {
                    ref: FrozenMap.overlay(
                        facts[ref],
                        FrozenMap(
                            {
                                "declared_frontier_order_index": index,
                                "declared_frontier_source_ref": policy_ref,
                            }
                        ),
                    )
                    for index, ref in enumerate(ordered_refs)
                }
            )
            return agenda.model_copy(
                update={
                    "candidates": tuple(candidates_by_ref[ref] for ref in ordered_refs),
                    "alternative_refs": ordered_refs,
                    "alternative_facts": ordered_facts,
                    "context_facts": FrozenMap.overlay(
                        agenda.context_facts,
                        FrozenMap(
                            {
                                "declared_frontier_origin_refs": ordered_refs,
                                "declared_frontier_source_ref": policy_ref,
                            }
                        ),
                    ),
                    "has_candidates": bool(ordered_refs),
                }
            )
        except PureCallBlocked:
            raise
        except (TypeError, ValueError, LookupError, KeyError) as error:
            raise PureCallBlocked(str(error)) from error

    def project_declared_delta(
        self,
        context: FunctionCallContext,
    ) -> object:
        """Project a small observation delta against immutable prior context."""

        context_facts = context.arguments.get("context_facts")
        descriptive_delta = context.arguments.get("descriptive_delta")
        if not isinstance(context_facts, Mapping) or not isinstance(
            descriptive_delta, Mapping
        ):
            raise PureCallBlocked(
                "declared delta projection requires context and delta mappings"
            )
        try:
            return project_declared_delta(
                drm=self._drm,
                selection_policy_ref=str(
                    context.arguments.get("selection_policy_ref") or ""
                ),
                alternative_refs=tuple(
                    str(item)
                    for item in context.arguments.get("alternative_refs", ())
                ),
                context_facts=context_facts,
                descriptive_delta=descriptive_delta,
            )
        except (TypeError, ValueError, LookupError) as error:
            raise PureCallBlocked(str(error)) from error

    def declared_selection_has_eligible_alternative(self, context: FunctionCallContext) -> bool:
        """Consult the exact SELECT evaluator without adding a fallback candidate."""
        declaration, _ = self._drm.resolve(str(context.arguments["selection_policy_ref"]))
        value = SelectionInput(
            goal_ref=context.arguments["goal_ref"],
            alternative_refs=tuple(context.arguments["alternative_refs"]),
            alternative_facts=context.arguments["alternative_facts"],
        )
        return bool(declared_lexicographic_order(value, declaration))

    def declared_meaning_is_applicable(self, context: FunctionCallContext) -> bool:
        """Evaluate the source-named templates; only an exact empty result yields false."""
        try:
            instantiate_declared_meaning(drm=self._drm,
                selection_policy_ref=str(context.arguments["selection_policy_ref"]),
                selected_ref=str(context.arguments["selected_ref"]),
                alternative_facts=context.arguments["alternative_facts"],
                evidence_refs=tuple(context.arguments["evidence_refs"]),
                state_revision=self._store.snapshot.revision)
        except NoApplicableDeclaredMeaning:
            return False
        return True

    def measure_current_shared_method_inventory(self, context: FunctionCallContext) -> FrozenMap:
        declaration, _ = self._drm.resolve(str(context.arguments["consultation_policy_ref"]))
        profile = declaration.fact_projections[0].static_facts["canonical_inventory_projection"]
        try:
            return measure_current_method_inventory(snapshot=self._store.snapshot,
                tracking=TemporalTrackingResult.model_validate(context.arguments["tracking"]), profile=profile)
        except (TypeError, ValueError, LookupError) as error:
            raise PureCallBlocked(str(error)) from error

    def measure_current_shared_effect_inventory(self, context: FunctionCallContext) -> FrozenMap:
        from agents.yf_arc3_v5.state.shared_effect_inventory import measure_current_effect_inventory
        declaration, _ = self._drm.resolve(str(context.arguments["consultation_policy_ref"]))
        profile = declaration.configuration["canonical_effect_inventory_projection"]
        snapshot = self._store.snapshot
        if not snapshot.terms_for_projection_contract(profile["sample_contract"]):
            return FrozenMap({"effect_inventory_rows": (), "has_inventory_delta": False,
                "evidence_refs": (), "state_revision": snapshot.revision})
        tracking = TemporalTrackingResult.model_validate(context.arguments["tracking"])
        if len(tracking.entities) > profile["maximum_bearers"]:
            raise PureCallBlocked("current effect bearer measurement bound exceeded")
        key_declaration, loaded = self._drm.resolve(profile["appearance_key_policy_ref"])
        key_facts = key_declaration.fact_projections[0].static_facts
        key_contract = AppearanceKeyContract(tiers=key_facts["appearance_key_contract"],
            rotation_quarter_turns=key_facts["rotation_quarter_turns"], source_hash=loaded.source_hash,
            digital_topology_contract=FrozenMap.overlay(FrozenMap({"source_hash": loaded.source_hash}),
                key_facts["digital_topology_contract"]))
        if self._capabilities is None:
            raise PureCallBlocked("current effect appearance requires the production capability registry")
        try:
            appearances = tuple(self._capabilities.invoke(RevisionedCapabilityRequest(
                capability_id="capability.bearer_appearance_measurement", input_revision=snapshot.revision,
                payload=BearerAppearanceInput(bearer_ref=entity.entity_id,
                    valued_pixels=tuple((r, c, entity.component.value) for r, c in entity.component.pixels),
                    measurement_basis_ref=profile["measurement_basis_ref"], cell_quantum=None,
                    key_contract=key_contract))).output for entity in tracking.entities
                if entity.current_frame_ref == tracking.frame_ref)
            return measure_current_effect_inventory(snapshot=snapshot, frame_ref=tracking.frame_ref,
                appearances=appearances, bearer_contexts=self._read_declared_bearer_contexts(
                    key_facts["canonical_index_projection"]["current_context_projection"], frame_ref=tracking.frame_ref),
                profile=profile)
        except (TypeError, ValueError, LookupError) as error:
            raise PureCallBlocked(str(error)) from error

    def measure_current_method_activity_coverage(self, context: FunctionCallContext) -> FrozenMap:
        declaration, _ = self._drm.resolve(str(context.arguments["coverage_policy_ref"]))
        profile = declaration.fact_projections[0].static_facts["canonical_coverage_projection"]
        producer, loaded = self._drm.resolve(profile["producer_policy_ref"])
        projection = next(item for item in producer.fact_projections if item.id == profile["producer_projection_ref"])
        agenda = _interaction_probe_agenda(context.arguments["agenda"])
        try:
            contract = GameMethodActivityContract.model_validate(projection.static_facts[profile["contract_field"]])
            registry_policy, _ = self._drm.resolve(profile["component_registry_policy_ref"])
            component_profile = registry_policy.fact_projections[0].static_facts["canonical_activity_projection"]
            return measure_current_method_coverage(
                snapshot=self._store.snapshot,
                tracking=TemporalTrackingResult.model_validate(context.arguments["tracking"]),
                alternative_facts=agenda.alternative_facts,
                mechanism_inputs=tuple(VerifiedMechanismEvidence.model_validate(item) for item in context.arguments["mechanism_evidence"]),
                profile=profile,
                declared_contract=contract,
                contract_source_hash=loaded.source_hash,
                component_attestations=self._declared_method_component_attestations(contract, component_profile),
                additional_contracts=tuple((
                    field,
                    GameMethodActivityContract.model_validate(projection.static_facts[field]),
                    self._declared_method_component_attestations(GameMethodActivityContract.model_validate(projection.static_facts[field]), component_profile),
                ) for field in profile["additional_contract_fields"]),
                dependency_contracts=tuple(FrozenMap.overlay(FrozenMap({
                    "composition_law_hash": self._drm.resolve(item["composition_principle_ref"])[1].source_hash,
                }), item) for item in component_profile["activity_dependencies"]),
            )
        except (TypeError, ValueError) as error:
            raise PureCallBlocked(str(error)) from error

    def _declared_method_component_attestations(self, contract: GameMethodActivityContract, profile: Mapping) -> tuple[FrozenMap, ...]:
        """Exact bounded registration lookup; DRM declares every correspondence."""
        correspondence = profile["component_workflow_refs"]
        if not frozenset(contract.component_refs).issubset(correspondence):
            return ()
        rows = []
        for component_ref in contract.component_refs:
            workflow_ref = correspondence[component_ref]
            entry = self._entries.get(workflow_ref)
            if entry is None or workflow_ref not in entry.workflow_ids:
                return ()
            rows.append(FrozenMap({
                "component_ref": component_ref, "workflow_definition_ref": workflow_ref,
                "source_hash": entry.source_hash, "semantic_hash": entry.semantic_hash,
                "transitive_source_hash": entry.transitive_source_hash,
            }))
        return tuple(rows)

    def project_current_method_activity_premises(self, context: FunctionCallContext) -> InteractionProbeAgenda:
        """Preserve the frozen binding refs even if one became invalid.

        DRM may reject that exact composition; the action-grounding gate still
        sees its stale premises. Never replace them with an ordinary cold why.
        """
        agenda = _interaction_probe_agenda(context.arguments["agenda"])
        rows = context.arguments["coverage"]["coverage_rows"]
        by_candidate = {}
        for row in rows:
            if self._store.snapshot.term(row["binding_term_ref"]) is not None:
                by_candidate.setdefault(row["candidate_ref"], []).append(row)
        facts = {}
        for reference in agenda.alternative_refs:
            candidate_rows = by_candidate.get(reference, ())
            if not candidate_rows:
                facts[reference] = agenda.alternative_facts[reference]
                continue
            term_refs = tuple(sorted({row["binding_term_ref"] for row in candidate_rows}))
            premise_refs = tuple(sorted({ref for row in candidate_rows for ref in (row["binding_term_ref"], row["binding_claim_ref"])}))
            checks_passed = all(
                self._store.snapshot.term(ref) is not None
                and canonical_reference_bindings_current(
                    self._store.snapshot, self._store.snapshot.term(ref),
                    frame_ref=context.arguments["frame_ref"],
                ) for ref in term_refs
            )
            facts[reference] = FrozenMap.overlay(agenda.alternative_facts[reference], FrozenMap({
                "canonical_method_activity_binding_refs": term_refs,
                "canonical_method_premise_refs": premise_refs,
                "canonical_method_attachment_checks_passed": checks_passed,
                "current_plan_basis_refs": context.arguments["basis_refs"],
            }))
        projected = project_declared_alternative_deltas(
            drm=self._drm, selection_policy_ref=str(context.arguments["plan_policy_ref"]),
            alternative_refs=agenda.alternative_refs, alternative_facts=FrozenMap.from_frozen_items(facts),
            context_facts=agenda.context_facts,
        )
        return self._projected_agenda(agenda, projected, agenda.context_facts)

    def _project_declared_canonical_joins(self, policy_ref, references, alternative_facts,
            context_facts, frame_ref):
        """Project exact proof joins declared by Source, without choosing a goal.

        Existing claims are transported, never synthesized from a reference
        pattern. Empty joins stay explicit; DRM determines their consequence.
        Only matching fact tuples consume the declared join-computation bound.
        """
        policy, loaded = self._drm.resolve(policy_ref)
        profiles = policy.configuration.get("canonical_claim_join_projections", ())
        if not profiles:
            return alternative_facts
        if len(profiles) > 8:
            raise ValueError("canonical join projection count exceeded")
        cache_key = (self._store.snapshot.revision, loaded.source_hash, frame_ref)
        if getattr(self, "_canonical_join_cache_key", None) != cache_key:
            self._canonical_join_cache_key = cache_key
            self._canonical_join_cache = {}
        result = {}
        for reference in references:
            source = alternative_facts[reference]
            facts = ChainMap({"measurement_frame_ref": frame_ref}, source, context_facts)
            delta = {}
            for ordinal, profile in enumerate(profiles):
                refs = ()
                if all(facts.get(field) == expected for field, expected in sorted(profile["fact_equalities"].items())):
                    key = (ordinal, tuple(facts.get(field) for field in profile["input_fields"]))
                    if key not in self._canonical_join_cache:
                        if len(self._canonical_join_cache) >= profile["maximum_computed_joins"]:
                            raise ValueError("canonical agenda join computation bound exceeded")
                        self._canonical_join_cache[key] = self._measure_declared_claim_join(profile["join"], facts) or ()
                    refs = self._canonical_join_cache[key]
                field = profile["output_field"]
                delta[field] = tuple(sorted(set((*delta.get(field, ()), *refs))))
                delta[profile["count_output_field"]] = len(delta[field])
            result[reference] = FrozenMap.overlay(FrozenMap(delta), source)
        return FrozenMap.from_frozen_items(result)

    def _measure_declared_claim_join(self, profile: Mapping, facts: Mapping) -> tuple[str, ...] | None:
        """Bounded relational equality/membership; every premise is Source-authored."""
        snapshot = self._store.snapshot

        def field(claim: Claim, path: str) -> object:
            parts = path.split(".")
            if parts[0] == "term_argument":
                index = int(parts[1])
                value = snapshot.term(claim.arguments[index]) if index < len(claim.arguments) else None
                parts = parts[2:]
                if parts == ["binding_current"]:
                    return value is not None and canonical_reference_bindings_current(
                        snapshot, value, frame_ref=facts.get("measurement_frame_ref"), require_projectable=False)
            else:
                value = claim
            for part in parts:
                if value is None:
                    return None
                if isinstance(value, Mapping):
                    value = value.get(part)
                elif isinstance(value, tuple):
                    index = int(part)
                    value = value[index] if index < len(value) else None
                else:
                    value = getattr(value, part, None)
            return value

        rows = [(FrozenMap(), ())]
        comparisons = 0
        for stage in profile["stages"]:
            candidate_refs_fact = stage.get("candidate_refs_fact", profile.get("candidate_refs_fact"))
            if candidate_refs_fact:
                refs = facts.get(candidate_refs_fact, ())
                if not isinstance(refs, tuple) or len(refs) > profile["max_candidates_per_stage"]:
                    return None
                candidates = tuple(candidate for ref in sorted(set(refs))
                    if (candidate := snapshot.claim(ref)) is not None
                    and candidate.predicate == stage["predicate"])
            else:
                candidates = snapshot.claims_for_predicate(stage["predicate"])
            if len(candidates) > profile["max_candidates_per_stage"]:
                return None
            joined = []
            for bindings, evidence in rows:
                for candidate in candidates:
                    comparisons += 1
                    if comparisons > profile["max_comparisons"]:
                        return None
                    if (
                        candidate.proof_rule not in stage["proof_rules"]
                        or candidate.epistemic_status.value not in stage["statuses"]
                        or candidate.disposition.value != "active"
                        or candidate.polarity.value != "positive"
                        or candidate.active_contradictions or candidate.defeated_by
                    ):
                        continue
                    matches = True
                    for constraint in stage["constraints"]:
                        actual = field(candidate, constraint["field"])
                        operand = constraint["operand"]
                        if "binding" in operand:
                            expected = bindings.get(operand["binding"])
                        elif "fact" in operand:
                            expected = facts.get(operand["fact"])
                        else:
                            expected = operand["literal"]
                        if actual is None or expected is None:
                            matches = False
                        elif constraint["operator"] == "equals":
                            matches = actual == expected
                        elif constraint["operator"] == "contains":
                            matches = isinstance(actual, tuple) and expected in actual
                        elif constraint["operator"] == "member_of":
                            matches = isinstance(expected, tuple) and actual in expected
                        else:
                            raise ValueError("unsupported declared claim-join comparison")
                        if not matches:
                            break
                    if not matches:
                        continue
                    values = FrozenMap({key: field(candidate, path) for key, path in sorted(stage["bindings"].items())})
                    if any(values[key] is None for key in sorted(values)):
                        continue
                    joined.append((FrozenMap.overlay(values, bindings), (*evidence, candidate.id)))
                    if len(joined) > profile["max_rows"]:
                        return None
            rows = joined
            if not rows:
                return None
        return tuple(sorted({ref for _, evidence in rows for ref in evidence}))

    def measure_winning_navigation(self, context: FunctionCallContext) -> FrozenMap:
        from agents.yf_arc3_v5.state.winning_navigation_memory import measure_winning_navigation
        declaration, _ = self._drm.resolve(context.arguments["measurement_policy_ref"])
        key_declaration, key_loaded = self._drm.resolve(context.arguments["key_policy_ref"])
        facts = key_declaration.fact_projections[0].static_facts
        contract = AppearanceKeyContract(tiers=facts["appearance_key_contract"],
            rotation_quarter_turns=facts["rotation_quarter_turns"], source_hash=key_loaded.source_hash,
            digital_topology_contract={**facts["digital_topology_contract"], "source_hash": key_loaded.source_hash})
        return measure_winning_navigation(snapshot=self._store.snapshot, store=self._store,
            repository=self._world_inputs, transition_ref=context.arguments["transition_ref"],
            profile=declaration.configuration["measurement_contract"], key_contract=contract)

    def measure_terminal_outcome_evidence(self, context: FunctionCallContext) -> FrozenMap:
        from agents.yf_arc3_v5.state.terminal_objective_memory import measure_terminal_outcome_evidence
        declaration, _ = self._drm.resolve(context.arguments["measurement_policy_ref"])
        return measure_terminal_outcome_evidence(snapshot=self._store.snapshot,
            store=self._store,
            repository=self._world_inputs,
            before_ref=context.arguments.get("before_raw_input_ref"),
            after_ref=context.arguments.get("after_raw_input_ref"),
            before_frame=context.arguments.get("before_frame"),
            after_frame=context.arguments.get("after_frame"),
            completion_facts=context.arguments["completion_facts"],
            transition_ref=context.arguments["transition_ref"],
            profile=declaration.configuration["measurement_contract"])

    def measure_local_field_type_inventory(self, context: FunctionCallContext) -> FrozenMap:
        from agents.yf_arc3_v5.capabilities.local_field_inventory import ArrayRegionMeasurementRequest
        from agents.yf_arc3_v5.capabilities.contracts import FrameGrid
        declaration, _ = self._drm.resolve(context.arguments["measurement_policy_ref"])
        profile = declaration.configuration["measurement_contract"]
        calibrations = self._store.snapshot.terms_for_projection_contract(profile["calibration_contract_ref"])
        world_context = context.arguments["context"]
        raw_scene = context.arguments["scene"]
        raw_frame = raw_scene.get("frame") if isinstance(raw_scene, Mapping) else None
        facts = {"enumeration_complete": False, "pattern_count": 0,
            "enlarged_matching_count": 0, "adjacent_matching_count": 0,
            "adjacent_inner_component_count": 0, "prior_calibration_count": len(calibrations),
            "prior_calibration_rows": tuple(dict(term.attributes) for term in calibrations),
            "measurement_observation_ref": world_context["id"],
            "measurement_level_ref": world_context.get("level_ref")}
        if raw_frame is not None:
            frame = FrameGrid.model_validate(raw_frame)
            tracking = context.arguments.get("tracking")
            entity_rows = tuple((row["component"]["component_id"], row["entity_id"])
                for row in (tracking.get("entities", ()) if isinstance(tracking, Mapping) else ()))
            request = ArrayRegionMeasurementRequest(frame=frame, component_entity_rows=entity_rows,
                maximum_square_side=profile["maximum_square_side"],
                minimum_array_size=profile["minimum_array_size"], maximum_array_size=profile["maximum_array_size"],
                maximum_band_height=max(1, frame.height // profile["band_height_denominator"]),
                maximum_pattern_rows=profile["maximum_pattern_rows"])
            measurement = self._capabilities.invoke(RevisionedCapabilityRequest(
                capability_id="capability.array_adjacent_region_measurement",
                input_revision=self._store.snapshot.revision, payload=request)).output
            facts.update(measurement.model_dump(mode="python"))
            if measurement.pattern_count == 1 and measurement.enumeration_complete:
                facts.update(dict(measurement.pattern_rows[0]))
        return FrozenMap({"descriptive_facts": FrozenMap(facts),
            "evidence_refs": (world_context["id"], *(term.id for term in calibrations)),
            "state_revision": self._store.snapshot.revision})

    def measure_declared_stock_dependency(self, context: FunctionCallContext) -> FrozenMap:
        from agents.yf_arc3_v5.capabilities.consumptive_dependency import (
            ConsumptiveDependencyRequest, ConsumptiveTemplate, SignatureStockItem,
        )
        facts = context.arguments["inventory_facts"]
        stock = tuple(SignatureStockItem(item_ref=row["item_ref"], signature_ref=row["signature_ref"])
            for row in facts["declared_stock_rows"])
        if facts["declared_recipe_template_order"] != "reverse_neighbor_sequence":
            raise PureCallBlocked("declared recipe order is unsupported")
        templates = tuple(ConsumptiveTemplate(template_ref=f"template:{before}:{after}",
            input_signature_refs=(before,) * facts["declared_recipe_input_count"], output_signature_ref=after)
            for before, after in reversed(facts["declared_recipe_signature_pairs"]))
        recipe_rows = {}
        for term in self._store.snapshot.terms_for_projection_contract(facts["empirical_recipe_projection_contract"]):
            row = dict(term.attributes)
            claim = self._store.snapshot.claim(row["supporting_claim_ref"])
            row.update({"source_claim_present": claim is not None,
                "source_claim_predicate": claim.predicate if claim is not None else None,
                "source_claim_epistemic_status": claim.epistemic_status.value if claim is not None else None,
                "source_claim_disposition": claim.disposition.value if claim is not None else None,
                "source_claim_contradiction_count": len(claim.active_contradictions) if claim is not None else 0,
                "source_claim_defeated_count": len(claim.defeated_by) if claim is not None else 0})
            recipe_rows[term.id] = FrozenMap(row)
        supplied_template_refs = {template.template_ref for template in templates}
        observed_refs = ()
        if recipe_rows:
            qualified = project_declared_alternatives(drm=self._drm,
                selection_policy_ref=facts["empirical_recipe_policy_ref"], alternative_refs=tuple(recipe_rows),
                descriptive_facts=FrozenMap(recipe_rows))
            observed_refs = tuple(qualified.alternative_facts[ref]["template_ref"] for ref in qualified.alternative_refs
                if qualified.alternative_facts[ref].get("declared_empirical_recipe_observed") is True
                and qualified.alternative_facts[ref]["template_ref"] in supplied_template_refs)
        request = ConsumptiveDependencyRequest(stock=stock, templates=templates,
            required_signature_refs=(facts["declared_required_signature_ref"],), observed_template_refs=observed_refs,
            signature_substitution_declared=facts["signature_substitution_declared"],
            virtual_output_prefix="virtual:declared-dependency",
            maximum_states=facts["dependency_maximum_states"], maximum_depth=facts["dependency_maximum_depth"])
        measurement = self._capabilities.invoke(RevisionedCapabilityRequest(
            capability_id="capability.consumptive_dependency_measurement",
            input_revision=self._store.snapshot.revision, payload=request)).output
        present_count = sum(item.signature_ref == facts["declared_required_signature_ref"] for item in stock)
        return FrozenMap({**dict(facts), **measurement.model_dump(mode="python"),
            "required_signature_present_count": present_count, "required_signature_present": present_count > 0})

    def measure_declared_pair_point_geometry(self, context: FunctionCallContext) -> FrozenMap:
        from agents.yf_arc3_v5.capabilities.local_field_pair_geometry import (
            FootprintRow, PairPointGeometryRequest,
        )
        facts = context.arguments["dependency_measurements"]
        declaration, _ = self._drm.resolve(context.arguments["measurement_policy_ref"])
        profile = declaration.configuration["measurement_contract"]
        applications = facts["applications"]
        radii = frozenset(row["positive_footprint_distance"] for row in facts["prior_calibration_rows"]
            if row.get("operational_status") == profile["calibration_status"])
        if len(radii) != 1 or not applications or len(applications[0]["input_item_refs"]) != 2:
            return FrozenMap({"has_candidates": False, "enumeration_complete": False,
                "alternative_refs": (), "descriptive_facts": FrozenMap()})
        rows = tuple(FootprintRow(item_ref=row["item_ref"], signature_ref=row["signature_ref"], bbox=row["bbox"])
            for row in (*facts["declared_stock_rows"], *facts["declared_unmapped_world_rows"]))
        signatures = {row.item_ref: row.signature_ref for row in rows}
        request = PairPointGeometryRequest(rows=rows,
            input_signature_refs=tuple(signatures[ref] for ref in applications[0]["input_item_refs"]),
            region_bbox=facts["declared_world_region_bbox"], radius=next(iter(radii)),
            maximum_point_checks=profile["maximum_point_checks"],
            representative_order=profile["representative_order"])
        measurement = self._capabilities.invoke(RevisionedCapabilityRequest(
            capability_id="capability.pair_point_geometry_measurement",
            input_revision=self._store.snapshot.revision, payload=request)).output
        return FrozenMap({**measurement.model_dump(mode="python"),
            "has_candidates": bool(measurement.candidate_rows) and measurement.enumeration_complete,
            "alternative_refs": tuple(row["candidate_ref"] for row in measurement.candidate_rows),
            "descriptive_facts": FrozenMap({row["candidate_ref"]: FrozenMap({**dict(row),
                "enumeration_complete": measurement.enumeration_complete,
                "measurement_observation_ref": facts["measurement_observation_ref"],
                "measurement_level_ref": facts["measurement_level_ref"],
                "measurement_workflow_instance_ref": context.workflow_instance_id,
                "before_stock_rows": facts["declared_stock_rows"],
                "before_other_rows": facts["declared_unmapped_world_rows"],
                "before_stock_lineage_rows": facts["declared_stock_lineage_rows"],
                "expected_template_ref": applications[0]["template_ref"],
                "expected_output_signature_ref": applications[0]["output_signature_ref"],
                "supplied_radius": request.radius,
                "first_template_observed": facts["first_template_observed"]})
                for row in measurement.candidate_rows})})

    def bind_declared_pair_point_agenda(self, context: FunctionCallContext) -> object:
        from agents.yf_arc3_v5.capabilities.contracts import InteractionProbeAgenda, InteractionProbeCandidate
        projected = context.arguments["projected"]
        rows = projected["alternative_facts"]
        refs = tuple(projected["alternative_refs"])
        candidates = tuple(InteractionProbeCandidate(candidate_ref=ref,
            action_ref=rows[ref]["action_ref"], point=rows[ref]["point"],
            action_data=FrozenMap({"x": rows[ref]["point"][1], "y": rows[ref]["point"][0]}),
            basis_refs=tuple(rows[ref]["premise_claim_refs"]),
            expectation_refs=tuple(rows[ref]["expectation_refs"])) for ref in refs)
        return InteractionProbeAgenda(candidates=candidates, alternative_refs=refs,
            alternative_facts=rows, has_candidates=bool(candidates)).model_dump(mode="python")

    def measure_current_pair_stock_change(self, context: FunctionCallContext) -> FrozenMap:
        from agents.yf_arc3_v5.capabilities.local_field_pair_geometry import FootprintRow
        from agents.yf_arc3_v5.capabilities.local_field_stock_change import StockChangeRequest
        from agents.yf_arc3_v5.logos.operations import EnvironmentActionDispatchedRecord
        declaration, _ = self._drm.resolve(context.arguments["measurement_policy_ref"])
        profile = declaration.configuration["measurement_contract"]
        facts = context.arguments["inventory_facts"]
        pointer = self._store.snapshot.term(profile["active_experiment_term_ref"])
        observed = self._store.latest_artifact(ActionObservedRecord)
        result = {"active_experiment_present": pointer is not None,
            "current_observation_ref": facts["measurement_observation_ref"],
            "action_intent_matches": False, "observed_frame_matches": False,
            "release_permit_matches": False, "dispatch_record_matches": False,
            "observation_changed": False}
        evidence = [facts["measurement_observation_ref"]]
        if pointer is not None:
            attrs = pointer.attributes
            result.update(dict(attrs))
            result["observation_changed"] = attrs["before_observation_ref"] != facts["measurement_observation_ref"]
            evidence.append(pointer.id)
            evidence.append(attrs["experiment_ref"])
            if observed is not None:
                permit = self._store.artifact(observed.action_release_permit_id)
                dispatch = self._store.latest_artifact(EnvironmentActionDispatchedRecord)
                executed_intent = self._store.artifact(observed.action_intent_id)
                result.update({
                    "action_intent_matches": isinstance(executed_intent, ActionIntent)
                        and (executed_intent.id == attrs["expected_intent_ref"]
                            or attrs["expected_intent_ref"] in executed_intent.provenance),
                    "observed_frame_matches": context.frame_id is not None and observed.frame_id == context.frame_id,
                    "release_permit_matches": isinstance(permit, ActionReleasePermit)
                        and permit.action_intent_id == observed.action_intent_id,
                    "dispatch_record_matches": dispatch is not None
                        and dispatch.action_intent_id == observed.action_intent_id
                        and dispatch.action_release_permit_id == observed.action_release_permit_id,
                    "observation_record_ref": observed.id,
                })
                evidence.append(observed.id)
            if facts.get("enumeration_complete") and facts.get("pattern_count") == 1:
                request = StockChangeRequest(
                    before_rows=tuple(FootprintRow(item_ref=row["item_ref"],signature_ref=row["signature_ref"],bbox=row["bbox"]) for row in attrs["before_stock_rows"]),
                    after_rows=tuple(FootprintRow(item_ref=row["item_ref"],signature_ref=row["signature_ref"],bbox=row["bbox"]) for row in facts["adjacent_matching_rows"]),
                    before_other_rows=tuple(FootprintRow(item_ref=row["item_ref"],signature_ref=row["signature_ref"],bbox=row["bbox"]) for row in attrs["before_other_rows"]),
                    after_other_rows=tuple(FootprintRow(item_ref=row["item_ref"],signature_ref=row["signature_ref"],bbox=row["bbox"]) for row in facts["adjacent_other_rows"]),
                    participant_refs=attrs["participant_refs"], output_signature_ref=attrs["expected_output_signature_ref"],
                    point=attrs["point"], before_observation_ref=attrs["before_observation_ref"],
                    after_observation_ref=facts["measurement_observation_ref"])
                measurement = self._capabilities.invoke(RevisionedCapabilityRequest(
                    capability_id="capability.stock_change_measurement", input_revision=self._store.snapshot.revision,
                    payload=request)).output
                result.update(dict(measurement.descriptive_facts))
                lineage_by_ref = {row["item_ref"]: row for row in attrs["before_stock_lineage_rows"]}
                result["parent_occurrence_refs"] = tuple(lineage_by_ref[ref]["declared_current_occurrence_ref"]
                    for ref in attrs["participant_refs"])
                leaf_refs = tuple(leaf for ref in attrs["participant_refs"]
                    for leaf in lineage_by_ref[ref]["declared_leaf_occurrence_refs"])
                result["parent_leaf_sets_disjoint"] = len(set(leaf_refs)) == len(leaf_refs)
                result["combined_leaf_occurrence_refs"] = tuple(sorted(set(leaf_refs)))
                result["output_occurrence_refs"] = tuple(
                    f"occurrence:{facts['measurement_observation_ref']}:{row['item_ref']}:{row['signature_ref']}"
                    for row in result["residual_rows"])
                result["residual_rows"] = tuple(FrozenMap({**dict(row), "occurrence_ref": ref})
                    for row,ref in zip(result["residual_rows"],result["output_occurrence_refs"]))
        return FrozenMap({"descriptive_facts": FrozenMap(result), "evidence_refs": tuple(evidence),
            "state_revision": self._store.snapshot.revision})

    def bind_current_stock_lineage(self, context: FunctionCallContext) -> FrozenMap:
        facts = context.arguments["inventory_facts"]
        rows = {}
        for body in facts["declared_stock_rows"]:
            prior = self._store.snapshot.term(
                f"{facts['stock_lineage_term_prefix']}{facts['measurement_level_ref']}.{body['item_ref']}")
            attrs = prior.attributes if prior is not None else FrozenMap()
            claim = self._store.snapshot.claim(attrs.get("supporting_claim_ref", ""))
            rows[body["item_ref"]] = FrozenMap({**dict(body),
                "measurement_level_ref": facts["measurement_level_ref"],
                "prior_lineage_present": prior is not None,
                "prior_signature_matches": attrs.get("signature_ref") == body["signature_ref"],
                "prior_occurrence_ref": attrs.get("occurrence_ref"),
                "prior_leaf_refs": attrs.get("leaf_occurrence_refs", ()),
                "source_claim_present": claim is not None,
                "source_claim_predicate": claim.predicate if claim is not None else None,
                "source_claim_epistemic_status": claim.epistemic_status.value if claim is not None else None,
                "source_claim_disposition": claim.disposition.value if claim is not None else None,
                "source_claim_contradiction_count": len(claim.active_contradictions) if claim is not None else 0,
                "source_claim_defeated_count": len(claim.defeated_by) if claim is not None else 0})
        projected = project_declared_alternatives(drm=self._drm,
            selection_policy_ref=facts["stock_lineage_policy_ref"], alternative_refs=tuple(rows),
            descriptive_facts=FrozenMap(rows))
        bound = tuple(projected.alternative_facts[ref] for ref in projected.alternative_refs
            if projected.alternative_facts[ref].get("declared_lineage_binding_present") is True)
        leaves = tuple(leaf for row in bound for leaf in row["declared_leaf_occurrence_refs"])
        return FrozenMap({**dict(facts), "declared_stock_lineage_rows": bound,
            "lineage_bindings_complete": len(bound) == len(rows) and len(leaves) == len(set(leaves))})

    def measure_current_local_field_delivery(self, context: FunctionCallContext) -> FrozenMap:
        from agents.yf_arc3_v5.capabilities.local_field_delivery import FilledContourRequest, PointProgressRequest
        from agents.yf_arc3_v5.capabilities.local_field_pair_geometry import FootprintRow
        from agents.yf_arc3_v5.capabilities.contracts import FrameGrid
        declaration, _ = self._drm.resolve(context.arguments["measurement_policy_ref"])
        profile = declaration.configuration["measurement_contract"]
        facts = context.arguments["inventory_facts"]
        scene = context.arguments["scene"]
        current_frame = FrameGrid.model_validate(scene["frame"])
        memories = self._store.snapshot.terms_for_projection_contract(profile["terminal_memory_contract"])
        episode_rows = {}
        for memory in memories:
            attrs = memory.attributes
            contract = attrs.get("executed_action_contract", {})
            intent = self._store.artifact(contract.get("action_intent_id", ""))
            data = intent.action_data if isinstance(intent, ActionIntent) else FrozenMap()
            episode_rows[memory.id] = FrozenMap({"official_boundary_present": attrs.get("official_boundary_present"),
                "before_frame_present": attrs.get("before_frame_present"),
                "frozen_action_ref": intent.action_ref if isinstance(intent, ActionIntent) else None,
                "point_payload_complete": type(data.get("x")) is int and type(data.get("y")) is int})
        if not episode_rows:
            return FrozenMap({"has_candidates": False,"alternative_refs": (),"descriptive_facts": FrozenMap(),
                "evidence_refs": (facts["measurement_observation_ref"],)})
        qualified = project_declared_alternatives(drm=self._drm,
            selection_policy_ref=profile["episode_policy_ref"],alternative_refs=tuple(episode_rows),
            descriptive_facts=FrozenMap(episode_rows))
        bodies = tuple(row for row in facts["declared_stock_rows"]
            if row["signature_ref"] == facts["declared_required_signature_ref"])
        radii = frozenset(row["positive_footprint_distance"] for row in facts["prior_calibration_rows"]
            if row.get("operational_status") == profile["calibration_status"])
        empty = FrozenMap({"has_candidates": False, "alternative_refs": (), "descriptive_facts": FrozenMap(),
            "evidence_refs": (facts["measurement_observation_ref"],)})
        if len(bodies) != 1 or len(radii) != 1:
            return empty
        matches = []
        previous_delivery = self._store.snapshot.term(profile["active_delivery_term_ref"])
        previous_recipient_bbox = (
            previous_delivery.attributes.get("recipient_bbox")
            if previous_delivery is not None
            and previous_delivery.attributes.get("measurement_level_ref") == facts["measurement_level_ref"]
            else None
        )
        for memory_ref in qualified.alternative_refs:
            if qualified.alternative_facts[memory_ref].get("declared_episode_admitted") is not True:
                continue
            memory = self._store.snapshot.term(memory_ref)
            attrs = memory.attributes
            contract = attrs.get("executed_action_contract", {})
            intent = self._store.artifact(contract.get("action_intent_id", ""))
            data = intent.action_data
            measurement = self._capabilities.invoke(RevisionedCapabilityRequest(
                capability_id="capability.filled_contour_correspondence",input_revision=self._store.snapshot.revision,
                payload=FilledContourRequest(before_frame=FrameGrid(rows=attrs["before_frame_rows"]),
                    current_frame=current_frame,before_point=(data["y"],data["x"]),
                    current_component_refs=tuple(row["component_ref"] for row in facts["declared_unmapped_world_rows"]),
                    current_occluder_bboxes=tuple(row["bbox"] for row in facts["declared_stock_rows"]),
                    recalled_current_bbox=previous_recipient_bbox,
                    maximum_stencil_pixels=profile["maximum_stencil_pixels"],
                    minimum_visible_fraction_numerator=profile["minimum_visible_fraction_numerator"],
                    minimum_visible_fraction_denominator=profile["minimum_visible_fraction_denominator"]))).output
            if not measurement.enumeration_complete:
                return empty
            for row in measurement.current_match_rows:
                matches.append((memory.id, row))
        if len(matches) != profile["required_recipient_count"]:
            return empty
        memory_ref, matched = matches[0]
        body = bodies[0]
        prior = self._store.snapshot.term(profile["active_delivery_term_ref"])
        prior_scope_matches = prior is not None and prior.attributes.get("measurement_level_ref") == facts["measurement_level_ref"]
        current_before = prior_scope_matches and prior.attributes.get("before_observation_ref") == facts["measurement_observation_ref"]
        observed = self._store.latest_artifact(ActionObservedRecord)
        proof_matches = False
        progressed = False
        stock_unchanged = False
        recipient_unchanged = False
        if prior_scope_matches and observed is not None:
            executed = self._store.artifact(observed.action_intent_id)
            permit = self._store.artifact(observed.action_release_permit_id)
            from agents.yf_arc3_v5.logos.operations import EnvironmentActionDispatchedRecord
            dispatch = self._store.latest_artifact(EnvironmentActionDispatchedRecord)
            proof_matches = isinstance(executed, ActionIntent) and isinstance(permit, ActionReleasePermit) and dispatch is not None
            proof_matches = proof_matches and prior.attributes["expected_intent_ref"] in (executed.id,*executed.provenance)
            proof_matches = proof_matches and observed.frame_id == context.frame_id and context.frame_id is not None
            proof_matches = proof_matches and permit.action_intent_id == dispatch.action_intent_id == observed.action_intent_id
            proof_matches = proof_matches and permit.id == dispatch.action_release_permit_id
            old = prior.attributes["before_body_row"]
            def gap(box, point):
                center=((box[0]+box[2])/2,(box[1]+box[3])/2)
                return sum((a-b)**2 for a,b in zip(center,point))
            progressed = old["item_ref"] == body["item_ref"] and old["signature_ref"] == body["signature_ref"]
            progressed = progressed and gap(body["bbox"],matched["center"]) < gap(old["bbox"],matched["center"])
            now = {row["item_ref"]: row for row in facts["declared_stock_rows"]}
            stock_unchanged = all(now.get(row["item_ref"]) == row for row in prior.attributes["before_stock_rows"]
                if row["item_ref"] != old["item_ref"])
            recipient_unchanged = matched["bbox"] == prior.attributes["recipient_bbox"] and matched["outer_contour_corresponds"]
        excluded = tuple(FootprintRow(item_ref=row["item_ref"],signature_ref=row["signature_ref"],bbox=row["bbox"])
            for row in (*facts["declared_stock_rows"],*facts["declared_unmapped_world_rows"])
            if row["item_ref"] != body["item_ref"] and row.get("component_ref") != matched["component_ref"])
        measurement = self._capabilities.invoke(RevisionedCapabilityRequest(
            capability_id="capability.point_progress_measurement",input_revision=self._store.snapshot.revision,
            payload=PointProgressRequest(body=FootprintRow(item_ref=body["item_ref"],signature_ref=body["signature_ref"],bbox=body["bbox"]),
                excluded_rows=excluded,region_bbox=facts["declared_world_region_bbox"],supplied_point=matched["center"],
                radius=next(iter(radii)),maximum_point_checks=profile["maximum_point_checks"]))).output
        rows = FrozenMap({row["candidate_ref"]: FrozenMap({**dict(row),
            "enumeration_complete": measurement.enumeration_complete,
            "measurement_observation_ref": facts["measurement_observation_ref"],
            "measurement_level_ref": facts["measurement_level_ref"],
            "measurement_workflow_instance_ref": context.workflow_instance_id,
            "required_signature_ref": facts["declared_required_signature_ref"],
            "recipient_component_ref": matched["component_ref"],"recipient_bbox": matched["bbox"],
            "recipient_center": matched["center"],"terminal_memory_ref": memory_ref,
            "outer_contour_corresponds": matched["outer_contour_corresponds"],
            "correspondence_under_body_occlusion": matched["correspondence_under_body_occlusion"],
            "filled_contour_pixel_count": matched["filled_contour_pixel_count"],
            "visible_corresponding_pixel_count": matched["visible_corresponding_pixel_count"],
            "corresponding_recipient_count": len(matches),
            "prior_delivery_scope_matches": prior_scope_matches,"prior_delivery_same_observation": current_before,
            "delivery_action_conditioned": proof_matches,"delivery_progress_measured": progressed,
            "delivery_other_stock_unchanged": stock_unchanged,"delivery_recipient_unchanged": recipient_unchanged,
            "before_stock_rows": facts["declared_stock_rows"],
            "before_body_row": body}) for row in measurement.candidate_rows})
        return FrozenMap({"has_candidates": bool(rows) and measurement.enumeration_complete,
            "alternative_refs": tuple(rows),"descriptive_facts": rows,
            "evidence_refs": (facts["measurement_observation_ref"],memory_ref)})

    def measure_reference_write_evidence(self, context: FunctionCallContext) -> FrozenMap:
        from agents.yf_arc3_v5.state.reference_write_evidence import measure_reference_write_evidence
        declaration, _ = self._drm.resolve(context.arguments["measurement_policy_ref"])
        return measure_reference_write_evidence(snapshot=self._store.snapshot,
            repository=self._world_inputs, scene=context.arguments.get("scene"),
            outcome=context.arguments["outcome"],
            evidence_refs=context.arguments["evidence_refs"],
            profile=declaration.configuration["measurement_contract"])

    def measure_terminal_coverage_evidence(self, context: FunctionCallContext) -> FrozenMap:
        from agents.yf_arc3_v5.state.terminal_objective_memory import measure_terminal_coverage_evidence
        declaration, _ = self._drm.resolve(context.arguments["measurement_policy_ref"])
        return measure_terminal_coverage_evidence(snapshot=self._store.snapshot,
            repository=self._world_inputs, store=self._store,
            scene=context.arguments.get("scene"), action_ref=context.arguments["action_ref"],
            frame_rows=context.arguments.get("frame_rows"),
            context_epoch=context.arguments.get("context_epoch"),
            transition_ref=context.arguments["transition_ref"],
            profile=declaration.configuration["measurement_contract"])

    def measure_recalled_navigation_lattices(self, context: FunctionCallContext) -> FrozenMap:
        from agents.yf_arc3_v5.state.navigation_lattice_memory import measure_recalled_navigation_lattices
        declaration, _ = self._drm.resolve(context.arguments["measurement_policy_ref"])
        return measure_recalled_navigation_lattices(snapshot=self._store.snapshot,
            tracking=TemporalTrackingResult.model_validate(context.arguments["tracking"]),
            repository=self._world_inputs, store=self._store,
            profile=declaration.configuration["measurement_contract"])

    def measure_typed_cell_lattices(self, context: FunctionCallContext) -> FrozenMap:
        from dataclasses import fields
        from agents.yf_arc3_v5.capabilities.typed_support_lattice import measure_typed_support_lattices
        declaration, _ = self._drm.resolve(context.arguments["measurement_policy_ref"])
        profile = declaration.configuration["measurement_contract"]
        frame_ref = str(context.arguments["observation_ref"])
        current = self._store.snapshot.terms_for_literal_attribute("operational_scene_ref", frame_ref)
        bound = any(term.attributes.get("projection_contract") == profile["current_grid_contract"]
            and canonical_reference_bindings_current(self._store.snapshot, term, frame_ref=frame_ref) for term in current)
        scene_payload = context.arguments.get("scene")
        support_family_unmeasured = bool(
            isinstance(scene_payload, Mapping)
            and scene_payload.get("schema_version") == "yf_arc3_v5.block_grid_scene_description.v1"
        )
        if scene_payload is None or bound or support_family_unmeasured:
            return FrozenMap({"has_measurements": False, "grid_rows": (), "enumeration_complete": bound,
                "has_unrecorded_measurements": False,
                "evidence_refs": (), "state_revision": self._store.snapshot.revision})
        scene = VisualSceneDescription.model_validate(scene_payload)
        measurement = measure_typed_support_lattices(scene.frame, scene.hole_sprites,
            maximum_cell_extent=profile["maximum_cell_extent"],
            maximum_candidates=profile["maximum_candidates"])
        frame_token = stable_digest(frame_ref)[:20]
        rows = []
        unrecorded_measurement_count = 0
        for candidate in measurement.candidates:
            geometry = (candidate.recount.grid_ref, candidate.row_offset, candidate.col_offset,
                candidate.cell_extent, candidate.cell_extent, candidate.cell_extent, candidate.cell_extent,
                candidate.logical_rows, candidate.logical_columns)
            grid_token = stable_digest(geometry)[:20]
            measurement_ref = profile["measurement_claim_pattern"].format(
                frame_token=frame_token, grid_token=grid_token)
            if self._store.snapshot.claim(measurement_ref) is None:
                unrecorded_measurement_count += 1
            proof_ref = profile["claim_pattern"].format(frame_token=frame_token, grid_token=grid_token)
            rows.append(FrozenMap({"frame_ref": frame_ref, "frame_token": frame_token,
                "grid_token": grid_token, "grid_geometry": geometry,
                "row_claim_refs": (proof_ref,), "grid_claim_ref": proof_ref,
                "repeated_multivalue_support_count": candidate.repeated_multivalue_support_count,
                "repeated_monochrome_support_count": candidate.repeated_monochrome_support_count,
                "type_patterns": candidate.recount.type_patterns,
                "cell_type_ids": candidate.recount.cell_type_ids,
                "cell_support_rows": tuple(FrozenMap({field.name: getattr(item, field.name)
                    for field in fields(item)}) for item in candidate.supports),
                "uncovered_bottom_extent": candidate.uncovered_bottom_extent,
                "uncovered_right_extent": candidate.uncovered_right_extent}))
        return FrozenMap({"has_measurements": bool(rows), "grid_rows": tuple(rows),
            "has_unrecorded_measurements": unrecorded_measurement_count > 0,
            "enumeration_complete": not measurement.enumeration_truncated,
            "evidence_refs": (frame_ref,), "state_revision": self._store.snapshot.revision})

    def measure_logical_command_steps(self, context: FunctionCallContext) -> FrozenMap:
        from agents.yf_arc3_v5.state.command_step_measurements import measure_logical_step_records
        declaration, _ = self._drm.resolve(context.arguments["measurement_policy_ref"])
        return measure_logical_step_records(snapshot=self._store.snapshot,
            tracking=(TemporalTrackingResult.model_validate(context.arguments["tracking"])
                if context.arguments["tracking"] is not None else None),
            profile=declaration.configuration["measurement_contract"])

    def measure_current_terminal_inventory(self, context: FunctionCallContext) -> FrozenMap:
        from agents.yf_arc3_v5.state.terminal_objective_memory import measure_current_terminal_inventory
        declaration, _ = self._drm.resolve(context.arguments["consultation_policy_ref"])
        return measure_current_terminal_inventory(snapshot=self._store.snapshot,
            tracking=TemporalTrackingResult.model_validate(context.arguments["tracking"]),
            profile=declaration.configuration["measurement_contract"], drm=self._drm)

    def measure_canonical_activity_support(self, context: FunctionCallContext) -> FrozenMap:
        """Exact bounded join of a Source contract and its committed proof.

        Python reports equality/membership only. DRM declares the admissible
        producer and proof; SRC decides whether to record the activity. An
        absent or malformed optional contract does not undo reconciliation.
        """
        empty = FrozenMap({"canonical_support_present": False, "evidence_refs": ()})
        declaration, _ = self._drm.resolve(str(context.arguments["admission_policy_ref"]))
        profile = declaration.fact_projections[0].static_facts["canonical_activity_projection"]
        producer, loaded = self._drm.resolve(profile["producer_policy_ref"])
        projection = next(
            item for item in producer.fact_projections
            if item.id == profile["producer_projection_ref"]
        )
        facts = context.arguments["activity_facts"]
        contract_facts = context.arguments.get("contract_facts")
        if contract_facts is None:
            contract_facts = facts
        if not isinstance(contract_facts, Mapping):
            return empty
        if any(contract_facts.get(field) != expected
               for field, expected in sorted(profile.get("required_contract_fact_values", {}).items())):
            return empty
        supplied = contract_facts.get(profile["contract_field"])
        if supplied is None:
            return empty
        try:
            contract = GameMethodActivityContract.model_validate(supplied)
            declared = GameMethodActivityContract.model_validate(
                projection.static_facts[profile["contract_field"]]
            )
        except (TypeError, ValueError):
            return empty
        if contract != declared:
            return empty
        component_attestations = self._declared_method_component_attestations(contract, profile)
        if not component_attestations:
            return empty
        candidates = context.arguments["canonical_claim_candidates"]
        if len(candidates) > profile["max_candidates"]:
            return empty
        transition_ref = context.arguments["transition_ref"]
        expected_id = profile["claim_id_pattern"].format(transition_ref=transition_ref)
        expected_arguments = tuple(facts.get(field) for field in profile["argument_fields"])
        if any(value is None for value in expected_arguments):
            return empty
        current_refs = frozenset(
            item.id for item in self._store.snapshot.claims_for_predicate(profile["link_predicate"])
        )
        support = []
        for candidate in candidates:
            try:
                submitted = Claim.model_validate(candidate)
                status_attribute = profile["candidate_status_attribute"]
                declared_status = submitted.attributes.get(status_attribute)
                if declared_status is not None:
                    attributes = submitted.attributes.to_shallow_dict()
                    attributes.pop(status_attribute)
                    normalized = submitted.model_dump()
                    normalized["epistemic_status"] = declared_status
                    normalized["attributes"] = FrozenMap(attributes)
                    submitted = Claim.model_validate(normalized)
            except (TypeError, ValueError):
                continue
            canonical = self._store.snapshot.claim(submitted.id)
            if canonical is None or canonical != submitted or canonical.id not in current_refs:
                continue
            if (
                canonical.id != expected_id
                or canonical.predicate != profile["link_predicate"]
                or canonical.proof_rule not in profile.get("proof_rules", (profile["proof_rule"],))
                or canonical.arguments != expected_arguments
                or canonical.epistemic_status.value not in profile["allowed_link_statuses"]
                or canonical.disposition.value != profile["link_disposition"]
                or canonical.polarity.value != profile["link_polarity"]
                or (profile["no_contradictions"] and canonical.active_contradictions)
                or (profile["no_defeaters"] and canonical.defeated_by)
            ):
                continue
            support.append(canonical.id)
        if not support:
            return empty
        premise_join = profile.get("premise_join")
        if premise_join is not None:
            premise_facts = contract_facts if profile.get("premise_facts_source") == "contract_facts" else facts
            premise_fields = profile.get("premise_activity_fact_fields", ())
            if premise_fields:
                premise_facts = FrozenMap.overlay(FrozenMap({field: facts.get(field)
                    for field in sorted(premise_fields)}), premise_facts)
            joined_refs = self._measure_declared_claim_join(premise_join, premise_facts)
            if not joined_refs:
                return empty
            support.extend(joined_refs)
        criterion_roles, criterion_associations, criterion_evidence = (), (), ()
        criterion = profile.get("optional_control_criterion")
        if criterion is not None:
            criterion_facts = contract_facts if criterion.get("facts_source") == "contract_facts" else facts
            criterion_evidence = self._measure_declared_claim_join(criterion["premise_join"], criterion_facts) or ()
            criterion_associations = tuple(sorted({
                self._store.snapshot.claim(ref).arguments[criterion["association_argument_index"]]
                for ref in criterion_evidence
                if self._store.snapshot.claim(ref).predicate == criterion["association_predicate"]
            }))
            if criterion_associations:
                criterion_roles = tuple(criterion["role_refs"])
        serialized = FrozenMap(contract.model_dump(mode="json"))
        primary_row = FrozenMap({
            "canonical_support_present": True,
            "method_ref": contract.method_ref,
            "activity_ref": contract.activity_ref,
            "activity_contract": serialized,
            "contract_source_hash": loaded.source_hash,
            "component_source_attestations": component_attestations,
            "observed_control_role_refs": criterion_roles,
            "control_criterion_association_refs": criterion_associations,
            "control_criterion_evidence_refs": criterion_evidence,
            "control_criterion_present": bool(criterion_associations),
            "contract_token": stable_digest((serialized, loaded.source_hash, component_attestations)),
            "transition_ref": transition_ref,
            "evidence_refs": tuple(sorted(set(support))),
            "state_revision": self._store.snapshot.revision,
        })
        activity_rows = [primary_row]
        rows_by_field = {profile["contract_field"]: primary_row}
        contracts_by_field = {profile["contract_field"]: contract}
        for additional in profile["additional_activity_profiles"]:
            field = additional["contract_field"]
            try:
                extra = GameMethodActivityContract.model_validate(facts.get(field))
                source_extra = GameMethodActivityContract.model_validate(projection.static_facts[field])
            except (TypeError, ValueError):
                continue
            if extra != source_extra or extra.method_ref != contract.method_ref:
                continue
            attestations = self._declared_method_component_attestations(extra, profile)
            if not attestations:
                continue
            proofs = self._store.snapshot.claims_for_predicate(additional["proof_predicate"])
            if len(proofs) > additional["max_proof_links"]:
                return empty
            arguments = tuple(facts.get(name) for name in additional["argument_fields"])
            proof_refs = tuple(sorted(proof.id for proof in proofs if (
                proof.arguments == arguments and proof.proof_rule == additional["proof_rule"]
                and proof.epistemic_status.value in additional["allowed_link_statuses"]
                and proof.disposition.value == profile["link_disposition"]
                and proof.polarity.value == profile["link_polarity"]
                and not (profile["no_contradictions"] and proof.active_contradictions)
                and not (profile["no_defeaters"] and proof.defeated_by)
            )))
            if not proof_refs:
                continue
            extra_serialized = FrozenMap(extra.model_dump(mode="json"))
            row = FrozenMap({
                "method_ref": extra.method_ref, "activity_ref": extra.activity_ref,
                "activity_contract": extra_serialized, "contract_source_hash": loaded.source_hash,
                "component_source_attestations": attestations,
                "observed_control_role_refs": (),
                "control_criterion_association_refs": (),
                "control_criterion_evidence_refs": (),
                "control_criterion_present": False,
                "contract_token": stable_digest((extra_serialized, loaded.source_hash, attestations)),
                "transition_ref": transition_ref, "evidence_refs": tuple(sorted(set(support).union(proof_refs))),
            })
            activity_rows.append(row)
            rows_by_field[field] = row
            contracts_by_field[field] = extra
            if len(activity_rows) > profile["max_activity_rows"]:
                return empty
        retained_term_refs = []
        for retained in profile.get("retained_activity_profiles", ()):
            retained_producer, _ = self._drm.resolve(retained["producer_policy_ref"])
            retained_projection = next(item for item in retained_producer.fact_projections
                if item.id == retained["producer_projection_ref"])
            expected_contract = GameMethodActivityContract.model_validate(
                retained_projection.static_facts[retained["source_contract_field"]])
            terms = self._store.snapshot.terms_for_projection_contract(retained["term_contract"])
            links = self._store.snapshot.claims_for_predicate(retained["link_predicate"])
            if len(terms) > retained["max_terms"] or len(links) > retained["max_links"]:
                return empty
            if any(len(link.grounds) > retained["max_proof_grounds"] for link in links):
                return empty

            def admitted_retained(proof):
                return (proof is not None
                    and proof.epistemic_status.value in retained["allowed_statuses"]
                    and proof.disposition.value == profile["link_disposition"]
                    and proof.polarity.value == profile["link_polarity"]
                    and not proof.active_contradictions and not proof.defeated_by)

            matches = []
            for term in sorted(terms, key=lambda item: item.id):
                attrs = term.attributes
                try:
                    retained_contract = GameMethodActivityContract.model_validate(attrs["activity_contract"])
                except (KeyError, TypeError, ValueError):
                    continue
                if retained_contract != expected_contract or retained_contract.method_ref != contract.method_ref:
                    continue
                serialized_retained = FrozenMap(retained_contract.model_dump(mode="json"))
                if not attrs.get("contract_source_hash") or not attrs.get("component_source_attestations"):
                    continue
                token = stable_digest((serialized_retained, attrs["contract_source_hash"],
                    attrs["component_source_attestations"]))
                if term.id != retained["term_id_pattern"].format(
                        activity_ref=retained_contract.activity_ref, contract_token=token):
                    continue
                proofs = tuple(link for link in sorted(links, key=lambda item: item.id)
                    if link.arguments == (contract.method_ref, term.id)
                    and link.proof_rule == retained["link_rule"] and admitted_retained(link)
                    and link.grounds and all(admitted_retained(self._store.snapshot.claim(ref)) for ref in link.grounds))
                if not proofs:
                    continue
                matches.append((retained_contract, FrozenMap({
                    "term_ref": term.id,
                    "activity_ref": retained_contract.activity_ref, "contract_token": token,
                    "evidence_refs": tuple(sorted({ref for proof in proofs for ref in (proof.id, *proof.grounds)})),
                })))
            # No temporal ordering or first-match choice resolves a tied provider.
            if len(matches) == 1:
                retained_contract, retained_row = matches[0]
                retained_term_refs.append(retained_row["term_ref"])
                rows_by_field[retained["contract_field"]] = retained_row
                contracts_by_field[retained["contract_field"]] = retained_contract
        dependency_rows = []
        for dependency in profile["activity_dependencies"]:
            provider = rows_by_field.get(dependency["provider_contract_field"])
            consumer = rows_by_field.get(dependency["consumer_contract_field"])
            if provider is None or consumer is None:
                continue
            provider_contract = contracts_by_field[dependency["provider_contract_field"]]
            consumer_contract = contracts_by_field[dependency["consumer_contract_field"]]
            evidence = tuple(sorted(set(provider["evidence_refs"]).union(consumer["evidence_refs"])))
            law, law_source = self._drm.resolve(dependency["composition_principle_ref"])
            row = FrozenMap({
                "method_ref": contract.method_ref,
                "provider_activity_ref": provider["activity_ref"], "provider_contract_token": provider["contract_token"],
                "consumer_activity_ref": consumer["activity_ref"], "consumer_contract_token": consumer["contract_token"],
                "predicate_ref": dependency["predicate_ref"], "composition_principle_ref": law.id,
                "composition_law_hash": law_source.source_hash,
                "falsifier_ref": dependency["falsifier_ref"], "preserved_invariant_refs": dependency["preserved_invariant_refs"],
                "transition_ref": transition_ref, "evidence_refs": evidence,
                "predicates_unify_exactly": dependency["predicate_ref"] in provider_contract.provides_predicate_refs and dependency["predicate_ref"] in consumer_contract.requires_predicate_refs,
                "invariants_preserved": all(ref in provider_contract.preserves_invariant_refs and ref in consumer_contract.preserves_invariant_refs for ref in dependency["preserved_invariant_refs"]),
            })
            dependency_rows.append(FrozenMap.overlay(FrozenMap({"dependency_token": stable_digest(row)}), row))
            if len(dependency_rows) > profile["max_dependency_rows"]:
                return empty
        evidence = tuple(sorted({ref for row in activity_rows
            for ref in (*row["evidence_refs"], *row["control_criterion_evidence_refs"])}
            | {ref for row in dependency_rows for ref in row["evidence_refs"]}))
        return FrozenMap.overlay(FrozenMap({
            "activity_rows": tuple(activity_rows), "dependency_rows": tuple(dependency_rows), "evidence_refs": evidence,
            "retained_term_refs": tuple(retained_term_refs),
        }), primary_row)

    def instantiate_declared_meaning(
        self,
        context: FunctionCallContext,
    ) -> object:
        facts = context.arguments.get("alternative_facts")
        if not isinstance(facts, Mapping):
            raise PureCallBlocked("declared meaning requires alternative facts")
        existing_term_refs = tuple(context.arguments.get("existing_term_refs", ()))
        if len(existing_term_refs) > 64 or any(self._store.snapshot.term(ref) is None for ref in existing_term_refs):
            raise PureCallBlocked("declared meaning requires bounded existing canonical terms")
        try:
            return instantiate_declared_meaning(
                drm=self._drm,
                selection_policy_ref=str(
                    context.arguments.get("selection_policy_ref") or ""
                ),
                selected_ref=str(context.arguments.get("selected_ref") or ""),
                alternative_facts=facts,
                evidence_refs=tuple(
                    str(item)
                    for item in context.arguments.get("evidence_refs", ())
                ),
                state_revision=self._store.snapshot.revision,
                existing_term_refs=existing_term_refs,
            )
        except (TypeError, ValueError, LookupError) as error:
            raise PureCallBlocked(str(error)) from error

    def prepare_optional_meaning_updates(self, context: FunctionCallContext) -> FrozenMap:
        try:
            prepared = prepare_declared_meaning_updates(
                snapshot=self._store.snapshot,
                terms=tuple(Term.model_validate(item) for item in context.arguments.get("terms", ())),
                claims=tuple(Claim.model_validate(item) for item in context.arguments.get("claims", ())),
                allow_empty=True,
            )
            return FrozenMap({"has_updates": prepared is not None,
                "update_items": runtime_value(prepared.update_items) if prepared is not None else ()})
        except (TypeError, ValueError) as error:
            raise PureCallBlocked(str(error)) from error

    def prepare_meaning_updates(self, context: FunctionCallContext) -> object:
        try:
            terms = tuple(
                Term.model_validate(item)
                for item in context.arguments.get("terms", ())
            )
            claims = tuple(
                Claim.model_validate(item)
                for item in context.arguments.get("claims", ())
            )
            return prepare_declared_meaning_updates(
                snapshot=self._store.snapshot,
                terms=terms,
                claims=claims,
            )
        except (TypeError, ValueError) as error:
            raise PureCallBlocked(str(error)) from error

    def link_disjoint_update_batch(self, context: FunctionCallContext) -> FrozenMap:
        from agents.yf_arc3_v5.state.update_batch import link_disjoint_update_batch
        try:
            return link_disjoint_update_batch(context.arguments['previous'],
                context.arguments['items'], context.arguments['evidence'])
        except (TypeError, ValueError) as error:
            raise PureCallBlocked(str(error)) from error

    def flatten_disjoint_update_batch(self, context: FunctionCallContext) -> FrozenMap:
        from agents.yf_arc3_v5.state.update_batch import flatten_disjoint_update_batch
        try:
            return flatten_disjoint_update_batch(context.arguments['chunks'],
                                                 context.arguments['maximum_items'])
        except (TypeError, ValueError, KeyError) as error:
            raise PureCallBlocked(str(error)) from error

    def prepare_optional_revisable_meaning_updates(self, context: FunctionCallContext) -> FrozenMap:
        try:
            prepared = prepare_declared_revisable_meaning_updates(
                snapshot=self._store.snapshot,
                terms=tuple(Term.model_validate(item) for item in context.arguments.get("terms", ())),
                claims=tuple(Claim.model_validate(item) for item in context.arguments.get("claims", ())),
                allow_empty=True,
            )
            return FrozenMap({"has_updates": prepared is not None,
                "update_items": runtime_value(prepared.update_items) if prepared is not None else ()})
        except (TypeError, ValueError) as error:
            raise PureCallBlocked(str(error)) from error

    def prepare_revisable_meaning_updates(
        self, context: FunctionCallContext
    ) -> object:
        try:
            terms = tuple(
                Term.model_validate(item)
                for item in context.arguments.get("terms", ())
            )
            claims = tuple(
                Claim.model_validate(item)
                for item in context.arguments.get("claims", ())
            )
            return prepare_declared_revisable_meaning_updates(
                snapshot=self._store.snapshot,
                terms=terms,
                claims=claims,
            )
        except (TypeError, ValueError) as error:
            raise PureCallBlocked(str(error)) from error

    def instantiate_dynamic_goal_frontier_activation(
        self, context: FunctionCallContext
    ) -> object:
        facts = context.arguments.get("facts")
        if not isinstance(facts, Mapping):
            raise PureCallBlocked("frontier activation requires DRM facts")
        try:
            return instantiate_dynamic_goal_frontier_activation(facts=facts)
        except (TypeError, ValueError, LookupError) as error:
            raise PureCallBlocked(str(error)) from error

    def dynamic_goal_frontier_payload_present(
        self, context: FunctionCallContext
    ) -> bool:
        """Check only whether DRM supplied a complete activation payload."""

        facts = context.arguments.get("facts")
        if not isinstance(facts, Mapping):
            raise PureCallBlocked("frontier payload check requires DRM facts")
        # Access-parent activation is legal only after the declared resolved
        # branch.  Partial and exhausted branches may carry the same static
        # frontier vocabulary, but must keep their committed opening plan.
        if (
            "access_resolution_status" in facts
            and facts.get("access_resolution_status") != "resolved"
        ):
            return False
        return all(
            bool(facts.get(name))
            for name in (
                "activation_ref",
                "activation_reason_refs",
                "scene_support_claim_refs",
                "terminal_goal_refs",
            )
        )

    def activation_preserves_selected_action_goal(
        self, context: FunctionCallContext
    ) -> bool:
        """Measure whether a proposed activation retains the selected action's terminal."""

        facts = context.arguments.get("facts")
        if not isinstance(facts, Mapping):
            raise PureCallBlocked("goal conservation requires DRM facts")
        intent = ActionIntent.model_validate(context.arguments.get("intent"))
        terminal = intent.active_terminal_goal_ref
        if not terminal or terminal not in intent.goal_dependency_chain_refs:
            return True
        activation_terminals = facts.get("terminal_goal_refs")
        if not isinstance(activation_terminals, (tuple, list)):
            return False
        return terminal in activation_terminals

    def dynamic_goal_frontier_activation_needs_commit(
        self, context: FunctionCallContext
    ) -> bool:
        """Compare one DRM-declared activation with canonical state exactly."""

        activation = context.arguments.get("activation")
        if not isinstance(activation, Mapping):
            raise PureCallBlocked("frontier commit check requires a typed activation")
        try:
            facts = dict(activation)
            meaning = instantiate_declared_meaning(
                drm=self._drm,
                selection_policy_ref="policy.declared_verified_goal_completion",
                selected_ref="dynamic_goal_frontier.activation",
                alternative_facts={"dynamic_goal_frontier.activation": facts},
                evidence_refs=tuple(
                    str(item) for item in facts.get("scene_support_claim_refs", ())
                ),
                state_revision=self._store.snapshot.revision,
                rule_ref_override="rule.dynamic_goal_frontier_activation_meaning",
            )
            snapshot = self._store.snapshot
            term_delta_required = any(
                (current := snapshot.term(term.id)) is None
                or current.attributes != term.attributes
                or current.label != term.label
                for term in meaning.terms
            )
            claim_delta_required = any(
                snapshot.claim(seed.id) is None for seed in meaning.conclusions
            )
            return term_delta_required or claim_delta_required
        except ValueError as error:
            raise PureCallBlocked(str(error)) from error
        except (TypeError, LookupError) as error:
            raise PureCallBlocked(str(error)) from error

    def access_goal_completion_payload_present(
        self, context: FunctionCallContext
    ) -> bool:
        """Check only whether SRC supplied a complete access-plan witness.

        The method deliberately does not decide completion.  DRM receives the
        bounded facts and selects resolved/partial/unresolved; Python only
        prevents an incomplete payload from entering that policy.
        """

        facts = context.arguments.get("facts")
        if not isinstance(facts, Mapping):
            return False
        if "goal_ref" not in facts:
            nested = facts.get("access_goal.resolved")
            if not isinstance(nested, Mapping):
                nested = facts.get("access_goal.partial_progress")
            if not isinstance(nested, Mapping):
                nested = facts.get("access_goal.unresolved")
            if not isinstance(nested, Mapping):
                nested = facts.get("access_goal.exhausted_but_insufficient")
            facts = nested if isinstance(nested, Mapping) else facts
        return all(
            facts.get(name) is not None
            for name in (
                "goal_ref",
                "transition_ref",
                "action_ref",
                "candidate_ref",
                "plan_kind",
                "parent_goal_ref",
                "active_terminal_goal_ref",
                "concrete_target_ref",
                "access_plan_parent_attached",
                "access_plan_membership",
                "plan_membership_observed",
                "expected_access_delta_present",
                "remaining_step_count",
                "remaining_step_count_before_action",
                "remaining_step_count_after_action",
                "residual_plan_present_after_action",
                "effect_observed",
                "observed_access_reachability_available",
            )
        ) and all(
            bool(facts.get(name))
            for name in (
                "parent_goal_ref",
                "active_terminal_goal_ref",
                "concrete_target_ref",
                "access_plan_parent_attached",
                "plan_membership_observed",
                "expected_access_delta_present",
            )
        )

    def committed_access_plan_present(
        self, context: FunctionCallContext
    ) -> bool:
        """Report whether the selected declared facts still carry a route cursor.

        This is a structural handoff check only.  It does not decide whether
        access is complete; it prevents generic frontier activation from
        replacing an unexhausted committed opening sequence.
        """

        facts = context.arguments.get("facts")
        selected_key = str(context.arguments.get("selected_key") or "")
        if not isinstance(facts, Mapping) or not selected_key:
            return False
        selected = facts.get(selected_key)
        if not isinstance(selected, Mapping):
            return False
        action_refs = (
            selected.get("verified_repetition_action_refs")
            or selected.get("topological_access_route_action_refs")
        )
        return bool(
            selected.get("plan_kind")
            in {
                "topological_access_expansion",
                "frontier_access_expansion",
                "local_mobility_expansion",
            }
            and int(selected.get("remaining_step_count") or 0) > 0
            and action_refs
        )

    def instantiate_dynamic_goal_frontier_meaning(
        self, context: FunctionCallContext
    ) -> object:
        activation = context.arguments.get("activation")
        if not isinstance(activation, Mapping):
            raise PureCallBlocked("frontier meaning requires a typed activation")
        try:
            facts = dict(activation)
            return instantiate_declared_meaning(
                drm=self._drm,
                selection_policy_ref="policy.declared_verified_goal_completion",
                selected_ref="dynamic_goal_frontier.activation",
                alternative_facts={"dynamic_goal_frontier.activation": facts},
                evidence_refs=tuple(
                    str(item) for item in facts.get("scene_support_claim_refs", ())
                ),
                state_revision=self._store.snapshot.revision,
                rule_ref_override="rule.dynamic_goal_frontier_activation_meaning",
            )
        except (TypeError, ValueError, LookupError) as error:
            raise PureCallBlocked(str(error)) from error

    def prepare_goal_priority_context_updates(
        self,
        context: FunctionCallContext,
    ) -> object:
        """Prepare the exact branch switch while preserving goal history.

        This call does not select a node.  SRC/DRM already selected it; the
        reducer-facing preparation only revisions the prior active priority
        claims and creates one new active revision for the selected node.
        """

        try:
            selected_ref = str(context.arguments.get("selected_ref") or "")
            disabled_reason_ref = str(
                context.arguments.get("disabled_reason_ref") or ""
            )
            if not selected_ref or not disabled_reason_ref:
                raise PureCallBlocked(
                    "goal priority update requires selected and disable reason refs"
                )
            terms = tuple(
                Term.model_validate(item)
                for item in context.arguments.get("terms", ())
            )
            claims = tuple(
                Claim.model_validate(item)
                for item in context.arguments.get("claims", ())
            )
            snapshot = self._store.snapshot
            next_revision = snapshot.revision + 1
            updates = []

            for term in terms:
                before = snapshot.term(term.id)
                after = term.model_copy(
                    update={
                        "created_at_state_revision": (
                            before.created_at_state_revision
                            if before is not None
                            else next_revision
                        ),
                        "last_changed_state_revision": next_revision,
                    }
                )
                if before is not None and (
                    before.attributes == after.attributes
                    and before.label == after.label
                ):
                    continue
                updates.append(UpdateItem(mutation=PutTerm(before=before, after=after)))

            current_priority_claims = tuple(
                claim
                for claim in snapshot.current_claims
                if claim.predicate == "goal_priority_context_selected"
            )
            selected_current = next(
                (
                    claim
                    for claim in current_priority_claims
                    if claim.attributes.get("priority_node_ref") == selected_ref
                ),
                None,
            )

            def unique_revision_id(base: str, marker: str) -> str:
                candidate = f"{base}:{marker}:r{next_revision}"
                suffix = 1
                existing = {claim.id for claim in snapshot.claims}
                existing.update(item.mutation.after.id for item in updates)
                while candidate in existing:
                    suffix += 1
                    candidate = f"{base}:{marker}:r{next_revision}:{suffix}"
                return candidate

            for claim in current_priority_claims:
                if (
                    claim.attributes.get("priority_node_ref") == selected_ref
                    or claim.attributes.get("active") is not True
                ):
                    continue
                attributes = claim.attributes.to_dict()
                attributes.update(
                    {
                        "active": False,
                        "disabled_reason_ref": disabled_reason_ref,
                        "disabled_at_state_revision": next_revision,
                    }
                )
                disabled = claim.model_copy(
                    update={
                        "id": unique_revision_id(claim.id, "disabled"),
                        "supersedes": claim.id,
                        "attributes": FrozenMap(attributes),
                    }
                )
                updates.append(
                    UpdateItem(mutation=PutClaim(before=claim, after=disabled))
                )

            selected_seed = next(
                (
                    claim
                    for claim in claims
                    if claim.attributes.get("priority_node_ref") == selected_ref
                ),
                None,
            )
            if selected_seed is None:
                raise PureCallBlocked(
                    "goal priority update has no selected claim seed"
                )
            selected_attributes = selected_seed.attributes.to_dict()
            selected_attributes.pop("declared_epistemic_status", None)
            if selected_current is None:
                selected_after = selected_seed.model_copy(
                    update={
                        "id": unique_revision_id(
                            selected_seed.id, "active"
                        ),
                        "supersedes": None,
                        "attributes": FrozenMap(selected_attributes),
                    }
                )
                updates.append(UpdateItem(mutation=PutClaim(after=selected_after)))
            elif selected_current.attributes.get("active") is not True:
                selected_attributes["active"] = True
                selected_attributes.pop("disabled_reason_ref", None)
                selected_after = selected_current.model_copy(
                    update={
                        "id": unique_revision_id(
                            selected_current.id, "active"
                        ),
                        "supersedes": selected_current.id,
                        "attributes": FrozenMap(selected_attributes),
                    }
                )
                updates.append(
                    UpdateItem(
                        mutation=PutClaim(
                            before=selected_current, after=selected_after
                        )
                    )
                )

            if not updates:
                raise PureCallBlocked("goal priority context has no canonical delta")
            return PreparedMeaningUpdates(update_items=tuple(updates))
        except PureCallBlocked:
            raise
        except (TypeError, ValueError) as error:
            raise PureCallBlocked(str(error)) from error

    def prepare_committed_proposal(self, context: FunctionCallContext) -> object:
        """Transport only the exact proposal already selected by SRC/DRM."""

        try:
            proposal = ProposalOutput.model_validate(context.arguments["proposal"])
            decision = SelectOutput.model_validate(context.arguments["decision"])
            unique_ref = str(decision.alternatives.selected or "")
            unique_claims = tuple(
                claim for claim in proposal.claims if claim.id == unique_ref
            )
            if len(unique_claims) != 1:
                raise PureCallBlocked(
                    "declarative decision must identify exactly one proposed claim"
                )
            claim = unique_claims[0]
            existing = self._store.snapshot.claim(claim.id)
            if existing is not None:
                if (
                    existing.predicate != claim.predicate
                    or existing.family != claim.family
                    or existing.arguments != claim.arguments
                    or existing.polarity != claim.polarity
                ):
                    raise PureCallBlocked(
                        "canonical proposal ref already denotes different meaning"
                    )
                return PreparedCommittedProposal(
                    canonical_claim_ref=claim.id,
                    canonical_delta_required=False,
                    update_items=(),
                )
            prepared = prepare_declared_meaning_updates(
                snapshot=self._store.snapshot, terms=(), claims=(claim,)
            )
            return PreparedCommittedProposal(
                canonical_claim_ref=claim.id,
                canonical_delta_required=True,
                update_items=prepared.update_items,
            )
        except (TypeError, ValueError) as error:
            raise PureCallBlocked(str(error)) from error

    def prepare_committed_derivation(self, context: FunctionCallContext) -> object:
        """Transport every claim produced by one declared DERIVE operation."""

        try:
            derivation = DerivationOutput.model_validate(
                context.arguments["derivation"]
            )
            new_claims: list[Claim] = []
            for claim in derivation.claims:
                existing = self._store.snapshot.claim(claim.id)
                if existing is None:
                    new_claims.append(claim)
                    continue
                comparable_fields = (
                    "predicate",
                    "family",
                    "arguments",
                    "polarity",
                    "epistemic_status",
                    "disposition",
                    "cardinality",
                    "scope",
                    "conditions",
                    "grounds",
                    "proof_rule",
                    "attributes",
                )
                if any(
                    getattr(existing, field) != getattr(claim, field)
                    for field in comparable_fields
                ):
                    raise PureCallBlocked(
                        "canonical derivation ref already denotes different meaning"
                    )
            if not new_claims:
                return PreparedCommittedDerivation(
                    canonical_claim_refs=tuple(
                        claim.id for claim in derivation.claims
                    ),
                    canonical_delta_required=False,
                    update_items=(),
                )
            prepared = prepare_declared_meaning_updates(
                snapshot=self._store.snapshot,
                terms=(),
                claims=tuple(new_claims),
            )
            return PreparedCommittedDerivation(
                canonical_claim_refs=tuple(claim.id for claim in derivation.claims),
                canonical_delta_required=True,
                update_items=prepared.update_items,
            )
        except PureCallBlocked:
            raise
        except (TypeError, ValueError) as error:
            raise PureCallBlocked(str(error)) from error

    def bind_derivation_basis(self, context: FunctionCallContext) -> tuple[str, ...]:
        """Append exact derived claim refs to a caller-declared proposal basis."""

        try:
            derivation = DerivationOutput.model_validate(
                context.arguments["derivation"]
            )
            declared_basis = tuple(
                str(item) for item in context.arguments.get("basis_refs", ())
            )
            ordered_refs = (
                *(claim.id for claim in derivation.claims),
                *declared_basis,
            )
            return tuple(dict.fromkeys(ordered_refs))
        except (TypeError, ValueError) as error:
            raise PureCallBlocked(str(error)) from error

    def bind_declared_scene_handoff(
        self,
        context: FunctionCallContext,
    ) -> FrozenMap:
        """Transport the unique canonical handoff already declared by DRM/SRC."""

        transition_ref = str(context.arguments.get("transition_ref") or "")
        matches = tuple(
            claim
            for claim in self._store.snapshot.current_claims
            if claim.predicate == "declares_level_boundary_handoff"
            and transition_ref in claim.arguments
        )
        if len(matches) != 1:
            raise PureCallBlocked(
                "level boundary requires exactly one canonical handoff claim"
            )
        unique_claim = matches[0]
        required_flags = (
            "requires_scene_rebinding",
            "rebind_preserved_knowledge_to_current_scene",
            "preserved_knowledge_remains_provisional_until_rebound",
            "reopen_current_level_goals",
            "clear_committed_local_plan",
            "save_hot_workflow_at_level_end",
            "load_hot_workflow_before_new_level_action",
            "persist_hot_workflow_through_game_end",
        )
        if any(unique_claim.attributes.get(flag) is not True for flag in required_flags):
            raise PureCallBlocked("declared level boundary is missing a required flag")
        return FrozenMap(
            {
                "handoff_claim_ref": unique_claim.id,
                "drop_local_reference_families": tuple(
                    unique_claim.attributes.get("drop_local_reference_families", ())
                ),
                "preserve_revisable_knowledge_families": tuple(
                    unique_claim.attributes.get(
                        "preserve_revisable_knowledge_families", ()
                    )
                ),
                **{
                    flag: unique_claim.attributes[flag]
                    for flag in required_flags
                },
            }
        )

    def refresh_world_context_snapshot(
        self,
        context: FunctionCallContext,
    ) -> FrozenMap:
        """Refresh only immutable authority metadata after reconciliation."""

        raw_context = context.arguments.get("context")
        if not isinstance(raw_context, Mapping):
            raise PureCallBlocked("world context is not a typed mapping")
        return FrozenMap(
            {
                **dict(raw_context),
                "snapshot_digest": self._store.snapshot.state_hash,
            }
        )

    def current_state_revision(self, context: FunctionCallContext) -> int:
        """Return only the exact current reducer revision."""

        return self._store.snapshot.revision

    def current_term_attribute_values(self, context: FunctionCallContext) -> tuple[str, ...]:
        """Project exact canonical fields requested by SRC, without assigning meaning."""
        from agents.yf_arc3_v5.state.reference_bindings import canonical_reference_bindings_current
        key = str(context.arguments["filter_attribute"])
        expected = context.arguments["filter_value"]
        field = str(context.arguments["returned_attribute"])
        bound = int(context.arguments["max_items"])
        result = []
        for term in sorted(self._store.snapshot.terms, key=lambda item: item.id):
            if term.attributes.get(key) != expected:
                continue
            if not canonical_reference_bindings_current(
                self._store.snapshot, term, frame_ref=context.frame_id,
                observation_ref=context.frame_id,
            ):
                continue
            excluded_attribute = context.arguments.get("exclude_attribute")
            if excluded_attribute is not None and term.attributes.get(excluded_attribute) == context.arguments["exclude_value"]:
                continue
            item = term.attributes.get(field)
            if not isinstance(item, str) or not item or len(result) >= bound:
                raise PureCallBlocked("canonical attribute projection is incomplete or exceeds its bound")
            result.append(item)
        return tuple(result)

    def _measure_declared_marker_signatures(self, *, tracking, policy_ref, maximum_rows):
        """Read declared marker hints and retain every innermost enclosure tie.

        A signature is an observation of a declared supposition, not Python's
        identification of the active control. SRC/DRM interpret the equality.
        """
        if not 1 <= maximum_rows <= 64:
            raise PureCallBlocked("invalid marker signature bound")
        declaration, _loaded = self._drm.resolve(policy_ref)
        projection = declaration.fact_projections[0].static_facts["marker_projection"]
        tracking = TemporalTrackingResult.model_validate(tracking)
        requests = self._current_compound_marker_requests(tracking=tracking, projection=projection)
        snapshot = self._store.snapshot
        hints = snapshot.claims_for_predicate(projection["hint_predicate"])
        if len(hints) > maximum_rows:
            raise PureCallBlocked("marker hint bound exceeded; no ties truncated")
        signatures, evidence = [], []
        for hint in sorted(hints, key=lambda item: item.id):
            if (hint.epistemic_status.value not in projection["allowed_hint_statuses"]
                    or hint.disposition.value != projection["hint_disposition"]
                    or hint.polarity.value != projection["hint_polarity"]
                    or hint.active_contradictions or hint.defeated_by
                    or len(hint.arguments) <= projection["schema_argument_index"]):
                continue
            schema = snapshot.term(hint.arguments[projection["schema_argument_index"]])
            if schema is None or projection["marker_value_attribute"] not in schema.attributes:
                continue
            marker_value = int(schema.attributes[projection["marker_value_attribute"]])
            rows, compounds = measure_marker_ownership(tracking, marker_value=marker_value,
                requests=requests, max_entities=projection["max_entities"],
                support_contract=BearerSupportContract.model_validate(projection["compound_support_contract"]))
            for row in (*rows, *compounds):
                if not row["encloses_more_specific_marker_carrier"]:
                    # Multiple canonical proofs of the same decoration on the
                    # same bearer are not multiple physical bearer candidates.
                    signatures.append((marker_value, row["carrier_identity"]))
                    if "bearer_ref" in row:
                        source = snapshot.term(row["bearer_ref"])
                        evidence.extend(source.attributes["role_premise_claim_refs"])
                        evidence.extend(item["claim_ref"] for item in source.attributes["canonical_claim_dependency_digests"])
            evidence.append(hint.id)
        signatures = tuple(sorted(set(signatures)))
        if len(signatures) > maximum_rows:
            raise PureCallBlocked("marker signature bound exceeded; no ties truncated")
        return signatures, tuple(sorted(set(evidence)))

    def measure_observed_extent_commands(self, context: FunctionCallContext) -> tuple[FrozenMap, ...]:
        from agents.yf_arc3_v5.capabilities.observed_extent_commands import measure_observed_extent_commands
        from agents.yf_arc3_v5.capabilities.contracts import ControlledTransitionAnalysis
        return measure_observed_extent_commands(
            analysis=ControlledTransitionAnalysis.model_validate(context.arguments["analysis"]),
            before_tracking=TemporalTrackingResult.model_validate(context.arguments["before_tracking"]),
            after_tracking=TemporalTrackingResult.model_validate(context.arguments["after_tracking"]),
            operator_scope_ref=context.arguments.get("operator_scope_ref"),
            operator_action_data=context.arguments.get("operator_action_data"),
            source_palette_value=context.arguments.get("source_palette_value"),
            action_ref=context.arguments["action_ref"], context_epoch=context.arguments["context_epoch"],
        )

    def measure_observed_axis_transport(self, context: FunctionCallContext) -> FrozenMap:
        from agents.yf_arc3_v5.capabilities.observed_axis_transport import measure_observed_axis_transport
        tracking = TemporalTrackingResult.model_validate(context.arguments["tracking"])
        return measure_observed_axis_transport(
            observations=context.arguments["observations"],
            present_entity_refs=tuple(item.entity_id for item in tracking.entities),
            context_epoch=context.arguments["context_epoch"],
        )

    def measure_carried_reference_center(self, context: FunctionCallContext) -> FrozenMap:
        from agents.yf_arc3_v5.capabilities.carried_reference_center import measure_carried_reference_center
        return measure_carried_reference_center(
            observations=context.arguments["observations"],
            tracking=TemporalTrackingResult.model_validate(context.arguments["tracking"]),
            context_epoch=context.arguments["context_epoch"],
        )

    def measure_observed_axis_frontier(self, context: FunctionCallContext) -> FrozenMap:
        from agents.yf_arc3_v5.capabilities.contracts import FrameGrid
        from agents.yf_arc3_v5.capabilities.observed_axis_configuration import measure_axis_frontier
        declaration, _ = self._drm.resolve(context.arguments['selection_policy_ref'])
        profile = declaration.configuration['measurement_contract']
        snapshot = self._store.snapshot
        observations = snapshot.terms_for_projection_contract(profile['extent_contract'])
        if len(observations)>profile['maximum_observations']:
            raise PureCallBlocked('observed axis frontier observation bound exceeded')
        graphs = snapshot.terms_for_projection_contract(profile['graph_contract'])
        goals = snapshot.terms_for_projection_contract(profile['goal_contract'])
        def supported_rows(terms, predicate):
            claims = snapshot.claims_for_predicate(predicate)
            return tuple(FrozenMap(dict(t.attributes,term_ref=t.id,
                premise_claim_refs=tuple(c.id for c in claims if t.id in c.arguments))) for t in terms)
        return measure_axis_frontier(
            observations=supported_rows(observations,profile['extent_predicate']),
            graphs=supported_rows(graphs,profile['graph_predicate']),
            goals=supported_rows(goals,profile['goal_predicate']),
            tracking=TemporalTrackingResult.model_validate(context.arguments['tracking']),
            frame=FrameGrid.model_validate(context.arguments['scene']['frame']).rows,
            agenda=_interaction_probe_agenda(context.arguments['agenda']),profile=profile,
        )

    def measure_transformation_pair_recurrence(self, context: FunctionCallContext) -> tuple[FrozenMap, ...]:
        from agents.yf_arc3_v5.capabilities.transformation_pairs import measure_pair_recurrence
        from agents.yf_arc3_v5.capabilities.coupling_context_geometry import measure_translation_context
        from agents.yf_arc3_v5.capabilities.coupling_observation_basis import measure_pair_observation_basis
        from agents.yf_arc3_v5.capabilities.coupling_render_accounting import measure_translation_render
        from agents.yf_arc3_v5.capabilities.shared_transformation_pattern import measure_shared_transformation_pattern
        from agents.yf_arc3_v5.capabilities.shared_effect_appearance import measure_shared_effect_slots, index_effect_bearers
        from agents.yf_arc3_v5.capabilities.registered_translation import (
            registration_delta, project_relative_entity_rows, offset_transformations,
        )
        from agents.yf_arc3_v5.capabilities.pair_observation_history import PairObservationHistory
        history = (PairObservationHistory.from_groups(context.arguments["previous_rows"])
                   if context.arguments.get("previous_rows_grouped", False)
                   else PairObservationHistory.from_rows(context.arguments["previous_rows"]))
        declaration, source = self._drm.resolve(context.arguments["shared_pattern_policy_ref"])
        pattern_profile = declaration.configuration["shared_pattern_measurement_contract"]
        facts = context.arguments["descriptive_facts"]
        if "transformation_entity_rows" not in facts or facts.get("transformation_entity_rows_truncated", True):
            return ()
        before_focus, before_focus_evidence = self._measure_declared_marker_signatures(
            tracking=context.arguments["before_tracking"], policy_ref=context.arguments["marker_policy_ref"],
            maximum_rows=context.arguments["maximum_focus_rows"])
        after_focus, after_focus_evidence = self._measure_declared_marker_signatures(
            tracking=context.arguments["after_tracking"], policy_ref=context.arguments["marker_policy_ref"],
            maximum_rows=context.arguments["maximum_focus_rows"])
        from agents.yf_arc3_v5.capabilities.transformation_units import measure_transformation_units
        before_tracking = TemporalTrackingResult.model_validate(context.arguments["before_tracking"])
        after_tracking = TemporalTrackingResult.model_validate(context.arguments["after_tracking"])
        registered_delta = registration_delta(context.arguments.get("registered_scene_translation"),
            before_frame_ref=before_tracking.frame_ref, after_frame_ref=after_tracking.frame_ref)
        collective_measurements = ()
        comparison_policy = context.arguments.get("comparison_policy_ref")
        if comparison_policy:
            comparison_declaration, _ = self._drm.resolve(comparison_policy)
            comparison_projection = project_declared_alternatives(drm=self._drm,
                selection_policy_ref=comparison_policy, alternative_refs=(), descriptive_facts=facts)
            comparison_ref = comparison_declaration.configuration["comparison_alternative_ref"]
            collective_measurements = comparison_projection.alternative_facts[comparison_ref]["collective_permutation_rows"]
        units = measure_transformation_units(
            before=before_tracking, after=after_tracking,
            entity_rows=project_relative_entity_rows(facts["transformation_entity_rows"], registered_delta),
            registered_delta=registered_delta, previous_rows=history,
            collective_measurements=collective_measurements,
            profile=declaration.configuration["comparison_unit_contract"],
            decoration_carrier_refs=tuple(sorted({ref for _, ref in (*before_focus, *after_focus)})))
        measured = measure_pair_recurrence(
            entity_rows=units["entity_rows"], previous_rows=history,
            action_ref=context.arguments["action_ref"], transition_ref=context.arguments["transition_ref"],
            context_epoch=context.arguments["context_epoch"], max_pairs=context.arguments["max_pairs"],
            operator_scope_ref=context.arguments.get("operator_scope_ref"),
        )
        if measured["pair_enumeration_truncated"]:
            raise PureCallBlocked("transformation pair enumeration truncated")
        if not measured["pair_rows"]:
            return ()
        observation_evidence = tuple(dict.fromkeys((*context.arguments["evidence_refs"],
            *before_focus_evidence, *after_focus_evidence)))
        appearance_declaration, appearance_source = self._drm.resolve(pattern_profile["appearance_key_policy_ref"])
        appearance_facts = appearance_declaration.fact_projections[0].static_facts
        appearance_contract = AppearanceKeyContract(
            tiers=appearance_facts["appearance_key_contract"],
            rotation_quarter_turns=appearance_facts["rotation_quarter_turns"],
            source_hash=appearance_source.source_hash,
            digital_topology_contract=FrozenMap.overlay(FrozenMap({"source_hash": appearance_source.source_hash}),
                appearance_facts["digital_topology_contract"]))
        before_entities = index_effect_bearers(before_tracking.entities,
            measured["pair_rows"], pattern_profile["maximum_bearers"])
        measured_descriptors = {item["entity_ref"]: item["descriptor"] for item in units["entity_rows"]}
        bearer_contexts = self._read_declared_bearer_contexts(
            appearance_facts["canonical_index_projection"]["current_context_projection"],
            frame_ref=before_tracking.frame_ref)
        appearances = {}
        rows = []
        for row in measured["pair_rows"]:
            basis = measure_pair_observation_basis(pair_row=row,
                entity_rows=units["entity_rows"], previous_rows=history.rows_for(row["entity_refs"]),
                before_focus_signatures=before_focus, after_focus_signatures=after_focus,
                transition_ref=context.arguments["transition_ref"], observed_return_rows=context.arguments["observed_return_rows"])
            pattern = measure_shared_transformation_pattern(action_ref=row["action_ref"],
                descriptors=row["observed_transformations"], profile=pattern_profile)
            previous_sample = self._store.snapshot.term(pattern_profile["sample_term_id_pattern"].format(
                mapping_token=row["mapping_token"]))
            if previous_sample is not None and previous_sample.attributes.get("effect_slots"):
                # An incompatible current effect cannot rewrite historical morphology.
                slots = previous_sample.attributes["effect_slots"]
            else:
                slot_appearances, slot_contexts = [], []
                for entity_ref, descriptor in zip(row["entity_refs"], row["observed_transformations"]):
                    entity = before_entities.get(entity_ref)
                    compound_pixels = units["before_valued_pixels_by_unit"].get(entity_ref)
                    if (entity is None and compound_pixels is None) or measured_descriptors.get(entity_ref) != descriptor:
                        slot_appearances.append(None)
                    else:
                        if entity_ref not in appearances:
                            if self._capabilities is None:
                                raise PureCallBlocked("effect appearance requires the production capability registry")
                            appearances[entity_ref] = self._capabilities.invoke(RevisionedCapabilityRequest(
                                capability_id="capability.bearer_appearance_measurement",
                                input_revision=self._store.snapshot.revision,
                                payload=BearerAppearanceInput(bearer_ref=entity_ref,
                                    valued_pixels=(compound_pixels if compound_pixels is not None else
                                        tuple((r, c, entity.component.value) for r, c in entity.component.pixels)),
                                    measurement_basis_ref=pattern_profile["measurement_basis_ref"],
                                    cell_quantum=None, key_contract=appearance_contract))).output
                        slot_appearances.append(appearances[entity_ref])
                    slot_contexts.append(bearer_contexts.get(entity_ref, ()))
                slots = measure_shared_effect_slots(descriptors=row["observed_transformations"],
                    appearances=tuple(slot_appearances), contexts=tuple(slot_contexts), profile=pattern_profile)
            row = FrozenMap.from_frozen_items(dict((*row.items(), *basis.items(), *pattern.items(),
                ("shared_effect_slots", slots),
                ("shared_pattern_contract_source_hash", source.source_hash),
                ("observation_evidence_refs", observation_evidence))))
            # Only a measured incompatible comparison requests this additional
            # raster accounting. No epistemic status or role is decided here.
            if not row["prior_incompatible_match_present"]:
                rows.append(row)
                continue
            geometry = measure_translation_context(
                scene=context.arguments["before_scene"], tracking=context.arguments["before_tracking"],
                entity_refs=row["entity_refs"], transformations=row["observed_transformations"],
                maximum_steps=context.arguments["maximum_translation_steps"],
                maximum_support_pixels=context.arguments["maximum_translation_support_pixels"],
            )
            render = measure_translation_render(
                before_scene=context.arguments["before_scene"], before_tracking=context.arguments["before_tracking"],
                after_scene=context.arguments["after_scene"], entity_refs=row["entity_refs"],
                transformations=offset_transformations(row["observed_transformations"], registered_delta),
                maximum_steps=context.arguments["maximum_translation_steps"],
                maximum_support_pixels=context.arguments["maximum_translation_support_pixels"],
            )
            rows.append(FrozenMap.from_frozen_items(dict((*row.items(), *geometry.items(), *render.items()))))
        return tuple(rows)

    def measure_remembered_coupling_render(self, context: FunctionCallContext) -> tuple[FrozenMap, ...]:
        from agents.yf_arc3_v5.capabilities.coupling_render_accounting import measure_translation_render
        from agents.yf_arc3_v5.capabilities.registered_translation import registration_delta, offset_transformations
        from agents.yf_arc3_v5.capabilities.scene_layer_composition import measure_scene_layer_composition
        from agents.yf_arc3_v5.capabilities.pair_observation_history import PairObservationHistory
        history = (PairObservationHistory.from_groups(context.arguments["previous_rows"])
                   if context.arguments.get("previous_rows_grouped", False)
                   else PairObservationHistory.from_rows(context.arguments["previous_rows"]))
        registration = context.arguments.get("registered_scene_translation")
        registered_delta = registration_delta(registration,
            before_frame_ref=TemporalTrackingResult.model_validate(context.arguments["before_tracking"]).frame_ref,
            after_frame_ref=context.arguments.get("after_frame_ref"))
        rows = []
        for row in history.iter_rows():
            screen_transformations = offset_transformations(row["observed_transformations"], registered_delta)
            render = measure_translation_render(
                before_scene=context.arguments["before_scene"], before_tracking=context.arguments["before_tracking"],
                after_scene=context.arguments["after_scene"], entity_refs=row["entity_refs"],
                transformations=screen_transformations,
                maximum_steps=context.arguments["maximum_translation_steps"],
                maximum_support_pixels=context.arguments["maximum_translation_support_pixels"],
            )
            if not render["translation_render_within_bound"]:
                raise PureCallBlocked("remembered coupling render accounting exceeds its bound")
            composition = ()
            if render["translation_render_available"] and not render["predicted_render_representation_equal"]:
                composition = measure_scene_layer_composition(
                    before_scene=context.arguments["before_scene"], before_tracking=context.arguments["before_tracking"],
                    after_scene=context.arguments["after_scene"], entity_refs=row["entity_refs"],
                    transformations=screen_transformations, render_account=render, action_ref=row["action_ref"],
                    maximum_layers=context.arguments["maximum_scene_layers"],
                    maximum_support_pixels=context.arguments["maximum_translation_support_pixels"],
                ).items()
            rows.append(FrozenMap.from_frozen_items(dict((*row.items(), *render.items(), *composition,
                ("render_observation_token", stable_digest((row["mapping_token"], context.arguments["transition_ref"]))),
                ("render_transition_ref", context.arguments["transition_ref"])))))
        return tuple(rows)

    def measure_current_coupling_context(self, context: FunctionCallContext) -> tuple[FrozenMap, ...]:
        from agents.yf_arc3_v5.capabilities.current_coupling_context import measure_current_coupling_context
        reviews = context.arguments["review_rows"]
        if len(reviews) > 64:
            raise PureCallBlocked("current coupling review bound exceeded")
        if not reviews:
            return ()
        focus, evidence = self._measure_declared_marker_signatures(
            tracking=context.arguments["tracking"], policy_ref=context.arguments["marker_policy_ref"],
            maximum_rows=context.arguments["maximum_focus_rows"])
        rows = []
        for review in reviews:
            measured = measure_current_coupling_context(review=review, scene=context.arguments["scene"],
                tracking=context.arguments["tracking"], context_epoch=context.arguments["context_epoch"],
                focus_signatures=focus, return_rows=context.arguments["return_rows"],
                maximum_steps=context.arguments["maximum_steps"],
                maximum_support_pixels=context.arguments["maximum_support_pixels"])
            rows.append(FrozenMap.from_frozen_items(dict((*measured.items(),
                ("current_marker_evidence_refs", evidence)))))
        return tuple(rows)

    def current_term_attribute_rows(self, context: FunctionCallContext) -> tuple[FrozenMap, ...]:
        """Bounded projection of fields chosen explicitly by the calling SRC."""
        key = str(context.arguments["filter_attribute"])
        expected = context.arguments["filter_value"]
        fields = tuple(context.arguments["returned_attributes"])
        bound = int(context.arguments["max_items"])
        extra_key = context.arguments.get("additional_filter_attribute")
        extra_value = context.arguments.get("additional_filter_value")
        if (extra_key is None) != (extra_value is None):
            raise PureCallBlocked("canonical row projection has an incomplete exact filter")
        if not 1 <= bound <= 64 or not 1 <= len(fields) <= 16:
            raise PureCallBlocked("invalid bounded canonical row projection")
        rows = []
        snapshot = self._store.snapshot
        indexed = snapshot.terms_by_projection_contract.get(expected, ()) if key == "projection_contract" else snapshot.terms
        for term in sorted(indexed, key=lambda item: item.id):
            if term.attributes.get(key) != expected:
                continue
            if extra_key is not None and term.attributes.get(extra_key) != extra_value:
                continue
            if len(rows) >= bound or any(field not in term.attributes for field in fields):
                raise PureCallBlocked("canonical row projection is incomplete or exceeds its bound")
            rows.append(FrozenMap({field:term.attributes[field] for field in fields}))
        return tuple(rows)

    def current_context_term_attribute_rows(self, context: FunctionCallContext) -> tuple[FrozenMap, ...]:
        """Exact Source-requested action/context slice, before the read bound."""
        key = str(context.arguments["filter_attribute"])
        expected = context.arguments["filter_value"]
        fields = tuple(context.arguments["returned_attributes"])
        optional_fields = tuple(context.arguments.get("optional_returned_attributes", ()))
        include_term_ref = context.arguments.get("include_term_ref", False)
        extra_key = context.arguments.get("additional_filter_attribute")
        extra_expected = context.arguments.get("additional_filter_value")
        if (extra_key is None) != (extra_expected is None):
            raise PureCallBlocked("canonical context projection has an incomplete extra filter")
        exact_filters = context.arguments.get("exact_attribute_filters") or FrozenMap()
        if not isinstance(exact_filters, Mapping) or len(exact_filters) > 4:
            raise PureCallBlocked("invalid bounded canonical exact filters")
        distinct_rows = context.arguments.get("distinct_rows", False)
        group_attribute = context.arguments.get("group_by_attribute")
        max_groups = int(context.arguments.get("max_groups") or 0)
        if group_attribute is not None and (not 1 <= max_groups <= 64 or distinct_rows):
            raise PureCallBlocked("invalid bounded canonical grouping")
        snapshot = self._store.snapshot
        if "only_term_refs" in context.arguments:
            raw_requested = context.arguments["only_term_refs"]
            requested = () if raw_requested is None else raw_requested
            if (not isinstance(requested, tuple) or len(requested) > 64
                    or any(not isinstance(ref, str) or not ref for ref in requested)
                    or len(set(requested)) != len(requested)):
                raise PureCallBlocked("invalid exact canonical term scope")
            # An explicitly empty request is not an unrestricted scan. Read
            # only the identities supplied by Source; missing targets remain
            # absent and never fall back to sibling or historical records.
            scoped_terms = tuple(term for ref in requested
                                 if (term := snapshot.term(ref)) is not None)
        else:
            scoped_terms = snapshot.terms_by_projection_contract.get(expected, ()) if key == "projection_contract" else snapshot.terms

        exact_filter_items = tuple(sorted(exact_filters.items()))

        def matches_context(term):
            return (term.attributes.get(key) == expected
                and (extra_key is None or term.attributes.get(extra_key) == extra_expected)
                and all(name in term.attributes and term.attributes[name] == value
                        for name, value in exact_filter_items)
                and (context.arguments["action_ref"] is None
                    or term.attributes.get("action_ref") == context.arguments["action_ref"])
                and term.attributes.get("context_epoch") == context.arguments["context_epoch"])

        claim_projection = None
        claim_policy = context.arguments.get("claim_projection_policy_ref")
        if claim_policy is not None:
            declaration, _ = self._drm.resolve(claim_policy)
            unique_profiles = tuple(projection.static_facts["term_claim_projection"]
                for projection in declaration.fact_projections if "term_claim_projection" in projection.static_facts)
            if len(unique_profiles) != 1:
                raise PureCallBlocked("canonical term claim projection is not unique")
            claim_projection = unique_profiles[0]
        entity_rows = context.arguments.get("present_entity_rows")
        membership_field = context.arguments.get("present_entity_members_attribute")
        present_tracking = context.arguments.get("present_tracking")
        present_refs = None
        if present_tracking is not None:
            if entity_rows is not None:
                raise PureCallBlocked("canonical context projection has two entity scopes")
            # Read only the Source-bound observation; do not resurrect an old
            # identity or reconstruct complete tracking/scene model objects.
            # The read bound applies to the requested join, not unrelated
            # entities in the scene. Keep every matching identity, including
            # ties; this operation does not choose a role or an alternative.
            requested_refs = frozenset(ref for term in scoped_terms if matches_context(term)
                for group in (term.attributes.get(membership_field) if membership_field else None)
                    or tuple((ref,) for ref in term.attributes.get("entity_refs", ()))
                for ref in group)
            present = set()
            for item in present_tracking["entities"]:
                ref = item["entity_id"]
                if ref not in requested_refs:
                    continue
                if ref in present:
                    raise PureCallBlocked("duplicate tracking entity projection")
                if len(present) >= 128:
                    raise PureCallBlocked("bounded tracking entity projection exceeded")
                present.add(ref)
            present_refs = frozenset(present)
        if entity_rows is not None:
            if len(entity_rows) > 128 or any("entity_ref" not in row for row in entity_rows):
                raise PureCallBlocked("invalid bounded current entity projection")
            present_refs = frozenset(row["entity_ref"] for row in entity_rows)
            if len(present_refs) != len(entity_rows):
                raise PureCallBlocked("duplicate current entity projection")
        bound = int(context.arguments["max_items"])
        if (not 1 <= bound <= 64 or not 1 <= len(fields) + len(optional_fields) + int(include_term_ref) + int(claim_policy is not None) <= 16
                or len(set((*fields, *optional_fields))) != len(fields) + len(optional_fields)):
            raise PureCallBlocked("invalid bounded canonical context projection")
        rows = []
        seen_rows = set()
        groups = {}
        for term in scoped_terms:
            if not matches_context(term):
                continue
            if present_refs is not None:
                refs = term.attributes.get("entity_refs")
                if refs is None:
                    raise PureCallBlocked("canonical context projection lacks requested entity scope")
                members = term.attributes.get(membership_field) if membership_field else None
                if members is not None:
                    if len(members) != len(refs) or any(not group or len(group) > 64 for group in members):
                        raise PureCallBlocked("invalid canonical compound membership")
                    refs = tuple(member for group in members for member in group)
                if not all(ref in present_refs for ref in refs):
                    continue
            evidence_claim_refs = ()
            if claim_projection is not None:
                index = claim_projection["term_argument_index"]
                candidates = tuple(claim for claim in self._store.snapshot.claims_for_predicate(claim_projection["predicate"])
                    if len(claim.arguments) > index and claim.arguments[index] == term.id)
                if len(candidates) > claim_projection["maximum_claims_per_term"]:
                    raise PureCallBlocked("canonical scoped term proof bound exceeded")
                evidence_claim_refs = tuple(sorted(claim.id for claim in candidates
                    if claim.epistemic_status.value in claim_projection["allowed_statuses"]
                    and claim.disposition.value == claim_projection["disposition"]
                    and claim.polarity.value == claim_projection["polarity"]
                    and claim.proof_rule in claim_projection["proof_rules"]
                    and not claim.active_contradictions and not claim.defeated_by))
                if not evidence_claim_refs:
                    continue
            if any(field not in term.attributes for field in fields):
                raise PureCallBlocked("canonical current context projection lacks required fields")
            row = FrozenMap.from_frozen_items(dict(
                [(field, term.attributes[field]) for field in fields]
                + [(field, term.attributes.get(field)) for field in optional_fields]
                + ([("term_ref", term.id)] if include_term_ref else [])
                + ([("evidence_claim_refs", evidence_claim_refs)] if claim_projection is not None else [])))
            if distinct_rows and row in seen_rows:
                continue
            target = rows
            if group_attribute is not None:
                if group_attribute not in term.attributes:
                    raise PureCallBlocked("canonical grouping lacks requested key")
                group_key = term.attributes[group_attribute]
                if group_key not in groups:
                    if len(groups) >= max_groups:
                        raise PureCallBlocked("canonical context group bound exceeded")
                    groups[group_key] = []
                target = groups[group_key]
            if len(target) >= bound:
                raise PureCallBlocked(
                    "canonical current context projection exceeds its bound; "
                    f"filter={key}:{str(expected)[:160]}; bound={bound}; "
                    f"action={context.arguments['action_ref']}; "
                    f"step={context.workflow_step_id}"
                )
            target.append((term.id, row))
            if distinct_rows:
                seen_rows.add(row)
        if group_attribute is not None:
            return tuple(FrozenMap({"group_key": key,
                "rows": tuple(row for _, row in sorted(values, key=lambda item: item[0]))})
                for key, values in sorted(groups.items()))
        return tuple(row for _, row in sorted(rows, key=lambda item: item[0]))

    def goal_priority_context_needs_commit(
        self,
        context: FunctionCallContext,
    ) -> bool:
        """Compare two declaratively selected node references mechanically."""

        selected_ref = str(context.arguments.get("selected_ref") or "")
        active_ref = str(context.arguments.get("active_ref") or "")
        return bool(selected_ref and selected_ref != active_ref)

    def probe_action_is_point(self, context: FunctionCallContext) -> bool:
        """Measure whether the source-selected candidate uses point input."""

        agenda = _interaction_probe_agenda(context.arguments["agenda"])
        selected_ref = str(context.arguments.get("selected_ref") or "")
        matches = tuple(
            candidate.action_ref
            for candidate in agenda.candidates
            if candidate.candidate_ref == selected_ref
        )
        if len(matches) != 1:
            raise PureCallBlocked("selected probe lacks one exact agenda action")
        return matches[0] == POINT_ACTION_REF

    def probe_fact_is_true(self, context: FunctionCallContext) -> bool:
        """Read one immutable Boolean fact for the exact source-bound candidate."""
        agenda = _interaction_probe_agenda(context.arguments["agenda"])
        candidate_ref = str(context.arguments.get("candidate_ref") or "")
        if sum(candidate.candidate_ref == candidate_ref for candidate in agenda.candidates) != 1:
            raise PureCallBlocked("probe fact lacks one exact agenda candidate")
        facts = agenda.alternative_facts.get(candidate_ref)
        return bool(facts is not None and facts.get(str(context.arguments["field_name"])) is True)

    def prepare_interaction_probe(
        self,
        context: FunctionCallContext,
    ) -> PreparedInteractionProbe:
        """Bind a source-selected candidate without choosing or reordering it."""

        agenda = _interaction_probe_agenda(context.arguments["agenda"])
        projected_agenda = context.arguments.get("projected_agenda")
        if projected_agenda is None:
            projected_agenda = FrozenMap(
                {"alternative_facts": agenda.alternative_facts}
            )
        if not isinstance(projected_agenda, Mapping):
            raise PureCallBlocked("declared agenda projection is not a mapping")
        projected_facts = projected_agenda.get("alternative_facts")
        if not isinstance(projected_facts, Mapping):
            raise PureCallBlocked("declared agenda facts are not a mapping")
        selection_policy_ref = str(
            context.arguments.get("selection_policy_ref") or ""
        )

        selection_source_ref = str(
            context.arguments.get("selection_source_ref") or ""
        )
        selected_ref = str(context.arguments.get("selected_ref") or "")
        selected = next(
            (
                candidate
                for candidate in agenda.candidates
                if candidate.candidate_ref == selected_ref
            ),
            None,
        )
        candidate_agenda = agenda
        if selected is None and projected_agenda.get("candidates") is not None:
            # DRM projection may replace a coarse measured point witness with
            # one bounded full-locus candidate.  This is identity transport,
            # not selection: SRC already selected ``selected_ref``.  Prepare
            # that exact projected candidate so grounding can accept it or
            # reject only it and preserve its siblings.
            projected_candidate_agenda = _interaction_probe_agenda(projected_agenda)
            selected = next(
                (
                    candidate
                    for candidate in projected_candidate_agenda.candidates
                    if candidate.candidate_ref == selected_ref
                ),
                None,
            )
            if selected is not None:
                candidate_agenda = projected_candidate_agenda
        if selected is None:
            raise PureCallBlocked("SRC selection did not identify an agenda candidate")
        if (
            selected.action_ref == POINT_ACTION_REF
            and candidate_agenda.context_facts.get(
                "declared_point_candidate_scope_ref"
            )
            == "probe_scope.access_witness_only"
            and context.arguments.get("point_locus_authorized") is not True
        ):
            raise PureCallBlocked(
                "point input requires full point-locus perception before preparation"
            )
        raw_context = context.arguments["context"]
        if not isinstance(raw_context, Mapping):
            raise PureCallBlocked("world context is not a typed mapping")
        parent = self._scheduler.current(context.workflow_instance_id)
        alternatives = tuple(
            candidate_ref
            for candidate_ref in candidate_agenda.alternative_refs
            if candidate_ref != selected_ref
        )
        selected_declared_facts = projected_facts.get(selected_ref, FrozenMap())
        if not isinstance(selected_declared_facts, Mapping):
            raise PureCallBlocked("selected declared agenda facts are not a mapping")
        # The DRM projection may attach the active goal only to the shared
        # projected context (rather than duplicating it on every candidate).
        # Preserve that context at the action boundary; otherwise a goal can
        # be instantiated correctly and then disappear before grounding.
        projected_context = projected_agenda.get("context_facts")
        if not isinstance(projected_context, Mapping):
            projected_context = FrozenMap()
        selected_facts = ChainMap(
            selected_declared_facts,
            projected_context,
            agenda.context_facts,
        )
        # Total MATCH branches use explicit negative transport facts.  Older
        # agenda projections may carry ``None`` for these optional route
        # markers; normalize only those absent route markers, without creating
        # a semantic selection in Python.
        selected_facts = FrozenMap(
            {
                **dict(selected_facts),
                **{
                    key: False
                    for key in (
                        "committed_verified_repetition",
                        "declared_complete_series_authorized",
                    )
                    if selected_facts.get(key) is None
                },
            }
        )
        consolidated_priority_ref = str(
            selected_facts.get("consolidated_priority_ref") or ""
        )
        if (
            consolidated_priority_ref
            and not selected_facts.get("consolidation_reason_refs")
        ):
            declared_reason_refs = self._declared_consolidation_reason_refs_for(
                consolidated_priority_ref
            )
            if declared_reason_refs:
                selected_facts = FrozenMap(
                    {
                        **dict(selected_facts),
                        "consolidation_reason_refs": declared_reason_refs,
                    }
                )
        candidate_local_goal_context_required = bool(
            context.arguments.get("candidate_local_goal_context_required") is True
        )
        dynamic_goal_context_present = bool(
            not candidate_local_goal_context_required
            and projected_context.get("dynamic_goal_selection_present") is True
            and projected_context.get("selected_dynamic_goal_ref")
            and projected_context.get("active_goal_ref")
        )
        # DRM may refine the selected goal's chain without replacing its owner.
        # Transport that declared refinement only under exact goal agreement.
        declared_chain_refines_current_goal = bool(
            selected_facts.get("release_dependency_chain_refines_selected_goal") is True
            and selected_facts.get("active_goal_ref") == projected_context.get("active_goal_ref")
            and selected_facts.get("active_terminal_goal_ref") == projected_context.get("active_terminal_goal_ref")
        )
        # A candidate's generic exploration grounding is only a fallback.  If
        # SRC has selected a declared dynamic goal, keep that exact goal as
        # the action's semantic owner and discard the candidate-local generic
        # hierarchy rather than combining two incompatible chains.
        action_active_goal_ref = (
            selected_facts.get("active_goal_ref")
            if candidate_local_goal_context_required
            else (
                projected_context.get("active_goal_ref")
                if dynamic_goal_context_present
                else selected_facts.get(
                    "release_active_goal_ref", selected_facts.get("active_goal_ref")
                )
            )
        )
        action_terminal_goal_ref = (
            selected_facts.get("active_terminal_goal_ref")
            if candidate_local_goal_context_required
            else (
                projected_context.get("active_terminal_goal_ref")
                if dynamic_goal_context_present
                else selected_facts.get("active_terminal_goal_ref")
            )
        )
        expected_terminal_refs = tuple(
            str(ref)
            for ref in (selected_facts.get("expected_terminal_refs") or ())
        )
        # A DRM-declared committed plan is already the executable semantic
        # branch.  Generic exploration grounding may also provide a
        # discriminating-experiment reference for the same candidate, but the
        # action contract intentionally permits exactly one of these roots.
        # Preserve the stronger declared plan and clear only the redundant
        # experiment transport field; no goal or priority is invented here.
        committed_plan_ref = selected_facts.get("committed_plan_ref")
        discriminating_experiment_ref = selected_facts.get(
            "release_discriminating_experiment_ref",
            selected_facts.get("discriminating_experiment_ref"),
        )
        grounding_data = {
            key: value
            for key, value in sorted({
                "active_goal_ref": action_active_goal_ref,
                "active_terminal_goal_ref": action_terminal_goal_ref,
                "declared_terminal_goal_ref": (
                    selected_facts.get("active_terminal_goal_ref")
                    if candidate_local_goal_context_required
                    else (
                        None
                        if dynamic_goal_context_present
                        else selected_facts.get("release_terminal_goal_ref")
                    )
                ),
                "subgoal_ref": (
                    selected_facts.get("subgoal_ref")
                    if candidate_local_goal_context_required
                    else selected_facts.get(
                        "release_subgoal_ref", selected_facts.get("subgoal_ref")
                    )
                ),
                "goal_graph_edge_refs": (
                    selected_facts.get("goal_graph_edge_refs")
                ),
                "goal_dependency_chain_refs": (
                    projected_context.get("goal_dependency_chain_refs")
                    if dynamic_goal_context_present and not declared_chain_refines_current_goal
                    else selected_facts.get("goal_dependency_chain_refs")
                ),
                "declared_dependency_chain_refs": (
                    selected_facts.get("goal_dependency_chain_refs")
                    if candidate_local_goal_context_required
                    else (
                        ()
                        if dynamic_goal_context_present and not declared_chain_refines_current_goal
                        else selected_facts.get("release_goal_dependency_chain_refs")
                    )
                ),
                "advances_active_goal_chain": (
                    True
                    if dynamic_goal_context_present
                    else selected_facts.get("advances_active_goal_chain")
                ),
                "inherited_priority_from_terminal_goal": selected_facts.get(
                    "inherited_priority_from_terminal_goal"
                ),
                "priority_class_ref": selected_facts.get("priority_class_ref"),
                "selection_principle_ref": selected_facts.get(
                    "selection_principle_ref"
                    if candidate_local_goal_context_required
                    else "release_selection_principle_ref",
                    selected_facts.get("selection_principle_ref"),
                ),
                "premise_claim_refs": selected_facts.get(
                    "premise_claim_refs"
                    if candidate_local_goal_context_required
                    else "release_premise_claim_refs",
                    selected_facts.get("premise_claim_refs"),
                ),
                "canonical_premise_claim_refs": selected_facts.get("canonical_premise_claim_refs", ()),
                "coupling_measurement_requested": selected_facts.get("coupling_measurement_requested"),
                "coupling_operator_scope_measurement_kind": selected_facts.get("coupling_operator_scope_measurement_kind"),
                "expected_effect_class": selected_facts.get(
                    "expected_effect_class"
                    if candidate_local_goal_context_required
                    else "release_expected_effect_class",
                    selected_facts.get("expected_effect_class"),
                ),
                "expected_configuration_or_goal_delta_ref": selected_facts.get(
                    "expected_configuration_or_goal_delta_ref"
                    if candidate_local_goal_context_required
                    else "release_expected_configuration_or_goal_delta_ref",
                    selected_facts.get("expected_configuration_or_goal_delta_ref"),
                ),
                "falsifier_refs": selected_facts.get(
                    "falsifier_refs"
                    if candidate_local_goal_context_required
                    else "release_falsifier_refs",
                    selected_facts.get("falsifier_refs"),
                ),
                "committed_plan_ref": selected_facts.get("committed_plan_ref"),
                "discriminating_experiment_ref": (
                    None
                    if committed_plan_ref
                    else discriminating_experiment_ref
                ),
                "lifecycle_engagement_ref": selected_facts.get(
                    "lifecycle_engagement_ref"
                ),
                "declared_plan_ref": selected_facts.get("declared_plan_ref"),
                "declared_plan_stage_refs": selected_facts.get(
                    "declared_plan_stage_refs"
                ),
                "declared_plan_ordered_member_refs": selected_facts.get(
                    "declared_plan_ordered_member_refs"
                ),
                "declared_plan_ordered_member_values": selected_facts.get(
                    "declared_plan_ordered_member_values"
                ),
                "declared_plan_current_stage_ref": selected_facts.get(
                    "declared_plan_current_stage_ref"
                ),
                "declared_plan_cursor_ordinal": selected_facts.get(
                    "declared_plan_cursor_ordinal"
                ),
                "declared_plan_member_count": selected_facts.get(
                    "declared_plan_member_count"
                ),
                "remaining_action_bound": selected_facts.get(
                    "remaining_action_bound"
                    if candidate_local_goal_context_required
                    else "release_remaining_action_bound",
                    selected_facts.get("remaining_action_bound"),
                ),
                "reconciliation_checkpoint_ref": selected_facts.get(
                    "reconciliation_checkpoint_ref"
                    if candidate_local_goal_context_required
                    else "release_reconciliation_checkpoint_ref",
                    selected_facts.get("reconciliation_checkpoint_ref"),
                ),
            }.items())
            if value is not None
        }
        # Transport a complete DRM-declared dynamic-frontier payload beside
        # the action grounding.  SRC may commit it before release; this code
        # neither completes missing fields nor originates semantic refs.
        frontier_activation_data = {
            key: value
            for key, value in sorted({
                "activation_ref": selected_facts.get("activation_ref"),
                "static_knowledge_graph_ref": selected_facts.get(
                    "static_knowledge_graph_ref"
                ),
                "guiding_principle_ref": selected_facts.get(
                    "guiding_principle_ref"
                ),
                "scene_support_claim_refs": selected_facts.get(
                    "scene_support_claim_refs"
                ),
                "terminal_goal_refs": selected_facts.get("terminal_goal_refs"),
                "instrumental_subgoal_refs": selected_facts.get(
                    "instrumental_subgoal_refs"
                ),
                "instrumental_parent_goal_refs": selected_facts.get(
                    "instrumental_parent_goal_refs"
                ),
                "goal_falsifier_refs": selected_facts.get(
                    "goal_falsifier_refs"
                ),
                "active_priority_node_ref": selected_facts.get(
                    "active_priority_node_ref"
                ),
                "source_static_priority_ref": selected_facts.get(
                    "source_static_priority_ref"
                ),
                "consolidated_priority_ref": selected_facts.get(
                    "consolidated_priority_ref"
                ),
                "consolidation_reason_refs": selected_facts.get(
                    "consolidation_reason_refs"
                ),
                "activation_reason_refs": selected_facts.get(
                    "activation_reason_refs"
                ),
                "disable_reason_refs": selected_facts.get("disable_reason_refs"),
                "alternative_goal_refs": selected_facts.get(
                    "alternative_goal_refs"
                ),
                "conflicting_goal_refs": selected_facts.get(
                    "conflicting_goal_refs"
                ),
                "best_credible_outcome_refs": selected_facts.get(
                    "best_credible_outcome_refs"
                ),
                "worst_credible_outcome_refs": selected_facts.get(
                    "worst_credible_outcome_refs"
                ),
            }.items())
            if value is not None
        }
        selected = selected.model_copy(
            update={
                "basis_refs": tuple(
                    str(item)
                    for item in selected_facts.get(
                        "release_declared_basis_refs",
                        selected_declared_facts.get(
                            "declared_basis_refs", selected.basis_refs
                        ),
                    )
                ),
                "expectation_refs": tuple(
                    dict.fromkeys(
                        (
                            *(
                                str(item)
                                for item in selected_facts.get(
                                    "release_declared_expectation_refs",
                                    selected_declared_facts.get(
                                        "declared_expectation_refs", ()
                                    ),
                                )
                            ),
                            *(str(item) for item in selected.expectation_refs),
                        )
                    )
                ),
            }
        )
        prepared_context = FrozenMap(
            {
                **dict(raw_context),
                "action_data": selected.action_data,
                "selected_candidate_ref": selected_ref,
                "selected_point_palette_value": (
                    selected_facts.get("candidate_palette_value")
                    if selected.action_ref == POINT_ACTION_REF
                    else None
                ),
                "selected_point_fills_bbox": (
                    selected_facts.get("unoccupied_bbox_pixel_count") == 0
                    if selected.action_ref == POINT_ACTION_REF
                    else None
                ),
                "selected_point_exact_morphology_group_size": (
                    selected_facts.get("shape_occurrence_count")
                    if selected.action_ref == POINT_ACTION_REF
                    else None
                ),
                "selected_point_intersects_transient_change_bbox": (
                    selected_facts.get("intersects_transient_change_bbox")
                    if selected.action_ref == POINT_ACTION_REF
                    else None
                ),
                "selection_basis_refs": selected.basis_refs,
                "selection_policy_ref": selection_policy_ref,
                "selection_source_ref": selection_source_ref,
                "candidate_local_goal_context_required": (
                    candidate_local_goal_context_required
                ),
                "candidate_local_goal_context_declared": (
                    selected_facts.get("candidate_local_goal_context_declared") is True
                ),
                "alternatives_preserved": alternatives,
                # A committed repetition/route is already selected by
                # DRM/SRC.  Preserve its immutable cursor as transport data
                # so CONTINUE can materialize the whole series once, before
                # the first environment trigger, without reselecting or
                # re-perceiving between actions.
                "committed_verified_repetition": selected_facts.get(
                    "committed_verified_repetition"
                ),
                "verified_repetition_action_refs": tuple(
                    str(ref)
                    for ref in (
                        selected_facts.get("verified_repetition_action_refs")
                        or ()
                    )
                ),
                "verified_repetition_action_payloads": tuple(
                    payload
                    for payload in (
                        selected_facts.get("verified_repetition_action_payloads")
                        or ()
                    )
                ),
                "verified_repetition_component_refs": tuple(
                    str(ref)
                    for ref in (
                        selected_facts.get("verified_repetition_component_refs")
                        or ()
                    )
                ),
                "verified_repetition_candidate_refs": tuple(
                    str(ref)
                    for ref in (
                        selected_facts.get("verified_repetition_candidate_refs")
                        or ()
                    )
                ),
                # Keep the complete declared continuation contract beside the
                # immutable route cursor.  Reconciliation receives this exact
                # selected DRM/SRC delta; it must not reconstruct it from the
                # measured candidate agenda after the action has crossed the
                # environment boundary.
                **{
                    key: selected_facts.get(key)
                    for key in (
                        "committed_cursor_advance_declared",
                        "post_action_committed_plan_transport_declared",
                        "plan_kind",
                        "interaction_kind",
                        "continuation_interaction_measurement_kind",
                        "remaining_step_count",
                        "remaining_action_bound",
                        "verified_repetition_cycle_release_declared",
                        "verified_repetition_full_observation_retry_declared",
                        "verified_alignment_selection_projection_complete",
                        "executed_verified_step_count",
                        "memorial_route_commitment_present",
                        "memorial_route_initial",
                        "memorial_route_grid_geometry",
                        "memorial_route_source_member_refs",
                        "memorial_route_reflected_member_refs",
                        "memorial_route_source_cells",
                        "memorial_route_reflected_cells",
                        "memorial_route_source_boxes",
                        "memorial_route_reflected_boxes",
                        "memorial_route_axis_coordinate_twice",
                        "memorial_route_dimension",
                        "prepared_method_activity_contract",
                        "memorial_source_goal_ref",
                        "recalled_method_conditions_match",
                        "recalled_method_preparation_needed_for_this_candidate",
                        "memorial_source_normal_residual",
                        "memorial_route_normal_change_count",
                        "memorial_route_premise_refs",
                        "memorial_route_premise_revalidation",
                        "memorial_route_cell_layers",
                        "full_observation_before_continuation_declared",
                        "post_action_full_goal_frontier_revalidation_required",
                        "fast_continuation_requires_meaningful_scene_change",
                        "expected_step_delta_row",
                        "expected_step_delta_col",
                        "marker_axis_residual_reconciliation_applicable",
                        "single_pair_residual_reconciliation_applicable",
                        "checkpoint_on_world_projection_change",
                        # Exact DRM-authored configuration/release evidence is
                        # retained for post-action temporal reconciliation.
                        # These fields carry no Python-authored role or policy.
                        "directed_accretive_port_status",
                        "directed_accretive_port_scope",
                        "directed_accretive_commit_risk_scope",
                        # A measured frontier route may contain several
                        # mechanically valid primitives while its later
                        # successors remain conditional on observation.  DRM
                        # therefore keeps the parent goal and explicitly asks
                        # SRC to release one primitive, reconcile, and
                        # re-project before continuing.  Transport this fact
                        # unchanged; do not infer it from route length.
                        "replan_after_each_primitive_declared",
                    )
                    if selected_facts.get(key) is not None
                },
                "expected_terminal_refs": expected_terminal_refs,
                "declared_terminal_marker_checkable": any(
                    _declared_terminal_marker_checkable(ref)
                    for ref in expected_terminal_refs
                ),
                "declared_deterministic_series_authorized": bool(
                    selected_facts.get(
                        "declared_deterministic_series_authorized"
                    )
                    is True
                ),
                "declared_complete_series_authorized": bool(
                    selected_facts.get(
                        "replan_after_each_primitive_declared"
                    )
                    is not True
                    and (
                        selected_facts.get(
                            "declared_deterministic_series_authorized"
                        )
                        is True
                        or any(
                            _declared_terminal_marker_checkable(ref)
                            for ref in expected_terminal_refs
                        )
                    )
                ),
                "terminal_objective_contract_ref": selected_facts.get(
                    "terminal_objective_contract_ref"
                ),
                "terminal_objective_measure_ref": selected_facts.get(
                    "terminal_objective_measure_ref"
                ),
                "terminal_objective_comparator": selected_facts.get(
                    "terminal_objective_comparator"
                ),
                "terminal_objective_target_value": selected_facts.get(
                    "terminal_objective_target_value"
                ),
                "terminal_objective_predicted_value_after_plan": selected_facts.get(
                    "terminal_objective_predicted_value_after_plan"
                ),
                "hot_workflow_ref": selected_facts.get("hot_workflow_ref"),
                "hot_workflow_activity_ref": selected_facts.get(
                    "hot_workflow_activity_ref"
                ),
                # Preserve the exact DRM-declared coordinate basis alongside
                # the frozen action. Absence stays absent; no mapping is
                # inferred here from reflection flags or measured motion.
                **{
                    key: selected_facts[key]
                    for key in (
                        "shape_peer_residual_transport_row_factor",
                        "shape_peer_residual_transport_col_factor",
                        "joint_peer_collision_route_present",
                        "joint_route_endpoint_positions_coincide",
                        "peer_signature_digest",
                        "peer_route_expected_pixel_boxes",
                        "peer_route_initial_pixel_boxes",
                    "peer_route_causal_steps",
                        "peer_current_step_pixels",
                        "peer_observed_blocker_palette_rows",
                        "peer_route_packet_reconciliation_declared",
                    "recalled_peer_step_requires_effect_reconciliation",
                        "independent_destination_validation_simulated",
                        "entity_typed_destination_validation_simulated",
                        "complete_swept_footprint_validation_simulated",
                        "opposed_column_coupling_measured",
                        "shared_row_coupling_measured",
                        "peer_method_signature_match_count",
                        "peer_method_goal_ref",
                        "peer_method_premise_refs",
                        "peer_palette_values",
                        "peer_cell_extent",
                        "peer_passable_palette_rows",
                        "peer_coupling_row",
                        "peer_coupling_column",
                        "peer_interface_delta_rows",
                    )
                    if key in selected_facts
                },
                # Keep the declarative goal chain inspectable for SRC
                # continuation, the controller report, and reconciliation.
                # These are transported facts only; semantic selection remains
                # owned by DRM/SRC above this boundary.
                **grounding_data,
                **frontier_activation_data,
                "point_pre_action_enclosure_residual": (
                    (
                        selected_facts.get(
                            "nearest_exact_enclosure_residual_row_signed"
                        ),
                        selected_facts.get(
                            "nearest_exact_enclosure_residual_col_signed"
                        ),
                )
                if selected.action_ref == POINT_ACTION_REF
                    and selected_facts.get(
                        "unique_exact_enclosure_assignment_present"
                    )
                    else None
                ),
                "point_pre_action_translated_peer_residual": (
                    (
                        -int(selected_facts.get(
                            "shape_peer_residual_row_signed"
                        ) or 0),
                        -int(selected_facts.get(
                            "shape_peer_residual_col_signed"
                        ) or 0),
                    )
                    if selected.action_ref == POINT_ACTION_REF
                    and selected_facts.get(
                        "is_exact_morphology_peer_of_translated_quantum_cell"
                    ) is True
                    and selected_facts.get(
                        "nonlocal_quantized_translation_observed"
                    ) is False
                    else None
                ),
            }
        )
        viability_id = f"viability:{context.workflow_instance_id}:{selected_ref}"
        viability = ViabilityAssessment(
            id=viability_id,
            source_unit=context.workflow_definition_id,
            source_hash=parent.source_hash,
            state_revision=self._store.snapshot.revision,
            workflow_definition_id=context.workflow_definition_id,
            workflow_instance_id=context.workflow_instance_id,
            workflow_step_id=context.workflow_step_id,
            timeline_id=context.timeline_id,
            frame_id=context.frame_id,
            provenance=tuple(selected.basis_refs),
            change_ref=selected_ref,
            status=ViabilityStatus.CONDITIONAL,
            unresolved_conditions=("condition.effect_not_yet_observed",),
            resource_consequences=FrozenMap(
                {
                    "cost": "one_controlled_action",
                    "information_gain": "tests_one_preserved_candidate",
                    "scarcity_status": "not_yet_grounded",
                }
            ),
            reversibility=Reversibility.UNKNOWN,
            alternatives_preserved=alternatives,
        )
        intent = ActionIntent(
            id=f"intent:{context.workflow_instance_id}:{selected_ref}",
            source_unit=context.workflow_definition_id,
            source_hash=parent.source_hash,
            state_revision=self._store.snapshot.revision,
            workflow_definition_id=context.workflow_definition_id,
            workflow_instance_id=context.workflow_instance_id,
            workflow_step_id=context.workflow_step_id,
            timeline_id=context.timeline_id,
            frame_id=context.frame_id,
            provenance=(viability.id, selected_ref),
            action_ref=selected.action_ref,
            available_action_ref=selected.action_ref,
            action_data=selected.action_data,
            world_context_ref=str(prepared_context.get("id") or ""),
            commitment_decision_id=f"commitment:{viability.id}",
            expectation_refs=selected.expectation_refs,
            active_goal_ref=(
                str(grounding_data.get("active_goal_ref"))
                if grounding_data.get("active_goal_ref")
                else None
            ),
            active_terminal_goal_ref=(
                str(grounding_data.get("active_terminal_goal_ref"))
                if grounding_data.get("active_terminal_goal_ref")
                else None
            ),
            declared_terminal_goal_ref=(
                str(grounding_data.get("declared_terminal_goal_ref"))
                if grounding_data.get("declared_terminal_goal_ref")
                else None
            ),
            subgoal_ref=(
                str(grounding_data.get("subgoal_ref"))
                if grounding_data.get("subgoal_ref")
                else None
            ),
            goal_graph_edge_refs=tuple(
                str(ref)
                for ref in (grounding_data.get("goal_graph_edge_refs") or ())
            ),
            goal_dependency_chain_refs=tuple(
                str(ref)
                for ref in (grounding_data.get("goal_dependency_chain_refs") or ())
            ),
            declared_dependency_chain_refs=tuple(
                str(ref)
                for ref in (
                    grounding_data.get("declared_dependency_chain_refs") or ()
                )
            ),
            advances_active_goal_chain=(
                bool(grounding_data.get("advances_active_goal_chain"))
                if grounding_data.get("advances_active_goal_chain") is not None
                else None
            ),
            inherited_priority_from_terminal_goal=bool(
                grounding_data.get("inherited_priority_from_terminal_goal")
            ),
            priority_class_ref=(
                str(grounding_data.get("priority_class_ref"))
                if grounding_data.get("priority_class_ref")
                else None
            ),
            selection_principle_ref=(
                str(grounding_data.get("selection_principle_ref"))
                if grounding_data.get("selection_principle_ref")
                else None
            ),
            premise_claim_refs=tuple(
                str(ref)
                for ref in (grounding_data.get("premise_claim_refs") or ())
            ),
            canonical_premise_claim_refs=tuple(grounding_data.get("canonical_premise_claim_refs") or ()),
            expected_effect_class=(
                str(grounding_data.get("expected_effect_class"))
                if grounding_data.get("expected_effect_class")
                else None
            ),
            expected_configuration_or_goal_delta_ref=(
                str(grounding_data.get("expected_configuration_or_goal_delta_ref"))
                if grounding_data.get("expected_configuration_or_goal_delta_ref")
                else None
            ),
            falsifier_refs=tuple(
                str(ref)
                for ref in (grounding_data.get("falsifier_refs") or ())
            ),
            committed_plan_ref=(
                str(grounding_data.get("committed_plan_ref"))
                if grounding_data.get("committed_plan_ref")
                else None
            ),
            discriminating_experiment_ref=(
                str(grounding_data.get("discriminating_experiment_ref"))
                if grounding_data.get("discriminating_experiment_ref")
                else None
            ),
            lifecycle_engagement_ref=(
                str(grounding_data.get("lifecycle_engagement_ref"))
                if grounding_data.get("lifecycle_engagement_ref")
                else None
            ),
            declared_plan_ref=(
                str(grounding_data.get("declared_plan_ref"))
                if grounding_data.get("declared_plan_ref")
                else None
            ),
            declared_plan_stage_refs=tuple(
                str(ref)
                for ref in (grounding_data.get("declared_plan_stage_refs") or ())
            ),
            declared_plan_ordered_member_refs=tuple(
                str(ref)
                for ref in (
                    grounding_data.get("declared_plan_ordered_member_refs") or ()
                )
            ),
            declared_plan_ordered_member_values=tuple(
                int(value)
                for value in (
                    grounding_data.get("declared_plan_ordered_member_values") or ()
                )
            ),
            declared_plan_current_stage_ref=(
                str(grounding_data.get("declared_plan_current_stage_ref"))
                if grounding_data.get("declared_plan_current_stage_ref")
                else None
            ),
            declared_plan_cursor_ordinal=(
                int(grounding_data.get("declared_plan_cursor_ordinal"))
                if grounding_data.get("declared_plan_cursor_ordinal") is not None
                else None
            ),
            declared_plan_member_count=(
                int(grounding_data.get("declared_plan_member_count"))
                if grounding_data.get("declared_plan_member_count") is not None
                else None
            ),
            remaining_action_bound=(
                int(grounding_data.get("remaining_action_bound"))
                if grounding_data.get("remaining_action_bound") is not None
                else None
            ),
            reconciliation_checkpoint_ref=(
                str(grounding_data.get("reconciliation_checkpoint_ref"))
                if grounding_data.get("reconciliation_checkpoint_ref")
                else None
            ),
        )
        return PreparedInteractionProbe(
            selected_candidate_ref=selected_ref,
            viability=viability,
            intent=intent,
            context=prepared_context,
            alternatives_preserved=alternatives,
            selection_policy_ref=str(
                context.arguments.get("selection_policy_ref") or ""
            ),
            selection_source_ref=str(
                context.arguments.get("selection_source_ref") or ""
            ),
        )

    def validate_action_grounding(
        self,
        context: FunctionCallContext,
    ) -> ActionGroundingContract:
        """Fail closed before a migrated exploration action is released."""

        intent = ActionIntent.model_validate(context.arguments["intent"])
        if not frozenset(intent.canonical_premise_claim_refs).issubset(intent.premise_claim_refs):
            raise PureCallBlocked("canonical premise obligation is absent from the frozen why")
        for reference in intent.canonical_premise_claim_refs:
            claim = self._store.snapshot.claim(reference)
            if (claim is None or claim.disposition.value != "active"
                    or claim.polarity.value != "positive" or claim.active_contradictions or claim.defeated_by
                    or not any(current.id == reference for current in self._store.snapshot.claims_for_predicate(claim.predicate))):
                raise PureCallBlocked("canonical premise proof is absent or inactive: " + reference)
        for reference in intent.premise_claim_refs:
            claim = self._store.snapshot.claim(reference)
            if claim is None:
                continue  # Existing symbolic/static premises are checked by SRC.
            bound_arguments = tuple(self._store.snapshot.term(argument) for argument in claim.arguments)
            has_reference_bindings = any(term is not None and (
                term.attributes.get("canonical_term_dependency_revisions")
                or term.attributes.get("canonical_claim_dependency_digests")
                or term.attributes.get("operational_scene_ref")
            ) for term in bound_arguments)
            if has_reference_bindings and not any(current.id == claim.id for current in self._store.snapshot.claims_for_predicate(claim.predicate)):
                raise PureCallBlocked("canonical premise reference binding is stale: " + reference)
            for term in bound_arguments:
                if term is not None and not canonical_reference_bindings_current(
                    self._store.snapshot, term,
                    frame_ref=context.frame_id, observation_ref=context.frame_id,
                    require_projectable=False,
                ):
                    raise PureCallBlocked("canonical premise reference binding is stale: " + reference)
        missing_grounding_fields = tuple(
            field
            for field, present in (
                ("falsifier_refs", bool(intent.falsifier_refs)),
                (
                    "terminal_goal_ref",
                    bool(
                        intent.active_terminal_goal_ref
                        or intent.declared_terminal_goal_ref
                    ),
                ),
                (
                    "dependency_chain_refs",
                    bool(
                        intent.goal_dependency_chain_refs
                        or intent.declared_dependency_chain_refs
                    ),
                ),
            )
            if not present
        )
        if missing_grounding_fields:
            raise PureCallBlocked(
                "no_grounded_action_available:missing="
                + ",".join(missing_grounding_fields)
            )
        try:
            return ActionGroundingContract(
                active_goal_ref=intent.active_goal_ref or "",
                active_terminal_goal_ref=intent.active_terminal_goal_ref,
                subgoal_ref=intent.subgoal_ref or "",
                goal_graph_edge_refs=intent.goal_graph_edge_refs,
                goal_dependency_chain_refs=intent.goal_dependency_chain_refs,
                advances_active_goal_chain=intent.advances_active_goal_chain,
                inherited_priority_from_terminal_goal=(
                    intent.inherited_priority_from_terminal_goal
                ),
                priority_class_ref=intent.priority_class_ref,
                selection_principle_ref=intent.selection_principle_ref or "",
                premise_claim_refs=intent.premise_claim_refs,
                canonical_premise_claim_refs=intent.canonical_premise_claim_refs,
                expected_effect_class=intent.expected_effect_class or "",
                expected_configuration_or_goal_delta_ref=(
                    intent.expected_configuration_or_goal_delta_ref or ""
                ),
                explicit_falsifier_ref=intent.falsifier_refs[0],
                committed_plan_ref=intent.committed_plan_ref,
                discriminating_experiment_ref=intent.discriminating_experiment_ref,
                declared_plan_ref=intent.declared_plan_ref,
                declared_plan_stage_refs=intent.declared_plan_stage_refs,
                declared_plan_ordered_member_refs=(
                    intent.declared_plan_ordered_member_refs
                ),
                declared_plan_ordered_member_values=(
                    intent.declared_plan_ordered_member_values
                ),
                declared_plan_current_stage_ref=intent.declared_plan_current_stage_ref,
                declared_plan_cursor_ordinal=intent.declared_plan_cursor_ordinal,
                declared_plan_member_count=intent.declared_plan_member_count,
                remaining_action_bound=intent.remaining_action_bound or 0,
                reconciliation_checkpoint_ref=(
                    intent.reconciliation_checkpoint_ref or ""
                ),
            )
        except Exception as exc:
            raise PureCallBlocked(
                "no_grounded_action_available:contract="
                + str(exc).replace("\n", " | ")[:800]
                + f":intent={intent.id}"
                + f":goal={intent.active_goal_ref}"
                + f":principle={intent.selection_principle_ref}"
                + f":bound={intent.remaining_action_bound}"
            ) from exc

    def action_grounding_is_valid(self, context: FunctionCallContext) -> bool:
        """Mechanically test the same fail-closed contract without ending SRC."""

        try:
            self.validate_action_grounding(context)
        except PureCallBlocked:
            return False
        return True

    def bind_interaction_agenda(
        self,
        context: FunctionCallContext,
    ) -> InteractionProbeAgenda:
        """Bind measured candidates to their immutable DRM projection."""

        agenda = _interaction_probe_agenda(context.arguments["agenda"])
        projected = context.arguments.get("projected_agenda")
        if isinstance(projected, InteractionProbeAgenda):
            return projected
        if not isinstance(projected, Mapping):
            raise PureCallBlocked("projected interaction agenda is not a mapping")
        projected_facts = projected.get("alternative_facts")
        if not isinstance(projected_facts, Mapping):
            raise PureCallBlocked("projected interaction facts are not a mapping")
        if set(projected_facts) != set(agenda.alternative_refs):
            raise PureCallBlocked("projected interaction facts do not index the agenda")
        projected_context = projected.get("selection_context_facts")
        if not isinstance(projected_context, Mapping):
            projected_context = projected.get("context_facts")
        if not isinstance(projected_context, Mapping):
            projected_context = agenda.context_facts
        return agenda.model_copy(
            update={
                "alternative_facts": FrozenMap(
                    {ref: projected_facts[ref] for ref in agenda.alternative_refs}
                ),
                "context_facts": FrozenMap(dict(projected_context)),
            }
        )

    def exclude_interaction_candidate_after_grounding_rejection(
        self,
        context: FunctionCallContext,
    ) -> InteractionProbeAgenda:
        """Remove only the exact rejected candidate and preserve every sibling."""

        agenda = _interaction_probe_agenda(context.arguments["agenda"])
        selected_ref = str(context.arguments.get("selected_ref") or "")
        failure_kind = str(context.arguments.get("failure_kind") or "")
        if selected_ref not in agenda.alternative_refs:
            raise PureCallBlocked(
                "grounding rejection does not identify an agenda candidate"
            )
        if not failure_kind:
            raise PureCallBlocked("grounding rejection requires a failure kind")
        retained_candidates = tuple(
            candidate
            for candidate in agenda.candidates
            if candidate.candidate_ref != selected_ref
        )
        retained_refs = tuple(
            ref for ref in agenda.alternative_refs if ref != selected_ref
        )
        retained_facts = FrozenMap(
            {ref: agenda.alternative_facts[ref] for ref in retained_refs}
        )
        retained_deltas = FrozenMap(
            {
                ref: agenda.descriptive_delta_facts[ref]
                for ref in retained_refs
                if ref in agenda.descriptive_delta_facts
            }
        )
        prior_rejections = tuple(
            zip(
                agenda.rejected_grounding_candidate_refs,
                agenda.rejected_grounding_failure_kinds,
            )
        )
        rejection_pairs = prior_rejections + ((selected_ref, failure_kind),)
        return agenda.model_copy(
            update={
                "candidates": retained_candidates,
                "alternative_refs": retained_refs,
                "alternative_facts": retained_facts,
                "descriptive_delta_facts": retained_deltas,
                "has_candidates": bool(retained_candidates),
                "rejected_grounding_candidate_refs": tuple(
                    ref for ref, _kind in rejection_pairs
                ),
                "rejected_grounding_failure_kinds": tuple(
                    kind for _ref, kind in rejection_pairs
                ),
            }
        )

    def operational_wisdom(self, context: FunctionCallContext) -> CommitmentDecision:
        viability = ViabilityAssessment.model_validate(context.arguments["viability"])
        self._record_lineage_once(viability)
        checkpoint = self._execute_gate(
            workflow_id=WISDOM_WORKFLOW_ID,
            context=context,
            inputs={"viability": viability},
            gate_name="wisdom",
        )
        outcomes = {
            (WorkflowStatus.COMPLETED, "commit"): CommitmentKind.COMMIT,
            (
                WorkflowStatus.COMPLETED,
                "conditional_commit",
            ): CommitmentKind.COMMIT,
            (WorkflowStatus.BLOCKED, "reject"): CommitmentKind.REJECT,
            (WorkflowStatus.DEFERRED, "wait"): CommitmentKind.DEFER,
        }
        decision = outcomes.get((checkpoint.status, checkpoint.terminal_result or ""))
        if decision is None:
            raise RuntimeError(
                "operational wisdom returned an undeclared source outcome: "
                f"{checkpoint.status.value}:{checkpoint.terminal_result}"
            )
        decision_record = CommitmentDecision(
            id=f"commitment:{viability.id}",
            source_unit=WISDOM_WORKFLOW_ID,
            source_hash=checkpoint.source_hash,
            state_revision=self._store.snapshot.revision,
            workflow_definition_id=checkpoint.workflow_definition_id,
            workflow_instance_id=checkpoint.workflow_instance_id,
            workflow_step_id=checkpoint.workflow_step_id,
            timeline_id=context.timeline_id,
            frame_id=context.frame_id,
            provenance=(viability.id, checkpoint.id),
            change_ref=viability.change_ref,
            decision=decision,
            viability_assessment_id=viability.id,
            policy_ref=WISDOM_WORKFLOW_ID,
            reason_refs=(
                f"source_outcome:{checkpoint.terminal_result}",
                f"source_checkpoint:{checkpoint.id}",
            ),
            alternatives_preserved=viability.alternatives_preserved,
        )
        self._record_lineage_once(decision_record)
        return decision_record

    def release_action(self, context: FunctionCallContext) -> ActionReleasePermit:
        intent = ActionIntent.model_validate(context.arguments["intent"])
        viability = ViabilityAssessment.model_validate(context.arguments["viability"])
        wisdom = CommitmentDecision.model_validate(context.arguments["wisdom"])
        world_context = context.arguments["context"]
        if not isinstance(world_context, Mapping):
            raise PureCallBlocked("world context is not a typed mapping")

        current_world = (
            self._world_inputs.get_by_frame_id(context.frame_id)
            if self._world_inputs is not None
            else None
        )
        action_data = world_context.get("action_data") or {}
        observed_fatal_replay_present = bool(
            current_world is not None
            and isinstance(action_data, Mapping)
            and measure_exact_observed_fatal_replay(
                claims=self._store.snapshot.current_claims,
                world=current_world,
                action_ref=intent.action_ref,
                action_data=action_data,
            )
        )

        checkpoint = self._execute_gate(
            workflow_id=RELEASE_WORKFLOW_ID,
            context=context,
            inputs={"viability": viability, "wisdom": wisdom,
                    "announced_data": world_context.get("announced_step", {}),
                    "observed_fatal_replay_facts": FrozenMap({
                        "observed_fatal_replay_present": observed_fatal_replay_present,
                    }),
                    "current_frame_ref": context.frame_id},
            gate_name="release",
        )
        if checkpoint.status is WorkflowStatus.DEFERRED:
            raise PureCallDeferred(checkpoint.terminal_result or "release unresolved")
        if checkpoint.status is not WorkflowStatus.COMPLETED:
            raise PureCallBlocked(checkpoint.terminal_result or "release blocked")
        if checkpoint.terminal_result != "release":
            raise PureCallBlocked("release source did not authorize dispatch")

        self._validate_release_lineage(
            context=context,
            intent=intent,
            viability=viability,
            wisdom=wisdom,
            world_context=world_context,
        )
        requires_full_revalidation = bool(
            world_context.get(
                "post_action_full_goal_frontier_revalidation_required"
            )
            is True
            or world_context.get("replan_after_each_primitive_declared") is True
        )
        if context.arguments.get("declared_series") is True:
            # SRC/DRM authorizes the full committed route.  Python only
            # materializes that decision; an unverified experiment remains a
            # singleton series followed by reconciliation.
            route_refs = (
                tuple(
                    str(ref)
                    for ref in (
                        world_context.get("verified_repetition_action_refs") or ()
                    )
                    if str(ref)
                )
                if context.arguments.get("full_route_authorized") is True
                and not requires_full_revalidation
                else ()
            ) or (intent.action_ref,)
            if route_refs[0] != intent.action_ref:
                raise PureCallBlocked(
                    "declared action series begins with another prepared action"
                )
            if len(route_refs) > 256:
                raise PureCallBlocked("declared action series exceeds hard bound")
            series_seed = stable_digest(
                {
                    "workflow_instance_id": context.workflow_instance_id,
                    "action_intent_id": intent.id,
                    "action_refs": route_refs,
                }
            )[:16]
            declared_series_ref = f"series:{series_seed}"
            declared_series_target = (
                f"workflow:continue-declared-series:{series_seed}"
            )
            declared_series_intent_ids = tuple(
                f"intent:declared-series:{series_seed}:{index:04d}"
                for index in range(len(route_refs))
            )
            declared_series_action_refs = route_refs
        else:
            declared_series_ref = (
                str(world_context.get("declared_series_ref"))
                if world_context.get("declared_series_ref")
                else None
            )
            declared_series_intent_ids = tuple(
                str(item)
                for item in (world_context.get("declared_series_intent_ids") or ())
            )
            declared_series_action_refs = tuple(
                str(item)
                for item in (world_context.get("declared_series_action_refs") or ())
            )
            declared_series_target = (
                str(world_context.get("declared_series_target_workflow_instance_id"))
                if world_context.get("declared_series_target_workflow_instance_id")
                else None
            )
        if declared_series_ref:
            if not declared_series_intent_ids:
                raise PureCallBlocked("declared action series has no intent ids")
            if len(declared_series_intent_ids) != len(declared_series_action_refs):
                raise PureCallBlocked("declared action series lineage is misaligned")
            if not declared_series_target:
                raise PureCallBlocked("declared action series has no target workflow")
        lease_intent_id = (
            declared_series_intent_ids[0]
            if declared_series_ref
            else intent.id
        )
        lease = self._validated_concentration_lease(
            context=context,
            intent=intent,
            viability=viability,
            wisdom=wisdom,
            world_context=world_context,
            action_intent_id=lease_intent_id,
            extension_intent_ids=(
                declared_series_intent_ids
                if declared_series_ref
                else (intent.id,)
            ),
            requires_full_revalidation=requires_full_revalidation,
            target_workflow_instance_id=(
                declared_series_target or context.workflow_instance_id
            ),
        )
        defer_acquisition = context.arguments.get("defer_concentration_acquisition") is True
        if defer_acquisition and not declared_series_ref:
            raise PureCallBlocked("deferred concentration requires an SRC-declared series")
        if not defer_acquisition:
            self._record_lineage_once(lease)
        return ActionReleasePermit(
            id=f"permit:{intent.id}",
            source_unit=RELEASE_WORKFLOW_ID,
            source_hash=checkpoint.source_hash,
            state_revision=self._store.snapshot.revision,
            workflow_definition_id=context.workflow_definition_id,
            workflow_instance_id=(
                declared_series_target or context.workflow_instance_id
            ),
            workflow_step_id=context.workflow_step_id,
            timeline_id=context.timeline_id,
            frame_id=context.frame_id,
            provenance=(intent.id, viability.id, wisdom.id, checkpoint.id),
            action_intent_id=(
                declared_series_intent_ids[0]
                if declared_series_ref
                else intent.id
            ),
            viability_assessment_id=viability.id,
            commitment_decision_id=wisdom.id,
            available_action_ref=intent.available_action_ref,
            pre_action_snapshot_digest=self._store.snapshot.state_hash,
            valid_through_state_revision=self._store.snapshot.revision,
            declared_series_ref=declared_series_ref,
            declared_series_intent_ids=declared_series_intent_ids,
            declared_series_action_refs=declared_series_action_refs,
            declared_series_target_workflow_instance_id=declared_series_target,
            concentration_acquisition_deferred=defer_acquisition,
            concentration_lease=lease,
        )

    def activate_dynamic_goal_frontier(
        self, context: FunctionCallContext
    ) -> PreparedMeaningUpdates:
        activation = context.arguments.get("activation")
        if not isinstance(activation, Mapping):
            raise PureCallBlocked("frontier activation is not a typed payload")
        checkpoint = self._execute_gate(
            workflow_id=ACTIVATE_DYNAMIC_GOAL_FRONTIER_WORKFLOW_ID,
            context=context,
            inputs={
                "activation": activation,
                "before_revision": int(context.arguments["before_revision"]),
            },
            gate_name="dynamic-goal-frontier",
        )
        if checkpoint.status is not WorkflowStatus.COMPLETED:
            raise PureCallBlocked(
                checkpoint.terminal_result or "dynamic frontier activation blocked"
            )
        prepared = checkpoint.local_bindings.get("prepared")
        if not isinstance(prepared, Mapping):
            raise PureCallBlocked("dynamic frontier activation returned no updates")
        return PreparedMeaningUpdates.model_validate(prepared)

    def execute_source_subworkflow(self, context: FunctionCallContext) -> object:
        """Dispatch one source-declared nested workflow without recreating meaning."""

        short_name = context.callee_ref.removeprefix("workflow.")
        workflow_ids = tuple(
            workflow_id
            for workflow_id in self._entries
            if workflow_id.endswith(f".{short_name}")
        )
        if len(workflow_ids) != 1:
            raise PureCallBlocked(
                "source subworkflow dispatch requires one exact registered workflow"
            )
        checkpoint = self._execute_gate(
            workflow_id=workflow_ids[0],
            context=context,
            inputs=context.arguments,
            gate_name=f"source-subworkflow-{short_name}",
        )
        if checkpoint.status is WorkflowStatus.DEFERRED:
            raise PureCallDeferred(
                checkpoint.terminal_result or "source subworkflow deferred"
            )
        if checkpoint.status is not WorkflowStatus.COMPLETED:
            raise PureCallBlocked(
                checkpoint.terminal_result or "source subworkflow blocked"
            )
        result = checkpoint.local_bindings.get("result")
        if result is not None:
            return result
        if checkpoint.terminal_result is not None:
            return checkpoint.terminal_result
        raise PureCallBlocked("source subworkflow returned no result binding")

    def _execute_gate(
        self,
        *,
        workflow_id: str,
        context: FunctionCallContext,
        inputs: Mapping[str, object],
        gate_name: str,
    ) -> WorkflowInstanceRecord:
        entry = self._entry(workflow_id)
        frozen_inputs = FrozenMap(
            {name: runtime_value(value) for name, value in sorted(inputs.items())}
        )
        identity = stable_digest(
            {
                "call_id": context.call_id,
                "workflow_id": workflow_id,
                "inputs": frozen_inputs,
            }
        )[:16]
        instance_id = f"{context.workflow_instance_id}:{gate_name}:{identity}"
        try:
            checkpoint = self._scheduler.current(instance_id)
        except WorkflowInstanceNotFound:
            from agents.yf_arc3_v5.scheduler.contracts import WorkflowStartRequest

            checkpoint = self._scheduler.start(
                WorkflowStartRequest(
                    workflow_definition_id=workflow_id,
                    workflow_instance_id=instance_id,
                    run_id=self._store.run_id,
                    timeline_id=context.timeline_id,
                    frame_id=context.frame_id,
                    expected_transitive_source_hash=entry.transitive_source_hash,
                    inputs=frozen_inputs,
                )
            )
        if checkpoint.status is WorkflowStatus.ACTIVE:
            # These source-controlled gates are pure, bounded and consumed only
            # through their terminal outcome.  Preserve the SRC execution path
            # while materializing one durable terminal checkpoint instead of a
            # checkpoint after every internal instruction.
            checkpoint = self._scheduler.advance(
                instance_id,
                durable_boundaries_only=True,
            ).checkpoint
        return checkpoint

    def _entry(self, workflow_id: str) -> RegistryEntry:
        entry = self._entries.get(workflow_id)
        if entry is None:
            raise RuntimeError(
                f"source-controlled gate is not registered: {workflow_id}"
            )
        self._workflows.resolve_workflow(
            workflow_id,
            expected_transitive_source_hash=entry.transitive_source_hash,
        )
        return entry

    def _validate_release_lineage(
        self,
        *,
        context: FunctionCallContext,
        intent: ActionIntent,
        viability: ViabilityAssessment,
        wisdom: CommitmentDecision,
        world_context: Mapping[str, object],
    ) -> None:
        parent = self._scheduler.current(context.workflow_instance_id)
        failures: list[str] = []
        if viability.state_revision != self._store.snapshot.revision:
            failures.append("stale viability")
        if intent.state_revision != self._store.snapshot.revision:
            failures.append("stale intent")
        if wisdom.state_revision != self._store.snapshot.revision:
            failures.append("stale wisdom")
        if wisdom.decision is not CommitmentKind.COMMIT:
            failures.append("wisdom did not commit")
        if viability.status not in {
            ViabilityStatus.VIABLE,
            ViabilityStatus.CONDITIONAL,
        }:
            failures.append("viability does not permit release")
        if wisdom.viability_assessment_id != viability.id:
            failures.append("wisdom viability lineage")
        if wisdom.change_ref != viability.change_ref:
            failures.append("wisdom change lineage")
        if intent.commitment_decision_id != wisdom.id:
            failures.append("intent commitment lineage")
        if intent.workflow_definition_id != context.workflow_definition_id:
            failures.append("intent workflow definition")
        if intent.workflow_instance_id != context.workflow_instance_id:
            failures.append("intent workflow instance")
        if intent.timeline_id != context.timeline_id:
            failures.append("intent timeline")
        if intent.frame_id != context.frame_id:
            failures.append("intent frame")
        if intent.source_hash != parent.source_hash:
            failures.append("intent source hash")
        if intent.world_context_ref != str(world_context.get("id") or ""):
            failures.append("intent world context")
        raw_available = world_context.get("available_action_refs", ())
        available = tuple(
            str(item)
            for item in (
                raw_available if isinstance(raw_available, (list, tuple)) else ()
            )
        )
        if intent.available_action_ref not in available:
            failures.append("action is unavailable")
        if world_context.get("snapshot_digest") != self._store.snapshot.state_hash:
            failures.append("pre-action snapshot")
        raw_invalidated = world_context.get("invalidated_refs", ())
        invalidated = {
            str(item)
            for item in (
                raw_invalidated if isinstance(raw_invalidated, (list, tuple)) else ()
            )
        }
        if invalidated & {
            intent.id,
            wisdom.id,
            viability.id,
            intent.action_ref,
        }:
            failures.append("critical lineage is invalidated")
        if self._intent_already_consumed(intent.id):
            failures.append("intent was already released or observed")
        if failures:
            raise PureCallBlocked("; ".join(failures))

    def _intent_already_consumed(self, intent_id: str) -> bool:
        for artifact_type in (ActionReleasePermit, ActionObservedRecord):
            for artifact in self._store.artifacts_of_type(artifact_type):
                if artifact.action_intent_id == intent_id:
                    return True
        return False

    def _validated_concentration_lease(
        self,
        *,
        context: FunctionCallContext,
        intent: ActionIntent,
        viability: ViabilityAssessment,
        wisdom: CommitmentDecision,
        world_context: Mapping[str, object],
        action_intent_id: str,
        extension_intent_ids: tuple[str, ...],
        requires_full_revalidation: bool,
        target_workflow_instance_id: str,
    ) -> ConcentrationLease:
        objective_ref = intent.subgoal_ref or intent.active_goal_ref
        information_step_ref = intent.discriminating_experiment_ref
        principle_ref = intent.selection_principle_ref
        expected_effect_ref = (
            intent.expected_configuration_or_goal_delta_ref
            or intent.expected_effect_class
        )
        scope = {
            key: world_context.get(key)
            for key in ("game_ref", "level_ref", "session_ref")
        }
        if any(not scope[key] for key in ("game_ref", "level_ref", "session_ref")):
            raise PureCallBlocked("concentration lease requires explicit game, level and session scope")
        if str(scope["session_ref"]) != self._store.run_id:
            raise PureCallBlocked("concentration lease session is not canonical")
        missing = tuple(
            name
            for name, value in (
                ("objective_or_information_step", objective_ref or information_step_ref),
                ("generic_principle", principle_ref),
                ("premises", intent.premise_claim_refs),
                ("expected_effect", expected_effect_ref),
                ("falsifiers", intent.falsifier_refs),
                ("drm_decision", viability.id),
                ("src_decision", wisdom.id),
            )
            if not value
        )
        if missing:
            raise PureCallBlocked(
                "concentration lease contract is incomplete: " + ", ".join(missing)
            )
        revision = self._store.snapshot.revision
        lease_seed = stable_digest(
            (
                context.timeline_id,
                context.frame_id,
                action_intent_id,
                revision,
                target_workflow_instance_id,
            )
        )[:16]
        return ConcentrationLease(
            id=f"concentration-lease:{lease_seed}",
            source_unit=RELEASE_WORKFLOW_ID,
            source_hash=intent.source_hash,
            state_revision=revision,
            workflow_definition_id=context.workflow_definition_id,
            workflow_instance_id=target_workflow_instance_id,
            workflow_step_id=context.workflow_step_id,
            timeline_id=context.timeline_id,
            frame_id=context.frame_id,
            provenance=(intent.id, wisdom.id),
            game_ref=str(scope["game_ref"]),
            level_ref=str(scope["level_ref"]),
            session_ref=str(scope["session_ref"]),
            frame_revision=revision,
            contract_revision=revision,
            active_objective_ref=objective_ref,
            information_step_ref=information_step_ref,
            committed_path_ref=(
                intent.committed_plan_ref or intent.declared_plan_ref or intent.id
            ),
            remaining_action_bound=intent.remaining_action_bound,
            drm_decision_ref=viability.id,
            src_decision_ref=wisdom.id,
            allowed_action_ref=intent.action_ref,
            action_intent_id=action_intent_id,
            premise_claim_refs=intent.premise_claim_refs,
            generic_principle_ref=str(principle_ref),
            expected_effect_ref=str(expected_effect_ref),
            falsifier_refs=intent.falsifier_refs,
            preserve_constraint_refs=tuple(
                str(ref) for ref in (world_context.get("preserve_constraint_refs") or ())
            ),
            overshoot_constraint_refs=tuple(
                str(ref)
                for ref in (
                    world_context.get("forbidden_overshoot_refs") or ()
                )
            ),
            state=ConcentrationLeaseState.ACQUIRED,
            acquisition_reason_ref=wisdom.id,
            expires_after_state_revision=revision + max(1, len(extension_intent_ids)),
            extension_intent_ids=extension_intent_ids,
            requires_post_action_full_revalidation=requires_full_revalidation,
        )

    def _record_lineage_once(
        self,
        decision: ViabilityAssessment | CommitmentDecision | ConcentrationLease,
    ) -> None:
        artifact = self._store.artifact(decision.id)
        if isinstance(artifact, type(decision)):
            return
        mutations: tuple[RecordArtifact, ...] = (RecordArtifact(artifact=decision),)
        if isinstance(decision, ConcentrationLease):
            validated = ConcentrationLeaseTransition(
                id=f"{decision.id}:validated",
                source_unit=decision.source_unit,
                source_hash=decision.source_hash,
                state_revision=decision.state_revision,
                workflow_definition_id=decision.workflow_definition_id,
                workflow_instance_id=decision.workflow_instance_id,
                workflow_step_id=decision.workflow_step_id,
                timeline_id=decision.timeline_id,
                frame_id=decision.frame_id,
                provenance=(decision.id, decision.src_decision_ref),
                concentration_lease_id=decision.id,
                from_state=ConcentrationLeaseState.ACQUIRED,
                to_state=ConcentrationLeaseState.VALIDATED,
                reason_ref=decision.src_decision_ref,
                action_intent_id=decision.action_intent_id,
            )
            governance = project_concentration_operation(
                lease=decision,
                lineage=validated,
                operation=ConcentrationOperation.ATTRIBUTE,
                reason_ref=decision.src_decision_ref,
            )
            mutations = (
                *mutations,
                RecordArtifact(artifact=validated),
                RecordArtifact(artifact=governance),
            )
        self._store.commit(
            TransactionRequest(
                transaction_id=f"{decision.id}:record",
                run_id=self._store.run_id,
                timeline_id=decision.timeline_id or "",
                frame_id=decision.frame_id,
                expected_revision=self._store.snapshot.revision,
                source_hash=decision.source_hash,
                mutations=mutations,
            )
        )

    def _artifacts(self) -> Sequence[object]:
        return self._store.artifacts
