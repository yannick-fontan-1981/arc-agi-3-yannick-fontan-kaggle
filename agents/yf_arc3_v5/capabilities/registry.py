"""Attested runtime registry and revision-owned cache for pure capabilities."""

from __future__ import annotations

import hashlib
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from agents.yf_arc3_v5.capabilities.announced_motion import (
    AnnouncedStepInput, AnnouncedStepMeasurements, measure_announced_step,
)
from agents.yf_arc3_v5.capabilities.abduction import measure_abductive_version_space
from agents.yf_arc3_v5.capabilities.appearance import (
    BearerAppearanceInput, BearerAppearanceMeasurement, measure_bearer_appearance,
)
from agents.yf_arc3_v5.capabilities.cursor_orientation import (
    AttachedRotationAnalysisInput, AttachedRotationAnalysis, analyze_attached_rotations,
)
from agents.yf_arc3_v5.capabilities.display_correspondence import (
    DisplayCorrespondenceInput, DisplayCorrespondenceMeasurements, measure_display_correspondence,
)

from agents.yf_arc3_v5.capabilities.components import (
    detect_components,
    enumerate_component_matches,
)
from agents.yf_arc3_v5.capabilities.cell_recount import recount_known_grid_cells
from agents.yf_arc3_v5.capabilities.cell_objects import assemble_known_grid_objects
from agents.yf_arc3_v5.capabilities.constraints import (
    compare_state_dimensions,
    simulate_quantities,
)
from agents.yf_arc3_v5.capabilities.contracts import (
    AbductiveVersionSpaceInput,
    AbductiveVersionSpaceMeasurements,
    BlockGridSceneDescription,
    CapabilityExecution,
    CommittedVerifiedAlignmentCursorInput,
    CommittedQuantizedPlanAdvanceInput,
    ComponentExtractionInput,
    ComponentExtractionResult,
    ComponentMatchInput,
    ComponentMatchResult,
    ControlledTransitionAnalysis,
    ControlledTransitionAnalysisInput,
    DeclaredGoalGapInput,
    DeclaredGoalGapResult,
    DeclaredGoalPlanAlternatives,
    DeclaredGoalPlanInput,
    DerivedGeometryStateInput,
    DerivedGeometryStateMeasurements,
    DimensionTransitionInput,
    DimensionTransitionResult,
    EliminationViabilityInput,
    EliminationViabilityMeasurements,
    RelationalEliminationGraphInput,
    RelationalEliminationGraphMeasurements,
    FrameDifferenceInput,
    FrameDifferenceResult,
    FrameNormalizationInput,
    FrameNormalizationResult,
    GeometryInput,
    GeometryResult,
    OrthogonalPerimeterAssemblyInput,
    OrthogonalPerimeterAssemblyMeasurements,
    GoalCompletionAnalysis,
    GoalCompletionAnalysisInput,
    GridPartitionInput,
    GridPartitionResult,
    PeriodicCellGridInput,
    PeriodicCellGridResult,
    KnownGridCellRecountInput,
    KnownGridCellRecountResult,
    KnownGridObjectAssemblyInput,
    KnownGridObjectAssemblyResult,
    HomologousRepairAgendaInput,
    HomologousRepairAnalysis,
    HomologousRepairInput,
    HomologousRepairReconciliationAnalysis,
    HomologousRepairReconciliationInput,
    InteractionProbeAgenda,
    InteractionProbeInput,
    InteractionTopologyInput,
    InteractionTopologyMeasurements,
    ArticulatedKinematicsInput,
    ArticulatedKinematicsMeasurements,
    ArticulatedSharedSnapshotInput,
    ArticulatedSharedSnapshotMeasurements,
    ArticulatedSuccessorInput,
    ArticulatedSuccessorMeasurements,
    TypedAssignmentDomainsInput,
    TypedAssignmentDomainsMeasurements,
    TypedEditSuccessorInput,
    TypedEditSuccessorMeasurements,
    TypedInterpreterInput,
    TypedInterpreterMeasurements,
    RecipePatternDifferenceInput,
    RecipePatternDifferenceMeasurements,
    RecipeSuccessorInput,
    RecipeSuccessorMeasurements,
    RecipeResizeSuccessorInput,
    RecipeResizeSuccessorMeasurements,
    RecipeResourceOrderInput,
    RecipeResourceOrderMeasurements,
    PiercingIntersectionInput,
    PiercingIntersectionMeasurements,
    PiercingPushSuccessorInput,
    PiercingPushSuccessorMeasurements,
    PiercingPierceSuccessorInput,
    PiercingPierceSuccessorMeasurements,
    PiercingWithdrawSuccessorInput,
    PiercingWithdrawSuccessorMeasurements,
    PiercingClearanceInput,
    PiercingClearanceMeasurements,
    InverseMeanAnchorDomainInput,
    InverseMeanAnchorDomainMeasurements,
    InverseAnchorSuccessorInput,
    InverseAnchorSuccessorMeasurements,
    InverseAnchorTypedClearanceInput,
    InverseAnchorTypedClearanceMeasurements,
    InverseAnchorWaypointCandidatesInput,
    InverseAnchorWaypointCandidatesMeasurements,
    OrderedReliefPoseInput,
    OrderedReliefPoseMeasurements,
    OrderedReliefTransitionInput,
    OrderedReliefTransitionMeasurements,
    OrderedReliefWriteInput,
    OrderedReliefWriteMeasurements,
    DirectedFlowColliderUnionInput,
    DirectedFlowColliderUnionMeasurements,
    DirectedFlowJointTickInput,
    DirectedFlowJointTickMeasurements,
    DirectedFlowRecoveryInput,
    DirectedFlowRecoveryMeasurements,
    LocalFieldAcceptanceWindowInput,
    LocalFieldAcceptanceWindowMeasurements,
    LocalFieldAutonomousSuccessorInput,
    LocalFieldAutonomousSuccessorMeasurements,
    LocalFieldSelectiveDomainInput,
    LocalFieldSelectiveDomainMeasurements,
    LocalFieldTransactionInput,
    LocalFieldTransactionMeasurements,
    ReconfigurableConnectionDomainInput,
    ReconfigurableConnectionDomainMeasurements,
    ReconfigurableRigidGeometryInput,
    ReconfigurableRigidGeometryMeasurements,
    ReconfigurableSupportPacketInput,
    ReconfigurableSupportPacketMeasurements,
    LocalConstraintCompositionAnalysis,
    LocalConstraintCompositionInput,
    LocalConstraintRepairAgendaInput,
    MultiResolutionViewInput,
    MultiResolutionViewMeasurements,
    MultiIdentityOpposedRoutesAnalysis,
    MultiIdentityOpposedRoutesInput,
    IntermediateSceneChangeInput,
    IntermediateSceneChangeMeasurements,
    PaletteCanonicalizationInput,
    PaletteCanonicalizationResult,
    SceneRebindingMeasurementInput,
    SceneRebindingMeasurements,
    QuantitySimulationInput,
    QuantitySimulationResult,
    RationalAnchorDependenceInput,
    RationalAnchorDependenceMeasurements,
    QuantizedPlanReconciliationAnalysis,
    QuantizedPlanReconciliationInput,
    ReachabilityInput,
    ReachabilityResult,
    RelationGraphInput,
    RelationGraphResult,
    RepeatedBoundaryDecreaseAnalysis,
    RepeatedBoundaryDecreaseAnalysisInput,
    ExternalTransportObservationInput,
    ExternalTransportObservations,
    ResourceGaugeQuantityAnalysis,
    ResourceGaugeQuantityAnalysisInput,
    RouteCompatibleGraphPlanInput,
    RevisionedCapabilityRequest,
    SceneTransitionMeasurementInput,
    SceneTransitionMeasurements,
    ExactMulticellPeerRelationInput,
    ExactMulticellPeerRelationMeasurements,
    TemporalTrackingInput,
    TemporalTrackingResult,
    TemporalConcurrencyInput,
    TemporalConcurrencyMeasurements,
    TemporalReplaySequenceInput,
    TemporalReplaySequenceMeasurements,
    TransitionEvidenceLedger,
    TransitionEvidenceLedgerInput,
    TransitionRetrodictionInput,
    TransitionRetrodictionMeasurements,
    TransitionPhenomenonInventoryInput,
    TransitionPhenomenonInventoryMeasurements,
    ViewportSupportTerrainInput,
    ViewportSupportTerrainMeasurements,
    UniformBlockReductionInput,
    UniformBlockReductionResult,
    VerifiedAlignmentActionInput,
    VisualSceneDescription,
    VisualSceneInput,
)
from agents.yf_arc3_v5.capabilities.frame import (
    canonicalize_palette,
    measure_scene_rebinding,
    difference_frames,
    enumerate_grid_partitions,
    enumerate_periodic_cell_grids,
    normalize_frame,
    reduce_uniform_blocks,
)
from agents.yf_arc3_v5.capabilities.elimination import measure_elimination_viability, measure_relational_elimination_graph
from agents.yf_arc3_v5.capabilities.geometry import (
    describe_component_geometry,
    measure_derived_geometry_states,
    measure_orthogonal_perimeter_assemblies,
)
from agents.yf_arc3_v5.capabilities.goals import measure_declared_goal_gaps
from agents.yf_arc3_v5.capabilities.goal_graph import (
    project_declared_goal_plan_alternatives,
)
from agents.yf_arc3_v5.capabilities.interaction import (
    advance_committed_verified_alignment_cursor,
    enumerate_interaction_probes,
    enumerate_verified_alignment_actions,
)
from agents.yf_arc3_v5.capabilities.inverse_anchor import (
    enumerate_inverse_mean_anchor_domain,
    measure_rational_anchor_dependence,
)
from agents.yf_arc3_v5.capabilities.multiscale import (
    analyze_local_constraint_compositions,
    analyze_homologous_repairs,
    enumerate_local_constraint_repair_actions,
    enumerate_homologous_repair_actions,
    reconcile_homologous_repair,
)
from agents.yf_arc3_v5.capabilities.opposed_routes import (
    simulate_multi_identity_opposed_routes,
)
from agents.yf_arc3_v5.capabilities.articulated_channel_kinematics import (
    measure_articulated_kinematics,
    measure_articulated_shared_snapshot,
    measure_articulated_successor,
)
from agents.yf_arc3_v5.capabilities.typed_recursive_program import (
    measure_typed_assignment_domains,
    measure_typed_edit_successor,
    measure_typed_interpreter,
)
from agents.yf_arc3_v5.capabilities.recipe_operator_compilation import (
    measure_recipe_pattern_difference,
    measure_recipe_resize_successor,
    measure_recipe_resource_order,
    measure_recipe_successor,
)
from agents.yf_arc3_v5.capabilities.piercing_rooted_order import (
    measure_piercing_clearance,
    measure_piercing_intersections,
    measure_piercing_pierce_successor,
    measure_piercing_push_successor,
    measure_piercing_withdraw_successor,
)
from agents.yf_arc3_v5.capabilities.inverse_anchor_signature_extensions import (
    measure_inverse_anchor_successor,
    measure_inverse_anchor_typed_clearance,
    measure_inverse_anchor_waypoint_candidates,
)
from agents.yf_arc3_v5.capabilities.ordered_relief_writing import (
    apply_ordered_relief_write,
    measure_ordered_relief_pose,
    measure_ordered_relief_transition,
)
from agents.yf_arc3_v5.capabilities.reconfigurable_support import (
    enumerate_reconfigurable_connection_domain,
    measure_reconfigurable_rigid_geometry,
    measure_reconfigurable_support_packet,
)
from agents.yf_arc3_v5.capabilities.directed_accretive_flow import (
    measure_directed_flow_collider_union,
    measure_directed_flow_joint_tick,
    measure_directed_flow_recovery,
)
from agents.yf_arc3_v5.capabilities.local_field_type_production import (
    measure_local_field_acceptance_windows,
    measure_local_field_autonomous_successors,
    measure_local_field_selective_domains,
    measure_local_field_transaction,
)
from agents.yf_arc3_v5.capabilities.local_field_inventory import (
    ArrayRegionMeasurementRequest, ArrayRegionMeasurements, measure_array_adjacent_regions,
)
from agents.yf_arc3_v5.capabilities.local_field_pair_geometry import (
    PairPointGeometryRequest, PairPointGeometryMeasurements, measure_pair_point_geometry,
)
from agents.yf_arc3_v5.capabilities.local_field_stock_change import (
    StockChangeRequest, StockChangeMeasurements, measure_stock_change,
)
from agents.yf_arc3_v5.capabilities.local_field_delivery import (
    FilledContourRequest, FilledContourMeasurements, measure_filled_contour_correspondence,
    PointProgressRequest, PointProgressMeasurements, measure_point_progress,
)
from agents.yf_arc3_v5.capabilities.consumptive_dependency import (
    ConsumptiveDependencyRequest, ConsumptiveDependencyMeasurements, measure_consumptive_dependency,
)
from agents.yf_arc3_v5.capabilities.planning import (
    advance_committed_quantized_plan,
    analyze_quantized_plan_reconciliation,
    enumerate_transferred_quantized_plans,
)
from agents.yf_arc3_v5.capabilities.relations import (
    analyze_relation_graph,
    measure_interaction_topology,
)
from agents.yf_arc3_v5.capabilities.retrodiction import (
    compact_transition_evidence,
    measure_transition_retrodiction,
)
from agents.yf_arc3_v5.capabilities.retrospection import (
    measure_transition_phenomenon_inventory,
)
from agents.yf_arc3_v5.capabilities.routing import (
    finite_grid_reachability,
    measure_viewport_support_terrain,
)
from agents.yf_arc3_v5.capabilities.route_compatible_graphs import (
    analyze_route_compatible_orders,
    enumerate_route_compatible_graph_plans,
)
from agents.yf_arc3_v5.capabilities.sprites import (
    describe_block_grid_scene,
    describe_visual_scene,
)
from agents.yf_arc3_v5.capabilities.tracking import (
    measure_exact_multicell_peer_relations,
    measure_temporal_concurrency,
    measure_temporal_replay_sequence,
    track_components,
)
from agents.yf_arc3_v5.capabilities.temporal_routes import (
    TemporalJointRouteInput,
    TemporalJointRouteMeasurements,
    measure_temporal_joint_routes,
)
from agents.yf_arc3_v5.capabilities.transition import (
    analyze_controlled_transition,
    analyze_goal_completion,
    analyze_repeated_boundary_decrease,
    measure_intermediate_scene_change,
    measure_scene_transition,
)
from agents.yf_arc3_v5.capabilities.resource_quantities import analyze_resource_gauge_quantities
from agents.yf_arc3_v5.capabilities.external_transport import measure_external_transport
from agents.yf_arc3_v5.capabilities.views import measure_multi_resolution_views
from agents.yf_arc3_v5.logos.operations import CapabilityDefinition
from agents.yf_arc3_v5.logos.types import (
    EffectClass,
    FrozenModel,
    schema_ordered_digest,
    stable_digest,
)
from agents.yf_arc3_v5.performance import begin_profile, finish_profile


class CapabilityRegistrationError(ValueError):
    """A capability implementation does not satisfy its declared contract."""


class CapabilityResolutionError(LookupError):
    """No exact registered capability matches an invocation."""


CapabilityImplementation = Callable[..., FrozenModel]


@dataclass(frozen=True)
class RegisteredCapability:
    definition: CapabilityDefinition
    input_model: type[FrozenModel]
    output_model: type[FrozenModel]
    implementation: CapabilityImplementation


class CapabilityRegistry:
    """Explicit registry with a bounded exact-input cache.

    A pure capability result depends on its source definition and immutable
    typed input, not on the controller's monotonically increasing semantic
    revision.  The revision remains provenance on the returned execution, but
    it is deliberately not part of the reuse key: a delta that changed no
    capability input must not force the same perception to run again.
    """

    _MAX_ACTIVE_REVISION_CACHE_ENTRIES = 512

    def __init__(self) -> None:
        self._entries: dict[str, RegisteredCapability] = {}
        self._cache: dict[str, CapabilityExecution] = {}
        self._input_digests: dict[
            int, tuple[weakref.ReferenceType[FrozenModel], str]
        ] = {}

    def register(self, entry: RegisteredCapability) -> None:
        definition = entry.definition
        if definition.effect_class is not EffectClass.PURE:
            raise CapabilityRegistrationError(
                "M5 computational capabilities must be pure"
            )
        existing = self._entries.get(definition.id)
        if existing is not None and existing != entry:
            raise CapabilityRegistrationError(
                f"capability already registered: {definition.id}"
            )
        self._entries[definition.id] = entry

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    @property
    def entries(self) -> tuple[RegisteredCapability, ...]:
        return tuple(self._entries[capability_id] for capability_id in self.ids())

    def resolve(self, capability_id: str) -> RegisteredCapability:
        entry = self._entries.get(capability_id)
        if entry is None:
            raise CapabilityResolutionError(f"unregistered capability: {capability_id}")
        return entry

    def invoke(self, request: RevisionedCapabilityRequest) -> CapabilityExecution:
        started = begin_profile()
        cache_size_before = len(self._cache) if started is not None else 0
        try:
            return self._invoke(request)
        finally:
            if started is not None:
                finish_profile(
                    started,
                    kind="capability",
                    name=request.capability_id,
                    metadata={
                        "cache_entries_added": max(
                            0, len(self._cache) - cache_size_before
                        )
                    },
                )

    def _invoke(self, request: RevisionedCapabilityRequest) -> CapabilityExecution:
        entry = self.resolve(request.capability_id)
        if not isinstance(request.payload, entry.input_model):
            raise CapabilityResolutionError(
                f"{request.capability_id} expects {entry.input_model.__name__}, "
                f"got {type(request.payload).__name__}"
            )
        payload_identity = id(request.payload)
        digest_entry = self._input_digests.get(payload_identity)
        if digest_entry is not None and digest_entry[0]() is request.payload:
            input_digest = digest_entry[1]
        else:
            input_digest = schema_ordered_digest(request.payload)
            self._input_digests[payload_identity] = (
                weakref.ref(
                    request.payload,
                    lambda _reference, identity=payload_identity: (
                        self._input_digests.pop(identity, None)
                    ),
                ),
                input_digest,
            )
        cache_key = stable_digest(
            {
                "capability_id": request.capability_id,
                "source_hash": entry.definition.source_hash,
                "input_digest": input_digest,
            }
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            if cached.input_revision == request.input_revision:
                return cached
            # Keep the exact immutable output and cache key while exposing the
            # current request revision as provenance.  model_copy updates only
            # the small envelope; it does not copy the raster-derived output.
            return cached.model_copy(update={"input_revision": request.input_revision})
        output = entry.implementation(request.payload)
        if not isinstance(output, entry.output_model):
            raise CapabilityResolutionError(
                f"{request.capability_id} returned {type(output).__name__}, "
                f"expected {entry.output_model.__name__}"
            )
        execution = CapabilityExecution(
            definition=entry.definition,
            input_revision=request.input_revision,
            input_digest=input_digest,
            cache_key=cache_key,
            output=output,
        )
        self._cache[cache_key] = execution
        while len(self._cache) > self._MAX_ACTIVE_REVISION_CACHE_ENTRIES:
            self._cache.pop(next(iter(self._cache)))
        return execution


def build_default_capability_registry() -> CapabilityRegistry:
    registry = CapabilityRegistry()
    for entry in _production_entries():
        registry.register(entry)
    return registry


def registered_capability_ids() -> tuple[str, ...]:
    return tuple(sorted(entry.definition.id for entry in _production_entries()))


def _production_entries() -> tuple[RegisteredCapability, ...]:
    return (
        _entry(
            capability_id="capability.bearer_appearance_measurement",
            implementation=measure_bearer_appearance,
            input_model=BearerAppearanceInput,
            output_model=BearerAppearanceMeasurement,
            source_file="appearance.py",
            applicability=("finite_explicit_bearer_support", "declared_appearance_key_contract"),
            invariants=("no_role_assignment", "no_old_position_in_signature", "exact_gcd_proportions"),
            falsifiers=("support_bound_exceeded", "unattested_structural_descriptor"),
        ),
        _entry(
            capability_id="capability.rational_anchor_dependence_measurement",
            implementation=measure_rational_anchor_dependence,
            input_model=RationalAnchorDependenceInput,
            output_model=RationalAnchorDependenceMeasurements,
            source_file="inverse_anchor.py",
            applicability=("source_supplied_finite_rational_dependence_and_quantizer",),
            invariants=(
                "exact_fraction_is_preserved_before_quantization",
                "quantizer_is_supplied_not_inferred",
                "no_anchor_configuration_is_selected",
            ),
            falsifiers=("missing_coefficient", "non_positive_denominator"),
        ),
        _entry(
            capability_id="capability.inverse_mean_anchor_domain_enumeration",
            implementation=enumerate_inverse_mean_anchor_domain,
            input_model=InverseMeanAnchorDomainInput,
            output_model=InverseMeanAnchorDomainMeasurements,
            source_file="inverse_anchor.py",
            applicability=("established_mean_and_bounded_integer_candidate_domain",),
            invariants=(
                "every_in_bound_compatible_position_is_returned",
                "quantizer_is_supplied_not_inferred",
                "compatible_positions_are_not_ranked_or_selected",
            ),
            falsifiers=("incomplete_fixed_member_set", "candidate_domain_exceeds_bound"),
        ),
        _entry(
            capability_id="capability.inverse_anchor_typed_clearance_measurement",
            implementation=measure_inverse_anchor_typed_clearance,
            input_model=InverseAnchorTypedClearanceInput,
            output_model=InverseAnchorTypedClearanceMeasurements,
            source_file="inverse_anchor_signature_extensions.py",
            applicability=("source_supplied_typed_footprints_terrain_and_path_policy",),
            invariants=(
                "anchor_and_dependent_legality_are_measured_separately",
                "unmeasured_swept_path_returns_unknown",
                "no_destination_is_selected_by_python",
            ),
            falsifiers=("incomplete_footprint_packet", "clearance_bound_reached"),
        ),
        _entry(
            capability_id="capability.inverse_anchor_successor_measurement",
            implementation=measure_inverse_anchor_successor,
            input_model=InverseAnchorSuccessorInput,
            output_model=InverseAnchorSuccessorMeasurements,
            source_file="inverse_anchor_signature_extensions.py",
            applicability=("source_supplied_system_operator_and_complete_result_packet",),
            invariants=(
                "control_transfer_and_placement_are_distinct",
                "affected_system_identity_is_preserved",
                "unaffected_systems_are_copied_exactly",
            ),
            falsifiers=("operator_ambiguous", "result_packet_incomplete"),
        ),
        _entry(
            capability_id="capability.inverse_anchor_waypoint_candidate_measurement",
            implementation=measure_inverse_anchor_waypoint_candidates,
            input_model=InverseAnchorWaypointCandidatesInput,
            output_model=InverseAnchorWaypointCandidatesMeasurements,
            source_file="inverse_anchor_signature_extensions.py",
            applicability=("source_supplied_bounded_corridor_waypoint_witnesses",),
            invariants=(
                "global_candidate_count_does_not_exceed_three",
                "supporting_anchor_configuration_remains_attached",
                "python_does_not_select_a_waypoint_or_route",
            ),
            falsifiers=("corridor_constraints_missing", "global_candidate_bound_reached"),
        ),
        _entry(
            capability_id="capability.articulated_kinematics_measurement",
            implementation=measure_articulated_kinematics,
            input_model=ArticulatedKinematicsInput,
            output_model=ArticulatedKinematicsMeasurements,
            source_file="articulated_channel_kinematics.py",
            applicability=("source_supplied_bounded_attachment_forest_and_local_frames",),
            invariants=(
                "local_frames_are_composed_exactly_before_raster_measurement",
                "roots_free_ends_and_markers_remain_distinct",
                "python_does_not_select_a_channel_receiver_or_action",
            ),
            falsifiers=("attachment_cycle", "body_bound_reached"),
        ),
        _entry(
            capability_id="capability.articulated_shared_snapshot_measurement",
            implementation=measure_articulated_shared_snapshot,
            input_model=ArticulatedSharedSnapshotInput,
            output_model=ArticulatedSharedSnapshotMeasurements,
            source_file="articulated_channel_kinematics.py",
            applicability=("source_supplied_local_operations_from_one_pre_action_snapshot",),
            invariants=(
                "only_repeated_reports_of_the_same_cause_and_receiver_are_deduplicated",
                "distinct_causes_are_composed_in_declared_order",
                "missing_composition_order_remains_ambiguous",
            ),
            falsifiers=("conflicting_repeated_cause", "operation_bound_reached"),
        ),
        _entry(
            capability_id="capability.articulated_successor_measurement",
            implementation=measure_articulated_successor,
            input_model=ArticulatedSuccessorInput,
            output_model=ArticulatedSuccessorMeasurements,
            source_file="articulated_channel_kinematics.py",
            applicability=("source_supplied_complete_before_after_and_transaction_packets",),
            invariants=(
                "all_extent_pose_and_marker_changes_share_one_joint_delta",
                "atomic_rejection_or_restoration_keeps_the_before_packet",
                "partial_receiver_commits_are_not_invented",
            ),
            falsifiers=("body_identity_changed", "transaction_packet_inconsistent"),
        ),
        _entry(
            capability_id="capability.typed_assignment_domains_measurement",
            implementation=measure_typed_assignment_domains,
            input_model=TypedAssignmentDomainsInput,
            output_model=TypedAssignmentDomainsMeasurements,
            source_file="typed_recursive_program.py",
            applicability=("source_supplied_typed_tokens_sockets_and_definition_identities",),
            invariants=(
                "physical_token_occurrences_remain_distinct",
                "slot_types_and_call_targets_are_checked_separately",
                "python_does_not_select_or_compose_an_assignment",
            ),
            falsifiers=("typed_inventory_incomplete", "domain_member_bound_reached"),
        ),
        _entry(
            capability_id="capability.typed_edit_successor_measurement",
            implementation=measure_typed_edit_successor,
            input_model=TypedEditSuccessorInput,
            output_model=TypedEditSuccessorMeasurements,
            source_file="typed_recursive_program.py",
            applicability=("source_supplied_complete_occupancy_selection_and_one_edit",),
            invariants=(
                "one_physical_token_occurrence_has_one_location",
                "an_exchange_returns_the_displaced_token_to_the_actual_source",
                "a_socket_click_without_selection_remains_unchanged_not_inert",
            ),
            falsifiers=("occupancy_packet_incomplete", "edit_target_absent"),
        ),
        _entry(
            capability_id="capability.typed_interpreter_measurement",
            implementation=measure_typed_interpreter,
            input_model=TypedInterpreterInput,
            output_model=TypedInterpreterMeasurements,
            source_file="typed_recursive_program.py",
            applicability=("source_supplied_typed_definitions_reference_boundary_and_bounds",),
            invariants=(
                "literal_emission_and_call_entry_remain_distinct",
                "return_stack_program_counter_and_output_index_remain_distinct",
                "prefix_match_termination_and_official_acceptance_remain_distinct",
            ),
            falsifiers=("call_target_absent", "independent_bound_reached"),
        ),
        _entry(capability_id="capability.recipe_pattern_difference_measurement", implementation=measure_recipe_pattern_difference, input_model=RecipePatternDifferenceInput, output_model=RecipePatternDifferenceMeasurements, source_file="recipe_operator_compilation.py", applicability=("source_supplied_lattice_current_target_and_edit_modality",), invariants=("input_and_world_lattices_remain_distinct", "already_entered_cells_are_preserved", "python_does_not_synthesize_a_missing_modality"), falsifiers=("lattice_mapping_incomplete", "edit_bound_reached")),
        _entry(capability_id="capability.recipe_successor_measurement", implementation=measure_recipe_successor, input_model=RecipeSuccessorInput, output_model=RecipeSuccessorMeasurements, source_file="recipe_operator_compilation.py", applicability=("source_supplied_recipe_edit_trigger_world_operation_and_reset",), invariants=("one_edit_precedes_trigger_evaluation", "world_operation_occurs_only_on_the_supplied_trigger", "input_reset_and_world_effect_share_one_successor"), falsifiers=("trigger_predicate_missing", "world_operation_packet_incomplete")),
        _entry(capability_id="capability.recipe_resize_successor_measurement", implementation=measure_recipe_resize_successor, input_model=RecipeResizeSuccessorInput, output_model=RecipeResizeSuccessorMeasurements, source_file="recipe_operator_compilation.py", applicability=("source_supplied_before_after_body_masks_and_world_layers",), invariants=("anchor_size_heading_and_mask_remain_distinct", "whole_body_contacts_are_measured", "python_does_not_choose_a_reanchoring_rule"), falsifiers=("body_mask_incomplete", "world_layer_packet_incomplete")),
        _entry(capability_id="capability.recipe_resource_order_measurement", implementation=measure_recipe_resource_order, input_model=RecipeResourceOrderInput, output_model=RecipeResourceOrderMeasurements, source_file="recipe_operator_compilation.py", applicability=("source_supplied_initial_quantity_and_ordered_events",), invariants=("cost_pickup_and_refill_order_is_preserved", "insufficient_cost_stops_before_mutation", "python_does_not_choose_a_route"), falsifiers=("event_order_unknown", "resource_event_bound_reached")),
        _entry(capability_id="capability.piercing_intersection_measurement", implementation=measure_piercing_intersections, input_model=PiercingIntersectionInput, output_model=PiercingIntersectionMeasurements, source_file="piercing_rooted_order.py", applicability=("source_supplied_rods_pieces_root_and_reference_order",), invariants=("intersection_is_independent_of_visible_layering", "order_is_read_from_each_declared_root", "unused_stock_remains_unconstrained"), falsifiers=("rod_or_piece_geometry_incomplete", "root_identity_unknown")),
        _entry(capability_id="capability.piercing_push_successor_measurement", implementation=measure_piercing_push_successor, input_model=PiercingPushSuccessorInput, output_model=PiercingPushSuccessorMeasurements, source_file="piercing_rooted_order.py", applicability=("source_supplied_contacted_subset_mobility_and_translation",), invariants=("only_supplied_movable_contacted_pieces_translate", "blocked_piece_identity_is_preserved", "support_meaning_is_not_inferred"), falsifiers=("contacted_subset_incomplete", "push_rule_unknown")),
        _entry(capability_id="capability.piercing_pierce_successor_measurement", implementation=measure_piercing_pierce_successor, input_model=PiercingPierceSuccessorInput, output_model=PiercingPierceSuccessorMeasurements, source_file="piercing_rooted_order.py", applicability=("source_supplied_after_rod_entry_face_and_immobilization",), invariants=("entry_face_and_intersection_are_both_required", "existing_supports_are_preserved", "python_does_not_generalize_entry_direction"), falsifiers=("entry_face_unknown", "immobilization_unknown")),
        _entry(capability_id="capability.piercing_withdraw_successor_measurement", implementation=measure_piercing_withdraw_successor, input_model=PiercingWithdrawSuccessorInput, output_model=PiercingWithdrawSuccessorMeasurements, source_file="piercing_rooted_order.py", applicability=("source_supplied_before_after_rods_and_piece_masks",), invariants=("released_pieces_keep_world_position", "multiple_supports_remain_simultaneous", "retained_and_released_relations_are_separate"), falsifiers=("after_rod_geometry_missing", "support_intersection_bound_reached")),
        _entry(capability_id="capability.piercing_clearance_measurement", implementation=measure_piercing_clearance, input_model=PiercingClearanceInput, output_model=PiercingClearanceMeasurements, source_file="piercing_rooted_order.py", applicability=("source_supplied_whole_tool_load_delta_obstacles_and_free_pieces",), invariants=("whole_assembly_is_translated", "free_pieces_can_block", "python_does_not_choose_a_route"), falsifiers=("assembly_mask_incomplete", "world_obstacles_incomplete")),
        _entry(
            capability_id="capability.ordered_relief_transition_measurement",
            implementation=measure_ordered_relief_transition,
            input_model=OrderedReliefTransitionInput,
            output_model=OrderedReliefTransitionMeasurements,
            source_file="ordered_relief_writing.py",
            applicability=("sealed_bounded_packet_with_source_declared_partitions",),
            invariants=(
                "partitions_are_supplied_not_inferred",
                "changed_and_known_same_value_writes_remain_distinct",
                "no_partition_is_promoted_to_a_semantic_role",
            ),
            falsifiers=("overlapping_partitions", "comparison_bound_exceeded"),
        ),
        _entry(
            capability_id="capability.ordered_relief_write_successor",
            implementation=apply_ordered_relief_write,
            input_model=OrderedReliefWriteInput,
            output_model=OrderedReliefWriteMeasurements,
            source_file="ordered_relief_writing.py",
            applicability=("source_supplied_canvas_footprint_and_write_value",),
            invariants=(
                "write_is_last_writer_replacement_on_the_supplied_footprint",
                "non_application_cells_are_preserved",
                "same_value_writes_are_reported_separately",
            ),
            falsifiers=("out_of_bounds_footprint", "canvas_bound_exceeded"),
        ),
        _entry(
            capability_id="capability.ordered_relief_pose_measurement",
            implementation=measure_ordered_relief_pose,
            input_model=OrderedReliefPoseInput,
            output_model=OrderedReliefPoseMeasurements,
            source_file="ordered_relief_writing.py",
            applicability=("source_requested_pose_and_observed_action_graph",),
            invariants=(
                "only_observed_edges_are_traversed",
                "aspect_and_size_are_tested_exactly",
                "no_rotation_action_is_invented",
            ),
            falsifiers=("unobserved_pose", "route_depth_bound_reached"),
        ),
        _entry(
            capability_id="capability.reconfigurable_support_packet_measurement",
            implementation=measure_reconfigurable_support_packet,
            input_model=ReconfigurableSupportPacketInput,
            output_model=ReconfigurableSupportPacketMeasurements,
            source_file="reconfigurable_support.py",
            applicability=("source_supplied_ordered_support_packet_and_connection_question",),
            invariants=(
                "actor_support_uses_the_complete_supplied_footprint",
                "support_owners_remain_separate_inside_the_union",
                "restoration_requires_a_measured_present_absent_present_sequence",
            ),
            falsifiers=("unstable_actor_footprint", "actor_pose_bound_reached"),
        ),
        _entry(
            capability_id="capability.reconfigurable_rigid_geometry_measurement",
            implementation=measure_reconfigurable_rigid_geometry,
            input_model=ReconfigurableRigidGeometryInput,
            output_model=ReconfigurableRigidGeometryMeasurements,
            source_file="reconfigurable_support.py",
            applicability=("source_supplied_masks_support_layers_and_pivot_candidates",),
            invariants=(
                "every_supplied_pivot_and_quarter_turn_is_tested_exactly",
                "pivot_candidates_are_not_ranked_or_selected",
                "staging_support_is_measured_before_and_after_separately",
            ),
            falsifiers=("no_exact_rigid_fit", "supplied_staging_mask_is_unsupported"),
        ),
        _entry(
            capability_id="capability.reconfigurable_connection_domain_enumeration",
            implementation=enumerate_reconfigurable_connection_domain,
            input_model=ReconfigurableConnectionDomainInput,
            output_model=ReconfigurableConnectionDomainMeasurements,
            source_file="reconfigurable_support.py",
            applicability=("source_supplied_bounded_support_configurations_and_terminal_footprint",),
            invariants=(
                "every_route_pose_is_supported_by_the_complete_active_union",
                "only_the_first_deterministic_shortest_witness_per_configuration_is_kept",
                "no_more_than_three_global_witnesses_are_returned",
            ),
            falsifiers=("actor_pose_bound_reached", "requested_connection_absent"),
        ),
        _entry(
            capability_id="capability.directed_flow_collider_union_measurement",
            implementation=measure_directed_flow_collider_union,
            input_model=DirectedFlowColliderUnionInput,
            output_model=DirectedFlowColliderUnionMeasurements,
            source_file="directed_accretive_flow.py",
            applicability=("source_supplied_bounded_directed_flow_collider_masks",),
            invariants=(
                "collider_identities_remain_separate_inside_the_union",
                "control_and_attachment_refs_are_preserved",
                "no_collider_role_is_inferred_from_palette",
            ),
            falsifiers=("collider_mask_missing", "collider_bound_reached"),
        ),
        _entry(
            capability_id="capability.directed_flow_joint_tick_measurement",
            implementation=measure_directed_flow_joint_tick,
            input_model=DirectedFlowJointTickInput,
            output_model=DirectedFlowJointTickMeasurements,
            source_file="directed_accretive_flow.py",
            applicability=("source_supplied_multiple_fronts_trace_and_bounded_schedules",),
            invariants=(
                "every_supplied_front_advances_one_cardinal_cell",
                "source_lineage_is_preserved_at_trace_merges",
                "all_supplied_schedules_are_returned_without_selection",
            ),
            falsifiers=("single_front_packet", "resulting_trace_bound_reached"),
        ),
        _entry(
            capability_id="capability.directed_flow_recovery_measurement",
            implementation=measure_directed_flow_recovery,
            input_model=DirectedFlowRecoveryInput,
            output_model=DirectedFlowRecoveryMeasurements,
            source_file="directed_accretive_flow.py",
            applicability=("source_supplied_failed_release_and_recovery_packet",),
            invariants=(
                "configuration_trace_receptacle_and_quantities_are_compared_separately",
                "quantity_identities_are_preserved",
                "no_reset_semantics_are_inferred_by_python",
            ),
            falsifiers=("failed_release_absent", "recovery_identity_packet_incomplete"),
        ),
        _entry(
            capability_id="capability.local_field_selective_domain_measurement",
            implementation=measure_local_field_selective_domains,
            input_model=LocalFieldSelectiveDomainInput,
            output_model=LocalFieldSelectiveDomainMeasurements,
            source_file="local_field_type_production.py",
            applicability=("source_supplied_required_excluded_bodies_and_bounded_click_candidates",),
            invariants=(
                "required_and_excluded_body_refs_remain_disjoint",
                "boundary_contacts_remain_uncertain",
                "no_click_candidate_is_selected_by_python",
            ),
            falsifiers=("candidate_domain_absent", "candidate_bound_reached"),
        ),
        _entry(
            capability_id="capability.array_adjacent_region_measurement",
            implementation=measure_array_adjacent_regions,
            input_model=ArrayRegionMeasurementRequest,
            output_model=ArrayRegionMeasurements,
            source_file="local_field_inventory.py",
            applicability=("source_supplied_array_and_component_bounds",),
            invariants=("no_hud_stock_or_terminal_role_is_assigned", "all_ambiguous_patterns_remain_measured"),
            falsifiers=("pattern_enumeration_incomplete", "array_or_adjacent_region_absent"),
        ),
        _entry(
            capability_id="capability.consumptive_dependency_measurement",
            implementation=measure_consumptive_dependency,
            input_model=ConsumptiveDependencyRequest,
            output_model=ConsumptiveDependencyMeasurements,
            source_file="consumptive_dependency.py",
            applicability=("source_supplied_rewrites_stock_and_signature_substitution",),
            invariants=("no_consumed_identity_supplies_two_inputs", "symbolic_witness_is_not_a_physical_route"),
            falsifiers=("enumeration_incomplete", "supplied_stock_has_no_symbolic_witness"),
        ),
        _entry(
            capability_id="capability.pair_point_geometry_measurement",
            implementation=measure_pair_point_geometry,
            input_model=PairPointGeometryRequest,
            output_model=PairPointGeometryMeasurements,
            source_file="local_field_pair_geometry.py",
            applicability=("source_supplied_footprints_signature_pair_region_and_radius",),
            invariants=("all_other_current_footprints_are_excluded", "no_radius_or_rewrite_is_confirmed"),
            falsifiers=("point_enumeration_incomplete", "strict_pair_or_progress_domain_absent"),
        ),
        _entry(
            capability_id="capability.stock_change_measurement",
            implementation=measure_stock_change,
            input_model=StockChangeRequest,
            output_model=StockChangeMeasurements,
            source_file="local_field_stock_change.py",
            applicability=("source_supplied_before_after_stock_and_distinct_participants",),
            invariants=("no_recipe_or_action_is_selected", "physical_identity_and_signature_are_compared_separately"),
            falsifiers=("duplicate_input_identity", "incomplete_stock_snapshot"),
        ),
        _entry(
            capability_id="capability.filled_contour_correspondence",
            implementation=measure_filled_contour_correspondence,
            input_model=FilledContourRequest, output_model=FilledContourMeasurements,
            source_file="local_field_delivery.py",
            applicability=("source_supplied_observed_point_and_current_component_scope",),
            invariants=("filled_holes_do_not_erase_open_concavities", "no_palette_assigns_a_destination"),
            falsifiers=("contour_enumeration_incomplete", "current_correspondence_absent_or_ambiguous"),
        ),
        _entry(
            capability_id="capability.point_progress_measurement",
            implementation=measure_point_progress,
            input_model=PointProgressRequest, output_model=PointProgressMeasurements,
            source_file="local_field_delivery.py",
            applicability=("source_supplied_body_exclusions_point_region_and_radius",),
            invariants=("all_strict_progress_points_are_preserved", "no_prior_body_displacement_is_copied"),
            falsifiers=("point_enumeration_incomplete", "strict_progress_domain_absent"),
        ),
        _entry(
            capability_id="capability.local_field_autonomous_successor_measurement",
            implementation=measure_local_field_autonomous_successors,
            input_model=LocalFieldAutonomousSuccessorInput,
            output_model=LocalFieldAutonomousSuccessorMeasurements,
            source_file="local_field_type_production.py",
            applicability=("source_supplied_named_events_occurrences_and_complete_schedules",),
            invariants=(
                "each_occurrence_is_advanced_at_most_once",
                "every_schedule_contains_every_occurrence",
                "alternative_orders_are_returned_without_selection",
            ),
            falsifiers=("unnamed_event", "duplicate_occurrence_advancement"),
        ),
        _entry(
            capability_id="capability.local_field_acceptance_window_measurement",
            implementation=measure_local_field_acceptance_windows,
            input_model=LocalFieldAcceptanceWindowInput,
            output_model=LocalFieldAcceptanceWindowMeasurements,
            source_file="local_field_type_production.py",
            applicability=("source_supplied_required_predicates_ordered_samples_and_acceptance_rules",),
            invariants=(
                "all_required_predicates_are_evaluated_jointly",
                "surplus_and_shared_occupancy_follow_supplied_rules",
                "prediction_stops_at_the_first_internal_interruption",
            ),
            falsifiers=("required_predicates_absent", "sample_bound_reached"),
        ),
        _entry(
            capability_id="capability.local_field_transaction_measurement",
            implementation=measure_local_field_transaction,
            input_model=LocalFieldTransactionInput,
            output_model=LocalFieldTransactionMeasurements,
            source_file="local_field_type_production.py",
            applicability=("source_supplied_before_transient_after_body_and_quantity_packet",),
            invariants=(
                "body_type_and_quantity_identities_are_preserved",
                "transient_and_final_deltas_are_reported_separately",
                "python_does_not_assign_commit_or_rollback_meaning",
            ),
            falsifiers=("identity_packet_incomplete", "post_event_packet_bound_reached"),
        ),
        _entry(
            capability_id="capability.abductive_version_space_measurement",
            implementation=measure_abductive_version_space,
            input_model=AbductiveVersionSpaceInput,
            output_model=AbductiveVersionSpaceMeasurements,
            source_file="abduction.py",
            applicability=("bounded_declared_explanation_obligation_sets",),
            invariants=("rejects_missing_or_contradictory_explanations", "no_weighted_selection"),
            falsifiers=("unbounded_explanation_set", "missing_falsifier_contract"),
        ),
        _entry(
            capability_id="capability.display_correspondence_measurements",
            implementation=measure_display_correspondence,
            input_model=DisplayCorrespondenceInput,
            output_model=DisplayCorrespondenceMeasurements,
            source_file="display_correspondence.py",
            applicability=("finite_frames_and_source_supplied_identity_descriptors",),
            invariants=("no_palette_role_binding", "bounded_exact_correspondence_rows_only"),
            falsifiers=("ambiguous_or_incomplete_geometry",),
        ),
        _entry(
            capability_id="capability.attached_rotation_analysis",
            implementation=analyze_attached_rotations,
            input_model=AttachedRotationAnalysisInput,
            output_model=AttachedRotationAnalysis,
            source_file="cursor_orientation.py",
            applicability=("current_temporal_tracking_and_supplied_body_identity",),
            invariants=("exact_bounded_part_rotations_only", "no_role_or_direction_binding",),
            falsifiers=("body_identity_or_part_correspondence_is_not_unique",),
        ),
        _entry(
            capability_id="capability.announced_step_measurements",
            implementation=measure_announced_step,
            input_model=AnnouncedStepInput,
            output_model=AnnouncedStepMeasurements,
            source_file="announced_motion.py",
            applicability=("finite_endpoint_and_obstacle_projection",),
            invariants=("no_role_or_model_is_selected", "counterfactual_endpoint_counts_are_separate",
                        "unknown_endpoint_is_not_replaced_by_stationarity", "hard_cardinality_bounds"),
            falsifiers=("missing_identity", "stale_endpoint_projection", "invalid_cardinality"),
        ),
        _entry(
            capability_id="capability.frame_normalization",
            implementation=normalize_frame,
            input_model=FrameNormalizationInput,
            output_model=FrameNormalizationResult,
            source_file="frame.py",
            applicability=("numeric_matrix_or_stack",),
            invariants=(
                "last_frame_selected",
                "pixel_values_preserved",
                "rectangle_required",
            ),
            falsifiers=("empty_payload", "ragged_matrix", "non_numeric_pixel"),
        ),
        _entry(
            capability_id="capability.connected_components",
            implementation=detect_components,
            input_model=ComponentExtractionInput,
            output_model=ComponentExtractionResult,
            source_file="components.py",
            applicability=("finite_rectangular_frame", "connectivity_4_or_8"),
            invariants=(
                "every_pixel_has_exactly_one_component",
                "component_pixels_share_value",
                "no_semantic_role_assigned",
            ),
            falsifiers=("unsupported_connectivity",),
        ),
        _entry(
            capability_id="capability.block_grid_scene_decomposition",
            implementation=describe_block_grid_scene,
            input_model=VisualSceneInput,
            output_model=BlockGridSceneDescription,
            source_file="sprites.py",
            applicability=("source_declared_block_grid_scope",),
            invariants=(
                "blocks_are_exhaustive_4_connected_components",
                "candidate_grids_are_exact_descriptions",
                "unmeasured_object_families_are_not_absence_evidence",
                "no_semantic_role_assigned",
            ),
            falsifiers=("invalid_frame_geometry", "undeclared_scope"),
        ),
        _entry(
            capability_id="capability.visual_scene_decomposition",
            implementation=describe_visual_scene,
            input_model=VisualSceneInput,
            output_model=VisualSceneDescription,
            source_file="sprites.py",
            applicability=("finite_rectangular_frame",),
            invariants=(
                "blocks_are_exhaustive_4_connected_components",
                "zones_are_exhaustive_8_connected_components",
                "sprite_crops_are_exact_and_uncommitted",
                "no_background_or_semantic_role_assigned",
            ),
            falsifiers=("invalid_frame_geometry",),
        ),
        _entry(
            capability_id="capability.known_grid_cell_recount",
            implementation=recount_known_grid_cells,
            input_model=KnownGridCellRecountInput,
            output_model=KnownGridCellRecountResult,
            source_file="cell_recount.py",
            applicability=("source_declared_current_grid_geometry",),
            invariants=(
                "exact_cell_patterns_and_occurrences_only",
                "no_grid_or_semantic_role_selection",
                "prior_cell_delta_is_exact",
            ),
            falsifiers=("geometry_outside_current_frame", "invalid_prior_type_index"),
        ),
        _entry(
            capability_id="capability.known_grid_object_assembly",
            implementation=assemble_known_grid_objects,
            input_model=KnownGridObjectAssemblyInput,
            output_model=KnownGridObjectAssemblyResult,
            source_file="cell_objects.py",
            applicability=("source_declared_retained_grid_recount",),
            invariants=(
                "all_cells_belong_to_exactly_one_object",
                "objects_are_maximal_orthogonally_connected_equal_type_sets",
                "shape_relations_are_linear_indexes_not_object_pairs",
                "no_cell_type_or_semantic_role_is_excluded",
            ),
            falsifiers=("cell_count_mismatch", "object_count_exceeds_hard_bound"),
        ),
        _entry(
            capability_id="capability.multi_resolution_view_measurement",
            implementation=measure_multi_resolution_views,
            input_model=MultiResolutionViewInput,
            output_model=MultiResolutionViewMeasurements,
            source_file="views.py",
            applicability=("visual_scene_description_available",),
            invariants=(
                "existing_descriptors_are_reused_without_pixel_rescan",
                "descriptions_are_bounded_per_family",
                "measurement_output_assigns_no_semantic_role",
            ),
            falsifiers=("visual_scene_description_missing",),
        ),
        _entry(
            capability_id="capability.frame_difference",
            implementation=difference_frames,
            input_model=FrameDifferenceInput,
            output_model=FrameDifferenceResult,
            source_file="frame.py",
            applicability=("equal_frame_dimensions",),
            invariants=("changes_are_exact", "action_reference_is_opaque"),
            falsifiers=("dimension_mismatch",),
        ),
        _entry(
            capability_id="capability.scene_transition_measurement",
            implementation=measure_scene_transition,
            input_model=SceneTransitionMeasurementInput,
            output_model=SceneTransitionMeasurements,
            source_file="transition.py",
            applicability=("equal_frame_dimensions", "action_conditioned_transition"),
            invariants=(
                "both_scene_equivalence_alternatives_are_preserved",
                "possible_nonzero_translation_is_not_erased",
                "no_resource_or_semantic_role_is_established",
            ),
            falsifiers=("dimension_mismatch",),
        ),
        _entry(
            capability_id="capability.intermediate_scene_change_measurement",
            implementation=measure_intermediate_scene_change,
            input_model=IntermediateSceneChangeInput,
            output_model=IntermediateSceneChangeMeasurements,
            source_file="transition.py",
            applicability=(
                "equal_frame_dimensions",
                "committed_continue_step_returned_frame",
            ),
            invariants=(
                "no_component_extraction_or_tracking",
                "ignored_regions_are_previously_grounded_named_entities",
                "raw_changes_partition_exactly_into_ignored_and_meaningful",
                "meaningful_changes_partition_exactly_into_permitted_and_outside_permitted",
                "all_guard_alternatives_are_preserved",
            ),
            falsifiers=("dimension_mismatch", "ungrounded_ignored_region"),
        ),
        _entry(
            capability_id="capability.temporal_component_tracking",
            implementation=track_components,
            input_model=TemporalTrackingInput,
            output_model=TemporalTrackingResult,
            source_file="tracking.py",
            applicability=("descriptive_components_available", "ordered_frames"),
            invariants=(
                "stationary_exact_identity_precedes_motion_matching",
                "unique_exact_translation_preserves_identity",
                "unique_one_axis_extent_transition_preserves_identity",
                "ambiguous_predecessors_remain_explicit",
                "no_semantic_role_assigned",
            ),
            falsifiers=("component_lineage_missing",),
        ),
        _entry(
            capability_id="capability.exact_multicell_peer_relation_measurement",
            implementation=measure_exact_multicell_peer_relations,
            input_model=ExactMulticellPeerRelationInput,
            output_model=ExactMulticellPeerRelationMeasurements,
            source_file="tracking.py",
            applicability=(
                "tracked_components_available",
                "eight_connected_zones_available",
            ),
            invariants=(
                "every_exact_multicell_pair_is_enumerated_within_the_global_bound",
                "same_orientation_is_preserved_by_exact_relative_pixels",
                "eight_connected_assemblies_keep_their_member_identities",
                "common_nonzero_member_translation_is_measured_without_assigning_meaning",
            ),
            falsifiers=(
                "exact_peer_enumeration_truncated",
                "assembly_member_union_differs_from_zone_support",
            ),
        ),
        _entry(
            capability_id="capability.controlled_transition_analysis",
            implementation=analyze_controlled_transition,
            input_model=ControlledTransitionAnalysisInput,
            output_model=ControlledTransitionAnalysis,
            source_file="transition.py",
            applicability=(
                "action_conditioned_transition",
                "temporal_tracking_available",
            ),
            invariants=(
                "all_interpretation_candidates_remain_explicit",
                "quantum_is_measured_from_clicked_component_extent",
                "complementary_extent_changes_are_exact",
                "aggregate_conservation_is_measured",
                "no_mechanism_role_assigned",
            ),
            falsifiers=("causal_lineage_missing", "ambiguous_exchange_pair"),
        ),
        _entry(
            capability_id="capability.goal_completion_analysis",
            implementation=analyze_goal_completion,
            input_model=GoalCompletionAnalysisInput,
            output_model=GoalCompletionAnalysis,
            source_file="transition.py",
            applicability=(
                "verified_goal_reducing_action_was_executed",
                "score_transition_observed",
            ),
            invariants=(
                "completion_requires_observed_score_increase",
                "completion_requires_expected_final_quantum",
                "goal_identity_is_preserved",
                "no_semantic_commitment_is_made",
            ),
            falsifiers=("score_did_not_increase", "more_than_one_step_was_expected"),
        ),
        _entry(
            capability_id="capability.transition_evidence_compaction",
            implementation=compact_transition_evidence,
            input_model=TransitionEvidenceLedgerInput,
            output_model=TransitionEvidenceLedger,
            source_file="retrodiction.py",
            applicability=("finite_observed_transition_history",),
            invariants=(
                "only_exact_observed_signatures_are_grouped",
                "at_most_three_witness_refs_are_retained_per_signature",
                "no_causal_or_semantic_class_is_inferred",
                "input_cardinality_is_hard_bounded",
            ),
            falsifiers=("duplicate_transition_ref", "unbounded_transition_history"),
        ),
        _entry(
            capability_id="capability.transition_retrodiction_measurement",
            implementation=measure_transition_retrodiction,
            input_model=TransitionRetrodictionInput,
            output_model=TransitionRetrodictionMeasurements,
            source_file="retrodiction.py",
            applicability=(
                "finite_observed_transition_history",
                "declared_partial_model_predictions_available",
            ),
            invariants=(
                "coverage_and_exactness_are_reported_separately",
                "first_divergence_follows_observed_sequence_order",
                "unknown_prediction_is_not_counted_as_success_or_failure",
                "no_model_is_selected_revised_or_committed",
                "input_cardinality_is_hard_bounded",
            ),
            falsifiers=("prediction_index_mismatch", "unbounded_transition_history"),
        ),
        _entry(
            capability_id="capability.transition_phenomenon_inventory",
            implementation=measure_transition_phenomenon_inventory,
            input_model=TransitionPhenomenonInventoryInput,
            output_model=TransitionPhenomenonInventoryMeasurements,
            source_file="retrospection.py",
            applicability=("reconciled_ordered_frame_packet",),
            invariants=(
                "every_changed_pixel_is_retained_or_counted_in_overflow",
                "transient_and_settled_changes_remain_distinct",
                "explanation_links_are_source_declared",
                "no_causal_meaning_is_inferred",
                "phenomenon_individuation_is_hard_bounded",
            ),
            falsifiers=("explanation_names_uninventoried_phenomenon",),
        ),
        _entry(
            capability_id="capability.repeated_boundary_decrease_analysis",
            implementation=analyze_repeated_boundary_decrease,
            input_model=RepeatedBoundaryDecreaseAnalysisInput,
            output_model=RepeatedBoundaryDecreaseAnalysis,
            source_file="transition.py",
            applicability=("action_conditioned_transition", "tracked_boundary_component"),
            invariants=(
                "single_observation_remains_distinct_from_repetition",
                "only_exact_collinear_boundary_diminution_is_reported",
                "no_resource_role_is_assigned",
            ),
            falsifiers=("boundary_change_is_not_exact", "component_identity_is_ambiguous"),
        ),
        _entry(
            capability_id="capability.resource_gauge_quantity_analysis",
            implementation=analyze_resource_gauge_quantities,
            input_model=ResourceGaugeQuantityAnalysisInput,
            output_model=ResourceGaugeQuantityAnalysis,
            source_file="resource_quantities.py",
            applicability=("tracked_boundary_rectangle", "source_supplied_rendering_descriptions"),
            invariants=("all_integer_compatible_budgets_are_preserved_as_intervals",
                        "unchanged_raster_observations_use_actual_primitive_ordinals",
                        "unbounded_intervals_do_not_become_exact_quantities",
                        "no_resource_role_or_rendering_model_is_selected"),
            falsifiers=("render_geometry_or_observed_count_contradicts_the_supplied_descriptions",),
        ),
        _entry(
            capability_id="capability.external_transport_observation",
            implementation=measure_external_transport,
            input_model=ExternalTransportObservationInput,
            output_model=ExternalTransportObservations,
            source_file="external_transport.py",
            applicability=("source_supplied_capacity_geometry", "exact_tracked_components"),
            invariants=("no_social_role_or_assignment_is_selected", "ambiguous_identity_is_not_delivery_evidence", "pair_domains_are_bounded"),
            falsifiers=("component_identity_or_geometry_is_contradicted",),
        ),
        _entry(
            capability_id="capability.component_matching",
            implementation=enumerate_component_matches,
            input_model=ComponentMatchInput,
            output_model=ComponentMatchResult,
            source_file="components.py",
            applicability=("descriptive_components_available",),
            invariants=(
                "all_exact_translation_candidates_preserved",
                "ambiguity_not_collapsed",
                "no_role_inference",
            ),
            falsifiers=("value_identity_absent", "relative_shape_identity_absent"),
        ),
        _entry(
            capability_id="capability.palette_canonicalization",
            implementation=canonicalize_palette,
            input_model=PaletteCanonicalizationInput,
            output_model=PaletteCanonicalizationResult,
            source_file="frame.py",
            applicability=("finite_rectangular_frame",),
            invariants=("equality_structure_preserved", "first_occurrence_order_used"),
            falsifiers=("non_rectangular_frame",),
        ),
        _entry(
            capability_id="capability.scene_rebinding_measurement",
            implementation=measure_scene_rebinding,
            input_model=SceneRebindingMeasurementInput,
            output_model=SceneRebindingMeasurements,
            source_file="frame.py",
            applicability=(
                "two_palette_canonicalized_frames",
                "current_temporal_tracking",
                "declared_boundary_handoff",
            ),
            invariants=(
                "only_exact_structure_and_tracking_counts_are_projected",
                "transported_references_are_preserved_without_interpretation",
                "no_correspondence_role_or_commitment_is_assigned",
            ),
            falsifiers=("tracking_frame_does_not_match_current_frame",),
        ),
        _entry(
            capability_id="capability.uniform_block_reduction",
            implementation=reduce_uniform_blocks,
            input_model=UniformBlockReductionInput,
            output_model=UniformBlockReductionResult,
            source_file="frame.py",
            applicability=("exact_block_partition",),
            invariants=("reduction_requires_uniform_blocks", "contradictions_reported"),
            falsifiers=("non_uniform_block", "partial_block"),
        ),
        _entry(
            capability_id="capability.grid_partition_enumeration",
            implementation=enumerate_grid_partitions,
            input_model=GridPartitionInput,
            output_model=GridPartitionResult,
            source_file="frame.py",
            applicability=("finite_rectangular_frame",),
            invariants=("all_exact_divisor_partitions_reported", "no_grid_selected"),
            falsifiers=("required_grid_extent_unavailable",),
        ),
        _entry(
            capability_id="capability.periodic_cell_grid_enumeration",
            implementation=enumerate_periodic_cell_grids,
            input_model=PeriodicCellGridInput,
            output_model=PeriodicCellGridResult,
            source_file="frame.py",
            applicability=("finite_rectangular_frame", "periodic_separator_candidate"),
            invariants=(
                "cell_patterns_and_separator_consistency_are_exact_measurements",
                "at_most_three_descriptions_cross_the_runtime_boundary",
                "no_cell_role_or_grid_meaning_is_assigned",
            ),
            falsifiers=("fewer_than_three_repeated_cells_per_axis",),
        ),
        _entry(
            capability_id="capability.component_geometry",
            implementation=describe_component_geometry,
            input_model=GeometryInput,
            output_model=GeometryResult,
            source_file="geometry.py",
            applicability=("descriptive_components_available",),
            invariants=("pair_relations_are_exact", "no_zone_or_role_assigned"),
            falsifiers=("component_geometry_invalid",),
        ),
        _entry(
            capability_id="capability.orthogonal_perimeter_assembly_measurement",
            implementation=measure_orthogonal_perimeter_assemblies,
            input_model=OrthogonalPerimeterAssemblyInput,
            output_model=OrthogonalPerimeterAssemblyMeasurements,
            source_file="geometry.py",
            applicability=(
                "finite_descriptive_components",
                "repeated_isotropic_component_lattice_candidate",
            ),
            invariants=(
                "closed_perimeter_order_is_an_exact_geometry_measurement",
                "off_lattice_quartets_are_bounded_to_three",
                "shared_values_and_separator_fit_do_not_assign_roles",
                "no_palette_value_is_bound_to_meaning",
            ),
            falsifiers=(
                "perimeter_is_not_closed_or_regular",
                "quartet_is_not_symmetric_or_gap_fitting",
            ),
        ),
        _entry(
            capability_id="capability.derived_geometry_state_measurement",
            implementation=measure_derived_geometry_states,
            input_model=DerivedGeometryStateInput,
            output_model=DerivedGeometryStateMeasurements,
            source_file="geometry.py",
            applicability=("finite_declared_geometry_or_state_trace",),
            invariants=(
                "affine_arithmetic_is_exact_and_rational",
                "only_observed_state_transitions_and_returns_are_reported",
                "no_cyclicity_or_geometric_law_is_inferred",
                "input_cardinality_is_hard_bounded",
            ),
            falsifiers=("invalid_or_unbounded_measurement_request",),
        ),
        _entry(
            capability_id="capability.viewport_support_terrain_measurement",
            implementation=measure_viewport_support_terrain,
            input_model=ViewportSupportTerrainInput,
            output_model=ViewportSupportTerrainMeasurements,
            source_file="routing.py",
            applicability=("finite_observed_viewport_or_configuration_trace",),
            invariants=(
                "configuration_edges_are_observed_not_invented",
                "terrain_and_support_deltas_are_exact_set_differences",
                "world_and_viewport_meanings_remain_unselected",
                "no_destination_or_route_is_selected",
                "input_cardinality_is_hard_bounded",
            ),
            falsifiers=("unknown_configuration_ref", "unbounded_measurement_request"),
        ),
        _entry(
            capability_id="capability.interaction_topology_measurement",
            implementation=measure_interaction_topology,
            input_model=InteractionTopologyInput,
            output_model=InteractionTopologyMeasurements,
            source_file="relations.py",
            applicability=("finite_observed_topology_or_landing_projection",),
            invariants=(
                "only_observed_graph_edges_are_compared",
                "connected_components_are_exact",
                "landing_fit_is_exact_set_containment_and_non_overlap",
                "no_interaction_role_or_topology_operator_is_assigned",
                "no_landing_candidate_is_selected",
                "input_cardinality_is_hard_bounded",
            ),
            falsifiers=("unknown_configuration_ref", "unbounded_measurement_request"),
        ),
        _entry(
            capability_id="capability.elimination_viability_measurement",
            implementation=measure_elimination_viability,
            input_model=EliminationViabilityInput,
            output_model=EliminationViabilityMeasurements,
            source_file="elimination.py",
            applicability=("finite_observed_state_graph_and_declared_terminal_set",),
            invariants=(
                "only_observed_transitions_are_traversed",
                "terminal_states_are_supplied_by_declared_meaning",
                "reverse_reachability_is_memoized_and_bounded",
                "dead_end_is_exact_only_after_complete_search",
                "no_operator_or_terminal_cardinality_is_inferred",
            ),
            falsifiers=("unknown_state_ref", "search_bound_reached"),
        ),
        _entry(
            capability_id="capability.relational_elimination_graph_measurement",
            implementation=measure_relational_elimination_graph,
            input_model=RelationalEliminationGraphInput,
            output_model=RelationalEliminationGraphMeasurements,
            source_file="elimination.py",
            applicability=("src_bound_atomic_operator_and_terminal_criterion",),
            invariants=("quantum_is_measured_not_assumed", "terminal_cardinality_is_supplied_by_src",
                "one_shared_graph_enumerator", "no_action_or_role_is_selected", "graph_and_terminal_set_are_bounded"),
            falsifiers=("unknown_cell_or_exceeded_state_bound",),
        ),
        _entry(
            capability_id="capability.temporal_concurrency_measurement",
            implementation=measure_temporal_concurrency,
            input_model=TemporalConcurrencyInput,
            output_model=TemporalConcurrencyMeasurements,
            source_file="tracking.py",
            applicability=("finite_ordered_multi_entity_samples",),
            invariants=(
                "frame_order_is_observed",
                "state_returns_and_instance_duplications_are_descriptive",
                "simultaneous_change_requires_shared_observed_intervals",
                "record_rewind_replay_and_phase_labels_are_not_inferred",
                "input_cardinality_is_hard_bounded",
            ),
            falsifiers=("duplicate_entity_frame_pair", "unbounded_timeline_request"),
        ),
        _entry(
            capability_id="capability.temporal_replay_sequence_measurement",
            implementation=measure_temporal_replay_sequence,
            input_model=TemporalReplaySequenceInput,
            output_model=TemporalReplaySequenceMeasurements,
            source_file="tracking.py",
            applicability=("two_source_declared_ordered_transition_sequences",),
            invariants=(
                "sequence_ownership_is_declared_outside_python",
                "comparison_is_exact_and_order_preserving",
                "endpoint_persistence_requires_post_exhaustion_observations",
                "record_rewind_replay_meaning_is_not_inferred",
                "input_cardinality_is_hard_bounded",
            ),
            falsifiers=("sequence_bound_exceeded", "replay_state_interval_mismatch"),
        ),
        _entry(
            capability_id="capability.temporal_joint_route_measurement",
            implementation=measure_temporal_joint_routes,
            input_model=TemporalJointRouteInput,
            output_model=TemporalJointRouteMeasurements,
            source_file="temporal_routes.py",
            applicability=("source_supplied_joint_routes_and_discrete_temporal_models",),
            invariants=(
                "no_model_route_role_or_action_is_selected",
                "all_supplied_models_are_preserved",
                "at_most_three_route_witnesses_and_1536_expansions",
                "blocked_actions_obey_the_supplied_clock_law",
                "unknown_endpoint_behavior_remains_unknown",
            ),
            falsifiers=("unknown_graph_cell", "ambiguous_graph_edge", "expansion_bound_exceeded"),
        ),
        _entry(
            capability_id="capability.interaction_probe_enumeration",
            implementation=enumerate_interaction_probes,
            input_model=InteractionProbeInput,
            output_model=InteractionProbeAgenda,
            source_file="interaction.py",
            applicability=(
                "finite_frame",
                "explicit_available_action_vocabulary",
            ),
            invariants=(
                "enumeration_does_not_choose_or_establish_a_role",
                "candidate_facts_are_neutral_and_unordered",
                "non_selected_candidates_remain_available",
            ),
            falsifiers=("no_available_action", "all_candidates_excluded"),
        ),
        _entry(
            capability_id="capability.local_constraint_composition_analysis",
            implementation=analyze_local_constraint_compositions,
            input_model=LocalConstraintCompositionInput,
            output_model=LocalConstraintCompositionAnalysis,
            source_file="multiscale.py",
            applicability=("rectangular_lattice_with_embedded_local_constraints",),
            invariants=(
                "at_most_three_global_composition_witnesses_survive",
                "projected_overlap_is_compared_by_exact_slot_identity",
                "shared_slots_are_deduplicated",
                "conflicts_remain_explicit_and_no_meaning_is_selected",
            ),
            falsifiers=("no_exact_embedded_constraint_lattice",),
        ),
        _entry(
            capability_id="capability.local_constraint_repair_action_enumeration",
            implementation=enumerate_local_constraint_repair_actions,
            input_model=LocalConstraintRepairAgendaInput,
            output_model=InteractionProbeAgenda,
            source_file="multiscale.py",
            applicability=("one_compatible_composition_selected_by_declared_policy",),
            invariants=(
                "conflicting_compositions_produce_no_action",
                "anchors_and_matching_slots_are_excluded",
                "only_deduplicated_mismatches_are_enumerated",
            ),
            falsifiers=("selected_composition_is_conflicting",),
        ),
        _entry(
            capability_id="capability.local_constraint_repair_reconciliation",
            implementation=reconcile_homologous_repair,
            input_model=HomologousRepairReconciliationInput,
            output_model=HomologousRepairReconciliationAnalysis,
            source_file="multiscale.py",
            applicability=("one_composed_mismatch_action_has_world_feedback",),
            invariants=(
                "target_change_and_boundary_display_change_are_distinct",
                "one_repair_must_reduce_the_deduplicated_mismatch_count_by_one",
                "terminal_completion_requires_score_feedback",
            ),
            falsifiers=("local_change_does_not_match_composed_prediction",),
        ),
        _entry(
            capability_id="capability.homologous_repair_analysis",
            implementation=analyze_homologous_repairs,
            input_model=HomologousRepairInput,
            output_model=HomologousRepairAnalysis,
            source_file="multiscale.py",
            applicability=("repeated_nested_grid_topology",),
            invariants=(
                "all_palette_bijections_are_enumerated",
                "normalized_positions_are_compared_across_scales",
                "center_anchor_is_excluded",
                "no_mapping_or_repair_is_selected",
            ),
            falsifiers=("no_repeated_nested_grid_cohort",),
        ),
        _entry(
            capability_id="capability.homologous_repair_action_enumeration",
            implementation=enumerate_homologous_repair_actions,
            input_model=HomologousRepairAgendaInput,
            output_model=InteractionProbeAgenda,
            source_file="multiscale.py",
            applicability=("one_mapping_selected_by_declared_policy",),
            invariants=(
                "only_predicted_mismatch_components_are_enumerated",
                "actions_are_component_grounded",
                "equivalent_mismatches_remain_preserved",
            ),
            falsifiers=("selected_mapping_has_no_component_grounded_mismatch",),
        ),
        _entry(
            capability_id="capability.homologous_repair_reconciliation",
            implementation=reconcile_homologous_repair,
            input_model=HomologousRepairReconciliationInput,
            output_model=HomologousRepairReconciliationAnalysis,
            source_file="multiscale.py",
            applicability=("one_predicted_mismatch_action_has_world_feedback",),
            invariants=(
                "target_change_and_boundary_display_change_are_distinct",
                "one_repair_must_reduce_mismatch_count_by_one",
                "terminal_completion_requires_score_feedback",
            ),
            falsifiers=("local_change_does_not_match_prediction",),
        ),
        _entry(
            capability_id="capability.committed_verified_alignment_cursor_advance",
            implementation=advance_committed_verified_alignment_cursor,
            input_model=CommittedVerifiedAlignmentCursorInput,
            output_model=InteractionProbeAgenda,
            source_file="interaction.py",
            applicability=("declared_verified_alignment_cursor_present",),
            invariants=(
                "one_predeclared_primitive_is_advanced",
                "no_mechanism_resimulation",
                "no_weighted_score",
            ),
            falsifiers=("committed_cursor_has_no_available_next_action",),
        ),
        _entry(
            capability_id="capability.verified_alignment_action_enumeration",
            implementation=enumerate_verified_alignment_actions,
            input_model=VerifiedAlignmentActionInput,
            output_model=InteractionProbeAgenda,
            source_file="interaction.py",
            applicability=(
                "observed_actuator_transition_available",
                "alignment_goal_available",
            ),
            invariants=(
                "only_observed_toward_goal_transitions_are_enumerated",
                "disabled_actuators_are_declared_ineligible",
                "remaining_steps_derive_from_exact_quantum",
                "no_weighted_score",
            ),
            falsifiers=(
                "goal_distance_not_quantized",
                "actuator_not_currently_tracked",
            ),
        ),
        _entry(
            capability_id="capability.route_compatible_order_analysis",
            implementation=analyze_route_compatible_orders,
            input_model=VerifiedAlignmentActionInput,
            output_model=InteractionProbeAgenda,
            source_file="route_compatible_graphs.py",
            applicability=(
                "typed_material_and_identity_graphs_share_persistent_nodes",
                "bounded_identity_routes_have_complete_target_correspondence",
            ),
            invariants=(
                "material_and_identity_edges_keep_distinct_effect_signatures",
                "only_exact_reverse_route_groups_are_measured",
                "no_execution_order_is_selected_by_python",
                "at_most_two_complete_route_orders_are_materialized",
            ),
            falsifiers=(
                "typed_graph_measurement_is_ambiguous",
                "target_correspondence_is_not_a_unique_exact_disambiguation",
            ),
        ),
        _entry(
            capability_id="capability.route_compatible_graph_plan_enumeration",
            implementation=enumerate_route_compatible_graph_plans,
            input_model=RouteCompatibleGraphPlanInput,
            output_model=InteractionProbeAgenda,
            source_file="route_compatible_graphs.py",
            applicability=(
                "one_route_order_was_declared_by_drm_and_src",
                "typed_graph_measurement_is_exact",
            ),
            invariants=(
                "only_the_declared_route_order_is_simulated",
                "each_stage_keeps_one_first_shortest_material_witness",
                "every_interface_uses_two_side_local_coordinates",
                "at_most_three_global_route_witnesses_are_materialized",
                "terminal_alignment_follows_every_identity_transport",
            ),
            falsifiers=(
                "declared_route_order_is_not_in_the_measured_alternatives",
                "a_state_action_or_route_bound_is_reached",
            ),
        ),
        _entry(
            capability_id="capability.transferred_quantized_plan_enumeration",
            implementation=enumerate_transferred_quantized_plans,
            input_model=VerifiedAlignmentActionInput,
            output_model=InteractionProbeAgenda,
            source_file="planning.py",
            applicability=(
                "prior_quantized_transfer_evidence_available",
                "current_scene_has_complete_structural_correspondence",
            ),
            invariants=(
                "every_enumerated_route_is_shortest_non_negative_and_conservative",
                "palette_and_absolute_screen_direction_do_not_ground_roles",
                "route_selection_remains_declarative",
                "every_planned_action_requires_observation_before_continuation",
            ),
            falsifiers=(
                "quantum_is_not_unique",
                "complete_chain_geometry_is_missing",
                "alignment_target_is_not_quantized",
            ),
        ),
        _entry(
            capability_id="capability.committed_quantized_plan_advance",
            implementation=advance_committed_quantized_plan,
            input_model=CommittedQuantizedPlanAdvanceInput,
            output_model=InteractionProbeAgenda,
            source_file="planning.py",
            applicability=("one_exact_route_is_already_committed",),
            invariants=(
                "no_route_search_or_simulation_is_repeated",
                "only_the_next_committed_cursor_delta_is_emitted",
                "action_coordinates_are_reused_from_the_committed_route",
            ),
            falsifiers=("committed_route_has_no_next_step",),
        ),
        _entry(
            capability_id="capability.quantized_plan_reconciliation",
            implementation=analyze_quantized_plan_reconciliation,
            input_model=QuantizedPlanReconciliationInput,
            output_model=QuantizedPlanReconciliationAnalysis,
            source_file="planning.py",
            applicability=("one_committed_predicted_plan_step_has_world_feedback",),
            invariants=(
                "expected_and_observed_quantities_are_compared_exactly",
                "terminal_score_feedback_is_distinct_from_same_level_frame_feedback",
                "reconciliation_does_not_select_the_next_action",
            ),
            falsifiers=("committed_plan_facts_are_missing",),
        ),
        _entry(
            capability_id="capability.multi_identity_opposed_routes",
            implementation=simulate_multi_identity_opposed_routes,
            input_model=MultiIdentityOpposedRoutesInput,
            output_model=MultiIdentityOpposedRoutesAnalysis,
            source_file="opposed_routes.py",
            applicability=(
                "one_measured_quantized_chain",
                "two_measured_opposed_identity_routes",
                "measured_interface_thresholds",
            ),
            invariants=(
                "no_more_than_three_crossing_witnesses",
                "one_first_shortest_material_preparation_per_stage",
                "crossing_transition_updates_both_identities_atomically",
                "identity_transports_precede_terminal_alignment",
                "route_choice_remains_declarative",
            ),
            falsifiers=(
                "opposed_route_measurements_are_incomplete",
                "a_capacity_or_action_bound_is_reached",
            ),
        ),
        _entry(
            capability_id="capability.finite_grid_reachability",
            implementation=finite_grid_reachability,
            input_model=ReachabilityInput,
            output_model=ReachabilityResult,
            source_file="routing.py",
            applicability=(
                "finite_grid",
                "explicit_traversable_set",
                "explicit_step_vectors",
            ),
            invariants=(
                "shortest_distances_exact",
                "equal_first_steps_preserved",
                "no_goal_or_route_selected",
            ),
            falsifiers=("start_not_traversable", "position_outside_grid"),
        ),
        _entry(
            capability_id="capability.relation_graph_analysis",
            implementation=analyze_relation_graph,
            input_model=RelationGraphInput,
            output_model=RelationGraphResult,
            source_file="relations.py",
            applicability=("finite_explicit_relation_graph",),
            invariants=(
                "connectivity_and_reachability_are_exact",
                "relation_identity_is_preserved",
                "no_semantic_role_assigned",
            ),
            falsifiers=("unknown_node_reference",),
        ),
        _entry(
            capability_id="capability.quantity_simulation",
            implementation=simulate_quantities,
            input_model=QuantitySimulationInput,
            output_model=QuantitySimulationResult,
            source_file="constraints.py",
            applicability=("finite_integer_quantity_transitions",),
            invariants=(
                "every_intermediate_state_is_returned",
                "non_negativity_is_exact",
                "capacity_is_exact",
                "declared_conservation_groups_are_exact",
            ),
            falsifiers=("unknown_quantity_entity",),
        ),
        _entry(
            capability_id="capability.declared_goal_gap_measurement",
            implementation=measure_declared_goal_gaps,
            input_model=DeclaredGoalGapInput,
            output_model=DeclaredGoalGapResult,
            source_file="goals.py",
            applicability=("drm_declared_open_goal_dimensions",),
            invariants=(
                "only_declared_dimensions_are_measured",
                "integer_deltas_and_equality_are_exact",
                "no_goal_is_opened_ranked_or_selected",
                "request_count_is_bounded",
            ),
            falsifiers=("goal_dimension_request_is_invalid_or_duplicated",),
        ),
        _entry(
            capability_id="capability.declared_goal_plan_enumeration",
            implementation=project_declared_goal_plan_alternatives,
            input_model=DeclaredGoalPlanInput,
            output_model=DeclaredGoalPlanAlternatives,
            source_file="goal_graph.py",
            applicability=(
                "drm_declared_open_goal_gaps_and_bounded_subgoal_dependencies",
            ),
            invariants=(
                "at_most_three_topological_witnesses_are_exposed",
                "every_witness_covers_only_declared_gap_dimensions",
                "goal_scope_and_open_gap_coverage_are_exact",
                "no_plan_is_ranked_selected_or_released",
            ),
            falsifiers=(
                "goal_scope_mismatch_or_open_gap_not_covered",
                "subgoal_dependency_cycle_has_no_bounded_witness",
            ),
        ),
        _entry(
            capability_id="capability.dimension_transition",
            implementation=compare_state_dimensions,
            input_model=DimensionTransitionInput,
            output_model=DimensionTransitionResult,
            source_file="constraints.py",
            applicability=("same_entity_dimension_pairs_before_and_after",),
            invariants=(
                "changed_and_preserved_dimensions_are_explicit",
                "entity_identity_is_not_recreated",
            ),
            falsifiers=("dimension_identity_mismatch",),
        ),
    )


def _entry(
    *,
    capability_id: str,
    implementation: CapabilityImplementation,
    input_model: type[FrozenModel],
    output_model: type[FrozenModel],
    source_file: str,
    applicability: tuple[str, ...],
    invariants: tuple[str, ...],
    falsifiers: tuple[str, ...],
) -> RegisteredCapability:
    source_path = Path(__file__).with_name(source_file)
    definition = CapabilityDefinition(
        id=capability_id,
        input_schema=str(input_model.model_fields["schema_version"].default),
        output_schema=str(output_model.model_fields["schema_version"].default),
        effect_class=EffectClass.PURE,
        applicability=applicability,
        invariants=invariants,
        falsifiers=falsifiers,
        source_unit=f"agents.yf_arc3_v5.capabilities.{source_path.stem}",
        source_hash=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        generalization_layer="generic_symbolic_computation",
    )
    return RegisteredCapability(
        definition=definition,
        input_model=input_model,
        output_model=output_model,
        implementation=implementation,
    )
