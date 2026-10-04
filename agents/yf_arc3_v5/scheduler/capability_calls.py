"""Mechanical SRC call adapters for the typed M5 capability registry."""

from __future__ import annotations

from agents.yf_arc3_v5.capabilities.display_correspondence import DisplayCorrespondenceInput
from agents.yf_arc3_v5.capabilities.contracts import (
    TypedAssignmentDomainsInput,
    TypedEditSuccessorInput,
    TypedInterpreterInput,
    RecipePatternDifferenceInput,
    RecipeSuccessorInput,
    RecipeResizeSuccessorInput,
    RecipeResourceOrderInput,
    ResourceGaugeQuantityAnalysisInput,
    PiercingIntersectionInput,
    PiercingPushSuccessorInput,
    PiercingPierceSuccessorInput,
    PiercingWithdrawSuccessorInput,
    PiercingClearanceInput,
)

from agents.yf_arc3_v5.capabilities.announced_motion import AnnouncedStepInput
from agents.yf_arc3_v5.capabilities.cursor_orientation import AttachedRotationAnalysisInput

from collections.abc import Callable

from agents.yf_arc3_v5.logos.types import FrozenMap
from agents.yf_arc3_v5.capabilities import (
    AbductiveVersionSpaceInput,
    CommittedVerifiedAlignmentCursorInput,
    CommittedQuantizedPlanAdvanceInput,
    ComponentExtractionInput,
    ComponentMatchInput,
    ControlledTransitionAnalysisInput,
    DeclaredGoalGapInput,
    DeclaredGoalPlanInput,
    DimensionTransitionInput,
    DerivedGeometryStateInput,
    EliminationViabilityInput,
    RelationalEliminationGraphInput,
    FrameDifferenceInput,
    FrameGrid,
    FrameNormalizationInput,
    GeometryInput,
    GoalCompletionAnalysisInput,
    GridPartitionInput,
    KnownGridCellRecountInput,
    HomologousRepairAgendaInput,
    HomologousRepairInput,
    HomologousRepairReconciliationInput,
    InteractionProbeInput,
    InteractionTopologyInput,
    ArticulatedKinematicsInput,
    ArticulatedSharedSnapshotInput,
    ArticulatedSuccessorInput,
    InverseMeanAnchorDomainInput,
    InverseAnchorSuccessorInput,
    InverseAnchorTypedClearanceInput,
    InverseAnchorWaypointCandidatesInput,
    OrderedReliefPoseInput,
    OrderedReliefTransitionInput,
    OrderedReliefWriteInput,
    ReconfigurableConnectionDomainInput,
    DirectedFlowColliderUnionInput,
    DirectedFlowJointTickInput,
    DirectedFlowRecoveryInput,
    LocalFieldAcceptanceWindowInput,
    LocalFieldAutonomousSuccessorInput,
    LocalFieldSelectiveDomainInput,
    LocalFieldTransactionInput,
    ReconfigurableRigidGeometryInput,
    ReconfigurableSupportPacketInput,
    IntermediateSceneChangeInput,
    LocalConstraintCompositionInput,
    LocalConstraintRepairAgendaInput,
    MultiResolutionViewInput,
    MultiIdentityOpposedRoutesInput,
    OrthogonalPerimeterAssemblyInput,
    PaletteCanonicalizationInput,
    QuantitySimulationInput,
    RationalAnchorDependenceInput,
    QuantizedPlanReconciliationInput,
    ReachabilityInput,
    RelationGraphInput,
    RepeatedBoundaryDecreaseAnalysisInput,
    ExternalTransportObservationInput,
    RouteCompatibleGraphPlanInput,
    RevisionedCapabilityRequest,
    SceneRebindingMeasurementInput,
    SceneTransitionMeasurementInput,
    ExactMulticellPeerRelationInput,
    TemporalTrackingInput,
    TemporalConcurrencyInput,
    TemporalReplaySequenceInput,
    TransitionEvidenceLedgerInput,
    TransitionRetrodictionInput,
    TransitionPhenomenonInventoryInput,
    UniformBlockReductionInput,
    VerifiedAlignmentActionInput,
    ViewportSupportTerrainInput,
    VisualSceneDescription,
    VisualSceneInput,
)
from agents.yf_arc3_v5.capabilities.registry import CapabilityRegistry
from agents.yf_arc3_v5.capabilities.temporal_routes import TemporalJointRouteInput
from agents.yf_arc3_v5.logos.types import FrozenModel
from agents.yf_arc3_v5.scheduler.contracts import FunctionCallContext
from agents.yf_arc3_v5.scheduler.evaluator import restore_runtime_model_refs
from agents.yf_arc3_v5.scheduler.registry import SchedulerRuntimeRegistry
from agents.yf_arc3_v5.src.production import capability_symbol_definitions


def register_capability_calls(
    runtime: SchedulerRuntimeRegistry,
    capabilities: CapabilityRegistry,
) -> None:
    symbols = {
        definition.id: definition
        for definition in capability_symbol_definitions(capabilities)
    }
    for entry in capabilities.entries:
        runtime.register_call(
            symbols[entry.definition.id],
            _call_adapter(capabilities, entry.definition.id),
        )


def _call_adapter(
    capabilities: CapabilityRegistry,
    capability_id: str,
) -> Callable[[FunctionCallContext], object]:
    def invoke(context: FunctionCallContext) -> object:
        payload = _payload(capability_id, context)
        execution = capabilities.invoke(
            RevisionedCapabilityRequest(
                capability_id=capability_id,
                input_revision=context.state_revision,
                payload=payload,
            )
        )
        return execution.output

    return invoke


def _payload(capability_id: str, context: FunctionCallContext) -> FrozenModel:
    arguments = restore_runtime_model_refs(context.arguments)
    if capability_id == "capability.abductive_version_space_measurement":
        return AbductiveVersionSpaceInput.model_validate(arguments["request"])
    if capability_id == "capability.frame_normalization":
        return FrameNormalizationInput.from_payload(arguments["payload"])
    if capability_id == "capability.connected_components":
        return ComponentExtractionInput(
            frame=FrameGrid.model_validate(arguments["frame"]),
            connectivity=int(arguments["connectivity"]),
        )
    if capability_id in {
        "capability.visual_scene_decomposition",
        "capability.block_grid_scene_decomposition",
    }:
        return VisualSceneInput(
            frame=FrameGrid.model_validate(arguments["frame"]),
            action_conditioned_grid_evidence=bool(
                arguments.get("action_conditioned_grid_evidence", True)
            ),
            retained_grid_geometry=arguments.get("retained_grid_geometry") or None,
            retained_grid_separator_consistency_ppm=int(
                arguments.get("retained_grid_separator_consistency_ppm") or 0
            ),
        )
    if capability_id == "capability.known_grid_cell_recount":
        return KnownGridCellRecountInput.model_validate(arguments["request"])
    if capability_id == "capability.multi_resolution_view_measurement":
        return MultiResolutionViewInput(
            scene=VisualSceneDescription.model_validate(arguments["scene"]),
            observation_ref=arguments["observation_ref"],
            frame_ref=context.frame_id,
            available_action_refs=tuple(arguments["available_action_refs"]),
            is_initial_observation=bool(arguments["is_initial_observation"]),
        )
    if capability_id == "capability.frame_difference":
        return FrameDifferenceInput(
            before=FrameGrid.model_validate(arguments["before"]),
            after=FrameGrid.model_validate(arguments["after"]),
            action_ref=arguments["action_ref"],
        )
    if capability_id == "capability.scene_transition_measurement":
        return SceneTransitionMeasurementInput(
            before=FrameGrid.model_validate(arguments["before"]),
            intermediate_frames=tuple(
                FrameGrid.model_validate(frame)
                for frame in arguments.get("intermediate_frames", ())
            ),
            after=FrameGrid.model_validate(arguments["after"]),
            transition_ref=str(arguments["transition_ref"]),
            action_ref=arguments["action_ref"],
            action_data=FrozenMap(arguments.get("action_data") or {}),
            candidate_ref=arguments.get("candidate_ref"),
            before_configuration_digest=arguments.get(
                "before_configuration_digest"
            ),
            after_configuration_digest=arguments.get("after_configuration_digest"),
            before_available_action_set_digest=arguments.get(
                "before_available_action_set_digest"
            ),
            boundary_indicator_quantum_delta_measurement=int(
                arguments.get("boundary_indicator_quantum_delta_measurement", 0)
            ),
            established_boundary_indicator_decrement_changed_pixel_count_measurement=int(
                arguments.get(
                    "established_boundary_indicator_decrement_changed_pixel_count_measurement",
                    0,
                )
            ),
            declared_boundary_indicator_decrease_changed_pixel_count_measurement=int(
                arguments.get(
                    "declared_boundary_indicator_decrease_changed_pixel_count_measurement",
                    0,
                )
            ),
            current_effect_signature_known=bool(
                arguments.get("current_effect_signature_known", False)
            ),
            current_effect_signature_known_cycle=bool(
                arguments.get("current_effect_signature_known_cycle", False)
            ),
            known_effect_signature_count=int(
                arguments.get("known_effect_signature_count", 0)
            ),
            known_cycle_transition_signature_count=int(
                arguments.get("known_cycle_transition_signature_count", 0)
            ),
            before_score=int(arguments.get("before_score", 0)),
            after_score=(
                int(arguments["after_score"])
                if arguments.get("after_score") is not None
                else None
            ),
        )
    if capability_id == "capability.intermediate_scene_change_measurement":
        return IntermediateSceneChangeInput.model_validate(arguments)
    if capability_id == "capability.temporal_component_tracking":
        return TemporalTrackingInput.model_validate(arguments)
    if capability_id == "capability.exact_multicell_peer_relation_measurement":
        return ExactMulticellPeerRelationInput.model_validate(arguments)
    if capability_id == "capability.temporal_concurrency_measurement":
        return TemporalConcurrencyInput.model_validate(arguments["request"])
    if capability_id == "capability.temporal_replay_sequence_measurement":
        return TemporalReplaySequenceInput.model_validate(arguments["request"])
    if capability_id == "capability.temporal_joint_route_measurement":
        return TemporalJointRouteInput.model_validate(arguments["request"])
    if capability_id == "capability.controlled_transition_analysis":
        return ControlledTransitionAnalysisInput.model_validate(arguments)
    if capability_id == "capability.attached_rotation_analysis":
        return AttachedRotationAnalysisInput.model_validate(arguments)
    if capability_id == "capability.display_correspondence_measurements":
        return DisplayCorrespondenceInput.model_validate(arguments)
    if capability_id == "capability.goal_completion_analysis":
        return GoalCompletionAnalysisInput.model_validate(arguments)
    if capability_id == "capability.declared_goal_gap_measurement":
        return DeclaredGoalGapInput.model_validate(arguments["request"])
    if capability_id == "capability.declared_goal_plan_enumeration":
        return DeclaredGoalPlanInput.model_validate(arguments)
    if capability_id == "capability.repeated_boundary_decrease_analysis":
        return RepeatedBoundaryDecreaseAnalysisInput.model_validate(arguments)
    if capability_id == "capability.resource_gauge_quantity_analysis":
        return ResourceGaugeQuantityAnalysisInput.model_validate(arguments)
    if capability_id == "capability.external_transport_observation":
        return ExternalTransportObservationInput.model_validate(arguments)
    if capability_id == "capability.announced_step_measurements":
        return AnnouncedStepInput.model_validate(arguments)
    if capability_id == "capability.component_matching":
        return ComponentMatchInput.model_validate(arguments)
    if capability_id == "capability.palette_canonicalization":
        return PaletteCanonicalizationInput.model_validate(arguments)
    if capability_id == "capability.scene_rebinding_measurement":
        return SceneRebindingMeasurementInput.model_validate(arguments)
    if capability_id == "capability.uniform_block_reduction":
        return UniformBlockReductionInput.model_validate(arguments)
    if capability_id == "capability.grid_partition_enumeration":
        return GridPartitionInput.model_validate(arguments)
    if capability_id == "capability.component_geometry":
        return GeometryInput.model_validate(arguments)
    if capability_id == "capability.derived_geometry_state_measurement":
        return DerivedGeometryStateInput.model_validate(arguments["request"])
    if capability_id == "capability.elimination_viability_measurement":
        return EliminationViabilityInput.model_validate(arguments["request"])
    if capability_id == "capability.relational_elimination_graph_measurement":
        return RelationalEliminationGraphInput.model_validate(arguments)
    if capability_id == "capability.viewport_support_terrain_measurement":
        return ViewportSupportTerrainInput.model_validate(arguments["request"])
    if capability_id == "capability.interaction_probe_enumeration":
        return InteractionProbeInput.model_validate(arguments)
    if capability_id == "capability.interaction_topology_measurement":
        return InteractionTopologyInput.model_validate(arguments["request"])
    if capability_id == "capability.orthogonal_perimeter_assembly_measurement":
        return OrthogonalPerimeterAssemblyInput.model_validate(arguments)
    if capability_id == "capability.local_constraint_composition_analysis":
        return LocalConstraintCompositionInput.model_validate(arguments)
    if capability_id == "capability.local_constraint_repair_action_enumeration":
        return LocalConstraintRepairAgendaInput.model_validate(arguments)
    if capability_id == "capability.local_constraint_repair_reconciliation":
        return HomologousRepairReconciliationInput.model_validate(arguments)
    if capability_id == "capability.homologous_repair_analysis":
        return HomologousRepairInput.model_validate(arguments)
    if capability_id == "capability.homologous_repair_action_enumeration":
        return HomologousRepairAgendaInput.model_validate(arguments)
    if capability_id == "capability.homologous_repair_reconciliation":
        return HomologousRepairReconciliationInput.model_validate(arguments)
    if capability_id == "capability.verified_alignment_action_enumeration":
        return VerifiedAlignmentActionInput.model_validate(arguments)
    if capability_id == "capability.committed_verified_alignment_cursor_advance":
        return CommittedVerifiedAlignmentCursorInput.model_validate(arguments)
    if capability_id == "capability.transferred_quantized_plan_enumeration":
        return VerifiedAlignmentActionInput.model_validate(arguments)
    if capability_id == "capability.route_compatible_order_analysis":
        return VerifiedAlignmentActionInput.model_validate(arguments)
    if capability_id == "capability.route_compatible_graph_plan_enumeration":
        return RouteCompatibleGraphPlanInput.model_validate(arguments)
    if capability_id == "capability.committed_quantized_plan_advance":
        return CommittedQuantizedPlanAdvanceInput.model_validate(arguments)
    if capability_id == "capability.quantized_plan_reconciliation":
        return QuantizedPlanReconciliationInput.model_validate(arguments)
    if capability_id == "capability.multi_identity_opposed_routes":
        return MultiIdentityOpposedRoutesInput.model_validate(arguments["request"])
    if capability_id == "capability.finite_grid_reachability":
        return ReachabilityInput.model_validate(arguments["request"])
    if capability_id == "capability.relation_graph_analysis":
        return RelationGraphInput.model_validate(arguments["request"])
    if capability_id == "capability.quantity_simulation":
        return QuantitySimulationInput.model_validate(arguments["request"])
    if capability_id == "capability.dimension_transition":
        return DimensionTransitionInput.model_validate(arguments["request"])
    if capability_id == "capability.rational_anchor_dependence_measurement":
        return RationalAnchorDependenceInput.model_validate(arguments["request"])
    if capability_id == "capability.inverse_mean_anchor_domain_enumeration":
        return InverseMeanAnchorDomainInput.model_validate(arguments["request"])
    if capability_id == "capability.inverse_anchor_typed_clearance_measurement":
        return InverseAnchorTypedClearanceInput.model_validate(arguments["request"])
    if capability_id == "capability.inverse_anchor_successor_measurement":
        return InverseAnchorSuccessorInput.model_validate(arguments["request"])
    if capability_id == "capability.inverse_anchor_waypoint_candidate_measurement":
        return InverseAnchorWaypointCandidatesInput.model_validate(arguments["request"])
    if capability_id == "capability.articulated_kinematics_measurement":
        return ArticulatedKinematicsInput.model_validate(arguments["request"])
    if capability_id == "capability.articulated_shared_snapshot_measurement":
        return ArticulatedSharedSnapshotInput.model_validate(arguments["request"])
    if capability_id == "capability.articulated_successor_measurement":
        return ArticulatedSuccessorInput.model_validate(arguments["request"])
    if capability_id == "capability.typed_assignment_domains_measurement":
        return TypedAssignmentDomainsInput.model_validate(arguments["request"])
    if capability_id == "capability.typed_edit_successor_measurement":
        return TypedEditSuccessorInput.model_validate(arguments["request"])
    if capability_id == "capability.typed_interpreter_measurement":
        return TypedInterpreterInput.model_validate(arguments["request"])
    if capability_id == "capability.recipe_pattern_difference_measurement":
        return RecipePatternDifferenceInput.model_validate(arguments["request"])
    if capability_id == "capability.recipe_successor_measurement":
        return RecipeSuccessorInput.model_validate(arguments["request"])
    if capability_id == "capability.recipe_resize_successor_measurement":
        return RecipeResizeSuccessorInput.model_validate(arguments["request"])
    if capability_id == "capability.recipe_resource_order_measurement":
        return RecipeResourceOrderInput.model_validate(arguments["request"])
    if capability_id == "capability.piercing_intersection_measurement":
        return PiercingIntersectionInput.model_validate(arguments["request"])
    if capability_id == "capability.piercing_push_successor_measurement":
        return PiercingPushSuccessorInput.model_validate(arguments["request"])
    if capability_id == "capability.piercing_pierce_successor_measurement":
        return PiercingPierceSuccessorInput.model_validate(arguments["request"])
    if capability_id == "capability.piercing_withdraw_successor_measurement":
        return PiercingWithdrawSuccessorInput.model_validate(arguments["request"])
    if capability_id == "capability.piercing_clearance_measurement":
        return PiercingClearanceInput.model_validate(arguments["request"])
    if capability_id == "capability.ordered_relief_transition_measurement":
        return OrderedReliefTransitionInput.model_validate(arguments["request"])
    if capability_id == "capability.ordered_relief_pose_measurement":
        return OrderedReliefPoseInput.model_validate(arguments["request"])
    if capability_id == "capability.ordered_relief_write_successor":
        return OrderedReliefWriteInput.model_validate(arguments["request"])
    if capability_id == "capability.reconfigurable_support_packet_measurement":
        return ReconfigurableSupportPacketInput.model_validate(arguments["request"])
    if capability_id == "capability.reconfigurable_rigid_geometry_measurement":
        return ReconfigurableRigidGeometryInput.model_validate(arguments["request"])
    if capability_id == "capability.reconfigurable_connection_domain_enumeration":
        return ReconfigurableConnectionDomainInput.model_validate(arguments["request"])
    if capability_id == "capability.directed_flow_collider_union_measurement":
        return DirectedFlowColliderUnionInput.model_validate(arguments["request"])
    if capability_id == "capability.directed_flow_joint_tick_measurement":
        return DirectedFlowJointTickInput.model_validate(arguments["request"])
    if capability_id == "capability.directed_flow_recovery_measurement":
        return DirectedFlowRecoveryInput.model_validate(arguments["request"])
    if capability_id == "capability.local_field_selective_domain_measurement":
        return LocalFieldSelectiveDomainInput.model_validate(arguments["request"])
    if capability_id == "capability.local_field_autonomous_successor_measurement":
        return LocalFieldAutonomousSuccessorInput.model_validate(arguments["request"])
    if capability_id == "capability.local_field_acceptance_window_measurement":
        return LocalFieldAcceptanceWindowInput.model_validate(arguments["request"])
    if capability_id == "capability.local_field_transaction_measurement":
        return LocalFieldTransactionInput.model_validate(arguments["request"])
    if capability_id == "capability.transition_evidence_compaction":
        return TransitionEvidenceLedgerInput.model_validate(arguments["request"])
    if capability_id == "capability.transition_retrodiction_measurement":
        return TransitionRetrodictionInput.model_validate(arguments["request"])
    if capability_id == "capability.transition_phenomenon_inventory":
        return TransitionPhenomenonInventoryInput.model_validate(arguments["request"])
    raise ValueError(f"no SRC adapter for capability: {capability_id}")
