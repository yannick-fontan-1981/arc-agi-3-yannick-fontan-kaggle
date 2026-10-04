"""Production SRC environment and exact source compilation for M6 workflows.

The former broad V0.7 workflow TODOs are covered for the six active games by
the generic temporal, repair, transport, quantized-plan, assignment, symbolic
contact and coupled-member workflows already registered below.  The coverage
crosswalk and its per-level evidence live in
`docs/v5/generic_todo_coverage_six_games.md` and
`docs/v5/level_resolution/*_L1_reasoning_frontier.json`.  New families still
require their own generic contract, focused falsifier and frontier before
promotion.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from agents.yf_arc3_v5.capabilities import (
    CapabilityRegistry,
    build_default_capability_registry,
)
from agents.yf_arc3_v5.drm import (
    DrmRegistry,
    build_production_drm_registry,
    load_production_drm,
)
from agents.yf_arc3_v5.determinism_guard import (
    assert_v5_determinism,
    v5_source_tree_digest,
)
from agents.yf_arc3_v5.src.compiler import CompiledModule, compile_source
from agents.yf_arc3_v5.src.production_contracts import (
    load_production_workflow_contracts,
    validate_production_workflow_contracts,
)
from agents.yf_arc3_v5.src.symbols import (
    CompilerEnvironment,
    ParameterSpec,
    SymbolDefinition,
    SymbolKind,
    generic_fixture_environment,
    type_spec,
)

_CAPABILITY_TYPES = (
    "ArrayRegionMeasurementRequest",
    "ArrayRegionMeasurements",
    "PairPointGeometryRequest",
    "StockChangeRequest",
    "StockChangeMeasurements",
    "FilledContourRequest",
    "FilledContourMeasurements",
    "PointProgressRequest",
    "PointProgressMeasurements",
    "PairPointGeometryMeasurements",
    "ConsumptiveDependencyRequest",
    "ConsumptiveDependencyMeasurements",
    "BearerAppearanceMeasurement",
    "FrameGrid",
    "DisplayCorrespondenceMeasurements",
    "AnnouncedStepMeasurements",
    "FrameNormalizationResult",
    "ComponentExtractionResult",
    "BlockGridSceneDescription",
    "VisualSceneDescription",
    "MultiResolutionViewMeasurements",
    "FrameDifferenceResult",
    "SceneTransitionMeasurements",
    "IntermediateSceneChangeMeasurements",
    "TemporalTrackingResult",
    "ExactMulticellPeerRelationMeasurements",
    "ControlledTransitionAnalysis",
    "AttachedRotationAnalysis",
    "GoalCompletionAnalysis",
    "DeclaredGoalGapResult",
    "DeclaredGoalGapInput",
    "DeclaredGoalPlanAlternatives",
    "GoalPlanRequest",
    "QuantizedPlanReconciliationAnalysis",
    "RepeatedBoundaryDecreaseAnalysis",
    "ResourceGaugeQuantityAnalysis",
    "ExternalTransportObservations",
    "DeclaredAlternativeFacts",
    "DeclaredMeaningInstantiation",
    "PreparedMeaningUpdates",
    "PreparedCommittedDerivation",
    "PreparedCommittedProposal",
    "ComponentMatchResult",
    "PaletteCanonicalizationResult",
    "SceneRebindingMeasurements",
    "UniformBlockReductionResult",
    "GridPartitionResult",
    "PeriodicCellGridResult",
    "KnownGridCellRecountInput",
    "KnownGridCellRecountResult",
    "HomologousRepairAnalysis",
    "HomologousRepairReconciliationAnalysis",
    "LocalConstraintCompositionAnalysis",
    "GeometryResult",
    "OrthogonalPerimeterAssemblyMeasurements",
    "DerivedGeometryStateMeasurements",
    "ViewportSupportTerrainMeasurements",
    "InteractionTopologyMeasurements",
    "EliminationViabilityMeasurements",
    "RelationalEliminationGraphMeasurements",
    "AbductiveVersionSpaceMeasurements",
    "TemporalConcurrencyMeasurements",
    "TemporalReplaySequenceMeasurements",
    "TemporalJointRouteInput",
    "TemporalJointRouteMeasurements",
    "TransitionEvidenceLedger",
    "TransitionRetrodictionMeasurements",
    "TransitionPhenomenonInventoryMeasurements",
    "InteractionProbeAgenda",
    "DynamicGoalFrontierActivation",
    "PreparedInteractionProbe",
    "ReachabilityResult",
    "RelationGraphResult",
    "QuantitySimulationInput",
    "QuantitySimulationResult",
    "DimensionTransitionResult",
    "RationalAnchorDependenceInput",
    "RationalAnchorDependenceMeasurements",
    "InverseMeanAnchorDomainInput",
    "InverseMeanAnchorDomainMeasurements",
    "InverseAnchorTypedClearanceInput",
    "InverseAnchorTypedClearanceMeasurements",
    "InverseAnchorSuccessorInput",
    "InverseAnchorSuccessorMeasurements",
    "InverseAnchorWaypointCandidatesInput",
    "InverseAnchorWaypointCandidatesMeasurements",
    "ArticulatedKinematicsInput",
    "ArticulatedKinematicsMeasurements",
    "ArticulatedSharedSnapshotInput",
    "ArticulatedSharedSnapshotMeasurements",
    "ArticulatedSuccessorInput",
    "ArticulatedSuccessorMeasurements",
    "TypedAssignmentDomainsInput",
    "TypedAssignmentDomainsMeasurements",
    "TypedEditSuccessorInput",
    "TypedEditSuccessorMeasurements",
    "TypedInterpreterInput",
    "TypedInterpreterMeasurements",
    "RecipePatternDifferenceInput",
    "RecipePatternDifferenceMeasurements",
    "RecipeSuccessorInput",
    "RecipeSuccessorMeasurements",
    "RecipeResizeSuccessorInput",
    "RecipeResizeSuccessorMeasurements",
    "RecipeResourceOrderInput",
    "RecipeResourceOrderMeasurements",
    "PiercingIntersectionInput",
    "PiercingIntersectionMeasurements",
    "PiercingPushSuccessorInput",
    "PiercingPushSuccessorMeasurements",
    "PiercingPierceSuccessorInput",
    "PiercingPierceSuccessorMeasurements",
    "PiercingWithdrawSuccessorInput",
    "PiercingWithdrawSuccessorMeasurements",
    "PiercingClearanceInput",
    "PiercingClearanceMeasurements",
    "OrderedReliefTransitionInput",
    "OrderedReliefTransitionMeasurements",
    "OrderedReliefWriteInput",
    "OrderedReliefWriteMeasurements",
    "OrderedReliefPoseInput",
    "OrderedReliefPoseMeasurements",
    "ReconfigurableSupportPacketInput",
    "ReconfigurableSupportPacketMeasurements",
    "ReconfigurableRigidGeometryInput",
    "ReconfigurableRigidGeometryMeasurements",
    "ReconfigurableConnectionDomainInput",
    "ReconfigurableConnectionDomainMeasurements",
    "DirectedFlowColliderUnionInput",
    "DirectedFlowColliderUnionMeasurements",
    "DirectedFlowJointTickInput",
    "DirectedFlowJointTickMeasurements",
    "DirectedFlowRecoveryInput",
    "DirectedFlowRecoveryMeasurements",
    "LocalFieldSelectiveDomainInput",
    "LocalFieldSelectiveDomainMeasurements",
    "LocalFieldAutonomousSuccessorInput",
    "LocalFieldAutonomousSuccessorMeasurements",
    "LocalFieldAcceptanceWindowInput",
    "LocalFieldAcceptanceWindowMeasurements",
    "LocalFieldTransactionInput",
    "LocalFieldTransactionMeasurements",
)


def production_control_symbol_definitions() -> tuple[SymbolDefinition, ...]:
    """Source-attested cognitive control calls used by action-bearing SRC."""

    root = Path(__file__).parent
    continue_path = root / "arc3" / "continue_declared_action.src"
    wisdom_path = root / "core" / "operational_wisdom.src"
    release_path = root / "core" / "release_action.src"
    construct_principle_path = root / "core" / "construct_applied_principle.src"
    instantiate_principle_path = root / "core" / "instantiate_general_principle.src"
    ontology_extension_path = root / "arc3" / "extend_insufficient_ontology.src"
    plan_goal_path = root / "arc3" / "plan_goal.src"
    evaluate_constrained_plan_path = root / "arc3" / "evaluate_constrained_plan.src"
    priority_frontier_path = root / "arc3" / "project_priority_obligation_frontier.src"
    probe_path = root / "arc3" / "choose_information_probe.src"
    meaning_path = root / "arc3" / "construct_transition_meaning.src"
    dynamic_synthesis_path = root.parent / "dynamic_workflow" / "synthesis.py"
    return (
        SymbolDefinition(
            id="workflow.operational_wisdom",
            kind=SymbolKind.WISDOM_WORKFLOW,
            parameters=(
                ParameterSpec(
                    name="viability",
                    type_spec=type_spec("ViabilityAssessment"),
                ),
            ),
            output_type=type_spec("CommitmentDecision"),
            source_unit="agents/yf_arc3_v5/src/core/operational_wisdom.src",
            source_hash=hashlib.sha256(wisdom_path.read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="workflow.release_action",
            kind=SymbolKind.ACTION_RELEASE_WORKFLOW,
            parameters=(
                ParameterSpec(name="intent", type_spec=type_spec("ActionIntent")),
                ParameterSpec(
                    name="viability",
                    type_spec=type_spec("ViabilityAssessment"),
                ),
                ParameterSpec(
                    name="wisdom",
                    type_spec=type_spec("CommitmentDecision"),
                ),
                ParameterSpec(
                    name="context",
                    type_spec=type_spec("WorldContext"),
                ),
                ParameterSpec(
                    name="declared_series",
                    type_spec=type_spec("Boolean"),
                    required=False,
                ),
                ParameterSpec(
                    name="full_route_authorized",
                    type_spec=type_spec("Boolean"),
                    required=False,
                ),
                ParameterSpec(
                    name="defer_concentration_acquisition",
                    type_spec=type_spec("Boolean"),
                    required=False,
                ),
            ),
            output_type=type_spec("ActionReleasePermit"),
            source_unit="agents/yf_arc3_v5/src/core/release_action.src",
            source_hash=hashlib.sha256(release_path.read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="workflow.construct_applied_principle",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(
                    name="premise_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(name="derivation_spec", type_spec=type_spec("Any")),
            ),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/core/construct_applied_principle.src",
            source_hash=hashlib.sha256(construct_principle_path.read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="workflow.instantiate_general_principle",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(
                    name="proposal_seeds",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="basis_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="proposal_scope",
                    type_spec=type_spec("Map", type_spec("Ref"), type_spec("Any")),
                ),
                ParameterSpec(
                    name="alternative_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="missing_evidence",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="falsifier_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
            ),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/core/instantiate_general_principle.src",
            source_hash=hashlib.sha256(instantiate_principle_path.read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="workflow.extend_insufficient_ontology",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="insufficiency_premises", type_spec=type_spec("List", type_spec("Ref"))),
                ParameterSpec(name="insufficiency_context", type_spec=type_spec("Map", type_spec("Ref"), type_spec("Any"))),
                ParameterSpec(name="extension_seeds", type_spec=type_spec("List", type_spec("Any"))),
                ParameterSpec(name="extension_basis", type_spec=type_spec("List", type_spec("Ref"))),
                ParameterSpec(name="extension_scope", type_spec=type_spec("Map", type_spec("Ref"), type_spec("Any"))),
                ParameterSpec(name="alternatives", type_spec=type_spec("List", type_spec("Ref"))),
                ParameterSpec(name="missing_evidence", type_spec=type_spec("List", type_spec("Ref"))),
                ParameterSpec(name="falsifiers", type_spec=type_spec("List", type_spec("Ref"))),
                ParameterSpec(name="selection_goal", type_spec=type_spec("Goal")),
                ParameterSpec(name="selection_requirements", type_spec=type_spec("List", type_spec("Ref"))),
                ParameterSpec(name="preservation_rules", type_spec=type_spec("List", type_spec("Ref"))),
                ParameterSpec(name="dominance_witnesses", type_spec=type_spec("List", type_spec("Any"))),
                ParameterSpec(name="before_revision", type_spec=type_spec("Integer")),
            ),
            output_type=type_spec("Assessment"),
            source_unit="agents/yf_arc3_v5/src/arc3/extend_insufficient_ontology.src",
            source_hash=hashlib.sha256(ontology_extension_path.read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="workflow.activate_dynamic_goal_frontier",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(
                    name="activation",
                    type_spec=type_spec("DynamicGoalFrontierActivation"),
                ),
                ParameterSpec(name="before_revision", type_spec=type_spec("Integer")),
            ),
            output_type=type_spec("PreparedMeaningUpdates"),
            source_unit=(
                "agents/yf_arc3_v5/src/arc3/activate_dynamic_goal_frontier.src"
            ),
            source_hash=hashlib.sha256(
                (root / "arc3" / "activate_dynamic_goal_frontier.src").read_bytes()
            ).hexdigest(),
        ),
        SymbolDefinition(
            id="workflow.plan_goal",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(
                    name="mechanism_gate",
                    type_spec=type_spec("Assessment", optional=True),
                ),
                ParameterSpec(
                    name="declared_gap_request",
                    type_spec=type_spec("DeclaredGoalGapInput", optional=True),
                ),
                ParameterSpec(
                    name="plan_request",
                    type_spec=type_spec("GoalPlanRequest", optional=True),
                ),
                ParameterSpec(
                    name="requirements",
                    type_spec=type_spec("List", type_spec("String")),
                ),
                ParameterSpec(
                    name="preservation_rules",
                    type_spec=type_spec("List", type_spec("String")),
                ),
            ),
            output_type=type_spec("Assessment"),
            source_unit="agents/yf_arc3_v5/src/arc3/plan_goal.src",
            source_hash=hashlib.sha256(plan_goal_path.read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="workflow.evaluate_constrained_plan",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(
                    name="simulation_request",
                    type_spec=type_spec("QuantitySimulationInput", optional=True),
                ),
            ),
            output_type=type_spec("TerminalResult"),
            source_unit=(
                "agents/yf_arc3_v5/src/arc3/evaluate_constrained_plan.src"
            ),
            source_hash=hashlib.sha256(
                evaluate_constrained_plan_path.read_bytes()
            ).hexdigest(),
        ),
        SymbolDefinition(
            id="control.assessment_selection_present",
            kind=SymbolKind.CAPABILITY,
            parameters=(
                ParameterSpec(
                    name="assessment", type_spec=type_spec("Any", optional=True)
                ),
            ),
            output_type=type_spec("Boolean"),
            source_unit=(
                "agents/yf_arc3_v5/src/arc3/project_priority_obligation_frontier.src"
            ),
            source_hash=hashlib.sha256(priority_frontier_path.read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.projected_alternative_boolean",
            kind=SymbolKind.CAPABILITY,
            parameters=(
                ParameterSpec(name="projection", type_spec=type_spec("Any")),
                ParameterSpec(name="alternative_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="field", type_spec=type_spec("String")),
            ),
            output_type=type_spec("Boolean"),
            source_unit="agents/yf_arc3_v5/src/arc3/choose_information_probe.src",
            source_hash=hashlib.sha256(probe_path.read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.prepare_interaction_probe",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(
                    name="agenda",
                    type_spec=type_spec("InteractionProbeAgenda"),
                ),
                ParameterSpec(
                    name="projected_agenda",
                    type_spec=type_spec("Any"),
                    required=False,
                ),
                ParameterSpec(name="selected_ref", type_spec=type_spec("Ref")),
                ParameterSpec(
                    name="context",
                    type_spec=type_spec("WorldContext"),
                ),
                ParameterSpec(
                    name="selection_policy_ref",
                    type_spec=type_spec("String"),
                    required=False,
                ),
                ParameterSpec(
                    name="selection_source_ref",
                    type_spec=type_spec("String"),
                    required=False,
                ),
                ParameterSpec(
                    name="candidate_local_goal_context_required",
                    type_spec=type_spec("Boolean"),
                    required=False,
                ),
                ParameterSpec(
                    name="point_locus_authorized",
                    type_spec=type_spec("Boolean", optional=True),
                    required=False,
                ),
            ),
            output_type=type_spec("PreparedInteractionProbe"),
            source_unit=("agents/yf_arc3_v5/src/arc3/choose_information_probe.src"),
            source_hash=hashlib.sha256(probe_path.read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.validate_action_grounding",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="intent", type_spec=type_spec("ActionIntent")),
            ),
            output_type=type_spec("ActionGroundingContract"),
            source_unit=(
                "agents/yf_arc3_v5/src/arc3/choose_information_probe.src"
            ),
            source_hash=hashlib.sha256(probe_path.read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.action_grounding_is_valid",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="intent", type_spec=type_spec("ActionIntent")),
            ),
            output_type=type_spec("Boolean"),
            source_unit=(
                "agents/yf_arc3_v5/src/arc3/choose_information_probe.src"
            ),
            source_hash=hashlib.sha256(probe_path.read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.bind_interaction_agenda",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(
                    name="agenda", type_spec=type_spec("InteractionProbeAgenda")
                ),
                ParameterSpec(name="projected_agenda", type_spec=type_spec("Any")),
            ),
            output_type=type_spec("InteractionProbeAgenda"),
            source_unit=(
                "agents/yf_arc3_v5/src/arc3/choose_information_probe.src"
            ),
            source_hash=hashlib.sha256(probe_path.read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.exclude_interaction_candidate_after_grounding_rejection",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(
                    name="agenda", type_spec=type_spec("InteractionProbeAgenda")
                ),
                ParameterSpec(name="selected_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="failure_kind", type_spec=type_spec("String")),
            ),
            output_type=type_spec("InteractionProbeAgenda"),
            source_unit=(
                "agents/yf_arc3_v5/src/arc3/choose_information_probe.src"
            ),
            source_hash=hashlib.sha256(probe_path.read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.project_declared_alternatives",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(
                    name="selection_policy_ref", type_spec=type_spec("String")
                ),
                ParameterSpec(
                    name="alternative_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(name="descriptive_facts", type_spec=type_spec("Any")),
            ),
            output_type=type_spec("DeclaredAlternativeFacts"),
            source_unit=(
                "agents/yf_arc3_v5/src/core/project_declared_alternatives.src"
            ),
            source_hash=hashlib.sha256(
                (root / "core" / "project_declared_alternatives.src").read_bytes()
            ).hexdigest(),
        ),
        SymbolDefinition(
            id="control.project_declared_agenda",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(
                    name="selection_policy_ref", type_spec=type_spec("String")
                ),
                ParameterSpec(
                    name="agenda", type_spec=type_spec("InteractionProbeAgenda")
                ),
            ),
            output_type=type_spec("InteractionProbeAgenda"),
            source_unit=(
                "agents/yf_arc3_v5/src/core/project_declared_alternatives.src"
            ),
            source_hash=hashlib.sha256(
                (root / "core" / "project_declared_alternatives.src").read_bytes()
            ).hexdigest(),
        ),
        SymbolDefinition(
            id="control.project_declared_agenda_delta",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(
                    name="selection_policy_ref", type_spec=type_spec("String")
                ),
                ParameterSpec(
                    name="agenda", type_spec=type_spec("InteractionProbeAgenda")
                ),
                ParameterSpec(
                    name="reasoning_strategy_ref",
                    type_spec=type_spec("Ref", optional=True),
                ),
                ParameterSpec(
                    name="active_priority_node_ref",
                    type_spec=type_spec("Ref", optional=True),
                ),
                ParameterSpec(
                    name="dynamic_goal_selection_present",
                    type_spec=type_spec("Boolean"),
                    required=False,
                ),
                ParameterSpec(
                    name="selected_dynamic_goal_ref",
                    type_spec=type_spec("Ref", optional=True),
                    required=False,
                ),
                ParameterSpec(
                    name="frozen_pre_reaction_click_frontier_exhausted",
                    type_spec=type_spec("Boolean"),
                    required=False,
                ),
            ),
            output_type=type_spec("Any"),
            source_unit=(
                "agents/yf_arc3_v5/src/core/project_declared_alternatives.src"
            ),
            source_hash=hashlib.sha256(
                (root / "core" / "project_declared_alternatives.src").read_bytes()
            ).hexdigest(),
        ),
        SymbolDefinition(
            id="control.materialize_declared_agenda_frontier",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(
                    name="selection_policy_ref", type_spec=type_spec("String")
                ),
                ParameterSpec(name="goal_ref", type_spec=type_spec("String")),
                ParameterSpec(
                    name="agenda", type_spec=type_spec("InteractionProbeAgenda")
                ),
                ParameterSpec(name="projected_agenda", type_spec=type_spec("Any")),
                ParameterSpec(
                    name="maximum_candidate_count", type_spec=type_spec("Integer")
                ),
            ),
            output_type=type_spec("InteractionProbeAgenda"),
            source_unit=(
                "agents/yf_arc3_v5/src/arc3/choose_information_probe.src"
            ),
            source_hash=hashlib.sha256(probe_path.read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.retain_selected_agenda",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(
                    name="agenda", type_spec=type_spec("InteractionProbeAgenda")
                ),
                ParameterSpec(name="selected_ref", type_spec=type_spec("Ref")),
            ),
            output_type=type_spec("InteractionProbeAgenda"),
            source_unit=(
                "agents/yf_arc3_v5/src/arc3/construct_transferred_quantized_plan.src"
            ),
            source_hash=hashlib.sha256(
                (root / "arc3" / "construct_transferred_quantized_plan.src").read_bytes()
            ).hexdigest(),
        ),
        SymbolDefinition(
            id="control.project_declared_delta",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(
                    name="selection_policy_ref", type_spec=type_spec("String")
                ),
                ParameterSpec(
                    name="alternative_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(name="context_facts", type_spec=type_spec("Any")),
                ParameterSpec(
                    name="descriptive_delta", type_spec=type_spec("Any")
                ),
            ),
            output_type=type_spec("DeclaredAlternativeFacts"),
            source_unit=(
                "agents/yf_arc3_v5/src/core/project_declared_alternatives.src"
            ),
            source_hash=hashlib.sha256(
                (root / "core" / "project_declared_alternatives.src").read_bytes()
            ).hexdigest(),
        ),
        SymbolDefinition(
            id="control.measure_declared_role_appearance",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="tracking", type_spec=type_spec("TemporalTrackingResult", optional=True)),
                ParameterSpec(name="before_tracking", type_spec=type_spec("TemporalTrackingResult", optional=True), required=False),
                ParameterSpec(name="canonical_term_candidates", type_spec=type_spec("List", type_spec("Any"))),
                ParameterSpec(name="canonical_claim_candidates", type_spec=type_spec("List", type_spec("Any"))),
                ParameterSpec(name="transition_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="learning_policy_ref", type_spec=type_spec("String")),
                ParameterSpec(name="key_policy_ref", type_spec=type_spec("String")),
                ParameterSpec(name="measurement_basis_ref", type_spec=type_spec("String")),
                ParameterSpec(name="cell_quantum", type_spec=type_spec("Integer", optional=True)),
                ParameterSpec(name="max_items", type_spec=type_spec("Integer")),
            ),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/interpret_controlled_transition.src",
            source_hash=hashlib.sha256((root / "arc3" / "interpret_controlled_transition.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.measure_typed_cell_lattices", kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="scene", type_spec=type_spec("Any", optional=True)),
                ParameterSpec(name="observation_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="measurement_policy_ref", type_spec=type_spec("String")),
            ),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/perceptual_memory.src",
            source_hash=hashlib.sha256((root / "arc3" / "perceptual_memory.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.measure_logical_command_steps", kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="tracking", type_spec=type_spec("TemporalTrackingResult", optional=True)),
                ParameterSpec(name="measurement_policy_ref", type_spec=type_spec("String")),
            ),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/perceptual_memory.src",
            source_hash=hashlib.sha256((root / "arc3" / "perceptual_memory.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.measure_recalled_navigation_lattices", kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="tracking", type_spec=type_spec("TemporalTrackingResult")),
                ParameterSpec(name="measurement_policy_ref", type_spec=type_spec("String")),
            ),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/perceptual_memory.src",
            source_hash=hashlib.sha256((root / "arc3" / "perceptual_memory.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.measure_current_terminal_inventory", kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="tracking", type_spec=type_spec("TemporalTrackingResult")),
                ParameterSpec(name="consultation_policy_ref", type_spec=type_spec("String")),
            ),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/perceptual_memory.src",
            source_hash=hashlib.sha256((root / "arc3" / "perceptual_memory.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.measure_winning_navigation", kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="transition_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="measurement_policy_ref", type_spec=type_spec("String")),
                ParameterSpec(name="key_policy_ref", type_spec=type_spec("String")),
            ),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/record_verified_goal_completion.src",
            source_hash=hashlib.sha256((root / "arc3" / "record_verified_goal_completion.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.measure_terminal_outcome_evidence", kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="before_raw_input_ref", type_spec=type_spec("Ref", optional=True)),
                ParameterSpec(name="after_raw_input_ref", type_spec=type_spec("Ref", optional=True)),
                ParameterSpec(name="before_frame", type_spec=type_spec("Any", optional=True)),
                ParameterSpec(name="after_frame", type_spec=type_spec("Any", optional=True)),
                ParameterSpec(name="completion_facts", type_spec=type_spec("Any")),
                ParameterSpec(name="transition_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="measurement_policy_ref", type_spec=type_spec("String")),
            ),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/record_verified_goal_completion.src",
            source_hash=hashlib.sha256((root / "arc3" / "record_verified_goal_completion.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.measure_local_field_type_inventory", kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="scene", type_spec=type_spec("Any")),
                ParameterSpec(name="tracking", type_spec=type_spec("Any")),
                ParameterSpec(name="context", type_spec=type_spec("WorldContext")),
                ParameterSpec(name="measurement_policy_ref", type_spec=type_spec("String")),
            ),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/choose_information_probe.src",
            source_hash=hashlib.sha256((root / "arc3" / "choose_information_probe.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.measure_declared_stock_dependency", kind=SymbolKind.WORKFLOW,
            parameters=(ParameterSpec(name="inventory_facts", type_spec=type_spec("Any")),),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/choose_information_probe.src",
            source_hash=hashlib.sha256((root / "arc3" / "choose_information_probe.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.measure_declared_pair_point_geometry", kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="dependency_measurements", type_spec=type_spec("Any")),
                ParameterSpec(name="measurement_policy_ref", type_spec=type_spec("String")),
            ),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/choose_information_probe.src",
            source_hash=hashlib.sha256((root / "arc3" / "choose_information_probe.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.bind_declared_pair_point_agenda", kind=SymbolKind.WORKFLOW,
            parameters=(ParameterSpec(name="projected", type_spec=type_spec("DeclaredAlternativeFacts")),),
            output_type=type_spec("InteractionProbeAgenda"),
            source_unit="agents/yf_arc3_v5/src/arc3/choose_information_probe.src",
            source_hash=hashlib.sha256((root / "arc3" / "choose_information_probe.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.measure_current_pair_stock_change", kind=SymbolKind.WORKFLOW,
            parameters=(ParameterSpec(name="inventory_facts", type_spec=type_spec("Any")),
                ParameterSpec(name="measurement_policy_ref", type_spec=type_spec("String"))),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/choose_information_probe.src",
            source_hash=hashlib.sha256((root / "arc3" / "choose_information_probe.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.bind_current_stock_lineage", kind=SymbolKind.WORKFLOW,
            parameters=(ParameterSpec(name="inventory_facts", type_spec=type_spec("Any")),),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/choose_information_probe.src",
            source_hash=hashlib.sha256((root / "arc3" / "choose_information_probe.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.measure_current_local_field_delivery", kind=SymbolKind.WORKFLOW,
            parameters=(ParameterSpec(name="inventory_facts", type_spec=type_spec("Any")),
                ParameterSpec(name="scene", type_spec=type_spec("Any")),
                ParameterSpec(name="measurement_policy_ref", type_spec=type_spec("String"))),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/choose_information_probe.src",
            source_hash=hashlib.sha256((root / "arc3" / "choose_information_probe.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.measure_reference_write_evidence", kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="scene", type_spec=type_spec("Any", optional=True)),
                ParameterSpec(name="outcome", type_spec=type_spec("Any")),
                ParameterSpec(name="evidence_refs", type_spec=type_spec("List", type_spec("Ref"))),
                ParameterSpec(name="measurement_policy_ref", type_spec=type_spec("String")),
            ),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/record_verified_goal_completion.src",
            source_hash=hashlib.sha256((root / "arc3" / "record_verified_goal_completion.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.measure_terminal_coverage_evidence", kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="scene", type_spec=type_spec("Any", optional=True)),
                ParameterSpec(name="frame_rows", type_spec=type_spec("Any", optional=True), required=False),
                ParameterSpec(name="action_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="transition_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="context_epoch", type_spec=type_spec("Integer", optional=True)),
                ParameterSpec(name="measurement_policy_ref", type_spec=type_spec("String")),
            ),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/record_verified_goal_completion.src",
            source_hash=hashlib.sha256((root / "arc3" / "record_verified_goal_completion.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.measure_canonical_activity_support",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="activity_facts", type_spec=type_spec("Any")),
                ParameterSpec(name="contract_facts", type_spec=type_spec("Any", optional=True), required=False),
                ParameterSpec(name="canonical_claim_candidates", type_spec=type_spec("List", type_spec("Any"))),
                ParameterSpec(name="transition_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="admission_policy_ref", type_spec=type_spec("String")),
            ),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/reconcile_transferred_quantized_plan.src",
            source_hash=hashlib.sha256((root / "arc3" / "reconcile_transferred_quantized_plan.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.measure_current_shared_method_inventory",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="tracking", type_spec=type_spec("TemporalTrackingResult")),
                ParameterSpec(name="consultation_policy_ref", type_spec=type_spec("String")),
            ),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/perceptual_memory.src",
            source_hash=hashlib.sha256((root / "arc3" / "perceptual_memory.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.measure_current_shared_effect_inventory",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="tracking", type_spec=type_spec("TemporalTrackingResult")),
                ParameterSpec(name="consultation_policy_ref", type_spec=type_spec("String")),
            ),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/perceptual_memory.src",
            source_hash=hashlib.sha256((root / "arc3" / "perceptual_memory.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.measure_current_method_activity_coverage",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="agenda", type_spec=type_spec("InteractionProbeAgenda")),
                ParameterSpec(name="tracking", type_spec=type_spec("TemporalTrackingResult")),
                ParameterSpec(name="mechanism_evidence", type_spec=type_spec("List", type_spec("Any"))),
                ParameterSpec(name="coverage_policy_ref", type_spec=type_spec("String")),
            ),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/construct_transferred_quantized_plan.src",
            source_hash=hashlib.sha256((root / "arc3" / "construct_transferred_quantized_plan.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.project_current_method_activity_premises",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="agenda", type_spec=type_spec("InteractionProbeAgenda")),
                ParameterSpec(name="coverage", type_spec=type_spec("Any")),
                ParameterSpec(name="basis_refs", type_spec=type_spec("List", type_spec("Ref"))),
                ParameterSpec(name="frame_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="plan_policy_ref", type_spec=type_spec("String")),
            ),
            output_type=type_spec("InteractionProbeAgenda"),
            source_unit="agents/yf_arc3_v5/src/arc3/construct_transferred_quantized_plan.src",
            source_hash=hashlib.sha256((root / "arc3" / "construct_transferred_quantized_plan.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.project_memorial_binding_rows",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="recognition_facts", type_spec=type_spec("Any")),
                ParameterSpec(name="key_policy_ref", type_spec=type_spec("String")),
            ),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/perceptual_memory.src",
            source_hash=hashlib.sha256((root / "arc3" / "perceptual_memory.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.read_perceptual_memory_matches",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="scene", type_spec=type_spec("VisualSceneDescription", optional=True)),
                ParameterSpec(name="tracking", type_spec=type_spec("TemporalTrackingResult", optional=True)),
                ParameterSpec(name="current_support_requests", type_spec=type_spec("List", type_spec("Any"), optional=True)),
                ParameterSpec(name="omit_current_bindings", type_spec=type_spec("Boolean")),
                ParameterSpec(name="observation_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="key_policy_ref", type_spec=type_spec("String")),
                ParameterSpec(name="measurement_basis_ref", type_spec=type_spec("String")),
                ParameterSpec(name="cell_quantum", type_spec=type_spec("Integer", optional=True)),
                ParameterSpec(name="max_bearers", type_spec=type_spec("Integer")),
                ParameterSpec(name="max_matches_per_bearer", type_spec=type_spec("Integer")),
                ParameterSpec(name="max_rows", type_spec=type_spec("Integer")),
            ),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/perceptual_memory.src",
            source_hash=hashlib.sha256((root / "arc3" / "perceptual_memory.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="workflow.assimilate_frame",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="raw_frame", type_spec=type_spec("Any")),
                ParameterSpec(name="raw_input_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="available_action_refs", type_spec=type_spec("List", type_spec("Ref"))),
                ParameterSpec(name="is_initial_observation", type_spec=type_spec("Boolean")),
                ParameterSpec(name="action_conditioned_grid_evidence", type_spec=type_spec("Boolean")),
                ParameterSpec(name="before_revision", type_spec=type_spec("Integer")),
                ParameterSpec(name="retained_grid_geometry", type_spec=type_spec("Any", optional=True)),
                ParameterSpec(name="retained_grid_separator_consistency_ppm", type_spec=type_spec("Integer", optional=True)),
            ),
            output_type=type_spec("VisualSceneDescription"),
            source_unit="agents/yf_arc3_v5/src/arc3/assimilate_frame.src",
            source_hash=hashlib.sha256((root / "arc3" / "assimilate_frame.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="workflow.assimilate_block_grid_frame",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="raw_frame", type_spec=type_spec("Any")),
                ParameterSpec(name="raw_input_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="action_conditioned_grid_evidence", type_spec=type_spec("Boolean")),
                ParameterSpec(name="before_revision", type_spec=type_spec("Integer")),
                ParameterSpec(name="retained_grid_geometry", type_spec=type_spec("Any", optional=True)),
                ParameterSpec(name="retained_grid_separator_consistency_ppm", type_spec=type_spec("Integer")),
            ),
            output_type=type_spec("BlockGridSceneDescription"),
            source_unit="agents/yf_arc3_v5/src/arc3/assimilate_frame.src",
            source_hash=hashlib.sha256((root / "arc3" / "assimilate_frame.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="workflow.track_scene",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="descriptive_components", type_spec=type_spec("ComponentExtractionResult")),
                ParameterSpec(name="frame_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="previous_tracking", type_spec=type_spec("Any", optional=True)),
                ParameterSpec(name="action_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="reset_correspondence", type_spec=type_spec("Boolean")),
            ),
            output_type=type_spec("TemporalTrackingResult"),
            source_unit="agents/yf_arc3_v5/src/arc3/track_scene.src",
            source_hash=hashlib.sha256((root / "arc3" / "track_scene.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="workflow.bind_typed_cell_lattices",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="scene", type_spec=type_spec("Any", optional=True)),
                ParameterSpec(name="frame_ref", type_spec=type_spec("Ref")),
            ),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/perceptual_memory.src",
            source_hash=hashlib.sha256((root / "arc3" / "perceptual_memory.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="workflow.retain_observed_logical_command_steps",
            kind=SymbolKind.WORKFLOW,
            parameters=(ParameterSpec(name="tracking", type_spec=type_spec("TemporalTrackingResult", optional=True)),),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/perceptual_memory.src",
            source_hash=hashlib.sha256((root / "arc3" / "perceptual_memory.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="workflow.recall_navigation_cell_lattice", kind=SymbolKind.WORKFLOW,
            parameters=(ParameterSpec(name="tracking", type_spec=type_spec("TemporalTrackingResult")),),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/perceptual_memory.src",
            source_hash=hashlib.sha256((root / "arc3" / "perceptual_memory.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="workflow.refresh_tracked_roles",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="tracking", type_spec=type_spec("TemporalTrackingResult")),
                ParameterSpec(name="scene", type_spec=type_spec("Any", optional=True)),
                ParameterSpec(name="current_support_requests", type_spec=type_spec("List", type_spec("Any"), optional=True)),
                ParameterSpec(name="before_revision", type_spec=type_spec("Integer")),
            ),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/perceptual_memory.src",
            source_hash=hashlib.sha256((root / "arc3" / "perceptual_memory.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="workflow.consult_current_shared_effects",
            kind=SymbolKind.WORKFLOW,
            parameters=(ParameterSpec(name="tracking", type_spec=type_spec("TemporalTrackingResult")),),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/perceptual_memory.src",
            source_hash=hashlib.sha256((root / "arc3" / "perceptual_memory.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="workflow.assess_mechanism_commitment_readiness",
            kind=SymbolKind.WORKFLOW,
            parameters=(ParameterSpec(name="descriptive_readiness_facts", type_spec=type_spec("Any")),
                ParameterSpec(name="evidence_refs", type_spec=type_spec("List", type_spec("Ref"))),
                ParameterSpec(name="before_revision", type_spec=type_spec("Integer"))),
            output_type=type_spec("Assessment"),
            source_unit="agents/yf_arc3_v5/src/arc3/assess_mechanism_commitment_readiness.src",
            source_hash=hashlib.sha256((root / "arc3" / "assess_mechanism_commitment_readiness.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="workflow.assess_elimination_viability",
            kind=SymbolKind.WORKFLOW,
            parameters=(ParameterSpec(name="mechanism_gate", type_spec=type_spec("Assessment")),
                ParameterSpec(name="measurement_request", type_spec=type_spec("Any")),
                ParameterSpec(name="evidence_refs", type_spec=type_spec("List", type_spec("Ref"))),
                ParameterSpec(name="before_revision", type_spec=type_spec("Integer"))),
            output_type=type_spec("Assessment"),
            source_unit="agents/yf_arc3_v5/src/arc3/assess_elimination_viability.src",
            source_hash=hashlib.sha256((root / "arc3" / "assess_elimination_viability.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="workflow.assess_orthogonal_perimeter_assembly",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="components", type_spec=type_spec("ComponentExtractionResult")),
                ParameterSpec(name="evidence_refs", type_spec=type_spec("List", type_spec("Ref"))),
                ParameterSpec(name="before_revision", type_spec=type_spec("Integer")),
            ),
            output_type=type_spec("Assessment"),
            source_unit="agents/yf_arc3_v5/src/arc3/assess_orthogonal_perimeter_assembly.src",
            source_hash=hashlib.sha256((root / "arc3" / "assess_orthogonal_perimeter_assembly.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="workflow.consult_current_shared_method",
            kind=SymbolKind.WORKFLOW,
            parameters=(ParameterSpec(name="tracking", type_spec=type_spec("TemporalTrackingResult")),),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/perceptual_memory.src",
            source_hash=hashlib.sha256((root / "arc3" / "perceptual_memory.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.measure_memorial_selection", kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="tracking", type_spec=type_spec("TemporalTrackingResult")),
                ParameterSpec(name="selection_policy_ref", type_spec=type_spec("String")),
            ),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/perceptual_memory.src",
            source_hash=hashlib.sha256((root / "arc3" / "perceptual_memory.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.instantiate_declared_meaning",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(
                    name="selection_policy_ref", type_spec=type_spec("String")
                ),
                ParameterSpec(name="selected_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="alternative_facts", type_spec=type_spec("Any")),
                ParameterSpec(name="existing_term_refs", type_spec=type_spec("List", type_spec("Ref")), required=False),
                ParameterSpec(
                    name="evidence_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
            ),
            output_type=type_spec("DeclaredMeaningInstantiation"),
            source_unit=(
                "agents/yf_arc3_v5/src/arc3/construct_transition_meaning.src"
            ),
            source_hash=hashlib.sha256(meaning_path.read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.prepare_optional_meaning_updates",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="terms", type_spec=type_spec("List", type_spec("Any"))),
                ParameterSpec(name="claims", type_spec=type_spec("List", type_spec("Any"))),
            ),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/assess_orthogonal_perimeter_assembly.src",
            source_hash=hashlib.sha256((root / "arc3" / "assess_orthogonal_perimeter_assembly.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.prepare_meaning_updates",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(
                    name="terms", type_spec=type_spec("List", type_spec("Any"))
                ),
                ParameterSpec(
                    name="claims", type_spec=type_spec("List", type_spec("Any"))
                ),
            ),
            output_type=type_spec("PreparedMeaningUpdates"),
            source_unit=(
                "agents/yf_arc3_v5/src/arc3/construct_transition_meaning.src"
            ),
            source_hash=hashlib.sha256(meaning_path.read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.link_disjoint_update_batch",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="previous", type_spec=type_spec("Any", optional=True)),
                ParameterSpec(name="items", type_spec=type_spec("List", type_spec("Any"))),
                ParameterSpec(name="evidence", type_spec=type_spec("List", type_spec("Ref"))),
            ),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/state/update_batch.py",
            source_hash=hashlib.sha256((root.parent / "state" / "update_batch.py").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.flatten_disjoint_update_batch",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="chunks", type_spec=type_spec("Any")),
                ParameterSpec(name="maximum_items", type_spec=type_spec("Integer")),
            ),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/state/update_batch.py",
            source_hash=hashlib.sha256((root.parent / "state" / "update_batch.py").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.prepare_revisable_meaning_updates",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(
                    name="terms", type_spec=type_spec("List", type_spec("Any"))
                ),
                ParameterSpec(
                    name="claims", type_spec=type_spec("List", type_spec("Any"))
                ),
            ),
            output_type=type_spec("PreparedMeaningUpdates"),
            source_unit=(
                "agents/yf_arc3_v5/src/arc3/assess_exact_multicell_peer_goals.src"
            ),
            source_hash=hashlib.sha256(
                (
                    root
                    / "arc3"
                    / "assess_exact_multicell_peer_goals.src"
                ).read_bytes()
            ).hexdigest(),
        ),
        SymbolDefinition(
            id="control.instantiate_dynamic_goal_frontier_activation",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="facts", type_spec=type_spec("Any")),
            ),
            output_type=type_spec("DynamicGoalFrontierActivation"),
            source_unit=(
                "agents/yf_arc3_v5/src/arc3/activate_dynamic_goal_frontier.src"
            ),
            source_hash=hashlib.sha256(
                (root / "arc3" / "activate_dynamic_goal_frontier.src").read_bytes()
            ).hexdigest(),
        ),
        SymbolDefinition(
            id="control.instantiate_dynamic_goal_frontier_meaning",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(
                    name="activation",
                    type_spec=type_spec("DynamicGoalFrontierActivation"),
                ),
            ),
            output_type=type_spec("DeclaredMeaningInstantiation"),
            source_unit=(
                "agents/yf_arc3_v5/src/arc3/activate_dynamic_goal_frontier.src"
            ),
            source_hash=hashlib.sha256(
                (root / "arc3" / "activate_dynamic_goal_frontier.src").read_bytes()
            ).hexdigest(),
        ),
        SymbolDefinition(
            id="control.dynamic_goal_frontier_payload_present",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="facts", type_spec=type_spec("Any")),
            ),
            output_type=type_spec("Boolean"),
            source_unit=(
                "agents/yf_arc3_v5/src/arc3/activate_dynamic_goal_frontier.src"
            ),
            source_hash=hashlib.sha256(
                (root / "arc3" / "activate_dynamic_goal_frontier.src").read_bytes()
            ).hexdigest(),
        ),
        SymbolDefinition(
            id="control.activation_preserves_selected_action_goal",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="facts", type_spec=type_spec("Any")),
                ParameterSpec(name="intent", type_spec=type_spec("ActionIntent")),
            ),
            output_type=type_spec("Boolean"),
            source_unit=(
                "agents/yf_arc3_v5/src/arc3/choose_information_probe.src"
            ),
            source_hash=hashlib.sha256(
                (root / "arc3" / "choose_information_probe.src").read_bytes()
            ).hexdigest(),
        ),
        SymbolDefinition(
            id="control.dynamic_goal_frontier_activation_needs_commit",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(
                    name="activation",
                    type_spec=type_spec("DynamicGoalFrontierActivation"),
                ),
            ),
            output_type=type_spec("Boolean"),
            source_unit=(
                "agents/yf_arc3_v5/src/arc3/choose_information_probe.src"
            ),
            source_hash=hashlib.sha256(
                (root / "arc3" / "choose_information_probe.src").read_bytes()
            ).hexdigest(),
        ),
        SymbolDefinition(
            id="control.access_goal_completion_payload_present",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="facts", type_spec=type_spec("Any")),
            ),
            output_type=type_spec("Boolean"),
            source_unit=(
                "agents/yf_arc3_v5/src/arc3/construct_transition_meaning.src"
            ),
            source_hash=hashlib.sha256(
                (root / "arc3" / "construct_transition_meaning.src").read_bytes()
            ).hexdigest(),
        ),
        SymbolDefinition(
            id="control.committed_access_plan_present",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="facts", type_spec=type_spec("Any")),
                ParameterSpec(name="selected_key", type_spec=type_spec("String")),
            ),
            output_type=type_spec("Boolean"),
            source_unit=(
                "agents/yf_arc3_v5/src/arc3/construct_transition_meaning.src"
            ),
            source_hash=hashlib.sha256(
                (root / "arc3" / "construct_transition_meaning.src").read_bytes()
            ).hexdigest(),
        ),
        SymbolDefinition(
            id="control.prepare_goal_priority_context_updates",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="terms", type_spec=type_spec("List", type_spec("Any"))),
                ParameterSpec(name="claims", type_spec=type_spec("List", type_spec("Any"))),
                ParameterSpec(name="selected_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="disabled_reason_ref", type_spec=type_spec("String")),
            ),
            output_type=type_spec("PreparedMeaningUpdates"),
            source_unit=(
                "agents/yf_arc3_v5/src/arc3/choose_information_probe.src"
            ),
            source_hash=hashlib.sha256(
                (root / "arc3" / "choose_information_probe.src").read_bytes()
            ).hexdigest(),
        ),
        SymbolDefinition(
            id="control.prepare_committed_derivation",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="derivation", type_spec=type_spec("Any")),
            ),
            output_type=type_spec("PreparedCommittedDerivation"),
            source_unit="agents/yf_arc3_v5/src/arc3/extend_insufficient_ontology.src",
            source_hash=hashlib.sha256(ontology_extension_path.read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.bind_derivation_basis",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="derivation", type_spec=type_spec("Any")),
                ParameterSpec(name="basis_refs", type_spec=type_spec("List", type_spec("Ref"))),
            ),
            output_type=type_spec("List", type_spec("Ref")),
            source_unit="agents/yf_arc3_v5/src/arc3/extend_insufficient_ontology.src",
            source_hash=hashlib.sha256(ontology_extension_path.read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.prepare_committed_proposal",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="proposal", type_spec=type_spec("Any")),
                ParameterSpec(name="decision", type_spec=type_spec("Assessment")),
            ),
            output_type=type_spec("PreparedCommittedProposal"),
            source_unit=(
                "agents/yf_arc3_v5/src/arc3/extend_insufficient_ontology.src"
            ),
            source_hash=hashlib.sha256(
                (root / "arc3" / "extend_insufficient_ontology.src").read_bytes()
            ).hexdigest(),
        ),
        SymbolDefinition(
            id="control.bind_declared_scene_handoff",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="transition_ref", type_spec=type_spec("Ref")),
            ),
            output_type=type_spec("Any"),
            source_unit=(
                "agents/yf_arc3_v5/src/arc3/record_verified_goal_completion.src"
            ),
            source_hash=hashlib.sha256(
                (
                    root / "arc3" / "record_verified_goal_completion.src"
                ).read_bytes()
            ).hexdigest(),
        ),
        SymbolDefinition(
            id="control.refresh_world_context_snapshot",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(
                    name="context", type_spec=type_spec("WorldContext")
                ),
            ),
            output_type=type_spec("WorldContext"),
            source_unit=(
                "agents/yf_arc3_v5/src/arc3/choose_verified_alignment_action.src"
            ),
            source_hash=hashlib.sha256(
                (
                    root / "arc3" / "choose_verified_alignment_action.src"
                ).read_bytes()
            ).hexdigest(),
        ),
        SymbolDefinition(
            id="control.current_state_revision",
            kind=SymbolKind.WORKFLOW,
            parameters=(),
            output_type=type_spec("Integer"),
            source_unit=(
                "agents/yf_arc3_v5/src/arc3/extend_insufficient_ontology.src"
            ),
            source_hash=hashlib.sha256(
                (root / "arc3" / "extend_insufficient_ontology.src").read_bytes()
            ).hexdigest(),
        ),
        SymbolDefinition(
            id="control.current_term_attribute_values",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="filter_attribute", type_spec=type_spec("String")),
                ParameterSpec(name="filter_value", type_spec=type_spec("String")),
                ParameterSpec(name="returned_attribute", type_spec=type_spec("String")),
                ParameterSpec(name="max_items", type_spec=type_spec("Integer")),
                ParameterSpec(name="exclude_attribute", type_spec=type_spec("String"), required=False),
                ParameterSpec(name="exclude_value", type_spec=type_spec("Any"), required=False),
            ),
            output_type=type_spec("List", type_spec("Ref")),
            source_unit="agents/yf_arc3_v5/src/arc3/interpret_controlled_transition.src",
            source_hash=hashlib.sha256((root / "arc3" / "interpret_controlled_transition.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.current_term_attribute_rows",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="filter_attribute", type_spec=type_spec("String")),
                ParameterSpec(name="filter_value", type_spec=type_spec("String")),
                ParameterSpec(name="additional_filter_attribute", type_spec=type_spec("String"), required=False),
                ParameterSpec(name="additional_filter_value", type_spec=type_spec("String"), required=False),
                ParameterSpec(name="returned_attributes", type_spec=type_spec("List", type_spec("String"))),
                ParameterSpec(name="max_items", type_spec=type_spec("Integer")),
            ),
            output_type=type_spec("List", type_spec("Any")),
            source_unit="agents/yf_arc3_v5/src/arc3/interpret_display_correspondence.src",
            source_hash=hashlib.sha256((root / "arc3" / "interpret_display_correspondence.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.current_context_term_attribute_rows", kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="only_term_refs", type_spec=type_spec("List", type_spec("Ref"), optional=True), required=False),
                ParameterSpec(name="filter_attribute", type_spec=type_spec("String")),
                ParameterSpec(name="filter_value", type_spec=type_spec("String")),
                ParameterSpec(name="additional_filter_attribute", type_spec=type_spec("String"), required=False),
                ParameterSpec(name="additional_filter_value", type_spec=type_spec("String"), required=False),
                ParameterSpec(name="exact_attribute_filters", type_spec=type_spec("Any"), required=False),
                ParameterSpec(name="distinct_rows", type_spec=type_spec("Boolean"), required=False),
                ParameterSpec(name="group_by_attribute", type_spec=type_spec("String"), required=False),
                ParameterSpec(name="max_groups", type_spec=type_spec("Integer"), required=False),
                ParameterSpec(name="claim_projection_policy_ref", type_spec=type_spec("String"), required=False),
                ParameterSpec(name="returned_attributes", type_spec=type_spec("List", type_spec("String"))),
                ParameterSpec(name="optional_returned_attributes", type_spec=type_spec("List", type_spec("String")), required=False),
                ParameterSpec(name="include_term_ref", type_spec=type_spec("Boolean"), required=False),
                ParameterSpec(name="present_entity_rows", type_spec=type_spec("List", type_spec("Any")), required=False),
                ParameterSpec(name="present_entity_members_attribute", type_spec=type_spec("String"), required=False),
                ParameterSpec(name="present_tracking", type_spec=type_spec("TemporalTrackingResult"), required=False),
                ParameterSpec(name="max_items", type_spec=type_spec("Integer")),
                ParameterSpec(name="action_ref", type_spec=type_spec("Ref", optional=True)),
                ParameterSpec(name="context_epoch", type_spec=type_spec("Integer")),
            ),
            output_type=type_spec("List", type_spec("Any")),
            source_unit="agents/yf_arc3_v5/src/arc3/interpret_controlled_transition.src",
            source_hash=hashlib.sha256((root / "arc3" / "interpret_controlled_transition.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.measure_observed_extent_commands", kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="analysis", type_spec=type_spec("ControlledTransitionAnalysis")),
                ParameterSpec(name="before_tracking", type_spec=type_spec("TemporalTrackingResult")),
                ParameterSpec(name="after_tracking", type_spec=type_spec("TemporalTrackingResult")),
                ParameterSpec(name="operator_scope_ref", type_spec=type_spec("Ref", optional=True)),
                ParameterSpec(name="operator_action_data", type_spec=type_spec("Any", optional=True)),
                ParameterSpec(name="source_palette_value", type_spec=type_spec("Integer", optional=True)),
                ParameterSpec(name="action_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="context_epoch", type_spec=type_spec("Integer")),
            ),
            output_type=type_spec("List", type_spec("Any")),
            source_unit="agents/yf_arc3_v5/src/arc3/interpret_controlled_transition.src",
            source_hash=hashlib.sha256((root / "arc3" / "interpret_controlled_transition.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.measure_transformation_pair_recurrence", kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="operator_scope_ref", type_spec=type_spec("Ref", optional=True), required=False),
                ParameterSpec(name="comparison_policy_ref", type_spec=type_spec("String"), required=False),
                ParameterSpec(name="registered_scene_translation", type_spec=type_spec("Any", optional=True), required=False),
                ParameterSpec(name="previous_rows_grouped", type_spec=type_spec("Boolean"), required=False),
                ParameterSpec(name="descriptive_facts", type_spec=type_spec("Any")),
                ParameterSpec(name="previous_rows", type_spec=type_spec("List", type_spec("Any"))),
                ParameterSpec(name="observed_return_rows", type_spec=type_spec("List", type_spec("Any"))),
                ParameterSpec(name="action_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="transition_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="context_epoch", type_spec=type_spec("Integer")),
                ParameterSpec(name="max_pairs", type_spec=type_spec("Integer")),
                ParameterSpec(name="before_scene", type_spec=type_spec("VisualSceneDescription")),
                ParameterSpec(name="after_scene", type_spec=type_spec("VisualSceneDescription")),
                ParameterSpec(name="before_tracking", type_spec=type_spec("TemporalTrackingResult")),
                ParameterSpec(name="after_tracking", type_spec=type_spec("TemporalTrackingResult")),
                ParameterSpec(name="marker_policy_ref", type_spec=type_spec("String")),
                ParameterSpec(name="maximum_focus_rows", type_spec=type_spec("Integer")),
                ParameterSpec(name="shared_pattern_policy_ref", type_spec=type_spec("String")),
                ParameterSpec(name="evidence_refs", type_spec=type_spec("List", type_spec("Ref"))),
                ParameterSpec(name="maximum_translation_steps", type_spec=type_spec("Integer")),
                ParameterSpec(name="maximum_translation_support_pixels", type_spec=type_spec("Integer")),
            ),
            output_type=type_spec("List", type_spec("Any")),
            source_unit="agents/yf_arc3_v5/src/arc3/interpret_controlled_transition.src",
            source_hash=hashlib.sha256((root / "arc3" / "interpret_controlled_transition.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.measure_remembered_coupling_render", kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="registered_scene_translation", type_spec=type_spec("Any", optional=True), required=False),
                ParameterSpec(name="after_frame_ref", type_spec=type_spec("Ref"), required=False),
                ParameterSpec(name="previous_rows_grouped", type_spec=type_spec("Boolean"), required=False),
                ParameterSpec(name="previous_rows", type_spec=type_spec("List", type_spec("Any"))),
                ParameterSpec(name="before_scene", type_spec=type_spec("VisualSceneDescription")),
                ParameterSpec(name="before_tracking", type_spec=type_spec("TemporalTrackingResult")),
                ParameterSpec(name="after_scene", type_spec=type_spec("VisualSceneDescription")),
                ParameterSpec(name="transition_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="maximum_translation_steps", type_spec=type_spec("Integer")),
                ParameterSpec(name="maximum_translation_support_pixels", type_spec=type_spec("Integer")),
                ParameterSpec(name="maximum_scene_layers", type_spec=type_spec("Integer")),
            ),
            output_type=type_spec("List", type_spec("Any")),
            source_unit="agents/yf_arc3_v5/src/arc3/interpret_controlled_transition.src",
            source_hash=hashlib.sha256((root / "arc3" / "interpret_controlled_transition.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.measure_current_coupling_context", kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="review_rows", type_spec=type_spec("List", type_spec("Any"))),
                ParameterSpec(name="return_rows", type_spec=type_spec("List", type_spec("Any"))),
                ParameterSpec(name="scene", type_spec=type_spec("VisualSceneDescription")),
                ParameterSpec(name="tracking", type_spec=type_spec("TemporalTrackingResult")),
                ParameterSpec(name="context_epoch", type_spec=type_spec("Integer")),
                ParameterSpec(name="marker_policy_ref", type_spec=type_spec("String")),
                ParameterSpec(name="maximum_focus_rows", type_spec=type_spec("Integer")),
                ParameterSpec(name="maximum_steps", type_spec=type_spec("Integer")),
                ParameterSpec(name="maximum_support_pixels", type_spec=type_spec("Integer")),
            ),
            output_type=type_spec("List", type_spec("Any")),
            source_unit="agents/yf_arc3_v5/src/arc3/interpret_controlled_transition.src",
            source_hash=hashlib.sha256((root / "arc3" / "interpret_controlled_transition.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.declared_meaning_is_applicable", kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="selection_policy_ref", type_spec=type_spec("String")),
                ParameterSpec(name="selected_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="alternative_facts", type_spec=type_spec("Any")),
                ParameterSpec(name="evidence_refs", type_spec=type_spec("List", type_spec("Ref"))),
            ),
            output_type=type_spec("Boolean"),
            source_unit="agents/yf_arc3_v5/src/arc3/interpret_display_correspondence.src",
            source_hash=hashlib.sha256((root / "arc3" / "interpret_display_correspondence.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.measure_observed_axis_transport", kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="observations", type_spec=type_spec("List", type_spec("Any"))),
                ParameterSpec(name="tracking", type_spec=type_spec("TemporalTrackingResult")),
                ParameterSpec(name="context_epoch", type_spec=type_spec("Integer")),
            ),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/interpret_controlled_transition.src",
            source_hash=hashlib.sha256((root / "arc3" / "interpret_controlled_transition.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.prepare_optional_revisable_meaning_updates", kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="terms", type_spec=type_spec("List", type_spec("Any"))),
                ParameterSpec(name="claims", type_spec=type_spec("List", type_spec("Any"))),
            ),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/interpret_controlled_transition.src",
            source_hash=hashlib.sha256((root / "arc3" / "interpret_controlled_transition.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.measure_carried_reference_center", kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="observations", type_spec=type_spec("List", type_spec("Any"))),
                ParameterSpec(name="tracking", type_spec=type_spec("TemporalTrackingResult")),
                ParameterSpec(name="context_epoch", type_spec=type_spec("Integer")),
            ),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/interpret_controlled_transition.src",
            source_hash=hashlib.sha256((root / "arc3" / "interpret_controlled_transition.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.measure_observed_axis_frontier", kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="selection_policy_ref", type_spec=type_spec("String")),
                ParameterSpec(name="agenda", type_spec=type_spec("InteractionProbeAgenda")),
                ParameterSpec(name="tracking", type_spec=type_spec("Any")),
                ParameterSpec(name="scene", type_spec=type_spec("Any")),
            ),
            output_type=type_spec("Any"),
            source_unit="agents/yf_arc3_v5/src/arc3/choose_information_probe.src",
            source_hash=hashlib.sha256((root / "arc3" / "choose_information_probe.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.declared_selection_has_eligible_alternative", kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="selection_policy_ref", type_spec=type_spec("String")),
                ParameterSpec(name="goal_ref", type_spec=type_spec("String")),
                ParameterSpec(name="alternative_refs", type_spec=type_spec("List", type_spec("Ref"))),
                ParameterSpec(name="alternative_facts", type_spec=type_spec("Any")),
            ),
            output_type=type_spec("Boolean"),
            source_unit="agents/yf_arc3_v5/src/arc3/interpret_controlled_transition.src",
            source_hash=hashlib.sha256((root / "arc3" / "interpret_controlled_transition.src").read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.goal_priority_context_needs_commit",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(name="selected_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="active_ref", type_spec=type_spec("Ref")),
            ),
            output_type=type_spec("Boolean"),
            source_unit=(
                "agents/yf_arc3_v5/src/arc3/choose_information_probe.src"
            ),
            source_hash=hashlib.sha256(
                (root / "arc3" / "choose_information_probe.src").read_bytes()
            ).hexdigest(),
        ),
        SymbolDefinition(
            id="control.synthesize_declared_dynamic_workflow_candidates",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(
                    name="synthesis_request",
                    type_spec=type_spec("Any"),
                ),
            ),
            output_type=type_spec("Any"),
            source_unit=(
                "agents/yf_arc3_v5/dynamic_workflow/synthesis.py"
            ),
            source_hash=hashlib.sha256(dynamic_synthesis_path.read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.probe_action_is_point",
            kind=SymbolKind.CAPABILITY,
            parameters=(
                ParameterSpec(name="agenda", type_spec=type_spec("InteractionProbeAgenda")),
                ParameterSpec(name="selected_ref", type_spec=type_spec("Ref")),
            ),
            output_type=type_spec("Boolean"),
            source_unit="agents/yf_arc3_v5/src/arc3/choose_information_probe.src",
            source_hash=hashlib.sha256(probe_path.read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.probe_fact_is_true",
            kind=SymbolKind.CAPABILITY,
            parameters=(
                ParameterSpec(name="agenda", type_spec=type_spec("InteractionProbeAgenda")),
                ParameterSpec(name="candidate_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="field_name", type_spec=type_spec("String")),
            ),
            output_type=type_spec("Boolean"),
            source_unit="agents/yf_arc3_v5/src/arc3/choose_information_probe.src",
            source_hash=hashlib.sha256(probe_path.read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.verify_declared_continue_terminal",
            kind=SymbolKind.CAPABILITY,
            parameters=(
                ParameterSpec(
                    name="observation", type_spec=type_spec("ObservationResult")
                ),
                ParameterSpec(
                    name="expected_terminal_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="reference_score", type_spec=type_spec("Integer")
                ),
            ),
            output_type=type_spec("TerminalResult"),
            source_unit=(
                "agents/yf_arc3_v5/src/arc3/continue_declared_action.src"
            ),
            source_hash=hashlib.sha256(continue_path.read_bytes()).hexdigest(),
        ),
        SymbolDefinition(
            id="control.reduce_dynamic_workflow_synthesis_facts",
            kind=SymbolKind.WORKFLOW,
            parameters=(
                ParameterSpec(
                    name="synthesis_result",
                    type_spec=type_spec("Any"),
                ),
            ),
            output_type=type_spec("Any"),
            source_unit=(
                "agents/yf_arc3_v5/dynamic_workflow/synthesis.py"
            ),
            source_hash=hashlib.sha256(dynamic_synthesis_path.read_bytes()).hexdigest(),
        ),
    )


def capability_symbol_definitions(
    registry: CapabilityRegistry,
) -> tuple[SymbolDefinition, ...]:
    signatures: dict[str, tuple[tuple[ParameterSpec, ...], str]] = {
        "capability.display_correspondence_measurements": (
            (
                ParameterSpec(name="before", type_spec=type_spec("FrameGrid")),
                ParameterSpec(name="after", type_spec=type_spec("FrameGrid")),
                ParameterSpec(name="observation_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="previous_observation_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="scope_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="action_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="announced_count", type_spec=type_spec("Integer")),
                ParameterSpec(name="before_completed", type_spec=type_spec("Integer")),
                ParameterSpec(name="after_completed", type_spec=type_spec("Integer")),
                ParameterSpec(name="direction", type_spec=type_spec("List", type_spec("Integer"))),
                ParameterSpec(name="body_rows", type_spec=type_spec("List", type_spec("Any"))),
                ParameterSpec(name="convention_rows", type_spec=type_spec("List", type_spec("Any"))),
                ParameterSpec(name="contact_rows", type_spec=type_spec("List", type_spec("Any"))),
                ParameterSpec(name="max_components", type_spec=type_spec("Integer")),
                ParameterSpec(name="max_rows", type_spec=type_spec("Integer")),
                ParameterSpec(name="boundary_inventory_only", type_spec=type_spec("Boolean"), required=False),
                ParameterSpec(name="lattice_geometry", type_spec=type_spec("Any"), required=False),
            ),
            "DisplayCorrespondenceMeasurements",
        ),
        "capability.announced_step_measurements": (
            (ParameterSpec(name="data", type_spec=type_spec("Any")),
             ParameterSpec(name="current_frame_ref", type_spec=type_spec("Ref")),
             ParameterSpec(name="alternative_refs", type_spec=type_spec("List", type_spec("Any")))),
            "AnnouncedStepMeasurements",
        ),
        "capability.frame_normalization": (
            (ParameterSpec(name="payload", type_spec=type_spec("Any")),),
            "FrameNormalizationResult",
        ),
        "capability.bearer_appearance_measurement": (
            (
                ParameterSpec(name="bearer_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="valued_pixels", type_spec=type_spec("List", type_spec("Any"))),
                ParameterSpec(name="measurement_basis_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="cell_quantum", type_spec=type_spec("Integer", optional=True)),
                ParameterSpec(name="key_contract", type_spec=type_spec("Any")),
                ParameterSpec(name="structural_descriptor", type_spec=type_spec("Any", optional=True)),
                ParameterSpec(name="structural_source_hash", type_spec=type_spec("Ref", optional=True)),
                ParameterSpec(name="max_bbox_area", type_spec=type_spec("Integer")),
            ),
            "BearerAppearanceMeasurement",
        ),
        "capability.connected_components": (
            (
                ParameterSpec(name="frame", type_spec=type_spec("FrameGrid")),
                ParameterSpec(name="connectivity", type_spec=type_spec("Integer")),
            ),
            "ComponentExtractionResult",
        ),
        "capability.block_grid_scene_decomposition": (
            (
                ParameterSpec(name="frame", type_spec=type_spec("FrameGrid")),
                ParameterSpec(
                    name="action_conditioned_grid_evidence",
                    type_spec=type_spec("Boolean"),
                ),
                ParameterSpec(
                    name="retained_grid_geometry", type_spec=type_spec("Any", optional=True)
                ),
                ParameterSpec(
                    name="retained_grid_separator_consistency_ppm",
                    type_spec=type_spec("Integer"),
                ),
            ),
            "BlockGridSceneDescription",
        ),
        "capability.visual_scene_decomposition": (
            (
                ParameterSpec(name="frame", type_spec=type_spec("FrameGrid")),
                ParameterSpec(
                    name="action_conditioned_grid_evidence",
                    type_spec=type_spec("Boolean"),
                ),
                ParameterSpec(name="retained_grid_geometry", type_spec=type_spec("Any", optional=True)),
                ParameterSpec(name="retained_grid_separator_consistency_ppm", type_spec=type_spec("Integer", optional=True)),
            ),
            "VisualSceneDescription",
        ),
        "capability.known_grid_cell_recount": (
            (
                ParameterSpec(
                    name="request", type_spec=type_spec("KnownGridCellRecountInput")
                ),
            ),
            "KnownGridCellRecountResult",
        ),
        "capability.known_grid_object_assembly": (
            (
                ParameterSpec(
                    name="request",
                    type_spec=type_spec("KnownGridObjectAssemblyInput"),
                ),
            ),
            "KnownGridObjectAssemblyResult",
        ),
        "capability.multi_resolution_view_measurement": (
            (
                ParameterSpec(name="scene", type_spec=type_spec("VisualSceneDescription")),
                ParameterSpec(name="observation_ref", type_spec=type_spec("Ref")),
                ParameterSpec(
                    name="available_action_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="is_initial_observation", type_spec=type_spec("Boolean")
                ),
            ),
            "MultiResolutionViewMeasurements",
        ),
        "capability.frame_difference": (
            (
                ParameterSpec(name="before", type_spec=type_spec("FrameGrid")),
                ParameterSpec(name="after", type_spec=type_spec("FrameGrid")),
                ParameterSpec(
                    name="action_ref", type_spec=type_spec("Ref", optional=True)
                ),
            ),
            "FrameDifferenceResult",
        ),
        "capability.scene_transition_measurement": (
            (
                ParameterSpec(name="before", type_spec=type_spec("FrameGrid")),
                ParameterSpec(
                    name="intermediate_frames",
                    type_spec=type_spec("List", type_spec("FrameGrid")),
                ),
                ParameterSpec(name="after", type_spec=type_spec("FrameGrid")),
                ParameterSpec(name="transition_ref", type_spec=type_spec("Ref")),
                ParameterSpec(
                    name="action_ref", type_spec=type_spec("Ref", optional=True)
                ),
                ParameterSpec(name="action_data", type_spec=type_spec("Any")),
                ParameterSpec(
                    name="candidate_ref", type_spec=type_spec("Ref", optional=True)
                ),
                ParameterSpec(name="context_epoch", type_spec=type_spec("Any")),
                ParameterSpec(
                    name="before_configuration_digest",
                    type_spec=type_spec("Ref", optional=True),
                ),
                ParameterSpec(
                    name="after_configuration_digest",
                    type_spec=type_spec("Ref", optional=True),
                ),
                ParameterSpec(
                    name="before_available_action_set_digest",
                    type_spec=type_spec("Ref", optional=True),
                ),
                ParameterSpec(
                    name="boundary_indicator_quantum_delta_measurement",
                    type_spec=type_spec("Integer"),
                ),
                ParameterSpec(
                    name="established_boundary_indicator_decrement_changed_pixel_count_measurement",
                    type_spec=type_spec("Integer"),
                ),
                ParameterSpec(
                    name="declared_boundary_indicator_decrease_changed_pixel_count_measurement",
                    type_spec=type_spec("Integer"),
                ),
                ParameterSpec(name="before_score", type_spec=type_spec("Integer")),
                ParameterSpec(
                    name="after_score", type_spec=type_spec("Integer", optional=True)
                ),
                ParameterSpec(
                    name="current_effect_signature_known",
                    type_spec=type_spec("Boolean"),
                ),
                ParameterSpec(
                    name="current_effect_signature_known_cycle",
                    type_spec=type_spec("Boolean"),
                ),
                ParameterSpec(
                    name="known_effect_signature_count",
                    type_spec=type_spec("Integer"),
                ),
                ParameterSpec(
                    name="known_cycle_transition_signature_count",
                    type_spec=type_spec("Integer"),
                ),
            ),
            "SceneTransitionMeasurements",
        ),
        "capability.intermediate_scene_change_measurement": (
            (
                ParameterSpec(name="before", type_spec=type_spec("FrameGrid")),
                ParameterSpec(name="after", type_spec=type_spec("FrameGrid")),
                ParameterSpec(
                    name="ignored_regions",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="permitted_regions",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="action_ref", type_spec=type_spec("Ref", optional=True)
                ),
                ParameterSpec(
                    name="periodic_cycle_workflow_facts",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
            ),
            "IntermediateSceneChangeMeasurements",
        ),
        "capability.temporal_component_tracking": (
            (
                ParameterSpec(
                    name="current",
                    type_spec=type_spec("ComponentExtractionResult"),
                ),
                ParameterSpec(name="frame_ref", type_spec=type_spec("Ref")),
                ParameterSpec(
                    name="previous", type_spec=type_spec("Any", optional=True)
                ),
                ParameterSpec(
                    name="action_ref", type_spec=type_spec("Ref", optional=True)
                ),
                ParameterSpec(
                    name="reset_correspondence", type_spec=type_spec("Boolean")
                ),
            ),
            "TemporalTrackingResult",
        ),
        "capability.exact_multicell_peer_relation_measurement": (
            (
                ParameterSpec(
                    name="tracking", type_spec=type_spec("TemporalTrackingResult")
                ),
                ParameterSpec(
                    name="zones", type_spec=type_spec("ComponentExtractionResult")
                ),
                ParameterSpec(
                    name="maximum_relations", type_spec=type_spec("Integer")
                ),
            ),
            "ExactMulticellPeerRelationMeasurements",
        ),
        "capability.controlled_transition_analysis": (
            (
                ParameterSpec(
                    name="before_scene", type_spec=type_spec("VisualSceneDescription")
                ),
                ParameterSpec(
                    name="after_scene", type_spec=type_spec("VisualSceneDescription")
                ),
                ParameterSpec(
                    name="before_tracking", type_spec=type_spec("TemporalTrackingResult")
                ),
                ParameterSpec(
                    name="after_tracking", type_spec=type_spec("TemporalTrackingResult")
                ),
                ParameterSpec(name="candidate_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="action_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="transition_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="context_epoch", type_spec=type_spec("Integer")),
                ParameterSpec(
                    name="available_directional_action_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="interface_action_translation_deltas",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(name="input_aligned_group_connectivity", type_spec=type_spec("Integer"), required=False),
                ParameterSpec(name="input_aligned_group_without_global_enclosure", type_spec=type_spec("Boolean"), required=False),
                ParameterSpec(
                    name="known_actuator_entity_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="canonical_known_actuator_term_facts",
                    type_spec=type_spec("List", type_spec("Any"), optional=True),
                ),
                ParameterSpec(
                    name="canonical_reversible_control_claim_facts",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="canonical_provisional_control_term_facts",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="canonical_action_translation_claim_facts",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="canonical_periodic_transition_claim_facts",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="canonical_relational_lattice_term_facts",
                    type_spec=type_spec("List", type_spec("Any")), required=False,
                ),
                ParameterSpec(
                    name="declared_cellular_mover_pattern_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="pre_action_input_aligned_entity_bboxes",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="post_action_input_aligned_entity_bboxes",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="input_aligned_episode_start_bbox",
                    type_spec=type_spec("Any", optional=True),
                ),
                ParameterSpec(
                    name="prior_context_effect_carried_reflection_axis_observed",
                    type_spec=type_spec("Boolean"),
                ),
                ParameterSpec(
                    name="ordered_point_trajectory", type_spec=type_spec("Any")
                ),
            ),
            "ControlledTransitionAnalysis",
        ),
        "capability.attached_rotation_analysis": (
            (
                ParameterSpec(name="previous", type_spec=type_spec("TemporalTrackingResult")),
                ParameterSpec(name="current", type_spec=type_spec("TemporalTrackingResult")),
                ParameterSpec(name="body_entity_refs", type_spec=type_spec("List", type_spec("Ref"))),
                ParameterSpec(name="part_entity_refs", type_spec=type_spec("List", type_spec("Ref"))),
                ParameterSpec(name="max_part_pairs", type_spec=type_spec("Integer")),
                ParameterSpec(name="max_structure_pixels", type_spec=type_spec("Integer")),
            ),
            "AttachedRotationAnalysis",
        ),
        "capability.goal_completion_analysis": (
            (
                ParameterSpec(name="before_score", type_spec=type_spec("Integer")),
                ParameterSpec(name="after_score", type_spec=type_spec("Integer")),
                ParameterSpec(name="context_epoch", type_spec=type_spec("Integer", optional=True)),
                ParameterSpec(
                    name="goal_ref", type_spec=type_spec("Ref", optional=True)
                ),
                ParameterSpec(
                    name="terminal_access_candidate_ref",
                    type_spec=type_spec("Ref", optional=True),
                ),
                ParameterSpec(
                    name="terminal_objective_contract_ref",
                    type_spec=type_spec("Ref", optional=True),
                ),
                ParameterSpec(
                    name="terminal_objective_measure_ref",
                    type_spec=type_spec("Ref", optional=True),
                ),
                ParameterSpec(
                    name="terminal_objective_comparator",
                    type_spec=type_spec("String", optional=True),
                ),
                ParameterSpec(
                    name="terminal_objective_target_value",
                    type_spec=type_spec("Integer", optional=True),
                ),
                ParameterSpec(
                    name="terminal_objective_predicted_value_after_plan",
                    type_spec=type_spec("Integer", optional=True),
                ),
                ParameterSpec(
                    name="retrospective_coverage_target_cell_count_before_winning_action",
                    type_spec=type_spec("Integer", optional=True),
                ),
                ParameterSpec(
                    name="retrospective_uncovered_target_cell_count_before_winning_action",
                    type_spec=type_spec("Integer", optional=True),
                ),
                ParameterSpec(
                    name="retrospective_stationary_peer_palette_nonboundary_cell_count_before_winning_action",
                    type_spec=type_spec("Integer", optional=True),
                ),
                ParameterSpec(
                    name="retrospective_predicted_stationary_peer_palette_remaining_cell_count_after_exact_coincidence",
                    type_spec=type_spec("Integer", optional=True),
                ),
                ParameterSpec(
                    name="retrospective_exact_stationary_peer_morphology_match",
                    type_spec=type_spec("Boolean"),
                ),
                ParameterSpec(
                    name="retrospective_hot_workflow_ref",
                    type_spec=type_spec("Ref", optional=True),
                ),
                ParameterSpec(
                    name="retrospective_hot_workflow_activity_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="retrospective_hot_workflow_observation_count",
                    type_spec=type_spec("Integer"),
                ),
                ParameterSpec(
                    name="moving_entity_ref",
                    type_spec=type_spec("Ref", optional=True),
                ),
                ParameterSpec(
                    name="fixed_entity_ref",
                    type_spec=type_spec("Ref", optional=True),
                ),
                ParameterSpec(
                    name="alignment_axis",
                    type_spec=type_spec("String", optional=True),
                ),
                ParameterSpec(
                    name="expected_remaining_step_count",
                    type_spec=type_spec("Integer", optional=True),
                ),
                ParameterSpec(name="action_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="candidate_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="transition_ref", type_spec=type_spec("Ref")),
                ParameterSpec(
                    name="distance_is_quantized",
                    type_spec=type_spec("Boolean"),
                ),
                ParameterSpec(
                    name="exact_terminal_relation_step_expected",
                    type_spec=type_spec("Boolean"),
                ),
                ParameterSpec(
                    name="lifecycle_revision_group_ref",
                    type_spec=type_spec("Ref", optional=True),
                ),
                ParameterSpec(
                    name="causally_effective_action_deltas",
                    type_spec=type_spec("List", type_spec("Integer")),
                ),
                ParameterSpec(
                    name="depth_claim_ref",
                    type_spec=type_spec("Ref", optional=True),
                ),
                ParameterSpec(
                    name="declared_minimum_effective_action_count",
                    type_spec=type_spec("Integer"),
                ),
            ),
            "GoalCompletionAnalysis",
        ),
        "capability.declared_goal_gap_measurement": (
            (
                ParameterSpec(
                    name="request", type_spec=type_spec("DeclaredGoalGapInput")
                ),
            ),
            "DeclaredGoalGapResult",
        ),
        "capability.declared_goal_plan_enumeration": (
            (
                ParameterSpec(
                    name="gap_measurements",
                    type_spec=type_spec("DeclaredGoalGapResult"),
                ),
                ParameterSpec(
                    name="plan_request", type_spec=type_spec("GoalPlanRequest")
                ),
            ),
            "DeclaredGoalPlanAlternatives",
        ),
        "capability.repeated_boundary_decrease_analysis": (
            (
                ParameterSpec(name="before", type_spec=type_spec("FrameGrid")),
                ParameterSpec(name="after", type_spec=type_spec("FrameGrid")),
                ParameterSpec(
                    name="before_tracking",
                    type_spec=type_spec("TemporalTrackingResult"),
                ),
                ParameterSpec(
                    name="prior_evidence", type_spec=type_spec("List", type_spec("Any"))
                ),
                ParameterSpec(name="action_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="transition_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="evidence_scope_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="primitive_count_after", type_spec=type_spec("Integer")),
            ),
            "RepeatedBoundaryDecreaseAnalysis",
        ),
        "capability.resource_gauge_quantity_analysis": (
            (
                ParameterSpec(name="before", type_spec=type_spec("FrameGrid")),
                ParameterSpec(name="after", type_spec=type_spec("FrameGrid")),
                ParameterSpec(name="before_tracking", type_spec=type_spec("TemporalTrackingResult")),
                ParameterSpec(name="primitive_count_after", type_spec=type_spec("Integer")),
                ParameterSpec(name="prior_evidence", type_spec=type_spec("List", type_spec("Any"))),
                ParameterSpec(name="current_evidence", type_spec=type_spec("List", type_spec("Any"))),
                ParameterSpec(name="prior_render_records", type_spec=type_spec("List", type_spec("Any"))),
                ParameterSpec(name="rendering_profiles", type_spec=type_spec("List", type_spec("Any"))),
                ParameterSpec(name="alternative_refs", type_spec=type_spec("List", type_spec("Ref"))),
                ParameterSpec(name="role_selection_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="evidence_scope_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="transition_ref", type_spec=type_spec("Ref")),
            ),
            "ResourceGaugeQuantityAnalysis",
        ),
        "capability.external_transport_observation": (
            (
                ParameterSpec(name="before_tracking", type_spec=type_spec("TemporalTrackingResult")),
                ParameterSpec(name="after_tracking", type_spec=type_spec("TemporalTrackingResult")),
                ParameterSpec(name="geometry", type_spec=type_spec("Any")),
                ParameterSpec(name="known_actuator_entity_refs", type_spec=type_spec("List", type_spec("Ref"))),
                ParameterSpec(name="prior_records", type_spec=type_spec("List", type_spec("Any"), optional=True)),
                ParameterSpec(name="transition_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="action_ref", type_spec=type_spec("Ref", optional=True)),
                ParameterSpec(name="pre_action_input_aligned_entity_bboxes", type_spec=type_spec("List", type_spec("Any"), optional=True)),
                ParameterSpec(name="post_action_input_aligned_entity_bboxes", type_spec=type_spec("List", type_spec("Any"), optional=True)),
                ParameterSpec(name="evidence_scope_ref", type_spec=type_spec("Ref", optional=True)),
            ),
            "ExternalTransportObservations",
        ),
        "capability.component_matching": (
            (
                ParameterSpec(
                    name="before", type_spec=type_spec("ComponentExtractionResult")
                ),
                ParameterSpec(
                    name="after", type_spec=type_spec("ComponentExtractionResult")
                ),
                ParameterSpec(
                    name="action_ref", type_spec=type_spec("Ref", optional=True)
                ),
            ),
            "ComponentMatchResult",
        ),
        "capability.palette_canonicalization": (
            (ParameterSpec(name="frame", type_spec=type_spec("FrameGrid")),),
            "PaletteCanonicalizationResult",
        ),
        "capability.scene_rebinding_measurement": (
            (
                ParameterSpec(
                    name="prior_structure",
                    type_spec=type_spec("PaletteCanonicalizationResult"),
                ),
                ParameterSpec(
                    name="current_structure",
                    type_spec=type_spec("PaletteCanonicalizationResult"),
                ),
                ParameterSpec(
                    name="current_tracking",
                    type_spec=type_spec("TemporalTrackingResult"),
                ),
                ParameterSpec(name="current_frame_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="handoff_claim_ref", type_spec=type_spec("Ref")),
                ParameterSpec(name="scope_ref", type_spec=type_spec("Ref")),
                ParameterSpec(
                    name="transported_reference_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
            ),
            "SceneRebindingMeasurements",
        ),
        "capability.uniform_block_reduction": (
            (
                ParameterSpec(name="frame", type_spec=type_spec("FrameGrid")),
                ParameterSpec(name="block_height", type_spec=type_spec("Integer")),
                ParameterSpec(name="block_width", type_spec=type_spec("Integer")),
                ParameterSpec(name="offset_row", type_spec=type_spec("Integer")),
                ParameterSpec(name="offset_col", type_spec=type_spec("Integer")),
            ),
            "UniformBlockReductionResult",
        ),
        "capability.grid_partition_enumeration": (
            (
                ParameterSpec(name="frame", type_spec=type_spec("FrameGrid")),
                ParameterSpec(name="minimum_grid_rows", type_spec=type_spec("Integer")),
                ParameterSpec(
                    name="minimum_grid_columns", type_spec=type_spec("Integer")
                ),
            ),
            "GridPartitionResult",
        ),
        "capability.periodic_cell_grid_enumeration": (
            (
                ParameterSpec(name="frame", type_spec=type_spec("FrameGrid")),
                ParameterSpec(name="maximum_pitch", type_spec=type_spec("Integer")),
                ParameterSpec(
                    name="maximum_candidates", type_spec=type_spec("Integer")
                ),
            ),
            "PeriodicCellGridResult",
        ),
        "capability.component_geometry": (
            (
                ParameterSpec(
                    name="components", type_spec=type_spec("ComponentExtractionResult")
                ),
            ),
            "GeometryResult",
        ),
        "capability.orthogonal_perimeter_assembly_measurement": (
            (
                ParameterSpec(
                    name="components",
                    type_spec=type_spec("ComponentExtractionResult"),
                ),
            ),
            "OrthogonalPerimeterAssemblyMeasurements",
        ),
        "capability.derived_geometry_state_measurement": (
            (ParameterSpec(name="request", type_spec=type_spec("Any")),),
            "DerivedGeometryStateMeasurements",
        ),
        "capability.viewport_support_terrain_measurement": (
            (ParameterSpec(name="request", type_spec=type_spec("Any")),),
            "ViewportSupportTerrainMeasurements",
        ),
        "capability.interaction_topology_measurement": (
            (ParameterSpec(name="request", type_spec=type_spec("Any")),),
            "InteractionTopologyMeasurements",
        ),
        "capability.elimination_viability_measurement": (
            (ParameterSpec(name="request", type_spec=type_spec("Any")),),
            "EliminationViabilityMeasurements",
        ),
        "capability.relational_elimination_graph_measurement": (
            tuple(ParameterSpec(name=name, type_spec=type_spec("Any")) for name in (
                "cell_positions", "occupied_positions", "half_translation_quantum", "operator_ref",
                "cell_palette_rows", "free_palette_values",
                "terminal_condition_ref", "terminal_occupancy_count", "measurement_context_ref",
                "premise_claim_refs", "frame_ref", "analysis_scope")),
            "RelationalEliminationGraphMeasurements",
        ),
        "capability.abductive_version_space_measurement": (
            (ParameterSpec(name="request", type_spec=type_spec("Any")),),
            "AbductiveVersionSpaceMeasurements",
        ),
        "capability.temporal_concurrency_measurement": (
            (ParameterSpec(name="request", type_spec=type_spec("Any")),),
            "TemporalConcurrencyMeasurements",
        ),
        "capability.temporal_replay_sequence_measurement": (
            (ParameterSpec(name="request", type_spec=type_spec("Any")),),
            "TemporalReplaySequenceMeasurements",
        ),
        "capability.temporal_joint_route_measurement": (
            (ParameterSpec(name="request", type_spec=type_spec("Any")),),
            "TemporalJointRouteMeasurements",
        ),
        "capability.transition_evidence_compaction": (
            (ParameterSpec(name="request", type_spec=type_spec("Any")),),
            "TransitionEvidenceLedger",
        ),
        "capability.transition_retrodiction_measurement": (
            (ParameterSpec(name="request", type_spec=type_spec("Any")),),
            "TransitionRetrodictionMeasurements",
        ),
        "capability.transition_phenomenon_inventory": (
            (ParameterSpec(name="request", type_spec=type_spec("Any")),),
            "TransitionPhenomenonInventoryMeasurements",
        ),
        "capability.interaction_probe_enumeration": (
            (
                ParameterSpec(name="frame", type_spec=type_spec("FrameGrid")),
                ParameterSpec(
                    name="components",
                    type_spec=type_spec("ComponentExtractionResult"),
                ),
                ParameterSpec(
                    name="zones",
                    type_spec=type_spec("ComponentExtractionResult"),
                ),
                ParameterSpec(
                    name="sprites", type_spec=type_spec("List", type_spec("Any"))
                ),
                ParameterSpec(name="periodic_cell_grids", type_spec=type_spec("Any")),
                ParameterSpec(
                    name="available_action_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="current_context_no_effect_candidate_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="current_configuration_executed_candidate_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="tested_candidate_refs_without_resolution_plan",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="known_scene_effect_component_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="reasoning_strategy_ref",
                    type_spec=type_spec("Ref", optional=True),
                ),
                ParameterSpec(
                    name="known_effectful_action_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="observed_ordered_temporal_transition_action_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="other_context_no_effect_action_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="known_no_effect_action_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="reverse_of_last_action_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="reverse_of_last_directional_action_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="last_action_ref", type_spec=type_spec("Ref", optional=True)
                ),
                ParameterSpec(
                    name="previous_action_ref",
                    type_spec=type_spec("Ref", optional=True),
                ),
                ParameterSpec(
                    name="last_point_position",
                    type_spec=type_spec("Any", optional=True),
                ),
                ParameterSpec(
                    name="point_pre_action_enclosure_residual",
                    type_spec=type_spec("Any", optional=True),
                ),
                ParameterSpec(
                    name="point_pre_action_translated_peer_residual",
                    type_spec=type_spec("Any", optional=True),
                ),
                ParameterSpec(
                    name="nonlocal_quantized_translation_observed",
                    type_spec=type_spec("Boolean"),
                ),
                ParameterSpec(
                    name="transient_change_bboxes",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="action_conditioned_appeared_bboxes",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="relational_cell_positions",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="relational_occupied_positions",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="action_execution_counts",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="current_spatial_configuration_hash",
                    type_spec=type_spec("Ref", optional=True),
                ),
                ParameterSpec(
                    name="visited_spatial_configuration_count",
                    type_spec=type_spec("Integer"),
                ),
                ParameterSpec(
                    name="actions_that_produced_novel_configuration",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="known_action_translation_deltas",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="observed_controlled_translation_count",
                    type_spec=type_spec("Integer"),
                ),
                ParameterSpec(
                    name="observed_translated_entity_bboxes",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="translated_entity_bboxes",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="input_aligned_entity_bboxes",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="visited_input_aligned_entity_bboxes",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="current_input_aligned_underlay_values",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="current_transition_change_bboxes",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="current_transition_intermediate_frame_count",
                    type_spec=type_spec("Integer"),
                ),
                ParameterSpec(
                    name="interface_action_translation_deltas",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="canonical_pose_term_facts",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="canonical_correspondence_term_facts",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="canonical_reversible_control_claim_facts",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="canonical_closed_support_term_facts",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="canonical_enclosed_locus_term_facts",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="canonical_closed_support_permutation_claim_facts",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="route_generation_policy",
                    type_spec=type_spec("Any"),
                ),
                ParameterSpec(
                    name="known_viewport_grid_shift_deltas",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="residual_axis_coupling_row",
                    type_spec=type_spec("Integer"),
                ),
                ParameterSpec(
                    name="residual_axis_coupling_column",
                    type_spec=type_spec("Integer"),
                ),
                ParameterSpec(
                    name="directional_frontier_saturated",
                    type_spec=type_spec("Boolean"),
                ),
                ParameterSpec(
                    name="cellular_mover_pattern_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="recent_moved_cell_pattern_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="cellular_actuator_pattern_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="cellular_traversable_pattern_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="open_contact_pattern_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="viewport_contact_projections",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
            ),
            "InteractionProbeAgenda",
        ),
        "capability.homologous_repair_analysis": (
            (
                ParameterSpec(
                    name="scene", type_spec=type_spec("VisualSceneDescription")
                ),
            ),
            "HomologousRepairAnalysis",
        ),
        "capability.local_constraint_composition_analysis": (
            (
                ParameterSpec(
                    name="scene", type_spec=type_spec("VisualSceneDescription")
                ),
                ParameterSpec(
                    name="transferred_mapping_pairs",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
            ),
            "LocalConstraintCompositionAnalysis",
        ),
        "capability.local_constraint_repair_action_enumeration": (
            (
                ParameterSpec(
                    name="analysis",
                    type_spec=type_spec("LocalConstraintCompositionAnalysis"),
                ),
                ParameterSpec(
                    name="selected_composition_ref", type_spec=type_spec("Ref")
                ),
                ParameterSpec(
                    name="available_action_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="verified_repair_count", type_spec=type_spec("Integer")
                ),
            ),
            "InteractionProbeAgenda",
        ),
        "capability.local_constraint_repair_reconciliation": (
            (
                ParameterSpec(name="before", type_spec=type_spec("FrameGrid")),
                ParameterSpec(name="after", type_spec=type_spec("FrameGrid")),
                ParameterSpec(name="selected_plan_facts", type_spec=type_spec("Any")),
                ParameterSpec(name="before_score", type_spec=type_spec("Integer")),
                ParameterSpec(name="after_score", type_spec=type_spec("Integer")),
                ParameterSpec(name="transition_ref", type_spec=type_spec("Ref")),
            ),
            "HomologousRepairReconciliationAnalysis",
        ),
        "capability.homologous_repair_action_enumeration": (
            (
                ParameterSpec(
                    name="analysis", type_spec=type_spec("HomologousRepairAnalysis")
                ),
                ParameterSpec(name="selected_mapping_ref", type_spec=type_spec("Ref")),
                ParameterSpec(
                    name="available_action_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(name="verified_repair_count", type_spec=type_spec("Integer")),
            ),
            "InteractionProbeAgenda",
        ),
        "capability.homologous_repair_reconciliation": (
            (
                ParameterSpec(name="before", type_spec=type_spec("FrameGrid")),
                ParameterSpec(name="after", type_spec=type_spec("FrameGrid")),
                ParameterSpec(name="selected_plan_facts", type_spec=type_spec("Any")),
                ParameterSpec(name="before_score", type_spec=type_spec("Integer")),
                ParameterSpec(name="after_score", type_spec=type_spec("Integer")),
                ParameterSpec(name="transition_ref", type_spec=type_spec("Ref")),
            ),
            "HomologousRepairReconciliationAnalysis",
        ),
        "capability.verified_alignment_action_enumeration": (
            (
                ParameterSpec(name="canonical_peer_method_facts", type_spec=type_spec("List", type_spec("Any")), required=False),
                ParameterSpec(name="canonical_terminal_inventory_facts", type_spec=type_spec("List", type_spec("Any")), required=False),
                ParameterSpec(name="canonical_provisional_control_term_facts", type_spec=type_spec("List", type_spec("Any")), required=False),
                ParameterSpec(
                    name="measurement_scope_ref",
                    type_spec=type_spec("Ref"),
                    required=False,
                ),
                ParameterSpec(
                    name="scene", type_spec=type_spec("VisualSceneDescription")
                ),
                ParameterSpec(
                    name="tracking", type_spec=type_spec("TemporalTrackingResult")
                ),
                ParameterSpec(
                    name="mechanism_evidence",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="calibrated_affine_evidence",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="effect_carried_reflection_axis_evidence",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="terminally_exhausted_action_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="calibrated_nary_affine_evidence",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="calibrated_local_point_evidence",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="available_action_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(name="periodic_cell_grids", type_spec=type_spec("Any")),
                ParameterSpec(
                    name="cellular_mover_pattern_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="recent_moved_cell_pattern_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="cellular_actuator_pattern_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="effectful_point_cell_pattern_pairs",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="cellular_traversable_pattern_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="current_transition_change_bboxes",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="cellular_entity_traversable_pattern_refs",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="cellular_entity_footprint_offsets",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="open_contact_pattern_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="established_hostile_contact_pattern_faces",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="observed_fatal_initial_action_facts",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="known_open_cell_coordinates",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="newly_reachable_cell_coordinates",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="committed_plan_facts", type_spec=type_spec("Any")
                ),
                ParameterSpec(
                    name="committed_plan_cursor_present",
                    type_spec=type_spec("Boolean"),
                ),
                ParameterSpec(
                    name="current_object_terminating_route_measurement_present",
                    type_spec=type_spec("Boolean"),
                ),
                ParameterSpec(
                    name="current_object_terminating_route_first_action_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="prior_context_orthogonal_reflection_direct_body_values",
                    type_spec=type_spec("List", type_spec("Integer")),
                ),
                ParameterSpec(
                    name="current_context_no_effect_candidate_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="current_configuration_executed_candidate_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="action_execution_counts",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="action_first_execution_ordinals",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="resource_only_effect_action_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="cyclic_pose_only_effect_action_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="known_action_translation_deltas",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="interface_action_translation_deltas",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="known_viewport_grid_shift_deltas",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="residual_axis_coupling_row",
                    type_spec=type_spec("Integer"),
                ),
                ParameterSpec(
                    name="residual_axis_coupling_column",
                    type_spec=type_spec("Integer"),
                ),
                ParameterSpec(
                    name="action_conditioned_marker_axis_overlap_reconstructed",
                    type_spec=type_spec("Boolean"),
                ),
                ParameterSpec(
                    name="marker_axis_residual_measurement_present",
                    type_spec=type_spec("Boolean"),
                ),
                ParameterSpec(
                    name="marker_axis_residual_transport_present",
                    type_spec=type_spec("Boolean"),
                ),
                ParameterSpec(
                    name="marker_axis_residual_row_signed",
                    type_spec=type_spec("Integer"),
                ),
                ParameterSpec(
                    name="marker_axis_residual_col_signed",
                    type_spec=type_spec("Integer"),
                ),
                ParameterSpec(
                    name="dynamic_goal_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="dynamic_goal_facts", type_spec=type_spec("Any")
                ),
                ParameterSpec(
                    name="configuration_revisit_action_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="continuation_policy", type_spec=type_spec("Any")
                ),
                ParameterSpec(
                    name="route_generation_policy", type_spec=type_spec("Any")
                ),
                ParameterSpec(
                    name="current_scope_attempted_route_suffixes",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
            ),
            "InteractionProbeAgenda",
        ),
        "capability.transferred_quantized_plan_enumeration": (
            (
                ParameterSpec(
                    name="scene", type_spec=type_spec("VisualSceneDescription")
                ),
                ParameterSpec(
                    name="tracking", type_spec=type_spec("TemporalTrackingResult")
                ),
                ParameterSpec(
                    name="mechanism_evidence",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="committed_plan_facts", type_spec=type_spec("Any")
                ),
                ParameterSpec(
                    name="continuation_policy", type_spec=type_spec("Any")
                ),
                ParameterSpec(
                    name="route_generation_policy", type_spec=type_spec("Any")
                ),
                ParameterSpec(
                    name="current_context_no_effect_candidate_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="current_scope_attempted_route_suffixes",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="available_action_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
            ),
            "InteractionProbeAgenda",
        ),
        "capability.route_compatible_order_analysis": (
            (
                ParameterSpec(
                    name="scene", type_spec=type_spec("VisualSceneDescription")
                ),
                ParameterSpec(
                    name="tracking", type_spec=type_spec("TemporalTrackingResult")
                ),
                ParameterSpec(
                    name="mechanism_evidence",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="available_action_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="committed_plan_facts", type_spec=type_spec("Any")
                ),
                ParameterSpec(
                    name="continuation_policy", type_spec=type_spec("Any")
                ),
                ParameterSpec(
                    name="route_generation_policy", type_spec=type_spec("Any")
                ),
                ParameterSpec(
                    name="current_context_no_effect_candidate_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="current_scope_attempted_route_suffixes",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
            ),
            "InteractionProbeAgenda",
        ),
        "capability.route_compatible_graph_plan_enumeration": (
            (
                ParameterSpec(
                    name="scene", type_spec=type_spec("VisualSceneDescription")
                ),
                ParameterSpec(
                    name="tracking", type_spec=type_spec("TemporalTrackingResult")
                ),
                ParameterSpec(
                    name="mechanism_evidence",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="available_action_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="committed_plan_facts", type_spec=type_spec("Any")
                ),
                ParameterSpec(
                    name="continuation_policy", type_spec=type_spec("Any")
                ),
                ParameterSpec(
                    name="route_generation_policy", type_spec=type_spec("Any")
                ),
                ParameterSpec(
                    name="current_context_no_effect_candidate_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="current_scope_attempted_route_suffixes",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="route_order_analysis",
                    type_spec=type_spec("InteractionProbeAgenda"),
                ),
                ParameterSpec(
                    name="declared_route_order_ref", type_spec=type_spec("Ref")
                ),
            ),
            "InteractionProbeAgenda",
        ),
        "capability.committed_verified_alignment_cursor_advance": (
            (
                ParameterSpec(name="canonical_peer_method_facts", type_spec=type_spec("List", type_spec("Any")), required=False),
                ParameterSpec(name="canonical_terminal_inventory_facts", type_spec=type_spec("List", type_spec("Any")), required=False),
                ParameterSpec(name="canonical_provisional_control_term_facts", type_spec=type_spec("List", type_spec("Any")), required=False),
                ParameterSpec(
                    name="tracking", type_spec=type_spec("TemporalTrackingResult")
                ),
                ParameterSpec(
                    name="committed_plan_facts", type_spec=type_spec("Any")
                ),
                ParameterSpec(
                    name="available_action_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="current_context_no_effect_candidate_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="current_configuration_executed_candidate_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="action_execution_counts",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="resource_only_effect_action_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="cyclic_pose_only_effect_action_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
                ParameterSpec(
                    name="known_action_translation_deltas",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="interface_action_translation_deltas",
                    type_spec=type_spec("List", type_spec("Any")),
                ),
                ParameterSpec(
                    name="residual_axis_coupling_row", type_spec=type_spec("Integer")
                ),
                ParameterSpec(
                    name="residual_axis_coupling_column",
                    type_spec=type_spec("Integer"),
                ),
                ParameterSpec(
                    name="route_generation_policy", type_spec=type_spec("Any")
                ),
            ),
            "InteractionProbeAgenda",
        ),
        "capability.committed_quantized_plan_advance": (
            (
                ParameterSpec(
                    name="committed_plan_facts", type_spec=type_spec("Any")
                ),
                ParameterSpec(
                    name="continuation_policy", type_spec=type_spec("Any")
                ),
                ParameterSpec(
                    name="available_action_refs",
                    type_spec=type_spec("List", type_spec("Ref")),
                ),
            ),
            "InteractionProbeAgenda",
        ),
        "capability.quantized_plan_reconciliation": (
            (
                ParameterSpec(
                    name="after_scene",
                    type_spec=type_spec("Any", optional=True),
                ),
                ParameterSpec(
                    name="after_tracking",
                    type_spec=type_spec("Any", optional=True),
                ),
                ParameterSpec(
                    name="selected_plan_facts", type_spec=type_spec("Any")
                ),
                ParameterSpec(name="before_score", type_spec=type_spec("Integer")),
                ParameterSpec(name="after_score", type_spec=type_spec("Integer")),
                ParameterSpec(name="transition_ref", type_spec=type_spec("Ref")),
            ),
            "QuantizedPlanReconciliationAnalysis",
        ),
        "capability.multi_identity_opposed_routes": (
            (ParameterSpec(name="request", type_spec=type_spec("Any")),),
            "MultiIdentityOpposedRoutesAnalysis",
        ),
        "capability.finite_grid_reachability": (
            (ParameterSpec(name="request", type_spec=type_spec("Any")),),
            "ReachabilityResult",
        ),
        "capability.relation_graph_analysis": (
            (ParameterSpec(name="request", type_spec=type_spec("Any")),),
            "RelationGraphResult",
        ),
        "capability.quantity_simulation": (
            (
                ParameterSpec(
                    name="request", type_spec=type_spec("QuantitySimulationInput")
                ),
            ),
            "QuantitySimulationResult",
        ),
        "capability.dimension_transition": (
            (ParameterSpec(name="request", type_spec=type_spec("Any")),),
            "DimensionTransitionResult",
        ),
        "capability.rational_anchor_dependence_measurement": (
            (
                ParameterSpec(
                    name="request",
                    type_spec=type_spec("RationalAnchorDependenceInput"),
                ),
            ),
            "RationalAnchorDependenceMeasurements",
        ),
        "capability.inverse_mean_anchor_domain_enumeration": (
            (
                ParameterSpec(
                    name="request",
                    type_spec=type_spec("InverseMeanAnchorDomainInput"),
                ),
            ),
            "InverseMeanAnchorDomainMeasurements",
        ),
        "capability.inverse_anchor_typed_clearance_measurement": (
            (
                ParameterSpec(
                    name="request",
                    type_spec=type_spec("InverseAnchorTypedClearanceInput"),
                ),
            ),
            "InverseAnchorTypedClearanceMeasurements",
        ),
        "capability.inverse_anchor_successor_measurement": (
            (
                ParameterSpec(
                    name="request",
                    type_spec=type_spec("InverseAnchorSuccessorInput"),
                ),
            ),
            "InverseAnchorSuccessorMeasurements",
        ),
        "capability.inverse_anchor_waypoint_candidate_measurement": (
            (
                ParameterSpec(
                    name="request",
                    type_spec=type_spec("InverseAnchorWaypointCandidatesInput"),
                ),
            ),
            "InverseAnchorWaypointCandidatesMeasurements",
        ),
        "capability.articulated_kinematics_measurement": (
            (
                ParameterSpec(
                    name="request",
                    type_spec=type_spec("ArticulatedKinematicsInput"),
                ),
            ),
            "ArticulatedKinematicsMeasurements",
        ),
        "capability.articulated_shared_snapshot_measurement": (
            (
                ParameterSpec(
                    name="request",
                    type_spec=type_spec("ArticulatedSharedSnapshotInput"),
                ),
            ),
            "ArticulatedSharedSnapshotMeasurements",
        ),
        "capability.articulated_successor_measurement": (
            (
                ParameterSpec(
                    name="request",
                    type_spec=type_spec("ArticulatedSuccessorInput"),
                ),
            ),
            "ArticulatedSuccessorMeasurements",
        ),
        "capability.typed_assignment_domains_measurement": (
            (
                ParameterSpec(
                    name="request",
                    type_spec=type_spec("TypedAssignmentDomainsInput"),
                ),
            ),
            "TypedAssignmentDomainsMeasurements",
        ),
        "capability.typed_edit_successor_measurement": (
            (
                ParameterSpec(
                    name="request",
                    type_spec=type_spec("TypedEditSuccessorInput"),
                ),
            ),
            "TypedEditSuccessorMeasurements",
        ),
        "capability.typed_interpreter_measurement": (
            (
                ParameterSpec(
                    name="request",
                    type_spec=type_spec("TypedInterpreterInput"),
                ),
            ),
            "TypedInterpreterMeasurements",
        ),
        "capability.recipe_pattern_difference_measurement": ((ParameterSpec(name="request", type_spec=type_spec("RecipePatternDifferenceInput")),), "RecipePatternDifferenceMeasurements"),
        "capability.recipe_successor_measurement": ((ParameterSpec(name="request", type_spec=type_spec("RecipeSuccessorInput")),), "RecipeSuccessorMeasurements"),
        "capability.recipe_resize_successor_measurement": ((ParameterSpec(name="request", type_spec=type_spec("RecipeResizeSuccessorInput")),), "RecipeResizeSuccessorMeasurements"),
        "capability.recipe_resource_order_measurement": ((ParameterSpec(name="request", type_spec=type_spec("RecipeResourceOrderInput")),), "RecipeResourceOrderMeasurements"),
        "capability.piercing_intersection_measurement": ((ParameterSpec(name="request", type_spec=type_spec("PiercingIntersectionInput")),), "PiercingIntersectionMeasurements"),
        "capability.piercing_push_successor_measurement": ((ParameterSpec(name="request", type_spec=type_spec("PiercingPushSuccessorInput")),), "PiercingPushSuccessorMeasurements"),
        "capability.piercing_pierce_successor_measurement": ((ParameterSpec(name="request", type_spec=type_spec("PiercingPierceSuccessorInput")),), "PiercingPierceSuccessorMeasurements"),
        "capability.piercing_withdraw_successor_measurement": ((ParameterSpec(name="request", type_spec=type_spec("PiercingWithdrawSuccessorInput")),), "PiercingWithdrawSuccessorMeasurements"),
        "capability.piercing_clearance_measurement": ((ParameterSpec(name="request", type_spec=type_spec("PiercingClearanceInput")),), "PiercingClearanceMeasurements"),
        "capability.ordered_relief_transition_measurement": (
            (
                ParameterSpec(
                    name="request",
                    type_spec=type_spec("OrderedReliefTransitionInput"),
                ),
            ),
            "OrderedReliefTransitionMeasurements",
        ),
        "capability.ordered_relief_write_successor": (
            (
                ParameterSpec(
                    name="request",
                    type_spec=type_spec("OrderedReliefWriteInput"),
                ),
            ),
            "OrderedReliefWriteMeasurements",
        ),
        "capability.ordered_relief_pose_measurement": (
            (
                ParameterSpec(
                    name="request",
                    type_spec=type_spec("OrderedReliefPoseInput"),
                ),
            ),
            "OrderedReliefPoseMeasurements",
        ),
        "capability.reconfigurable_support_packet_measurement": (
            (
                ParameterSpec(
                    name="request",
                    type_spec=type_spec("ReconfigurableSupportPacketInput"),
                ),
            ),
            "ReconfigurableSupportPacketMeasurements",
        ),
        "capability.reconfigurable_rigid_geometry_measurement": (
            (
                ParameterSpec(
                    name="request",
                    type_spec=type_spec("ReconfigurableRigidGeometryInput"),
                ),
            ),
            "ReconfigurableRigidGeometryMeasurements",
        ),
        "capability.reconfigurable_connection_domain_enumeration": (
            (
                ParameterSpec(
                    name="request",
                    type_spec=type_spec("ReconfigurableConnectionDomainInput"),
                ),
            ),
            "ReconfigurableConnectionDomainMeasurements",
        ),
        "capability.directed_flow_collider_union_measurement": (
            (
                ParameterSpec(
                    name="request",
                    type_spec=type_spec("DirectedFlowColliderUnionInput"),
                ),
            ),
            "DirectedFlowColliderUnionMeasurements",
        ),
        "capability.directed_flow_joint_tick_measurement": (
            (
                ParameterSpec(
                    name="request",
                    type_spec=type_spec("DirectedFlowJointTickInput"),
                ),
            ),
            "DirectedFlowJointTickMeasurements",
        ),
        "capability.directed_flow_recovery_measurement": (
            (
                ParameterSpec(
                    name="request",
                    type_spec=type_spec("DirectedFlowRecoveryInput"),
                ),
            ),
            "DirectedFlowRecoveryMeasurements",
        ),
        "capability.local_field_selective_domain_measurement": (
            (
                ParameterSpec(
                    name="request",
                    type_spec=type_spec("LocalFieldSelectiveDomainInput"),
                ),
            ),
            "LocalFieldSelectiveDomainMeasurements",
        ),
        "capability.array_adjacent_region_measurement": (
            (ParameterSpec(name="request", type_spec=type_spec("ArrayRegionMeasurementRequest")),),
            "ArrayRegionMeasurements",
        ),
        "capability.consumptive_dependency_measurement": (
            (ParameterSpec(name="request", type_spec=type_spec("ConsumptiveDependencyRequest")),),
            "ConsumptiveDependencyMeasurements",
        ),
        "capability.pair_point_geometry_measurement": (
            (ParameterSpec(name="request", type_spec=type_spec("PairPointGeometryRequest")),),
            "PairPointGeometryMeasurements",
        ),
        "capability.stock_change_measurement": (
            (ParameterSpec(name="request", type_spec=type_spec("StockChangeRequest")),),
            "StockChangeMeasurements",
        ),
        "capability.filled_contour_correspondence": (
            (ParameterSpec(name="request", type_spec=type_spec("FilledContourRequest")),),
            "FilledContourMeasurements",
        ),
        "capability.point_progress_measurement": (
            (ParameterSpec(name="request", type_spec=type_spec("PointProgressRequest")),),
            "PointProgressMeasurements",
        ),
        "capability.local_field_autonomous_successor_measurement": (
            (
                ParameterSpec(
                    name="request",
                    type_spec=type_spec("LocalFieldAutonomousSuccessorInput"),
                ),
            ),
            "LocalFieldAutonomousSuccessorMeasurements",
        ),
        "capability.local_field_acceptance_window_measurement": (
            (
                ParameterSpec(
                    name="request",
                    type_spec=type_spec("LocalFieldAcceptanceWindowInput"),
                ),
            ),
            "LocalFieldAcceptanceWindowMeasurements",
        ),
        "capability.local_field_transaction_measurement": (
            (
                ParameterSpec(
                    name="request",
                    type_spec=type_spec("LocalFieldTransactionInput"),
                ),
            ),
            "LocalFieldTransactionMeasurements",
        ),
    }
    definitions: list[SymbolDefinition] = []
    for entry in registry.entries:
        parameters, output_name = signatures[entry.definition.id]
        definitions.append(
            SymbolDefinition(
                id=entry.definition.id,
                kind=SymbolKind.CAPABILITY,
                parameters=parameters,
                output_type=type_spec(output_name),
                effect_class=entry.definition.effect_class,
                source_unit=entry.definition.source_unit,
                source_hash=entry.definition.source_hash,
            )
        )
    return tuple(definitions)


def production_environment(
    *,
    capabilities: CapabilityRegistry | None = None,
    drm: DrmRegistry | None = None,
) -> CompilerEnvironment:
    capability_registry = capabilities or build_default_capability_registry()
    drm_registry = drm or build_production_drm_registry(load_production_drm())
    base = generic_fixture_environment()
    control_symbols = {
        item.id: item for item in production_control_symbol_definitions()
    }
    base_symbol_ids = {symbol.id for symbol in base.symbols}
    environment = base.model_copy(
        update={
            "types": (*base.types, *_CAPABILITY_TYPES),
            "symbols": (
                *(control_symbols.get(symbol.id, symbol) for symbol in base.symbols),
                *(
                    symbol
                    for symbol in (control_symbols[__yf_order_key] for __yf_order_key in sorted(control_symbols))
                    if symbol.id not in base_symbol_ids
                ),
                *capability_symbol_definitions(capability_registry),
            ),
        }
    )
    return drm_registry.extend_environment(environment)


def production_src_paths() -> tuple[Path, ...]:
    root = Path(__file__).parent
    return (
        root / "arc3" / "initialize_relation_graph_dimensions.src",
        root / "arc3" / "assimilate_frame.src",
        root / "arc3" / "perceptual_memory.src",
        root / "arc3" / "activate_dynamic_goal_frontier.src",
        root / "arc3" / "choose_information_probe.src",
        root / "arc3" / "choose_repeated_pair_program_action.src",
        root / "arc3" / "choose_symbol_tile_action.src",
        root / "arc3" / "choose_homologous_repair_action.src",
        root / "arc3" / "continue_homologous_repair.src",
        root / "arc3" / "choose_composed_constraint_repair_action.src",
        root / "arc3" / "continue_composed_constraint_repair.src",
        root / "arc3" / "reconcile_composed_constraint_repair.src",
        root / "arc3" / "record_local_constraint_conflict.src",
        root / "arc3" / "classify_interaction_effect.src",
        root / "arc3" / "revise_minimum_effective_depth.src",
        root / "arc3" / "verify_intermediate_continue.src",
        root / "arc3" / "verify_reconciled_state_reuse.src",
        root / "arc3" / "track_scene.src",
        root / "arc3" / "assess_exact_multicell_peer_goals.src",
        root / "arc3" / "interpret_controlled_transition.src",
        root / "arc3" / "interpret_display_correspondence.src",
        root / "arc3" / "construct_transition_meaning.src",
        root / "arc3" / "choose_verified_alignment_action.src",
        root / "arc3" / "construct_transferred_quantized_plan.src",
        root / "arc3" / "construct_reference_guided_regional_repair.src",
        root / "arc3" / "advance_committed_transferred_quantized_plan.src",
        root / "arc3" / "continue_transferred_quantized_plan.src",
        root / "arc3" / "prepare_declared_action_series.src",
        root / "arc3" / "continue_declared_action.src",
        root / "arc3" / "reconcile_transferred_quantized_plan.src",
        root / "arc3" / "reconcile_homologous_repair.src",
        root / "arc3" / "record_verified_goal_completion.src",
        root / "arc3" / "recognize_action_resource_indicator.src",
        root / "arc3" / "resolve_role_by_probe.src",
        root / "arc3" / "assess_mechanism_commitment_readiness.src",
        root / "arc3" / "test_mechanism.src",
        root / "arc3" / "plan_goal.src",
        root / "arc3" / "execute_verified_plan.src",
        root / "arc3" / "rebind_transformed_scene.src",
        root / "arc3" / "evaluate_constrained_plan.src",
        root / "arc3" / "extend_insufficient_ontology.src",
        root / "arc3" / "verify_independent_transition.src",
        root / "arc3" / "assess_derived_geometry_state.src",
        root / "arc3" / "measure_inverse_anchor_signature.src",
        root / "arc3" / "assess_inverse_anchor_signature_extensions.src",
        root / "arc3" / "assess_articulated_channel_kinematics.src",
        root / "arc3" / "assess_typed_recursive_program.src",
        root / "arc3" / "assess_recipe_operator_compilation.src",
        root / "arc3" / "assess_piercing_rooted_order.src",
        root / "arc3" / "assess_ordered_relief_writing.src",
        root / "arc3" / "assess_reconfigurable_support.src",
        root / "arc3" / "assess_directed_accretive_flow_extensions.src",
        root / "arc3" / "assess_local_field_type_production_extensions.src",
        root / "arc3" / "assess_mechanism_application_question.src",
        root / "arc3" / "assess_consumptive_type_dependency.src",
        root / "arc3" / "assess_orthogonal_perimeter_assembly.src",
        root / "arc3" / "assess_viewport_support_terrain.src",
        root / "arc3" / "screen_mobile_support_transform.src",
        root / "arc3" / "assess_interaction_topology.src",
        root / "arc3" / "assess_elimination_viability.src",
        root / "arc3" / "assess_temporal_concurrency.src",
        root / "arc3" / "measure_temporal_observations.src",
        root / "arc3" / "assess_temporal_replay_sequence.src",
        root / "arc3" / "form_temporal_concurrency_plan.src",
        root / "arc3" / "assess_transition_retrodiction.src",
        root / "arc3" / "assess_abductive_explanations.src",
        root / "arc3" / "select_hot_symbolic_program.src",
        root / "arc3" / "synthesize_dynamic_workflow.src",
        root / "arc3" / "conditional_reasoning_tree.src",
        root / "arc3" / "reasoning_tree_v11_todo_projection.src",
        root / "arc3" / "select_reasoning_strategy.src",
        root / "arc3" / "project_priority_obligation_frontier.src",
        root / "arc3" / "project_functional_usefulness_frontier.src",
        root / "arc3" / "classify_action_contribution.src",
        root / "arc3" / "select_post_action_continuation.src",
        root / "arc3" / "execute_staged_reconciled_strategy.src",
        root / "core" / "dependency_propagation.src",
        root / "core" / "operational_wisdom.src",
        root / "core" / "construct_applied_principle.src",
        root / "core" / "instantiate_general_principle.src",
        root / "core" / "epistemic_principle_lifecycle.src",
    )


def _compile_production_workflows(
    environment: CompilerEnvironment,
) -> tuple[CompiledModule, ...]:
    modules = tuple(
        compile_source(
            path.read_text(encoding="utf-8"),
            source_name=str(path),
            environment=environment,
        )
        for path in production_src_paths()
    )
    contracts, _source_hash = load_production_workflow_contracts()
    validate_production_workflow_contracts(modules, contracts)
    return modules


@lru_cache(maxsize=4)
def _cached_default_production_workflows(
    source_tree_digest: str,
) -> tuple[CompiledModule, ...]:
    # The digest is the cache key. A source edit must miss this cache and rerun
    # the complete determinism audit before any newly compiled graph is usable.
    del source_tree_digest
    assert_v5_determinism()
    return _compile_production_workflows(production_environment())


def compile_production_workflows(
    *,
    environment: CompilerEnvironment | None = None,
) -> tuple[CompiledModule, ...]:
    if environment is None:
        # The full source digest invalidates the compiled graph on every edit.
        # A cache miss audits that exact tree before compiling it; re-auditing
        # unchanged sources on every runtime lookup has no decision consumer.
        return _cached_default_production_workflows(v5_source_tree_digest())
    # Explicit environments bypass that cache and still need a full audit.
    assert_v5_determinism()
    # The delegated helper is the only place that invokes compile_source(
    # for this explicit path.
    return _compile_production_workflows(environment)
