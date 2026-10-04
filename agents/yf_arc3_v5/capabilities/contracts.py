"""Immutable descriptive contracts for generic V5 capabilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

from pydantic import Field, model_validator

from agents.yf_arc3_v5.logos.operations import CapabilityDefinition
from agents.yf_arc3_v5.logos.types import (
    DeclaredCausalClass,
    EffectSignatureNovelty,
    FrozenMap,
    FrozenModel,
    GoalGapDimension,
    GoalEdgeKind,
    GoalKind,
    GoalLifecycleStatus,
    GoalNodeKind,
    StaticKnowledgeEdgeKind,
    StaticKnowledgeNodeKind,
    NonNegativeRevision,
    ObservableChangeScope,
    PeriodicCycleWorkflowStatus,
    Ref,
    RelationDimension,
    require_unique,
)

Position = tuple[int, int]
GoalGapValue: TypeAlias = bool | int | Ref | tuple[int, int]


Rational = tuple[int, int]
RationalPoint = tuple[Rational, Rational]
RationalMatrix2 = tuple[tuple[Rational, Rational], tuple[Rational, Rational]]


class RationalAnchorDependenceInput(FrozenModel):
    """A source-supplied finite rational dependence and explicit quantizer."""

    anchor_positions: tuple[Position, ...] = Field(min_length=1, max_length=16)
    coefficient_numerators: tuple[int, ...] = Field(min_length=1, max_length=16)
    coefficient_denominators: tuple[int, ...] = Field(min_length=1, max_length=16)
    bias_numerators: Position = (0, 0)
    bias_denominator: int = Field(default=1, ge=1)
    quantizer: Literal[
        "floor",
        "ceil",
        "truncate_toward_zero",
        "nearest_ties_to_even",
    ]
    schema_version: Ref = "yf_arc3_v5.rational_anchor_dependence_input.v1"

    @model_validator(mode="after")
    def validate_dependence(self) -> "RationalAnchorDependenceInput":
        count = len(self.anchor_positions)
        if len(self.coefficient_numerators) != count:
            raise ValueError("one coefficient numerator is required per anchor")
        if len(self.coefficient_denominators) != count:
            raise ValueError("one coefficient denominator is required per anchor")
        if any(item <= 0 for item in self.coefficient_denominators):
            raise ValueError("coefficient denominators must be positive")
        return self


class RationalAnchorDependenceMeasurements(FrozenModel):
    exact_coordinates: tuple[Rational, Rational]
    quantized_position: Position
    quantizer: Literal[
        "floor",
        "ceil",
        "truncate_toward_zero",
        "nearest_ties_to_even",
    ]
    schema_version: Ref = "yf_arc3_v5.rational_anchor_dependence_measurements.v1"


class InverseMeanAnchorDomainInput(FrozenModel):
    """Bounded integer domain for one unknown member of an established mean."""

    fixed_anchor_positions: tuple[Position, ...] = Field(min_length=1, max_length=15)
    anchor_count: int = Field(ge=2, le=16)
    desired_quantized_position: Position
    candidate_top_left: Position
    candidate_bottom_right: Position
    quantizer: Literal[
        "floor",
        "ceil",
        "truncate_toward_zero",
        "nearest_ties_to_even",
    ]
    maximum_tested_positions: int = Field(ge=1, le=4096)
    schema_version: Ref = "yf_arc3_v5.inverse_mean_anchor_domain_input.v1"

    @model_validator(mode="after")
    def validate_domain(self) -> "InverseMeanAnchorDomainInput":
        if len(self.fixed_anchor_positions) != self.anchor_count - 1:
            raise ValueError("fixed anchors must account for every non-variable member")
        top, left = self.candidate_top_left
        bottom, right = self.candidate_bottom_right
        if bottom < top or right < left:
            raise ValueError("candidate bounds must be ordered")
        domain_size = (bottom - top + 1) * (right - left + 1)
        if domain_size > self.maximum_tested_positions:
            raise ValueError("candidate domain exceeds the declared hard bound")
        return self


class InverseMeanAnchorDomainMeasurements(FrozenModel):
    compatible_positions: tuple[Position, ...]
    tested_position_count: int = Field(ge=1, le=4096)
    domain_exhaustive: bool
    quantizer: Literal[
        "floor",
        "ceil",
        "truncate_toward_zero",
        "nearest_ties_to_even",
    ]
    schema_version: Ref = "yf_arc3_v5.inverse_mean_anchor_domain_measurements.v1"


class InverseAnchorTypedClearanceInput(FrozenModel):
    """Source-supplied typed footprints and transition-path evidence."""

    anchor_footprint: tuple[Position, ...] = Field(min_length=1, max_length=4096)
    dependent_footprint: tuple[Position, ...] = Field(min_length=1, max_length=4096)
    anchor_allowed_positions: tuple[Position, ...] = Field(max_length=4096)
    dependent_allowed_positions: tuple[Position, ...] = Field(max_length=4096)
    transition_path_policy: Literal[
        "endpoint_only",
        "supplied_swept_footprints",
        "unmeasured",
    ]
    dependent_swept_footprints: tuple[tuple[Position, ...], ...] = Field(
        default=(), max_length=128
    )
    maximum_examined_positions: int = Field(ge=1, le=65536)
    schema_version: Ref = "yf_arc3_v5.inverse_anchor_typed_clearance_input.v1"

    @model_validator(mode="after")
    def validate_typed_clearance(self) -> "InverseAnchorTypedClearanceInput":
        require_unique(self.anchor_footprint, "anchor footprint positions")
        require_unique(self.dependent_footprint, "dependent footprint positions")
        require_unique(self.anchor_allowed_positions, "anchor allowed positions")
        require_unique(
            self.dependent_allowed_positions, "dependent allowed positions"
        )
        if (
            self.transition_path_policy == "supplied_swept_footprints"
            and not self.dependent_swept_footprints
        ):
            raise ValueError("swept footprints are required by the supplied path policy")
        if (
            self.transition_path_policy != "supplied_swept_footprints"
            and self.dependent_swept_footprints
        ):
            raise ValueError("swept footprints require the supplied swept-path policy")
        for index, footprint in enumerate(self.dependent_swept_footprints):
            if not footprint:
                raise ValueError(f"swept footprint {index} must not be empty")
            require_unique(footprint, f"swept footprint {index} positions")
        examined = (
            len(self.anchor_footprint)
            + len(self.dependent_footprint)
            + sum(len(item) for item in self.dependent_swept_footprints)
        )
        if examined > self.maximum_examined_positions:
            raise ValueError("typed clearance packet exceeds the declared hard bound")
        return self


class InverseAnchorTypedClearanceMeasurements(FrozenModel):
    anchor_blocked_positions: tuple[Position, ...]
    dependent_blocked_positions: tuple[Position, ...]
    transition_blocked_positions: tuple[Position, ...]
    anchor_destination_clear: bool
    dependent_destination_clear: bool
    transition_path_status: Literal["clear", "blocked", "unknown"]
    examined_position_count: int = Field(ge=1, le=65536)
    schema_version: Ref = "yf_arc3_v5.inverse_anchor_typed_clearance_measurements.v1"


class InverseAnchorWireState(FrozenModel):
    wire_ref: Ref
    endpoint_refs: tuple[Ref, Ref]
    raster_positions: tuple[Position, ...] = Field(max_length=4096)

    @model_validator(mode="after")
    def validate_wire(self) -> "InverseAnchorWireState":
        if self.endpoint_refs[0] == self.endpoint_refs[1]:
            raise ValueError("wire endpoints must be distinct")
        require_unique(self.raster_positions, f"wire raster {self.wire_ref}")
        return self


class InverseAnchorSystemState(FrozenModel):
    system_ref: Ref
    active_anchor_ref: Ref
    anchor_positions: tuple[tuple[Ref, Position], ...] = Field(
        min_length=2, max_length=16
    )
    dependent_position: Position
    dependent_signature: tuple[tuple[Position, Ref], ...] = Field(max_length=4096)
    pickup_refs: tuple[Ref, ...] = Field(max_length=256)
    wires: tuple[InverseAnchorWireState, ...] = Field(max_length=32)

    @model_validator(mode="after")
    def validate_system(self) -> "InverseAnchorSystemState":
        anchor_refs = tuple(item[0] for item in self.anchor_positions)
        require_unique(anchor_refs, f"anchor refs for {self.system_ref}")
        if self.active_anchor_ref not in anchor_refs:
            raise ValueError("active anchor must belong to its system")
        signature_positions = tuple(item[0] for item in self.dependent_signature)
        require_unique(signature_positions, f"dependent signature for {self.system_ref}")
        require_unique(self.pickup_refs, f"pickup refs for {self.system_ref}")
        require_unique(
            tuple(item.wire_ref for item in self.wires),
            f"wire refs for {self.system_ref}",
        )
        return self


class InverseAnchorSuccessorInput(FrozenModel):
    systems: tuple[InverseAnchorSystemState, ...] = Field(min_length=1, max_length=16)
    affected_system_ref: Ref
    operator_kind: Literal["transfer_control", "place_active_anchor"]
    requested_anchor_ref: Ref
    destination: Position | None = None
    supplied_dependent_position: Position | None = None
    supplied_pickup_refs: tuple[Ref, ...] | None = None
    supplied_wires: tuple[InverseAnchorWireState, ...] | None = None
    schema_version: Ref = "yf_arc3_v5.inverse_anchor_successor_input.v1"

    @model_validator(mode="after")
    def validate_successor(self) -> "InverseAnchorSuccessorInput":
        require_unique(tuple(item.system_ref for item in self.systems), "system refs")
        systems = {item.system_ref: item for item in self.systems}
        if self.affected_system_ref not in systems:
            raise ValueError("affected system is absent")
        affected = systems[self.affected_system_ref]
        anchor_refs = {item[0] for item in affected.anchor_positions}
        if self.requested_anchor_ref not in anchor_refs:
            raise ValueError("requested anchor is absent from the affected system")
        if self.operator_kind == "transfer_control":
            if any(
                item is not None
                for item in (
                    self.destination,
                    self.supplied_dependent_position,
                    self.supplied_pickup_refs,
                    self.supplied_wires,
                )
            ):
                raise ValueError("control transfer must not include placement deltas")
        else:
            if self.requested_anchor_ref != affected.active_anchor_ref:
                raise ValueError("placement must address the active anchor")
            if self.destination is None or self.supplied_dependent_position is None:
                raise ValueError("placement requires destination and dependent position")
            if self.supplied_pickup_refs is None or self.supplied_wires is None:
                raise ValueError("placement requires complete pickup and wire results")
            require_unique(self.supplied_pickup_refs, "supplied pickup refs")
            require_unique(
                tuple(item.wire_ref for item in self.supplied_wires),
                "supplied wire refs",
            )
        return self


class InverseAnchorSuccessorMeasurements(FrozenModel):
    resulting_systems: tuple[InverseAnchorSystemState, ...]
    affected_system_ref: Ref
    unaffected_system_refs: tuple[Ref, ...]
    operator_kind: Literal["transfer_control", "place_active_anchor"]
    schema_version: Ref = "yf_arc3_v5.inverse_anchor_successor_measurements.v1"


class InverseAnchorWaypointCandidate(FrozenModel):
    candidate_ref: Ref
    dependent_waypoint: Position
    anchor_configuration: tuple[tuple[Ref, Position], ...] = Field(
        min_length=2, max_length=16
    )
    required_corridor_constraint_refs: tuple[Ref, ...] = Field(max_length=64)
    satisfied_corridor_constraint_refs: tuple[Ref, ...] = Field(max_length=64)
    transition_path_status: Literal["clear", "blocked", "unknown"]
    configuration_complete: bool

    @model_validator(mode="after")
    def validate_waypoint_candidate(self) -> "InverseAnchorWaypointCandidate":
        require_unique(
            tuple(item[0] for item in self.anchor_configuration),
            f"anchor configuration for {self.candidate_ref}",
        )
        require_unique(
            self.required_corridor_constraint_refs,
            f"required corridor constraints for {self.candidate_ref}",
        )
        require_unique(
            self.satisfied_corridor_constraint_refs,
            f"satisfied corridor constraints for {self.candidate_ref}",
        )
        return self


class InverseAnchorWaypointCandidatesInput(FrozenModel):
    candidates: tuple[InverseAnchorWaypointCandidate, ...] = Field(
        min_length=1, max_length=3
    )
    maximum_candidates: int = Field(default=3, ge=1, le=3)
    schema_version: Ref = "yf_arc3_v5.inverse_anchor_waypoint_candidates_input.v1"

    @model_validator(mode="after")
    def validate_waypoint_candidates(self) -> "InverseAnchorWaypointCandidatesInput":
        require_unique(tuple(item.candidate_ref for item in self.candidates), "candidate refs")
        if len(self.candidates) > self.maximum_candidates:
            raise ValueError("waypoint candidates exceed the declared global bound")
        return self


class InverseAnchorWaypointCandidateMeasurement(FrozenModel):
    candidate_ref: Ref
    missing_corridor_constraint_refs: tuple[Ref, ...]
    transition_path_status: Literal["clear", "blocked", "unknown"]
    configuration_complete: bool


class InverseAnchorWaypointCandidatesMeasurements(FrozenModel):
    compatible_candidate_refs: tuple[Ref, ...]
    uncertain_candidate_refs: tuple[Ref, ...]
    rejected_candidate_refs: tuple[Ref, ...]
    candidate_measurements: tuple[InverseAnchorWaypointCandidateMeasurement, ...]
    candidate_bound_respected: bool
    schema_version: Ref = "yf_arc3_v5.inverse_anchor_waypoint_candidates_measurements.v1"


def _require_rational(value: Rational, label: str) -> None:
    if value[1] <= 0:
        raise ValueError(f"{label} denominator must be positive")


def _require_rational_point(value: RationalPoint, label: str) -> None:
    _require_rational(value[0], f"{label} x")
    _require_rational(value[1], f"{label} y")


class ArticulatedBodyState(FrozenModel):
    """One source-grounded local body frame in a bounded attachment forest."""

    body_ref: Ref
    parent_ref: Ref | None = None
    channel_refs: tuple[Ref, ...] = Field(default=(), max_length=8)
    local_linear: RationalMatrix2 = (
        (((1, 1), (0, 1))),
        (((0, 1), (1, 1))),
    )
    local_translation: RationalPoint = ((0, 1), (0, 1))
    local_axis: RationalPoint = ((1, 1), (0, 1))
    local_extent: Rational = (0, 1)
    marker_offsets: tuple[tuple[Ref, RationalPoint], ...] = Field(
        default=(), max_length=64
    )

    @model_validator(mode="after")
    def validate_body(self) -> "ArticulatedBodyState":
        for row_index, row in enumerate(self.local_linear):
            for column_index, value in enumerate(row):
                _require_rational(
                    value, f"local linear {row_index},{column_index} for {self.body_ref}"
                )
        _require_rational_point(
            self.local_translation, f"local translation for {self.body_ref}"
        )
        _require_rational_point(self.local_axis, f"local axis for {self.body_ref}")
        _require_rational(self.local_extent, f"local extent for {self.body_ref}")
        require_unique(self.channel_refs, f"channel refs for {self.body_ref}")
        require_unique(
            tuple(item[0] for item in self.marker_offsets),
            f"marker refs for {self.body_ref}",
        )
        for marker_ref, offset in self.marker_offsets:
            _require_rational_point(offset, f"marker offset {marker_ref}")
        return self


def _validate_articulated_forest(bodies: tuple[ArticulatedBodyState, ...]) -> None:
    refs = tuple(item.body_ref for item in bodies)
    require_unique(refs, "articulated body refs")
    known = set(refs)
    parent_by_ref = {item.body_ref: item.parent_ref for item in bodies}
    for body in bodies:
        if body.parent_ref is not None and body.parent_ref not in known:
            raise ValueError(f"parent {body.parent_ref} is absent")
        if body.parent_ref == body.body_ref:
            raise ValueError("an articulated body cannot parent itself")
        seen: set[str] = set()
        cursor: str | None = body.body_ref
        while cursor is not None:
            if cursor in seen:
                raise ValueError("articulated attachment graph must be acyclic")
            seen.add(cursor)
            cursor = parent_by_ref[cursor]


class ArticulatedBodyGeometry(FrozenModel):
    body_ref: Ref
    root_position: RationalPoint
    free_end_position: RationalPoint
    world_axis: RationalPoint
    local_extent: Rational
    marker_positions: tuple[tuple[Ref, RationalPoint], ...]


class ArticulatedKinematicsInput(FrozenModel):
    bodies: tuple[ArticulatedBodyState, ...] = Field(min_length=1, max_length=32)
    maximum_bodies: int = Field(default=32, ge=1, le=32)
    schema_version: Ref = "yf_arc3_v5.articulated_kinematics_input.v1"

    @model_validator(mode="after")
    def validate_kinematics(self) -> "ArticulatedKinematicsInput":
        _validate_articulated_forest(self.bodies)
        if len(self.bodies) > self.maximum_bodies:
            raise ValueError("articulated body packet exceeds the declared hard bound")
        return self


class ArticulatedKinematicsMeasurements(FrozenModel):
    body_geometries: tuple[ArticulatedBodyGeometry, ...]
    body_count: int = Field(ge=1, le=32)
    schema_version: Ref = "yf_arc3_v5.articulated_kinematics_measurements.v1"


class ArticulatedLocalOperation(FrozenModel):
    operation_ref: Ref
    cause_ref: Ref
    receiver_ref: Ref
    order_index: int | None = Field(default=None, ge=0, le=63)
    composition_side: Literal["before_local", "after_local"]
    linear_delta: RationalMatrix2 = (
        (((1, 1), (0, 1))),
        (((0, 1), (1, 1))),
    )
    translation_delta: RationalPoint = ((0, 1), (0, 1))
    extent_delta: Rational = (0, 1)

    @model_validator(mode="after")
    def validate_operation(self) -> "ArticulatedLocalOperation":
        for row_index, row in enumerate(self.linear_delta):
            for column_index, value in enumerate(row):
                _require_rational(
                    value, f"linear delta {row_index},{column_index} for {self.operation_ref}"
                )
        _require_rational_point(
            self.translation_delta, f"translation delta for {self.operation_ref}"
        )
        _require_rational(self.extent_delta, f"extent delta for {self.operation_ref}")
        return self


class ArticulatedSharedSnapshotInput(FrozenModel):
    bodies: tuple[ArticulatedBodyState, ...] = Field(min_length=1, max_length=32)
    operations: tuple[ArticulatedLocalOperation, ...] = Field(
        min_length=1, max_length=64
    )
    maximum_operations: int = Field(default=64, ge=1, le=64)
    schema_version: Ref = "yf_arc3_v5.articulated_shared_snapshot_input.v1"

    @model_validator(mode="after")
    def validate_snapshot(self) -> "ArticulatedSharedSnapshotInput":
        _validate_articulated_forest(self.bodies)
        require_unique(
            tuple(item.operation_ref for item in self.operations),
            "articulated operation refs",
        )
        if len(self.operations) > self.maximum_operations:
            raise ValueError("articulated operation packet exceeds the declared hard bound")
        body_refs = {item.body_ref for item in self.bodies}
        reports: dict[tuple[str, str], ArticulatedLocalOperation] = {}
        for operation in self.operations:
            if operation.receiver_ref not in body_refs:
                raise ValueError("articulated operation receiver is absent")
            key = (operation.cause_ref, operation.receiver_ref)
            previous = reports.get(key)
            if previous is not None:
                comparable_previous = previous.model_copy(
                    update={"operation_ref": operation.operation_ref}
                )
                if comparable_previous != operation:
                    raise ValueError("repeated cause report has conflicting payload")
            reports[key] = operation
        return self


class ArticulatedSharedSnapshotMeasurements(FrozenModel):
    status: Literal["applied", "ambiguous_order"]
    resulting_bodies: tuple[ArticulatedBodyState, ...]
    deduplicated_operation_refs: tuple[Ref, ...]
    unresolved_order_refs: tuple[Ref, ...]
    applied_operation_count: int = Field(ge=0, le=64)
    schema_version: Ref = "yf_arc3_v5.articulated_shared_snapshot_measurements.v1"


class ArticulatedSuccessorInput(FrozenModel):
    before_bodies: tuple[ArticulatedBodyState, ...] = Field(min_length=1, max_length=32)
    supplied_after_bodies: tuple[ArticulatedBodyState, ...] = Field(
        min_length=1, max_length=32
    )
    transaction_outcome: Literal[
        "committed", "rejected", "partial_prefix", "restored"
    ]
    atomic_model: bool
    committed_packet: Literal["before", "supplied_after"]
    schema_version: Ref = "yf_arc3_v5.articulated_successor_input.v1"

    @model_validator(mode="after")
    def validate_successor(self) -> "ArticulatedSuccessorInput":
        _validate_articulated_forest(self.before_bodies)
        _validate_articulated_forest(self.supplied_after_bodies)
        if tuple(item.body_ref for item in self.before_bodies) != tuple(
            item.body_ref for item in self.supplied_after_bodies
        ):
            raise ValueError("successor packets must retain ordered body identity")
        if self.atomic_model and self.transaction_outcome in {"rejected", "restored"}:
            if self.committed_packet != "before":
                raise ValueError("atomic rejection or restoration must retain the before packet")
        if self.transaction_outcome in {"committed", "partial_prefix"}:
            if self.committed_packet != "supplied_after":
                raise ValueError("committed outcomes require the supplied after packet")
        return self


class ArticulatedSuccessorMeasurements(FrozenModel):
    resulting_bodies: tuple[ArticulatedBodyState, ...]
    changed_extent_refs: tuple[Ref, ...]
    changed_pose_refs: tuple[Ref, ...]
    changed_marker_refs: tuple[Ref, ...]
    transaction_outcome: Literal[
        "committed", "rejected", "partial_prefix", "restored"
    ]
    atomic_model: bool
    schema_version: Ref = "yf_arc3_v5.articulated_successor_measurements.v1"


class TypedInstructionResource(FrozenModel):
    """One physical instruction occurrence supplied by the descriptive layer."""

    token_ref: Ref
    instruction_kind: Literal["literal", "call"]
    value_ref: Ref | None = None
    target_definition_ref: Ref | None = None

    @model_validator(mode="after")
    def validate_instruction(self) -> "TypedInstructionResource":
        if self.instruction_kind == "literal":
            if self.value_ref is None or self.target_definition_ref is not None:
                raise ValueError("a literal requires only a value ref")
        elif self.target_definition_ref is None or self.value_ref is not None:
            raise ValueError("a call requires only a target definition ref")
        return self


class TypedSocketState(FrozenModel):
    socket_ref: Ref
    accepted_instruction_kinds: tuple[Literal["literal", "call"], ...] = Field(
        min_length=1, max_length=2
    )
    occupant_token_ref: Ref | None = None

    @model_validator(mode="after")
    def validate_socket(self) -> "TypedSocketState":
        require_unique(self.accepted_instruction_kinds, f"accepted kinds for {self.socket_ref}")
        return self


class TypedSocketDomain(FrozenModel):
    socket_ref: Ref
    compatible_token_refs: tuple[Ref, ...]


class TypedAssignmentDomainsInput(FrozenModel):
    tokens: tuple[TypedInstructionResource, ...] = Field(min_length=1, max_length=32)
    sockets: tuple[TypedSocketState, ...] = Field(min_length=1, max_length=16)
    definition_refs: tuple[Ref, ...] = Field(min_length=1, max_length=16)
    maximum_domain_members: int = Field(default=32, ge=1, le=32)
    schema_version: Ref = "yf_arc3_v5.typed_assignment_domains_input.v1"

    @model_validator(mode="after")
    def validate_packet(self) -> "TypedAssignmentDomainsInput":
        require_unique(tuple(item.token_ref for item in self.tokens), "typed token refs")
        require_unique(tuple(item.socket_ref for item in self.sockets), "typed socket refs")
        require_unique(self.definition_refs, "typed definition refs")
        known_tokens = {item.token_ref for item in self.tokens}
        occupied = tuple(
            item.occupant_token_ref
            for item in self.sockets
            if item.occupant_token_ref is not None
        )
        require_unique(occupied, "typed occupied token refs")
        if any(item not in known_tokens for item in occupied):
            raise ValueError("a socket occupant is absent from the supplied token inventory")
        return self


class TypedAssignmentDomainsMeasurements(FrozenModel):
    socket_domains: tuple[TypedSocketDomain, ...]
    deficit_socket_refs: tuple[Ref, ...]
    incompatible_call_token_refs: tuple[Ref, ...]
    physical_token_count: int = Field(ge=1, le=32)
    schema_version: Ref = "yf_arc3_v5.typed_assignment_domains_measurements.v1"


class TypedResourceLocation(FrozenModel):
    token_ref: Ref
    location_ref: Ref


class TypedEditSuccessorInput(FrozenModel):
    locations: tuple[TypedResourceLocation, ...] = Field(min_length=1, max_length=32)
    sockets: tuple[TypedSocketState, ...] = Field(min_length=1, max_length=16)
    selection_token_ref: Ref | None = None
    edit_kind: Literal["token_click", "socket_click"]
    edit_ref: Ref
    schema_version: Ref = "yf_arc3_v5.typed_edit_successor_input.v1"

    @model_validator(mode="after")
    def validate_edit(self) -> "TypedEditSuccessorInput":
        token_refs = tuple(item.token_ref for item in self.locations)
        require_unique(token_refs, "typed edit token refs")
        require_unique(tuple(item.socket_ref for item in self.sockets), "typed edit socket refs")
        if self.selection_token_ref is not None and self.selection_token_ref not in token_refs:
            raise ValueError("the supplied selection token is absent")
        if self.edit_kind == "token_click" and self.edit_ref not in token_refs:
            raise ValueError("the clicked token is absent")
        if self.edit_kind == "socket_click" and self.edit_ref not in {
            item.socket_ref for item in self.sockets
        }:
            raise ValueError("the clicked socket is absent")
        occupants = tuple(
            item.occupant_token_ref
            for item in self.sockets
            if item.occupant_token_ref is not None
        )
        require_unique(occupants, "typed edit occupied token refs")
        if any(item not in token_refs for item in occupants):
            raise ValueError("a socket occupant is absent from the location packet")
        return self


class TypedEditSuccessorMeasurements(FrozenModel):
    status: Literal["selection_changed", "unchanged_without_selection", "transferred", "exchanged"]
    resulting_locations: tuple[TypedResourceLocation, ...]
    resulting_sockets: tuple[TypedSocketState, ...]
    selection_token_ref: Ref | None = None
    displaced_token_ref: Ref | None = None
    schema_version: Ref = "yf_arc3_v5.typed_edit_successor_measurements.v1"


class TypedProgramDefinition(FrozenModel):
    definition_ref: Ref
    instruction_token_refs: tuple[Ref, ...] = Field(max_length=32)


class TypedEmissionProvenance(FrozenModel):
    output_index: int = Field(ge=0, le=255)
    value_ref: Ref
    literal_token_ref: Ref
    definition_ref: Ref
    instruction_index: int = Field(ge=0, le=31)
    call_site_chain: tuple[tuple[Ref, int], ...] = Field(max_length=32)


class TypedInterpreterInput(FrozenModel):
    tokens: tuple[TypedInstructionResource, ...] = Field(min_length=1, max_length=64)
    definitions: tuple[TypedProgramDefinition, ...] = Field(min_length=1, max_length=16)
    entry_definition_ref: Ref
    reference_values: tuple[Ref, ...] = Field(min_length=1, max_length=256)
    boundary_kind: Literal["full_termination", "finite_prefix"]
    official_acceptance_evidence: bool
    maximum_steps: int = Field(default=256, ge=1, le=1024)
    maximum_calls: int = Field(default=64, ge=0, le=256)
    maximum_stack_depth: int = Field(default=32, ge=0, le=64)
    maximum_output: int = Field(default=256, ge=1, le=256)
    schema_version: Ref = "yf_arc3_v5.typed_interpreter_input.v1"

    @model_validator(mode="after")
    def validate_interpreter(self) -> "TypedInterpreterInput":
        token_refs = tuple(item.token_ref for item in self.tokens)
        definition_refs = tuple(item.definition_ref for item in self.definitions)
        require_unique(token_refs, "interpreter token refs")
        require_unique(definition_refs, "interpreter definition refs")
        if self.entry_definition_ref not in definition_refs:
            raise ValueError("entry definition is absent")
        known_tokens = set(token_refs)
        referenced: list[str] = []
        for definition in self.definitions:
            referenced.extend(definition.instruction_token_refs)
            if any(item not in known_tokens for item in definition.instruction_token_refs):
                raise ValueError("a definition references an absent instruction token")
        if len(referenced) != len(set(referenced)):
            raise ValueError("one physical instruction token cannot occupy multiple sockets")
        known_definitions = set(definition_refs)
        for token in self.tokens:
            if (
                token.instruction_kind == "call"
                and token.target_definition_ref not in known_definitions
            ):
                raise ValueError("a call target is absent")
        if self.maximum_output < len(self.reference_values):
            raise ValueError("the output bound must cover the supplied reference")
        return self


class TypedInterpreterMeasurements(FrozenModel):
    status: Literal[
        "matched_finite_prefix",
        "normal_termination",
        "mismatch_localized",
        "nonproductive_cycle",
        "budget_reached",
        "stack_bound_reached",
        "call_bound_reached",
    ]
    emitted_values: tuple[Ref, ...]
    emissions: tuple[TypedEmissionProvenance, ...]
    first_mismatch_index: int | None = Field(default=None, ge=0, le=255)
    pending_returns: tuple[tuple[Ref, int], ...]
    prefix_match: bool
    normal_termination: bool
    official_acceptance_observed: bool
    steps_executed: int = Field(ge=0, le=1024)
    calls_executed: int = Field(ge=0, le=256)
    schema_version: Ref = "yf_arc3_v5.typed_interpreter_measurements.v1"


class RecipePatternDifferenceInput(FrozenModel):
    lattice_refs: tuple[Ref, ...] = Field(min_length=1, max_length=64)
    current_active_refs: tuple[Ref, ...] = Field(max_length=64)
    target_active_refs: tuple[Ref, ...] = Field(max_length=64)
    supported_edit_refs: tuple[Ref, ...] = Field(max_length=64)
    schema_version: Ref = "yf_arc3_v5.recipe_pattern_difference_input.v1"

    @model_validator(mode="after")
    def validate_pattern(self) -> "RecipePatternDifferenceInput":
        require_unique(self.lattice_refs, "recipe lattice refs")
        require_unique(self.current_active_refs, "current recipe refs")
        require_unique(self.target_active_refs, "target recipe refs")
        require_unique(self.supported_edit_refs, "supported recipe edit refs")
        lattice = set(self.lattice_refs)
        if any(item not in lattice for item in (*self.current_active_refs, *self.target_active_refs, *self.supported_edit_refs)):
            raise ValueError("recipe refs must belong to the supplied lattice")
        return self


class RecipePatternDifferenceMeasurements(FrozenModel):
    missing_active_refs: tuple[Ref, ...]
    extra_active_refs: tuple[Ref, ...]
    supported_required_edit_refs: tuple[Ref, ...]
    unsupported_required_edit_refs: tuple[Ref, ...]
    schema_version: Ref = "yf_arc3_v5.recipe_pattern_difference_measurements.v1"


class RecipeWorldState(FrozenModel):
    values: tuple[tuple[Ref, Ref], ...] = Field(max_length=64)

    @model_validator(mode="after")
    def validate_values(self) -> "RecipeWorldState":
        require_unique(tuple(item[0] for item in self.values), "recipe world value refs")
        return self


class RecipeSuccessorInput(FrozenModel):
    lattice_refs: tuple[Ref, ...] = Field(min_length=1, max_length=64)
    current_active_refs: tuple[Ref, ...] = Field(max_length=64)
    edit_ref: Ref
    target_active_refs: tuple[Ref, ...] = Field(max_length=64)
    completion_trigger: Literal["exact_match"]
    auto_clear_on_trigger: bool
    world_before: RecipeWorldState
    supplied_world_after: RecipeWorldState
    schema_version: Ref = "yf_arc3_v5.recipe_successor_input.v1"

    @model_validator(mode="after")
    def validate_successor(self) -> "RecipeSuccessorInput":
        require_unique(self.lattice_refs, "recipe successor lattice refs")
        lattice = set(self.lattice_refs)
        if self.edit_ref not in lattice or any(item not in lattice for item in (*self.current_active_refs, *self.target_active_refs)):
            raise ValueError("recipe successor refs must belong to the lattice")
        return self


class RecipeSuccessorMeasurements(FrozenModel):
    active_refs_after_edit: tuple[Ref, ...]
    active_refs_after_transaction: tuple[Ref, ...]
    trigger_satisfied: bool
    resulting_world: RecipeWorldState
    schema_version: Ref = "yf_arc3_v5.recipe_successor_measurements.v1"


class RecipeBodyState(FrozenModel):
    anchor_position: Position
    occupied_positions: tuple[Position, ...] = Field(min_length=1, max_length=256)
    size_ref: Ref
    heading_ref: Ref

    @model_validator(mode="after")
    def validate_body(self) -> "RecipeBodyState":
        require_unique(self.occupied_positions, "recipe body positions")
        return self


class RecipeResourceRegion(FrozenModel):
    resource_ref: Ref
    positions: tuple[Position, ...] = Field(min_length=1, max_length=64)


class RecipeResizeSuccessorInput(FrozenModel):
    before_body: RecipeBodyState
    supplied_after_body: RecipeBodyState
    substrate_positions: tuple[Position, ...] = Field(max_length=4096)
    wall_positions: tuple[Position, ...] = Field(max_length=4096)
    resources: tuple[RecipeResourceRegion, ...] = Field(max_length=32)
    schema_version: Ref = "yf_arc3_v5.recipe_resize_successor_input.v1"


class RecipeResizeSuccessorMeasurements(FrozenModel):
    anchor_delta: Position
    heading_preserved: bool
    substrate_missing_positions: tuple[Position, ...]
    wall_conflict_positions: tuple[Position, ...]
    contacted_resource_refs: tuple[Ref, ...]
    schema_version: Ref = "yf_arc3_v5.recipe_resize_successor_measurements.v1"


class RecipeResourceEvent(FrozenModel):
    event_ref: Ref
    event_kind: Literal["cost", "pickup", "refill"]
    amount: int = Field(ge=0, le=1000000)


class RecipeResourceOrderInput(FrozenModel):
    initial_quantity: int = Field(ge=0, le=1000000)
    events: tuple[RecipeResourceEvent, ...] = Field(max_length=128)
    schema_version: Ref = "yf_arc3_v5.recipe_resource_order_input.v1"


class RecipeResourceOrderMeasurements(FrozenModel):
    quantities_after_events: tuple[tuple[Ref, int], ...]
    first_insufficient_event_ref: Ref | None = None
    final_quantity: int = Field(ge=0, le=1000000)
    schema_version: Ref = "yf_arc3_v5.recipe_resource_order_measurements.v1"


class PiercingRodState(FrozenModel):
    rod_ref: Ref
    root_ref: Ref
    root_to_tip_positions: tuple[Position, ...] = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def validate_positions(self) -> "PiercingRodState":
        require_unique(self.root_to_tip_positions, f"positions for {self.rod_ref}")
        return self


class PiercingPieceState(FrozenModel):
    piece_ref: Ref
    type_ref: Ref
    occupied_positions: tuple[Position, ...] = Field(min_length=1, max_length=256)
    support_refs: tuple[Ref, ...] = Field(max_length=16)

    @model_validator(mode="after")
    def validate_piece(self) -> "PiercingPieceState":
        require_unique(self.occupied_positions, f"positions for {self.piece_ref}")
        require_unique(self.support_refs, f"supports for {self.piece_ref}")
        return self


class PiercingIntersectionInput(FrozenModel):
    rods: tuple[PiercingRodState, ...] = Field(min_length=1, max_length=16)
    pieces: tuple[PiercingPieceState, ...] = Field(max_length=128)
    requested_root_ref: Ref
    requested_type_order: tuple[Ref, ...] = Field(max_length=128)
    schema_version: Ref = "yf_arc3_v5.piercing_intersection_input.v1"


class PiercingRodIntersection(FrozenModel):
    rod_ref: Ref
    piece_ref: Ref
    intersection_positions: tuple[Position, ...]
    first_root_index: int = Field(ge=0, le=511)


class PiercingIntersectionMeasurements(FrozenModel):
    intersections: tuple[PiercingRodIntersection, ...]
    ordered_piece_refs_by_rod: tuple[tuple[Ref, tuple[Ref, ...]], ...]
    requested_support_ref: Ref | None = None
    requested_support_piece_refs: tuple[Ref, ...]
    requested_support_type_order: tuple[Ref, ...]
    rooted_match: bool
    remaining_unconstrained_piece_refs: tuple[Ref, ...]
    schema_version: Ref = "yf_arc3_v5.piercing_intersection_measurements.v1"


class PiercingPushSuccessorInput(FrozenModel):
    pieces: tuple[PiercingPieceState, ...] = Field(max_length=128)
    contacted_piece_refs: tuple[Ref, ...] = Field(max_length=128)
    movable_piece_refs: tuple[Ref, ...] = Field(max_length=128)
    translation_delta: Position
    schema_version: Ref = "yf_arc3_v5.piercing_push_successor_input.v1"


class PiercingPushSuccessorMeasurements(FrozenModel):
    resulting_pieces: tuple[PiercingPieceState, ...]
    translated_piece_refs: tuple[Ref, ...]
    blocked_piece_refs: tuple[Ref, ...]
    schema_version: Ref = "yf_arc3_v5.piercing_push_successor_measurements.v1"


class PiercingPierceSuccessorInput(FrozenModel):
    supplied_after_rod: PiercingRodState
    pieces: tuple[PiercingPieceState, ...] = Field(max_length=128)
    candidate_piece_ref: Ref
    observed_entry_face: Ref
    permitted_entry_face: Ref
    immobilized_piece_refs: tuple[Ref, ...] = Field(max_length=128)
    schema_version: Ref = "yf_arc3_v5.piercing_pierce_successor_input.v1"


class PiercingPierceSuccessorMeasurements(FrozenModel):
    resulting_pieces: tuple[PiercingPieceState, ...]
    pierced: bool
    intersection_positions: tuple[Position, ...]
    resulting_support_refs: tuple[Ref, ...]
    schema_version: Ref = "yf_arc3_v5.piercing_pierce_successor_measurements.v1"


class PiercingWithdrawSuccessorInput(FrozenModel):
    before_rods: tuple[PiercingRodState, ...] = Field(min_length=1, max_length=16)
    supplied_after_rods: tuple[PiercingRodState, ...] = Field(min_length=1, max_length=16)
    pieces: tuple[PiercingPieceState, ...] = Field(max_length=128)
    schema_version: Ref = "yf_arc3_v5.piercing_withdraw_successor_input.v1"


class PiercingWithdrawSuccessorMeasurements(FrozenModel):
    resulting_pieces: tuple[PiercingPieceState, ...]
    retained_relations: tuple[tuple[Ref, Ref], ...]
    released_relations: tuple[tuple[Ref, Ref], ...]
    multi_support_piece_refs: tuple[Ref, ...]
    world_positions_preserved: bool
    schema_version: Ref = "yf_arc3_v5.piercing_withdraw_successor_measurements.v1"


class PiercingObstacleRegion(FrozenModel):
    obstacle_ref: Ref
    occupied_positions: tuple[Position, ...] = Field(min_length=1, max_length=4096)


class PiercingClearanceInput(FrozenModel):
    tool_and_load_positions: tuple[Position, ...] = Field(min_length=1, max_length=4096)
    requested_delta: Position
    obstacles: tuple[PiercingObstacleRegion, ...] = Field(max_length=256)
    free_pieces: tuple[PiercingPieceState, ...] = Field(max_length=128)
    schema_version: Ref = "yf_arc3_v5.piercing_clearance_input.v1"


class PiercingClearanceMeasurements(FrozenModel):
    translated_positions: tuple[Position, ...]
    blocking_obstacle_refs: tuple[Ref, ...]
    blocking_free_piece_refs: tuple[Ref, ...]
    clear: bool
    schema_version: Ref = "yf_arc3_v5.piercing_clearance_measurements.v1"


class OrderedReliefPartition(FrozenModel):
    """One source-declared, non-overlapping region of a sealed frame packet."""

    partition_ref: Ref
    positions: tuple[Position, ...] = Field(max_length=4096)

    @model_validator(mode="after")
    def validate_positions(self) -> "OrderedReliefPartition":
        require_unique(self.positions, f"positions for {self.partition_ref}")
        return self


class OrderedReliefTransitionInput(FrozenModel):
    """Bounded before/after pixels plus source-declared descriptive partitions."""

    before_rows: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=64)
    after_rows: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=64)
    partitions: tuple[OrderedReliefPartition, ...] = Field(
        min_length=1, max_length=64
    )
    application_positions: tuple[Position, ...] = Field(default=(), max_length=4096)
    supplied_write_value: int | None = None
    maximum_compared_cells: int = Field(default=4096, ge=1, le=4096)
    schema_version: Ref = "yf_arc3_v5.ordered_relief_transition_input.v1"

    @model_validator(mode="after")
    def validate_packet(self) -> "OrderedReliefTransitionInput":
        for name, rows in (("before", self.before_rows), ("after", self.after_rows)):
            width = len(rows[0])
            if width < 1 or width > 64 or any(len(row) != width for row in rows):
                raise ValueError(f"{name} rows must form a rectangular grid")
            if len(rows) * width > self.maximum_compared_cells:
                raise ValueError(f"{name} grid exceeds the declared comparison bound")
        refs = tuple(partition.partition_ref for partition in self.partitions)
        require_unique(refs, "ordered relief partition refs")
        partition_positions = tuple(
            position for partition in self.partitions for position in partition.positions
        )
        require_unique(partition_positions, "ordered relief partition positions")
        require_unique(self.application_positions, "ordered relief application positions")
        return self


class OrderedReliefTransitionMeasurements(FrozenModel):
    before_shape: Position
    after_shape: Position
    extent_changed: bool
    changed_positions: tuple[Position, ...]
    changed_count_by_partition: tuple[tuple[Ref, int], ...]
    unpartitioned_changed_count: int = Field(ge=0)
    application_position_count: int = Field(ge=0)
    changed_application_positions: tuple[Position, ...]
    known_same_value_application_positions: tuple[Position, ...]
    compared_cell_count: int = Field(ge=0, le=4096)
    schema_version: Ref = "yf_arc3_v5.ordered_relief_transition_measurements.v1"


class OrderedReliefWriteInput(FrozenModel):
    """One supplied last-writer update over a bounded canvas."""

    canvas_rows: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=64)
    application_positions: tuple[Position, ...] = Field(min_length=1, max_length=4096)
    write_value: int
    maximum_canvas_cells: int = Field(default=4096, ge=1, le=4096)
    schema_version: Ref = "yf_arc3_v5.ordered_relief_write_input.v1"

    @model_validator(mode="after")
    def validate_write(self) -> "OrderedReliefWriteInput":
        width = len(self.canvas_rows[0])
        if width < 1 or width > 64 or any(len(row) != width for row in self.canvas_rows):
            raise ValueError("canvas rows must form a rectangular grid")
        if len(self.canvas_rows) * width > self.maximum_canvas_cells:
            raise ValueError("canvas exceeds the declared cell bound")
        require_unique(self.application_positions, "ordered relief application positions")
        if any(
            row < 0
            or column < 0
            or row >= len(self.canvas_rows)
            or column >= width
            for row, column in self.application_positions
        ):
            raise ValueError("application positions must stay inside the canvas")
        return self


class OrderedReliefWriteMeasurements(FrozenModel):
    resulting_rows: tuple[tuple[int, ...], ...]
    affected_positions: tuple[Position, ...]
    changed_positions: tuple[Position, ...]
    same_value_positions: tuple[Position, ...]
    non_application_cells_preserved: bool
    schema_version: Ref = "yf_arc3_v5.ordered_relief_write_measurements.v1"


class ObservedPoseRecord(FrozenModel):
    pose_ref: Ref
    aspect_ref: Ref
    applicator_size: int = Field(ge=1, le=4096)


class ObservedPoseEdge(FrozenModel):
    source_pose_ref: Ref
    action_ref: Ref
    target_pose_ref: Ref


class OrderedReliefPoseInput(FrozenModel):
    """A source-requested pose tested only against an observed action graph."""

    current_pose_ref: Ref
    requested_pose_ref: Ref
    required_aspect_ref: Ref
    required_applicator_size: int = Field(ge=1, le=4096)
    pose_records: tuple[ObservedPoseRecord, ...] = Field(min_length=1, max_length=32)
    observed_edges: tuple[ObservedPoseEdge, ...] = Field(max_length=128)
    maximum_route_depth: int = Field(default=16, ge=0, le=32)
    schema_version: Ref = "yf_arc3_v5.ordered_relief_pose_input.v1"

    @model_validator(mode="after")
    def validate_pose_graph(self) -> "OrderedReliefPoseInput":
        pose_refs = tuple(record.pose_ref for record in self.pose_records)
        require_unique(pose_refs, "ordered relief pose refs")
        if self.current_pose_ref not in pose_refs:
            raise ValueError("current pose must be present in the observed graph")
        if any(
            edge.source_pose_ref not in pose_refs or edge.target_pose_ref not in pose_refs
            for edge in self.observed_edges
        ):
            raise ValueError("observed pose edges must reference observed poses")
        edge_keys = tuple(
            (edge.source_pose_ref, edge.action_ref, edge.target_pose_ref)
            for edge in self.observed_edges
        )
        require_unique(edge_keys, "ordered relief pose edges")
        return self


class OrderedReliefPoseMeasurements(FrozenModel):
    requested_pose_observed: bool
    aspect_matches: bool
    applicator_size_matches: bool
    route_available: bool
    route_action_refs: tuple[Ref, ...]
    route_pose_refs: tuple[Ref, ...]
    tested_edge_count: int = Field(ge=0, le=128)
    route_depth_bound_reached: bool
    schema_version: Ref = "yf_arc3_v5.ordered_relief_pose_measurements.v1"


class ReconfigurableSupportLayer(FrozenModel):
    """One source-supplied support owner and its exact occupied cells."""

    owner_ref: Ref
    positions: tuple[Position, ...] = Field(max_length=4096)

    @model_validator(mode="after")
    def validate_positions(self) -> "ReconfigurableSupportLayer":
        require_unique(self.positions, f"support positions for {self.owner_ref}")
        return self


class ReconfigurableSupportStep(FrozenModel):
    action_ref: Ref
    delta_row: int
    delta_column: int

    @model_validator(mode="after")
    def validate_translation(self) -> "ReconfigurableSupportStep":
        if (self.delta_row == 0) == (self.delta_column == 0):
            raise ValueError("support steps must be nonzero cardinal translations")
        return self


class ReconfigurableSupportSnapshot(FrozenModel):
    snapshot_ref: Ref
    actor_positions: tuple[Position, ...] = Field(min_length=1, max_length=4096)
    active_layers: tuple[ReconfigurableSupportLayer, ...] = Field(
        min_length=1, max_length=32
    )

    @model_validator(mode="after")
    def validate_snapshot(self) -> "ReconfigurableSupportSnapshot":
        require_unique(self.actor_positions, f"actor positions for {self.snapshot_ref}")
        require_unique(
            tuple(layer.owner_ref for layer in self.active_layers),
            f"support owners for {self.snapshot_ref}",
        )
        return self


class ReconfigurableSupportPacketInput(FrozenModel):
    """Ordered support states and one supplied connection question."""

    snapshots: tuple[ReconfigurableSupportSnapshot, ...] = Field(
        min_length=2, max_length=16
    )
    requested_terminal_actor_positions: tuple[Position, ...] = Field(
        min_length=1, max_length=4096
    )
    movement_steps: tuple[ReconfigurableSupportStep, ...] = Field(
        min_length=1, max_length=16
    )
    maximum_tested_actor_poses: int = Field(default=512, ge=1, le=4096)
    schema_version: Ref = "yf_arc3_v5.reconfigurable_support_packet_input.v1"

    @model_validator(mode="after")
    def validate_packet(self) -> "ReconfigurableSupportPacketInput":
        require_unique(
            tuple(snapshot.snapshot_ref for snapshot in self.snapshots),
            "reconfigurable support snapshot refs",
        )
        require_unique(
            self.requested_terminal_actor_positions,
            "requested terminal actor positions",
        )
        require_unique(
            tuple(step.action_ref for step in self.movement_steps),
            "reconfigurable support action refs",
        )
        if any(len(snapshot.actor_positions) != len(self.snapshots[0].actor_positions) for snapshot in self.snapshots):
            raise ValueError("actor footprint cardinality must remain stable across snapshots")
        if len(self.requested_terminal_actor_positions) != len(self.snapshots[-1].actor_positions):
            raise ValueError("requested terminal footprint must match actor cardinality")
        return self


class ReconfigurableSupportSnapshotReport(FrozenModel):
    snapshot_ref: Ref
    support_union_positions: tuple[Position, ...]
    unsupported_actor_positions: tuple[Position, ...]
    supporting_owner_refs_by_actor_position: tuple[
        tuple[Position, tuple[Ref, ...]], ...
    ]


class ReconfigurableSupportPacketMeasurements(FrozenModel):
    snapshot_reports: tuple[ReconfigurableSupportSnapshotReport, ...]
    support_union_changed: bool
    restoration_positions: tuple[Position, ...]
    initial_final_support_union_equal: bool
    final_actor_fully_supported: bool
    final_connection_present: bool
    final_route_action_refs: tuple[Ref, ...]
    tested_actor_pose_count: int = Field(ge=1, le=4096)
    actor_pose_bound_reached: bool
    final_active_layer_count: int = Field(ge=1, le=32)
    final_multi_owner_actor_position_count: int = Field(ge=0, le=4096)
    schema_version: Ref = "yf_arc3_v5.reconfigurable_support_packet_measurements.v1"


class ReconfigurablePivotCandidate(FrozenModel):
    candidate_ref: Ref
    position: Position
    staging_actor_positions: tuple[Position, ...] = Field(min_length=1, max_length=4096)

    @model_validator(mode="after")
    def validate_staging_positions(self) -> "ReconfigurablePivotCandidate":
        require_unique(self.staging_actor_positions, f"staging positions for {self.candidate_ref}")
        return self


class ReconfigurableRigidGeometryInput(FrozenModel):
    """Two exact masks and caller-supplied pivot candidates."""

    before_positions: tuple[Position, ...] = Field(min_length=1, max_length=4096)
    after_positions: tuple[Position, ...] = Field(min_length=1, max_length=4096)
    pivot_candidates: tuple[ReconfigurablePivotCandidate, ...] = Field(
        min_length=1, max_length=32
    )
    before_support_layers: tuple[ReconfigurableSupportLayer, ...] = Field(
        min_length=1, max_length=32
    )
    after_support_layers: tuple[ReconfigurableSupportLayer, ...] = Field(
        min_length=1, max_length=32
    )
    schema_version: Ref = "yf_arc3_v5.reconfigurable_rigid_geometry_input.v1"

    @model_validator(mode="after")
    def validate_geometry(self) -> "ReconfigurableRigidGeometryInput":
        require_unique(self.before_positions, "before rigid mask positions")
        require_unique(self.after_positions, "after rigid mask positions")
        require_unique(
            tuple(candidate.candidate_ref for candidate in self.pivot_candidates),
            "reconfigurable pivot candidate refs",
        )
        return self


class ReconfigurableRigidFit(FrozenModel):
    candidate_ref: Ref
    quarter_turns_clockwise: int = Field(ge=1, le=3)
    pivot_outside_transformed_masks: bool
    staging_supported_before: bool
    staging_supported_after: bool


class ReconfigurableRigidGeometryMeasurements(FrozenModel):
    exact_translation_deltas: tuple[Position, ...]
    exact_quarter_turn_fits: tuple[ReconfigurableRigidFit, ...]
    tested_pivot_turn_count: int = Field(ge=3, le=96)
    schema_version: Ref = "yf_arc3_v5.reconfigurable_rigid_geometry_measurements.v1"


class ReconfigurableSupportConfiguration(FrozenModel):
    configuration_ref: Ref
    active_layers: tuple[ReconfigurableSupportLayer, ...] = Field(
        min_length=1, max_length=32
    )


class ReconfigurableConnectionDomainInput(FrozenModel):
    """Bounded configurations for one supplied whole-footprint connection."""

    actor_positions: tuple[Position, ...] = Field(min_length=1, max_length=4096)
    requested_terminal_actor_positions: tuple[Position, ...] = Field(
        min_length=1, max_length=4096
    )
    configurations: tuple[ReconfigurableSupportConfiguration, ...] = Field(
        min_length=1, max_length=32
    )
    movement_steps: tuple[ReconfigurableSupportStep, ...] = Field(
        min_length=1, max_length=16
    )
    maximum_tested_actor_poses: int = Field(default=512, ge=1, le=4096)
    maximum_witnesses: int = Field(default=3, ge=1, le=3)
    schema_version: Ref = "yf_arc3_v5.reconfigurable_connection_domain_input.v1"

    @model_validator(mode="after")
    def validate_domain(self) -> "ReconfigurableConnectionDomainInput":
        require_unique(self.actor_positions, "inverse connection actor positions")
        require_unique(
            self.requested_terminal_actor_positions,
            "inverse connection terminal actor positions",
        )
        if len(self.actor_positions) != len(self.requested_terminal_actor_positions):
            raise ValueError("inverse connection footprints must have equal cardinality")
        require_unique(
            tuple(configuration.configuration_ref for configuration in self.configurations),
            "reconfigurable support configuration refs",
        )
        require_unique(
            tuple(step.action_ref for step in self.movement_steps),
            "inverse connection action refs",
        )
        return self


class ReconfigurableConnectionWitness(FrozenModel):
    configuration_ref: Ref
    action_refs: tuple[Ref, ...]
    actor_pose_positions: tuple[tuple[Position, ...], ...]


class ReconfigurableConnectionDomainMeasurements(FrozenModel):
    witnesses: tuple[ReconfigurableConnectionWitness, ...] = Field(max_length=3)
    tested_configuration_count: int = Field(ge=1, le=32)
    tested_actor_pose_count: int = Field(ge=1, le=4096)
    actor_pose_bound_reached: bool
    witness_bound_reached: bool
    schema_version: Ref = "yf_arc3_v5.reconfigurable_connection_domain_measurements.v1"


class DirectedFlowColliderMask(FrozenModel):
    """One source-supplied collider identity and its exact occupied cells."""

    collider_ref: Ref
    positions: tuple[Position, ...] = Field(min_length=1, max_length=4096)
    control_refs: tuple[Ref, ...] = Field(default=(), max_length=32)
    attachment_refs: tuple[Ref, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def validate_mask(self) -> "DirectedFlowColliderMask":
        require_unique(self.positions, f"collider positions for {self.collider_ref}")
        require_unique(self.control_refs, f"collider controls for {self.collider_ref}")
        require_unique(
            self.attachment_refs,
            f"collider attachments for {self.collider_ref}",
        )
        return self


class DirectedFlowColliderUnionInput(FrozenModel):
    collider_masks: tuple[DirectedFlowColliderMask, ...] = Field(
        min_length=1, max_length=32
    )
    schema_version: Ref = "yf_arc3_v5.directed_flow_collider_union_input.v1"

    @model_validator(mode="after")
    def validate_colliders(self) -> "DirectedFlowColliderUnionInput":
        require_unique(
            tuple(mask.collider_ref for mask in self.collider_masks),
            "directed flow collider refs",
        )
        return self


class DirectedFlowColliderUnionMeasurements(FrozenModel):
    union_positions: tuple[Position, ...]
    owner_refs_by_position: tuple[tuple[Position, tuple[Ref, ...]], ...]
    overlap_positions: tuple[Position, ...]
    control_refs: tuple[Ref, ...]
    attachment_refs: tuple[Ref, ...]
    schema_version: Ref = "yf_arc3_v5.directed_flow_collider_union_measurements.v1"


class DirectedFlowFront(FrozenModel):
    front_ref: Ref
    source_ref: Ref
    position: Position
    delta_row: int
    delta_column: int

    @model_validator(mode="after")
    def validate_step(self) -> "DirectedFlowFront":
        if (self.delta_row == 0) == (self.delta_column == 0):
            raise ValueError("directed flow fronts require one cardinal unit axis")
        if abs(self.delta_row) + abs(self.delta_column) != 1:
            raise ValueError("directed flow fronts advance exactly one supplied cell")
        return self


class DirectedFlowTraceCell(FrozenModel):
    position: Position
    lineage_refs: tuple[Ref, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_lineage(self) -> "DirectedFlowTraceCell":
        require_unique(self.lineage_refs, f"lineage refs at {self.position}")
        return self


class DirectedFlowTickSchedule(FrozenModel):
    schedule_ref: Ref
    ordered_front_refs: tuple[Ref, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_order(self) -> "DirectedFlowTickSchedule":
        require_unique(
            self.ordered_front_refs,
            f"front order for {self.schedule_ref}",
        )
        return self


class DirectedFlowJointTickInput(FrozenModel):
    fronts: tuple[DirectedFlowFront, ...] = Field(min_length=2, max_length=32)
    existing_trace: tuple[DirectedFlowTraceCell, ...] = Field(
        default=(), max_length=4096
    )
    schedules: tuple[DirectedFlowTickSchedule, ...] = Field(
        min_length=1, max_length=3
    )
    maximum_resulting_trace_positions: int = Field(default=4096, ge=1, le=4096)
    schema_version: Ref = "yf_arc3_v5.directed_flow_joint_tick_input.v1"

    @model_validator(mode="after")
    def validate_tick(self) -> "DirectedFlowJointTickInput":
        front_refs = tuple(front.front_ref for front in self.fronts)
        require_unique(front_refs, "directed flow front refs")
        require_unique(
            tuple(cell.position for cell in self.existing_trace),
            "directed flow trace positions",
        )
        require_unique(
            tuple(schedule.schedule_ref for schedule in self.schedules),
            "directed flow schedule refs",
        )
        front_set = set(front_refs)
        if any(set(schedule.ordered_front_refs) != front_set for schedule in self.schedules):
            raise ValueError("every tick schedule must contain every front exactly once")
        return self


class DirectedFlowProjectedFront(FrozenModel):
    front_ref: Ref
    source_ref: Ref
    before_position: Position
    after_position: Position
    contacted_preexisting_trace: bool
    contacted_same_tick_trace: bool


class DirectedFlowJointTickSuccessor(FrozenModel):
    schedule_ref: Ref
    projected_fronts: tuple[DirectedFlowProjectedFront, ...]
    resulting_trace: tuple[DirectedFlowTraceCell, ...]
    simultaneous_arrival_positions: tuple[Position, ...]
    preexisting_trace_contact_positions: tuple[Position, ...]
    same_tick_trace_contact_positions: tuple[Position, ...]


class DirectedFlowJointTickMeasurements(FrozenModel):
    successors: tuple[DirectedFlowJointTickSuccessor, ...] = Field(
        max_length=3
    )
    source_ref_count: int = Field(ge=1, le=32)
    front_count: int = Field(ge=2, le=32)
    schedule_count: int = Field(ge=1, le=3)
    alternative_schedule_count: int = Field(ge=0, le=2)
    resulting_trace_bound_reached: bool
    schema_version: Ref = "yf_arc3_v5.directed_flow_joint_tick_measurements.v1"


class DirectedFlowQuantity(FrozenModel):
    quantity_ref: Ref
    value: int = Field(ge=0)


class DirectedFlowRecoveryInput(FrozenModel):
    configuration_before_release: tuple[Position, ...] = Field(max_length=4096)
    configuration_after_recovery: tuple[Position, ...] = Field(max_length=4096)
    trace_after_failed_release: tuple[Position, ...] = Field(max_length=4096)
    trace_after_recovery: tuple[Position, ...] = Field(max_length=4096)
    receptacle_before_release: tuple[Position, ...] = Field(max_length=4096)
    receptacle_after_failed_release: tuple[Position, ...] = Field(max_length=4096)
    receptacle_after_recovery: tuple[Position, ...] = Field(max_length=4096)
    quantities_before_release: tuple[DirectedFlowQuantity, ...] = Field(
        default=(), max_length=32
    )
    quantities_after_recovery: tuple[DirectedFlowQuantity, ...] = Field(
        default=(), max_length=32
    )
    schema_version: Ref = "yf_arc3_v5.directed_flow_recovery_input.v1"

    @model_validator(mode="after")
    def validate_recovery_packet(self) -> "DirectedFlowRecoveryInput":
        for label, positions in (
            ("configuration before release", self.configuration_before_release),
            ("configuration after recovery", self.configuration_after_recovery),
            ("trace after failed release", self.trace_after_failed_release),
            ("trace after recovery", self.trace_after_recovery),
            ("receptacle before release", self.receptacle_before_release),
            ("receptacle after failed release", self.receptacle_after_failed_release),
            ("receptacle after recovery", self.receptacle_after_recovery),
        ):
            require_unique(positions, label)
        before_refs = tuple(item.quantity_ref for item in self.quantities_before_release)
        after_refs = tuple(item.quantity_ref for item in self.quantities_after_recovery)
        require_unique(before_refs, "pre-release directed flow quantities")
        require_unique(after_refs, "post-recovery directed flow quantities")
        if set(before_refs) != set(after_refs):
            raise ValueError("recovery quantities must preserve exact identities")
        return self


class DirectedFlowQuantityDelta(FrozenModel):
    quantity_ref: Ref
    before_value: int = Field(ge=0)
    after_value: int = Field(ge=0)
    signed_delta: int


class DirectedFlowRecoveryMeasurements(FrozenModel):
    retained_configuration_positions: tuple[Position, ...]
    removed_configuration_positions: tuple[Position, ...]
    added_configuration_positions: tuple[Position, ...]
    configuration_exactly_preserved: bool
    cleared_trace_positions: tuple[Position, ...]
    retained_trace_positions: tuple[Position, ...]
    added_trace_positions: tuple[Position, ...]
    receptacle_exactly_restored: bool
    restored_receptacle_positions: tuple[Position, ...]
    quantity_deltas: tuple[DirectedFlowQuantityDelta, ...]
    schema_version: Ref = "yf_arc3_v5.directed_flow_recovery_measurements.v1"


class LocalFieldClickCandidate(FrozenModel):
    candidate_ref: Ref
    position: Position
    strictly_influenced_body_refs: tuple[Ref, ...] = Field(max_length=64)
    boundary_body_refs: tuple[Ref, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def validate_candidate(self) -> "LocalFieldClickCandidate":
        require_unique(
            self.strictly_influenced_body_refs,
            f"strict influence refs for {self.candidate_ref}",
        )
        require_unique(
            self.boundary_body_refs,
            f"boundary refs for {self.candidate_ref}",
        )
        if set(self.strictly_influenced_body_refs) & set(self.boundary_body_refs):
            raise ValueError("a body cannot be strict and boundary-uncertain")
        return self


class LocalFieldSelectiveDomainInput(FrozenModel):
    required_body_refs: tuple[Ref, ...] = Field(min_length=1, max_length=64)
    excluded_body_refs: tuple[Ref, ...] = Field(default=(), max_length=64)
    candidates: tuple[LocalFieldClickCandidate, ...] = Field(
        min_length=1, max_length=256
    )
    schema_version: Ref = "yf_arc3_v5.local_field_selective_domain_input.v1"

    @model_validator(mode="after")
    def validate_domain(self) -> "LocalFieldSelectiveDomainInput":
        require_unique(self.required_body_refs, "required local-field body refs")
        require_unique(self.excluded_body_refs, "excluded local-field body refs")
        require_unique(
            tuple(item.candidate_ref for item in self.candidates),
            "local-field candidate refs",
        )
        if set(self.required_body_refs) & set(self.excluded_body_refs):
            raise ValueError("required and excluded body refs must be disjoint")
        return self


class LocalFieldCandidateMeasurement(FrozenModel):
    candidate_ref: Ref
    position: Position
    missing_required_body_refs: tuple[Ref, ...]
    strictly_included_excluded_body_refs: tuple[Ref, ...]
    boundary_required_body_refs: tuple[Ref, ...]
    boundary_excluded_body_refs: tuple[Ref, ...]


class LocalFieldSelectiveDomainMeasurements(FrozenModel):
    exact_candidate_refs: tuple[Ref, ...]
    boundary_uncertain_candidate_refs: tuple[Ref, ...]
    rejected_candidate_refs: tuple[Ref, ...]
    candidate_measurements: tuple[LocalFieldCandidateMeasurement, ...]
    schema_version: Ref = (
        "yf_arc3_v5.local_field_selective_domain_measurements.v1"
    )


class LocalFieldAutonomousOccurrence(FrozenModel):
    occurrence_ref: Ref
    body_ref: Ref
    event_ref: Ref
    before_position: Position
    after_position: Position


class LocalFieldAutonomousSchedule(FrozenModel):
    schedule_ref: Ref
    ordered_occurrence_refs: tuple[Ref, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_order(self) -> "LocalFieldAutonomousSchedule":
        require_unique(
            self.ordered_occurrence_refs,
            f"local-field schedule order for {self.schedule_ref}",
        )
        return self


class LocalFieldAutonomousSuccessorInput(FrozenModel):
    occurrences: tuple[LocalFieldAutonomousOccurrence, ...] = Field(
        min_length=1, max_length=64
    )
    schedules: tuple[LocalFieldAutonomousSchedule, ...] = Field(
        min_length=1, max_length=3
    )
    already_advanced_occurrence_refs: tuple[Ref, ...] = Field(
        default=(), max_length=64
    )
    schema_version: Ref = "yf_arc3_v5.local_field_autonomous_successor_input.v1"

    @model_validator(mode="after")
    def validate_successors(self) -> "LocalFieldAutonomousSuccessorInput":
        occurrence_refs = tuple(item.occurrence_ref for item in self.occurrences)
        require_unique(occurrence_refs, "local-field autonomous occurrence refs")
        require_unique(
            tuple(item.schedule_ref for item in self.schedules),
            "local-field autonomous schedule refs",
        )
        require_unique(
            self.already_advanced_occurrence_refs,
            "already advanced local-field occurrence refs",
        )
        occurrence_set = set(occurrence_refs)
        if not set(self.already_advanced_occurrence_refs) <= occurrence_set:
            raise ValueError("already advanced refs must name supplied occurrences")
        if any(
            set(schedule.ordered_occurrence_refs) != occurrence_set
            for schedule in self.schedules
        ):
            raise ValueError("every autonomous schedule must contain every occurrence")
        return self


class LocalFieldAutonomousScheduleMeasurement(FrozenModel):
    schedule_ref: Ref
    ordered_occurrence_refs: tuple[Ref, ...]
    resulting_body_positions: tuple[tuple[Ref, Position], ...]


class LocalFieldAutonomousSuccessorMeasurements(FrozenModel):
    schedule_measurements: tuple[LocalFieldAutonomousScheduleMeasurement, ...] = Field(
        max_length=3
    )
    event_refs: tuple[Ref, ...]
    duplicate_advancement_refs: tuple[Ref, ...]
    alternative_schedule_count: int = Field(ge=0, le=2)
    schema_version: Ref = (
        "yf_arc3_v5.local_field_autonomous_successor_measurements.v1"
    )


class LocalFieldAcceptanceSample(FrozenModel):
    sample_ref: Ref
    satisfied_predicate_refs: tuple[Ref, ...] = Field(max_length=64)
    retained_predicate_refs: tuple[Ref, ...] = Field(max_length=64)
    surplus_body_count: int = Field(default=0, ge=0, le=64)
    shared_zone_occupancy_count: int = Field(default=0, ge=0, le=64)
    internal_interruption_observed: bool = False

    @model_validator(mode="after")
    def validate_predicates(self) -> "LocalFieldAcceptanceSample":
        require_unique(
            self.satisfied_predicate_refs,
            f"satisfied predicates for {self.sample_ref}",
        )
        require_unique(
            self.retained_predicate_refs,
            f"retained predicates for {self.sample_ref}",
        )
        if not set(self.retained_predicate_refs) <= set(
            self.satisfied_predicate_refs
        ):
            raise ValueError("retained predicates must also be satisfied")
        return self


class LocalFieldAcceptanceWindowInput(FrozenModel):
    required_predicate_refs: tuple[Ref, ...] = Field(min_length=1, max_length=64)
    samples: tuple[LocalFieldAcceptanceSample, ...] = Field(
        min_length=1, max_length=256
    )
    surplus_admissible: bool
    shared_zone_occupancy_admissible: bool
    schema_version: Ref = "yf_arc3_v5.local_field_acceptance_window_input.v1"

    @model_validator(mode="after")
    def validate_samples(self) -> "LocalFieldAcceptanceWindowInput":
        require_unique(
            self.required_predicate_refs,
            "required local-field acceptance predicates",
        )
        require_unique(
            tuple(item.sample_ref for item in self.samples),
            "local-field acceptance sample refs",
        )
        return self


class LocalFieldAcceptanceWindow(FrozenModel):
    start_sample_ref: Ref
    end_sample_ref: Ref
    sample_count: int = Field(ge=1, le=256)
    retained_through_window: bool


class LocalFieldAcceptanceWindowMeasurements(FrozenModel):
    windows: tuple[LocalFieldAcceptanceWindow, ...] = Field(max_length=256)
    jointly_satisfied_sample_refs: tuple[Ref, ...]
    transient_only_sample_refs: tuple[Ref, ...]
    retained_completion_sample_refs: tuple[Ref, ...]
    first_interruption_sample_ref: Ref | None = None
    schema_version: Ref = (
        "yf_arc3_v5.local_field_acceptance_window_measurements.v1"
    )


class LocalFieldTypedBody(FrozenModel):
    body_ref: Ref
    type_ref: Ref
    positions: tuple[Position, ...] = Field(min_length=1, max_length=4096)

    @model_validator(mode="after")
    def validate_body(self) -> "LocalFieldTypedBody":
        require_unique(self.positions, f"positions for {self.body_ref}")
        return self


class LocalFieldTypedSnapshot(FrozenModel):
    snapshot_ref: Ref
    bodies: tuple[LocalFieldTypedBody, ...] = Field(max_length=128)

    @model_validator(mode="after")
    def validate_body_refs(self) -> "LocalFieldTypedSnapshot":
        require_unique(
            tuple(item.body_ref for item in self.bodies),
            f"body refs for {self.snapshot_ref}",
        )
        return self


class LocalFieldQuantity(FrozenModel):
    quantity_ref: Ref
    value: int = Field(ge=0)


class LocalFieldTransactionInput(FrozenModel):
    before: LocalFieldTypedSnapshot
    transient: LocalFieldTypedSnapshot
    after_packet: tuple[LocalFieldTypedSnapshot, ...] = Field(
        min_length=1, max_length=32
    )
    quantities_before: tuple[LocalFieldQuantity, ...] = Field(
        default=(), max_length=32
    )
    quantities_after: tuple[LocalFieldQuantity, ...] = Field(
        default=(), max_length=32
    )
    schema_version: Ref = "yf_arc3_v5.local_field_transaction_input.v1"

    @model_validator(mode="after")
    def validate_transaction(self) -> "LocalFieldTransactionInput":
        require_unique(
            tuple(item.snapshot_ref for item in self.after_packet),
            "post-event local-field snapshot refs",
        )
        before_refs = tuple(item.quantity_ref for item in self.quantities_before)
        after_refs = tuple(item.quantity_ref for item in self.quantities_after)
        require_unique(before_refs, "local-field quantities before")
        require_unique(after_refs, "local-field quantities after")
        if set(before_refs) != set(after_refs):
            raise ValueError("transaction quantities must preserve exact identities")
        return self


class LocalFieldTypeTransition(FrozenModel):
    body_ref: Ref
    before_type_ref: Ref
    after_type_ref: Ref


class LocalFieldTransactionQuantityDelta(FrozenModel):
    quantity_ref: Ref
    before_value: int = Field(ge=0)
    after_value: int = Field(ge=0)
    signed_delta: int


class LocalFieldTransactionMeasurements(FrozenModel):
    final_added_body_refs: tuple[Ref, ...]
    final_removed_body_refs: tuple[Ref, ...]
    final_type_transitions: tuple[LocalFieldTypeTransition, ...]
    transient_only_body_refs: tuple[Ref, ...]
    restored_body_refs: tuple[Ref, ...]
    bodies_surviving_entire_after_packet: tuple[Ref, ...]
    stable_type_body_refs: tuple[Ref, ...]
    quantity_deltas: tuple[LocalFieldTransactionQuantityDelta, ...]
    schema_version: Ref = "yf_arc3_v5.local_field_transaction_measurements.v1"


class BoundingBox(FrozenModel):
    top: int
    left: int
    bottom: int
    right: int

    @model_validator(mode="after")
    def validate_extents(self) -> "BoundingBox":
        if self.bottom < self.top or self.right < self.left:
            raise ValueError("bounding box extents must be ordered")
        return self

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1

    @property
    def width(self) -> int:
        return self.right - self.left + 1


class FrameGrid(FrozenModel):
    rows: tuple[tuple[int, ...], ...] = Field(min_length=1)
    schema_version: Ref = "yf_arc3_v5.frame_grid.v1"

    def __hash__(self) -> int:
        # This is an in-process cache key, never a canonical source/state digest.
        # A copied model may carry the cache: bind it to both immutable fields
        # so model_copy(update=...) cannot retain a stale raster hash.
        cached = self.__dict__.get("_frame_hash_cache")
        if (
            cached is not None
            and cached[0] is self.rows
            and cached[1] is self.schema_version
        ):
            return cached[2]
        value = FrozenMap.from_frozen_items({
            "rows": self.rows,
            "schema_version": self.schema_version,
        }).__hash__()
        object.__setattr__(self, "_frame_hash_cache", (self.rows, self.schema_version, value))
        return value

    @model_validator(mode="after")
    def validate_rectangle(self) -> "FrameGrid":
        width = len(self.rows[0])
        if width == 0 or any(len(row) != width for row in self.rows):
            raise ValueError("frame must be a non-empty rectangular matrix")
        return self

    @property
    def height(self) -> int:
        return len(self.rows)

    @property
    def width(self) -> int:
        return len(self.rows[0])


class FrameNormalizationInput(FrozenModel):
    frames: tuple[FrameGrid, ...] = Field(min_length=1)
    schema_version: Ref = "yf_arc3_v5.frame_normalization_input.v1"

    @classmethod
    def from_payload(cls, payload: Any) -> "FrameNormalizationInput":
        """Accept one numeric matrix or a non-empty stack of numeric matrices."""

        if not _is_sequence(payload) or not payload:
            raise ValueError("expected a non-empty 2D frame or 3D frame stack")
        first = payload[0]
        if not _is_sequence(first) or not first:
            raise ValueError("expected a non-empty 2D frame or 3D frame stack")
        first_value = first[0]
        raw_frames = payload if _is_sequence(first_value) else (payload,)
        frames = tuple(FrameGrid(rows=_coerce_matrix(frame)) for frame in raw_frames)
        return cls(frames=frames)


class FrameNormalizationResult(FrozenModel):
    frame: FrameGrid
    transition_frames: tuple[FrameGrid, ...] = ()
    source_frame_count: int = Field(ge=1)
    schema_version: Ref = "yf_arc3_v5.frame_normalization_result.v2"

    @model_validator(mode="after")
    def validate_source_frame_count(self) -> "FrameNormalizationResult":
        if self.source_frame_count != len(self.transition_frames) + 1:
            raise ValueError(
                "source frame count must include every transition frame and the final frame"
            )
        return self


class ComponentDescription(FrozenModel):
    component_id: Ref
    value: int
    pixels: tuple[Position, ...] = Field(min_length=1)
    relative_pixels: tuple[Position, ...] = Field(min_length=1)
    bbox: BoundingBox
    area: int = Field(ge=1)
    touches_frame_boundary: bool
    schema_version: Ref = "yf_arc3_v5.component_description.v1"

    @model_validator(mode="after")
    def validate_geometry(self) -> "ComponentDescription":
        if self.area != len(self.pixels) or self.area != len(self.relative_pixels):
            raise ValueError("component area must equal its pixel counts")
        if len(set(self.pixels)) != len(self.pixels):
            raise ValueError("component pixels must be unique")
        expected_relative = tuple(
            (row - self.bbox.top, col - self.bbox.left) for row, col in self.pixels
        )
        if self.relative_pixels != expected_relative:
            raise ValueError("relative pixels must be derived from the component bbox")
        return self


class ComponentExtractionInput(FrozenModel):
    frame: FrameGrid
    connectivity: int = 4
    schema_version: Ref = "yf_arc3_v5.component_extraction_input.v1"

    @model_validator(mode="after")
    def validate_connectivity(self) -> "ComponentExtractionInput":
        if self.connectivity not in {4, 8}:
            raise ValueError("connectivity must be 4 or 8")
        return self


class ComponentExtractionResult(FrozenModel):
    components: tuple[ComponentDescription, ...]
    schema_version: Ref = "yf_arc3_v5.component_extraction_result.v1"


class PeriodicCellGridInput(FrozenModel):
    frame: FrameGrid
    maximum_pitch: int = Field(default=16, ge=2, le=32)
    maximum_candidates: int = Field(default=3, ge=1, le=8)
    action_conditioned_grid_evidence: bool = True
    observed_translation_quantum: int = Field(default=0, ge=0, le=32)
    schema_version: Ref = "yf_arc3_v5.periodic_cell_grid_input.v1"


class PeriodicCellDescription(FrozenModel):
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    bbox: BoundingBox
    point: Position
    pattern_ref: Ref
    palette_value_count: int = Field(ge=1)
    palette_values: tuple[int, ...] = Field(default=(), max_length=64)
    pattern_occurrence_count: int = Field(ge=1)
    schema_version: Ref = "yf_arc3_v5.periodic_cell_description.v1"


class ClippedPeriodicCell(FrozenModel):
    """Visible crop of an existing lattice position, never a new complete type."""

    row: int
    col: int
    bbox: BoundingBox
    crop_offset: Position
    compatible_pattern_refs: tuple[Ref, ...] = ()


class PeriodicCellGridCandidate(FrozenModel):
    candidate_ref: Ref
    row_offset: int = Field(ge=0)
    col_offset: int = Field(ge=0)
    cell_height: int = Field(ge=1)
    cell_width: int = Field(ge=1)
    row_gap: int = Field(ge=0)
    col_gap: int = Field(ge=0)
    row_pitch: int = Field(ge=2)
    col_pitch: int = Field(ge=2)
    logical_rows: int = Field(ge=2)
    logical_columns: int = Field(ge=2)
    separator_consistency_ppm: int = Field(ge=0, le=1_000_000)
    row_boundary_contrast_ppm: int = Field(default=0, ge=0, le=1_000_000)
    col_boundary_contrast_ppm: int = Field(default=0, ge=0, le=1_000_000)
    row_proper_divisor_boundary_gain_line_ppm: int = Field(default=0, ge=0)
    col_proper_divisor_boundary_gain_line_ppm: int = Field(default=0, ge=0)
    repeated_pattern_count: int = Field(ge=0)
    distinct_pattern_count: int = Field(ge=1)
    cells: tuple[PeriodicCellDescription, ...] = Field(min_length=4)
    clipped_cells: tuple[ClippedPeriodicCell, ...] = ()
    visible_row_range: tuple[int, int] | None = None
    visible_column_range: tuple[int, int] | None = None
    schema_version: Ref = "yf_arc3_v5.periodic_cell_grid_candidate.v1"

    @model_validator(mode="after")
    def validate_cells(self) -> "PeriodicCellGridCandidate":
        if len(self.cells) != self.logical_rows * self.logical_columns:
            raise ValueError("periodic grid cells must fill the logical rectangle")
        return self


class PeriodicCellGridResult(FrozenModel):
    candidates: tuple[PeriodicCellGridCandidate, ...]
    enumeration_truncated: bool = False
    schema_version: Ref = "yf_arc3_v5.periodic_cell_grid_result.v1"


class KnownGridCellRecountInput(FrozenModel):
    """Exact raster recount on a geometry already supplied by DRM/SRC."""

    frame: FrameGrid
    grid_ref: Ref
    row_offset: int = Field(ge=0)
    col_offset: int = Field(ge=0)
    cell_height: int = Field(ge=1)
    cell_width: int = Field(ge=1)
    row_pitch: int = Field(ge=1)
    col_pitch: int = Field(ge=1)
    logical_rows: int = Field(ge=1)
    logical_columns: int = Field(ge=1)
    prior_type_patterns: tuple[tuple[tuple[int, ...], ...], ...] = ()
    prior_cell_type_ids: tuple[int, ...] = ()
    prior_separator_values: tuple[int, ...] = ()
    schema_version: Ref = "yf_arc3_v5.known_grid_cell_recount_input.v1"

    @model_validator(mode="after")
    def validate_geometry(self) -> "KnownGridCellRecountInput":
        cell_count = self.logical_rows * self.logical_columns
        if cell_count > 4096:
            raise ValueError("known grid cell count exceeds hard bound")
        if self.row_pitch < self.cell_height or self.col_pitch < self.cell_width:
            raise ValueError("known grid cells may not overlap")
        span_height = (self.logical_rows - 1) * self.row_pitch + self.cell_height
        span_width = (self.logical_columns - 1) * self.col_pitch + self.cell_width
        separator_count = (
            span_height * span_width
            - cell_count * self.cell_height * self.cell_width
        )
        if self.prior_separator_values and len(self.prior_separator_values) != separator_count:
            raise ValueError("prior separator count differs from geometry")
        if (
            self.row_offset + (self.logical_rows - 1) * self.row_pitch
            + self.cell_height > self.frame.height
            or self.col_offset + (self.logical_columns - 1) * self.col_pitch
            + self.cell_width > self.frame.width
        ):
            raise ValueError("known grid geometry no longer fits current frame")
        if self.prior_cell_type_ids:
            if len(self.prior_cell_type_ids) != cell_count:
                raise ValueError("prior grid cell count differs from geometry")
            if not self.prior_type_patterns or any(
                type_id < 0 or type_id >= len(self.prior_type_patterns)
                for type_id in self.prior_cell_type_ids
            ):
                raise ValueError("prior grid type index is invalid")
        elif self.prior_type_patterns:
            raise ValueError("prior patterns require prior cell type ids")
        return self


class KnownGridCellRecountResult(FrozenModel):
    """Compact exact types, occurrences, and changed logical cells."""

    grid_ref: Ref
    type_patterns: tuple[tuple[tuple[int, ...], ...], ...]
    cell_type_ids: tuple[int, ...]
    type_occurrences: tuple[int, ...]
    changed_cell_indexes: tuple[int, ...]
    separator_values: tuple[int, ...] = ()
    separator_values_changed: bool = False
    unique_multivalue_cell_indexes: tuple[int, ...] = ()
    schema_version: Ref = "yf_arc3_v5.known_grid_cell_recount_result.v1"


class KnownGridObjectAssemblyInput(FrozenModel):
    """Exact retained-grid cells supplied for bounded mechanical grouping."""

    recount: KnownGridCellRecountResult
    logical_rows: int = Field(ge=1)
    logical_columns: int = Field(ge=1)
    schema_version: Ref = "yf_arc3_v5.known_grid_object_assembly_input.v1"

    @model_validator(mode="after")
    def validate_cell_count(self) -> "KnownGridObjectAssemblyInput":
        if self.logical_rows * self.logical_columns != len(
            self.recount.cell_type_ids
        ):
            raise ValueError("retained-grid dimensions differ from cell count")
        return self


class KnownGridCellObject(FrozenModel):
    """One orthogonally connected exact cell-type component."""

    object_ref: Ref
    cell_type_id: int = Field(ge=0)
    member_cell_indexes: tuple[int, ...] = Field(min_length=1, max_length=4096)
    top: int = Field(ge=0)
    left: int = Field(ge=0)
    height: int = Field(ge=1)
    width: int = Field(ge=1)
    shape_class_ref: Ref
    visibility: Literal["visible", "partially_hidden", "hidden"]
    schema_version: Ref = "yf_arc3_v5.known_grid_cell_object.v1"


class KnownGridShapeClass(FrozenModel):
    """Exact normalized shape index; relations never enumerate object pairs."""

    shape_class_ref: Ref
    normalized_cells: tuple[Position, ...] = Field(min_length=1, max_length=4096)
    object_refs: tuple[Ref, ...] = Field(min_length=1, max_length=256)
    horizontal_reflection_class_ref: Ref | None = None
    vertical_reflection_class_ref: Ref | None = None
    schema_version: Ref = "yf_arc3_v5.known_grid_shape_class.v1"


class KnownGridObjectAssemblyResult(FrozenModel):
    """Bounded complete object inventory plus linear exact-shape indexes."""

    grid_ref: Ref
    logical_rows: int = Field(ge=1)
    logical_columns: int = Field(ge=1)
    objects: tuple[KnownGridCellObject, ...] = Field(max_length=256)
    shape_classes: tuple[KnownGridShapeClass, ...] = Field(max_length=256)
    singleton_object_count: int = Field(ge=0, le=256)
    multicell_object_count: int = Field(ge=0, le=256)
    schema_version: Ref = "yf_arc3_v5.known_grid_object_assembly_result.v1"


class KnownGridCellRecountDelta(FrozenModel):
    """Exact changed-cell/separator payload against one retained grid result."""

    grid_ref: Ref
    cell_count: int = Field(ge=1, le=4096)
    separator_count: int = Field(ge=0, le=4096)
    changed_cells: tuple[tuple[int, tuple[tuple[int, ...], ...]], ...] = ()
    changed_separators: tuple[tuple[int, int], ...] = ()
    schema_version: Ref = "yf_arc3_v5.known_grid_cell_recount_delta.v1"


class VisualSceneInput(FrozenModel):
    frame: FrameGrid
    action_conditioned_grid_evidence: bool = True
    # Existing retained geometry is an immutable transport fact. When
    # supplied by the declared block-grid scope, perception recounts that
    # geometry instead of rediscovering every lattice candidate.
    retained_grid_geometry: tuple[Any, ...] | None = None
    retained_grid_separator_consistency_ppm: int = Field(
        default=0, ge=0, le=1_000_000
    )
    schema_version: Ref = "yf_arc3_v5.visual_scene_input.v1"


class VisualSpriteDescription(FrozenModel):
    """One exact crop candidate; `sprite` is descriptive, never a role."""

    sprite_id: Ref
    anchor_component_ref: Ref
    bbox: BoundingBox
    pattern: tuple[tuple[int, ...], ...] = Field(min_length=1)
    member_component_refs: tuple[Ref, ...] = Field(min_length=1)
    member_zone_refs: tuple[Ref, ...] = Field(min_length=1)
    color_histogram: tuple[tuple[int, int], ...] = Field(min_length=1)
    internal_component_count: int = Field(ge=0)
    pattern_occurrence_count: int = Field(ge=1)
    exact_dezoom_factors: tuple[int, ...] = ()
    touches_frame_boundary: bool
    schema_version: Ref = "yf_arc3_v5.visual_sprite_description.v1"

    @model_validator(mode="after")
    def validate_crop(self) -> "VisualSpriteDescription":
        if len(self.pattern) != self.bbox.height or any(
            len(row) != self.bbox.width for row in self.pattern
        ):
            raise ValueError("sprite pattern must exactly fill its bounding box")
        if self.anchor_component_ref not in self.member_component_refs:
            raise ValueError("sprite anchor must remain one of its component members")
        require_unique(self.member_component_refs, "sprite component members")
        require_unique(self.member_zone_refs, "sprite zone members")
        if self.internal_component_count != len(self.member_component_refs) - 1:
            raise ValueError("sprite internal count must derive from component members")
        if tuple(sorted(self.exact_dezoom_factors)) != self.exact_dezoom_factors:
            raise ValueError("sprite exact dezoom factors must be sorted")
        return self


class HoleSpriteDescription(FrozenModel):
    """Exact valued content of one enclosed complement region."""

    hole_sprite_id: Ref
    enclosing_component_ref: Ref
    bbox: BoundingBox
    pixels: tuple[Position, ...] = Field(min_length=1)
    value_count: int = Field(ge=1)
    schema_version: Ref = "yf_arc3_v5.hole_sprite_description.v1"

    @model_validator(mode="after")
    def validate_geometry(self) -> "HoleSpriteDescription":
        require_unique(self.pixels, "hole sprite pixels")
        if any(
            row < self.bbox.top
            or row > self.bbox.bottom
            or col < self.bbox.left
            or col > self.bbox.right
            for row, col in self.pixels
        ):
            raise ValueError("hole sprite pixels must lie inside its bounding box")
        return self


class BlockGridSceneDescription(FrozenModel):
    """Exact measured subset; omitted object families remain unknown."""

    frame: FrameGrid
    blocks: ComponentExtractionResult
    periodic_cell_grids: PeriodicCellGridResult
    frame_enclosing_component_refs: tuple[Ref, ...] = ()
    hole_sprite_enumeration_pruned_by_periodic_partition: bool
    schema_version: Ref = "yf_arc3_v5.block_grid_scene_description.v1"

    @model_validator(mode="after")
    def validate_block_refs(self) -> "BlockGridSceneDescription":
        block_refs = {item.component_id for item in self.blocks.components}
        require_unique(
            self.frame_enclosing_component_refs,
            "frame-enclosing component references",
        )
        if not set(self.frame_enclosing_component_refs) <= block_refs:
            raise ValueError("frame-enclosing measurements must reference blocks")
        return self


class VisualSceneDescription(FrozenModel):
    frame: FrameGrid
    blocks: ComponentExtractionResult
    zones: ComponentExtractionResult
    sprites: tuple[VisualSpriteDescription, ...]
    hole_sprites: tuple[HoleSpriteDescription, ...] = ()
    hole_sprite_measurement_count: int = Field(default=0, ge=0)
    hole_sprite_enumeration_truncated: bool = False
    periodic_cell_grids: PeriodicCellGridResult = Field(
        default_factory=lambda: PeriodicCellGridResult(candidates=())
    )
    frame_enclosing_component_refs: tuple[Ref, ...] = ()
    schema_version: Ref = "yf_arc3_v5.visual_scene_description.v3"

    @model_validator(mode="after")
    def validate_sprite_projection(self) -> "VisualSceneDescription":
        block_refs = {item.component_id for item in self.blocks.components}
        require_unique(
            self.frame_enclosing_component_refs,
            "frame-enclosing component references",
        )
        if not set(self.frame_enclosing_component_refs) <= block_refs:
            raise ValueError("frame-enclosing measurements must reference blocks")
        if any(
            sprite.anchor_component_ref in self.frame_enclosing_component_refs
            for sprite in self.sprites
        ):
            raise ValueError("a frame-enclosing component cannot be projected as a sprite")
        require_unique(
            tuple(item.hole_sprite_id for item in self.hole_sprites),
            "hole sprite references",
        )
        if self.hole_sprite_measurement_count < len(self.hole_sprites):
            raise ValueError("hole sprite count cannot be smaller than emitted items")
        if self.hole_sprite_enumeration_truncated != (
            self.hole_sprite_measurement_count > len(self.hole_sprites)
        ):
            raise ValueError("hole sprite truncation must derive from total and emitted counts")
        return self


class MultiResolutionViewInput(FrozenModel):
    scene: VisualSceneDescription
    observation_ref: Ref
    frame_ref: Ref | None = None
    maximum_descriptions_per_family: int = Field(default=16, ge=1, le=16)
    maximum_distinct_visual_elements: int = Field(default=256, ge=1, le=512)
    maximum_distinct_visual_element_pair_relations: int = Field(
        default=64, ge=1, le=4096
    )
    maximum_distinct_visual_element_repeated_pair_thin_relations: int = Field(
        default=256, ge=1, le=1024
    )
    is_initial_observation: bool = False
    available_action_refs: tuple[Ref, ...] = ()
    schema_version: Ref = "yf_arc3_v5.multi_resolution_view_input.v3"

    @model_validator(mode="after")
    def validate_available_action_refs(self) -> "MultiResolutionViewInput":
        require_unique(self.available_action_refs, "available action references")
        return self


class MultiResolutionViewMeasurements(FrozenModel):
    """Compact geometry summaries; semantic interpretation belongs to DRM."""

    descriptive_facts: FrozenMap
    measurement_refs: tuple[Ref, ...]
    description_truncated: bool = False
    schema_version: Ref = "yf_arc3_v5.multi_resolution_view_measurements.v2"

    @model_validator(mode="after")
    def validate_measurement_refs(self) -> "MultiResolutionViewMeasurements":
        require_unique(self.measurement_refs, "multi-resolution measurement references")
        return self


class InteractionProbeCandidate(FrozenModel):
    """One descriptive, still-uncommitted environment experiment."""

    candidate_ref: Ref
    action_ref: Ref
    action_data: FrozenMap = Field(default_factory=FrozenMap)
    component_ref: Ref | None = None
    sprite_ref: Ref | None = None
    zone_refs: tuple[Ref, ...] = ()
    point: Position | None = None
    basis_refs: tuple[Ref, ...] = ()
    expectation_refs: tuple[Ref, ...] = ()
    schema_version: Ref = "yf_arc3_v5.interaction_probe_candidate.v1"


class ViewportContactProjection(FrozenModel):
    """One bounded, mechanical projection of an observed contact cell."""

    contact_pattern_ref: Ref
    source_transition_ref: Ref
    projected_row: int
    projected_col: int
    viewport_shift_support_count: int = Field(ge=1)
    supporting_transition_refs: tuple[Ref, ...] = Field(min_length=1, max_length=32)
    schema_version: Ref = "yf_arc3_v5.viewport_contact_projection.v1"


class InteractionProbeInput(FrozenModel):
    canonical_instance_sequence_facts: tuple[FrozenMap, ...] = Field(default=(), max_length=3)
    canonical_temporal_contact_inventory_facts: tuple[FrozenMap, ...] = Field(default=(), max_length=1)
    canonical_contact_source_pattern_method_facts: tuple[FrozenMap, ...] = Field(default=(), max_length=64)
    canonical_phase_slot_method_facts: tuple[FrozenMap, ...] = Field(default=(), max_length=64)
    # Canonical conditional dependencies, not a creator-side route or oracle.
    canonical_relational_reception_facts: tuple[FrozenMap, ...] = Field(default=(), max_length=3)
    canonical_relational_transport_facts: tuple[FrozenMap, ...] = Field(default=(), max_length=3)
    canonical_relational_lattice_facts: tuple[FrozenMap, ...] = Field(default=(), max_length=3)
    relational_reception_measurement_contract: FrozenMap = Field(default_factory=FrozenMap)
    linked_ordered_family_measurement_contract: FrozenMap = Field(default_factory=FrozenMap)
    oriented_contour_network_contract: FrozenMap = Field(default_factory=FrozenMap)
    canonical_reference_write_facts: tuple[FrozenMap, ...] = Field(default=(), max_length=16)
    canonical_reference_goal_facts: tuple[FrozenMap, ...] = Field(default=(), max_length=64)
    frame: FrameGrid
    # Same established DRM sprite/face constraints as the verified route input.
    established_hostile_contact_pattern_faces: tuple[tuple[Ref, str], ...] = Field(
        default=(), max_length=16
    )
    canonical_peer_method_facts: tuple[FrozenMap, ...] = Field(default=(), max_length=16)
    canonical_contact_transport_facts: tuple[FrozenMap, ...] = Field(default=(), max_length=64)
    canonical_terminal_inventory_facts: tuple[FrozenMap, ...] = Field(default=(), max_length=64)
    canonical_method_inventory_facts: tuple[FrozenMap, ...] = Field(default=(), max_length=64)
    canonical_recipe_operator_facts: tuple[FrozenMap, ...] = Field(default=(), max_length=64)
    canonical_entry_relation_facts: tuple[FrozenMap, ...] = Field(default=(), max_length=64)
    canonical_recipe_edit_facts: tuple[FrozenMap, ...] = Field(default=(), max_length=64)
    canonical_navigation_method_facts: tuple[FrozenMap, ...] = Field(default=(), max_length=16)
    canonical_navigation_association_facts: tuple[FrozenMap, ...] = Field(default=(), max_length=64)
    navigation_dependency_contract: FrozenMap = Field(default_factory=FrozenMap)
    rigid_marker_pose_contract: FrozenMap = Field(default_factory=FrozenMap)
    rigid_marker_epoch: Ref | None = None
    rigid_marker_frame_ref: Ref | None = None
    canonical_rigid_marker_target_facts: tuple[FrozenMap, ...] = Field(default=(), max_length=8)
    canonical_rigid_marker_body_facts: tuple[FrozenMap, ...] = Field(default=(), max_length=8)
    canonical_unselected_control_facts: tuple[FrozenMap, ...] = Field(default=(), max_length=64)
    isolated_reasoning_scope_ref: Ref | None = None
    boundary_search_band_px: int = Field(default=7, ge=1, le=64)
    # Bounded raster measurements only; never raw animation frames or decisions.
    ordered_square_packets: tuple[Any, ...] = Field(default=(), max_length=4)
    layer_transition_measurements: tuple[Any, ...] = Field(default=(), max_length=32)
    axial_packet_measurements: tuple[Any, ...] = Field(default=(), max_length=32)
    periodic_last_delta: Any | None = None
    reflection_last_delta: Any | None = None
    concentric_frame_additions: tuple[tuple[int, int, int, int, int], ...] = Field(default=(), max_length=32)
    observed_level_index: int | None = Field(default=None, ge=1)
    diagram_left_removal_right_appearance_count: int = Field(default=0, ge=0, le=32)
    contour_replacement_delta: Any | None = None
    # Exact raster currently reported by the environment.  ``frame`` may be a
    # deliberately retained semantic reference frame during a bounded cursor;
    # capabilities that explicitly measure current geometry may use this fact.
    current_observed_frame: FrameGrid | None = None
    # Exact prior observed supports transported by a declared route, not targets
    # or inferred geometry. Current pixels outside control overlays revalidate them.
    prior_observed_open_perimeter_components: tuple[ComponentDescription, ...] = Field(
        default=(), max_length=16
    )
    # Exact closed perimeter supports observed earlier in the current scene.
    # This is bounded geometry memory only: DRM decides whether a support is a
    # destination.  Current pixels outside measured mover overlays must still
    # revalidate it, so a real structural change falsifies the remembered shape.
    prior_observed_exact_perimeter_components: tuple[
        ComponentDescription, ...
    ] = Field(default=(), max_length=16)
    prior_capture_receiver_bbox: tuple[int, int, int, int] | None = None
    prior_capture_request_pixels: tuple[Position, ...] = Field(default=(), max_length=256)
    prior_capture_request_peer_count: int = Field(default=0, ge=0)
    prior_capture_request_peer_bbox: tuple[int, int, int, int] | None = None
    prior_capture_stationary_action_count: int = Field(default=0, ge=0, le=32)
    # Same-game, cross-level morphology only.  No bbox, coordinate, component
    # identity, route or action survives in this signature.
    transmitted_input_aligned_corner_signatures: tuple[FrozenMap, ...] = Field(
        default=(), max_length=8
    )
    # Every causally associated same-game role keeps all observed color+shape
    # signatures across a level boundary. Coordinates, component ids and routes
    # are deliberately absent; the next level must relocate matching bearers.
    transmitted_level_role_signatures: tuple[FrozenMap, ...] = Field(
        default=(), max_length=24
    )
    # Bounded previous-frame measurements for detecting one action-conditioned
    # marker-pair absorption. These are mechanical counts/residuals only; they
    # carry no palette role, objective, or action choice.
    prior_bounded_marker_pair_cover_candidate_count: int = Field(
        default=0, ge=0, le=3
    )
    prior_bounded_marker_pair_cover_marker_occurrence_count: int = Field(
        default=0, ge=0, le=24
    )
    prior_bounded_marker_pair_cover_edge_count: int = Field(default=0, ge=0, le=6)
    prior_bounded_marker_pair_cover_satisfied_edge_count: int = Field(
        default=0, ge=0, le=6
    )
    prior_bounded_marker_pair_cover_unique: bool = False
    prior_bounded_marker_pair_pose_candidate_count: int = Field(
        default=0, ge=0, le=3
    )
    prior_bounded_marker_pair_residual_row_twice: int = Field(
        default=0, ge=-256, le=256
    )
    prior_bounded_marker_pair_residual_col_twice: int = Field(
        default=0, ge=-256, le=256
    )
    prior_bounded_marker_pair_rotation_degrees: int = Field(
        default=0, ge=0, le=270
    )
    # Bounded process handoff across reconciled focus and pose primitives.
    # Exact boxes rebind the declared moving/fixed edge to current components;
    # pose state keeps that process open while its structural edge remains valid.
    # No action identity or payload is retained as continuation state.
    prior_bounded_marker_pair_cover_focus_transfer_required: bool = False
    prior_bounded_marker_pair_cover_moving_bbox: (
        tuple[int, int, int, int] | None
    ) = None
    prior_bounded_marker_pair_cover_fixed_bbox: (
        tuple[int, int, int, int] | None
    ) = None
    # A repeated projection of one immutable observation revalidates but must
    # not apply the same transition delta to the process a second time.
    prior_bounded_marker_pair_measurement_is_current_observation: bool = False
    # Exact visual signatures established only by an action-conditioned marker
    # absorption. They suppress consequence rasters from becoming new markers;
    # no palette value has consequence meaning before that causal transition.
    established_marker_pair_consequence_signature_digests: tuple[str, ...] = ()
    prior_oriented_square_transport_capacity_bbox: (
        tuple[int, int, int, int] | None
    ) = None
    canonical_external_transport_records: tuple[FrozenMap, ...] = Field(default=(), max_length=8)
    current_observation_frame_ref: Ref | None = None
    # Structural grid hypotheses become action-conditioned only after the
    # first observed transition.  Direct capability fixtures default to the
    # post-transition mode; the live controller sets the initial frame false.
    action_conditioned_grid_evidence: bool = True
    components: ComponentExtractionResult
    zones: ComponentExtractionResult | None = None
    sprites: tuple[VisualSpriteDescription, ...] = ()
    # Full bounded sprite measurements remain available to mechanisms whose
    # DRM gate is structural and directional even when ``sprites`` is scoped
    # to point-candidate enumeration.  This is descriptive evidence only; it
    # neither creates candidates nor assigns a role.
    structural_sprites: tuple[VisualSpriteDescription, ...] | None = None
    # Bounded descriptive snapshot from the frame immediately before the last
    # point action.  It carries no role or selection: interaction measurement
    # may compare translation-invariant sprite signatures and DRM alone decides
    # what a recurrence means.
    prior_point_frame_sprites: tuple[VisualSpriteDescription, ...] = Field(
        default=(), max_length=128
    )
    prior_point_frame_zones: ComponentExtractionResult | None = None
    hole_sprites: tuple[HoleSpriteDescription, ...] = ()
    prior_context_effect_carried_reflection_axis_observed: bool = False
    # Canonical DRM CLAIM evidence that an action-conditioned appearance
    # transfer previously identified a change of controlled support.  The
    # capability transports this fact but never assigns selection meaning.
    action_conditioned_control_focus_transfer_supported: bool = False
    # Canonical DRM-owned connector evidence from the current level only.
    # Python transports this fact; DRM decides whether it changes probe order.
    current_context_observed_connector_activation_supported: bool = False
    # Canonical DRM-owned evidence that a connector changed state in this
    # level.  Activation is included, while a local path substitution keeps
    # the blocked/passing direction unresolved.
    current_context_observed_connector_state_transition_supported: bool = False
    # Bounded geometry for one uniquely outlined member of a horizontal row.
    # Python transports only exact rank and normalized-shape measurements;
    # declared knowledge decides whether they constitute a reading focus.
    horizontal_contoured_member_row_count: int = Field(default=0, ge=0, le=3)
    horizontal_contoured_member_count: int = Field(default=0, ge=0, le=8)
    horizontal_contoured_member_distinct_shape_count: int = Field(
        default=0, ge=0, le=8
    )
    horizontal_member_order_digest: str | None = None
    outlined_member_ordinal: int | None = Field(default=None, ge=0, le=7)
    outlined_member_shape_digest: str | None = None
    canonical_ordered_focus_action_claim_facts: tuple[FrozenMap, ...] = Field(
        default=(), max_length=32
    )
    canonical_ordered_member_action_schema_claim_facts: tuple[
        FrozenMap, ...
    ] = Field(default=(), max_length=16)
    prior_context_effect_carried_reflection_axis_carrier_values: tuple[int, ...] = (
        ()
    )
    prior_context_effect_carried_reflection_axis_quanta: tuple[int, ...] = ()
    # Thin transport of the support identity committed by the preceding
    # declarative reflection plan; measurement may preserve it but cannot
    # originate or prioritize a new support value from this field.
    prior_context_orthogonal_reflection_support_values: tuple[int, ...] = ()
    # Same-game provisional target-layer attribute value authored by the
    # preceding DRM commitment and revalidated as a query criterion per scene.
    prior_game_episode_orthogonal_reflection_support_values: tuple[int, ...] = ()
    # Thin transport of the direct body identity already committed by DRM.
    # Measurement may preserve this palette value across an active residual
    # series but cannot originate a source role from colour alone.
    prior_context_orthogonal_reflection_direct_body_values: tuple[int, ...] = ()
    # Current-game episode memory of a DRM-confirmed attribute equivalence.
    # It is a provisional query criterion only: current-level operational
    # commitment remains empty until the scene independently revalidates it.
    prior_game_episode_orthogonal_reflection_direct_body_values: tuple[int, ...] = ()
    # Translation-invariant logical-cell morphology of the direct body most
    # recently committed by DRM.  Python transports this bounded measurement;
    # the declared focus-transfer workflow decides when it excludes that
    # completed identity from the remaining-body frontier.
    prior_context_orthogonal_reflection_direct_body_normalized_cell_offsets: tuple[
        Position, ...
    ] = Field(default=(), max_length=128)
    # Bounded translation-invariant morphologies measured for every member of
    # the current level's DRM-confirmed direct-body role query.  They preserve
    # layer identity when equal-value bodies later become adjacent or overlap.
    prior_context_orthogonal_reflection_direct_body_morphology_collections: tuple[
        tuple[Position, ...], ...
    ] = Field(default=(), max_length=16)
    # Canonical DRM claim arguments transported from prior successful level
    # boundaries.  They remain provisional here; a current candidate must
    # explicitly bind the same terminal objective before DRM may prioritize it.
    prior_context_transmitted_terminal_objective_contract_refs: tuple[Ref, ...] = (
        Field(default=(), max_length=32)
    )
    # Canonical boundary CLAIM references for recurrent activity schemas. They
    # are only transport evidence: the current scene must independently
    # revalidate a DRM-authored workflow and one of its open activities.
    prior_context_transmitted_workflow_activity_schema_refs: tuple[Ref, ...] = (
        Field(default=(), max_length=32)
    )
    prior_context_transmitted_workflow_activity_refs: tuple[Ref, ...] = Field(
        default=(), max_length=64
    )
    # Previous pre-action measurement transported from the last dispatched
    # alternative in this level epoch.  Python compares counts only; DRM owns
    # whether an unchanged objective measure makes an action family ineligible.
    prior_dispatched_orthogonal_reflection_uncovered_target_cell_count: (
        int | None
    ) = Field(default=None, ge=0)
    # Mechanical upper bound for cycling among the measured reflection carriers
    # and compatible movable bodies until the declared focus morphology binds.
    orthogonal_reflection_focus_transfer_cycle_bound: int = Field(
        default=0, ge=0, le=8
    )
    # Number of consecutive focus-transfer primitives already reconciled in
    # the current selector series.  Python uses it only to reduce the current
    # scene's mechanically measured finite cycle capacity.
    orthogonal_reflection_recent_focus_transfer_count: int = Field(
        default=0, ge=0, le=8
    )
    prior_context_orthogonal_reflection_completed_cells: tuple[Position, ...] = (
        ()
    )
    prior_context_orthogonal_reflection_row_phase: int | None = None
    prior_context_orthogonal_reflection_col_phase: int | None = None
    prior_context_orthogonal_reflection_row_axis_coordinate_twice: int | None = None
    prior_context_orthogonal_reflection_col_axis_coordinate_twice: int | None = None
    # Last carrier coordinates measured while both declared axis residuals were
    # zero.  These are immutable geometry facts used to survive temporary
    # rendering occlusion; they do not assign an axis role or release actions.
    prior_context_orthogonal_reflection_carrier_row_axis_coordinate_twice: (
        int | None
    ) = None
    prior_context_orthogonal_reflection_carrier_col_axis_coordinate_twice: (
        int | None
    ) = None
    prior_context_orthogonal_reflection_axis_zero_count: int = Field(
        default=0, ge=0, le=2
    )
    transient_change_bboxes: tuple[BoundingBox, ...] = Field(
        default=(), max_length=3
    )
    action_conditioned_appeared_bboxes: tuple[BoundingBox, ...] = Field(
        default=(), max_length=3
    )
    periodic_cell_grids: PeriodicCellGridResult = Field(
        default_factory=lambda: PeriodicCellGridResult(candidates=())
    )
    # Exact geometry already retained by the declared grid workflow. This is a
    # mechanical filter, not a new semantic selection.
    retained_grid_geometry: tuple[int, int, int, int, int, int, int, int] | None = None
    retained_grid_ref: Ref | None = None
    retained_grid_separator_consistency_ppm: int = Field(
        default=0, ge=0, le=1_000_000
    )
    # Compact synthesized memory for the retained grid. The full cell objects
    # are materialized lazily only when a declared consumer such as K09 needs
    # them; this avoids retaining a duplicate scene-sized representation.
    retained_grid_recount: KnownGridCellRecountResult | None = None
    # Bounded ray-extension measurement from the retained cursor cell.  The
    # capability reports only the action whose continuation reaches a
    # different reachable region; DRM decides whether that witness outranks
    # unrelated exploratory actions.
    retained_grid_exploratory_frontier_action_ref: Ref | None = None
    retained_grid_exploratory_frontier_target_cell_index: int | None = Field(
        default=None, ge=0
    )
    retained_grid_exploratory_frontier_present: bool = False
    # Bounded cell indices demanded by the active declared retained-grid
    # reasoning leaf. Empty means that leaf has not supplied an exact witness.
    retained_grid_informative_cell_indices: tuple[int, ...] = Field(
        default=(), max_length=3
    )
    # Exact cells exposed by a bounded viewport-shift measurement.  They remain
    # descriptive evidence; DRM decides whether the prior point effect should
    # be reused to advance the newly revealed frontier.
    retained_grid_revealed_effectful_cell_indices: tuple[int, ...] = Field(
        default=(), max_length=3
    )
    # Exact same-pattern cells demanded by the reconciled delta of the latest
    # point action.  Python transports only bounded cell indexes; DRM decides
    # whether continuing that causal interaction chain advances an objective.
    retained_grid_point_delta_informative_cell_indices: tuple[int, ...] = Field(
        default=(), max_length=3
    )
    retained_grid_effectful_point_source_pattern_refs: tuple[Ref, ...] = Field(
        default=(), max_length=3
    )
    # Mechanical terminal checkpoint selected by the retained-grid SRC route.
    # DRM alone assigns access/frontier meaning to this transported fact.
    retained_grid_opened_column_route_reached_terminal: bool = False
    available_action_refs: tuple[Ref, ...] = Field(min_length=1)
    # Bounded raster census used only to decide whether the initial point
    # frontier should be materialized before an uninformative directional
    # probe.  It carries no role or mechanism meaning.
    bounded_point_candidate_component_count: int = Field(default=0, ge=0, le=256)
    # Explicit environment/session control transported by the adapter. At the
    # level-start causal configuration it may be probed once without destroying
    # progress; display-only feedback does not close that safe window. After a
    # world effect or progress it is exposed only as dedicated recovery.
    level_start_checkpoint_action_ref: Ref | None = None
    level_start_checkpoint_execution_count: int = Field(default=0, ge=0, le=3)
    at_level_start_checkpoint_configuration: bool = False
    level_start_checkpoint_probe_is_lossless_in_current_observed_context: bool = False
    current_context_no_effect_candidate_refs: tuple[Ref, ...] = ()
    # Exact action refs whose prior execution in this same hashed context
    # produced a positive delta in an already-established boundary indicator.
    # Python only matches immutable measurements; DRM assigns replenishment
    # meaning and selection priority.
    current_context_positive_boundary_indicator_delta_action_refs: tuple[
        Ref, ...
    ] = Field(default=(), max_length=4)
    # Exact current-run evidence that at least one established boundary
    # indicator increased after an action.  This transports no role, route or
    # selection; DRM decides how recovered budget changes probe ordering.
    positive_boundary_indicator_delta_observed_in_current_attempt: bool = False
    # Exact action refs whose prior-attempt execution in this same hashed
    # context produced a world-spatial effect. Python only matches immutable
    # observations; DRM decides whether repeating one advances an open goal.
    current_context_prior_attempt_world_effect_action_refs: tuple[
        Ref, ...
    ] = Field(default=(), max_length=8)
    # Exact primitive refs whose already-committed residual cursor reached
    # zero in the still-current control context.  This is measured cursor
    # lifecycle state; DRM decides whether the primitive remains eligible.
    terminally_exhausted_action_refs: tuple[Ref, ...] = Field(
        default=(), max_length=3
    )
    closed_goal_route_stop_lifecycle_count: int = Field(default=0, ge=0, le=3)
    # Exact DRM-authored goal + route/experiment + stop lifecycle refs closed by
    # the bounded action budget. Python transports only the identities; DRM
    # decides whether the exact engagement remains excluded or retry-hard
    # explicitly reopens it. Other successors of the same goal stay eligible.
    terminally_exhausted_goal_route_stop_lifecycle_refs: tuple[Ref, ...] = Field(
        default=(), max_length=3
    )
    # Exact DRM-authored goal refs whose standard BFS cursor reached its target
    # in this level. Python transports the bounded identities; DRM decides that
    # the standard frontier must move to a distinct target.
    visited_bfs_goal_refs: tuple[Ref, ...] = Field(default=(), max_length=3)
    current_scope_attempted_route_suffixes: tuple[Ref, ...] = Field(
        default=(), max_length=3
    )
    other_context_no_effect_action_refs: tuple[Ref, ...] = ()
    # Exact directional sterility measured since the most recent observed
    # non-directional scene effect. DRM decides its eligibility consequence.
    control_epoch_no_effect_directional_action_refs: tuple[Ref, ...] = Field(
        default=(), max_length=4
    )
    tested_candidate_refs_without_resolution_plan: tuple[Ref, ...] = ()
    known_scene_effect_component_refs: tuple[Ref, ...] = Field(
        default=(), max_length=64
    )
    # Mechanical evidence from the most recent effectful point action.  The
    # count excludes the clicked component's bounded local footprint and keeps
    # palette values descriptive: DRM alone may interpret the observation as a
    # revisable colour-indexed actuator relation.
    last_effectful_point_remote_same_palette_changed_pixel_count: int = Field(
        default=0, ge=0
    )
    # The invitation was measured before the effect because the tutorial
    # contour may disappear as soon as the first control is used.
    last_effectful_point_source_had_white_invitation: bool = False
    # Exact pre-effect point loci that carried a measured white component or
    # enclosure.  They are bounded scene facts, not persistent role labels.
    pre_effect_white_invited_point_loci: tuple[Position, ...] = Field(
        default=(), max_length=64
    )
    # Exact declarative metaplan output transported for DRM selection.  Python
    # does not derive or reinterpret this cognitive reference.
    reasoning_strategy_ref: Ref | None = None
    # SRC/DRM-declared scope for point-locus materialization.  The default
    # preserves the full unknown-world inventory; Python never infers the
    # narrower scope from a game name, palette, or available action family.
    declared_point_candidate_scope_ref: Ref = "probe_scope.full_point_loci"
    # Exact DRM demand, independent of the point-locus optimization.
    declared_required_measurement_families: tuple[
        Literal["measurement_family.paired_peer_shared_path", "measurement_family.exact_enclosure_assignment", "measurement_family.oriented_square_transport", "measurement_family.alternating_material_graph"], ...
    ] = Field(default=(), max_length=3)
    # SRC/DRM-declared scope for the interaction frontier itself.  The narrow
    # scopes measure only the declared bounded route family; an empty result
    # must be followed by a source-declared reopening of the full scope.
    interaction_measurement_scope_ref: Ref = "measurement_scope.full_interaction_frontier"
    # Exact SRC-selected subgoal transported from the immediately preceding
    # action authority.  Python does not derive or interpret this reference.
    active_subgoal_ref: Ref | None = None
    # The initial singleton-cell SRC leaf can defer the separate multi-sprite
    # common-clearance search; any later branch requiring it reopens full scope.
    declared_shared_clearance_scope_ref: Ref = "clearance_scope.full"
    # SRC/DRM may defer the independent three-overlay push measurement while
    # testing a unique-cell directional witness; later probes default to full.
    declared_cell_transport_scope_ref: Ref = "cell_transport_scope.full"
    # The currently active configuration-priority node is a TERM/CLAIM fact
    # selected by DRM/SRC.  A missing value means that no priority branch has
    # been committed; capabilities must not infer one from geometry.
    active_priority_node_ref: Ref | None = None
    # Mechanical transport of the currently declared dynamic frontier.  The
    # graph projection owns these facts; this contract does not select a goal
    # or assign a semantic order.
    # Canonical TERM/CLAIM remains the complete durable memory.  The transport
    # accepts a large explicit safety ceiling so ordinary exact-pair agendas
    # are never silently truncated before DRM selection; capabilities publish
    # the current count and whether this fail-closed ceiling was reached.
    # Mechanical TERM/CLAIM census for DRM priority projection; it does not
    # select a goal or an action.
    declared_goal_kind_counts: tuple[tuple[Ref, int], ...] = Field(
        default=(), max_length=64
    )
    dynamic_goal_refs: tuple[Ref, ...] = Field(default=(), max_length=4096)
    dynamic_goal_facts: FrozenMap = Field(default_factory=FrozenMap)
    # Thin transport of already-observed action/effect and spatial novelty facts
    # (measured outside DRM). Selection remains declarative.
    committed_plan_present: bool = False
    # Exact SRC lifecycle boundary: a DRM-declared standard object-terminating
    # BFS consumed its final primitive in the current reconciled observation.
    # Capabilities expose the boundary; DRM decides whether another target is due.
    standard_bfs_target_cursor_end_observed: bool = False
    # Exact SRC lifecycle boundary for any verified repetition cursor. This is
    # separate from the target-reaching subtype so DRM can choose continuation.
    verified_repetition_cursor_end_observed: bool = False
    # Exact declarative plan-kind reference dispatched on the preceding step.
    # Python transports this cognitive reference without interpreting it.
    committed_plan_kind_ref: Ref | None = None
    # Exact SRC boundary for a bounded-zone route whose final primitive was
    # just dispatched.  This closes the route before generic equivalent-frame
    # retries can reopen the same directional frontier.
    prior_dispatched_bounded_zone_state_route_terminal_observed: bool = False
    prior_bounded_zone_state_route_step_no_effect: bool = False
    prior_bounded_zone_state_route_was_dispatched: bool = False
    # Exact presence bit from the incoming DRM/SRC-authored route engagement.
    # Selection and deferral remain declarative; Python only transports it.
    incoming_declared_route_engagement_present: bool = False
    incoming_declared_route_remaining_step_count: int = Field(default=0, ge=0)
    # Exact measurements from the preceding DRM-authored object-terminating
    # route experiment. Python transports the declared route length; DRM alone
    # combines it with the current measured translation to decide whether it
    # licenses a contact interaction at the reached frontier.
    prior_dispatched_provisional_object_terminating_route_length: int = Field(
        default=0, ge=0, le=128
    )
    # Exact DRM-authored fact transported with a focus-transfer plan.  It is
    # false unless the declarative source established that the preceding body
    # context reached its terminal residual before releasing the selector.
    committed_focus_transfer_body_context_exhausted_declared: bool = False
    # Thin transport of a DRM-authored exception: the preceding primitive was
    # allowed to reopen a lower-priority shape axis because it advanced the
    # still-active priority goal. Python preserves the boolean only; DRM owns
    # continuation and invalidation when the active priority changes.
    prior_dispatched_lower_shape_axis_reopening_was_instrumental_to_active_priority: bool = False
    # Mechanical snapshot of the four exact residuals (axis row/column, direct
    # body row/column) that preceded the last dispatched declarative action.
    # Capabilities compare it with the current measurement; DRM alone decides
    # whether observed progress reopens that same primitive.
    prior_dispatched_orthogonal_reflection_residuals: tuple[
        int, int, int, int
    ] | None = None
    # Bounded mechanical memory of recently released directional primitives
    # and the exact repeated-embedded residual at release time.  Capabilities
    # expose exact composition reuse; DRM decides its eligibility consequence.
    recent_dispatched_repeated_embedded_action_residuals: tuple[
        tuple[Ref, int, int], ...
    ] = Field(default=(), max_length=3)
    # Bounded transport of the exact peer identity bound by a recently
    # released shared-path projection. The following observation must still
    # confirm that same identity before DRM can complete its lateral probe.
    recent_reconciled_paired_peer_shared_path_component_refs: tuple[Ref, ...] = (
        Field(default=(), max_length=3)
    )
    known_effectful_action_refs: tuple[Ref, ...] = ()
    # Exact action refs whose canonical contextual-effect claim records an
    # ordered temporal transition.  This is observed evidence, not a claim
    # that the transition advanced an objective.
    observed_ordered_temporal_transition_action_refs: tuple[Ref, ...] = ()
    known_no_effect_action_refs: tuple[Ref, ...] = ()
    # Thin transport of DRM-classified observable-effect scopes. These refs
    # preserve declared exceptions to the ordinary no-effect repetition bound.
    resource_only_effect_action_refs: tuple[Ref, ...] = ()
    cyclic_pose_only_effect_action_refs: tuple[Ref, ...] = ()
    reverse_of_last_action_refs: tuple[Ref, ...] = ()
    reverse_of_last_directional_action_refs: tuple[Ref, ...] = ()
    last_action_ref: Ref | None = None
    last_directional_action_ref: Ref | None = None
    previous_distinct_directional_action_ref: Ref | None = None
    previous_action_ref: Ref | None = None
    last_point_position: Position | None = None
    previous_point_position: Position | None = None
    # Bounded exact geometry from the latest point-conditioned moving transient
    # component.  Python transports measurements only; DRM decides whether the
    # path is a launch, another autonomous process, or irrelevant animation.
    prior_transient_component_path_measurement: FrozenMap = Field(
        default_factory=FrozenMap
    )
    # Latest unique thickness-1 prefix/suffix bar. Python transports the
    # count and the bar's geometry. DRM decides whether that count is a
    # magnitude to vary after a curved transient path.
    prior_scalar_fill_display_measurement: FrozenMap = Field(
        default_factory=FrozenMap
    )
    prior_curved_launch_point: Position | None = None
    prior_launched_point_scalar_fill_pairs: tuple[tuple[int, int, int], ...] = Field(
        default=(), max_length=48
    )
    prior_scalar_fill_increase_action_ref: Ref | None = None
    prior_scalar_fill_decrease_action_ref: Ref | None = None
    prior_scalar_fill_unchanged_action_refs: tuple[Ref, ...] = Field(
        default=(), max_length=16
    )
    # Cell and fill of a fitted arc that meets the one distinctive stationary
    # body, or of a longer flight toward it when that arc does not yet meet.
    # Python transports the cells. DRM decides which one is released.
    prior_ballistic_contact_solution_point: Position | None = None
    prior_ballistic_contact_solution_fill_count: int | None = None
    prior_ballistic_curve_disclosure_point: Position | None = None
    prior_ballistic_curve_disclosure_fill_count: int | None = None
    # Bounded exact action-conditioned memory.  These are observed loci, not
    # inferred roles: DRM remains responsible for deciding whether revisiting
    # one can advance the current goal.
    known_effectful_point_loci: tuple[tuple[Ref, int, int], ...] = Field(
        default=(), max_length=32
    )
    # Exact pre-action component footprints for the same bounded loci.  They
    # are geometry only; declared knowledge decides whether a current fragment
    # preserves the prior component identity.
    known_effectful_point_footprints: tuple[BoundingBox, ...] = Field(
        default=(), max_length=32
    )
    # Subset whose pre-action point lay in an exactly measured enclosed region.
    known_effectful_point_enclosing_footprints: tuple[BoundingBox, ...] = Field(
        default=(), max_length=32
    )
    # Exact point candidates whose observed transition left the largest aligned
    # rectangular collection unchanged while it was at the maximum cardinality
    # seen in this level epoch.  This is a count relation, not a capacity role.
    point_candidate_refs_with_unchanged_collection_at_observed_maximum: tuple[
        Ref, ...
    ] = Field(default=(), max_length=32)
    point_positions_with_unchanged_collection_at_observed_maximum: tuple[
        Position, ...
    ] = Field(default=(), max_length=32)
    ordered_collection_observed_maximum_count: int = Field(default=0, ge=0, le=12)
    ordered_collection_reduction_observed_after_unchanged_maximum: bool = False
    latest_point_ordered_collection_delta: FrozenMap = Field(
        default_factory=FrozenMap
    )
    # Bounded exact raster-transition measurements for one isolated
    # intersecting-square-cycle experiment.  Geometry and signed offsets are
    # measurements only; DRM owns their track/control interpretation.
    isolated_distributed_square_transition_measurements: tuple[FrozenMap, ...] = Field(
        default=(), max_length=6
    )
    # The last point action changed a revisable cell and must be reconciled
    # before a stale directional subgoal is resumed.
    last_point_was_access_opening: bool = False
    point_pre_action_enclosure_residual: Position | None = None
    point_designated_enclosure_residual_present: bool = False
    nonlocal_translation_and_open_enclosure_residual_present: bool = False
    point_pre_action_translated_peer_residual: Position | None = None
    nonlocal_quantized_translation_observed: bool = False
    relational_cell_positions: tuple[Position, ...] = Field(default=(), max_length=64)
    relational_occupied_positions: tuple[Position, ...] = Field(
        default=(), max_length=64
    )
    action_execution_counts: tuple[tuple[Ref, int], ...] = ()
    candidate_execution_counts: tuple[tuple[Ref, int], ...] = ()
    # Bounded mechanical count of consecutive frontier-retry aliases in the
    # current route engagement.  The controller resets it at a non-retry
    # action; DRM uses it only with the measured route budget.
    frontier_retry_execution_count: int = Field(default=0, ge=0, le=64)
    frontier_retry_route_length: int = Field(default=0, ge=0, le=40)
    consecutive_directional_repeat_count: int = Field(default=0, ge=0, le=32)
    preceding_consecutive_distinct_point_locus_count: int = 0
    last_selected_point_palette_value: int = -1
    last_selected_point_fills_bbox: bool = False
    last_selected_point_exact_morphology_group_size: int = 0
    point_streak_origin_intersects_transient_change_bbox: bool = False
    current_spatial_configuration_hash: Ref | None = None
    current_spatial_configuration_changed: bool = False
    visited_spatial_configuration_count: int = Field(default=0, ge=0)
    actions_that_produced_novel_configuration: tuple[Ref, ...] = ()
    configuration_revisit_action_refs: tuple[Ref, ...] = ()
    configuration_revisit_candidate_refs: tuple[Ref, ...] = ()
    # Exact candidate refs already dispatched from the currently observed
    # spatial configuration. This is bounded mechanical memory; DRM decides
    # whether a repeated interaction remains semantically justified.
    current_configuration_executed_candidate_refs: tuple[Ref, ...] = ()
    # Exact actions whose already-observed transition from the current spatial
    # configuration restores the configuration immediately preceding the last
    # meaningful spatial change. This is a measurement, not an undo role.
    immediate_predecessor_restoring_action_refs: tuple[Ref, ...] = ()
    directional_frontier_saturated: bool = False
    # Mechanical memory for the generic last-resort action/configuration
    # census. Keys are opaque measured experiment identities, never semantic
    # roles, pose requirements, or a level-specific plan. DRM/SRC decide when
    # this bounded frontier is admissible and what its result means.
    bounded_exhaustive_experiment_attempt_keys: tuple[Ref, ...] = Field(
        default=(), max_length=16
    )
    # Mechanical presence of a DRM-declared cursor-role supposition.  The
    # declaration is revisable, but while it remains live the cursor class no
    # longer needs a generic directional identity probe.
    declared_cursor_role_present: bool = False
    cellular_mover_pattern_refs: tuple[Ref, ...] = ()
    recent_moved_cell_pattern_refs: tuple[Ref, ...] = ()
    cellular_actuator_pattern_refs: tuple[Ref, ...] = ()
    cellular_traversable_pattern_refs: tuple[Ref, ...] = ()
    open_contact_pattern_refs: tuple[Ref, ...] = ()
    viewport_contact_projections: tuple[ViewportContactProjection, ...] = Field(
        default=(), max_length=3
    )
    # Measured (action_ref, delta_row, delta_col) from prior effectful transitions.
    known_action_translation_deltas: tuple[tuple[Ref, int, int], ...] = ()
    # Exact action-aligned translation from the latest controlled transition.
    # This local measurement may differ from a durable quantum learned for the
    # same interface action on another controlled body.
    observed_input_aligned_action_translation: tuple[Ref, int, int] | None = None
    # Exact zero-gap cell quantum measured before a DRM/SRC-authorized
    # directional identification action. Numeric transport only: the
    # declarative projection decides whether this measurement may persist.
    retained_action_quantized_grid_quantum: int = Field(default=0, ge=0, le=32)
    # Exact twice-centre residual transported through the latest observed
    # input-aligned translation when visible morphology is fragmented.
    transported_input_aligned_shape_peer_residual: Position | None = None
    transported_input_aligned_shape_peer_quantum: int = Field(default=0, ge=0)
    # Exact axes on which the transported residual crossed a previously closed
    # terminal (zero -> nonzero) or changed sign. Python transports only this
    # measurement; DRM decides whether it opens a recovery procedure.
    transported_shape_peer_residual_crossed_terminal_axes: tuple[bool, bool] = (
        False,
        False,
    )
    transported_residual_origin_was_unique_unilateral_enclosed_change: bool = False
    # Bounded exact marker-axis geometry carried across an observed controlled
    # translation.  These are numeric measurements, not a goal or role; DRM
    # decides whether a zero residual is a still-active invariant.
    transported_marker_axis_residual_measurement_present: bool = False
    transported_marker_axis_group_digest: Ref | None = None
    transported_marker_axis_residual_row_signed: int = 0
    transported_marker_axis_residual_col_signed: int = 0
    transported_marker_axis_translation_quantum: int = Field(default=0, ge=0)
    # Number of entity translations measured in the latest controlled
    # transition. Python transports this count; DRM decides its meaning.
    observed_controlled_translation_count: int = Field(default=0, ge=0, le=64)
    # Maximum cardinality of one exact equal-vector component cohort observed
    # in this level. Retained measurement; it asserts neither attachment nor
    # a current co-translation after a later local or blocked operation.
    observed_equal_exact_translation_occurrence_count_max: int = Field(
        default=0, ge=0, le=64
    )
    # Exact remaining quanta of the current-level boundary indicator, but only
    # after DRM established its resource-indicator role and repeated decrement
    # unit. Python transports the arithmetic measurement; DRM decides whether
    # it is a deadline, stock, score, or another resource kind.
    established_boundary_indicator_present: bool = False
    boundary_indicator_remaining_quantum_count: int = Field(
        default=0, ge=0, le=128
    )
    # Exact palette values of that already-established indicator's current
    # tracked component.  This is visual correspondence evidence only; it does
    # not assign a resource role to any same-valued scene object.
    boundary_indicator_palette_values: tuple[int, ...] = Field(
        default=(), max_length=64
    )
    # Last non-empty bounded delta packet for the same continuously repeated
    # action. It survives one changed frame with ambiguous identity tracking.
    recent_same_action_controlled_translation_action_ref: Ref | None = None
    recent_same_action_controlled_translation_deltas: tuple[
        tuple[int, int], ...
    ] = Field(default=(), max_length=64)
    # Largest number of distinct non-zero translation deltas measured together
    # in this level epoch. This is bounded temporal evidence, not a role claim.
    observed_distinct_controlled_translation_delta_count_max: int = Field(
        default=0, ge=0, le=64
    )
    # Current exact boxes of every identity with an observed non-zero
    # translation. This supports mechanical co-mover/stationary separation.
    observed_translated_entity_bboxes: tuple[BoundingBox, ...] = Field(
        default=(), max_length=64
    )
    # Historical translated tracking-ref cardinality for this level epoch.
    # Identifier revisions can make this scalar exceed the bounded current
    # component collections. It is not an exact physical body count and does
    # not by itself turn an untested lookalike into a controlled entity.
    observed_distinct_translated_entity_count: int = Field(default=0, ge=0)
    # Exact current-transition component ref and measured row/column delta.
    # Historical controlled identities are not current motion evidence.
    current_exact_component_translations: tuple[tuple[Ref, int, int], ...] = Field(
        default=(), max_length=64
    )
    # Prior-frame pixels at each uniquely translated component's current box.
    # Each row is current-observation evidence, independent of retained poses.
    current_translation_underlay_rows: tuple[FrozenMap, ...] = Field(default=(), max_length=64)
    # Current exact boxes of entities previously identified by controlled
    # translation. This is geometry transport, not a role assignment.
    translated_entity_bboxes: tuple[BoundingBox, ...] = Field(
        default=(), max_length=64
    )
    # A retained last-translation pose is still available for continuity, but
    # is not an additional body observed at that pose in the current frame.
    translated_entity_bboxes_from_prior_frame: bool = False
    # Current exact boxes whose observed delta equals the declared interface
    # delta. This is an action-conditioned geometric measurement only.
    input_aligned_entity_bboxes: tuple[BoundingBox, ...] = Field(
        default=(), max_length=64
    )
    # Canonical TERM facts authored by DRM for provisional controlled
    # identities, including every appearance alias and its proof. Exact
    # descriptions share one transport row; source_term_rows retains every
    # supporting TERM and its own premises. The bound is 64 descriptions,
    # each with at most 64 proofs, not the three-route ceiling.
    # Python transports the declared entity ref only; it assigns no role.
    canonical_provisional_control_term_facts: tuple[FrozenMap, ...] = Field(
        default=(), max_length=64
    )
    # Source-declared interface supports, with explicit exclusion and falsifier.
    # Transporting them does not assign a role to any measured component.
    canonical_interface_member_facts: tuple[FrozenMap, ...] = Field(default=(), max_length=128)
    # Current exact geometry joined to the singleton canonical control term.
    # It is transported separately from generic input alignment so a simultaneous
    # foreign fragment cannot erase a revalidated zone-cell occupancy witness.
    canonical_provisional_control_geometry_bboxes: tuple[BoundingBox, ...] = Field(
        default=(), max_length=3
    )
    # Current exact visible-outline geometry measured by action-conditioned
    # control tracking.  It is transported separately from provisional TERM
    # cardinality so transparent underlays cannot duplicate the controlled
    # outline.  Consumers may use it only when the bbox remains singleton.
    # The producer joins at most one exact current bbox per canonical role
    # bearer and deduplicates boxes: its bound is the 64 canonical TERM rows,
    # not the independent ceiling of three executable route witnesses.
    current_action_conditioned_control_outline_bboxes: tuple[BoundingBox, ...] = Field(
        default=(), max_length=64
    )
    # True only between a DRM-selected control-focus transfer and the first
    # action-conditioned translation observed for the rebound focus.
    control_identity_rebinding_confirmation_pending: bool = False
    # Last direction whose scene effect was measured in the preceding level
    # epoch.  Object, coordinate, and route bindings are deliberately absent;
    # DRM may use this action-only witness for one falsifiable rebound test.
    prior_control_epoch_effectful_directional_action_ref: Ref | None = None
    # Exact singleton candidates produced by a reciprocal value swap between
    # enclosed cells of the prior controlled footprint and another current
    # component. This remains a control supposition until a directional effect.
    reciprocal_enclosed_change_candidate_bboxes: tuple[BoundingBox, ...] = Field(
        default=(), max_length=3
    )
    # Exact stationary regions carrying the reciprocal appearance transition
    # of the prior controlled footprint.  Values and interior morphology stay
    # descriptive; DRM alone may interpret this as a revisable focus transfer.
    reciprocal_stationary_appearance_candidate_bboxes: tuple[
        BoundingBox, ...
    ] = Field(default=(), max_length=3)
    # Exact singleton candidates whose enclosed cells changed in one
    # direction when the reciprocal endpoint is visually occluded.
    unilateral_enclosed_change_candidate_bboxes: tuple[BoundingBox, ...] = Field(
        default=(), max_length=3
    )
    # Bounded exact pose history for frontier novelty.  Python transports
    # geometry only; DRM decides whether revisiting a pose is useful or errant.
    visited_input_aligned_entity_bboxes: tuple[BoundingBox, ...] = Field(
        default=(), max_length=64
    )
    # Exact terminal control poses recorded only when SRC closes a declared
    # standard object-terminating BFS. They are bounded mechanical history;
    # DRM decides whether such a completed carrier can be selected again.
    completed_standard_bfs_control_bboxes: tuple[BoundingBox, ...] = Field(
        default=(), max_length=3
    )
    # Exact aligned-entry regions carried by the DRM-committed standard BFS.
    # Unlike the final control pose, this region preserves one logical cell-zone
    # when the occupying body later occludes and fragments its visible contour.
    completed_standard_bfs_aligned_entry_region_bboxes: tuple[
        BoundingBox, ...
    ] = Field(default=(), max_length=3)
    # Exact non-boundary regions changed by an interaction whose remote regular
    # indicator family was declaratively established as objective progress.
    # These are bounded mechanical episode memory; DRM owns their closure role.
    completed_objective_progress_interaction_region_bboxes: tuple[
        BoundingBox, ...
    ] = Field(default=(), max_length=6)
    # Compact transport of a canonical DRM-authored objective-progress TERM.
    # Python measures neither the role nor the goal; it only carries the
    # declared positive remainder into bounded route measurement.
    canonical_objective_progress_indicator_present: bool = False
    canonical_objective_progress_indicator_remaining_member_count: int = Field(
        default=0, ge=0, le=12
    )
    # Compact DRM-authored terminal-access candidates.  Python may compare the
    # supplied exact bbox with route endpoints, but cannot originate the role.
    canonical_terminal_access_candidate_rows: tuple[FrozenMap, ...] = Field(
        default=(), max_length=3
    )
    # Exact values observed in the prior frame at the current translated
    # footprint.  This is measured underlay, not a terrain-role assignment.
    current_input_aligned_underlay_values: tuple[int, ...] = Field(
        default=(), max_length=64
    )
    # Bounded union of exact underlay values observed beneath action-aligned
    # footprints during the current level episode.  Point-based focus transfer
    # may clear the current footprint measurement but does not erase this map
    # evidence.  DRM policy still decides whether it grounds a floor graph.
    episode_input_aligned_underlay_values: tuple[int, ...] = Field(
        default=(), max_length=64
    )
    # Bounded connected changed-pixel envelopes for the current transition.
    # They carry no role.  A capability may compare their current full-valued
    # patterns with periodic cell sprites before DRM assigns any meaning.
    current_transition_change_bboxes: tuple[BoundingBox, ...] = Field(
        default=(), max_length=3
    )
    # Number of observed intermediate frames before the stable frame in the
    # current transition. This is temporal evidence only, never a role.
    current_transition_intermediate_frame_count: int = Field(
        default=0, ge=0, le=128
    )
    # Bounded retrospective evidence extracted from an animated point packet.
    # These are mechanical observations; DRM decides whether they commit a
    # mechanism model and reopen a route.
    retrospective_animation_effect_evidence: bool = False
    retrospective_animation_retry_requires_safe_context: bool = False
    retrospective_animation_reconfiguration: FrozenMap = Field(
        default_factory=FrozenMap
    )
    # Exact controlled-body translation measured for the last action in the
    # current visual transition.  This remains a mechanical raster delta: it
    # may differ from the unit intent declared by the input interface.
    last_action_measured_translation_delta: tuple[int, int] | None = None
    # Directional intent declared by the official input interface.  This is a
    # counterfactual control description until an observed transition verifies
    # (or falsifies) the corresponding game effect.
    interface_action_translation_deltas: tuple[tuple[Ref, int, int], ...] = ()
    # Thin canonical TERM transport.  DRM authored these meanings; the
    # capability may only join their refs to exact geometry and local no-effect
    # measurements.
    canonical_pose_term_facts: tuple[FrozenMap, ...] = Field(
        default=(), max_length=64
    )
    canonical_correspondence_term_facts: tuple[FrozenMap, ...] = Field(
        default=(), max_length=64
    )
    canonical_reversible_control_claim_facts: tuple[FrozenMap, ...] = Field(
        default=(), max_length=64
    )
    canonical_paired_peer_shared_path_claim_facts: tuple[FrozenMap, ...] = Field(
        default=(), max_length=64
    )
    canonical_closed_support_term_facts: tuple[FrozenMap, ...] = Field(
        default=(), max_length=3
    )
    canonical_enclosed_locus_term_facts: tuple[FrozenMap, ...] = Field(
        default=(), max_length=3
    )
    canonical_joint_slot_relation_facts: tuple[FrozenMap, ...] = Field(
        default=(), max_length=64
    )
    canonical_joint_slot_claim_facts: tuple[FrozenMap, ...] = Field(
        default=(), max_length=64
    )
    canonical_ordered_trait_claim_facts: tuple[FrozenMap, ...] = Field(
        default=(), max_length=64
    )
    canonical_complete_word_translation_facts: tuple[FrozenMap, ...] = Field(
        default=(), max_length=64
    )
    canonical_closed_support_permutation_claim_facts: tuple[FrozenMap, ...] = Field(
        default=(), max_length=64
    )
    # Declaratively transported mechanical route bounds.  The capability may
    # materialize a path only inside these fixed limits; it does not choose the
    # semantic plan.
    route_generation_policy: FrozenMap = Field(default_factory=FrozenMap)
    known_viewport_grid_shift_deltas: tuple[tuple[Ref, int, int], ...] = Field(
        default=(), max_length=32
    )
    # Coupling of co-mover residual to controller delta per axis: +1 same, -1 opposite.
    residual_axis_coupling_row: int = Field(default=1)
    residual_axis_coupling_column: int = Field(default=1)
    schema_version: Ref = "yf_arc3_v5.interaction_probe_input.v1"

    @model_validator(mode="after")
    def validate_probe_scope(self) -> "InteractionProbeInput":
        if self.interaction_measurement_scope_ref not in {
            "measurement_scope.full_interaction_frontier",
            "measurement_scope.unique_dynamic_goal_continuation",
            "measurement_scope.recalled_peer_method",
            "measurement_scope.provisional_object_terminating_route",
            "measurement_scope.bounded_zone_state_route",
        }:
            raise ValueError("interaction measurement scope must be source-declared")
        if self.declared_point_candidate_scope_ref not in {
            "probe_scope.full_point_loci",
            "probe_scope.access_witness_only",
            "probe_scope.retained_grid_informative_cells",
        }:
            raise ValueError("point candidate scope must be source-declared")
        if self.declared_shared_clearance_scope_ref not in {
            "clearance_scope.full",
            "clearance_scope.defer_shared_multi_sprite",
        }:
            raise ValueError("shared clearance scope must be source-declared")
        if len(set(self.available_action_refs)) != len(self.available_action_refs):
            raise ValueError("available action refs must be unique")
        if (
            self.level_start_checkpoint_action_ref is not None
            and self.level_start_checkpoint_action_ref not in self.available_action_refs
        ):
            raise ValueError(
                "level-start checkpoint action must be in the available vocabulary"
            )
        if len(set(self.immediate_predecessor_restoring_action_refs)) != len(
            self.immediate_predecessor_restoring_action_refs
        ):
            raise ValueError(
                "immediate-predecessor restoring action refs must be unique"
            )
        if any(
            ref not in self.available_action_refs
            for ref in self.immediate_predecessor_restoring_action_refs
        ):
            raise ValueError(
                "immediate-predecessor restoring actions must be available"
            )
        if len(set(self.current_context_no_effect_candidate_refs)) != len(
            self.current_context_no_effect_candidate_refs
        ):
            raise ValueError("current-context no-effect candidate refs must be unique")
        if len(
            set(
                self.point_candidate_refs_with_unchanged_collection_at_observed_maximum
            )
        ) != len(
            self.point_candidate_refs_with_unchanged_collection_at_observed_maximum
        ):
            raise ValueError(
                "unchanged-at-observed-maximum point candidate refs must be unique"
            )
        if len(
            set(self.current_context_positive_boundary_indicator_delta_action_refs)
        ) != len(self.current_context_positive_boundary_indicator_delta_action_refs):
            raise ValueError(
                "current-context positive boundary-indicator delta action refs "
                "must be unique"
            )
        if len(
            set(self.current_context_prior_attempt_world_effect_action_refs)
        ) != len(self.current_context_prior_attempt_world_effect_action_refs):
            raise ValueError(
                "current-context prior-attempt world-effect action refs must be unique"
            )
        if len(set(self.other_context_no_effect_action_refs)) != len(
            self.other_context_no_effect_action_refs
        ):
            raise ValueError("other-context no-effect action refs must be unique")
        if len(set(self.tested_candidate_refs_without_resolution_plan)) != len(
            self.tested_candidate_refs_without_resolution_plan
        ):
            raise ValueError("tested candidate refs must be unique")
        if len(set(self.known_scene_effect_component_refs)) != len(
            self.known_scene_effect_component_refs
        ):
            raise ValueError("known scene-effect component refs must be unique")
        if len(set(self.observed_ordered_temporal_transition_action_refs)) != len(
            self.observed_ordered_temporal_transition_action_refs
        ):
            raise ValueError(
                "observed ordered-temporal-transition action refs must be unique"
            )
        count_refs = tuple(ref for ref, _count in self.action_execution_counts)
        if len(set(count_refs)) != len(count_refs):
            raise ValueError("action execution count refs must be unique")
        if any(count < 0 for _ref, count in self.action_execution_counts):
            raise ValueError("action execution counts must be non-negative")
        candidate_count_refs = tuple(
            ref for ref, _count in self.candidate_execution_counts
        )
        if len(set(candidate_count_refs)) != len(candidate_count_refs):
            raise ValueError("candidate execution count refs must be unique")
        if any(count < 0 for _ref, count in self.candidate_execution_counts):
            raise ValueError("candidate execution counts must be non-negative")
        if self.preceding_consecutive_distinct_point_locus_count < 0:
            raise ValueError(
                "preceding consecutive distinct point locus count must be non-negative"
            )
        if self.last_selected_point_exact_morphology_group_size < 0:
            raise ValueError(
                "last selected point exact morphology group size must be non-negative"
            )
        if len(set(self.dynamic_goal_refs)) != len(self.dynamic_goal_refs):
            raise ValueError("dynamic goal refs must be unique")
        if set(self.dynamic_goal_facts) != set(self.dynamic_goal_refs):
            raise ValueError("dynamic goal facts must index every dynamic goal")
        if any(
            not isinstance(self.dynamic_goal_facts[item], FrozenMap)
            for item in self.dynamic_goal_refs
        ):
            raise ValueError("dynamic goal facts must be immutable mappings")
        return self


class InteractionProbeAgenda(FrozenModel):
    candidates: tuple[InteractionProbeCandidate, ...]
    alternative_refs: tuple[Ref, ...]
    alternative_facts: FrozenMap = Field(default_factory=FrozenMap)
    descriptive_delta_facts: FrozenMap = Field(default_factory=FrozenMap)
    context_facts: FrozenMap = Field(default_factory=FrozenMap)
    has_candidates: bool
    current_context_no_effect_candidate_refs: tuple[Ref, ...] = ()
    tested_candidate_refs_without_resolution_plan: tuple[Ref, ...] = ()
    # Mechanical memory of bounded route variants rejected during
    # reconciliation.  DRM/SRC interpret the rejection; the capability only
    # transports the exact variant and a non-semantic failure kind.
    rejected_route_variant_refs: tuple[Ref, ...] = Field(default=(), max_length=3)
    rejected_route_variant_kinds: tuple[Ref, ...] = Field(default=(), max_length=3)
    rejected_grounding_candidate_refs: tuple[Ref, ...] = Field(
        default=(), max_length=64
    )
    rejected_grounding_failure_kinds: tuple[Ref, ...] = Field(
        default=(), max_length=64
    )
    requires_ontology_extension: bool = False
    ontology_extension_context: FrozenMap = Field(default_factory=FrozenMap)
    schema_version: Ref = "yf_arc3_v5.interaction_probe_agenda.v3"

    @model_validator(mode="after")
    def validate_candidate_index(self) -> "InteractionProbeAgenda":
        candidate_refs = tuple(item.candidate_ref for item in self.candidates)
        if len(set(candidate_refs)) != len(candidate_refs):
            raise ValueError("interaction probe candidate refs must be unique")
        if self.alternative_refs != candidate_refs:
            raise ValueError("interaction alternatives must index every candidate")
        if set(self.alternative_facts) != set(candidate_refs):
            raise ValueError("interaction facts must index every candidate")
        if any(
            not isinstance(self.alternative_facts[item], FrozenMap)
            for item in candidate_refs
        ):
            raise ValueError("interaction facts for each candidate must be a mapping")
        if self.descriptive_delta_facts and set(
            self.descriptive_delta_facts
        ) != set(candidate_refs):
            raise ValueError("interaction deltas must index every candidate")
        if self.has_candidates != bool(self.candidates):
            raise ValueError("has_candidates must reflect the candidate collection")
        if len(set(self.rejected_route_variant_refs)) != len(
            self.rejected_route_variant_refs
        ):
            raise ValueError("rejected route variants must be unique")
        if len(self.rejected_route_variant_refs) != len(
            self.rejected_route_variant_kinds
        ):
            raise ValueError(
                "rejected route variants and failure kinds must be aligned"
            )
        if len(set(self.rejected_grounding_candidate_refs)) != len(
            self.rejected_grounding_candidate_refs
        ):
            raise ValueError("rejected grounding candidates must be unique")
        if len(self.rejected_grounding_candidate_refs) != len(
            self.rejected_grounding_failure_kinds
        ):
            raise ValueError(
                "rejected grounding candidates and failure kinds must be aligned"
            )
        return self


class HomologousRepairAnalysis(FrozenModel):
    """Exact candidate multiscale correspondences; no mapping is chosen here."""

    mapping_alternative_refs: tuple[Ref, ...]
    mapping_alternative_facts: FrozenMap = Field(default_factory=FrozenMap)
    has_mapping_candidates: bool
    schema_version: Ref = "yf_arc3_v5.homologous_repair_analysis.v1"

    @model_validator(mode="after")
    def validate_mapping_index(self) -> "HomologousRepairAnalysis":
        require_unique(
            self.mapping_alternative_refs,
            "homologous repair mapping alternatives",
        )
        if set(self.mapping_alternative_facts) != set(
            self.mapping_alternative_refs
        ):
            raise ValueError("mapping facts must index every mapping alternative")
        if self.has_mapping_candidates != bool(self.mapping_alternative_refs):
            raise ValueError("has_mapping_candidates must reflect mapping alternatives")
        return self


class HomologousRepairInput(FrozenModel):
    scene: VisualSceneDescription | BlockGridSceneDescription
    schema_version: Ref = "yf_arc3_v5.homologous_repair_input.v1"


class LocalConstraintCompositionAnalysis(FrozenModel):
    """Exact bounded compositions; compatible and conflicting witnesses survive."""

    composition_alternative_refs: tuple[Ref, ...]
    composition_alternative_facts: FrozenMap = Field(default_factory=FrozenMap)
    compatible_alternative_refs: tuple[Ref, ...] = ()
    conflicting_alternative_refs: tuple[Ref, ...] = ()
    has_composition_candidates: bool
    schema_version: Ref = "yf_arc3_v5.local_constraint_composition_analysis.v1"

    @model_validator(mode="after")
    def validate_composition_index(self) -> "LocalConstraintCompositionAnalysis":
        require_unique(
            self.composition_alternative_refs,
            "local constraint composition alternatives",
        )
        if set(self.composition_alternative_facts) != set(
            self.composition_alternative_refs
        ):
            raise ValueError("composition facts must index every alternative")
        classified = set(self.compatible_alternative_refs) | set(
            self.conflicting_alternative_refs
        )
        if classified != set(self.composition_alternative_refs):
            raise ValueError("every composition must be compatible or conflicting")
        if set(self.compatible_alternative_refs) & set(
            self.conflicting_alternative_refs
        ):
            raise ValueError("composition compatibility classes must be disjoint")
        if len(self.composition_alternative_refs) > 3:
            raise ValueError("at most three global composition witnesses may survive")
        if self.has_composition_candidates != bool(
            self.composition_alternative_refs
        ):
            raise ValueError("candidate flag must reflect composition alternatives")
        return self


class LocalConstraintCompositionInput(FrozenModel):
    scene: VisualSceneDescription | BlockGridSceneDescription
    transferred_mapping_pairs: tuple[tuple[int, int], ...] = ()
    schema_version: Ref = "yf_arc3_v5.local_constraint_composition_input.v1"


class LocalConstraintRepairAgendaInput(FrozenModel):
    analysis: LocalConstraintCompositionAnalysis
    selected_composition_ref: Ref
    available_action_refs: tuple[Ref, ...] = Field(min_length=1)
    verified_repair_count: int = Field(default=0, ge=0)
    schema_version: Ref = "yf_arc3_v5.local_constraint_repair_agenda_input.v1"


class HomologousRepairAgendaInput(FrozenModel):
    analysis: HomologousRepairAnalysis
    selected_mapping_ref: Ref
    available_action_refs: tuple[Ref, ...] = Field(min_length=1)
    verified_repair_count: int = Field(default=0, ge=0)
    schema_version: Ref = "yf_arc3_v5.homologous_repair_agenda_input.v1"


class HomologousRepairReconciliationInput(FrozenModel):
    before: FrameGrid
    after: FrameGrid
    selected_plan_facts: FrozenMap
    before_score: int = Field(ge=0)
    after_score: int = Field(ge=0)
    transition_ref: Ref
    schema_version: Ref = "yf_arc3_v5.homologous_repair_reconciliation_input.v1"


class HomologousRepairReconciliationAnalysis(FrozenModel):
    descriptive_delta: FrozenMap
    evidence_refs: tuple[Ref, ...] = Field(min_length=1)
    transition_ref: Ref
    schema_version: Ref = "yf_arc3_v5.homologous_repair_reconciliation.v1"

class CalibratedAffineActionEvidence(FrozenModel):
    action_ref: Ref
    full_entity_ref: Ref
    derived_entity_ref: Ref
    fixed_entity_ref: Ref
    relation_ref: Ref
    structure_ref: Ref
    goal_ref: Ref
    enclosure_center_row_twice: int
    enclosure_center_column_twice: int
    action_point_offset_row_twice: int
    action_point_offset_column_twice: int
    evidence_ref: Ref
    symmetric_plan_witnesses: tuple[FrozenMap, ...] = ()
    symmetric_plan_candidate_limit: int = Field(default=3, ge=1, le=3)
    goal_lifecycle_status: Literal["proposed", "pursuing"] = "proposed"
    operational_status: Literal["enabled", "disabled"] = "enabled"


class EffectCarriedReflectionAxisPlanEvidence(FrozenModel):
    context_epoch: int = Field(ge=1)
    measurement_frame_ref: Ref | None = None
    action_ref: Ref
    relation_ref: Ref
    goal_ref: Ref
    carrier_entity_ref: Ref
    carrier_observed_value: int
    moving_support_ref: Ref
    terminal_support_ref: Ref
    negative_support_ref: Ref | None = None
    positive_support_ref: Ref | None = None
    reflection_scope: Literal["exact_support_pair"] | None = None
    role_ref: Ref | None = None
    orientation: Literal["horizontal", "vertical"]
    axis_delta_row: int
    axis_delta_col: int
    remaining_action_count: int = Field(gt=0, le=40)
    terminal_candidate_count: int | None = Field(default=None, ge=0)
    terminal_residual_is_zero: bool | None = None
    evidence_ref: Ref
    commitment: Literal["revisable"] = "revisable"
    palette_equality_required: Literal[False] = False
    operational_status: Literal["enabled", "disabled"] = "enabled"

    @model_validator(mode="after")
    def validate_axis_delta(self) -> "EffectCarriedReflectionAxisPlanEvidence":
        if self.terminal_residual_is_zero is True:
            raise ValueError("a closed terminal residual cannot be an executable axis plan")
        if self.axis_delta_row == 0 and self.axis_delta_col == 0:
            raise ValueError("effect-carried reflection axis delta must be nonzero")
        if self.reflection_scope is not None and not (
            self.negative_support_ref and self.positive_support_ref and self.role_ref
        ):
            raise ValueError("scoped reflection evidence requires its pair and declared role")
        return self


class CalibratedNaryAffineRelationEvidence(FrozenModel):
    action_ref: Ref
    relation_ref: Ref
    derived_entity_ref: Ref
    contributor_entity_refs: tuple[Ref, ...] = Field(min_length=2)
    active_contributor_entity_ref: Ref
    coefficient_numerators: tuple[int, ...] = Field(min_length=2)
    coefficient_denominator: int = Field(gt=0)
    evidence_ref: Ref
    discriminating_transition_count: int = Field(default=1, ge=1)
    operational_status: Literal["enabled", "disabled"] = "enabled"

    @model_validator(mode="after")
    def validate_coefficient_arity(self) -> "CalibratedNaryAffineRelationEvidence":
        if len(self.coefficient_numerators) != len(self.contributor_entity_refs):
            raise ValueError("affine coefficient and contributor arities must match")
        if self.active_contributor_entity_ref not in self.contributor_entity_refs:
            raise ValueError("active affine contributor must belong to contributors")
        return self


class CalibratedLocalPointTransportEvidence(FrozenModel):
    action_ref: Ref
    moving_entity_ref: Ref
    delta_row: int
    delta_col: int
    positive_footprint_distance: int = Field(gt=0)
    evidence_ref: Ref
    commitment: Literal["revisable"] = "revisable"
    operational_status: Literal["enabled", "disabled"] = "enabled"

    @model_validator(mode="after")
    def validate_nonzero_delta(self) -> "CalibratedLocalPointTransportEvidence":
        if self.delta_row == 0 and self.delta_col == 0:
            raise ValueError("local point transport delta must be nonzero")
        return self


class VerifiedMechanismEvidence(FrozenModel):
    actuator_entity_ref: Ref
    action_ref: Ref
    goal_ref: Ref
    moving_entity_ref: Ref
    fixed_entity_ref: Ref
    alignment_axis: Literal["row", "column"]
    relation_to_goal: Literal["toward", "away", "aligned", "unchanged"]
    quantum: int = Field(gt=0)
    alignment_distance_twice: int = Field(ge=0)
    operational_status: Literal["enabled", "disabled"] = "enabled"
    evidence_ref: Ref
    mechanism_kind: Literal["material_transfer", "identity_transport"] = (
        "material_transfer"
    )
    changed_dimension_ref: Ref | None = None
    preserved_dimension_refs: tuple[Ref, ...] = ()
    activation_precondition_kind: Ref | None = None
    activation_threshold: int | None = Field(default=None, ge=0)
    supports_transferred_quantized_plan_seed: bool = False


class VerifiedAlignmentActionInput(FrozenModel):
    canonical_peer_method_facts: tuple[FrozenMap, ...] = Field(default=(), max_length=16)
    canonical_terminal_inventory_facts: tuple[FrozenMap, ...] = Field(default=(), max_length=64)
    canonical_provisional_control_term_facts: tuple[FrozenMap, ...] = Field(default=(), max_length=64)
    # SRC/DRM-declared measurement scope. Python may use it only to skip
    # unrelated candidate families; it never chooses this reference.
    measurement_scope_ref: Ref | None = None
    scene: VisualSceneDescription | BlockGridSceneDescription
    concentric_frame_additions: tuple[tuple[int, int, int, int, int], ...] = Field(default=(), max_length=32)
    observed_level_index: int | None = Field(default=None, ge=1)
    tracking: TemporalTrackingResult
    mechanism_evidence: tuple[VerifiedMechanismEvidence, ...]
    calibrated_affine_evidence: tuple[CalibratedAffineActionEvidence, ...] = ()
    effect_carried_reflection_axis_evidence: tuple[
        EffectCarriedReflectionAxisPlanEvidence, ...
    ] = ()
    # Thin transport of the direct body identity already committed by DRM.
    # Its cardinality distinguishes an unresolved control transfer from a
    # reflection plan whose direct body must remain bound.
    prior_context_orthogonal_reflection_direct_body_values: tuple[int, ...] = ()
    calibrated_nary_affine_evidence: tuple[
        CalibratedNaryAffineRelationEvidence, ...
    ] = ()
    calibrated_local_point_evidence: tuple[
        CalibratedLocalPointTransportEvidence, ...
    ] = ()
    available_action_refs: tuple[Ref, ...] = Field(min_length=1)
    periodic_cell_grids: PeriodicCellGridResult = Field(
        default_factory=lambda: PeriodicCellGridResult(candidates=())
    )
    current_observed_frame: FrameGrid | None = None
    retained_grid_geometry: tuple[int, int, int, int, int, int, int, int] | None = None
    retained_grid_ref: Ref | None = None
    retained_grid_separator_consistency_ppm: int = Field(
        default=0, ge=0, le=1_000_000
    )
    retained_grid_recount: KnownGridCellRecountResult | None = None
    retained_action_quantized_grid_quantum: int = Field(default=0, ge=0, le=32)
    action_conditioned_grid_evidence: bool = False
    cellular_mover_pattern_refs: tuple[Ref, ...] = ()
    recent_moved_cell_pattern_refs: tuple[Ref, ...] = Field(
        default=(), max_length=3
    )
    cellular_actuator_pattern_refs: tuple[Ref, ...] = ()
    effectful_point_cell_pattern_pairs: tuple[tuple[Ref, Ref], ...] = Field(
        default=(), max_length=3
    )
    cellular_traversable_pattern_refs: tuple[Ref, ...] = ()
    # Current action delta only; used for bounded exact comparison between a
    # changed display sprite and stable periodic-cell sprites.
    current_transition_change_bboxes: tuple[BoundingBox, ...] = Field(
        default=(), max_length=3
    )
    # Positive action-conditioned support pairs
    # (moving_pattern_ref, terrain_pattern_ref). Empty preserves the exact
    # legacy case where one measured support set applies to every mover.
    cellular_entity_traversable_pattern_refs: tuple[tuple[Ref, Ref], ...] = Field(
        default=(), max_length=64
    )
    # Complete logical-cell footprint offsets per moving pattern. Missing
    # entries mean the measured singleton footprint ((0, 0),).
    cellular_entity_footprint_offsets: tuple[tuple[Ref, int, int], ...] = Field(
        default=(), max_length=128
    )
    open_contact_pattern_refs: tuple[Ref, ...] = Field(default=(), max_length=3)
    # Established DRM contact constraints, transported as exact sprite/face
    # pairs for mechanical route clearance. Proposed color analogies stay in
    # the cognitive frontier and do not become a simulator fact.
    established_hostile_contact_pattern_faces: tuple[tuple[Ref, str], ...] = Field(
        default=(), max_length=16
    )
    # An already-fatal primitive in this measured initial cell configuration.
    # Search may remove that exact first edge; later states are reconciled.
    observed_fatal_initial_action_facts: tuple[tuple[Ref, FrozenMap], ...] = Field(
        default=(), max_length=16
    )
    known_open_cell_coordinates: tuple[tuple[int, int], ...] = Field(
        default=(), max_length=32
    )
    newly_reachable_cell_coordinates: tuple[tuple[int, int], ...] = Field(
        default=(), max_length=32
    )
    configuration_revisit_action_refs: tuple[Ref, ...] = ()
    # Exact candidate refs already dispatched from the current non-HUD spatial
    # configuration at the current score. Python only measures membership;
    # DRM owns whether the candidate remains eligible.
    current_configuration_executed_candidate_refs: tuple[Ref, ...] = ()
    committed_plan_facts: FrozenMap | None = None
    committed_plan_cursor_present: bool = False
    # Exact current-frame route measurement used only to falsify a stale
    # committed object-terminating cursor before its next primitive.
    current_object_terminating_route_measurement_present: bool = False
    current_object_terminating_route_first_action_refs: tuple[Ref, ...] = Field(
        default=(), max_length=4
    )
    # Exact SRC lifecycle fact: the committed repetition consumed its final
    # primitive in the current reconciled observation.  Capabilities may use
    # it only to avoid mechanically replaying that exhausted cursor.
    verified_repetition_cursor_end_observed: bool = False
    continuation_policy: FrozenMap = Field(default_factory=FrozenMap)
    route_generation_policy: FrozenMap = Field(default_factory=FrozenMap)
    current_context_no_effect_candidate_refs: tuple[Ref, ...] = ()
    terminally_exhausted_action_refs: tuple[Ref, ...] = Field(
        default=(), max_length=3
    )
    action_execution_counts: tuple[tuple[Ref, int], ...] = ()
    # Exact causal ordinal of every first execution. The route capability
    # measures agreement with this sequence; DRM owns the preference.
    action_first_execution_ordinals: tuple[tuple[Ref, int], ...] = ()
    resource_only_effect_action_refs: tuple[Ref, ...] = ()
    cyclic_pose_only_effect_action_refs: tuple[Ref, ...] = ()
    current_scope_attempted_route_suffixes: tuple[Ref, ...] = Field(
        default=(), max_length=3
    )
    # Measured controller deltas (action_ref, delta_row, delta_col).
    known_action_translation_deltas: tuple[tuple[Ref, int, int], ...] = ()
    # Directional intent declared by the official input interface.  A
    # measured transition may also contain an orthogonal automatic settle;
    # bounded route search keeps the declared pre-settle entry axis separate
    # from that observed post-settle displacement.
    interface_action_translation_deltas: tuple[tuple[Ref, int, int], ...] = ()
    # Measured viewport-grid deltas, never inferred world coordinates.
    known_viewport_grid_shift_deltas: tuple[tuple[Ref, int, int], ...] = Field(
        default=(), max_length=32
    )
    residual_axis_coupling_row: int = Field(default=1)
    residual_axis_coupling_column: int = Field(default=1)
    action_conditioned_marker_axis_overlap_reconstructed: bool = False
    marker_axis_residual_measurement_present: bool = False
    marker_axis_residual_transport_present: bool = False
    marker_axis_residual_row_signed: int = 0
    marker_axis_residual_col_signed: int = 0
    dynamic_goal_refs: tuple[Ref, ...] = Field(default=(), max_length=4096)
    dynamic_goal_facts: FrozenMap = Field(default_factory=FrozenMap)
    schema_version: Ref = "yf_arc3_v5.verified_alignment_action_input.v1"

    @model_validator(mode="after")
    def validate_route_generation_scope(self) -> "VerifiedAlignmentActionInput":
        if len(set(self.dynamic_goal_refs)) != len(self.dynamic_goal_refs):
            raise ValueError("dynamic goal refs must be unique")
        if set(self.dynamic_goal_facts) != set(self.dynamic_goal_refs):
            raise ValueError("dynamic goal facts must index every dynamic goal")
        if any(
            not isinstance(self.dynamic_goal_facts[item], FrozenMap)
            for item in self.dynamic_goal_refs
        ):
            raise ValueError("dynamic goal facts must be immutable mappings")
        if len(set(self.current_context_no_effect_candidate_refs)) != len(
            self.current_context_no_effect_candidate_refs
        ):
            raise ValueError("current-context no-effect candidate refs must be unique")
        if len(set(self.current_scope_attempted_route_suffixes)) != len(
            self.current_scope_attempted_route_suffixes
        ):
            raise ValueError("current-scope attempted route suffixes must be unique")
        if len(
            set(self.current_object_terminating_route_first_action_refs)
        ) != len(self.current_object_terminating_route_first_action_refs):
            raise ValueError(
                "current object-terminating route first-action refs must be unique"
            )
        first_ordinal_refs = tuple(
            ref for ref, _ordinal in self.action_first_execution_ordinals
        )
        first_ordinals = tuple(
            ordinal for _ref, ordinal in self.action_first_execution_ordinals
        )
        if len(set(first_ordinal_refs)) != len(first_ordinal_refs):
            raise ValueError("action first-execution refs must be unique")
        if any(ordinal < 0 for ordinal in first_ordinals):
            raise ValueError("action first-execution ordinals must be non-negative")
        if len(set(first_ordinals)) != len(first_ordinals):
            raise ValueError("action first-execution ordinals must be unique")
        return self


class CommittedVerifiedAlignmentCursorInput(FrozenModel):
    """Minimal mechanical payload for one DRM/SRC-committed cursor advance."""

    tracking: TemporalTrackingResult
    canonical_peer_method_facts: tuple[FrozenMap, ...] = Field(default=(), max_length=16)
    canonical_terminal_inventory_facts: tuple[FrozenMap, ...] = Field(default=(), max_length=64)
    canonical_provisional_control_term_facts: tuple[FrozenMap, ...] = Field(default=(), max_length=64)
    committed_plan_facts: FrozenMap
    available_action_refs: tuple[Ref, ...] = Field(min_length=1)
    # Fresh current-frame route evidence must travel with the compact cursor
    # payload too.  The shared cursor advancer uses it to reject a stale next
    # primitive when the visible object route has changed.
    current_object_terminating_route_measurement_present: bool = False
    current_object_terminating_route_first_action_refs: tuple[Ref, ...] = Field(
        default=(), max_length=4
    )
    current_context_no_effect_candidate_refs: tuple[Ref, ...] = ()
    current_configuration_executed_candidate_refs: tuple[Ref, ...] = ()
    action_execution_counts: tuple[tuple[Ref, int], ...] = ()
    resource_only_effect_action_refs: tuple[Ref, ...] = ()
    cyclic_pose_only_effect_action_refs: tuple[Ref, ...] = ()
    known_action_translation_deltas: tuple[tuple[Ref, int, int], ...] = ()
    interface_action_translation_deltas: tuple[tuple[Ref, int, int], ...] = ()
    residual_axis_coupling_row: int = 1
    residual_axis_coupling_column: int = 1
    route_generation_policy: FrozenMap = Field(default_factory=FrozenMap)
    schema_version: Ref = "yf_arc3_v5.committed_verified_alignment_cursor_input.v1"


class RouteCompatibleGraphPlanInput(VerifiedAlignmentActionInput):
    """One DRM-declared route order plus immutable visual measurements."""

    route_order_analysis: InteractionProbeAgenda
    declared_route_order_ref: Ref
    schema_version: Ref = "yf_arc3_v5.route_compatible_graph_plan_input.v1"

    @model_validator(mode="after")
    def validate_declared_route_order(self) -> "RouteCompatibleGraphPlanInput":
        if self.declared_route_order_ref not in self.route_order_analysis.alternative_refs:
            raise ValueError("declared route order must index the measured alternatives")
        if not self.route_order_analysis.has_candidates:
            raise ValueError("declared route order requires measured alternatives")
        return self


class QuantizedTransferControlMeasurement(FrozenModel):
    source_index: int = Field(ge=0)
    destination_index: int = Field(ge=0)
    component_ref: Ref
    schema_version: Ref = "yf_arc3_v5.quantized_transfer_control_measurement.v1"


class QuantizedInterfaceMeasurement(FrozenModel):
    interface_ref: Ref
    lower_index: int = Field(ge=0)
    upper_index: int = Field(ge=0)
    activation_threshold: int = Field(ge=0)
    schema_version: Ref = "yf_arc3_v5.quantized_interface_measurement.v1"


class CompositeIdentityRouteMeasurement(FrozenModel):
    identity_ref: Ref
    carrier_entity_ref: Ref
    attribute_entity_ref: Ref
    current_zone_index: int = Field(ge=0)
    target_zone_index: int = Field(ge=0)
    attribute_offset: Position
    schema_version: Ref = "yf_arc3_v5.composite_identity_route_measurement.v1"


class MultiIdentityOpposedRoutesInput(FrozenModel):
    """Exact measured chain supplied to a bounded counterfactual simulator."""

    material_refs: tuple[Ref, ...] = Field(min_length=2)
    initial_quantities: tuple[int, ...] = Field(min_length=2)
    capacities: tuple[int, ...] = Field(min_length=2)
    terminal_quantities: tuple[int, ...] = ()
    terminal_quantity_requirements: tuple[tuple[int, int], ...] = ()
    controls: tuple[QuantizedTransferControlMeasurement, ...] = Field(min_length=2)
    interfaces: tuple[QuantizedInterfaceMeasurement, ...] = Field(
        min_length=1, max_length=3
    )
    identities: tuple[CompositeIdentityRouteMeasurement, ...] = Field(
        min_length=2, max_length=2
    )
    control_order_variant: Literal[
        "canonical", "reverse_canonical", "canonical_rotation_1"
    ]
    max_witnesses: int = Field(default=3, ge=1, le=3)
    max_expanded_states_per_stage: int = Field(default=4096, ge=1, le=4096)
    max_actions_per_witness: int = Field(default=128, ge=1, le=128)
    schema_version: Ref = "yf_arc3_v5.multi_identity_opposed_routes_input.v2"

    @model_validator(mode="after")
    def validate_exact_chain(self) -> "MultiIdentityOpposedRoutesInput":
        zone_count = len(self.material_refs)
        if not len(self.initial_quantities) == len(self.capacities) == zone_count:
            raise ValueError("quantity and capacity vectors must index every material")
        exact_terminal = bool(self.terminal_quantities)
        partial_terminal = bool(self.terminal_quantity_requirements)
        if exact_terminal == partial_terminal:
            raise ValueError(
                "exactly one exact or partial terminal constraint is required"
            )
        if exact_terminal and len(self.terminal_quantities) != zone_count:
            raise ValueError("exact terminal quantities must index every material")
        require_unique(self.material_refs, "material references")
        require_unique(
            tuple(item.component_ref for item in self.controls),
            "transfer control references",
        )
        require_unique(
            tuple(item.interface_ref for item in self.interfaces),
            "interface references",
        )
        require_unique(
            tuple(item.identity_ref for item in self.identities),
            "identity references",
        )
        if any(
            value < 0
            for value in (
                *self.initial_quantities,
                *self.capacities,
                *self.terminal_quantities,
                *(value for _index, value in self.terminal_quantity_requirements),
            )
        ):
            raise ValueError("quantities and capacities must be non-negative")
        if any(
            value > capacity
            for values in (self.initial_quantities, self.terminal_quantities)
            for value, capacity in zip(values, self.capacities)
        ):
            raise ValueError("initial and terminal quantities must respect capacities")
        if exact_terminal and sum(self.initial_quantities) != sum(self.terminal_quantities):
            raise ValueError("terminal quantities must conserve the measured aggregate")
        expected_edges = tuple((index, index + 1) for index in range(zone_count - 1))
        observed_edges = tuple(
            sorted((item.lower_index, item.upper_index) for item in self.interfaces)
        )
        if observed_edges != expected_edges:
            raise ValueError("interfaces must form one complete consecutive chain")
        if any(
            item.source_index >= zone_count
            or item.destination_index >= zone_count
            or item.source_index == item.destination_index
            for item in self.controls
        ):
            raise ValueError("transfer controls must reference distinct known zones")
        directed_edges = {
            (item.source_index, item.destination_index) for item in self.controls
        }
        if any(
            (left, right) not in directed_edges or (right, left) not in directed_edges
            for left, right in expected_edges
        ):
            raise ValueError("every adjacent material pair requires bidirectional controls")
        ordered_identities = sorted(self.identities, key=lambda item: item.current_zone_index)
        if (
            ordered_identities[0].current_zone_index != 0
            or ordered_identities[0].target_zone_index != zone_count - 1
            or ordered_identities[1].current_zone_index != zone_count - 1
            or ordered_identities[1].target_zone_index != 0
        ):
            raise ValueError("the two measured routes must be exact opposites on the chain")
        if partial_terminal:
            requirement_indices = tuple(
                int(index) for index, _value in self.terminal_quantity_requirements
            )
            require_unique(requirement_indices, "terminal quantity requirement indices")
            target_indices = tuple(
                sorted(item.target_zone_index for item in self.identities)
            )
            if tuple(sorted(requirement_indices)) != target_indices:
                raise ValueError(
                    "partial terminal constraints must cover both measured identity targets"
                )
            if len(requirement_indices) >= zone_count:
                raise ValueError("partial terminal constraints must leave a residual unknown")
            requirement_total = sum(
                int(value) for _index, value in self.terminal_quantity_requirements
            )
            aggregate = sum(self.initial_quantities)
            if requirement_total > aggregate:
                raise ValueError("partial terminal requirements exceed conserved aggregate")
            requirement_by_index = dict(self.terminal_quantity_requirements)
            if any(
                index < 0
                or index >= zone_count
                or value > self.capacities[index]
                for index, value in sorted(requirement_by_index.items())
            ):
                raise ValueError("partial terminal requirements must respect capacities")
            residual_capacity = sum(
                capacity
                for index, capacity in enumerate(self.capacities)
                if index not in requirement_by_index
            )
            if aggregate - requirement_total > residual_capacity:
                raise ValueError(
                    "conserved terminal residual exceeds unknown-zone capacity"
                )
        return self


class MultiIdentityRouteStep(FrozenModel):
    step_index: int = Field(ge=1)
    stage: Literal[
        "pre_crossing", "atomic_crossing", "post_crossing", "terminal_alignment"
    ]
    operation_kind: Literal["quantity_transfer", "interface_transition"]
    component_ref: Ref
    source_index: int = Field(ge=0)
    destination_index: int = Field(ge=0)
    moved_identity_refs: tuple[Ref, ...] = Field(default=(), max_length=2)
    quantities_after: tuple[int, ...] = Field(min_length=2)
    identity_zone_indices_after: tuple[tuple[Ref, int], ...] = Field(min_length=2)
    composite_offsets_after: tuple[tuple[Ref, Position], ...] = Field(min_length=2)
    schema_version: Ref = "yf_arc3_v5.multi_identity_route_step.v1"


class MultiIdentityRouteWitness(FrozenModel):
    witness_ref: Ref
    crossing_interface_ref: Ref
    crossing_interface_index: int = Field(ge=0)
    steps: tuple[MultiIdentityRouteStep, ...] = Field(min_length=1, max_length=128)
    material_action_count: int = Field(ge=0)
    interface_event_count: int = Field(ge=1)
    primitive_action_count: int = Field(ge=1, le=128)
    maximum_remaining_route_distance_after_crossing: int = Field(ge=0)
    remaining_route_distance_imbalance_after_crossing: int = Field(ge=0)
    expanded_state_count: int = Field(ge=0)
    aggregate_conserved: bool
    all_intermediate_states_valid: bool
    all_identity_transports_before_terminal_alignment: bool
    atomic_crossing_identity_count: int = Field(ge=0, le=2)
    final_quantities: tuple[int, ...] = Field(min_length=2)
    final_identity_zone_indices: tuple[tuple[Ref, int], ...] = Field(min_length=2)
    schema_version: Ref = "yf_arc3_v5.multi_identity_route_witness.v1"


class MultiIdentityRouteRejection(FrozenModel):
    interface_ref: Ref
    failure_kind: Literal[
        "material_preparation_unreachable", "action_bound_reached", "state_bound_reached"
    ]
    schema_version: Ref = "yf_arc3_v5.multi_identity_route_rejection.v1"


class MultiIdentityOpposedRoutesAnalysis(FrozenModel):
    descriptor_kind: Literal["multi_identity_opposed_routes"] = (
        "multi_identity_opposed_routes"
    )
    witnesses: tuple[MultiIdentityRouteWitness, ...] = Field(max_length=3)
    rejections: tuple[MultiIdentityRouteRejection, ...] = Field(max_length=3)
    witness_limit: int = Field(ge=1, le=3)
    schema_version: Ref = "yf_arc3_v5.multi_identity_opposed_routes_analysis.v1"


class CommittedQuantizedPlanAdvanceInput(FrozenModel):
    """Small immutable cursor advance over one already committed simulation."""

    committed_plan_facts: FrozenMap
    available_action_refs: tuple[Ref, ...] = Field(min_length=1)
    continuation_policy: FrozenMap = Field(default_factory=FrozenMap)
    schema_version: Ref = "yf_arc3_v5.committed_quantized_plan_advance_input.v1"


class IgnoredSceneRegion(FrozenModel):
    """A previously grounded component region ignored by a fast guard."""

    entity_ref: Ref
    reason_kind: Ref
    pixels: tuple[Position, ...] = Field(min_length=1)
    schema_version: Ref = "yf_arc3_v5.ignored_scene_region.v1"

    @model_validator(mode="after")
    def validate_unique_pixels(self) -> "IgnoredSceneRegion":
        if len(set(self.pixels)) != len(self.pixels):
            raise ValueError("ignored scene region pixels must be unique")
        return self


class IntermediateSceneChangeInput(FrozenModel):
    """Frames and already grounded exclusions for a cheap CONTINUE guard."""

    before: FrameGrid
    after: FrameGrid
    ignored_regions: tuple[IgnoredSceneRegion, ...] = ()
    permitted_regions: tuple[IgnoredSceneRegion, ...] = ()
    action_ref: Ref | None = None
    # DRM-owned cycle workflow facts transported as immutable observations.
    # The capability only performs the exact universal-status measurement;
    # policy decides whether CONTINUE may be released.
    periodic_cycle_workflow_facts: tuple[FrozenMap, ...] = Field(
        default=(), max_length=64
    )
    schema_version: Ref = "yf_arc3_v5.intermediate_scene_change_input.v1"

    @model_validator(mode="after")
    def validate_periodic_cycle_workflow_facts(self) -> "IntermediateSceneChangeInput":
        allowed = {status.value for status in PeriodicCycleWorkflowStatus}
        for fact in self.periodic_cycle_workflow_facts:
            status = fact.get("workflow_status")
            if status not in allowed:
                raise ValueError(
                    "periodic cycle workflow facts require a declared workflow_status"
                )
            if not fact.get("cycle_ref"):
                raise ValueError("periodic cycle workflow facts require cycle_ref")
        return self


class IntermediateSceneChangeMeasurements(FrozenModel):
    """Exact changed-pixel counts; selection remains declarative in DRM."""

    alternative_refs: tuple[Ref, ...] = Field(min_length=3)
    alternative_facts: FrozenMap
    raw_changed_count: int = Field(ge=0)
    ignored_changed_count: int = Field(ge=0)
    meaningful_changed_count: int = Field(ge=0)
    permitted_region_count: int = Field(ge=0)
    permitted_changed_count: int = Field(ge=0)
    outside_permitted_changed_count: int = Field(ge=0)
    ignored_region_refs: tuple[Ref, ...] = ()
    action_ref: Ref | None = None
    periodic_cycle_workflow_statuses: tuple[Ref, ...] = ()
    all_periodic_cycles_understood: bool = True
    schema_version: Ref = "yf_arc3_v5.intermediate_scene_change_measurements.v1"

    @model_validator(mode="after")
    def validate_alternative_index(self) -> "IntermediateSceneChangeMeasurements":
        require_unique(self.alternative_refs, "intermediate scene alternatives")
        require_unique(self.ignored_region_refs, "ignored scene region refs")
        if set(self.alternative_facts) != set(self.alternative_refs):
            raise ValueError("intermediate scene facts must index every alternative")
        if self.ignored_changed_count + self.meaningful_changed_count != self.raw_changed_count:
            raise ValueError("intermediate scene change counts must partition raw changes")
        if (
            self.permitted_changed_count + self.outside_permitted_changed_count
            != self.meaningful_changed_count
        ):
            raise ValueError(
                "permitted and outside-permitted counts must partition meaningful changes"
            )
        return self


class PixelChange(FrozenModel):
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    before: int
    after: int


class FrameDifferenceInput(FrozenModel):
    before: FrameGrid
    after: FrameGrid
    action_ref: Ref | None = None
    schema_version: Ref = "yf_arc3_v5.frame_difference_input.v1"


class FrameDifferenceResult(FrozenModel):
    changes: tuple[PixelChange, ...]
    changed_count: int = Field(ge=0)
    unchanged_count: int = Field(ge=0)
    changed_bbox: BoundingBox | None = None
    action_ref: Ref | None = None
    schema_version: Ref = "yf_arc3_v5.frame_difference_result.v1"


class SceneTransitionMeasurementInput(FrozenModel):
    """Frames supplied to a neutral transition measurement capability."""

    before: FrameGrid
    intermediate_frames: tuple[FrameGrid, ...] = ()
    after: FrameGrid
    # User-confirmed geometric search band, not a palette or semantic role.
    boundary_search_band_px: int = Field(default=7, ge=1, le=64)
    transition_ref: Ref
    action_ref: Ref | None = None
    action_data: FrozenMap = FrozenMap()
    candidate_ref: Ref | None = None
    context_epoch: int | None = None
    before_configuration_digest: Ref | None = None
    after_configuration_digest: Ref | None = None
    before_available_action_set_digest: Ref | None = None
    boundary_indicator_quantum_delta_measurement: int = Field(
        default=0, ge=-128, le=128
    )
    established_boundary_indicator_decrement_changed_pixel_count_measurement: int = (
        Field(default=0, ge=0, le=4096)
    )
    declared_boundary_indicator_decrease_changed_pixel_count_measurement: int = Field(
        default=0, ge=0, le=4096
    )
    current_effect_signature_known: bool = False
    current_effect_signature_known_cycle: bool = False
    known_effect_signature_count: int = Field(default=0, ge=0)
    known_cycle_transition_signature_count: int = Field(default=0, ge=0)
    before_score: int = 0
    after_score: int | None = None
    schema_version: Ref = "yf_arc3_v5.scene_transition_measurement_input.v8"


class PixelTransitionDelta(FrozenModel):
    """Compact exact pixel delta for one ordered environment frame packet."""

    transition_ref: Ref
    action_ref: Ref | None = None
    packet_dimensions_consistent: bool
    intermediate_frame_count: int = Field(ge=0)
    consecutive_changed_counts: tuple[int, ...] = Field(min_length=1)
    intermediate_changed_counts_from_before: tuple[int, ...] = ()
    intermediate_changed_counts_to_after: tuple[int, ...] = ()
    final_changed_count: int = Field(ge=0)
    changed_bbox: BoundingBox | None = None
    changed_edges: int = Field(
        ge=0,
        description="Ordered consecutive frame pairs containing a pixel change.",
    )
    changed_transition_pair_count: int = Field(ge=0)
    potential_nonzero_translation_count: int = Field(ge=0)
    changed_union_is_collinear: bool
    changed_union_touches_boundary: bool
    final_scene_equivalent_with_intermediate_change: bool
    schema_version: Ref = "yf_arc3_v5.pixel_transition_delta.v2"

    @model_validator(mode="after")
    def validate_packet_counts(self) -> "PixelTransitionDelta":
        if len(self.consecutive_changed_counts) != self.intermediate_frame_count + 1:
            raise ValueError("consecutive change counts must cover the whole frame packet")
        if (
            len(self.intermediate_changed_counts_from_before)
            != self.intermediate_frame_count
            or len(self.intermediate_changed_counts_to_after)
            != self.intermediate_frame_count
        ):
            raise ValueError("intermediate change counts must index every intermediate frame")
        return self


class SceneTransitionMeasurements(FrozenModel):
    """Exact facts for declarative scene-equivalence alternatives.

    The capability exposes both alternatives and never chooses between them.
    Selection belongs to the DRM policy invoked from SRC.
    """

    alternative_refs: tuple[Ref, ...] = Field(min_length=2)
    alternative_facts: FrozenMap
    # Local SRC/DRM projection payload.  It remains available to the in-process
    # evaluator but is excluded from public/result serialization so exact path
    # measurements are not duplicated in diagnostic or model-server payloads.
    ordered_packet_facts: FrozenMap = Field(default_factory=FrozenMap, exclude=True)
    changed_count: int = Field(ge=0)
    potential_nonzero_translation_count: int = Field(ge=0)
    pixel_delta: PixelTransitionDelta
    action_ref: Ref | None = None
    candidate_ref: Ref | None = None
    before_configuration_digest: Ref | None = None
    after_configuration_digest: Ref | None = None
    before_available_action_set_digest: Ref | None = None
    measured_delta_signature_ref: Ref | None = None
    configuration_transition_changed: bool | None = None
    score_delta: int | None = None
    effect_signature_novelty_measurement: EffectSignatureNovelty = EffectSignatureNovelty.NONE
    observable_change_scope_measurement: ObservableChangeScope = ObservableChangeScope.NONE
    schema_version: Ref = "yf_arc3_v5.scene_transition_measurements.v5"

    @model_validator(mode="after")
    def validate_alternative_index(self) -> "SceneTransitionMeasurements":
        require_unique(self.alternative_refs, "scene transition alternatives")
        if set(self.alternative_facts) != set(self.alternative_refs):
            raise ValueError("scene transition facts must index every alternative")
        if any(
            not isinstance(self.alternative_facts[item], FrozenMap)
            for item in self.alternative_refs
        ):
            raise ValueError("scene transition alternative facts must be mappings")
        if self.changed_count != self.pixel_delta.final_changed_count:
            raise ValueError("scene and pixel-delta changed counts must agree")
        if (
            self.potential_nonzero_translation_count
            != self.pixel_delta.potential_nonzero_translation_count
        ):
            raise ValueError("scene and pixel-delta motion witnesses must agree")
        return self


class EffectContextKey(FrozenModel):
    """Exact context identity used to scope effect observations."""

    configuration_digest: Ref
    available_action_set_digest: Ref | None = None
    score_or_level_revision: Ref | None = None
    resource_state_digest: Ref | None = None
    mode_or_pose_state_refs: tuple[Ref, ...] = ()
    candidate_payload_identity: Ref | None = None
    world_revision_ref: Ref | None = None
    schema_version: Ref = "yf_arc3_v5.effect_context_key.v1"


class EffectObservation(FrozenModel):
    """Immutable, deduplicable action-conditioned effect observation."""

    observation_ref: Ref
    action_ref: Ref
    exact_payload_ref: Ref
    before_context_ref: Ref
    after_context_ref: Ref | None = None
    measured_delta_signature_ref: Ref
    signature_novelty: EffectSignatureNovelty
    observable_change_scope: ObservableChangeScope
    declared_causal_class_ref: DeclaredCausalClass | None = None
    affected_entity_refs: tuple[Ref, ...] = ()
    affected_relation_refs: tuple[Ref, ...] = ()
    resource_delta: int | None = None
    score_delta: int | None = None
    configuration_novelty_ref: Ref | None = None
    support_claim_refs: tuple[Ref, ...] = ()
    competing_interpretation_refs: tuple[Ref, ...] = ()
    falsifier_refs: tuple[Ref, ...] = ()
    observation_count: int = Field(default=1, ge=1)
    first_observed_revision: NonNegativeRevision = 0
    last_observed_revision: NonNegativeRevision = 0
    schema_version: Ref = "yf_arc3_v5.effect_observation.v1"

    @model_validator(mode="after")
    def validate_revisions(self) -> "EffectObservation":
        if self.last_observed_revision < self.first_observed_revision:
            raise ValueError("effect observation revisions must be ordered")
        return self


class EffectContextAggregate(FrozenModel):
    """Bounded counters for one action/signature/context family."""

    aggregate_ref: Ref
    action_ref: Ref
    exact_payload_refs: tuple[Ref, ...] = Field(max_length=16)
    context_refs: tuple[Ref, ...] = Field(max_length=32)
    known_signature_refs: tuple[Ref, ...] = Field(max_length=32)
    novel_signature_refs: tuple[Ref, ...] = Field(max_length=32)
    no_effect_observation_count: int = Field(default=0, ge=0)
    known_effect_observation_count: int = Field(default=0, ge=0)
    novel_effect_observation_count: int = Field(default=0, ge=0)
    execution_count: int = Field(default=0, ge=0)
    reversible: bool | None = None
    schema_version: Ref = "yf_arc3_v5.effect_context_aggregate.v1"


class EffectObservationMemory(FrozenModel):
    """Canonical immutable effect ledger projected from TERM/CLAIM."""

    observations: tuple[EffectObservation, ...] = Field(max_length=512)
    aggregates: tuple[EffectContextAggregate, ...] = Field(max_length=256)
    cycle_edge_refs: tuple[Ref, ...] = Field(max_length=256)
    schema_version: Ref = "yf_arc3_v5.effect_observation_memory.v1"

    @model_validator(mode="after")
    def validate_observations(self) -> "EffectObservationMemory":
        require_unique(
            tuple(item.observation_ref for item in self.observations),
            "effect observation refs",
        )
        require_unique(
            tuple(item.aggregate_ref for item in self.aggregates),
            "effect aggregate refs",
        )
        return self


class GoalOutcomeEnvelope(FrozenModel):
    """Declared best/worst credible consequences for a candidate plan."""

    strongest_goal_advanced_refs: tuple[Ref, ...] = ()
    prerequisite_or_goal_refs_unlocked: tuple[Ref, ...] = ()
    alternatives_falsifiable_refs: tuple[Ref, ...] = ()
    expected_effect_novelty: EffectSignatureNovelty = EffectSignatureNovelty.NONE
    expected_configuration_novelty: bool = False
    invariant_loss_refs: tuple[Ref, ...] = ()
    irreversible_state_change_refs: tuple[Ref, ...] = ()
    resource_exhaustion_refs: tuple[Ref, ...] = ()
    stronger_goal_blocked_refs: tuple[Ref, ...] = ()
    access_loss_refs: tuple[Ref, ...] = ()
    exact_context_no_effect: bool = False
    recovery_action_bound: int | None = Field(default=None, ge=0)
    schema_version: Ref = "yf_arc3_v5.goal_outcome_envelope.v1"


class OutcomeEnvelope(GoalOutcomeEnvelope):
    """A consequence envelope, deliberately not a goal node."""

    schema_version: Ref = "yf_arc3_v5.outcome_envelope.v1"


class StaticKnowledgeNode(FrozenModel):
    """One reusable principle/question node, independent of a scene."""

    knowledge_ref: Ref
    node_kind: StaticKnowledgeNodeKind
    parent_knowledge_refs: tuple[Ref, ...] = ()
    activation_condition_refs: tuple[Ref, ...] = ()
    expected_effect_refs: tuple[Ref, ...] = ()
    falsifier_refs: tuple[Ref, ...] = ()
    dependency_refs: tuple[Ref, ...] = ()
    schema_version: Ref = "yf_arc3_v5.static_knowledge_node.v1"


class StaticKnowledgeEdge(FrozenModel):
    """A generic relation; it cannot carry instance lifecycle or coordinates."""

    edge_ref: Ref
    edge_kind: StaticKnowledgeEdgeKind
    source_ref: Ref
    target_ref: Ref
    schema_version: Ref = "yf_arc3_v5.static_knowledge_edge.v1"


class StaticKnowledgePriority(FrozenModel):
    """One reusable ordered priority declaration from DRM.

    The list is concrete and ordered, but it remains scene-independent.  A
    dynamic priority node may select a different list after scene evidence.
    """

    priority_ref: Ref
    ordered_alternative_refs: tuple[Ref, ...] = Field(min_length=1, max_length=64)
    activation_condition_refs: tuple[Ref, ...] = ()
    expected_effect_refs: tuple[Ref, ...] = ()
    falsifier_refs: tuple[Ref, ...] = ()
    dependency_refs: tuple[Ref, ...] = ()
    schema_version: Ref = "yf_arc3_v5.static_knowledge_priority.v1"

    @model_validator(mode="after")
    def validate_priority_order(self) -> "StaticKnowledgePriority":
        require_unique(self.ordered_alternative_refs, "static priority alternatives")
        return self


class StaticKnowledgeGraph(FrozenModel):
    """Acyclic reusable knowledge tree, separate from the live goal graph."""

    knowledge_graph_ref: Ref = "knowledge.graph.generic"
    # The migration adds a bounded obligation vocabulary beside the existing
    # mechanism vocabulary. Keep an explicit ceiling while allowing both
    # source-hashed declarative sets to coexist during the shadow phase.
    # Compile-time catalogue bound, not per-frame cognitive memory.  The
    # editable v0.2 tree adds source-declared lazy gates while live transport
    # remains capped separately at 10 KiB.
    nodes: tuple[StaticKnowledgeNode, ...] = Field(max_length=1024)
    edges: tuple[StaticKnowledgeEdge, ...] = Field(max_length=1024)
    priorities: tuple[StaticKnowledgePriority, ...] = Field(default=(), max_length=64)
    root_knowledge_refs: tuple[Ref, ...] = Field(min_length=1)
    schema_version: Ref = "yf_arc3_v5.static_knowledge_graph.v1"

    @model_validator(mode="after")
    def validate_tree(self) -> "StaticKnowledgeGraph":
        node_refs = tuple(item.knowledge_ref for item in self.nodes)
        edge_refs = tuple(item.edge_ref for item in self.edges)
        priority_refs = tuple(item.priority_ref for item in self.priorities)
        require_unique(node_refs, "static knowledge refs")
        require_unique(edge_refs, "static knowledge edge refs")
        require_unique(priority_refs, "static knowledge priority refs")
        node_set = set(node_refs)
        if not set(self.root_knowledge_refs) <= node_set:
            raise ValueError("static knowledge roots must reference known nodes")
        if any(
            ref not in node_set
            for node in self.nodes
            for ref in (
                *node.parent_knowledge_refs,
                *node.activation_condition_refs,
                *node.expected_effect_refs,
                *node.falsifier_refs,
                *node.dependency_refs,
            )
        ):
            raise ValueError("static knowledge links must reference known nodes")
        if any(
            ref not in node_set
            for priority in self.priorities
            for ref in (
                *priority.activation_condition_refs,
                *priority.expected_effect_refs,
                *priority.falsifier_refs,
                *priority.dependency_refs,
            )
        ):
            raise ValueError("static priority links must reference known nodes")
        if any(
            edge.source_ref not in node_set or edge.target_ref not in node_set
            for edge in self.edges
        ):
            raise ValueError("static knowledge edges must reference known nodes")

        edge_adjacency: dict[str, tuple[str, ...]] = {ref: () for ref in node_refs}
        for edge in self.edges:
            edge_adjacency[edge.source_ref] = tuple(
                sorted(set(edge_adjacency[edge.source_ref]) | {edge.target_ref})
            )
        edge_visiting: set[str] = set()
        edge_visited: set[str] = set()

        def visit_edge(ref: str) -> None:
            if ref in edge_visiting:
                raise ValueError("static knowledge edges must be acyclic")
            if ref in edge_visited:
                return
            edge_visiting.add(ref)
            for target in edge_adjacency[ref]:
                visit_edge(target)
            edge_visiting.remove(ref)
            edge_visited.add(ref)

        for ref in node_refs:
            visit_edge(ref)

        parents = {
            node.knowledge_ref: node.parent_knowledge_refs
            for node in self.nodes
        }
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(ref: str) -> None:
            if ref in visiting:
                raise ValueError("static knowledge parents must be acyclic")
            if ref in visited:
                return
            visiting.add(ref)
            for parent in parents[ref]:
                visit(parent)
            visiting.remove(ref)
            visited.add(ref)

        for ref in node_refs:
            visit(ref)
        return self


class GoalIdentity(FrozenModel):
    """Exact immutable identity projection of a canonical goal node."""

    goal_ref: Ref
    node_kind: GoalNodeKind
    goal_kind: GoalKind | None
    priority_class_ref: Ref | None
    concrete_target_ref: Ref | None
    epistemic_status: Ref
    schema_version: Ref


class GoalRelations(FrozenModel):
    """Exact immutable relation projection of a canonical goal node."""

    active_terminal_goal_ref: Ref | None
    parent_goal_refs: tuple[Ref, ...]
    prerequisite_goal_refs: tuple[Ref, ...]
    enabled_goal_refs: tuple[Ref, ...]
    alternative_goal_refs: tuple[Ref, ...]
    conflicting_goal_refs: tuple[Ref, ...]


class GoalEvidence(FrozenModel):
    """Exact immutable evidence projection of a canonical goal node."""

    support_claim_refs: tuple[Ref, ...]
    falsifier_refs: tuple[Ref, ...]
    expected_effect_signature_refs: tuple[Ref, ...]
    expected_goal_delta_refs: tuple[Ref, ...]
    observed_goal_delta_refs: tuple[Ref, ...]
    achieved_claim_refs: tuple[Ref, ...]
    effect_observation_refs: tuple[Ref, ...]
    best_credible_outcome_ref: Ref | None
    worst_credible_outcome_ref: Ref | None


class GoalLifecycle(FrozenModel):
    """Exact immutable lifecycle projection of a canonical goal node."""

    lifecycle_status: GoalLifecycleStatus
    status_history_refs: tuple[Ref, ...]
    activation_reason_refs: tuple[Ref, ...]
    disable_reason_refs: tuple[Ref, ...]
    enable_reason_refs: tuple[Ref, ...]
    blocker_refs: tuple[Ref, ...]
    access_condition_refs: tuple[Ref, ...]
    access_state: Ref
    reopen_count: int
    first_opened_revision: NonNegativeRevision | None
    last_revised_revision: NonNegativeRevision | None
    last_pursued_revision: NonNegativeRevision | None


class GoalCommitment(FrozenModel):
    """Exact immutable commitment projection of a canonical goal node."""

    candidate_plan_refs: tuple[Ref, ...]
    committed_plan_ref: Ref | None
    committed_path_ref: Ref | None
    resource_budget_ref: Ref | None
    reversibility_ref: Ref | None


class GoalProgress(FrozenModel):
    """Exact immutable progress projection of a canonical goal node."""

    plan_cursor: int
    remaining_action_bound: int | None
    attempt_count: int
    primitive_action_use_count: int
    successful_transition_count: int
    no_effect_count: int
    failed_plan_variant_refs: tuple[Ref, ...]


class GoalGraphNode(FrozenModel):
    """Full bounded memory for one persistent goal node."""

    goal_ref: Ref
    node_kind: GoalNodeKind
    goal_kind: GoalKind | None = None
    # A scene-local goal may be an instance of a reusable DRM priority class.
    # The class reference is static knowledge provenance, not another open
    # scene goal and therefore does not become a dynamic node by itself.
    priority_class_ref: Ref | None = None
    # Scene-local binding for an instrumental prerequisite.  These are
    # references, never coordinates or game-specific identities.
    active_terminal_goal_ref: Ref | None = None
    concrete_target_ref: Ref | None = None
    epistemic_status: Ref = "supposed"
    lifecycle_status: GoalLifecycleStatus = GoalLifecycleStatus.PROPOSED
    # Immutable TERM/CLAIM references for every lifecycle transition.  The
    # current status remains a projection, while this history prevents a
    # re-evaluation from erasing why the node was disabled or reopened.
    status_history_refs: tuple[Ref, ...] = ()
    parent_goal_refs: tuple[Ref, ...] = ()
    prerequisite_goal_refs: tuple[Ref, ...] = ()
    enabled_goal_refs: tuple[Ref, ...] = ()
    alternative_goal_refs: tuple[Ref, ...] = ()
    conflicting_goal_refs: tuple[Ref, ...] = ()
    support_claim_refs: tuple[Ref, ...] = ()
    falsifier_refs: tuple[Ref, ...] = ()
    activation_reason_refs: tuple[Ref, ...] = ()
    disable_reason_refs: tuple[Ref, ...] = ()
    enable_reason_refs: tuple[Ref, ...] = ()
    blocker_refs: tuple[Ref, ...] = ()
    access_condition_refs: tuple[Ref, ...] = ()
    access_state: Ref = "unknown"
    candidate_plan_refs: tuple[Ref, ...] = ()
    committed_plan_ref: Ref | None = None
    committed_path_ref: Ref | None = None
    plan_cursor: int = Field(default=0, ge=0)
    remaining_action_bound: int | None = Field(default=None, ge=0)
    expected_effect_signature_refs: tuple[Ref, ...] = ()
    expected_goal_delta_refs: tuple[Ref, ...] = ()
    observed_goal_delta_refs: tuple[Ref, ...] = ()
    attempt_count: int = Field(default=0, ge=0)
    primitive_action_use_count: int = Field(default=0, ge=0)
    successful_transition_count: int = Field(default=0, ge=0)
    no_effect_count: int = Field(default=0, ge=0)
    achieved_claim_refs: tuple[Ref, ...] = ()
    effect_observation_refs: tuple[Ref, ...] = ()
    failed_plan_variant_refs: tuple[Ref, ...] = ()
    reopen_count: int = Field(default=0, ge=0)
    first_opened_revision: NonNegativeRevision | None = None
    last_revised_revision: NonNegativeRevision | None = None
    last_pursued_revision: NonNegativeRevision | None = None
    best_credible_outcome_ref: Ref | None = None
    worst_credible_outcome_ref: Ref | None = None
    resource_budget_ref: Ref | None = None
    reversibility_ref: Ref | None = None
    schema_version: Ref = "yf_arc3_v5.goal_graph_node.v1"

    @property
    def identity_view(self) -> GoalIdentity:
        return GoalIdentity.model_validate(self, from_attributes=True)

    @property
    def relations_view(self) -> GoalRelations:
        return GoalRelations.model_validate(self, from_attributes=True)

    @property
    def evidence_view(self) -> GoalEvidence:
        return GoalEvidence.model_validate(self, from_attributes=True)

    @property
    def lifecycle_view(self) -> GoalLifecycle:
        return GoalLifecycle.model_validate(self, from_attributes=True)

    @property
    def commitment_view(self) -> GoalCommitment:
        return GoalCommitment.model_validate(self, from_attributes=True)

    @property
    def progress_view(self) -> GoalProgress:
        return GoalProgress.model_validate(self, from_attributes=True)

    @model_validator(mode="after")
    def validate_revision_memory(self) -> "GoalGraphNode":
        # Instrumental nodes are scene-local means, never free-standing
        # objectives.  Their activation must be justified by at least one
        # stronger scene goal; terminal world goals and access states remain
        # valid roots of the dynamic graph.
        instrumental_kinds = {
            GoalNodeKind.MECHANISM_DISCRIMINATION,
            GoalNodeKind.EFFECT_CALIBRATION,
            GoalNodeKind.CONFIGURATION_DISCOVERY,
            GoalNodeKind.RECOVERY,
        }
        if self.node_kind in instrumental_kinds and not self.parent_goal_refs:
            raise ValueError(
                "instrumental goal nodes must reference at least one parent goal"
            )
        revisions = tuple(
            value
            for value in (
                self.first_opened_revision,
                self.last_pursued_revision,
                self.last_revised_revision,
            )
            if value is not None
        )
        if revisions != tuple(sorted(revisions)):
            raise ValueError("goal revision memory must be monotonic")
        return self


class GoalGraphEdge(FrozenModel):
    edge_ref: Ref
    edge_kind: GoalEdgeKind
    source_ref: Ref
    target_ref: Ref
    support_claim_refs: tuple[Ref, ...] = ()
    falsifier_refs: tuple[Ref, ...] = ()
    active: bool = True
    schema_version: Ref = "yf_arc3_v5.goal_graph_edge.v1"


class GoalPriorityNode(FrozenModel):
    """One configuration-dependent ordered frontier of the goal graph.

    The order is declarative and concrete: consumers take the first eligible
    goal in ``ordered_goal_refs``.  A different observed context may select a
    child priority node with a different list; no scalar score is involved.
    """

    priority_node_ref: Ref
    parent_priority_node_ref: Ref | None = None
    # Provenance of the consolidated dynamic branch.  DRM declares the
    # static priority family; SRC/TERM activates this scene-specific node.
    source_static_priority_ref: Ref | None = None
    consolidated_priority_ref: Ref | None = None
    consolidation_reason_refs: tuple[Ref, ...] = ()
    context_claim_refs: tuple[Ref, ...] = ()
    ordered_goal_refs: tuple[Ref, ...] = Field(min_length=1, max_length=64)
    best_credible_outcome_refs: tuple[Ref, ...] = ()
    worst_credible_outcome_refs: tuple[Ref, ...] = ()
    child_priority_node_refs: tuple[Ref, ...] = ()
    transition_reason_refs: tuple[Ref, ...] = ()
    schema_version: Ref = "yf_arc3_v5.goal_priority_node.v1"


class GoalGraph(FrozenModel):
    """Persistent goal/dependency graph; selection remains DRM-owned."""

    nodes: tuple[GoalGraphNode, ...] = Field(max_length=256)
    edges: tuple[GoalGraphEdge, ...] = Field(max_length=768)
    priority_nodes: tuple[GoalPriorityNode, ...] = Field(default=(), max_length=256)
    static_knowledge_graph_ref: Ref | None = None
    revision: NonNegativeRevision = 0
    schema_version: Ref = "yf_arc3_v5.goal_graph.v1"

    @model_validator(mode="after")
    def validate_graph(self) -> "GoalGraph":
        node_refs = tuple(item.identity_view.goal_ref for item in self.nodes)
        edge_refs = tuple(item.edge_ref for item in self.edges)
        priority_refs = tuple(item.priority_node_ref for item in self.priority_nodes)
        require_unique(node_refs, "goal graph node refs")
        require_unique(edge_refs, "goal graph edge refs")
        require_unique(priority_refs, "goal priority node refs")
        node_set = set(node_refs)
        if any(
            edge.source_ref not in node_set or edge.target_ref not in node_set
            for edge in self.edges
        ):
            raise ValueError("goal graph edges must reference known nodes")
        priority_set = set(priority_refs)
        if any(
            goal_ref not in node_set
            for priority in self.priority_nodes
            for goal_ref in priority.ordered_goal_refs
        ):
            raise ValueError("goal priority lists must reference known goals")
        if any(
            child_ref not in priority_set
            for priority in self.priority_nodes
            for child_ref in priority.child_priority_node_refs
        ):
            raise ValueError("goal priority children must reference known priority nodes")
        if any(
            priority.parent_priority_node_ref is not None
            and priority.parent_priority_node_ref not in priority_set
            for priority in self.priority_nodes
        ):
            raise ValueError("goal priority parents must reference known priority nodes")

        # Configuration-dependent ordered lists form a tree of contexts even
        # though the live goal graph itself may contain arbitrary cycles.  A
        # cycle here would make the declared frontier ambiguous and would
        # force selection to invent a context transition.
        priority_adjacency: dict[str, set[str]] = {
            ref: set() for ref in priority_set
        }
        for priority in self.priority_nodes:
            if priority.parent_priority_node_ref is not None:
                priority_adjacency[priority.parent_priority_node_ref].add(
                    priority.priority_node_ref
                )
            priority_adjacency[priority.priority_node_ref].update(
                priority.child_priority_node_refs
            )
        priority_visiting: set[str] = set()
        priority_visited: set[str] = set()

        def visit_priority(ref: str) -> None:
            if ref in priority_visiting:
                raise ValueError("goal priority contexts must be acyclic")
            if ref in priority_visited:
                return
            priority_visiting.add(ref)
            for child_ref in priority_adjacency[ref]:
                visit_priority(child_ref)
            priority_visiting.remove(ref)
            priority_visited.add(ref)

        for ref in priority_set:
            visit_priority(ref)
        return self


class DynamicGoalGraph(GoalGraph):
    """Live scene graph; static knowledge is referenced, never embedded."""

    schema_version: Ref = "yf_arc3_v5.dynamic_goal_graph.v1"


class InstrumentalSubgoal(FrozenModel):
    """A temporary means toward one or more stronger scene goals."""

    subgoal_ref: Ref
    node_kind: GoalNodeKind
    parent_goal_refs: tuple[Ref, ...] = Field(min_length=1)
    activation_reason_refs: tuple[Ref, ...] = Field(min_length=1)
    expected_effect_refs: tuple[Ref, ...] = ()
    observed_effect_refs: tuple[Ref, ...] = ()
    lifecycle_status: GoalLifecycleStatus = GoalLifecycleStatus.PROPOSED
    schema_version: Ref = "yf_arc3_v5.instrumental_subgoal.v1"

    @model_validator(mode="after")
    def validate_instrumental_kind(self) -> "InstrumentalSubgoal":
        if self.node_kind in {
            GoalNodeKind.OFFICIAL_COMPLETION,
            GoalNodeKind.WORLD_ACHIEVEMENT,
            GoalNodeKind.SAFETY_INVARIANT,
        }:
            raise ValueError("instrumental subgoals cannot be terminal world goals")
        return self


class DynamicGoalFrontierActivation(FrozenModel):
    """Scene-scoped activation emitted by SRC from DRM and CLAIM evidence.

    This is an immutable update description, not a Python-created goal.  The
    guiding principle and support claims identify why the dynamic frontier is
    open; terminal goals remain distinct from instrumental subgoals and
    outcome envelopes remain consequence references only.
    """

    activation_ref: Ref
    static_knowledge_graph_ref: Ref
    guiding_principle_ref: Ref
    scene_support_claim_refs: tuple[Ref, ...] = Field(min_length=1)
    terminal_goal_refs: tuple[Ref, ...] = Field(min_length=1)
    instrumental_subgoal_refs: tuple[Ref, ...] = ()
    instrumental_parent_goal_refs: tuple[tuple[Ref, Ref], ...] = ()
    # Each mapping preserves which declared falsifier disables a terminal
    # goal or one of its instrumental subgoals.  The payload is scene-scoped,
    # while the falsifier vocabulary remains generic DRM knowledge.
    goal_falsifier_refs: tuple[tuple[Ref, Ref], ...] = ()
    # Optional generic terminal goal selected by the active priority branch;
    # instance-specific terminal relation refs remain in terminal_goal_refs.
    active_goal_ref: Ref | None = None
    active_priority_node_ref: Ref | None = None
    source_static_priority_ref: Ref | None = None
    consolidated_priority_ref: Ref | None = None
    consolidation_reason_refs: tuple[Ref, ...] = ()
    activation_reason_refs: tuple[Ref, ...] = Field(min_length=1)
    disable_reason_refs: tuple[Ref, ...] = ()
    alternative_goal_refs: tuple[Ref, ...] = ()
    conflicting_goal_refs: tuple[Ref, ...] = ()
    best_credible_outcome_refs: tuple[Ref, ...] = ()
    worst_credible_outcome_refs: tuple[Ref, ...] = ()
    schema_version: Ref = "yf_arc3_v5.dynamic_goal_frontier_activation.v1"

    @model_validator(mode="after")
    def validate_frontier_activation(self) -> "DynamicGoalFrontierActivation":
        for refs, label in (
            (self.scene_support_claim_refs, "scene support claims"),
            (self.terminal_goal_refs, "terminal goals"),
            (self.instrumental_subgoal_refs, "instrumental subgoals"),
            (self.consolidation_reason_refs, "consolidation reasons"),
            (self.activation_reason_refs, "activation reasons"),
            (self.alternative_goal_refs, "alternative goals"),
            (self.conflicting_goal_refs, "conflicting goals"),
        ):
            require_unique(refs, label)
        terminal_refs = set(self.terminal_goal_refs)
        instrumental_refs = set(self.instrumental_subgoal_refs)
        require_unique(
            tuple(pair[0] for pair in self.instrumental_parent_goal_refs),
            "instrumental subgoal parent mappings",
        )
        parent_pairs = dict(self.instrumental_parent_goal_refs)
        if set(parent_pairs) != instrumental_refs:
            raise ValueError(
                "every instrumental subgoal must declare one terminal parent"
            )
        if any(
            parent_ref not in terminal_refs for parent_ref in (parent_pairs[__yf_order_key] for __yf_order_key in sorted(parent_pairs))
        ):
            raise ValueError(
                "instrumental subgoals must reference active terminal goals"
            )
        if terminal_refs & instrumental_refs:
            raise ValueError(
                "terminal goals and instrumental subgoals must remain distinct"
            )
        if (
            self.source_static_priority_ref is not None
            or self.consolidated_priority_ref is not None
        ):
            if not self.source_static_priority_ref or not self.consolidated_priority_ref:
                raise ValueError(
                    "consolidated priority needs static provenance and a consolidated ref"
                )
            require_unique(
                self.consolidation_reason_refs,
                "consolidation reasons",
            )
            if not self.consolidation_reason_refs:
                raise ValueError(
                    "consolidated priority needs an explicit consolidation reason"
                )
        falsifier_pairs = dict(self.goal_falsifier_refs)
        if len(falsifier_pairs) != len(self.goal_falsifier_refs):
            raise ValueError("goal falsifier mappings must be unique")
        if any(
            goal_ref not in terminal_refs | instrumental_refs or not falsifier_ref
            for goal_ref, falsifier_ref in sorted(falsifier_pairs.items())
        ):
            raise ValueError(
                "goal falsifiers must reference an activated terminal or instrumental goal"
            )
        if (
            self.guiding_principle_ref in terminal_refs
            or self.guiding_principle_ref in instrumental_refs
        ):
            raise ValueError("guiding principle cannot be a scene goal reference")
        if (
            self.active_priority_node_ref is not None
            and self.active_priority_node_ref.startswith("policy.")
        ):
            raise ValueError(
                "active priority context must reference a goal-priority node, "
                "not a DRM selection policy"
            )
        if set(self.alternative_goal_refs) & set(self.conflicting_goal_refs):
            raise ValueError("alternative and conflicting goals must remain distinct")
        return self


class GoalGraphComponent(FrozenModel):
    component_ref: Ref
    node_refs: tuple[Ref, ...] = Field(min_length=1)
    outgoing_component_refs: tuple[Ref, ...] = ()
    schema_version: Ref = "yf_arc3_v5.goal_graph_component.v1"


class GoalGraphAnalysis(FrozenModel):
    graph_revision: NonNegativeRevision
    static_knowledge_graph_ref: Ref | None = None
    components: tuple[GoalGraphComponent, ...]
    node_component_refs: tuple[tuple[Ref, Ref], ...]
    active_edge_refs: tuple[Ref, ...]
    schema_version: Ref = "yf_arc3_v5.goal_graph_analysis.v1"


class GoalProposal(FrozenModel):
    """A DRM-grounded candidate goal; Python may only transport/validate it."""

    proposal_ref: Ref
    goal_ref: Ref
    node_kind: GoalNodeKind
    grounding_claim_refs: tuple[Ref, ...] = Field(min_length=1)
    predicted_screen_configuration_delta_ref: Ref | None = None
    predicted_action_effect_class: DeclaredCausalClass | None = None
    alternatives_separated: tuple[Ref, ...] = ()
    goals_unlocked: tuple[Ref, ...] = ()
    access_plan_bound: Ref | None = None
    probe_and_recovery_cost: int = Field(ge=0)
    explicit_falsifier_refs: tuple[Ref, ...] = Field(min_length=1)
    schema_version: Ref = "yf_arc3_v5.goal_proposal.v1"


class GoalPlanStep(FrozenModel):
    subgoal_ref: Ref
    prerequisite_subgoal_refs: tuple[Ref, ...] = ()
    primitive_action_refs: tuple[Ref, ...] = Field(min_length=1, max_length=16)
    addressed_gap_dimensions: tuple[GoalGapDimension, ...] = Field(min_length=1)
    expected_effect_class: DeclaredCausalClass | None = None
    expected_delta_ref: Ref | None = None
    falsifier_refs: tuple[Ref, ...] = Field(min_length=1)
    action_bound: int = Field(default=1, ge=1, le=16)
    schema_version: Ref = "yf_arc3_v5.goal_plan_step.v1"

    @model_validator(mode="after")
    def validate_gap_dimensions(self) -> "GoalPlanStep":
        require_unique(self.addressed_gap_dimensions, "addressed goal-gap dimensions")
        return self


class GoalPlanRequest(FrozenModel):
    goal_ref: Ref
    steps: tuple[GoalPlanStep, ...] = Field(min_length=1, max_length=32)
    maximum_witnesses: int = Field(default=3, ge=1, le=3)
    schema_version: Ref = "yf_arc3_v5.goal_plan_request.v1"

    @model_validator(mode="after")
    def validate_steps(self) -> "GoalPlanRequest":
        refs = tuple(item.subgoal_ref for item in self.steps)
        require_unique(refs, "goal plan subgoal refs")
        if any(
            prerequisite not in set(refs)
            for item in self.steps
            for prerequisite in item.prerequisite_subgoal_refs
        ):
            raise ValueError("goal plan prerequisites must reference known steps")
        return self


class GoalPlanWitness(FrozenModel):
    plan_ref: Ref
    goal_ref: Ref
    subgoal_refs: tuple[Ref, ...] = Field(min_length=1)
    primitive_action_refs: tuple[Ref, ...] = Field(min_length=1, max_length=128)
    addressed_gap_dimensions: tuple[GoalGapDimension, ...] = Field(min_length=1)
    remaining_action_bound: int = Field(ge=1, le=128)
    cursor: int = Field(default=0, ge=0)
    schema_version: Ref = "yf_arc3_v5.goal_plan_witness.v1"

    @model_validator(mode="after")
    def validate_gap_dimensions(self) -> "GoalPlanWitness":
        require_unique(self.addressed_gap_dimensions, "witness goal-gap dimensions")
        return self


class GoalPlanResult(FrozenModel):
    goal_ref: Ref
    witnesses: tuple[GoalPlanWitness, ...] = Field(max_length=3)
    schema_version: Ref = "yf_arc3_v5.goal_plan_result.v1"


class OrderedSupportConfiguration(FrozenModel):
    """One finite arrangement of strictly ordered extents on repeated supports."""

    support_extents: tuple[tuple[int, ...], ...] = Field(min_length=3, max_length=8)
    schema_version: Ref = "yf_arc3_v5.ordered_support_configuration.v1"

    @model_validator(mode="after")
    def validate_strict_extents(self) -> "OrderedSupportConfiguration":
        flattened = tuple(
            extent for support in self.support_extents for extent in support
        )
        if not flattened:
            raise ValueError("an ordered-support configuration needs at least one extent")
        if any(extent <= 0 for extent in flattened):
            raise ValueError("ordered-support extents must be positive")
        if len(set(flattened)) != len(flattened):
            raise ValueError("ordered-support extents must be globally unique")
        if any(
            lower <= upper
            for support in self.support_extents
            for lower, upper in zip(support, support[1:])
        ):
            raise ValueError("each support must be strictly decreasing bottom-to-top")
        return self


class OrderedSupportTransferRequest(FrozenModel):
    """Mechanical search request whose endpoint and orders were declared upstream."""

    configuration: OrderedSupportConfiguration
    endpoint_support_index: int = Field(ge=0, le=7)
    traversal_orders: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=3)
    maximum_state_expansions: int = Field(default=4096, ge=1, le=65536)
    maximum_transfer_depth: int = Field(default=128, ge=1, le=128)
    schema_version: Ref = "yf_arc3_v5.ordered_support_transfer_request.v1"

    @model_validator(mode="after")
    def validate_declared_orders(self) -> "OrderedSupportTransferRequest":
        support_count = len(self.configuration.support_extents)
        if self.endpoint_support_index >= support_count:
            raise ValueError("endpoint support index is outside the configuration")
        expected = tuple(range(support_count))
        if any(tuple(sorted(order)) != expected for order in self.traversal_orders):
            raise ValueError("each traversal order must be a support permutation")
        if len(set(self.traversal_orders)) != len(self.traversal_orders):
            raise ValueError("declared traversal orders must be unique")
        return self


class OrderedSupportTransferStep(FrozenModel):
    extent: int = Field(gt=0)
    source_support_index: int = Field(ge=0, le=7)
    destination_support_index: int = Field(ge=0, le=7)
    schema_version: Ref = "yf_arc3_v5.ordered_support_transfer_step.v1"


class OrderedSupportTransferWitness(FrozenModel):
    traversal_order_ordinal: int = Field(ge=0, le=2)
    steps: tuple[OrderedSupportTransferStep, ...] = Field(min_length=1, max_length=128)
    expanded_state_count: int = Field(ge=1, le=65536)
    schema_version: Ref = "yf_arc3_v5.ordered_support_transfer_witness.v1"


class OrderedSupportTransferResult(FrozenModel):
    witnesses: tuple[OrderedSupportTransferWitness, ...] = Field(max_length=3)
    total_expanded_state_count: int = Field(ge=0, le=65536)
    state_expansion_bound_reached: bool = False
    transfer_depth_bound_reached: bool = False
    failure_kinds: tuple[Ref, ...] = Field(default=(), max_length=3)
    schema_version: Ref = "yf_arc3_v5.ordered_support_transfer_result.v1"


class OrderedSupportDescription(FrozenModel):
    support_ref: Ref
    center_column: int = Field(ge=0)
    point: Position
    structural_extent: int = Field(gt=0)
    layer_extents: tuple[int, ...] = Field(default=(), max_length=8)
    schema_version: Ref = "yf_arc3_v5.ordered_support_description.v1"


class OrderedSupportGeometryMeasurements(FrozenModel):
    supports: tuple[OrderedSupportDescription, ...] = Field(default=(), max_length=8)
    repeated_support_count: int = Field(ge=0)
    strict_layer_order_observed: bool = False
    globally_unique_layer_extents_observed: bool = False
    unique_widest_support_index: int | None = Field(default=None, ge=0, le=7)
    schema_version: Ref = "yf_arc3_v5.ordered_support_geometry_measurements.v1"


class ActionGroundingContract(FrozenModel):
    """Complete symbolic why required before a migrated action release."""

    active_goal_ref: Ref
    active_terminal_goal_ref: Ref | None = None
    subgoal_ref: Ref
    goal_graph_edge_refs: tuple[Ref, ...] = Field(min_length=1)
    goal_dependency_chain_refs: tuple[Ref, ...] = ()
    advances_active_goal_chain: bool | None = None
    inherited_priority_from_terminal_goal: bool = False
    priority_class_ref: Ref | None = None
    selection_principle_ref: Ref
    premise_claim_refs: tuple[Ref, ...] = Field(min_length=1)
    canonical_premise_claim_refs: tuple[Ref, ...] = Field(default=(), max_length=64)
    expected_effect_class: DeclaredCausalClass
    expected_configuration_or_goal_delta_ref: Ref
    explicit_falsifier_ref: Ref
    committed_plan_ref: Ref | None = None
    discriminating_experiment_ref: Ref | None = None
    declared_plan_ref: Ref | None = None
    declared_plan_stage_refs: tuple[Ref, ...] = Field(default=(), max_length=16)
    declared_plan_ordered_member_refs: tuple[Ref, ...] = Field(
        default=(), max_length=16
    )
    declared_plan_ordered_member_values: tuple[int, ...] = Field(
        default=(), max_length=16
    )
    declared_plan_current_stage_ref: Ref | None = None
    declared_plan_cursor_ordinal: int | None = Field(default=None, ge=0)
    declared_plan_member_count: int | None = Field(default=None, ge=1)
    remaining_action_bound: int = Field(ge=1, le=128)
    reconciliation_checkpoint_ref: Ref
    exploration_purpose_refs: tuple[Ref, ...] = ()
    schema_version: Ref = "yf_arc3_v5.action_grounding_contract.v1"

    @model_validator(mode="after")
    def validate_commitment(self) -> "ActionGroundingContract":
        if bool(self.committed_plan_ref) == bool(self.discriminating_experiment_ref):
            raise ValueError(
                "grounding requires exactly one committed plan or experiment"
            )
        if self.active_terminal_goal_ref:
            if self.advances_active_goal_chain is not True:
                raise ValueError(
                    "goal grounding must declare progress on the active terminal chain"
                )
            if (
                self.active_goal_ref != self.active_terminal_goal_ref
                and not self.inherited_priority_from_terminal_goal
                and not self.goal_dependency_chain_refs
            ):
                raise ValueError(
                    "instrumental goal grounding requires its dependency chain"
                )
            if self.goal_dependency_chain_refs and (
                self.active_terminal_goal_ref not in self.goal_dependency_chain_refs
                or self.active_goal_ref not in self.goal_dependency_chain_refs
            ):
                raise ValueError(
                    "goal dependency chain does not contain the active goal pair"
                )
        if self.declared_plan_ref:
            if not self.declared_plan_stage_refs:
                raise ValueError("declared complete plan requires its ordered stages")
            if self.declared_plan_current_stage_ref not in self.declared_plan_stage_refs:
                raise ValueError("current complete-plan stage is absent from its stage order")
            if not self.active_terminal_goal_ref:
                raise ValueError("declared complete plan requires one terminal goal")
            required_chain_refs = {
                self.active_terminal_goal_ref,
                self.active_goal_ref,
                self.subgoal_ref,
            }
            if not required_chain_refs.issubset(set(self.goal_dependency_chain_refs)):
                raise ValueError("declared complete plan is not grounded through its full goal chain")
            if bool(self.declared_plan_cursor_ordinal is not None) != bool(
                self.declared_plan_member_count is not None
            ):
                raise ValueError(
                    "declared complete plan must provide both member cursor and count"
                )
            if (
                self.declared_plan_cursor_ordinal is not None
                and self.declared_plan_member_count is not None
                and self.declared_plan_cursor_ordinal
                >= self.declared_plan_member_count
            ):
                raise ValueError("declared complete plan has an invalid ordered-member cursor")
            if bool(self.declared_plan_ordered_member_refs) != bool(
                self.declared_plan_ordered_member_values
            ):
                raise ValueError(
                    "declared ordered plan members require refs and observed values"
                )
            if self.declared_plan_ordered_member_refs and (
                len(self.declared_plan_ordered_member_refs)
                != self.declared_plan_member_count
                or len(self.declared_plan_ordered_member_values)
                != self.declared_plan_member_count
            ):
                raise ValueError(
                    "declared ordered plan members must match the declared member count"
                )
        return self


class CompactTransitionEvidence(FrozenModel):
    """One bounded immutable transition record containing observations only."""

    transition_ref: Ref
    attempt_ref: Ref
    level_index: int = Field(ge=1)
    sequence_index: int = Field(ge=0)
    before_digest: Ref
    action_ref: Ref
    action_payload_digest: Ref
    after_digest: Ref
    observed_delta_digest: Ref
    observed_changed_count: int = Field(ge=0)
    observed_terminal_ref: Ref | None = None
    schema_version: Ref = "yf_arc3_v5.compact_transition_evidence.v1"


class DeclaredTransitionPrediction(FrozenModel):
    """Mechanical prediction supplied by an already-declared partial model."""

    transition_ref: Ref
    prediction_applies: bool
    predicted_after_digest: Ref | None = None
    predicted_delta_digest: Ref | None = None
    predicted_changed_count: int | None = Field(default=None, ge=0)
    predicted_terminal_ref: Ref | None = None
    schema_version: Ref = "yf_arc3_v5.declared_transition_prediction.v1"

    @model_validator(mode="after")
    def validate_prediction_payload(self) -> "DeclaredTransitionPrediction":
        payload = (
            self.predicted_after_digest,
            self.predicted_delta_digest,
            self.predicted_changed_count,
            self.predicted_terminal_ref,
        )
        if self.prediction_applies and all(item is None for item in payload):
            raise ValueError("an applicable prediction must expose a measured output")
        if not self.prediction_applies and any(item is not None for item in payload):
            raise ValueError("an uncovered transition cannot carry predicted output")
        return self


class TransitionEvidenceLedgerInput(FrozenModel):
    """Finite observed transition history to compact by exact structural signature."""

    transitions: tuple[CompactTransitionEvidence, ...] = Field(max_length=256)
    schema_version: Ref = "yf_arc3_v5.transition_evidence_ledger_input.v1"

    @model_validator(mode="after")
    def validate_transition_refs(self) -> "TransitionEvidenceLedgerInput":
        require_unique(
            tuple(item.transition_ref for item in self.transitions),
            "transition evidence refs",
        )
        return self


class TransitionEvidenceClass(FrozenModel):
    """Exact equality class; no causal or semantic meaning is inferred."""

    signature_digest: Ref
    action_ref: Ref
    action_payload_digest: Ref
    observed_delta_digest: Ref
    observed_changed_count: int = Field(ge=0)
    observed_terminal_ref: Ref | None = None
    occurrence_count: int = Field(ge=1)
    witness_transition_refs: tuple[Ref, ...] = Field(min_length=1, max_length=3)
    schema_version: Ref = "yf_arc3_v5.transition_evidence_class.v1"


class TransitionEvidenceLedger(FrozenModel):
    observed_transition_count: int = Field(ge=0)
    evidence_classes: tuple[TransitionEvidenceClass, ...] = Field(max_length=256)
    schema_version: Ref = "yf_arc3_v5.transition_evidence_ledger.v1"


class TransitionRetrodictionInput(FrozenModel):
    """Observed evidence plus outputs of one declared partial model."""

    model_ref: Ref
    transitions: tuple[CompactTransitionEvidence, ...] = Field(max_length=256)
    predictions: tuple[DeclaredTransitionPrediction, ...] = Field(max_length=256)
    schema_version: Ref = "yf_arc3_v5.transition_retrodiction_input.v1"

    @model_validator(mode="after")
    def validate_prediction_index(self) -> "TransitionRetrodictionInput":
        transition_refs = tuple(item.transition_ref for item in self.transitions)
        prediction_refs = tuple(item.transition_ref for item in self.predictions)
        require_unique(transition_refs, "retrodiction transition refs")
        require_unique(prediction_refs, "retrodiction prediction refs")
        if set(transition_refs) != set(prediction_refs):
            raise ValueError("predictions must index every observed transition exactly once")
        return self


class TransitionDivergence(FrozenModel):
    transition_ref: Ref
    sequence_index: int = Field(ge=0)
    observed_after_digest: Ref
    predicted_after_digest: Ref | None = None
    observed_delta_digest: Ref
    predicted_delta_digest: Ref | None = None
    observed_changed_count: int = Field(ge=0)
    predicted_changed_count: int | None = Field(default=None, ge=0)
    observed_terminal_ref: Ref | None = None
    predicted_terminal_ref: Ref | None = None
    mismatch_dimensions: tuple[Ref, ...] = Field(min_length=1)
    schema_version: Ref = "yf_arc3_v5.transition_divergence.v1"


class TransitionRetrodictionMeasurements(FrozenModel):
    """Separate coverage and exactness measures with the first ordered mismatch."""

    model_ref: Ref
    observed_count: int = Field(ge=0)
    covered_count: int = Field(ge=0)
    exact_count: int = Field(ge=0)
    uncovered_transition_refs: tuple[Ref, ...] = ()
    divergent_transition_refs: tuple[Ref, ...] = ()
    first_divergence: TransitionDivergence | None = None
    descriptive_facts: FrozenMap = Field(default_factory=FrozenMap)
    schema_version: Ref = "yf_arc3_v5.transition_retrodiction_measurements.v1"

    @model_validator(mode="after")
    def validate_counts(self) -> "TransitionRetrodictionMeasurements":
        if self.covered_count > self.observed_count:
            raise ValueError("covered count cannot exceed observed count")
        if self.exact_count > self.covered_count:
            raise ValueError("exact count cannot exceed covered count")
        if len(self.uncovered_transition_refs) != self.observed_count - self.covered_count:
            raise ValueError("uncovered refs must explain the coverage gap")
        if len(self.divergent_transition_refs) != self.covered_count - self.exact_count:
            raise ValueError("divergent refs must explain the exactness gap")
        if bool(self.divergent_transition_refs) != (self.first_divergence is not None):
            raise ValueError("first divergence must exist exactly when mismatches exist")
        return self


class TransitionPhenomenonExplanationLink(FrozenModel):
    """One source-declared explanation for one mechanically inventoried phenomenon."""

    phenomenon_ref: Ref
    explanation_ref: Ref
    evidence_refs: tuple[Ref, ...] = Field(min_length=1, max_length=16)
    falsifier_ref: Ref
    schema_version: Ref = "yf_arc3_v5.transition_phenomenon_explanation_link.v1"


class TransitionPhenomenonInventoryInput(FrozenModel):
    """A bounded ordered frame packet plus externally declared explanation links."""

    before: FrameGrid
    intermediate_frames: tuple[FrameGrid, ...] = Field(default=(), max_length=64)
    after: FrameGrid
    transition_ref: Ref
    action_ref: Ref | None = None
    explanation_links: tuple[TransitionPhenomenonExplanationLink, ...] = Field(
        default=(), max_length=128
    )
    maximum_phenomenon_groups: int = Field(default=128, ge=1, le=128)
    prior_same_declared_interaction_effectful_pair_count: int = Field(
        default=0, ge=0, le=65
    )
    official_terminal_success: bool = False
    terminal_packet_excludes_next_level_scene: bool = False
    schema_version: Ref = "yf_arc3_v5.transition_phenomenon_inventory_input.v1"


class TransitionPhenomenonGroup(FrozenModel):
    """One connected set of cells sharing the exact same ordered value trace."""

    phenomenon_ref: Ref
    trace_digest: Ref
    pixel_count: int = Field(gt=0)
    changed_pair_indices: tuple[int, ...] = Field(min_length=1, max_length=65)
    first_changed_pair_index: int = Field(ge=0, le=64)
    last_changed_pair_index: int = Field(ge=0, le=64)
    returns_to_initial: bool
    differs_in_settled_frame: bool
    bounding_box: BoundingBox
    schema_version: Ref = "yf_arc3_v5.transition_phenomenon_group.v1"


class TransitionPhenomenonRelation(FrozenModel):
    """Exact pairwise temporal and spatial relation between retained groups."""

    first_phenomenon_ref: Ref
    second_phenomenon_ref: Ref
    shares_changed_pair: bool
    first_finishes_before_second_starts: bool
    second_finishes_before_first_starts: bool
    bounding_boxes_overlap: bool
    bounding_boxes_touch_orthogonally: bool
    schema_version: Ref = "yf_arc3_v5.transition_phenomenon_relation.v1"


class OrderedAnimationExtentChange(FrozenModel):
    """Exact extent change temporally following one activation candidate."""

    extent_change_ref: Ref
    effect_pair_index: int = Field(ge=0, le=64)
    bounding_box: BoundingBox
    changed_pixel_count: int = Field(gt=0)
    palette_value: int
    signed_extent_delta: int
    before_extent: int = Field(ge=0)
    after_extent: int = Field(ge=0)
    bound_extent_candidates: tuple[int, ...] = Field(default=(), max_length=8)
    full_rectangular_change: bool
    schema_version: Ref = "yf_arc3_v5.ordered_animation_extent_change.v1"


class OrderedAnimationHomologousOccurrence(FrozenModel):
    """One exact translated occurrence of an activated morphology."""

    occurrence_ref: Ref
    bounding_box: BoundingBox
    observed_palette_values: tuple[int, ...] = Field(min_length=1, max_length=8)
    schema_version: Ref = "yf_arc3_v5.ordered_animation_homologous_occurrence.v1"


class OrderedAnimationGlyphCandidate(FrozenModel):
    """Mechanically grouped activation/replay evidence for one displayed morphology."""

    glyph_candidate_ref: Ref
    morphology_digest: Ref
    destination_bounding_box: BoundingBox
    observed_palette_values: tuple[int, ...] = Field(min_length=1, max_length=8)
    activation_pair_indices: tuple[int, ...] = Field(min_length=1, max_length=32)
    newly_present_after_first_activation: bool
    present_before_action: bool
    persists_after_last_local_effect: bool
    homologous_occurrences: tuple[OrderedAnimationHomologousOccurrence, ...] = Field(
        default=(), max_length=128
    )
    extent_changes: tuple[OrderedAnimationExtentChange, ...] = Field(
        default=(), max_length=64
    )
    schema_version: Ref = "yf_arc3_v5.ordered_animation_glyph_candidate.v1"


class TransitionPhenomenonInventoryMeasurements(FrozenModel):
    """Exhaustive pixel accounting with bounded individuation and causal coverage."""

    transition_ref: Ref
    action_ref: Ref | None = None
    packet_frame_count: int = Field(ge=2, le=66)
    changed_pixel_count: int = Field(ge=0)
    unchanged_pixel_count: int = Field(ge=0)
    phenomenon_groups: tuple[TransitionPhenomenonGroup, ...] = Field(
        default=(), max_length=128
    )
    phenomenon_relations: tuple[TransitionPhenomenonRelation, ...] = Field(
        default=(), max_length=8128
    )
    ordered_animation_glyph_candidates: tuple[OrderedAnimationGlyphCandidate, ...] = Field(
        default=(), max_length=64
    )
    phenomenon_group_count: int = Field(ge=0)
    transient_group_refs: tuple[Ref, ...] = ()
    settled_change_group_refs: tuple[Ref, ...] = ()
    linked_group_refs: tuple[Ref, ...] = ()
    explained_group_refs: tuple[Ref, ...] = ()
    unexplained_group_refs: tuple[Ref, ...] = ()
    overflow_group_count: int = Field(ge=0)
    overflow_pixel_count: int = Field(ge=0)
    inventory_complete: bool
    descriptive_facts: FrozenMap = Field(default_factory=FrozenMap)
    schema_version: Ref = "yf_arc3_v5.transition_phenomenon_inventory_measurements.v1"

    @model_validator(mode="after")
    def validate_inventory_partition(self) -> "TransitionPhenomenonInventoryMeasurements":
        retained_pixels = sum(group.pixel_count for group in self.phenomenon_groups)
        if retained_pixels + self.overflow_pixel_count != self.changed_pixel_count:
            raise ValueError("retained and overflow groups must cover every changed pixel")
        if self.phenomenon_group_count != len(self.phenomenon_groups) + self.overflow_group_count:
            raise ValueError("phenomenon group count must include retained and overflow groups")
        retained_refs = {group.phenomenon_ref for group in self.phenomenon_groups}
        if not set(self.explained_group_refs).issubset(retained_refs):
            raise ValueError("explained refs must name retained phenomenon groups")
        if set(self.explained_group_refs) & set(self.unexplained_group_refs):
            raise ValueError("explained and unexplained groups must be disjoint")
        if self.inventory_complete != (self.overflow_group_count == 0):
            raise ValueError("inventory completeness must match the overflow count")
        return self


class BoundaryDecreaseEvidence(FrozenModel):
    """One exact action-conditioned diminution of a tracked boundary component."""

    indicator_entity_ref: Ref
    transition_ref: Ref
    action_ref: Ref
    evidence_scope_ref: Ref = "scope.current"
    changed_pixel_count: int = Field(gt=0)
    before_pixel_count: int = Field(gt=0)
    after_pixel_count: int = Field(ge=0)
    primitive_count_after: int | None = Field(default=None, ge=0)
    render_extent_bbox: BoundingBox | None = None
    render_pixel_thickness: int | None = Field(default=None, ge=1)
    regular_family_member_count: int = Field(default=0, ge=0, le=12)
    changed_family_member_count: int = Field(default=0, ge=0, le=12)
    remaining_family_member_count: int = Field(default=0, ge=0, le=12)
    homologous_multi_member_state_transfer: bool = False
    schema_version: Ref = "yf_arc3_v5.boundary_decrease_evidence.v1"


class RepeatedBoundaryDecreaseAnalysisInput(FrozenModel):
    before: FrameGrid
    after: FrameGrid
    boundary_search_band_px: int = Field(default=7, ge=1, le=64)
    before_tracking: "TemporalTrackingResult"
    prior_evidence: tuple[BoundaryDecreaseEvidence, ...] = ()
    action_ref: Ref
    transition_ref: Ref
    evidence_scope_ref: Ref = "scope.current"
    primitive_count_after: int | None = Field(default=None, ge=0)
    schema_version: Ref = "yf_arc3_v5.repeated_boundary_decrease_input.v2"


class RepeatedBoundaryDecreaseAnalysis(FrozenModel):
    alternative_refs: tuple[Ref, ...] = Field(min_length=3)
    alternative_facts: FrozenMap
    evidence: tuple[BoundaryDecreaseEvidence, ...] = ()
    evidence_refs: tuple[Ref, ...] = ()
    schema_version: Ref = "yf_arc3_v5.repeated_boundary_decrease_analysis.v1"

    @model_validator(mode="after")
    def validate_alternative_index(self) -> "RepeatedBoundaryDecreaseAnalysis":
        require_unique(self.alternative_refs, "boundary decrease alternatives")
        if set(self.alternative_facts) != set(self.alternative_refs):
            raise ValueError("boundary decrease facts must index every alternative")
        return self


class ResourceGaugeQuantityAnalysisInput(FrozenModel):
    before: FrameGrid
    after: FrameGrid
    before_tracking: "TemporalTrackingResult"
    primitive_count_after: int = Field(ge=0)
    prior_evidence: tuple[BoundaryDecreaseEvidence, ...] = ()
    current_evidence: tuple[BoundaryDecreaseEvidence, ...] = ()
    prior_render_records: tuple[FrozenMap, ...] = Field(default=(), max_length=3)
    rendering_profiles: tuple[FrozenMap, ...] = Field(min_length=1, max_length=6)
    alternative_refs: tuple[Ref, ...] = Field(min_length=2, max_length=3)
    role_selection_ref: Ref
    evidence_scope_ref: Ref
    transition_ref: Ref
    schema_version: Ref = "yf_arc3_v5.resource_gauge_quantity_input.v1"


class ResourceGaugeQuantityAnalysis(FrozenModel):
    alternative_refs: tuple[Ref, ...]
    alternative_facts: FrozenMap
    evidence_refs: tuple[Ref, ...] = ()
    schema_version: Ref = "yf_arc3_v5.resource_gauge_quantity_analysis.v1"


class ComponentMatchCandidate(FrozenModel):
    before_component_id: Ref
    after_component_id: Ref
    delta_row: int
    delta_col: int
    exact_value_identity: bool = True
    exact_shape_identity: bool = True
    schema_version: Ref = "yf_arc3_v5.component_match_candidate.v1"


class ComponentMatchInput(FrozenModel):
    before: ComponentExtractionResult
    after: ComponentExtractionResult
    action_ref: Ref | None = None
    schema_version: Ref = "yf_arc3_v5.component_match_input.v1"


class ComponentMatchResult(FrozenModel):
    candidates: tuple[ComponentMatchCandidate, ...]
    ambiguous_before_component_ids: tuple[Ref, ...]
    ambiguous_after_component_ids: tuple[Ref, ...]
    unmatched_before_component_ids: tuple[Ref, ...]
    unmatched_after_component_ids: tuple[Ref, ...]
    action_ref: Ref | None = None
    schema_version: Ref = "yf_arc3_v5.component_match_result.v1"


class TrackedComponent(FrozenModel):
    """One current visual entity with an identity preserved across frames."""

    entity_id: Ref
    component: ComponentDescription
    first_seen_frame_ref: Ref
    current_frame_ref: Ref
    observation_count: int = Field(ge=1)
    identity_status: Literal["established", "special", "new"]
    match_kind: Literal[
        "initial",
        "stationary_exact",
        "stationary_value_transition",
        "stationary_appearance_transition",
        "translated_exact",
        "extent_transition",
        "ambiguous",
        "appeared",
    ]
    previous_component_ref: Ref | None = None
    possible_predecessor_entity_refs: tuple[Ref, ...] = ()
    delta_row: int = 0
    delta_col: int = 0
    changed_extent_axis: Literal["row", "column"] | None = None
    schema_version: Ref = "yf_arc3_v5.tracked_component.v1"

    @model_validator(mode="after")
    def validate_lineage(self) -> "TrackedComponent":
        require_unique(
            self.possible_predecessor_entity_refs,
            "possible predecessor entity references",
        )
        if self.identity_status == "special" and not self.possible_predecessor_entity_refs:
            raise ValueError("special tracking requires preserved predecessor alternatives")
        if self.match_kind == "initial" and self.previous_component_ref is not None:
            raise ValueError("an initial tracking observation has no predecessor")
        return self


class TemporalTrackingInput(FrozenModel):
    current: ComponentExtractionResult
    frame_ref: Ref
    previous: "TemporalTrackingResult | None" = None
    action_ref: Ref | None = None
    reset_correspondence: bool = False
    schema_version: Ref = "yf_arc3_v5.temporal_tracking_input.v1"


class TemporalTrackingResult(FrozenModel):
    frame_ref: Ref
    entities: tuple[TrackedComponent, ...]
    disappeared_entity_refs: tuple[Ref, ...] = ()
    ambiguous_current_component_refs: tuple[Ref, ...] = ()
    next_entity_sequence: int = Field(ge=1)
    action_ref: Ref | None = None
    schema_version: Ref = "yf_arc3_v5.temporal_tracking_result.v1"

    @model_validator(mode="after")
    def validate_tracking_index(self) -> "TemporalTrackingResult":
        require_unique(tuple(item.entity_id for item in self.entities), "tracked entities")
        require_unique(self.disappeared_entity_refs, "disappeared tracked entities")
        require_unique(
            self.ambiguous_current_component_refs,
            "ambiguous current components",
        )
        return self


class ExactMulticellPeerRelationMeasurement(FrozenModel):
    """One exact translation-invariant silhouette relation between tracked bodies."""

    relation_ref: Ref
    first_entity_ref: Ref
    second_entity_ref: Ref
    first_component_ref: Ref
    second_component_ref: Ref
    first_member_entity_refs: tuple[Ref, ...] = Field(min_length=1, max_length=64)
    second_member_entity_refs: tuple[Ref, ...] = Field(min_length=1, max_length=64)
    first_member_component_refs: tuple[Ref, ...] = Field(min_length=1, max_length=64)
    second_member_component_refs: tuple[Ref, ...] = Field(min_length=1, max_length=64)
    relative_shape_ref: Ref
    area: int = Field(ge=2)
    # Cardinality of the exact silhouette class in the current scene.  A
    # cardinality above two makes a pairwise terminal assignment ambiguous;
    # DRM preserves it as evidence without activating an arbitrary pair goal.
    exact_shape_occurrence_count: int = Field(default=2, ge=2)
    first_value: int = Field(ge=0)
    second_value: int = Field(ge=0)
    residual_row_twice: int
    residual_column_twice: int
    residual_manhattan_twice: int = Field(ge=0)
    joint_observation_count: int = Field(ge=1)
    first_observation_sequence: int = Field(ge=0)
    second_observation_sequence: int = Field(ge=0)
    first_bbox_top: int = Field(ge=0)
    first_bbox_left: int = Field(ge=0)
    first_bbox_bottom: int = Field(ge=0)
    first_bbox_right: int = Field(ge=0)
    second_bbox_top: int = Field(ge=0)
    second_bbox_left: int = Field(ge=0)
    second_bbox_bottom: int = Field(ge=0)
    second_bbox_right: int = Field(ge=0)
    first_is_eight_connected_assembly: bool = False
    second_is_eight_connected_assembly: bool = False
    first_assembly_cotranslation_observed: bool = False
    second_assembly_cotranslation_observed: bool = False
    containing_exact_peer_relation_refs: tuple[Ref, ...] = Field(
        default=(), max_length=256
    )
    strictly_contained_in_larger_exact_peer_relation: bool = False
    internal_to_larger_exact_peer_endpoint_refs: tuple[Ref, ...] = Field(
        default=(), max_length=256
    )
    internal_to_larger_exact_peer_endpoint: bool = False
    schema_version: Ref = "yf_arc3_v5.exact_multicell_peer_relation_measurement.v3"

    @model_validator(mode="after")
    def validate_relation(self) -> "ExactMulticellPeerRelationMeasurement":
        require_unique(self.first_member_entity_refs, "first assembly members")
        require_unique(self.second_member_entity_refs, "second assembly members")
        require_unique(
            self.first_member_component_refs, "first assembly member components"
        )
        require_unique(
            self.second_member_component_refs, "second assembly member components"
        )
        require_unique(
            self.containing_exact_peer_relation_refs,
            "containing exact peer relations",
        )
        require_unique(
            self.internal_to_larger_exact_peer_endpoint_refs,
            "larger exact peer endpoints containing an internal relation",
        )
        if self.first_entity_ref == self.second_entity_ref:
            raise ValueError("an exact peer relation requires two distinct entities")
        if self.residual_manhattan_twice != (
            abs(self.residual_row_twice) + abs(self.residual_column_twice)
        ):
            raise ValueError("exact peer residual magnitude must match signed axes")
        if self.strictly_contained_in_larger_exact_peer_relation != bool(
            self.containing_exact_peer_relation_refs
        ):
            raise ValueError(
                "strict exact-peer containment must match its measured witnesses"
            )
        if self.internal_to_larger_exact_peer_endpoint != bool(
            self.internal_to_larger_exact_peer_endpoint_refs
        ):
            raise ValueError(
                "internal exact-peer endpoint status must match its witnesses"
            )
        return self


class ExactMulticellPeerRelationInput(FrozenModel):
    tracking: TemporalTrackingResult
    zones: ComponentExtractionResult
    maximum_relations: int = Field(default=256, ge=1, le=256)
    schema_version: Ref = "yf_arc3_v5.exact_multicell_peer_relation_input.v1"


class ExactMulticellPeerRelationMeasurements(FrozenModel):
    observation_ref: Ref
    relations: tuple[ExactMulticellPeerRelationMeasurement, ...] = Field(
        default=(), max_length=256
    )
    total_relation_count: int = Field(ge=0)
    deferred_relation_count: int = Field(ge=0)
    priority_frontier_complete: bool = True
    priority_criterion_refs: tuple[Ref, ...] = Field(default=(), max_length=16)
    enumeration_truncated: bool = False
    tracked_component_count: int = Field(ge=0)
    eight_connected_assembly_count: int = Field(ge=0)
    confirmed_cotranslating_assembly_count: int = Field(ge=0)
    descriptive_facts: FrozenMap = Field(default_factory=FrozenMap)
    schema_version: Ref = "yf_arc3_v5.exact_multicell_peer_relation_measurements.v2"

    @model_validator(mode="after")
    def validate_measurements(self) -> "ExactMulticellPeerRelationMeasurements":
        require_unique(
            tuple(item.relation_ref for item in self.relations),
            "exact multicell peer relation references",
        )
        if self.total_relation_count < len(self.relations):
            raise ValueError("total exact peer relation count cannot be below payload")
        if self.deferred_relation_count != self.total_relation_count - len(
            self.relations
        ):
            raise ValueError("deferred exact peer count must match the bounded frontier")
        if self.priority_frontier_complete == self.enumeration_truncated:
            raise ValueError(
                "exact peer frontier completeness must oppose enumeration truncation"
            )
        require_unique(
            self.priority_criterion_refs,
            "exact peer priority criterion references",
        )
        return self


class ExtentTransition(FrozenModel):
    entity_ref: Ref
    before_component_ref: Ref
    after_component_ref: Ref
    axis: Literal["row", "column"]
    before_extent: int = Field(ge=1)
    after_extent: int = Field(ge=1)
    delta_extent: int
    stable_edge: Literal["top", "bottom", "left", "right"]


class TranslationTransition(FrozenModel):
    entity_ref: Ref
    component_ref: Ref
    delta_row: int
    delta_col: int

    @model_validator(mode="after")
    def validate_motion(self) -> "TranslationTransition":
        if self.delta_row == 0 and self.delta_col == 0:
            raise ValueError("translation transition must move")
        return self


class PeerDistanceChangeCandidate(FrozenModel):
    translated_entity_ref: Ref
    stationary_peer_entity_ref: Ref
    axis: Literal["row", "column"]
    before_distance_twice: int = Field(ge=0)
    after_distance_twice: int = Field(ge=0)
    distance_change_kind: Literal["decreased", "increased", "zero", "unchanged"]


class QuantizedExchangeCandidate(FrozenModel):
    source_entity_ref: Ref
    destination_entity_ref: Ref
    axis: Literal["row", "column"]
    quantum: int = Field(gt=0)
    source_quantity_before: int = Field(ge=0)
    source_quantity_after: int = Field(ge=0)
    destination_quantity_before: int = Field(ge=0)
    destination_quantity_after: int = Field(ge=0)
    aggregate_quantity_before: int = Field(ge=0)
    aggregate_quantity_after: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_exchange(self) -> "QuantizedExchangeCandidate":
        if self.source_quantity_after >= self.source_quantity_before:
            raise ValueError("exchange source quantity must decrease")
        if self.destination_quantity_after <= self.destination_quantity_before:
            raise ValueError("exchange destination quantity must increase")
        return self


class ExternalTransportObservationInput(FrozenModel):
    before_tracking: TemporalTrackingResult
    after_tracking: TemporalTrackingResult
    geometry: FrozenMap | None = None
    known_actuator_entity_refs: tuple[Ref, ...] = ()
    pre_action_input_aligned_entity_bboxes: tuple[BoundingBox, ...] = Field(default=(), max_length=3)
    post_action_input_aligned_entity_bboxes: tuple[BoundingBox, ...] = Field(default=(), max_length=3)
    prior_records: tuple[FrozenMap, ...] | None = Field(default=None, max_length=8)
    transition_ref: Ref
    action_ref: Ref | None = None
    evidence_scope_ref: Ref | None = None
    maximum_pairs: int = Field(default=8, ge=1, le=8)
    schema_version: Ref = "yf_arc3_v5.external_transport_observation_input.v1"


class ExternalTransportObservations(FrozenModel):
    rows: tuple[FrozenMap, ...] = ()
    pair_bound_reached: bool = False
    schema_version: Ref = "yf_arc3_v5.external_transport_observations.v1"


class ControlledTransitionAnalysisInput(FrozenModel):
    before_scene: VisualSceneDescription | BlockGridSceneDescription
    after_scene: VisualSceneDescription | BlockGridSceneDescription
    before_tracking: TemporalTrackingResult
    after_tracking: TemporalTrackingResult
    candidate_ref: Ref
    action_ref: Ref
    transition_ref: Ref
    context_epoch: int = Field(default=1, ge=1)
    available_directional_action_refs: tuple[Ref, ...] = ()
    interface_action_translation_deltas: tuple[tuple[Ref, int, int], ...] = ()
    input_aligned_group_connectivity: Literal[0, 4, 8] = 0
    input_aligned_group_without_global_enclosure: bool = False
    known_actuator_entity_refs: tuple[Ref, ...] = ()
    canonical_known_actuator_term_facts: tuple[FrozenMap, ...] | None = Field(
        default=None, max_length=64
    )
    canonical_reversible_control_claim_facts: tuple[FrozenMap, ...] = Field(
        default=(), max_length=64
    )
    canonical_provisional_control_term_facts: tuple[FrozenMap, ...] = Field(
        default=(), max_length=64
    )
    canonical_action_translation_claim_facts: tuple[FrozenMap, ...] = Field(
        default=(), max_length=64
    )
    canonical_periodic_transition_claim_facts: tuple[FrozenMap, ...] = Field(
        default=(), max_length=64
    )
    canonical_relational_lattice_term_facts: tuple[FrozenMap, ...] = Field(default=(), max_length=3)
    declared_cellular_mover_pattern_refs: tuple[Ref, ...] = Field(
        default=(), max_length=64
    )
    # Bounded geometry transported from the control episode. Python only
    # measures exact pose recurrence; DRM decides whether it requires a new
    # control binding after an action-conditioned animation.
    pre_action_input_aligned_entity_bboxes: tuple[BoundingBox, ...] = Field(
        default=(), max_length=3
    )
    post_action_input_aligned_entity_bboxes: tuple[BoundingBox, ...] = Field(
        default=(), max_length=3
    )
    input_aligned_episode_start_bbox: BoundingBox | None = None
    prior_context_effect_carried_reflection_axis_observed: bool = False
    ordered_point_trajectory: FrozenMap = Field(default_factory=FrozenMap)
    schema_version: Ref = "yf_arc3_v5.controlled_transition_analysis_input.v1"


class ControlledTransitionAnalysis(FrozenModel):
    descriptive_facts: FrozenMap
    evidence_refs: tuple[Ref, ...] = Field(min_length=1)
    transition_ref: Ref
    candidate_ref: Ref
    clicked_entity_ref: Ref | None = None
    extent_transitions: tuple[ExtentTransition, ...] = ()
    exchange_candidates: tuple[QuantizedExchangeCandidate, ...] = ()
    translations: tuple[TranslationTransition, ...] = ()
    peer_distance_change_candidates: tuple[PeerDistanceChangeCandidate, ...] = ()
    boundary_component_refs: tuple[Ref, ...] = ()
    local_neighborhood_component_refs: tuple[Ref, ...] = ()
    schema_version: Ref = "yf_arc3_v5.controlled_transition_analysis.v1"

    @model_validator(mode="after")
    def validate_analysis(self) -> "ControlledTransitionAnalysis":
        require_unique(self.evidence_refs, "controlled transition evidence")
        require_unique(self.boundary_component_refs, "boundary component references")
        require_unique(
            self.local_neighborhood_component_refs,
            "local neighborhood component references",
        )
        return self


class GoalCompletionAnalysisInput(FrozenModel):
    before_score: int = Field(ge=0)
    after_score: int = Field(ge=0)
    context_epoch: int | None = Field(default=None, ge=1)
    goal_ref: Ref | None = None
    # Scene-local candidate reference already authored by DRM on the winning
    # route. It is used only for retrospective attribution at this boundary.
    terminal_access_candidate_ref: Ref | None = None
    # The active terminal objective declared on the winning action.  This is a
    # reusable schema reference, never a scene-local route or object binding.
    terminal_objective_contract_ref: Ref | None = None
    terminal_objective_measure_ref: Ref | None = None
    terminal_objective_comparator: Literal["equals"] | None = None
    terminal_objective_target_value: int | None = None
    terminal_objective_predicted_value_after_plan: int | None = Field(
        default=None, ge=0
    )
    # Raw pre-action layer measurements.  At an official success boundary DRM
    # may retrospectively assign their reusable terminal meaning; Python only
    # transports the bounded counts.
    retrospective_coverage_target_cell_count_before_winning_action: (
        int | None
    ) = Field(default=None, ge=0)
    retrospective_uncovered_target_cell_count_before_winning_action: (
        int | None
    ) = Field(default=None, ge=0)
    retrospective_stationary_peer_palette_nonboundary_cell_count_before_winning_action: (
        int | None
    ) = Field(default=None, ge=0)
    retrospective_predicted_stationary_peer_palette_remaining_cell_count_after_exact_coincidence: (
        int | None
    ) = Field(default=None, ge=0)
    retrospective_exact_stationary_peer_morphology_match: bool = False
    # Bounded DRM-authored activity references observed during the completed
    # level. Python transports their compressed order and recurrence count;
    # DRM alone assigns reusable workflow-schema meaning at the boundary.
    retrospective_hot_workflow_ref: Ref | None = None
    retrospective_hot_workflow_activity_refs: tuple[Ref, ...] = Field(
        default=(), max_length=32
    )
    retrospective_hot_workflow_observation_count: int = Field(
        default=0, ge=0, le=4096
    )
    moving_entity_ref: Ref | None = None
    fixed_entity_ref: Ref | None = None
    alignment_axis: Literal["row", "column"] | None = None
    expected_remaining_step_count: int | None = Field(default=None, ge=0)
    action_ref: Ref
    candidate_ref: Ref
    transition_ref: Ref
    distance_is_quantized: bool = False
    exact_terminal_relation_step_expected: bool = False
    lifecycle_revision_group_ref: Ref | None = None
    causally_effective_action_deltas: tuple[int, ...] = Field(
        default=(), max_length=256
    )
    depth_claim_ref: Ref | None = None
    declared_minimum_effective_action_count: int = Field(default=0, ge=0)
    schema_version: Ref = "yf_arc3_v5.goal_completion_analysis_input.v8"

    @model_validator(mode="after")
    def validate_effective_action_deltas(self) -> "GoalCompletionAnalysisInput":
        if any(delta not in (0, 1) for delta in self.causally_effective_action_deltas):
            raise ValueError("effective action deltas must be binary DRM ledger values")
        terminal_bundle_present = (
            self.terminal_objective_contract_ref is not None,
            self.terminal_objective_measure_ref is not None,
            self.terminal_objective_comparator is not None,
            self.terminal_objective_target_value is not None,
            self.terminal_objective_predicted_value_after_plan is not None,
        )
        if any(terminal_bundle_present) and not all(terminal_bundle_present):
            raise ValueError(
                "terminal objective evidence requires contract, measure, comparator, "
                "target, and predicted post-plan value together"
            )
        stationary_peer_counts_present = (
            self.retrospective_stationary_peer_palette_nonboundary_cell_count_before_winning_action
            is not None,
            self.retrospective_predicted_stationary_peer_palette_remaining_cell_count_after_exact_coincidence
            is not None,
        )
        if any(stationary_peer_counts_present) and not all(
            stationary_peer_counts_present
        ):
            raise ValueError(
                "stationary peer cover evidence requires current and predicted "
                "remaining counts together"
            )
        if self.retrospective_exact_stationary_peer_morphology_match and not all(
            stationary_peer_counts_present
        ):
            raise ValueError(
                "an exact stationary peer morphology match requires both mechanical "
                "cell counts"
            )
        hot_workflow_bundle_present = (
            self.retrospective_hot_workflow_ref is not None,
            bool(self.retrospective_hot_workflow_activity_refs),
            self.retrospective_hot_workflow_observation_count > 0,
        )
        if any(hot_workflow_bundle_present) and not all(hot_workflow_bundle_present):
            raise ValueError(
                "hot workflow boundary evidence requires one workflow ref, at "
                "least one activity ref, and a positive observation count"
            )
        return self


class GoalCompletionAnalysis(FrozenModel):
    descriptive_facts: FrozenMap
    evidence_refs: tuple[Ref, ...] = Field(min_length=1)
    transition_ref: Ref
    schema_version: Ref = "yf_arc3_v5.goal_completion_analysis.v1"

class QuantizedPlanReconciliationInput(FrozenModel):
    after_scene: VisualSceneDescription | None = None
    after_tracking: TemporalTrackingResult | None = None
    selected_plan_facts: FrozenMap
    before_score: int = Field(ge=0)
    after_score: int = Field(ge=0)
    transition_ref: Ref
    schema_version: Ref = "yf_arc3_v5.quantized_plan_reconciliation_input.v1"


class QuantizedPlanReconciliationAnalysis(FrozenModel):
    descriptive_delta: FrozenMap
    evidence_refs: tuple[Ref, ...] = Field(min_length=1)
    transition_ref: Ref
    schema_version: Ref = "yf_arc3_v5.quantized_plan_reconciliation_analysis.v1"

class PaletteCanonicalizationInput(FrozenModel):
    frame: FrameGrid
    schema_version: Ref = "yf_arc3_v5.palette_canonicalization_input.v1"


class PaletteCanonicalizationResult(FrozenModel):
    canonical_frame: FrameGrid
    source_to_canonical: tuple[tuple[int, int], ...]
    schema_version: Ref = "yf_arc3_v5.palette_canonicalization_result.v1"


class SceneRebindingMeasurementInput(FrozenModel):
    prior_structure: PaletteCanonicalizationResult
    current_structure: PaletteCanonicalizationResult
    current_tracking: TemporalTrackingResult
    current_frame_ref: Ref
    handoff_claim_ref: Ref
    scope_ref: Ref
    transported_reference_refs: tuple[Ref, ...] = Field(default=(), max_length=4096)
    schema_version: Ref = "yf_arc3_v5.scene_rebinding_measurement_input.v1"

    @model_validator(mode="after")
    def validate_transported_references(self) -> "SceneRebindingMeasurementInput":
        require_unique(
            self.transported_reference_refs,
            "transported scene-rebinding references",
        )
        return self


class SceneRebindingMeasurements(FrozenModel):
    descriptive_facts: FrozenMap
    schema_version: Ref = "yf_arc3_v5.scene_rebinding_measurements.v1"


class UniformBlockReductionInput(FrozenModel):
    frame: FrameGrid
    block_height: int = Field(ge=1)
    block_width: int = Field(ge=1)
    offset_row: int = Field(default=0, ge=0)
    offset_col: int = Field(default=0, ge=0)
    schema_version: Ref = "yf_arc3_v5.uniform_block_reduction_input.v1"


class UniformBlockReductionResult(FrozenModel):
    reduced_frame: FrameGrid | None
    non_uniform_blocks: tuple[BoundingBox, ...]
    covered_bbox: BoundingBox | None
    schema_version: Ref = "yf_arc3_v5.uniform_block_reduction_result.v1"


class GridPartitionInput(FrozenModel):
    frame: FrameGrid
    minimum_grid_rows: int = Field(default=2, ge=1)
    minimum_grid_columns: int = Field(default=2, ge=1)
    schema_version: Ref = "yf_arc3_v5.grid_partition_input.v1"


class GridPartitionCandidate(FrozenModel):
    candidate_id: Ref
    cell_height: int = Field(ge=1)
    cell_width: int = Field(ge=1)
    grid_rows: int = Field(ge=1)
    grid_columns: int = Field(ge=1)
    distinct_pattern_count: int = Field(ge=1)
    repeated_pattern_count: int = Field(ge=0)
    uniform_cell_count: int = Field(ge=0)
    schema_version: Ref = "yf_arc3_v5.grid_partition_candidate.v1"


class GridPartitionResult(FrozenModel):
    candidates: tuple[GridPartitionCandidate, ...]
    selected_candidate_id: Ref | None = None
    schema_version: Ref = "yf_arc3_v5.grid_partition_result.v1"


class GeometryInput(FrozenModel):
    components: ComponentExtractionResult
    schema_version: Ref = "yf_arc3_v5.geometry_input.v1"


class ComponentPairGeometry(FrozenModel):
    first_component_id: Ref
    second_component_id: Ref
    overlap: bool
    edge_contact: bool
    corner_contact: bool
    first_bbox_contains_second: bool
    second_bbox_contains_first: bool
    same_center_row: bool
    same_center_column: bool
    center_delta_twice: tuple[int, int]
    minimum_manhattan_pixel_gap: int = Field(ge=0)
    schema_version: Ref = "yf_arc3_v5.component_pair_geometry.v1"


class GeometryResult(FrozenModel):
    pairs: tuple[ComponentPairGeometry, ...]
    schema_version: Ref = "yf_arc3_v5.geometry_result.v1"


class OrthogonalPerimeterCandidate(FrozenModel):
    candidate_ref: Ref
    component_refs: tuple[Ref, ...] = Field(min_length=8, max_length=64)
    slot_centers_twice: tuple[Position, ...] = Field(min_length=8, max_length=64)
    slot_values: tuple[int, ...] = Field(min_length=8, max_length=64)
    cell_height: int = Field(ge=1)
    cell_width: int = Field(ge=1)
    row_gap: int = Field(ge=1)
    column_gap: int = Field(ge=1)
    schema_version: Ref = "yf_arc3_v5.orthogonal_perimeter_candidate.v1"

    @model_validator(mode="after")
    def validate_perimeter_index(self) -> "OrthogonalPerimeterCandidate":
        if not (
            len(self.component_refs)
            == len(self.slot_centers_twice)
            == len(self.slot_values)
        ):
            raise ValueError("perimeter components, slots, and values must align")
        require_unique(self.component_refs, "perimeter component refs")
        require_unique(self.slot_centers_twice, "perimeter slot centres")
        return self


class OffLatticeQuartetCandidate(FrozenModel):
    candidate_ref: Ref
    component_refs: tuple[Ref, Ref, Ref, Ref]
    component_centers_twice: tuple[Position, Position, Position, Position]
    shared_value: int
    component_height: int = Field(ge=1)
    component_width: int = Field(ge=1)
    common_center_twice: Position
    perimeter_candidate_ref: Ref
    coincident_perimeter_slot_index: int = Field(ge=0, le=63)
    unique_same_value_perimeter_component_ref: Ref
    fits_row_separator_gap: bool
    fits_column_separator_gap: bool
    off_perimeter_lattice: bool
    schema_version: Ref = "yf_arc3_v5.off_lattice_quartet_candidate.v1"

    @model_validator(mode="after")
    def validate_quartet(self) -> "OffLatticeQuartetCandidate":
        require_unique(self.component_refs, "off-lattice quartet component refs")
        require_unique(
            self.component_centers_twice,
            "off-lattice quartet component centres",
        )
        return self


class OrthogonalPerimeterAssemblyInput(FrozenModel):
    components: ComponentExtractionResult
    maximum_perimeter_candidates: int = Field(default=3, ge=1, le=3)
    maximum_quartet_candidates: int = Field(default=3, ge=1, le=3)
    schema_version: Ref = "yf_arc3_v5.orthogonal_perimeter_assembly_input.v1"


class OrthogonalPerimeterAssemblyMeasurements(FrozenModel):
    perimeter_candidates: tuple[OrthogonalPerimeterCandidate, ...] = Field(
        default=(), max_length=3
    )
    off_lattice_quartet_candidates: tuple[OffLatticeQuartetCandidate, ...] = Field(
        default=(), max_length=3
    )
    descriptive_facts: FrozenMap = Field(default_factory=FrozenMap)
    schema_version: Ref = "yf_arc3_v5.orthogonal_perimeter_assembly_measurements.v1"


class MeasuredPosition(FrozenModel):
    position_ref: Ref
    row: int
    column: int


class AffinePositionTerm(FrozenModel):
    position_ref: Ref
    integer_coefficient: int


class AffinePositionRequest(FrozenModel):
    request_ref: Ref
    terms: tuple[AffinePositionTerm, ...] = Field(min_length=1)
    positive_denominator: int = Field(ge=1)
    row_offset_numerator: int = 0
    column_offset_numerator: int = 0
    observed_target_position_ref: Ref | None = None

    @model_validator(mode="after")
    def validate_terms(self) -> "AffinePositionRequest":
        require_unique(
            tuple(item.position_ref for item in self.terms),
            "affine position term refs",
        )
        return self


class DiscreteStateSample(FrozenModel):
    entity_ref: Ref
    observation_index: NonNegativeRevision
    state_ref: Ref


class DerivedGeometryStateInput(FrozenModel):
    measurement_context_ref: Ref
    positions: tuple[MeasuredPosition, ...] = Field(default=(), max_length=64)
    affine_requests: tuple[AffinePositionRequest, ...] = Field(
        default=(), max_length=64
    )
    discrete_state_samples: tuple[DiscreteStateSample, ...] = Field(
        default=(), max_length=128
    )
    schema_version: Ref = "yf_arc3_v5.derived_geometry_state_input.v1"

    @model_validator(mode="after")
    def validate_measurement_scope(self) -> "DerivedGeometryStateInput":
        if not self.affine_requests and not self.discrete_state_samples:
            raise ValueError("at least one finite geometry or state trace is required")
        require_unique(
            tuple(item.position_ref for item in self.positions),
            "measured position refs",
        )
        require_unique(
            tuple(item.request_ref for item in self.affine_requests),
            "affine request refs",
        )
        require_unique(
            tuple(
                f"{item.entity_ref}:{item.observation_index}"
                for item in self.discrete_state_samples
            ),
            "discrete state entity/index pairs",
        )
        known_positions = {item.position_ref for item in self.positions}
        required_positions = {
            term.position_ref
            for request in self.affine_requests
            for term in request.terms
        }
        missing = required_positions - known_positions
        if missing:
            raise ValueError(f"affine request references unknown positions: {sorted(missing)}")
        return self


class AffinePositionMeasurement(FrozenModel):
    request_ref: Ref
    row_numerator: int
    column_numerator: int
    positive_denominator: int = Field(ge=1)
    integral_position: bool
    derived_position: Position | None = None
    observed_target_matches: bool | None = None

    @model_validator(mode="after")
    def validate_integral_position(self) -> "AffinePositionMeasurement":
        if self.integral_position != (self.derived_position is not None):
            raise ValueError("integral status and derived position must agree")
        return self


class DiscreteStateTransitionMeasurement(FrozenModel):
    entity_ref: Ref
    before_observation_index: NonNegativeRevision
    after_observation_index: NonNegativeRevision
    before_state_ref: Ref
    after_state_ref: Ref


class DiscreteStateReturnMeasurement(FrozenModel):
    entity_ref: Ref
    earlier_observation_index: NonNegativeRevision
    later_observation_index: NonNegativeRevision
    state_ref: Ref


class DiscreteStateTraceMeasurement(FrozenModel):
    entity_ref: Ref
    observed_state_refs: tuple[Ref, ...] = Field(min_length=1)
    transitions: tuple[DiscreteStateTransitionMeasurement, ...] = ()
    returns: tuple[DiscreteStateReturnMeasurement, ...] = ()


class DerivedGeometryStateMeasurements(FrozenModel):
    affine_positions: tuple[AffinePositionMeasurement, ...] = ()
    discrete_state_traces: tuple[DiscreteStateTraceMeasurement, ...] = ()
    descriptive_facts: FrozenMap = Field(default_factory=FrozenMap)
    schema_version: Ref = "yf_arc3_v5.derived_geometry_state_measurements.v1"


class ViewportShiftObservation(FrozenModel):
    observation_ref: Ref
    delta_row: int
    delta_col: int
    anchor_refs: tuple[Ref, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def validate_anchor_refs(self) -> "ViewportShiftObservation":
        require_unique(self.anchor_refs, "viewport shift anchor refs")
        return self


class TerrainConfigurationObservation(FrozenModel):
    configuration_ref: Ref
    traversable_positions: tuple[Position, ...] = Field(default=(), max_length=512)
    support_contact_positions: tuple[Position, ...] = Field(default=(), max_length=128)

    @model_validator(mode="after")
    def validate_positions(self) -> "TerrainConfigurationObservation":
        if len(set(self.traversable_positions)) != len(self.traversable_positions):
            raise ValueError("traversable positions must be unique per configuration")
        if len(set(self.support_contact_positions)) != len(
            self.support_contact_positions
        ):
            raise ValueError("support contact positions must be unique per configuration")
        return self


class MobileLayerObservation(FrozenModel):
    configuration_ref: Ref
    present: bool
    occupied_positions: tuple[Position, ...] = Field(default=(), max_length=128)

    @model_validator(mode="after")
    def validate_mobile_layer(self) -> "MobileLayerObservation":
        if len(set(self.occupied_positions)) != len(self.occupied_positions):
            raise ValueError("mobile occupied positions must be unique")
        if self.present != bool(self.occupied_positions):
            raise ValueError(
                "present mobile requires occupied positions and absent mobile requires none"
            )
        return self


class TerrainConfigurationTransitionObservation(FrozenModel):
    transition_ref: Ref
    before_configuration_ref: Ref
    after_configuration_ref: Ref
    action_ref: Ref | None = None


class ViewportSupportTerrainInput(FrozenModel):
    measurement_context_ref: Ref
    viewport_shifts: tuple[ViewportShiftObservation, ...] = Field(
        default=(), max_length=32
    )
    configurations: tuple[TerrainConfigurationObservation, ...] = Field(
        default=(), max_length=64
    )
    transitions: tuple[TerrainConfigurationTransitionObservation, ...] = Field(
        default=(), max_length=128
    )
    mobile_layers: tuple[MobileLayerObservation, ...] = Field(
        default=(), max_length=64
    )
    declared_directional_action_refs: tuple[Ref, ...] = Field(
        default=(), max_length=16
    )
    schema_version: Ref = "yf_arc3_v5.viewport_support_terrain_input.v1"

    @model_validator(mode="after")
    def validate_measurement_scope(self) -> "ViewportSupportTerrainInput":
        if not self.viewport_shifts and not self.configurations:
            raise ValueError("at least one viewport shift or configuration is required")
        require_unique(
            tuple(item.observation_ref for item in self.viewport_shifts),
            "viewport shift observation refs",
        )
        require_unique(
            tuple(item.configuration_ref for item in self.configurations),
            "terrain configuration refs",
        )
        require_unique(
            tuple(item.transition_ref for item in self.transitions),
            "terrain transition refs",
        )
        require_unique(
            tuple(item.configuration_ref for item in self.mobile_layers),
            "mobile layer configuration refs",
        )
        require_unique(
            self.declared_directional_action_refs,
            "declared directional action refs",
        )
        known = {item.configuration_ref for item in self.configurations}
        referenced = {
            ref
            for item in self.transitions
            for ref in (
                item.before_configuration_ref,
                item.after_configuration_ref,
            )
        }
        missing = referenced - known
        if missing:
            raise ValueError(
                f"terrain transitions reference unknown configurations: {sorted(missing)}"
            )
        missing_mobile_configurations = {
            item.configuration_ref for item in self.mobile_layers
        } - known
        if missing_mobile_configurations:
            raise ValueError(
                "mobile layers reference unknown configurations: "
                f"{sorted(missing_mobile_configurations)}"
            )
        return self


class ConfigurationGraphEdge(FrozenModel):
    transition_ref: Ref
    before_configuration_ref: Ref
    after_configuration_ref: Ref
    action_ref: Ref | None = None


class TerrainTransitionMeasurement(FrozenModel):
    transition_ref: Ref
    added_traversable_positions: tuple[Position, ...] = ()
    removed_traversable_positions: tuple[Position, ...] = ()
    gained_support_contact_positions: tuple[Position, ...] = ()
    lost_support_contact_positions: tuple[Position, ...] = ()


class MobileUnderlayTransitionMeasurement(FrozenModel):
    transition_ref: Ref
    action_ref: Ref | None = None
    action_is_declared_directional: bool = False
    before_mobile_present: bool | None = None
    after_mobile_present: bool | None = None
    same_mobile_occupied_positions: bool | None = None
    mobile_pose_changed: bool | None = None
    before_complete_traversable_underlay: bool | None = None
    after_complete_traversable_underlay: bool | None = None
    incomplete_underlay_position_count_after: int | None = Field(default=None, ge=0)


class ViewportSupportTerrainMeasurements(FrozenModel):
    configuration_edges: tuple[ConfigurationGraphEdge, ...] = ()
    terrain_transitions: tuple[TerrainTransitionMeasurement, ...] = ()
    mobile_underlay_transitions: tuple[MobileUnderlayTransitionMeasurement, ...] = ()
    persistent_anchor_refs: tuple[Ref, ...] = ()
    descriptive_facts: FrozenMap = Field(default_factory=FrozenMap)
    schema_version: Ref = "yf_arc3_v5.viewport_support_terrain_measurements.v1"


class ObservedTopologyEdge(FrozenModel):
    left_ref: Ref
    right_ref: Ref

    @model_validator(mode="after")
    def validate_canonical_edge(self) -> "ObservedTopologyEdge":
        if self.left_ref >= self.right_ref:
            raise ValueError("topology edges must use distinct canonical endpoints")
        return self


class TopologyConfigurationObservation(FrozenModel):
    configuration_ref: Ref
    node_refs: tuple[Ref, ...] = Field(default=(), max_length=128)
    adjacency_edges: tuple[ObservedTopologyEdge, ...] = Field(
        default=(), max_length=512
    )

    @model_validator(mode="after")
    def validate_graph(self) -> "TopologyConfigurationObservation":
        require_unique(self.node_refs, "topology configuration node refs")
        edge_keys = tuple(
            f"{edge.left_ref}\0{edge.right_ref}" for edge in self.adjacency_edges
        )
        require_unique(edge_keys, "topology configuration edges")
        known = set(self.node_refs)
        missing = {
            endpoint
            for edge in self.adjacency_edges
            for endpoint in (edge.left_ref, edge.right_ref)
            if endpoint not in known
        }
        if missing:
            raise ValueError(f"topology edges reference unknown nodes: {sorted(missing)}")
        return self


class TopologyTransitionObservation(FrozenModel):
    transition_ref: Ref
    before_configuration_ref: Ref
    after_configuration_ref: Ref
    action_ref: Ref | None = None


class LandingProjectionObservation(FrozenModel):
    candidate_ref: Ref
    entity_ref: Ref
    region_ref: Ref
    projected_positions: tuple[Position, ...] = Field(min_length=1, max_length=128)
    permissible_positions: tuple[Position, ...] = Field(min_length=1, max_length=512)
    occupied_positions: tuple[Position, ...] = Field(default=(), max_length=512)

    @model_validator(mode="after")
    def validate_position_sets(self) -> "LandingProjectionObservation":
        for values, label in (
            (self.projected_positions, "projected positions"),
            (self.permissible_positions, "permissible positions"),
            (self.occupied_positions, "occupied positions"),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{label} must be unique")
        return self


class InteractionTopologyInput(FrozenModel):
    measurement_context_ref: Ref
    configurations: tuple[TopologyConfigurationObservation, ...] = Field(
        default=(), max_length=64
    )
    transitions: tuple[TopologyTransitionObservation, ...] = Field(
        default=(), max_length=128
    )
    landing_projections: tuple[LandingProjectionObservation, ...] = Field(
        default=(), max_length=64
    )
    schema_version: Ref = "yf_arc3_v5.interaction_topology_input.v1"

    @model_validator(mode="after")
    def validate_measurement_scope(self) -> "InteractionTopologyInput":
        if not self.configurations and not self.landing_projections:
            raise ValueError("at least one configuration or landing projection is required")
        require_unique(
            tuple(item.configuration_ref for item in self.configurations),
            "topology configuration refs",
        )
        require_unique(
            tuple(item.transition_ref for item in self.transitions),
            "topology transition refs",
        )
        require_unique(
            tuple(item.candidate_ref for item in self.landing_projections),
            "landing projection candidate refs",
        )
        known = {item.configuration_ref for item in self.configurations}
        referenced = {
            ref
            for item in self.transitions
            for ref in (item.before_configuration_ref, item.after_configuration_ref)
        }
        missing = referenced - known
        if missing:
            raise ValueError(
                f"topology transitions reference unknown configurations: {sorted(missing)}"
            )
        return self


class TopologyTransitionMeasurement(FrozenModel):
    transition_ref: Ref
    added_adjacency_edges: tuple[ObservedTopologyEdge, ...] = ()
    removed_adjacency_edges: tuple[ObservedTopologyEdge, ...] = ()
    before_connected_component_count: int = Field(ge=0)
    after_connected_component_count: int = Field(ge=0)


class LandingFitMeasurement(FrozenModel):
    candidate_ref: Ref
    entity_ref: Ref
    region_ref: Ref
    outside_permissible_count: int = Field(ge=0)
    occupied_overlap_count: int = Field(ge=0)
    exact_fit: bool


class InteractionTopologyMeasurements(FrozenModel):
    topology_transitions: tuple[TopologyTransitionMeasurement, ...] = ()
    landing_fits: tuple[LandingFitMeasurement, ...] = ()
    descriptive_facts: FrozenMap = Field(default_factory=FrozenMap)
    schema_version: Ref = "yf_arc3_v5.interaction_topology_measurements.v1"


class EliminationStateObservation(FrozenModel):
    state_ref: Ref
    occupied_ref_set: tuple[Ref, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def validate_occupied_refs(self) -> "EliminationStateObservation":
        require_unique(self.occupied_ref_set, "elimination occupied refs")
        return self


class EliminationTransitionObservation(FrozenModel):
    transition_ref: Ref
    before_state_ref: Ref
    after_state_ref: Ref
    operator_ref: Ref
    removed_ref_set: tuple[Ref, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def validate_removed_refs(self) -> "EliminationTransitionObservation":
        require_unique(self.removed_ref_set, "elimination removed refs")
        return self


class EliminationViabilityInput(FrozenModel):
    measurement_context_ref: Ref
    states: tuple[EliminationStateObservation, ...] = Field(
        min_length=1, max_length=128
    )
    transitions: tuple[EliminationTransitionObservation, ...] = Field(
        default=(), max_length=512
    )
    start_state_ref: Ref
    declared_terminal_state_refs: tuple[Ref, ...] = Field(min_length=1, max_length=32)
    max_expanded_states: int = Field(default=128, ge=1, le=4096)
    max_depth: int = Field(default=128, ge=1, le=128)
    transition_graph_complete: bool = True
    operator_ref: Ref | None = None
    terminal_condition_ref: Ref | None = None
    premise_claim_refs: tuple[Ref, ...] = ()
    frame_ref: Ref | None = None
    analysis_scope: str | None = None
    schema_version: Ref = "yf_arc3_v5.elimination_viability_input.v1"

    @model_validator(mode="after")
    def validate_search_scope(self) -> "EliminationViabilityInput":
        require_unique(
            tuple(item.state_ref for item in self.states), "elimination state refs"
        )
        require_unique(
            tuple(item.transition_ref for item in self.transitions),
            "elimination transition refs",
        )
        require_unique(
            self.declared_terminal_state_refs, "declared terminal state refs"
        )
        known = {item.state_ref for item in self.states}
        referenced = {
            self.start_state_ref,
            *self.declared_terminal_state_refs,
            *(
                ref
                for item in self.transitions
                for ref in (item.before_state_ref, item.after_state_ref)
            ),
        }
        missing = referenced - known
        if missing:
            raise ValueError(
                f"elimination search references unknown states: {sorted(missing)}"
            )
        return self


class EliminationStateViability(FrozenModel):
    state_ref: Ref
    terminal_reachable: bool | None = None
    shortest_terminal_distance: int | None = Field(default=None, ge=0)
    exact_dead_end: bool = False

    @model_validator(mode="after")
    def validate_viability(self) -> "EliminationStateViability":
        if (self.shortest_terminal_distance is not None) != (
            self.terminal_reachable is True
        ):
            raise ValueError("reachable status and terminal distance must agree")
        if self.exact_dead_end != (self.terminal_reachable is False):
            raise ValueError("exact dead end requires exact unreachable status")
        return self


class EliminationViabilityMeasurements(FrozenModel):
    state_viability: tuple[EliminationStateViability, ...]
    descriptive_facts: FrozenMap = Field(default_factory=FrozenMap)
    schema_version: Ref = "yf_arc3_v5.elimination_viability_measurements.v1"


class RelationalEliminationGraphInput(FrozenModel):
    schema_version: Ref = "yf_arc3_v5.relational_elimination_graph_input.v1"
    cell_positions: tuple[Position, ...] = Field(min_length=1, max_length=32)
    occupied_positions: tuple[Position, ...] = Field(min_length=1, max_length=32)
    half_translation_quantum: int = Field(ge=1)
    cell_palette_rows: tuple[tuple[Position, int], ...] = Field(min_length=1, max_length=32)
    free_palette_values: tuple[int, ...] = Field(min_length=1, max_length=16)
    operator_ref: Ref
    terminal_condition_ref: Ref
    terminal_occupancy_count: int = Field(ge=1, le=32)
    measurement_context_ref: Ref
    premise_claim_refs: tuple[Ref, ...] = Field(min_length=1, max_length=64)
    frame_ref: Ref
    analysis_scope: str

    @model_validator(mode="after")
    def validate_cells(self) -> "RelationalEliminationGraphInput":
        require_unique(self.cell_positions, "relational graph cells")
        require_unique(self.occupied_positions, "relational graph occupied cells")
        # Current occupancy and the measured static landing domain are distinct.
        # A transported member remains observed outside that domain; its vacated
        # position does not thereby become an admitted static landing.
        if len(set(self.occupied_positions) | set(self.cell_positions)) > 32:
            raise ValueError("relational graph combined measured positions exceed their bound")
        require_unique(tuple(position for position, _ in self.cell_palette_rows), "relational graph palette positions")
        if {position for position, _ in self.cell_palette_rows} != set(self.cell_positions):
            raise ValueError("relational graph palette measurements must cover its cells exactly")
        return self


class RelationalEliminationGraphMeasurements(FrozenModel):
    viability_request: EliminationViabilityInput
    reception_observation_facts: FrozenMap = Field(default_factory=FrozenMap)
    transport_observation_facts: FrozenMap = Field(default_factory=FrozenMap)
    schema_version: Ref = "yf_arc3_v5.relational_elimination_graph_measurements.v1"


class AbductiveExplanationObservation(FrozenModel):
    """One declared explanation projected into a finite mechanical comparison."""

    explanation_ref: Ref
    covered_obligation_refs: tuple[Ref, ...] = Field(default=(), max_length=32)
    dependency_refs: tuple[Ref, ...] = Field(default=(), max_length=32)
    contradiction_refs: tuple[Ref, ...] = Field(default=(), max_length=32)
    forbidden_instance_fact_refs: tuple[Ref, ...] = Field(default=(), max_length=32)
    falsifier_refs: tuple[Ref, ...] = Field(default=(), max_length=32)
    discriminating_test_refs: tuple[Ref, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def validate_ref_sets(self) -> "AbductiveExplanationObservation":
        for label, refs in (
            ("covered obligations", self.covered_obligation_refs),
            ("dependencies", self.dependency_refs),
            ("contradictions", self.contradiction_refs),
            ("forbidden instance facts", self.forbidden_instance_fact_refs),
            ("falsifiers", self.falsifier_refs),
            ("discriminating tests", self.discriminating_test_refs),
        ):
            require_unique(refs, f"abductive {label}")
        return self


class AbductiveVersionSpaceInput(FrozenModel):
    """Bounded declared explanations and the obligations they must account for."""

    isolated_reasoning_scope_ref: Ref
    measurement_context_ref: Ref
    required_obligation_refs: tuple[Ref, ...] = Field(min_length=1, max_length=32)
    explanations: tuple[AbductiveExplanationObservation, ...] = Field(
        min_length=1, max_length=12
    )
    schema_version: Ref = "yf_arc3_v5.abductive_version_space_input.v1"

    @model_validator(mode="after")
    def validate_explanations(self) -> "AbductiveVersionSpaceInput":
        require_unique(self.required_obligation_refs, "abductive required obligations")
        require_unique(
            tuple(item.explanation_ref for item in self.explanations),
            "abductive explanation refs",
        )
        required = set(self.required_obligation_refs)
        for item in self.explanations:
            unknown = set(item.covered_obligation_refs) - required
            if unknown:
                raise ValueError(
                    f"explanation {item.explanation_ref!r} covers undeclared obligations: "
                    f"{sorted(unknown)}"
                )
        return self


class AbductiveExplanationRejection(FrozenModel):
    explanation_ref: Ref
    reason_refs: tuple[Ref, ...] = Field(min_length=1, max_length=4)
    missing_obligation_refs: tuple[Ref, ...] = Field(default=(), max_length=32)


class AbductiveDominanceObservation(FrozenModel):
    dominant_explanation_ref: Ref
    dominated_explanation_ref: Ref
    strict_dimension_refs: tuple[Ref, ...] = Field(min_length=1, max_length=2)


class AbductiveVersionSpaceMeasurements(FrozenModel):
    """Exact filtering and Pareto dominance; symbolic commitment stays declarative."""

    admissible_explanation_refs: tuple[Ref, ...] = Field(default=(), max_length=12)
    rejections: tuple[AbductiveExplanationRejection, ...] = Field(
        default=(), max_length=12
    )
    dominance_observations: tuple[AbductiveDominanceObservation, ...] = Field(
        default=(), max_length=132
    )
    undominated_explanation_refs: tuple[Ref, ...] = Field(default=(), max_length=12)
    surviving_discriminating_test_refs: FrozenMap = Field(default_factory=FrozenMap)
    descriptive_facts: FrozenMap = Field(default_factory=FrozenMap)
    schema_version: Ref = "yf_arc3_v5.abductive_version_space_measurements.v1"


class TimelineSample(FrozenModel):
    entity_ref: Ref
    frame_ref: Ref | None = None
    scene_observation_value: int | str | None = None
    frame_index: NonNegativeRevision
    state_ref: Ref
    instance_refs: tuple[Ref, ...] = Field(min_length=1, max_length=32)
    occupied_slot_refs: tuple[Ref, ...] = Field(default=(), max_length=64)
    preceding_action_ref: Ref | None = None
    instance_pose_refs: FrozenMap = Field(default_factory=FrozenMap)
    instance_value_refs: FrozenMap = Field(default_factory=FrozenMap)
    instance_bounds: FrozenMap = Field(default_factory=FrozenMap)
    observed_cell_facts: FrozenMap = Field(default_factory=FrozenMap)

    @model_validator(mode="after")
    def validate_sample_refs(self) -> "TimelineSample":
        require_unique(self.instance_refs, "timeline instance refs")
        require_unique(self.occupied_slot_refs, "timeline occupied slot refs")
        for projection in (self.instance_pose_refs, self.instance_value_refs, self.instance_bounds):
            if projection and set(projection) != set(self.instance_refs):
                raise ValueError("instance projections must cover exactly the observed instances")
        return self


class TemporalConcurrencyInput(FrozenModel):
    measurement_context_ref: Ref
    # Mechanical carrier envelope: 64 morphology groups x 65 observed states.
    # Per-recording and search bounds remain independent and unchanged.
    samples: tuple[TimelineSample, ...] = Field(min_length=1, max_length=64 * 65)
    requested_detail: Literal["full", "presence"] = "full"
    recorded_rendered_frame_refs: tuple[Ref, ...] = Field(default=(), max_length=4096)
    candidate_reverse_rendered_frame_refs: tuple[Ref, ...] = Field(default=(), max_length=64)
    source_authorized_reset_observed: bool | None = None
    source_reset_action_ref: Ref | None = None
    measured_translation_action_refs: tuple[Ref, ...] = Field(default=(), max_length=16)
    observation_evidence_refs: tuple[Ref, ...] = Field(default=(), max_length=65)
    observation_revision: NonNegativeRevision = 0
    schema_version: Ref = "yf_arc3_v5.temporal_concurrency_input.v1"

    @model_validator(mode="after")
    def validate_timeline_scope(self) -> "TemporalConcurrencyInput":
        if (self.recorded_rendered_frame_refs and self.candidate_reverse_rendered_frame_refs
                and not self.observation_evidence_refs):
            raise ValueError("rendered history comparison requires observation evidence refs")
        require_unique(self.observation_evidence_refs, "temporal observation evidence refs")
        require_unique(
            tuple(f"{item.entity_ref}\0{item.frame_index}" for item in self.samples),
            "timeline entity/frame pairs",
        )
        if len({item.entity_ref for item in self.samples}) > 64:
            raise ValueError("timeline entity count exceeds hard bound")
        return self


class TimelineEntityMeasurement(FrozenModel):
    entity_ref: Ref
    observed_frame_indices: tuple[NonNegativeRevision, ...]
    ordered_transition_count: int = Field(ge=0)
    state_return_count: int = Field(ge=0)
    duplicated_instance_frame_count: int = Field(ge=0)


class TemporalPairChangeMeasurement(FrozenModel):
    left_entity_ref: Ref
    right_entity_ref: Ref
    shared_transition_interval_count: int = Field(ge=0)
    simultaneous_state_change_count: int = Field(ge=0)


class TemporalConcurrencyMeasurements(FrozenModel):
    entity_timelines: tuple[TimelineEntityMeasurement, ...]
    pair_changes: tuple[TemporalPairChangeMeasurement, ...] = Field(
        default=(), max_length=2016
    )
    descriptive_facts: FrozenMap = Field(default_factory=FrozenMap)
    schema_version: Ref = "yf_arc3_v5.temporal_concurrency_measurements.v1"


class TemporalReplaySequenceInput(FrozenModel):
    """Two source-declared ordered sequences for exact mechanical comparison."""

    measurement_context_ref: Ref
    observation_frame_ref: Ref | None = None
    recorded_transition_delta_refs: tuple[Ref, ...] = Field(
        default=(), max_length=64
    )
    replay_transition_delta_refs: tuple[Ref, ...] = Field(
        default=(), max_length=128
    )
    replay_state_refs: tuple[Ref, ...] = Field(default=(), max_length=129)
    sequence_measurement_kind: Literal[
        "ordered_transition_deltas", "ordered_observed_states"
    ] = "ordered_transition_deltas"
    recorded_observed_state_refs: tuple[Ref, ...] = Field(default=(), max_length=64)
    candidate_observed_state_refs: tuple[Ref, ...] = Field(default=(), max_length=128)
    independent_command_observed: bool = False
    recorded_entity_ref: Ref | None = None
    candidate_entity_ref: Ref | None = None
    candidate_identifier_change_count: int = Field(default=0, ge=0, le=127)
    recorded_instance_bounds: tuple[BoundingBox, ...] = Field(default=(), max_length=65)
    recorded_action_refs: tuple[Ref, ...] = Field(default=(), max_length=64)
    candidate_instance_bounds: tuple[BoundingBox, ...] = Field(default=(), max_length=128)
    recording_frame_index: NonNegativeRevision | None = None
    replay_phase_frame_index: NonNegativeRevision | None = None
    phase_action_ref: Ref | None = None
    recorded_rendered_frame_refs: tuple[Ref, ...] = Field(default=(), max_length=4096)
    candidate_reverse_rendered_frame_refs: tuple[Ref, ...] = Field(default=(), max_length=4096)
    source_authorized_reset_observed: bool | None = None
    phase_action_conditioned: bool = False
    state_return_observed: bool = False
    duplicate_instance_observed: bool = False
    schema_version: Ref = "yf_arc3_v5.temporal_replay_sequence_input.v1"

    @model_validator(mode="after")
    def validate_replay_scope(self) -> "TemporalReplaySequenceInput":
        if self.sequence_measurement_kind == "ordered_observed_states":
            if not self.recorded_observed_state_refs:
                raise ValueError("ordered state comparison requires a recorded sequence")
            if self.recorded_instance_bounds and len(self.recorded_instance_bounds) != len(self.recorded_observed_state_refs) + 1:
                raise ValueError("recording bounds must include the initial state and every observed transition")
            if self.recorded_action_refs and len(self.recorded_action_refs) != len(self.recorded_observed_state_refs):
                raise ValueError("recording actions must delimit exactly the observed transitions")
            if self.candidate_instance_bounds and len(self.candidate_instance_bounds) != len(self.candidate_observed_state_refs):
                raise ValueError("candidate bounds must cover exactly the observed sequence")
            if (self.recorded_transition_delta_refs or self.replay_transition_delta_refs
                    or self.replay_state_refs):
                raise ValueError("state and transition sequence projections cannot be mixed")
            return self
        if self.recorded_observed_state_refs or self.candidate_observed_state_refs:
            raise ValueError("observed state sequences require their explicit measurement kind")
        if not self.recorded_transition_delta_refs:
            raise ValueError("transition comparison requires a recorded sequence")
        if len(self.replay_state_refs) != len(self.replay_transition_delta_refs) + 1:
            raise ValueError("replay states must delimit every replay transition")
        if not self.replay_transition_delta_refs and not self.recorded_rendered_frame_refs:
            raise ValueError("an empty replay prefix requires rendered history evidence")
        return self


class TemporalReplaySequenceMeasurements(FrozenModel):
    comparable_prefix_count: int = Field(ge=0)
    matching_prefix_count: int = Field(ge=0)
    first_divergence_index: int | None = Field(default=None, ge=0)
    replay_exhaustion_index: int = Field(ge=1)
    post_exhaustion_observation_count: int = Field(ge=0)
    persistent_post_exhaustion_count: int = Field(ge=0)
    descriptive_facts: FrozenMap = Field(default_factory=FrozenMap)
    schema_version: Ref = "yf_arc3_v5.temporal_replay_sequence_measurements.v1"


class StepVector(FrozenModel):
    label: Ref
    delta_row: int
    delta_col: int

    @model_validator(mode="after")
    def validate_nonzero(self) -> "StepVector":
        if self.delta_row == 0 and self.delta_col == 0:
            raise ValueError("a step vector must move")
        return self


class ReachabilityInput(FrozenModel):
    rows: int = Field(ge=1, le=64)
    columns: int = Field(ge=1, le=64)
    start: Position
    traversable: tuple[Position, ...]
    goals: tuple[Position, ...]
    steps: tuple[StepVector, ...] = Field(min_length=1, max_length=16)
    schema_version: Ref = "yf_arc3_v5.reachability_input.v1"

    @model_validator(mode="after")
    def validate_domain(self) -> "ReachabilityInput":
        positions = (self.start, *self.traversable, *self.goals)
        if any(
            row < 0 or col < 0 or row >= self.rows or col >= self.columns
            for row, col in positions
        ):
            raise ValueError("reachability positions must be inside the finite grid")
        if self.start not in self.traversable:
            raise ValueError("start must be traversable")
        if len(set(self.traversable)) != len(self.traversable):
            raise ValueError("traversable positions must be unique")
        labels = tuple(step.label for step in self.steps)
        if len(labels) != len(set(labels)):
            raise ValueError("step labels must be unique")
        return self


class PositionDistance(FrozenModel):
    position: Position
    distance: int = Field(ge=0)


class GoalReachability(FrozenModel):
    goal: Position
    reachable: bool
    distance: int | None = Field(default=None, ge=0)
    shortest_first_step_labels: tuple[Ref, ...] = ()
    first_shortest_step_labels: tuple[Ref, ...] = ()

    @model_validator(mode="after")
    def validate_reachability(self) -> "GoalReachability":
        if self.reachable != (self.distance is not None):
            raise ValueError("reachable and distance must agree")
        if not self.reachable and self.first_shortest_step_labels:
            raise ValueError("an unreachable goal cannot expose a route witness")
        if self.distance is not None and len(self.first_shortest_step_labels) != self.distance:
            raise ValueError("route witness length must equal goal distance")
        return self


class ReachabilityResult(FrozenModel):
    distances: tuple[PositionDistance, ...]
    goals: tuple[GoalReachability, ...]
    selected_goal: Position | None = None
    selected_first_step: Ref | None = None
    schema_version: Ref = "yf_arc3_v5.reachability_result.v1"


class RelationEdge(FrozenModel):
    edge_id: Ref
    source_ref: Ref
    target_ref: Ref


class RelationQuery(FrozenModel):
    query_id: Ref
    source_ref: Ref
    target_ref: Ref


class RelationGraphInput(FrozenModel):
    graph_id: Ref
    dimension: RelationDimension
    relation_ref: Ref
    node_refs: tuple[Ref, ...] = Field(min_length=1, max_length=512)
    edges: tuple[RelationEdge, ...] = Field(default=(), max_length=2048)
    queries: tuple[RelationQuery, ...] = Field(default=(), max_length=64)
    directed: bool = False
    schema_version: Ref = "yf_arc3_v5.relation_graph_input.v2"

    @model_validator(mode="after")
    def validate_graph(self) -> "RelationGraphInput":
        node_set = set(self.node_refs)
        if len(node_set) != len(self.node_refs):
            raise ValueError("relation graph nodes must be unique")
        if len({edge.edge_id for edge in self.edges}) != len(self.edges):
            raise ValueError("relation graph edge ids must be unique")
        if len({query.query_id for query in self.queries}) != len(self.queries):
            raise ValueError("relation graph query ids must be unique")
        referenced = {
            ref for edge in self.edges for ref in (edge.source_ref, edge.target_ref)
        } | {
            ref
            for query in self.queries
            for ref in (query.source_ref, query.target_ref)
        }
        if not referenced <= node_set:
            raise ValueError("relation graph edges and queries must reference nodes")
        return self


class RelationReachability(FrozenModel):
    query_id: Ref
    source_ref: Ref
    target_ref: Ref
    reachable: bool
    minimum_edge_count: int | None = Field(default=None, ge=0)
    first_shortest_path: tuple[Ref, ...] = ()
    required_intermediate_refs: tuple[Ref, ...] = ()

    @model_validator(mode="after")
    def validate_reachability(self) -> "RelationReachability":
        if self.reachable != (self.minimum_edge_count is not None):
            raise ValueError("reachable and minimum edge count must agree")
        if self.reachable != bool(self.first_shortest_path):
            raise ValueError("reachable and first shortest path must agree")
        if self.first_shortest_path and (
            self.first_shortest_path[0] != self.source_ref
            or self.first_shortest_path[-1] != self.target_ref
            or len(self.first_shortest_path) != int(self.minimum_edge_count) + 1
        ):
            raise ValueError("first shortest path must connect the query endpoints")
        if not set(self.required_intermediate_refs) <= set(
            self.first_shortest_path[1:-1]
        ):
            raise ValueError("required intermediates must belong to the shortest path")
        return self


class RelationGraphResult(FrozenModel):
    graph_id: Ref
    dimension: RelationDimension
    relation_ref: Ref
    directed: bool
    weak_components: tuple[tuple[Ref, ...], ...]
    reachability: tuple[RelationReachability, ...]
    schema_version: Ref = "yf_arc3_v5.relation_graph_result.v2"


class QuantityValue(FrozenModel):
    entity_ref: Ref
    quantity: int = Field(ge=0)
    capacity: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_capacity(self) -> "QuantityValue":
        if self.capacity is not None and self.quantity > self.capacity:
            raise ValueError("initial quantity cannot exceed capacity")
        return self


class QuantityTransfer(FrozenModel):
    transition_ref: Ref
    source_ref: Ref
    target_ref: Ref
    amount: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_endpoints(self) -> "QuantityTransfer":
        if self.source_ref == self.target_ref:
            raise ValueError("a quantity transfer requires distinct endpoints")
        return self


class ConservationGroup(FrozenModel):
    group_ref: Ref
    member_refs: tuple[Ref, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_members(self) -> "ConservationGroup":
        if len(set(self.member_refs)) != len(self.member_refs):
            raise ValueError("conservation group members must be unique")
        return self


class QuantitySimulationInput(FrozenModel):
    initial: tuple[QuantityValue, ...] = Field(min_length=1)
    transfers: tuple[QuantityTransfer, ...]
    conservation_groups: tuple[ConservationGroup, ...] = ()
    schema_version: Ref = "yf_arc3_v5.quantity_simulation_input.v1"

    @model_validator(mode="after")
    def validate_simulation(self) -> "QuantitySimulationInput":
        entities = tuple(item.entity_ref for item in self.initial)
        if len(set(entities)) != len(entities):
            raise ValueError("initial quantity entities must be unique")
        if len({item.transition_ref for item in self.transfers}) != len(self.transfers):
            raise ValueError("quantity transition refs must be unique")
        entity_set = set(entities)
        endpoints = {
            ref
            for transfer in self.transfers
            for ref in (transfer.source_ref, transfer.target_ref)
        }
        if not endpoints <= entity_set:
            raise ValueError("quantity transfers must reference known entities")
        if len({group.group_ref for group in self.conservation_groups}) != len(
            self.conservation_groups
        ):
            raise ValueError("conservation group refs must be unique")
        if any(
            not set(group.member_refs) <= entity_set
            for group in self.conservation_groups
        ):
            raise ValueError("conservation groups must reference known entities")
        return self


class QuantityState(FrozenModel):
    after_transition_ref: Ref | None = None
    quantities: tuple[tuple[Ref, int], ...]


class ConstraintViolation(FrozenModel):
    transition_ref: Ref
    constraint_ref: Ref
    entity_refs: tuple[Ref, ...] = Field(min_length=1)


class QuantitySimulationResult(FrozenModel):
    states: tuple[QuantityState, ...] = Field(min_length=1)
    violations: tuple[ConstraintViolation, ...]
    all_valid: bool
    schema_version: Ref = "yf_arc3_v5.quantity_simulation_result.v1"

    @model_validator(mode="after")
    def validate_outcome(self) -> "QuantitySimulationResult":
        if self.all_valid == bool(self.violations):
            raise ValueError("all_valid must be the inverse of violations")
        return self


class DimensionValue(FrozenModel):
    entity_ref: Ref
    dimension_ref: Ref
    value_ref: Ref


class DimensionTransitionInput(FrozenModel):
    before: tuple[DimensionValue, ...] = Field(min_length=1)
    after: tuple[DimensionValue, ...] = Field(min_length=1)
    expected_changed_dimensions: tuple[Ref, ...] = ()
    expected_preserved_dimensions: tuple[Ref, ...] = ()
    schema_version: Ref = "yf_arc3_v5.dimension_transition_input.v1"

    @model_validator(mode="after")
    def validate_dimensions(self) -> "DimensionTransitionInput":
        before_keys = tuple(
            (item.entity_ref, item.dimension_ref) for item in self.before
        )
        after_keys = tuple((item.entity_ref, item.dimension_ref) for item in self.after)
        if len(set(before_keys)) != len(before_keys) or len(set(after_keys)) != len(
            after_keys
        ):
            raise ValueError("entity/dimension pairs must be unique")
        if set(before_keys) != set(after_keys):
            raise ValueError("before and after must describe identical dimensions")
        overlap = set(self.expected_changed_dimensions) & set(
            self.expected_preserved_dimensions
        )
        if overlap:
            raise ValueError("a dimension cannot be expected changed and preserved")
        return self


class DimensionTransitionResult(FrozenModel):
    changed_dimensions: tuple[Ref, ...]
    preserved_dimensions: tuple[Ref, ...]
    expectation_violations: tuple[Ref, ...]
    exact: bool
    schema_version: Ref = "yf_arc3_v5.dimension_transition_result.v1"


class DeclaredGoalGapRequest(FrozenModel):
    """One exact dimension requested by a DRM-declared open goal."""

    goal_ref: Ref
    goal_kind: GoalKind
    dimension: GoalGapDimension
    observed_value: GoalGapValue
    target_value: GoalGapValue
    schema_version: Ref = "yf_arc3_v5.declared_goal_gap_request.v1"


class DeclaredGoalGapMeasurement(FrozenModel):
    goal_ref: Ref
    goal_kind: GoalKind
    dimension: GoalGapDimension
    observed_value: GoalGapValue
    target_value: GoalGapValue
    satisfied: bool
    signed_integer_delta: int | None = None
    schema_version: Ref = "yf_arc3_v5.declared_goal_gap_measurement.v1"


class DeclaredGoalGapInput(FrozenModel):
    requests: tuple[DeclaredGoalGapRequest, ...] = Field(max_length=64)
    schema_version: Ref = "yf_arc3_v5.declared_goal_gap_input.v1"

    @model_validator(mode="after")
    def validate_unique_dimensions(self) -> "DeclaredGoalGapInput":
        keys = tuple((item.goal_ref, item.dimension) for item in self.requests)
        if len(set(keys)) != len(keys):
            raise ValueError("goal/dimension requests must be unique")
        return self


class DeclaredGoalGapResult(FrozenModel):
    measurements: tuple[DeclaredGoalGapMeasurement, ...]
    all_satisfied: bool
    schema_version: Ref = "yf_arc3_v5.declared_goal_gap_result.v1"


class DeclaredGoalPlanInput(FrozenModel):
    """Exact measured gaps plus one DRM-declared bounded plan structure."""

    gap_measurements: DeclaredGoalGapResult
    plan_request: GoalPlanRequest
    schema_version: Ref = "yf_arc3_v5.declared_goal_plan_input.v1"


class DeclaredGoalPlanAlternatives(FrozenModel):
    """Mechanical witnesses and facts exposed for DRM admissibility."""

    goal_ref: Ref
    witnesses: tuple[GoalPlanWitness, ...] = Field(max_length=3)
    alternative_refs: tuple[Ref, ...] = Field(max_length=3)
    alternative_facts: FrozenMap
    has_candidates: bool
    schema_version: Ref = "yf_arc3_v5.declared_goal_plan_alternatives.v1"

    @model_validator(mode="after")
    def validate_projection(self) -> "DeclaredGoalPlanAlternatives":
        witness_refs = tuple(item.plan_ref for item in self.witnesses)
        if witness_refs != self.alternative_refs:
            raise ValueError("goal-plan alternatives must preserve witness order")
        if set(self.alternative_facts) != set(self.alternative_refs):
            raise ValueError("goal-plan facts must index every witness exactly once")
        if self.has_candidates != bool(self.alternative_refs):
            raise ValueError("goal-plan candidate flag must match alternatives")
        return self


class RevisionedCapabilityRequest(FrozenModel):
    capability_id: Ref
    input_revision: NonNegativeRevision
    payload: FrozenModel
    schema_version: Ref = "yf_arc3_v5.revisioned_capability_request.v1"


class CapabilityExecution(FrozenModel):
    definition: CapabilityDefinition
    input_revision: NonNegativeRevision
    input_digest: Ref
    cache_key: Ref
    output: FrozenModel
    schema_version: Ref = "yf_arc3_v5.capability_execution.v1"


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _coerce_matrix(value: Any) -> tuple[tuple[int, ...], ...]:
    if not _is_sequence(value) or not value:
        raise ValueError("frame must be a non-empty rectangular matrix")
    width: int | None = None
    rows: list[tuple[int, ...]] = []
    for raw_row in value:
        if not _is_sequence(raw_row):
            raise ValueError("frame must be a non-empty rectangular matrix")
        row = tuple(int(item) for item in raw_row)
        if width is None:
            width = len(row)
        if not row or len(row) != width:
            raise ValueError("frame must be a non-empty rectangular matrix")
        rows.append(row)
    return tuple(rows)
