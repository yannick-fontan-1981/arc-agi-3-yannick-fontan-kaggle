"""Bounded joint simulation under explicitly supplied discrete laws.

No law, route, role or action is chosen here.  Every model remains visible in
the result; source declarations decide whether its premises permit commitment.
"""

from typing import Literal

from pydantic import Field, model_validator

from agents.yf_arc3_v5.capabilities.contracts import FrozenModel, Ref, FrozenMap


class TemporalRouteWitness(FrozenModel):
    alternative_ref: Ref
    action_refs: tuple[Ref, ...] = Field(min_length=1, max_length=64)


class TemporalRouteModel(FrozenModel):
    model_ref: Ref
    occupancy_response: Literal["while_occupied", "latched_on_entry"]
    advance_order: Literal["before_motion", "after_motion"]
    advance_when: Literal["every_attempt", "successful_motion"]
    endpoint_behavior: Literal["retain", "remove", "unknown"]
    initial_latched_value: bool

    @model_validator(mode="after")
    def validate_tick_order(self) -> "TemporalRouteModel":
        if self.advance_order == "before_motion" and self.advance_when == "successful_motion":
            raise ValueError("successful-motion clock requires an explicit after-motion order")
        return self


class TemporalJointRouteInput(FrozenModel):
    measurement_context_ref: Ref
    observation_revision: int = Field(ge=0)
    simulation_premise_refs: tuple[Ref, ...] = Field(min_length=1, max_length=32)
    simulation_premises_committed: bool
    node_refs: tuple[Ref, ...] = Field(min_length=1, max_length=256)
    # Each edge is (origin, source-declared action, destination).
    edges: tuple[tuple[Ref, Ref, Ref], ...] = Field(min_length=1, max_length=1024)
    initial_cell_ref: Ref
    required_terminal_cell_ref: Ref
    recorded_cell_refs: tuple[Ref, ...] = Field(min_length=1, max_length=65)
    replay_read_index: int = Field(ge=0, le=64)
    occupancy_cell_refs: tuple[Ref, ...] = Field(min_length=1, max_length=64)
    blocked_when_inactive: tuple[Ref, ...] = Field(default=(), max_length=256)
    blocked_when_active: tuple[Ref, ...] = Field(default=(), max_length=256)
    fixed_blocked_cell_refs: tuple[Ref, ...] = Field(default=(), max_length=256)
    models: tuple[TemporalRouteModel, ...] = Field(min_length=1, max_length=8)
    witnesses: tuple[TemporalRouteWitness, ...] = Field(min_length=1, max_length=3)
    expansion_bound: int = Field(default=1536, ge=1, le=1536)
    schema_version: Ref = "yf_arc3_v5.temporal_joint_route_input.v1"

    @model_validator(mode="after")
    def validate_graph(self) -> "TemporalJointRouteInput":
        nodes = set(self.node_refs)
        if len(nodes) != len(self.node_refs):
            raise ValueError("node refs must be unique")
        referenced = (self.initial_cell_ref, self.required_terminal_cell_ref,
                      *self.recorded_cell_refs, *self.occupancy_cell_refs,
                      *self.blocked_when_inactive, *self.blocked_when_active,
                      *self.fixed_blocked_cell_refs)
        if not set(referenced).issubset(nodes):
            raise ValueError("all cells must belong to the supplied graph")
        if self.replay_read_index >= len(self.recorded_cell_refs):
            raise ValueError("read index is outside the supplied recording")
        keys = [(origin, action) for origin, action, _ in self.edges]
        if len(keys) != len(set(keys)):
            raise ValueError("graph must have one destination per origin/action")
        if any(origin not in nodes or dest not in nodes for origin, _, dest in self.edges):
            raise ValueError("edge endpoint is outside graph")
        for refs in (tuple(m.model_ref for m in self.models),
                     tuple(w.alternative_ref for w in self.witnesses),
                     self.simulation_premise_refs):
            if len(refs) != len(set(refs)):
                raise ValueError("model, witness and premise refs must be unique")
        if sum(len(w.action_refs) for w in self.witnesses) * len(self.models) > self.expansion_bound:
            raise ValueError("joint simulation exceeds the supplied expansion bound")
        return self


class TemporalJointRouteMeasurements(FrozenModel):
    alternative_refs: tuple[Ref, ...]
    alternative_facts: FrozenMap
    expanded_transition_count: int = Field(ge=0, le=1536)
    schema_version: Ref = "yf_arc3_v5.temporal_joint_route_measurements.v1"


def measure_temporal_joint_routes(value: TemporalJointRouteInput) -> TemporalJointRouteMeasurements:
    """Evaluate at most three supplied routes under every supplied model."""
    edges = {(origin, action): dest for origin, action, dest in value.edges}
    action_refs = frozenset(action for _, action, _ in value.edges)
    occupancy = frozenset(value.occupancy_cell_refs)
    fixed = frozenset(value.fixed_blocked_cell_refs)
    inactive = frozenset(value.blocked_when_inactive) | fixed
    active = frozenset(value.blocked_when_active) | fixed
    expanded = 0
    facts = {}
    for witness in value.witnesses:
        rows = []
        for model in value.models:
            cell = value.initial_cell_ref
            index = value.replay_read_index
            replay_cell = value.recorded_cell_refs[index]
            latch = model.initial_latched_value
            previous_occupied = replay_cell in occupancy
            first_occupied = 0 if previous_occupied else None
            first_blocked = None
            blocked_count = 0
            unknown_endpoint = False
            support_indices = []
            changed_cell_crossings = []
            for step, action in enumerate(witness.action_refs, 1):
                expanded += 1
                if action not in action_refs:
                    blocked_count += 1
                    if first_blocked is None:
                        first_blocked = step
                    continue
                # Models state whether an attempt advances time; no invented WAIT.
                if model.advance_order == "before_motion":
                    index += 1
                    replay_cell, uncertain = _cell_at(value, model, index)
                    unknown_endpoint |= uncertain
                occupied = replay_cell in occupancy if replay_cell is not None else False
                if occupied and not previous_occupied:
                    latch = True
                previous_occupied = occupied
                output = occupied if model.occupancy_response == "while_occupied" else latch
                if occupied and first_occupied is None:
                    first_occupied = step
                if output:
                    support_indices.append(step)
                destination = edges.get((cell, action))
                moved = (destination is not None and destination != cell
                         and destination not in (active if output else inactive))
                if moved:
                    cell = destination
                    if destination in inactive - active:
                        changed_cell_crossings.append(step)
                else:
                    blocked_count += 1
                    if first_blocked is None:
                        first_blocked = step
                if model.advance_order == "after_motion" and (
                    model.advance_when == "every_attempt" or moved
                ):
                    index += 1
                    replay_cell, uncertain = _cell_at(value, model, index)
                    unknown_endpoint |= uncertain
                    occupied_after = replay_cell in occupancy if replay_cell is not None else False
                    if occupied_after and not previous_occupied:
                        latch = True
                    previous_occupied = occupied_after
                    if occupied_after and first_occupied is None:
                        first_occupied = step
            rows.append(FrozenMap({
                "model_ref": model.model_ref,
                "terminal_cell_ref": cell,
                "terminal_cell_matches": cell == value.required_terminal_cell_ref,
                "first_occupied_tick": first_occupied,
                "first_blocked_action_index": first_blocked,
                "blocked_action_count": blocked_count,
                "output_active_action_indices": tuple(support_indices),
                "changed_cell_crossing_action_indices": tuple(changed_cell_crossings),
                "replay_read_index": index,
                "endpoint_state_unknown": unknown_endpoint,
            }))
        facts[witness.alternative_ref] = FrozenMap({
            "alternative_ref": witness.alternative_ref,
            "measurement_context_ref": value.measurement_context_ref,
            "observation_revision": value.observation_revision,
            "simulation_premise_refs": value.simulation_premise_refs,
            "simulation_premises_committed": value.simulation_premises_committed,
            "route_action_refs": witness.action_refs,
            "initial_cell_ref": value.initial_cell_ref,
            "required_terminal_cell_ref": value.required_terminal_cell_ref,
            "primitive_action_count": len(witness.action_refs),
            "model_measurements": tuple(rows),
            "all_model_terminal_cells_match": all(row["terminal_cell_matches"] for row in rows),
            "all_model_routes_unblocked": all(row["blocked_action_count"] == 0 for row in rows),
            "all_model_endpoint_states_known": all(not row["endpoint_state_unknown"] for row in rows),
            "all_models_cross_changed_cells": all(bool(row["changed_cell_crossing_action_indices"]) for row in rows),
            "recorded_endpoint_inside_occupancy": value.recorded_cell_refs[-1] in occupancy,
            "all_models_retain_endpoint": all(m.endpoint_behavior == "retain" for m in value.models),
        })
    return TemporalJointRouteMeasurements(
        alternative_refs=tuple(w.alternative_ref for w in value.witnesses),
        alternative_facts=FrozenMap(facts),
        expanded_transition_count=expanded,
    )


def _cell_at(value: TemporalJointRouteInput, model: TemporalRouteModel, index: int) -> tuple[Ref | None, bool]:
    if index < len(value.recorded_cell_refs):
        return value.recorded_cell_refs[index], False
    if model.endpoint_behavior == "retain":
        return value.recorded_cell_refs[-1], False
    return None, model.endpoint_behavior == "unknown"
