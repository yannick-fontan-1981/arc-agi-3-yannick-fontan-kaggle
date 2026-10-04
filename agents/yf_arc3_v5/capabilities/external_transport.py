"""Exact tracked pair motion and separation, without assigning social roles."""
from agents.yf_arc3_v5.capabilities.contracts import (
    ExternalTransportObservationInput, ExternalTransportObservations,
)
from agents.yf_arc3_v5.logos.types import FrozenMap


def measure_external_transport(value: ExternalTransportObservationInput) -> ExternalTransportObservations:
    geometry = value.geometry or FrozenMap()
    if value.evidence_scope_ref is None:
        return ExternalTransportObservations()
    quantum = geometry.get("quantum")
    core_value = geometry.get("member_core_value")
    core_height = geometry.get("member_core_height")
    core_width = geometry.get("member_core_width")
    capacity = geometry.get("capacity_bbox")
    if not (isinstance(quantum, int) and quantum > 1 and isinstance(core_value, int)
            and isinstance(core_height, int) and isinstance(core_width, int)
            and capacity is not None and len(capacity) == 4):
        return ExternalTransportObservations()
    before = {entity.entity_id: entity for entity in value.before_tracking.entities}
    after = {entity.entity_id: entity for entity in value.after_tracking.entities}
    # Match only the morphology supplied by the source-recorded oriented
    # descriptor. No palette-to-role mapping or choice among equal matches.
    body_value = geometry.get("actor_body_value")
    def bodies(entities):
        return tuple(entity for entity in sorted(entities.values(), key=lambda e: e.entity_id)
            if isinstance(body_value, int) and entity.component.value == body_value
            and entity.component.area == quantum * (quantum - 1)
            and sorted((entity.component.bbox.bottom - entity.component.bbox.top + 1,
                        entity.component.bbox.right - entity.component.bbox.left + 1))
                == [quantum - 1, quantum])
    old_bodies, new_bodies = bodies(before), bodies(after)
    declared_body_stationary = False
    if len(old_bodies) == len(new_bodies) == 1:
        old_body, new_body = old_bodies[0], new_bodies[0]
        declared_body_stationary = (
            old_body.entity_id == new_body.entity_id
            and new_body.identity_status == "established"
            and not new_body.possible_predecessor_entity_refs
            and new_body.component.bbox == old_body.component.bbox
            and new_body.component.relative_pixels == old_body.component.relative_pixels
            and (new_body.delta_row, new_body.delta_col) == (0, 0)
        )
    def displacement(entity):
        previous = before.get(entity.entity_id)
        if previous is None:
            return (0, 0)
        return (entity.component.bbox.top - previous.component.bbox.top,
                entity.component.bbox.left - previous.component.bbox.left)
    actuators = frozenset(value.known_actuator_entity_refs)
    cores = [entity for entity in value.after_tracking.entities
             if entity.component.value == core_value
             and entity.component.bbox.bottom - entity.component.bbox.top + 1 == core_height
             and entity.component.bbox.right - entity.component.bbox.left + 1 == core_width]
    squares = [entity for entity in value.after_tracking.entities
               if entity.entity_id not in actuators and entity.component.area == quantum * quantum
               and entity.component.bbox.bottom - entity.component.bbox.top + 1 == quantum
               and entity.component.bbox.right - entity.component.bbox.left + 1 == quantum]
    prior = {(record.get("carrier_entity_ref"), record.get("cargo_entity_ref")): record
             for record in (value.prior_records or ())
             if record.get("evidence_scope_ref") == value.evidence_scope_ref}
    pairs = set(prior)
    for square in squares:
        for core in cores:
            sb, cb = square.component.bbox, core.component.bbox
            # One cell of separation between centres; excludes two touching
            # fragments of the same bicolour sprite.
            centres = (sb.top + sb.bottom, sb.left + sb.right,
                       cb.top + cb.bottom, cb.left + cb.right)
            adjacent = ((centres[0] == centres[2] and abs(centres[1] - centres[3]) == 2 * quantum)
                        or (centres[1] == centres[3] and abs(centres[0] - centres[2]) == 2 * quantum))
            if adjacent and square.entity_id in before and core.entity_id in before:
                sd = displacement(square)
                cd = displacement(core)
                if sd == cd and sd != (0, 0):
                    pairs.add((square.entity_id, core.entity_id))
    if len(pairs) > value.maximum_pairs:
        return ExternalTransportObservations(pair_bound_reached=True)
    rows = []
    for carrier_ref, cargo_ref in sorted(pairs):
        old = prior.get((carrier_ref, cargo_ref), FrozenMap())
        carrier, cargo = after.get(carrier_ref), after.get(cargo_ref)
        exact = (carrier is not None and cargo is not None
                 and carrier_ref in before and cargo_ref in before
                 and carrier.identity_status == "established" and cargo.identity_status == "established"
                 and not carrier.possible_predecessor_entity_refs and not cargo.possible_predecessor_entity_refs)
        carrier_delta = displacement(carrier) if carrier else (0, 0)
        cargo_delta = displacement(cargo) if cargo else (0, 0)
        if exact:
            exact = all(
                current.component.value == before[current.entity_id].component.value
                and current.component.relative_pixels == before[current.entity_id].component.relative_pixels
                and displacement(current) == (current.delta_row, current.delta_col)
                for current in (carrier, cargo)
            )
        cotranslated = exact and carrier_delta == cargo_delta and cargo_delta != (0, 0)
        inside = False
        if cargo:
            box = cargo.component.bbox
            inside = capacity[0] <= box.top <= box.bottom <= capacity[2] and capacity[1] <= box.left <= box.right <= capacity[3]
        separated = exact and carrier_delta != (0, 0) and cargo_delta == (0, 0)
        adjacent_now = False
        if carrier is not None and cargo is not None:
            sb, cb = carrier.component.bbox, cargo.component.bbox
            row_gap = (sb.top + sb.bottom) - (cb.top + cb.bottom)
            col_gap = (sb.left + sb.right) - (cb.left + cb.right)
            adjacent_now = (row_gap == 0 and abs(col_gap) == 2 * quantum) or (col_gap == 0 and abs(row_gap) == 2 * quantum)
        prior_current_count = (
            int(old.get("pair_current_motion_count", 0))
            if old.get("observation_frame_ref") == value.before_tracking.frame_ref else 0
        )
        current_count = prior_current_count + int(cotranslated) if exact and adjacent_now else 0
        controlled_stationary = bool(value.known_actuator_entity_refs) and all(
            ref in before and ref in after
            and before[ref].component.bbox == after[ref].component.bbox
            and before[ref].component.relative_pixels == after[ref].component.relative_pixels
            and before[ref].component.value == after[ref].component.value
            and after[ref].identity_status == "established"
            and not after[ref].possible_predecessor_entity_refs
            for ref in value.known_actuator_entity_refs
        )
        boxes_before = tuple(sorted((b.top,b.left,b.bottom,b.right)
            for b in value.pre_action_input_aligned_entity_bboxes))
        boxes_after = tuple(sorted((b.top,b.left,b.bottom,b.right)
            for b in value.post_action_input_aligned_entity_bboxes))
        rows.append(FrozenMap({
            "carrier_entity_ref": carrier_ref, "cargo_entity_ref": cargo_ref,
            "transition_ref": value.transition_ref, "evidence_scope_ref": value.evidence_scope_ref,
            "pair_identity_exact": exact, "pair_cotranslated": bool(cotranslated),
            "action_ref": value.action_ref,
            "controlled_entities_stationary": controlled_stationary,
            "input_aligned_boxes_stationary": bool(boxes_before) and boxes_before == boxes_after,
            "declared_oriented_body_stationary": declared_body_stationary,
            "pair_motion_count": int(old.get("pair_motion_count", 0)) + int(cotranslated),
            "pair_current_motion_count": current_count,
            "cargo_inside_capacity": inside, "carrier_departed_cargo_stationary": bool(separated),
            "pair_adjacent_now": adjacent_now,
            "observation_frame_ref": value.after_tracking.frame_ref,
            "cargo_core_bbox": ((cargo.component.bbox.top, cargo.component.bbox.left,
                                 cargo.component.bbox.bottom, cargo.component.bbox.right) if cargo else None),
            "prior_deposit_observed": bool(old.get("deposit_observed", False)),
            "cargo_missing_or_outside": cargo is None or not inside,
            "carrier_delta_row": carrier_delta[0], "carrier_delta_col": carrier_delta[1],
            "cargo_delta_row": cargo_delta[0], "cargo_delta_col": cargo_delta[1],
        }))
    return ExternalTransportObservations(rows=tuple(rows))
