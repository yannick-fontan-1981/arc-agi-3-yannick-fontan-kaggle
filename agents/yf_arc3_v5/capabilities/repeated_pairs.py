"""Neutral measurements for repeated orthogonal pair state/effect relations."""

from __future__ import annotations

from dataclasses import dataclass

from agents.yf_arc3_v5.capabilities.components import detect_components
from agents.yf_arc3_v5.capabilities.contracts import (
    ComponentDescription,
    ComponentExtractionInput,
    FrameGrid,
    InteractionProbeAgenda,
    InteractionProbeCandidate,
)
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest


POINT_ACTION_REF = "ACTION6"
MAX_PAIR_GROUPS = 16
MAX_EFFECT_RELATIONS = 3


@dataclass(frozen=True, slots=True)
class OrthogonalPairGroup:
    """Two equal-area line components sharing one column anchor."""

    ordinal: int
    horizontal_bbox: tuple[int, int, int, int]
    vertical_bbox: tuple[int, int, int, int]
    horizontal_value: int
    vertical_value: int


@dataclass(frozen=True, slots=True)
class RepeatedPairStateEffectRelation:
    """Exact arithmetic relation; DRM alone assigns its meaning."""

    relation_ref: str
    pair_groups: tuple[OrthogonalPairGroup, ...]
    measured_pair_value: int
    translated_component_value: int
    translated_component_bbox: tuple[int, int, int, int]
    translated_component_relative_pixels: tuple[tuple[int, int], ...]
    translated_component_area: int
    stationary_larger_component_bbox: tuple[int, int, int, int]
    measured_delta_row: int
    measured_delta_col: int
    measured_quantum_row: int
    measured_quantum_col: int
    full_group_residual_row: int
    full_group_residual_col: int
    observed_effect_support_count: int
    observed_final_frame_equals_before: bool
    point_x: int
    point_y: int


def measure_repeated_pair_state_effect_relations(
    *,
    before: FrameGrid,
    ordered_frames: tuple[FrameGrid, ...],
    point_x: int,
    point_y: int,
) -> tuple[RepeatedPairStateEffectRelation, ...]:
    """Enumerate exact repeated-pair/state/translation correspondences."""

    groups = _repeated_orthogonal_pair_groups(before)
    if len(groups) < 3 or len(groups) > MAX_PAIR_GROUPS or not ordered_frames:
        return ()
    uniform_counts: dict[int, int] = {}
    for group in groups:
        if group.horizontal_value == group.vertical_value:
            uniform_counts[group.horizontal_value] = (
                uniform_counts.get(group.horizontal_value, 0) + 1
            )
    if len(uniform_counts) < 2:
        return ()

    before_components = detect_components(
        ComponentExtractionInput(frame=before, connectivity=4)
    ).components
    signatures: dict[tuple[int, tuple[tuple[int, int], ...], int], list[ComponentDescription]] = {}
    for component in before_components:
        signature = (component.value, component.relative_pixels, component.area)
        signatures.setdefault(signature, []).append(component)

    frame_components = tuple(
        detect_components(ComponentExtractionInput(frame=frame, connectivity=4)).components
        for frame in ordered_frames
    )
    relations: list[RepeatedPairStateEffectRelation] = []
    for signature, base_matches in sorted(signatures.items()):
        if len(base_matches) != 1:
            continue
        moving = base_matches[0]
        if moving.touches_frame_boundary:
            continue
        trajectory: list[tuple[int, int]] = []
        for components in frame_components:
            matches = [
                component
                for component in components
                if (component.value, component.relative_pixels, component.area)
                == signature
            ]
            if len(matches) != 1:
                continue
            trajectory.append(
                (
                    matches[0].bbox.top - moving.bbox.top,
                    matches[0].bbox.left - moving.bbox.left,
                )
            )
        nonzero_deltas = tuple(delta for delta in trajectory if delta != (0, 0))
        if not nonzero_deltas:
            continue
        larger_peers = tuple(
            component
            for component in before_components
            if component.component_id != moving.component_id
            and component.value == moving.value
            and component.area > moving.area
            and component.bbox.height >= moving.bbox.height
            and component.bbox.width >= moving.bbox.width
            and not component.touches_frame_boundary
        )
        for peer in larger_peers:
            residual_row = peer.bbox.top - moving.bbox.top
            residual_col_twice = (
                peer.bbox.left
                + peer.bbox.right
                - moving.bbox.left
                - moving.bbox.right
            )
            if residual_col_twice % 2:
                continue
            residual_col = residual_col_twice // 2
            for pair_value, support_count in sorted(uniform_counts.items()):
                for delta_row, delta_col in nonzero_deltas:
                    if delta_row % support_count or delta_col % support_count:
                        continue
                    quantum_row = delta_row // support_count
                    quantum_col = delta_col // support_count
                    if quantum_row == 0 and quantum_col == 0:
                        continue
                    if (
                        quantum_row * len(groups) != residual_row
                        or quantum_col * len(groups) != residual_col
                    ):
                        continue
                    relation_ref = (
                        "measurement.repeated_pair_state_effect."
                        + stable_digest(
                            (
                                tuple(
                                    (
                                        group.horizontal_bbox,
                                        group.vertical_bbox,
                                        group.horizontal_value,
                                        group.vertical_value,
                                    )
                                    for group in groups
                                ),
                                pair_value,
                                moving.component_id,
                                peer.component_id,
                                delta_row,
                                delta_col,
                            )
                        )[:16]
                    )
                    relations.append(
                        RepeatedPairStateEffectRelation(
                            relation_ref=relation_ref,
                            pair_groups=groups,
                            measured_pair_value=pair_value,
                            translated_component_value=moving.value,
                            translated_component_bbox=_bbox_tuple(moving),
                            translated_component_relative_pixels=(
                                moving.relative_pixels
                            ),
                            translated_component_area=moving.area,
                            stationary_larger_component_bbox=_bbox_tuple(peer),
                            measured_delta_row=delta_row,
                            measured_delta_col=delta_col,
                            measured_quantum_row=quantum_row,
                            measured_quantum_col=quantum_col,
                            full_group_residual_row=residual_row,
                            full_group_residual_col=residual_col,
                            observed_effect_support_count=support_count,
                            observed_final_frame_equals_before=(
                                ordered_frames[-1] == before
                            ),
                            point_x=point_x,
                            point_y=point_y,
                        )
                    )
    unique = {relation.relation_ref: relation for relation in relations}
    if not unique or len(unique) > MAX_EFFECT_RELATIONS:
        return ()
    return tuple(unique[key] for key in sorted(unique))


def enumerate_repeated_pair_state_actions(
    *,
    frame: FrameGrid,
    relations: tuple[RepeatedPairStateEffectRelation, ...],
    available_action_refs: tuple[str, ...],
) -> InteractionProbeAgenda:
    """Enumerate edits and source revisits without assigning program meaning."""

    if POINT_ACTION_REF not in available_action_refs or not relations:
        return _empty_agenda()
    current_groups = _repeated_orthogonal_pair_groups(frame)
    current_components = detect_components(
        ComponentExtractionInput(frame=frame, connectivity=4)
    ).components
    candidates: list[InteractionProbeCandidate] = []
    facts_by_ref: dict[str, FrozenMap] = {}
    for relation in relations:
        if len(current_groups) != len(relation.pair_groups):
            continue
        successor_offsets = tuple(
            (
                relation.measured_quantum_row * (slot_index + 1),
                relation.measured_quantum_col * (slot_index + 1),
            )
            for slot_index in range(len(current_groups))
        )
        exact_terminal_position = bool(
            successor_offsets
            and successor_offsets[-1]
            == (
                relation.full_group_residual_row,
                relation.full_group_residual_col,
            )
        )
        observed_effect_closes_terminal_position = bool(
            relation.measured_delta_row == relation.full_group_residual_row
            and relation.measured_delta_col == relation.full_group_residual_col
        )
        observed_effect_support_is_incomplete = bool(
            relation.observed_effect_support_count < len(current_groups)
        )
        current_translated_matches = tuple(
            component
            for component in current_components
            if component.value == relation.translated_component_value
            and component.relative_pixels
            == relation.translated_component_relative_pixels
            and component.area == relation.translated_component_area
        )
        current_translated_offset = (
            (
                current_translated_matches[0].bbox.top
                - relation.translated_component_bbox[0],
                current_translated_matches[0].bbox.left
                - relation.translated_component_bbox[1],
            )
            if len(current_translated_matches) == 1
            else None
        )
        current_actor_matches_prelaunch_position = current_translated_offset == (0, 0)
        current_actor_matches_observed_effect_endpoint = (
            current_translated_offset
            == (relation.measured_delta_row, relation.measured_delta_col)
        )
        checkpoint_actor_state_scope_measured = bool(
            current_actor_matches_prelaunch_position
            or current_actor_matches_observed_effect_endpoint
        )
        mismatches: list[tuple[OrthogonalPairGroup, int, tuple[int, int, int, int]]] = []
        for group in current_groups:
            if group.horizontal_value != relation.measured_pair_value:
                mismatches.append((group, 0, group.horizontal_bbox))
            if group.vertical_value != relation.measured_pair_value:
                mismatches.append((group, 1, group.vertical_bbox))
        if mismatches:
            for group, member_order, bbox in mismatches:
                row = (bbox[0] + bbox[2]) // 2
                col = (bbox[1] + bbox[3]) // 2
                candidate_ref = (
                    f"probe:repeated_pair_member:{relation.relation_ref}:"
                    f"g{group.ordinal}:m{member_order}:r{row}:c{col}"
                )
                candidate = InteractionProbeCandidate(
                    candidate_ref=candidate_ref,
                    action_ref=POINT_ACTION_REF,
                    action_data=FrozenMap({"x": col, "y": row}),
                    point=(row, col),
                    basis_refs=(relation.relation_ref,),
                )
                candidates.append(candidate)
                facts_by_ref[candidate_ref] = FrozenMap(
                    {
                        "alternative_ref": candidate_ref,
                        "candidate_ref": candidate_ref,
                        "interaction_measurement_kind": "point",
                        "action_ref": POINT_ACTION_REF,
                        "current_action_available": True,
                        "repeated_orthogonal_pair_group_count": len(current_groups),
                        "pair_state_effect_support_count": sum(
                            group.horizontal_value == relation.measured_pair_value
                            and group.vertical_value == relation.measured_pair_value
                            for group in current_groups
                        ),
                        "measured_pair_value": relation.measured_pair_value,
                        "measured_quantum_row": relation.measured_quantum_row,
                        "measured_quantum_col": relation.measured_quantum_col,
                        "full_group_residual_row": relation.full_group_residual_row,
                        "full_group_residual_col": relation.full_group_residual_col,
                        "all_repeated_pair_groups_measured": True,
                        "full_group_quantum_closes_exact_residual": bool(
                            relation.measured_quantum_row * len(current_groups)
                            == relation.full_group_residual_row
                            and relation.measured_quantum_col * len(current_groups)
                            == relation.full_group_residual_col
                        ),
                        "remaining_nonmatching_pair_member_count": len(mismatches),
                        "remaining_pair_member_edit_action_count": len(mismatches),
                        "required_effect_source_action_count": 1,
                        "bounded_stage_action_count": len(mismatches) + 1,
                        "bounded_stage_action_cost_measured": True,
                        "next_complete_pair_configuration_grounded": True,
                        "pair_group_order": group.ordinal,
                        "pair_member_order": member_order,
                        "repeated_pair_member_measurement": True,
                        "effect_source_revisit_measurement": False,
                        "observed_effect_support_count": (
                            relation.observed_effect_support_count
                        ),
                        "observed_effect_support_is_incomplete": (
                            observed_effect_support_is_incomplete
                        ),
                        "observed_effect_closes_terminal_position": (
                            observed_effect_closes_terminal_position
                        ),
                        "observed_final_frame_equals_before": (
                            relation.observed_final_frame_equals_before
                        ),
                        "current_translated_component_offset_measured": (
                            current_translated_offset is not None
                        ),
                        "current_translated_component_offset": (
                            current_translated_offset
                        ),
                        "current_actor_matches_prelaunch_position": (
                            current_actor_matches_prelaunch_position
                        ),
                        "current_actor_matches_observed_effect_endpoint": (
                            current_actor_matches_observed_effect_endpoint
                        ),
                        "checkpoint_actor_state_scope_measured": (
                            checkpoint_actor_state_scope_measured
                        ),
                        "instruction_successor_translation_measured": True,
                        "program_successor_slot_count": len(current_groups),
                        "program_successor_offsets": successor_offsets,
                        "program_successor_sequence_measured": True,
                        "program_successor_all_slots_supported": False,
                        "program_successor_has_unsupported_slot": True,
                        "program_successor_final_position_match": (
                            exact_terminal_position
                        ),
                        "typed_position_acceptance_measured": (
                            exact_terminal_position
                        ),
                        "typed_nonposition_property_measurement_count": 0,
                    }
                )
        else:
            candidate_ref = (
                f"probe:repeated_pair_effect_source:{relation.relation_ref}:"
                f"r{relation.point_y}:c{relation.point_x}"
            )
            candidate = InteractionProbeCandidate(
                candidate_ref=candidate_ref,
                action_ref=POINT_ACTION_REF,
                action_data=FrozenMap({"x": relation.point_x, "y": relation.point_y}),
                point=(relation.point_y, relation.point_x),
                basis_refs=(relation.relation_ref,),
            )
            candidates.append(candidate)
            facts_by_ref[candidate_ref] = FrozenMap(
                {
                    "alternative_ref": candidate_ref,
                    "candidate_ref": candidate_ref,
                    "interaction_measurement_kind": "point",
                    "action_ref": POINT_ACTION_REF,
                    "current_action_available": True,
                    "repeated_orthogonal_pair_group_count": len(current_groups),
                    "pair_state_effect_support_count": len(current_groups),
                    "measured_pair_value": relation.measured_pair_value,
                    "measured_quantum_row": relation.measured_quantum_row,
                    "measured_quantum_col": relation.measured_quantum_col,
                    "full_group_residual_row": relation.full_group_residual_row,
                    "full_group_residual_col": relation.full_group_residual_col,
                    "all_repeated_pair_groups_measured": True,
                    "full_group_quantum_closes_exact_residual": bool(
                        relation.measured_quantum_row * len(current_groups)
                        == relation.full_group_residual_row
                        and relation.measured_quantum_col * len(current_groups)
                        == relation.full_group_residual_col
                    ),
                    "remaining_nonmatching_pair_member_count": 0,
                    "remaining_pair_member_edit_action_count": 0,
                    "required_effect_source_action_count": 1,
                    "bounded_stage_action_count": 1,
                    "bounded_stage_action_cost_measured": True,
                    "next_complete_pair_configuration_grounded": True,
                    "pair_group_order": len(current_groups),
                    "pair_member_order": 2,
                    "repeated_pair_member_measurement": False,
                    "effect_source_revisit_measurement": True,
                    "observed_effect_support_count": (
                        relation.observed_effect_support_count
                    ),
                    "observed_effect_support_is_incomplete": (
                        observed_effect_support_is_incomplete
                    ),
                    "observed_effect_closes_terminal_position": (
                        observed_effect_closes_terminal_position
                    ),
                    "observed_final_frame_equals_before": (
                        relation.observed_final_frame_equals_before
                    ),
                    "current_translated_component_offset_measured": (
                        current_translated_offset is not None
                    ),
                    "current_translated_component_offset": current_translated_offset,
                    "current_actor_matches_prelaunch_position": (
                        current_actor_matches_prelaunch_position
                    ),
                    "current_actor_matches_observed_effect_endpoint": (
                        current_actor_matches_observed_effect_endpoint
                    ),
                    "checkpoint_actor_state_scope_measured": (
                        checkpoint_actor_state_scope_measured
                    ),
                    "instruction_successor_translation_measured": True,
                    "program_successor_slot_count": len(current_groups),
                    "program_successor_offsets": successor_offsets,
                    "program_successor_sequence_measured": True,
                    "program_successor_all_slots_supported": True,
                    "program_successor_has_unsupported_slot": False,
                    "program_successor_final_position_match": exact_terminal_position,
                    "typed_position_acceptance_measured": exact_terminal_position,
                    "typed_nonposition_property_measurement_count": 0,
                }
            )
    candidates.sort(key=lambda item: item.candidate_ref)
    refs = tuple(item.candidate_ref for item in candidates)
    return InteractionProbeAgenda(
        candidates=tuple(candidates),
        alternative_refs=refs,
        alternative_facts=FrozenMap.from_frozen_items(
            {ref: facts_by_ref[ref] for ref in refs}
        ),
        context_facts=FrozenMap(
            {
                "repeated_pair_state_effect_relation_count": len(relations),
                "repeated_pair_state_action_candidate_count": len(candidates),
            }
        ),
        has_candidates=bool(candidates),
    )


def _repeated_orthogonal_pair_groups(frame: FrameGrid) -> tuple[OrthogonalPairGroup, ...]:
    components = detect_components(
        ComponentExtractionInput(frame=frame, connectivity=4)
    ).components
    horizontal = tuple(
        component
        for component in components
        if 2 <= component.area <= 8
        and component.bbox.height == 1
        and component.bbox.width == component.area
        and not component.touches_frame_boundary
    )
    vertical = tuple(
        component
        for component in components
        if 2 <= component.area <= 8
        and component.bbox.width == 1
        and component.bbox.height == component.area
        and not component.touches_frame_boundary
    )
    families: dict[tuple[int, int, int], list[tuple[ComponentDescription, ComponentDescription]]] = {}
    for first in horizontal:
        first_center_col_twice = first.bbox.left + first.bbox.right
        for second in vertical:
            if first.area != second.area:
                continue
            if 2 * second.bbox.left != first_center_col_twice:
                continue
            row_gap = second.bbox.top - first.bbox.bottom
            if row_gap <= 0 or row_gap > first.area:
                continue
            key = (first.bbox.top, second.bbox.top, first.area)
            families.setdefault(key, []).append((first, second))
    viable: list[tuple[tuple[int, int, int], list[tuple[ComponentDescription, ComponentDescription]]]] = []
    for key, pairs in sorted(families.items()):
        ordered = sorted(pairs, key=lambda pair: pair[0].bbox.left)
        if len(ordered) < 3 or len(ordered) > MAX_PAIR_GROUPS:
            continue
        centers = tuple(
            (component.bbox.left + component.bbox.right) // 2
            for component, _ in ordered
        )
        spacings = tuple(right - left for left, right in zip(centers, centers[1:]))
        if not spacings or len(set(spacings)) != 1:
            continue
        viable.append((key, ordered))
    if len(viable) != 1:
        return ()
    _key, pairs = viable[0]
    return tuple(
        OrthogonalPairGroup(
            ordinal=index,
            horizontal_bbox=_bbox_tuple(first),
            vertical_bbox=_bbox_tuple(second),
            horizontal_value=first.value,
            vertical_value=second.value,
        )
        for index, (first, second) in enumerate(pairs, start=1)
    )


def _bbox_tuple(component: ComponentDescription) -> tuple[int, int, int, int]:
    bbox = component.bbox
    return (bbox.top, bbox.left, bbox.bottom, bbox.right)


def _empty_agenda() -> InteractionProbeAgenda:
    return InteractionProbeAgenda(
        candidates=(),
        alternative_refs=(),
        alternative_facts=FrozenMap(),
        context_facts=FrozenMap(
            {
                "repeated_pair_state_effect_relation_count": 0,
                "repeated_pair_state_action_candidate_count": 0,
            }
        ),
        has_candidates=False,
    )
