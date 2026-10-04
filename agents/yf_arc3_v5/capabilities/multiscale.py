"""Exact multiscale homology and predicted-exception repair evidence.

The capability enumerates geometry and palette correspondences.  It never
chooses a mapping, assigns universal color meaning, or reads game/oracle data.
Selection and semantic commitment remain declared in SRC/DRM.
"""

from __future__ import annotations

from collections import ChainMap
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations

from agents.yf_arc3_v5.capabilities.contracts import (
    BoundingBox,
    ComponentDescription,
    FrameGrid,
    HomologousRepairAgendaInput,
    HomologousRepairAnalysis,
    HomologousRepairInput,
    HomologousRepairReconciliationAnalysis,
    HomologousRepairReconciliationInput,
    InteractionProbeAgenda,
    InteractionProbeCandidate,
    LocalConstraintCompositionAnalysis,
    LocalConstraintCompositionInput,
    LocalConstraintRepairAgendaInput,
)
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest

POINT_ACTION_REF = "ACTION6"
MAX_COMPOSITION_WITNESSES = 3


@dataclass(frozen=True)
class _Panel:
    top: int
    left: int
    order: int
    macro_size: int
    gap: int
    outer_values: tuple[int, ...]
    inner_values: tuple[int, ...]
    outer_components: tuple[str, ...]

    @property
    def extent(self) -> int:
        return self.order * self.macro_size + (self.order - 1) * self.gap

    @property
    def bbox(self) -> BoundingBox:
        return BoundingBox(
            top=self.top,
            left=self.left,
            bottom=self.top + self.extent - 1,
            right=self.left + self.extent - 1,
        )


@dataclass(frozen=True)
class _EmbeddedConstraint:
    row_index: int
    col_index: int
    order: int
    values: tuple[int, ...]
    component_ref: str

    @property
    def center_value(self) -> int:
        center = self.order // 2
        return self.values[center * self.order + center]

    @property
    def perimeter_values(self) -> tuple[int, ...]:
        center = self.order // 2
        return tuple(
            self.values[row * self.order + col]
            for row in range(self.order)
            for col in range(self.order)
            if (row, col) != (center, center)
        )


@dataclass(frozen=True)
class _LocalConstraintLattice:
    top: int
    left: int
    row_count: int
    col_count: int
    cell_size: int
    gap: int
    ordinary_values: tuple[tuple[int, ...], ...]
    ordinary_components: tuple[tuple[str, ...], ...]
    constraints: tuple[_EmbeddedConstraint, ...]

    @property
    def stride(self) -> int:
        return self.cell_size + self.gap


def analyze_local_constraint_compositions(
    value: LocalConstraintCompositionInput,
) -> LocalConstraintCompositionAnalysis:
    """Enumerate at most three exact global compositions of local constraints."""

    scene = value.scene
    component_by_bbox = {
        (
            item.bbox.top,
            item.bbox.left,
            item.bbox.bottom,
            item.bbox.right,
        ): item.component_id
        for item in scene.blocks.components
    }
    transferred = tuple(sorted(value.transferred_mapping_pairs))
    described: list[tuple[str, FrozenMap]] = []
    boundary_refs = tuple(
        sorted(
            item.component_id
            for item in scene.blocks.components
            if item.touches_frame_boundary
            and min(item.bbox.height, item.bbox.width) == 1
            and max(item.bbox.height, item.bbox.width)
            * 2
            >= (scene.frame.width if item.bbox.height == 1 else scene.frame.height)
        )
    )
    for lattice in _local_constraint_lattices(scene, component_by_bbox):
        inner_domains = {
            tuple(sorted(set(item.perimeter_values)))
            for item in lattice.constraints
        }
        observed_lattice_values = {
            cell
            for row in lattice.ordinary_values
            for cell in row
            if cell >= 0
        }
        same_scale_display_values = {
            item.value
            for item in scene.blocks.components
            if item.bbox.height == lattice.cell_size
            and item.bbox.width == lattice.cell_size
            and item.area == lattice.cell_size * lattice.cell_size
        }
        transferred_inner_domain = tuple(
            sorted({inner_value for inner_value, _outer_value in transferred})
        )
        transferred_outer_domain = {
            outer_value for _inner_value, outer_value in transferred
        }
        eligible_transferred_outer_domain = (
            transferred_outer_domain
            if transferred_inner_domain in inner_domains
            and len(transferred_outer_domain) == len(transferred_inner_domain)
            else set()
        )
        constraint_center_values = {
            item.center_value for item in lattice.constraints
        }
        scene_local_outer_domain = (
            observed_lattice_values
            | same_scale_display_values
            | constraint_center_values
        )
        supplemented_outer_domain = (
            scene_local_outer_domain
            if len(scene_local_outer_domain) >= 2
            else scene_local_outer_domain | eligible_transferred_outer_domain
        )
        outer_domain = tuple(
            sorted(supplemented_outer_domain)
        )
        if len(inner_domains) != 1 or len(outer_domain) != 2:
            continue
        inner_domain = next(iter(inner_domains))
        if len(inner_domain) != 2:
            continue
        used_component_refs = {
            component_ref
            for row in lattice.ordinary_components
            for component_ref in row
            if component_ref
        }
        value_domain_refs = tuple(
            sorted(
                item.component_id
                for item in scene.blocks.components
                if item.value in outer_domain
                and item.component_id not in used_component_refs
                and item.bbox.height == lattice.cell_size
                and item.bbox.width == lattice.cell_size
                and item.area == lattice.cell_size * lattice.cell_size
            )
        )
        for mapped_outer in permutations(outer_domain):
            mapping = dict(zip(inner_domain, mapped_outer, strict=True))
            mapping_pairs = tuple(sorted(mapping.items()))
            transferred_pairs_in_current_domain = tuple(
                (inner_value, outer_value)
                for inner_value, outer_value in transferred
                if inner_value in mapping and outer_value in outer_domain
            )
            transferred_mapping_agreement_count = sum(
                mapping[inner_value] == outer_value
                for inner_value, outer_value in transferred_pairs_in_current_domain
            )
            transferred_mapping_conflict_count = sum(
                mapping[inner_value] != outer_value
                for inner_value, outer_value in transferred_pairs_in_current_domain
            )
            predicted: dict[tuple[int, int], list[tuple[str, int]]] = {}
            constraint_refs: list[str] = []
            for constraint in lattice.constraints:
                constraint_ref = (
                    "constraint.local_projection."
                    f"r{constraint.row_index:02d}c{constraint.col_index:02d}"
                )
                constraint_refs.append(constraint_ref)
                center = constraint.order // 2
                for local_row in range(constraint.order):
                    for local_col in range(constraint.order):
                        if (local_row, local_col) == (center, center):
                            continue
                        global_row = constraint.row_index + local_row - center
                        global_col = constraint.col_index + local_col - center
                        inner_value = constraint.values[
                            local_row * constraint.order + local_col
                        ]
                        predicted.setdefault((global_row, global_col), []).append(
                            (constraint_ref, mapping[inner_value])
                        )
            overlap_slots = tuple(
                sorted(slot for slot, predictions in sorted(predicted.items()) if len(predictions) > 1)
            )
            conflict_facts = tuple(
                FrozenMap(
                    {
                        "global_slot": slot,
                        "prediction_facts": tuple(predicted[slot]),
                        "distinct_predicted_values": tuple(
                            sorted({prediction[1] for prediction in predicted[slot]})
                        ),
                    }
                )
                for slot in overlap_slots
                if len({prediction[1] for prediction in predicted[slot]}) > 1
            )
            compatible_overlap_count = sum(
                len({prediction[1] for prediction in predicted[slot]}) == 1
                for slot in overlap_slots
            )
            expected_by_slot = {
                slot: predictions[0][1]
                for slot, predictions in sorted(predicted.items())
                if len({prediction[1] for prediction in predictions}) == 1
            }
            mismatch_facts: list[FrozenMap] = []
            for (row_index, col_index), expected_value in sorted(
                expected_by_slot.items()
            ):
                if any(
                    item.row_index == row_index and item.col_index == col_index
                    for item in lattice.constraints
                ):
                    continue
                observed_value = lattice.ordinary_values[row_index][col_index]
                component_ref = lattice.ordinary_components[row_index][col_index]
                if observed_value < 0 or not component_ref or observed_value == expected_value:
                    continue
                top = lattice.top + row_index * lattice.stride
                left = lattice.left + col_index * lattice.stride
                bottom = top + lattice.cell_size - 1
                right = left + lattice.cell_size - 1
                mismatch_facts.append(
                    FrozenMap(
                        {
                            "component_ref": component_ref,
                            "global_slot": (row_index, col_index),
                            "normalized_position_ref": (
                                f"position.global_r{row_index:02d}c{col_index:02d}"
                            ),
                            "normalized_position_index": (
                                row_index * lattice.col_count + col_index
                            ),
                            "point": ((top + bottom) // 2, (left + right) // 2),
                            "target_bbox": (top, left, bottom, right),
                            "observed_outer_value": observed_value,
                            "expected_outer_value": expected_value,
                        }
                    )
                )
            expected_field = tuple(
                tuple(expected_by_slot.get((row, col)) for col in range(lattice.col_count))
                for row in range(lattice.row_count)
            )
            schema_ref = stable_digest(
                {
                    "top": lattice.top,
                    "left": lattice.left,
                    "shape": (lattice.row_count, lattice.col_count),
                    "cell_size": lattice.cell_size,
                    "gap": lattice.gap,
                    "constraints": tuple(constraint_refs),
                    "mapping": mapping_pairs,
                }
            )[:16]
            composition_ref = f"composition.local_constraints.{schema_ref}"
            conflict_count = len(conflict_facts)
            described.append(
                (
                    composition_ref,
                    FrozenMap(
                        {
                            "composition_ref": composition_ref,
                            "lattice_ref": f"lattice.rectangular.{schema_ref}",
                            "lattice_shape": (
                                lattice.row_count,
                                lattice.col_count,
                            ),
                            "cell_size": lattice.cell_size,
                            "inter_cell_gap": lattice.gap,
                            "local_constraint_count": len(lattice.constraints),
                            "constraint_refs": tuple(constraint_refs),
                            "anchor_component_refs": tuple(
                                item.component_ref for item in lattice.constraints
                            ),
                            "mapping_pairs": mapping_pairs,
                            "transferred_mapping_available": bool(transferred),
                            "transferred_mapping_consistent": (
                                bool(transferred_pairs_in_current_domain)
                                and transferred_mapping_agreement_count > 0
                                and transferred_mapping_conflict_count == 0
                            ),
                            "transferred_mapping_exactly_reused": (
                                bool(transferred) and mapping_pairs == transferred
                            ),
                            "transferred_mapping_agreement_count": (
                                transferred_mapping_agreement_count
                            ),
                            "transferred_mapping_conflict_count": (
                                transferred_mapping_conflict_count
                            ),
                            "scene_local_outer_domain": tuple(
                                sorted(scene_local_outer_domain)
                            ),
                            "normalized_position_projection": True,
                            "projected_overlap_count": len(overlap_slots),
                            "compatible_overlap_count": compatible_overlap_count,
                            "overlap_slot_refs": tuple(
                                f"slot.global_r{row:02d}c{col:02d}"
                                for row, col in overlap_slots
                            ),
                            "overlap_conflict_count": conflict_count,
                            "overlap_conflict_facts": conflict_facts,
                            "overlap_is_compatible": conflict_count == 0,
                            "shared_slots_deduplicated": True,
                            "composed_slot_count": len(expected_by_slot),
                            "composed_expected_field": expected_field,
                            "center_anchor_excluded": True,
                            "mismatch_count": len(mismatch_facts),
                            "mismatch_facts": tuple(mismatch_facts),
                            "mismatch_component_refs": tuple(
                                str(item["component_ref"]) for item in mismatch_facts
                            ),
                            "value_domain_component_refs": value_domain_refs,
                            "boundary_component_refs": boundary_refs,
                            "transition_order_observed": False,
                            "cyclicity_observed": False,
                        }
                    ),
                )
            )
    ordered = tuple(
        sorted(
            described,
            key=lambda item: (
                int(item[1].get("overlap_conflict_count") or 0) > 0,
                -int(item[1].get("composed_slot_count") or 0),
                -int(item[1].get("local_constraint_count") or 0),
                item[0],
            ),
        )[:MAX_COMPOSITION_WITNESSES]
    )
    compatible = tuple(
        ref for ref, facts in ordered if facts.get("overlap_is_compatible") is True
    )
    conflicting = tuple(ref for ref, _facts in ordered if ref not in compatible)
    return LocalConstraintCompositionAnalysis(
        composition_alternative_refs=tuple(ref for ref, _facts in ordered),
        composition_alternative_facts=FrozenMap(dict(ordered)),
        compatible_alternative_refs=compatible,
        conflicting_alternative_refs=conflicting,
        has_composition_candidates=bool(ordered),
    )


def enumerate_local_constraint_repair_actions(
    value: LocalConstraintRepairAgendaInput,
) -> InteractionProbeAgenda:
    """Enumerate exact mismatches only for one compatible SRC-selected union."""

    if POINT_ACTION_REF not in value.available_action_refs:
        return _empty_agenda()
    facts = value.analysis.composition_alternative_facts.get(
        value.selected_composition_ref
    )
    if not isinstance(facts, FrozenMap) or facts.get("overlap_is_compatible") is not True:
        return _empty_agenda()
    mismatch_count = int(facts.get("mismatch_count") or 0)
    described: list[tuple[InteractionProbeCandidate, FrozenMap]] = []
    for raw in facts.get("mismatch_facts") or ():
        if not isinstance(raw, FrozenMap):
            continue
        component_ref = str(raw.get("component_ref") or "")
        point = tuple(raw.get("point") or ())
        if not component_ref or len(point) != 2:
            continue
        candidate_ref = f"repair:{component_ref}"
        candidate_delta = FrozenMap(
            ChainMap(
                {
                    "candidate_ref": candidate_ref,
                    "component_ref": component_ref,
                    "belongs_to_selected_composed_difference_set": True,
                    "verified_repair_count": value.verified_repair_count,
                    "mismatch_count_before": mismatch_count,
                    "expected_mismatch_count_after": mismatch_count - 1,
                    "difference_sequence_ref": (
                        f"sequence:{candidate_ref}:{mismatch_count}"
                    ),
                },
                raw,
            )
        )
        described.append(
            (
                InteractionProbeCandidate(
                    candidate_ref=candidate_ref,
                    action_ref=POINT_ACTION_REF,
                    action_data=FrozenMap({"x": point[1], "y": point[0]}),
                    component_ref=component_ref,
                    point=(int(point[0]), int(point[1])),
                ),
                candidate_delta,
            )
        )
    ordered = tuple(sorted(described, key=lambda item: item[0].candidate_ref))
    candidates = tuple(item[0] for item in ordered)
    return InteractionProbeAgenda(
        candidates=candidates,
        alternative_refs=tuple(item.candidate_ref for item in candidates),
        alternative_facts=FrozenMap(
            {candidate.candidate_ref: item_facts for candidate, item_facts in ordered}
        ),
        context_facts=facts,
        has_candidates=bool(candidates),
    )


def analyze_homologous_repairs(
    value: HomologousRepairInput,
) -> HomologousRepairAnalysis:
    """Enumerate exact same-topology, different-scale symbol mappings."""

    scene = value.scene
    frame = scene.frame
    component_by_bbox = {
        (
            item.bbox.top,
            item.bbox.left,
            item.bbox.bottom,
            item.bbox.right,
        ): item.component_id
        for item in scene.blocks.components
    }
    grouped = _nested_panels(frame, component_by_bbox)
    described: list[tuple[str, FrozenMap]] = []
    boundary_refs = tuple(
        sorted(
            item.component_id
            for item in scene.blocks.components
            if item.touches_frame_boundary
            and min(item.bbox.height, item.bbox.width) == 1
            and max(item.bbox.height, item.bbox.width)
            * 2
            >= (frame.width if item.bbox.height == 1 else frame.height)
        )
    )
    for schema, panels in sorted(grouped.items()):
        if len(panels) < 4 or not _pairwise_disjoint(panels):
            continue
        outer_symbols = tuple(sorted({item for panel in panels for item in panel.outer_values}))
        inner_symbols = tuple(sorted({item for panel in panels for item in panel.inner_values}))
        if len(outer_symbols) != 2 or len(inner_symbols) != 2:
            continue
        order, macro_size, gap = schema
        panel_refs = tuple(
            f"panel.nested_grid.order_{order}.macro_{macro_size}.instance_{index:03d}"
            for index in range(1, len(panels) + 1)
        )
        for mapped_outer in permutations(outer_symbols):
            mapping = dict(zip(inner_symbols, mapped_outer, strict=True))
            mismatch_by_panel = tuple(
                tuple(
                    index
                    for index, (outer, inner) in enumerate(
                        zip(panel.outer_values, panel.inner_values, strict=True)
                    )
                    if outer != mapping[inner]
                )
                for panel in panels
            )
            coherent_refs = tuple(
                panel_ref
                for panel_ref, mismatches in zip(
                    panel_refs, mismatch_by_panel, strict=True
                )
                if not mismatches
            )
            exceptions = tuple(
                index
                for index, mismatches in enumerate(mismatch_by_panel)
                if mismatches
            )
            exception_index = exceptions[0] if len(exceptions) == 1 else None
            exception_panel_ref = (
                panel_refs[exception_index] if exception_index is not None else ""
            )
            exception_panel = (
                panels[exception_index] if exception_index is not None else None
            )
            mismatch_indexes = (
                mismatch_by_panel[exception_index]
                if exception_index is not None
                else ()
            )
            mapping_pairs = tuple(sorted(mapping.items()))
            schema_ref = stable_digest(
                {
                    "order": order,
                    "macro_size": macro_size,
                    "gap": gap,
                    "panel_count": len(panels),
                    "mapping": mapping_pairs,
                }
            )[:16]
            mapping_ref = f"mapping.normalized_multiscale.{schema_ref}"
            attention = _corner_attention_evidence(
                scene.blocks.components,
                exception_panel.bbox if exception_panel is not None else None,
            )
            mismatch_facts = tuple(
                _mismatch_fact(
                    panel=exception_panel,
                    panel_ref=exception_panel_ref,
                    flat_index=index,
                    mapping=mapping,
                    correspondence_ref=mapping_ref,
                )
                for index in mismatch_indexes
                if exception_panel is not None
            )
            described.append(
                (
                    mapping_ref,
                    FrozenMap(
                        {
                            "mapping_ref": mapping_ref,
                            "panel_count": len(panels),
                            "panel_order": order,
                            "macro_cell_size": macro_size,
                            "inter_cell_gap": gap,
                            "coherent_panel_count": len(coherent_refs),
                            "coherent_panel_refs": coherent_refs,
                            "exception_panel_count": len(exceptions),
                            "exception_panel_ref": exception_panel_ref,
                            "mismatch_count": len(mismatch_indexes),
                            "mismatch_facts": mismatch_facts,
                            "mismatch_component_refs": tuple(
                                str(item.get("component_ref"))
                                for item in mismatch_facts
                            ),
                            "mapping_pairs": mapping_pairs,
                            "normalized_position_correspondence": True,
                            "center_anchor_excluded": True,
                            "corner_pattern_detected": attention[0],
                            "corner_pattern_component_refs": attention[1],
                            "boundary_component_refs": boundary_refs,
                        }
                    ),
                )
            )
    ordered = tuple(sorted(described, key=lambda item: item[0]))
    return HomologousRepairAnalysis(
        mapping_alternative_refs=tuple(item[0] for item in ordered),
        mapping_alternative_facts=FrozenMap(dict(ordered)),
        has_mapping_candidates=bool(ordered),
    )


def enumerate_homologous_repair_actions(
    value: HomologousRepairAgendaInput,
) -> InteractionProbeAgenda:
    """Materialize only component-grounded mismatches for one SRC-selected map."""

    if POINT_ACTION_REF not in value.available_action_refs:
        return _empty_agenda()
    facts = value.analysis.mapping_alternative_facts.get(value.selected_mapping_ref)
    if not isinstance(facts, FrozenMap):
        return _empty_agenda()
    described: list[tuple[InteractionProbeCandidate, FrozenMap]] = []
    mismatch_count = int(facts.get("mismatch_count") or 0)
    for raw in facts.get("mismatch_facts") or ():
        if not isinstance(raw, FrozenMap):
            continue
        component_ref = str(raw.get("component_ref") or "")
        if not component_ref:
            continue
        candidate_ref = f"repair:{component_ref}"
        point = tuple(raw.get("point") or ())
        if len(point) != 2:
            continue
        expected_after = mismatch_count - 1
        candidate_delta = FrozenMap(
            ChainMap(
                {
                "candidate_ref": candidate_ref,
                "component_ref": component_ref,
                "belongs_to_selected_difference_set": True,
                "verified_repair_count": value.verified_repair_count,
                "mismatch_count_before": mismatch_count,
                "expected_mismatch_count_after": expected_after,
                "difference_sequence_ref": f"sequence:{candidate_ref}:{mismatch_count}",
                },
                raw,
            )
        )
        described.append(
            (
                InteractionProbeCandidate(
                    candidate_ref=candidate_ref,
                    action_ref=POINT_ACTION_REF,
                    action_data=FrozenMap({"x": point[1], "y": point[0]}),
                    component_ref=component_ref,
                    point=(int(point[0]), int(point[1])),
                ),
                candidate_delta,
            )
        )
    ordered = tuple(sorted(described, key=lambda item: item[0].candidate_ref))
    candidates = tuple(item[0] for item in ordered)
    return InteractionProbeAgenda(
        candidates=candidates,
        alternative_refs=tuple(item.candidate_ref for item in candidates),
        alternative_facts=FrozenMap(
            {candidate.candidate_ref: facts for candidate, facts in ordered}
        ),
        context_facts=facts,
        has_candidates=bool(candidates),
    )


def reconcile_homologous_repair(
    value: HomologousRepairReconciliationInput,
) -> HomologousRepairReconciliationAnalysis:
    """Compare one committed local repair prediction with its exact feedback."""

    facts = value.selected_plan_facts
    expected_after = int(facts.get("expected_mismatch_count_after") or 0)
    verified_before = int(facts.get("verified_repair_count") or 0)
    target_bbox = tuple(facts.get("target_bbox") or ())
    expected_value = facts.get("expected_outer_value")
    dimensions_match = (
        value.before.height == value.after.height
        and value.before.width == value.after.width
    )
    target_exact = False
    outside_changes_are_boundary_only = False
    changed_inside_count = 0
    changed_outside_count = 0
    changed_boundary_count = 0
    changed_nonboundary_outside_count = 0
    before_target_values: set[int] = set()
    after_target_values: set[int] = set()
    if dimensions_match and len(target_bbox) == 4 and isinstance(expected_value, int):
        top, left, bottom, right = (int(item) for item in target_bbox)
        outside_changes_are_boundary_only = True
        target_exact = True
        for row in range(value.before.height):
            for col in range(value.before.width):
                before_pixel = value.before.rows[row][col]
                after_pixel = value.after.rows[row][col]
                inside = top <= row <= bottom and left <= col <= right
                if inside:
                    before_target_values.add(before_pixel)
                    after_target_values.add(after_pixel)
                    target_exact = target_exact and after_pixel == expected_value
                    changed_inside_count += before_pixel != after_pixel
                elif before_pixel != after_pixel:
                    changed_outside_count += 1
                    on_boundary = row in {0, value.before.height - 1} or col in {
                        0,
                        value.before.width - 1,
                    }
                    if on_boundary:
                        changed_boundary_count += 1
                    else:
                        changed_nonboundary_outside_count += 1
                        outside_changes_are_boundary_only = False
    observed_value_edge: tuple[int, int] | tuple[()] = ()
    if (
        len(before_target_values) == 1
        and len(after_target_values) == 1
        and before_target_values != after_target_values
    ):
        observed_value_edge = (
            next(iter(before_target_values)),
            next(iter(after_target_values)),
        )
    local_observation_matches = (
        dimensions_match
        and target_exact
        and changed_inside_count > 0
        and outside_changes_are_boundary_only
    )
    score_increased = value.after_score > value.before_score
    descriptive_delta = {
        "transition_ref": value.transition_ref,
        "before_score": value.before_score,
        "after_score": value.after_score,
        "observed_mismatch_count_if_prediction_holds": expected_after,
        "verified_repair_count_if_prediction_holds": verified_before + 1,
        "frame_dimensions_match": dimensions_match,
        "target_pixels_match_prediction": target_exact,
        "local_observation_matches_prediction": local_observation_matches,
        "outside_changes_are_boundary_only": outside_changes_are_boundary_only,
        "changed_inside_count": changed_inside_count,
        "changed_outside_count": changed_outside_count,
        "changed_boundary_count": changed_boundary_count,
        "changed_nonboundary_outside_count": changed_nonboundary_outside_count,
        "selected_carrier_before_values": tuple(sorted(before_target_values)),
        "selected_carrier_after_values": tuple(sorted(after_target_values)),
        "selected_carrier_value_transition_observed": bool(observed_value_edge),
        "selected_carrier_value_transition_edge": observed_value_edge,
        "selected_carrier_self_only_nonboundary_mask_matches": (
            changed_inside_count > 0 and changed_nonboundary_outside_count == 0
        ),
        "score_increased": score_increased,
        "terminal_step_expected": expected_after == 0,
        "score_transition_consistent": score_increased == (expected_after == 0),
    }
    return HomologousRepairReconciliationAnalysis(
        descriptive_delta=FrozenMap(descriptive_delta),
        evidence_refs=(
            value.transition_ref,
            str(facts.get("candidate_ref") or "repair:unknown"),
        ),
        transition_ref=value.transition_ref,
    )


def _local_constraint_lattices(
    scene,
    component_by_bbox: dict[tuple[int, int, int, int], str],
) -> tuple[_LocalConstraintLattice, ...]:
    frame = scene.frame
    square_components = tuple(
        item
        for item in scene.blocks.components
        if item.bbox.height == item.bbox.width
        and item.bbox.height >= 2
        and item.area == item.bbox.height * item.bbox.width
        and not item.touches_frame_boundary
    )
    cell_sizes = tuple(sorted({item.bbox.height for item in square_components}))
    described: dict[tuple[object, ...], _LocalConstraintLattice] = {}
    for cell_size in cell_sizes:
        cohort = tuple(
            item for item in square_components if item.bbox.height == cell_size
        )
        row_sequences = _axis_sequences(
            {item.bbox.top for item in cohort}, cell_size
        )
        col_sequences = _axis_sequences(
            {item.bbox.left for item in cohort}, cell_size
        )
        for rows in row_sequences:
            for cols in col_sequences:
                if len(rows) * len(cols) > 49:
                    continue
                row_step = rows[1] - rows[0]
                col_step = cols[1] - cols[0]
                if row_step != col_step:
                    continue
                ordinary_values: list[list[int]] = []
                ordinary_components: list[list[str]] = []
                constraints: list[_EmbeddedConstraint] = []
                valid = True
                for row_index, top in enumerate(rows):
                    value_row: list[int] = []
                    component_row: list[str] = []
                    for col_index, left in enumerate(cols):
                        uniform = _uniform_value(
                            frame, top, left, cell_size, cell_size
                        )
                        bbox_key = (
                            top,
                            left,
                            top + cell_size - 1,
                            left + cell_size - 1,
                        )
                        component_ref = component_by_bbox.get(bbox_key, "")
                        if uniform is not None and component_ref:
                            value_row.append(uniform)
                            component_row.append(component_ref)
                            continue
                        pattern = _centered_inner_pattern(
                            frame, top, left, cell_size
                        )
                        if pattern is None:
                            valid = False
                            break
                        order, values = pattern
                        anchor_ref = "constraint.anchor." + stable_digest(
                            {
                                "bbox": bbox_key,
                                "order": order,
                                "values": values,
                            }
                        )[:16]
                        constraints.append(
                            _EmbeddedConstraint(
                                row_index=row_index,
                                col_index=col_index,
                                order=order,
                                values=values,
                                component_ref=anchor_ref,
                            )
                        )
                        value_row.append(-1)
                        component_row.append("")
                    if not valid:
                        break
                    ordinary_values.append(value_row)
                    ordinary_components.append(component_row)
                if not valid or not (2 <= len(constraints) <= 3):
                    continue
                orders = {item.order for item in constraints}
                if len(orders) != 1:
                    continue
                order = next(iter(orders))
                radius = order // 2
                if any(
                    item.row_index < radius
                    or item.col_index < radius
                    or item.row_index + radius >= len(rows)
                    or item.col_index + radius >= len(cols)
                    for item in constraints
                ):
                    continue
                lattice = _LocalConstraintLattice(
                    top=rows[0],
                    left=cols[0],
                    row_count=len(rows),
                    col_count=len(cols),
                    cell_size=cell_size,
                    gap=row_step - cell_size,
                    ordinary_values=tuple(tuple(row) for row in ordinary_values),
                    ordinary_components=tuple(
                        tuple(row) for row in ordinary_components
                    ),
                    constraints=tuple(
                        sorted(
                            constraints,
                            key=lambda item: (item.row_index, item.col_index),
                        )
                    ),
                )
                key = (
                    lattice.top,
                    lattice.left,
                    lattice.row_count,
                    lattice.col_count,
                    lattice.cell_size,
                    lattice.gap,
                    tuple(
                        (item.row_index, item.col_index, item.values)
                        for item in lattice.constraints
                    ),
                )
                described[key] = lattice
    return tuple(
        sorted(
            described.values(),
            key=lambda item: (
                -(item.row_count * item.col_count),
                item.top,
                item.left,
                item.cell_size,
            ),
        )
    )


def _axis_sequences(coordinates: set[int], cell_size: int) -> tuple[tuple[int, ...], ...]:
    sequences: set[tuple[int, ...]] = set()
    maximum_length = 7
    for step in range(cell_size, 2 * cell_size + 1):
        for start in sorted(coordinates):
            run: list[int] = []
            current = start
            while current in coordinates and len(run) < maximum_length:
                run.append(current)
                current += step
            for length in range(3, len(run) + 1):
                sequences.add(tuple(run[:length]))
    return tuple(sorted(sequences, key=lambda item: (-len(item), item)))


def _centered_inner_pattern(
    frame: FrameGrid,
    top: int,
    left: int,
    cell_size: int,
) -> tuple[int, tuple[int, ...]] | None:
    candidates: list[tuple[int, int, int, tuple[int, ...]]] = []
    for order in range(3, min(5, cell_size) + 1, 2):
        for micro_size in range(1, cell_size // order + 1):
            for gap in range(cell_size + 1):
                extent = order * micro_size + (order - 1) * gap
                remaining = cell_size - extent
                if remaining < 0 or remaining % 2:
                    continue
                offset = remaining // 2
                values: list[int] = []
                valid = True
                for row in range(order):
                    for col in range(order):
                        sampled = _uniform_value(
                            frame,
                            top + offset + row * (micro_size + gap),
                            left + offset + col * (micro_size + gap),
                            micro_size,
                            micro_size,
                        )
                        if sampled is None:
                            valid = False
                            break
                        values.append(sampled)
                    if not valid:
                        break
                if not valid or len(set(values)) < 2:
                    continue
                center = order // 2
                perimeter = tuple(
                    values[row * order + col]
                    for row in range(order)
                    for col in range(order)
                    if (row, col) != (center, center)
                )
                if len(set(perimeter)) != 2:
                    continue
                candidates.append((extent, micro_size, -gap, tuple(values)))
    if not candidates:
        return None
    extent, micro_size, negative_gap, values = max(candidates)
    del extent, micro_size, negative_gap
    order = int(len(values) ** 0.5)
    return order, values


def _nested_panels(
    frame: FrameGrid,
    component_by_bbox: dict[tuple[int, int, int, int], str],
) -> dict[tuple[int, int, int], tuple[_Panel, ...]]:
    groups: dict[tuple[int, int, int], list[_Panel]] = {}
    square_component_origins: dict[int, list[tuple[int, int]]] = {}
    for top, left, bottom, right in component_by_bbox:
        height = bottom - top + 1
        if height == right - left + 1:
            square_component_origins.setdefault(height, []).append((top, left))
    maximum_order = min(frame.height, frame.width, 7)
    for order in range(3, maximum_order + 1, 2):
        center = order // 2
        for macro_size in range(order, min(frame.height, frame.width) // order + 1):
            if macro_size % order:
                continue
            micro_size = macro_size // order
            for gap in range(macro_size + 1):
                stride = macro_size + gap
                extent = order * macro_size + (order - 1) * gap
                if extent > frame.height or extent > frame.width:
                    continue
                # Every valid perimeter includes its upper-left macro cell,
                # which must be an exact component with this square extent.
                # Indexing those origins is therefore an exact lazy bound,
                # not a structural selection.
                for top, left in square_component_origins.get(macro_size, ()):
                    if top + extent > frame.height or left + extent > frame.width:
                        continue
                    outer: list[int] = []
                    component_refs: list[str] = []
                    valid = True
                    for row_index, col_index in _perimeter_positions(order):
                        cell_top = top + row_index * stride
                        cell_left = left + col_index * stride
                        pixel = _uniform_value(
                            frame,
                            cell_top,
                            cell_left,
                            macro_size,
                            macro_size,
                        )
                        component_ref = component_by_bbox.get(
                            (
                                cell_top,
                                cell_left,
                                cell_top + macro_size - 1,
                                cell_left + macro_size - 1,
                            )
                        )
                        if pixel is None or component_ref is None:
                            valid = False
                            break
                        outer.append(pixel)
                        component_refs.append(component_ref)
                    if not valid or len(set(outer)) > 2:
                        continue
                    inner: list[int] = []
                    inner_top = top + center * stride
                    inner_left = left + center * stride
                    for row_index, col_index in _perimeter_positions(order):
                        pixel = _uniform_value(
                            frame,
                            inner_top + row_index * micro_size,
                            inner_left + col_index * micro_size,
                            micro_size,
                            micro_size,
                        )
                        if pixel is None:
                            valid = False
                            break
                        inner.append(pixel)
                    if not valid or len(set(inner)) != 2:
                        continue
                    groups.setdefault((order, macro_size, gap), []).append(
                        _Panel(
                            top=top,
                            left=left,
                            order=order,
                            macro_size=macro_size,
                            gap=gap,
                            outer_values=tuple(outer),
                            inner_values=tuple(inner),
                            outer_components=tuple(component_refs),
                        )
                    )
    return {
        schema: tuple(sorted(panels, key=lambda item: (item.top, item.left)))
        for schema, panels in sorted(groups.items())
    }


def _mismatch_fact(
    *,
    panel: _Panel,
    panel_ref: str,
    flat_index: int,
    mapping: dict[int, int],
    correspondence_ref: str,
) -> FrozenMap:
    positions = _perimeter_positions(panel.order)
    row_index, col_index = positions[flat_index]
    component_ref = panel.outer_components[flat_index]
    stride = panel.macro_size + panel.gap
    top = panel.top + row_index * stride
    left = panel.left + col_index * stride
    bottom = top + panel.macro_size - 1
    right = left + panel.macro_size - 1
    return FrozenMap(
        {
            "panel_ref": panel_ref,
            "correspondence_ref": correspondence_ref,
            "component_ref": component_ref,
            "normalized_position_ref": _position_ref(
                panel.order, row_index, col_index
            ),
            "normalized_position_index": col_index * panel.order + row_index,
            "point": ((top + bottom) // 2, (left + right) // 2),
            "target_bbox": (top, left, bottom, right),
            "expected_outer_value": mapping[panel.inner_values[flat_index]],
        }
    )


def _corner_attention_evidence(
    components: tuple[ComponentDescription, ...],
    focus_bbox: BoundingBox | None,
) -> tuple[bool, tuple[str, ...]]:
    if focus_bbox is None:
        return False, ()
    enclosing = tuple(
        item
        for item in components
        if item.bbox.top < focus_bbox.top
        and item.bbox.left < focus_bbox.left
        and item.bbox.bottom > focus_bbox.bottom
        and item.bbox.right > focus_bbox.right
    )
    for surface in sorted(
        enclosing,
        key=lambda item: (item.bbox.height * item.bbox.width, item.component_id),
    ):
        groups: dict[tuple[int, int, int, int], list[ComponentDescription]] = {}
        for item in components:
            if item.component_id == surface.component_id:
                continue
            key = (item.bbox.height, item.bbox.width, item.area, item.value)
            groups.setdefault(key, []).append(item)
        for group in (groups[__yf_order_key] for __yf_order_key in sorted(groups)):
            if len(group) < 4:
                continue
            height = group[0].bbox.height
            width = group[0].bbox.width
            corners = {
                (surface.bbox.top, surface.bbox.left),
                (surface.bbox.top, surface.bbox.right - width + 1),
                (surface.bbox.bottom - height + 1, surface.bbox.left),
                (
                    surface.bbox.bottom - height + 1,
                    surface.bbox.right - width + 1,
                ),
            }
            indexed = {
                (item.bbox.top, item.bbox.left): item.component_id for item in group
            }
            if corners <= set(indexed):
                return True, tuple(sorted(indexed[corner] for corner in corners))
    return False, ()


def _uniform_value(
    frame: FrameGrid,
    top: int,
    left: int,
    height: int,
    width: int,
) -> int | None:
    first = frame.rows[top][left]
    expected_row = (first,) * width
    return (
        first
        if all(
            frame.rows[row][left : left + width] == expected_row
            for row in range(top, top + height)
        )
        else None
    )


@lru_cache(maxsize=64)
def _perimeter_positions(order: int) -> tuple[tuple[int, int], ...]:
    center = order // 2
    return tuple(
        (row, col)
        for row in range(order)
        for col in range(order)
        if (row, col) != (center, center)
    )


def _position_ref(order: int, row: int, col: int) -> str:
    if order == 3:
        vertical = ("top", "middle", "bottom")[row]
        horizontal = ("left", "middle", "right")[col]
        return f"position.{vertical}_{horizontal}"
    return f"position.normalized_{row}_{col}_of_{order}"


def _pairwise_disjoint(panels: tuple[_Panel, ...]) -> bool:
    for index, first in enumerate(panels):
        for second in panels[index + 1 :]:
            if not (
                first.bbox.bottom < second.bbox.top
                or second.bbox.bottom < first.bbox.top
                or first.bbox.right < second.bbox.left
                or second.bbox.right < first.bbox.left
            ):
                return False
    return True


def _empty_agenda() -> InteractionProbeAgenda:
    return InteractionProbeAgenda(
        candidates=(),
        alternative_refs=(),
        alternative_facts=FrozenMap(),
        has_candidates=False,
    )


__all__ = (
    "analyze_local_constraint_compositions",
    "analyze_homologous_repairs",
    "enumerate_local_constraint_repair_actions",
    "enumerate_homologous_repair_actions",
    "reconcile_homologous_repair",
)
