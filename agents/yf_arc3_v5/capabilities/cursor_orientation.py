"""Exact temporal rotation of an explicitly supplied body/attached-part pair.

This does not discover a cursor, choose a decomposition, infer facing from a
palette, or bind the requested direction to an observed orientation. The caller
must preserve competing decompositions. DRM alone licenses their meaning.
"""

from __future__ import annotations

from pydantic import Field

from agents.yf_arc3_v5.capabilities.contracts import (
    ComponentDescription, TemporalTrackingResult,
)
from agents.yf_arc3_v5.logos.types import FrozenMap, FrozenModel, Ref, stable_digest


class AttachedStructureRotationInput(FrozenModel):
    previous: TemporalTrackingResult
    current: TemporalTrackingResult
    body_entity_ref: Ref
    previous_part_component_ref: Ref
    current_part_component_ref: Ref
    max_structure_pixels: int = Field(default=256, ge=1, le=4096)


class AttachedRotationAnalysisInput(FrozenModel):
    previous: TemporalTrackingResult
    current: TemporalTrackingResult
    body_entity_refs: tuple[Ref, ...] = Field(max_length=64)
    part_entity_refs: tuple[Ref, ...] = Field(default=(), max_length=64)
    max_part_pairs: int = Field(ge=1, le=256)
    max_structure_pixels: int = Field(ge=1, le=4096)
    schema_version: Ref = "yf_arc3_v5.attached_rotation_analysis_input.v1"


class AttachedRotationAnalysis(FrozenModel):
    descriptive_facts: FrozenMap
    evidence_refs: tuple[Ref, ...]
    schema_version: Ref = "yf_arc3_v5.attached_rotation_analysis.v1"


def analyze_attached_rotations(value: AttachedRotationAnalysisInput) -> AttachedRotationAnalysis:
    """Exhaustively measure bounded adjacent part pairs for supplied identities."""
    facts = {"body_reference_count": len(value.body_entity_refs),
             "structural_correspondence_count": 0,
             "rotation_enumeration_complete": True, "attached_rotation_unique": False,
             "attached_reorientation_unique": False}
    evidence = (value.previous.frame_ref, value.current.frame_ref)
    # No identity selection: multiple supplied bodies stay unresolved.
    if len(value.body_entity_refs) != 1 or len(value.part_entity_refs) > 1:
        return AttachedRotationAnalysis(descriptive_facts=FrozenMap(facts), evidence_refs=evidence)
    body_ref = value.body_entity_refs[0]
    old = tuple(e for e in value.previous.entities if e.entity_id == body_ref)
    new = tuple(e for e in value.current.entities if e.entity_id == body_ref)
    if len(old) != 1 or len(new) != 1:
        return AttachedRotationAnalysis(descriptive_facts=FrozenMap(facts), evidence_refs=evidence)
    old_parts = tuple(e.component for e in value.previous.entities
                      if (not value.part_entity_refs or e.entity_id in value.part_entity_refs)
                      and _attached(e.component, old[0].component))
    new_parts = tuple(e.component for e in value.current.entities if _attached(e.component, new[0].component))
    if len(old_parts) * len(new_parts) > value.max_part_pairs:
        facts["rotation_enumeration_complete"] = False
        return AttachedRotationAnalysis(descriptive_facts=FrozenMap(facts), evidence_refs=evidence)
    rotations = []
    for old_part in old_parts:
        for new_part in new_parts:
            result = measure_attached_structure_rotation(AttachedStructureRotationInput(
                previous=value.previous, current=value.current, body_entity_ref=body_ref,
                previous_part_component_ref=old_part.component_id,
                current_part_component_ref=new_part.component_id,
                max_structure_pixels=value.max_structure_pixels,
            ))
            if result.get("measurement_issue") == "structure_pixel_bound_reached":
                # Large adjacent regions cannot be silently discarded to manufacture uniqueness.
                facts["rotation_enumeration_complete"] = False
            if result.get("attached_reorientation_unique"):
                rotations.append(result)
    facts["structural_correspondence_count"] = len(rotations)
    if len(rotations) == 1:
        facts.update(rotations[0])
        evidence = (*evidence, str(rotations[0]["measurement_ref"]))
    return AttachedRotationAnalysis(descriptive_facts=FrozenMap(facts), evidence_refs=evidence)


def _offsets_twice(
    part: ComponentDescription, body: ComponentDescription,
) -> tuple[tuple[int, int], ...]:
    # Doubled coordinates retain off-centre detail around even-sized cores.
    row_sum = body.bbox.top + body.bbox.bottom
    col_sum = body.bbox.left + body.bbox.right
    return tuple(sorted((2 * r - row_sum, 2 * c - col_sum) for r, c in part.pixels))


def _quarter_turn(
    points: tuple[tuple[int, int], ...], count: int,
) -> tuple[tuple[int, int], ...]:
    for _ in range(count):
        points = tuple((c, -r) for r, c in points)
    return tuple(sorted(points))


def _attached(part: ComponentDescription, body: ComponentDescription) -> bool:
    body_pixels = frozenset(body.pixels)
    if body_pixels.intersection(part.pixels):
        return False
    return any(
        (r + dr, c + dc) in body_pixels
        for r, c in part.pixels
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
    )


def _side_normal(part: ComponentDescription, body: ComponentDescription):
    """Unique outward normal of a whole part outside one body-box side."""
    sides = tuple(normal for holds, normal in (
        (part.bbox.bottom < body.bbox.top, (-1, 0)),
        (part.bbox.top > body.bbox.bottom, (1, 0)),
        (part.bbox.right < body.bbox.left, (0, -1)),
        (part.bbox.left > body.bbox.right, (0, 1)),
    ) if holds)
    return sides[0] if len(sides) == 1 else None


def measure_attached_structure_rotation(value: AttachedStructureRotationInput) -> FrozenMap:
    """Measure at most four exact transforms; never turn absence into evidence.

    A stable body lineage is mandatory. Rotation is evaluated about that body's
    centre after its translation has been removed, including zero translation.
    A unique non-identity rigid transform or a changed unique attachment side
    with preserved part silhouette supports reorientation. Their evidence is
    kept distinct; a discrete side change never asserts rigid pixel rotation.
    """
    def absent(issue: str) -> FrozenMap:
        return FrozenMap({"attached_rotation_unique": False, "attached_reorientation_unique": False,
                          "measurement_issue": issue})

    before, after = value.previous, value.current
    if before.frame_ref == after.frame_ref or not after.action_ref:
        return absent("missing_action_conditioned_transition")
    previous_bodies = tuple(e for e in before.entities if e.entity_id == value.body_entity_ref)
    current_bodies = tuple(e for e in after.entities if e.entity_id == value.body_entity_ref)
    if len(previous_bodies) != 1 or len(current_bodies) != 1:
        return absent("body_lineage_not_unique")
    old_body, new_body = previous_bodies[0], current_bodies[0]
    if (
        old_body.current_frame_ref != before.frame_ref
        or new_body.current_frame_ref != after.frame_ref
        or new_body.identity_status != "established"
        or new_body.match_kind not in ("stationary_exact", "translated_exact")
        or new_body.previous_component_ref != old_body.component.component_id
        or new_body.first_seen_frame_ref != old_body.first_seen_frame_ref
    ):
        return absent("body_lineage_not_current_exact")
    old_core, new_core = old_body.component, new_body.component
    delta = (new_core.bbox.top - old_core.bbox.top, new_core.bbox.left - old_core.bbox.left)
    if (
        old_core.value != new_core.value
        or old_core.relative_pixels != new_core.relative_pixels
        or delta != (new_body.delta_row, new_body.delta_col)
    ):
        return absent("body_not_exactly_translated")
    old_parts = tuple(e for e in before.entities if e.component.component_id == value.previous_part_component_ref)
    new_parts = tuple(e for e in after.entities if e.component.component_id == value.current_part_component_ref)
    if len(old_parts) != 1 or len(new_parts) != 1:
        return absent("part_correspondence_not_unique")
    old_part_entity, new_part_entity = old_parts[0], new_parts[0]
    if old_part_entity.current_frame_ref != before.frame_ref or new_part_entity.current_frame_ref != after.frame_ref:
        return absent("part_geometry_is_stale")
    old_part, new_part = old_part_entity.component, new_part_entity.component
    if old_part.value != new_part.value or old_part.area != new_part.area:
        return absent("part_appearance_not_preserved")
    if max(old_core.area + old_part.area, new_core.area + new_part.area) > value.max_structure_pixels:
        # Reject impossible bounding-box rotations before expanding large masks.
        def corners(part, core):
            return tuple(sorted((2*r-core.bbox.top-core.bbox.bottom,
                                 2*c-core.bbox.left-core.bbox.right)
                                for r in (part.bbox.top, part.bbox.bottom)
                                for c in (part.bbox.left, part.bbox.right)))
        old_corners, new_corners = corners(old_part, old_core), corners(new_part, new_core)
        old_side, new_side = _side_normal(old_part, old_core), _side_normal(new_part, new_core)
        possible_side_change = (old_side is not None and new_side is not None and old_side != new_side
                                and old_part.bbox.height == new_part.bbox.height
                                and old_part.bbox.width == new_part.bbox.width)
        if not possible_side_change and not any(_quarter_turn(old_corners, k) == new_corners for k in (1, 2, 3)):
            return absent("part_bounds_exclude_non_identity_rotation")
        return absent("structure_pixel_bound_reached")
    if not _attached(old_part, old_core) or not _attached(new_part, new_core):
        return absent("part_not_attached_in_both_frames")
    old_offsets = _offsets_twice(old_part, old_core)
    new_offsets = _offsets_twice(new_part, new_core)
    turns = tuple(k for k in range(4) if _quarter_turn(old_offsets, k) == new_offsets)
    rigid_match = len(turns) == 1 and turns[0] != 0
    old_normal, new_normal = _side_normal(old_part, old_core), _side_normal(new_part, new_core)
    side_turns = tuple(k for k in (1, 2, 3)
                       if old_normal is not None and new_normal is not None
                       and _quarter_turn((old_normal,), k) == (new_normal,))
    discrete_match = (len(side_turns) == 1
                      and old_part.relative_pixels == new_part.relative_pixels)
    if len(turns) == 1 and turns[0] == 0:
        return absent("relative_structure_unchanged")
    if not rigid_match and not discrete_match:
        return absent("part_rotation_not_unique" if turns else "not_an_exact_quarter_rotation")
    measured = {
        "body_entity_ref": value.body_entity_ref,
        "previous_frame_ref": before.frame_ref,
        "current_frame_ref": after.frame_ref,
        "action_ref": after.action_ref,
        "body_delta_row": delta[0], "body_delta_col": delta[1],
        "body_relative_pixels": old_core.relative_pixels,
        "body_value": old_core.value, "part_value": old_part.value,
        "previous_part_component_ref": old_part.component_id,
        "current_part_component_ref": new_part.component_id,
        "previous_part_entity_ref": old_part_entity.entity_id,
        "current_part_entity_ref": new_part_entity.entity_id,
        "previous_part_offsets_twice": old_offsets,
        "current_part_offsets_twice": new_offsets,
        "quarter_turns_clockwise": turns[0] if rigid_match else None,
        "orientation_quarter_turns_clockwise": turns[0] if rigid_match else side_turns[0],
        "previous_side_normal": old_normal, "current_side_normal": new_normal,
        "part_relative_pixels": old_part.relative_pixels,
        "current_part_relative_pixels": new_part.relative_pixels,
        "rigid_rotation_match": rigid_match,
        "attached_rotation_unique": rigid_match,
        "attached_reorientation_unique": True,
    }
    measured["measurement_ref"] = "measurement.attached_rotation." + stable_digest(measured)[:24]
    return FrozenMap(measured)
