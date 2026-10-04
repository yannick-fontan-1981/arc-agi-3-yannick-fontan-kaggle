"""Exact union of explicitly supplied current members, without role inference.

This does not discover groups or choose which neighbouring components belong
together. The caller supplies the bounded membership; every pixel keeps its
observed value. A support reference is not a new tracked temporal identity.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from agents.yf_arc3_v5.capabilities.contracts import TemporalTrackingResult
from agents.yf_arc3_v5.logos.types import FrozenModel, Ref, stable_digest


class BearerSupportContract(FrozenModel):
    max_members: int = Field(strict=True, ge=2, le=64)
    max_pixels: int = Field(strict=True, ge=1, le=4096)
    max_bbox_area: int = Field(strict=True, ge=1, le=4096)
    max_supports: int = Field(strict=True, ge=1, le=1024)


class CurrentBearerSupportRequest(FrozenModel):
    bearer_ref: Ref
    member_entity_refs: tuple[Ref, ...] = Field(min_length=2, max_length=64)
    observation_ref: Ref

    @model_validator(mode="after")
    def unique_members(self) -> "CurrentBearerSupportRequest":
        if len(set(self.member_entity_refs)) != len(self.member_entity_refs):
            raise ValueError("bearer support repeats a member")
        return self


class MeasuredBearerSupport(FrozenModel):
    bearer_ref: Ref
    member_entity_refs: tuple[Ref, ...]
    observation_ref: Ref
    valued_pixels: tuple[tuple[int, int, int], ...]
    bbox_top: int
    bbox_left: int
    bbox_height: int
    bbox_width: int
    observed_values: tuple[int, ...]
    support_digest: Ref


def measure_current_bearer_supports(
    *, tracking: TemporalTrackingResult,
    requests: tuple[CurrentBearerSupportRequest, ...],
    contract: BearerSupportContract,
) -> tuple[MeasuredBearerSupport, ...]:
    if len(requests) > contract.max_supports:
        raise ValueError("bearer support enumeration bound exceeded; no support truncated")
    if len({request.bearer_ref for request in requests}) != len(requests):
        raise ValueError("bearer support repeats a reference")
    memberships = tuple(frozenset(request.member_entity_refs) for request in requests)
    if len(set(memberships)) != len(memberships):
        raise ValueError("bearer support repeats the same member union")
    entities = {entity.entity_id: entity for entity in tracking.entities}
    output = []
    for request in requests:
        if request.observation_ref != tracking.frame_ref:
            raise ValueError("bearer support observation is not the current tracking frame")
        if request.bearer_ref in entities:
            raise ValueError("a composed support cannot alias an individual tracked entity")
        if len(request.member_entity_refs) > contract.max_members:
            raise ValueError("bearer support member bound exceeded")
        members = tuple(entities.get(reference) for reference in sorted(request.member_entity_refs))
        if any(member is None for member in members):
            raise ValueError("bearer support member is absent from current tracking")
        if any(member.current_frame_ref != tracking.frame_ref for member in members):
            raise ValueError("bearer support member belongs to a stale frame")
        if sum(len(member.component.pixels) for member in members) > contract.max_pixels:
            raise ValueError("bearer support pixel bound exceeded before union")
        valued = {}
        for member in members:
            for row, col in member.component.pixels:
                if (row, col) in valued:
                    raise ValueError("bearer support members have overlapping pixels")
                valued[(row, col)] = member.component.value
        top = min(row for row, _col in valued)
        left = min(col for _row, col in valued)
        height = max(row for row, _col in valued) - top + 1
        width = max(col for _row, col in valued) - left + 1
        if height * width > contract.max_bbox_area:
            raise ValueError("bearer support bbox bound exceeded")
        pixels = tuple((row, col, value) for (row, col), value in sorted(valued.items()))
        output.append(MeasuredBearerSupport(
            bearer_ref=request.bearer_ref, member_entity_refs=tuple(sorted(request.member_entity_refs)),
            observation_ref=tracking.frame_ref, valued_pixels=pixels,
            bbox_top=top, bbox_left=left, bbox_height=height, bbox_width=width,
            observed_values=tuple(sorted(set(valued.values()))),
            support_digest=stable_digest(tuple((row - top, col - left, value) for row, col, value in pixels)),
        ))
    return tuple(output)


def measure_proper_bbox_enclosing_members(
    *, tracking: TemporalTrackingResult, support: MeasuredBearerSupport, max_members: int,
) -> tuple[Ref, ...]:
    """All actual members whose bbox properly contains the other members.

    This is geometry only: no role or new temporal identity is returned, and
    multiple enclosing candidates remain multiple candidates.
    """
    if not 2 <= max_members <= 64 or len(support.member_entity_refs) > max_members:
        raise ValueError("compound anchor member bound exceeded")
    by_reference = {item.entity_id: item for item in tracking.entities}
    members = tuple(by_reference[reference] for reference in support.member_entity_refs)
    return tuple(sorted(member.entity_id for member in members if all(
        other.entity_id == member.entity_id or (
            member.component.bbox != other.component.bbox
            and member.component.bbox.top <= other.component.bbox.top
            and member.component.bbox.left <= other.component.bbox.left
            and member.component.bbox.bottom >= other.component.bbox.bottom
            and member.component.bbox.right >= other.component.bbox.right
        ) for other in members
    )))


def measure_adjacent_translation_groups(
    *, tracking: TemporalTrackingResult,
    member_translations: tuple[tuple[Ref, int, int], ...],
    connectivity: int, max_members: int, max_pixels: int,
) -> tuple[tuple[Ref, ...], ...]:
    """Exact connected groups of adjacent members sharing one displacement.

    The caller supplies command-aligned translations. This measures all groups,
    including singleton groups, without selecting a group or assigning a role.
    """
    if connectivity not in (4, 8):
        raise ValueError("unsupported declared adjacency connectivity")
    if not 1 <= max_members <= 64 or len(member_translations) > max_members:
        raise ValueError("adjacency member bound exceeded")
    if not 1 <= max_pixels <= 4096:
        raise ValueError("adjacency pixel bound exceeded")
    deltas = {reference: (row, col) for reference, row, col in member_translations}
    if len(deltas) != len(member_translations):
        raise ValueError("adjacency repeats a member")
    if any(delta == (0, 0) for delta in sorted(deltas.values())):
        raise ValueError("adjacency requires nonzero translations")
    current = {entity.entity_id: entity for entity in tracking.entities}
    pixel_owner: dict[tuple[int, int], Ref] = {}
    for reference in sorted(deltas):
        entity = current.get(reference)
        if entity is None or entity.current_frame_ref != tracking.frame_ref:
            raise ValueError("adjacency member missing or stale")
        for pixel in entity.component.pixels:
            if pixel in pixel_owner:
                raise ValueError("adjacency member pixels overlap")
            pixel_owner[pixel] = reference
            if len(pixel_owner) > max_pixels:
                raise ValueError("adjacency pixel bound exceeded")
    offsets = tuple((row, col) for row in (-1, 0, 1) for col in (-1, 0, 1)
                    if (row, col) != (0, 0) and (connectivity == 8 or abs(row) + abs(col) == 1))
    neighbors: dict[Ref, set[Ref]] = {reference: set() for reference in deltas}
    for (row, col), reference in sorted(pixel_owner.items()):
        for delta_row, delta_col in offsets:
            other = pixel_owner.get((row + delta_row, col + delta_col))
            if other is not None and other != reference and deltas[other] == deltas[reference]:
                neighbors[reference].add(other)
    remaining = set(deltas)
    groups: list[tuple[Ref, ...]] = []
    while remaining:
        pending = [min(remaining)]
        members: set[Ref] = set()
        while pending:
            reference = pending.pop()
            if reference in members:
                continue
            members.add(reference)
            pending.extend(sorted(neighbors[reference] - members))
        remaining.difference_update(members)
        groups.append(tuple(sorted(members)))
    return tuple(sorted(groups))


def measure_translated_enclosed_member_refs(
    *, before: TemporalTrackingResult, after: TemporalTrackingResult,
    anchor_entity_ref: Ref, max_members: int, max_pixels: int,
) -> tuple[Ref, ...] | None:
    """Compare complete valued supports without identities for repeated marks.

    Only the declared anchor needs temporal lineage. Other current members
    are geometrically enclosed; the complete before/after union must equal
    one exact nonzero translation. This does not select a role or a palette.
    """
    if not 1 <= max_members <= 64 or not 1 <= max_pixels <= 4096:
        raise ValueError("translated support declaration bound exceeded")
    before_entities = {entity.entity_id: entity for entity in before.entities}
    after_entities = {entity.entity_id: entity for entity in after.entities}
    old_anchor = before_entities.get(anchor_entity_ref)
    new_anchor = after_entities.get(anchor_entity_ref)
    if old_anchor is None or new_anchor is None:
        return None
    if old_anchor.current_frame_ref != before.frame_ref or new_anchor.current_frame_ref != after.frame_ref:
        return None
    delta = (new_anchor.delta_row, new_anchor.delta_col)
    if delta == (0, 0):
        return None

    def enclosed_support(tracking: TemporalTrackingResult, anchor):
        box = anchor.component.bbox
        members = tuple(entity for entity in tracking.entities if entity.entity_id == anchor_entity_ref or (
            entity.component.bbox != box
            and box.top <= entity.component.bbox.top and box.left <= entity.component.bbox.left
            and box.bottom >= entity.component.bbox.bottom and box.right >= entity.component.bbox.right
        ))
        if len(members) > max_members or sum(len(entity.component.pixels) for entity in members) > max_pixels:
            raise ValueError("translated enclosed support bound exceeded")
        if any(entity.current_frame_ref != tracking.frame_ref for entity in members):
            return None
        pixels = tuple(sorted((row, col, entity.component.value)
                              for entity in members for row, col in entity.component.pixels))
        if len({(row, col) for row, col, _value in pixels}) != len(pixels):
            raise ValueError("translated enclosed support has overlapping members")
        return tuple(sorted(entity.entity_id for entity in members)), pixels

    old_support = enclosed_support(before, old_anchor)
    new_support = enclosed_support(after, new_anchor)
    if old_support is None or new_support is None:
        return None
    translated_pixels = tuple(sorted((row + delta[0], col + delta[1], value)
                                    for row, col, value in old_support[1]))
    if translated_pixels != new_support[1]:
        return None
    return new_support[0]
