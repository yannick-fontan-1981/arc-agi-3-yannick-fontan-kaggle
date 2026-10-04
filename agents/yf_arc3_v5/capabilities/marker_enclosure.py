"""Bounded enclosure measurements of a supplied, remembered marker value.

No marker or bearer is assigned a role here. Nested candidates and all ties
are measured explicitly so Source can distinguish a carrier from its backdrop.
"""

from agents.yf_arc3_v5.capabilities.contracts import TemporalTrackingResult
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest
from collections.abc import Mapping
from agents.yf_arc3_v5.capabilities.bearer_support import (
    BearerSupportContract, CurrentBearerSupportRequest, measure_current_bearer_supports,
)


def _field(value, name):
    return value[name] if isinstance(value, Mapping) else getattr(value, name)


def measure_marker_enclosures(tracking: TemporalTrackingResult, *, marker_value: int,
                              max_entities: int, max_members: int) -> tuple[FrozenMap, ...]:
    entities = _field(tracking, "entities")
    frame_ref = _field(tracking, "frame_ref")
    if max_entities < 1 or max_members < 1 or len(entities) > max_entities:
        raise ValueError("marker enclosure entity bound exceeded")
    if any(_field(entity, "current_frame_ref") != frame_ref for entity in entities):
        raise ValueError("marker enclosure contains a stale entity")
    markers = tuple(entity for entity in entities if _field(_field(entity, "component"), "value") == marker_value)
    candidates = []
    for body in entities:
        component = _field(body, "component")
        if _field(component, "value") == marker_value:
            continue
        box = _field(component, "bbox")
        members = tuple(marker for marker in markers if (
            _field(box, "top") < _field(_field(_field(marker, "component"), "bbox"), "top")
            and _field(box, "left") < _field(_field(_field(marker, "component"), "bbox"), "left")
            and _field(box, "bottom") > _field(_field(_field(marker, "component"), "bbox"), "bottom")
            and _field(box, "right") > _field(_field(_field(marker, "component"), "bbox"), "right")
        ))
        if not members:
            continue
        if len(members) + 1 > max_members:
            raise ValueError("marker enclosure member bound exceeded; no members truncated")
        candidates.append((body, members))
    rows = []
    for body, members in candidates:
        component = _field(body, "component")
        box = _field(component, "bbox")
        smaller = tuple(other for other, _marks in candidates if _field(other, "entity_id") != _field(body, "entity_id") and (
            _field(box, "top") <= _field(_field(_field(other, "component"), "bbox"), "top")
            and _field(box, "left") <= _field(_field(_field(other, "component"), "bbox"), "left")
            and _field(box, "bottom") >= _field(_field(_field(other, "component"), "bbox"), "bottom")
            and _field(box, "right") >= _field(_field(_field(other, "component"), "bbox"), "right")
            and box != _field(_field(other, "component"), "bbox")
        ))
        rows.append(FrozenMap({
            "entity_ref": _field(body, "entity_id"), "component_ref": _field(component, "component_id"),
            "shape_digest": stable_digest(_field(component, "relative_pixels"))[:16],
            "bbox_height": _field(box, "bottom") - _field(box, "top") + 1,
            "bbox_width": _field(box, "right") - _field(box, "left") + 1,
            "observed_value": _field(component, "value"),
            "marker_member_refs": tuple(sorted(_field(member, "entity_id") for member in members)),
            "marker_component_count": len(members), "encloses_more_specific_marker_carrier": bool(smaller),
        }))
    return tuple(rows)


def measure_marker_ownership(tracking: TemporalTrackingResult, *, marker_value: int,
                             requests: tuple[CurrentBearerSupportRequest, ...],
                             max_entities: int, support_contract: BearerSupportContract):
    """Measure enclosure and marker membership, preserving every supplied proof.

    No group is discovered or interpreted here. Supplied unions are current,
    exact, disjoint-valued supports. A surrounding component is measured as
    containing a more specific carrier only when it encloses that union and
    shares at least one of its actual marker members.
    """
    if len(requests) > support_contract.max_supports:
        raise ValueError("marker ownership support bound exceeded; no rows truncated")
    atomic = measure_marker_enclosures(tracking, marker_value=marker_value,
        max_entities=max_entities, max_members=support_contract.max_members)
    by_ref = {entity.entity_id: entity for entity in tracking.entities}
    if len(by_ref) != len(tracking.entities):
        raise ValueError("marker ownership repeats a tracked identity")
    compound = []
    supports = []
    for request in requests:
        support, = measure_current_bearer_supports(tracking=tracking,
            requests=(request,), contract=support_contract)
        markers = tuple(ref for ref in support.member_entity_refs
                        if by_ref[ref].component.value == marker_value)
        if not markers:
            continue
        supports.append((support, markers))
        compound.append(FrozenMap({
            "bearer_ref": support.bearer_ref,
            "carrier_identity": stable_digest(("member_union", support.member_entity_refs)),
            "member_refs": support.member_entity_refs, "support_digest": support.support_digest,
            "bbox_height": support.bbox_height, "bbox_width": support.bbox_width,
            "marker_member_refs": markers, "marker_component_count": len(markers),
            "encloses_more_specific_marker_carrier": False,
        }))
    rows = []
    for row in atomic:
        box = by_ref[row["entity_ref"]].component.bbox
        more_specific = any(set(markers).intersection(row["marker_member_refs"])
            and box.top <= support.bbox_top and box.left <= support.bbox_left
            and box.bottom >= support.bbox_top + support.bbox_height - 1
            and box.right >= support.bbox_left + support.bbox_width - 1
            and (box.top, box.left, box.height, box.width) != (
                support.bbox_top, support.bbox_left, support.bbox_height, support.bbox_width)
            for support, markers in supports)
        rows.append(FrozenMap.overlay(FrozenMap({"carrier_identity": row["entity_ref"],
            "encloses_more_specific_marker_carrier": row["encloses_more_specific_marker_carrier"] or more_specific}), row))
    return tuple(rows), tuple(compound)
