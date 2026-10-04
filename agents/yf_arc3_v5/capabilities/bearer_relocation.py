"""Bounded exact raster relocation; no membership combinations or role choice."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

from pydantic import Field, model_validator

from agents.yf_arc3_v5.capabilities.bearer_support import (
    BearerSupportContract, CurrentBearerSupportRequest,
)
from agents.yf_arc3_v5.capabilities.contracts import TemporalTrackingResult
from agents.yf_arc3_v5.logos.types import FrozenModel, stable_digest


class BearerRelocationContract(FrozenModel):
    max_templates: int = Field(strict=True, ge=1, le=1024)
    max_scene_pixels: int = Field(strict=True, ge=1, le=65536)
    max_anchor_probes: int = Field(strict=True, ge=1, le=65536)
    max_pixel_checks: int = Field(strict=True, ge=1, le=4194304)
    rotation_quarter_turns: tuple[int, ...] = Field(min_length=1, max_length=4)
    support_match_modes: tuple[Literal["valued_pixels", "support_palette"], ...] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def exact_rotations(self) -> "BearerRelocationContract":
        if len(set(self.rotation_quarter_turns)) != len(self.rotation_quarter_turns) or any(
            turn not in (0, 1, 2, 3) for turn in self.rotation_quarter_turns
        ):
            raise ValueError("relocation requires distinct declared quarter turns")
        if len(set(self.support_match_modes)) != len(self.support_match_modes):
            raise ValueError("relocation repeats a declared support match mode")
        return self


def relocate_exact_bearer_templates(
    *, tracking: TemporalTrackingResult,
    templates: tuple[tuple[tuple[int, int, int], ...], ...],
    contract: BearerRelocationContract, support_contract: BearerSupportContract,
) -> tuple[CurrentBearerSupportRequest, ...]:
    """Translate declared colored rasters, then require complete current members.

    Every translation is anchored on an observed same-valued pixel. Membership
    follows pixel ownership, not a search over subsets of neighbouring entities.
    Rotations and support/palette equality generate measurements only; the
    ordinary key index still decides the strongest compatible signature.
    Reflected or scaled variants are not fabricated by this operation.
    """
    if len(templates) > contract.max_templates:
        raise ValueError("relocation template bound exceeded; no template truncated")
    if not templates:
        return ()
    if sum(len(entity.component.pixels) for entity in tracking.entities) > contract.max_scene_pixels:
        raise ValueError("relocation scene pixel bound exceeded before indexing")
    entities = {entity.entity_id: entity for entity in tracking.entities}
    if len(entities) != len(tracking.entities):
        raise ValueError("relocation repeats a tracked member")
    pixels = {}
    by_value = defaultdict(list)
    for entity in tracking.entities:
        if entity.current_frame_ref != tracking.frame_ref:
            raise ValueError("relocation member belongs to a stale frame")
        for position in entity.component.pixels:
            if position in pixels:
                raise ValueError("relocation members have overlapping pixels")
            pixels[position] = (entity.component.value, entity.entity_id)
            by_value[entity.component.value].append(position)
    variants = set()
    for template in templates:
        if not template or len(template) > support_contract.max_pixels:
            raise ValueError("relocation template pixel bound exceeded")
        if len({(row, col) for row, col, _value in template}) != len(template):
            raise ValueError("relocation template repeats a pixel")
        for turns in contract.rotation_quarter_turns:
            rotated = template
            for _ in range(turns):
                rotated = tuple((col, -row, value) for row, col, value in rotated)
            top = min(row for row, _col, _value in rotated)
            left = min(col for _row, col, _value in rotated)
            normalized = tuple(sorted((row - top, col - left, value) for row, col, value in rotated))
            height = max(row for row, _col, _value in normalized) + 1
            width = max(col for _row, col, _value in normalized) + 1
            if height * width > support_contract.max_bbox_area:
                raise ValueError("relocation template bbox bound exceeded")
            mask = tuple((row, col) for row, col, _value in normalized)
            palette = tuple(sorted({value for _row, _col, value in normalized}))
            for mode in contract.support_match_modes:
                variants.add((mode, mask, palette, normalized if mode == "valued_pixels" else ()))
    probes = checks = 0
    memberships = set()
    complete_member_pixels_by_size = {}
    for mode, mask, palette, template in sorted(variants):
        # A relocation must contain every pixel of each contributing member.
        # A member larger than the whole template cannot contribute even its
        # anchor. Exclude this impossible partial membership before counting
        # probes; preserve every complete member that can still fit.
        if len(mask) not in complete_member_pixels_by_size:
            eligible_by_value = defaultdict(list)
            for entity in tracking.entities:
                if len(entity.component.pixels) <= len(mask):
                    eligible_by_value[entity.component.value].extend(entity.component.pixels)
            complete_member_pixels_by_size[len(mask)] = eligible_by_value
        eligible_by_value = complete_member_pixels_by_size[len(mask)]
        if (any(not eligible_by_value[value] for value in palette)
                or sum(len(eligible_by_value[value]) for value in palette) < len(mask)):
            continue
        # Rare-value anchoring only reduces mechanical work, never role eligibility.
        if mode == "valued_pixels":
            anchor_row, anchor_col, anchor_value = min(
                template, key=lambda point: (len(eligible_by_value[point[2]]), point),
            )
            anchors = eligible_by_value[anchor_value]
            probes_to_check = tuple((row, col, (value,)) for row, col, value in template)
        else:
            anchor_row, anchor_col = mask[0]
            anchors = tuple(position for value in palette for position in eligible_by_value[value])
            probes_to_check = tuple((row, col, palette) for row, col in mask)
        if probes + len(anchors) > contract.max_anchor_probes:
            raise ValueError("relocation anchor probe bound exceeded; no candidate truncated")
        probes += len(anchors)
        for current_row, current_col in anchors:
            offset_row, offset_col = current_row - anchor_row, current_col - anchor_col
            member_refs = set()
            observed_values = set()
            for row, col, allowed_values in probes_to_check:
                checks += 1
                if checks > contract.max_pixel_checks:
                    raise ValueError("relocation pixel check bound exceeded; no candidate truncated")
                observed = pixels.get((row + offset_row, col + offset_col))
                if observed is None or observed[0] not in allowed_values:
                    break
                member_refs.add(observed[1])
                observed_values.add(observed[0])
            else:
                if len(member_refs) < 2 or observed_values != set(palette):
                    continue
                # All template pixels were visited and owned. Equal total size
                # excludes taking only a fragment of any complete member.
                if sum(len(entities[ref].component.pixels) for ref in member_refs) != len(mask):
                    continue
                if len(member_refs) > support_contract.max_members:
                    raise ValueError("relocation member bound exceeded")
                memberships.add(tuple(sorted(member_refs)))
                if len(memberships) > support_contract.max_supports:
                    raise ValueError("relocation support bound exceeded; no candidate truncated")
    return tuple(CurrentBearerSupportRequest(
        bearer_ref="measurement.composed." + stable_digest((tracking.frame_ref, members)),
        member_entity_refs=members, observation_ref=tracking.frame_ref,
    ) for members in sorted(memberships))
