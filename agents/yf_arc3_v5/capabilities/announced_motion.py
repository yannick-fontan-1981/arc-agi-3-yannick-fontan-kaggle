"""Bounded endpoint arithmetic, not interpretation or action selection.

Inputs are observation/declared-model projections. Both unmodified and
obstacle-cancelled endpoint coincidences are measured, never chosen here.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Literal
from pydantic import Field

from agents.yf_arc3_v5.logos.types import FrozenMap, FrozenModel


class AnnouncedStepInput(FrozenModel):
    schema_version: Literal["yf_arc3_v5.announced_step_input.v1"] = "yf_arc3_v5.announced_step_input.v1"
    data: FrozenMap = Field(default_factory=FrozenMap)
    current_frame_ref: str
    alternative_refs: tuple[str, ...]


class AnnouncedStepMeasurements(FrozenModel):
    schema_version: Literal["yf_arc3_v5.announced_step_measurements.v1"] = "yf_arc3_v5.announced_step_measurements.v1"
    alternative_refs: tuple[str, ...]
    alternative_facts: FrozenMap


MAX_BODIES = 64
MAX_PAIRS = 128
MAX_CELLS = 4096


def _point(value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("endpoint must have two coordinates")
    if any(type(item) is not int for item in value):
        raise ValueError("endpoint coordinates must be integers")
    return tuple(value)


def measure_announced_step(value: AnnouncedStepInput) -> AnnouncedStepMeasurements:
    data = value.data
    bodies = tuple(data.get("bodies", ()))
    pairs = tuple(data.get("excluded_coincidence_pairs", ()))
    cells = tuple(data.get("post_action_blocked_cells", ()))
    if len(bodies) > MAX_BODIES or len(pairs) > MAX_PAIRS or len(cells) > MAX_CELLS:
        raise ValueError("announced step cardinality bound reached")
    if not 1 <= len(value.alternative_refs) <= 4:
        raise ValueError("announced step alternative bound")
    if len(set(value.alternative_refs)) != len(value.alternative_refs):
        raise ValueError("duplicate announced step alternative")
    walls = frozenset(_point(cell) for cell in cells)
    endpoints = {}
    cancelled = {}
    originals = {}
    missing = 0
    obstructed = 0
    origin_obstructions = 0
    destination_obstructions = 0
    for body in bodies:
        if not isinstance(body, Mapping):
            raise ValueError("body must be a mapping")
        ref = body.get("entity_ref")
        if not isinstance(ref, str) or not ref or ref in originals:
            raise ValueError("body identity missing or duplicated")
        point = _point(body.get("position"))
        originals[ref] = point
        destination = body.get("destination")
        if destination is None:
            missing += 1
            continue
        endpoint = _point(destination)
        endpoints[ref] = endpoint
        intersects = point in walls or endpoint in walls
        origin_obstructions += int(point in walls)
        destination_obstructions += int(endpoint in walls)
        obstructed += int(intersects)
        cancelled[ref] = point if intersects else endpoint
    pair_set = set()
    current_count = 0
    announced_count = 0
    cancelled_count = 0
    uncovered_pairs = 0
    for pair in pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("coincidence pair must contain two identities")
        left, right = pair
        if left == right or left not in originals or right not in originals:
            raise ValueError("coincidence pair references missing/distinct bodies")
        identity = tuple(sorted((left, right)))
        if identity in pair_set:
            raise ValueError("duplicate coincidence pair")
        pair_set.add(identity)
        current_count += int(originals[left] == originals[right])
        if left not in endpoints or right not in endpoints:
            uncovered_pairs += 1
            continue
        announced_count += int(endpoints[left] == endpoints[right])
        cancelled_count += int(cancelled[left] == cancelled[right])
    counts = {}
    for key in ("announced_match_count", "announced_divergence_count",
                "blocked_stay_count", "blocked_reroute_count",
                "origin_stay_count", "origin_reroute_count"):
        item = data.get(key, 0)
        if type(item) is not int or not 0 <= item <= MAX_CELLS:
            raise ValueError("invalid observed transition count")
        counts[key] = item
    facts = FrozenMap({
        "body_count": len(bodies), "pair_count": len(pairs),
        "missing_destination_count": missing, "obstructed_body_count": obstructed,
        "origin_obstruction_count": origin_obstructions,
        "destination_obstruction_count": destination_obstructions,
        "current_coincidence_count": current_count,
        "announced_coincidence_count": announced_count,
        "cancelled_coincidence_count": cancelled_count,
        "uncovered_pair_count": uncovered_pairs,
        "observation_revision": data.get("observation_revision"),
        "prediction_revision": data.get("prediction_revision"),
        "prediction_frame_ref": data.get("prediction_frame_ref"),
        "current_frame_ref": value.current_frame_ref,
        **counts,
    })
    return AnnouncedStepMeasurements(alternative_refs=value.alternative_refs,
        alternative_facts=FrozenMap({ref: facts for ref in value.alternative_refs}))
