"""Bounded geometric correspondences of overlapping binary supports.

All palette assignments below are conditional raster descriptions. No colour
has a built-in role, and these measurements do not choose an action or commit
a composition law. Closed holes and boundary-open notches are both compared.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import hashlib
import json
from functools import lru_cache

from agents.yf_arc3_v5.capabilities.components import detect_components
from agents.yf_arc3_v5.capabilities.contracts import ComponentExtractionInput, FrameGrid

MAX_FRAME_AREA = 4096
MAX_BOXES = 8
MAX_PAIR_DESCRIPTIONS = 3
MAX_MATCHES = 3
MAX_PATCH_AREA = 256


@dataclass(frozen=True)
class RasterBox:
    top: int
    left: int
    height: int
    width: int
    value: int
    visible_sides: int

    @property
    def interior(self) -> tuple[int, int, int, int]:
        return self.top + 1, self.left + 1, self.height - 2, self.width - 2


@dataclass(frozen=True)
class MaskPose:
    clockwise_quarters: int
    window_row: int
    window_col: int
    observed_cell_count: int
    row_delta: int
    col_delta: int


@dataclass(frozen=True)
class LayerCorrespondence:
    value: int
    extra_value: int
    external_cell_count: int
    poses: tuple[MaskPose, ...]
    maximal_observed_match_count: int
    current_patch_difference_count: int


@dataclass(frozen=True)
class LayerField:
    first_box: RasterBox
    second_box: RasterBox
    background_value: int
    extra_value: int
    layers: tuple[LayerCorrespondence, ...]
    exact_current_pattern_difference_count: int


@dataclass(frozen=True)
class LayerRasterSample:
    fields: tuple[LayerField, ...]
    patch_hashes: tuple[tuple[RasterBox, str], ...]


@dataclass(frozen=True)
class LayerTransitionMeasure:
    action_ref: str
    before: LayerRasterSample
    after: LayerRasterSample
    point_value: int | None
    added_marker_value: int | None
    changed_pixel_count: int
    # (value, clockwise turn, row translation, column translation)
    motions: tuple[tuple[int, int, int, int], ...]
    intersection_observed_count: int
    intersection_mismatch_count: int


def rectangular_outline_measurements(frame: FrameGrid) -> tuple[RasterBox, ...]:
    """Complete rectangles and a congruent three-sided outline interrupted by a band."""
    if frame.height * frame.width > MAX_FRAME_AREA:
        return ()
    components = detect_components(ComponentExtractionInput(frame=frame, connectivity=4)).components
    complete = []
    for component in components:
        box = component.bbox
        h, w = box.height, box.width
        if h < 3 or w < 3 or (h - 2) * (w - 2) > MAX_PATCH_AREA:
            continue
        border = {(r, c) for r in range(box.top, box.bottom + 1) for c in range(box.left, box.right + 1)
                  if r in (box.top, box.bottom) or c in (box.left, box.right)}
        if frozenset(component.pixels) == border:
            complete.append(RasterBox(box.top, box.left, h, w, component.value, 4))
    result = list(complete)
    for component in components:
        pixels = frozenset(component.pixels)
        for exemplar in complete:
            if component.value != exemplar.value:
                continue
            # Any one side may be interrupted by a uniform raster-spanning
            # band. Enumerate all four directions; no screen position is fixed.
            h, w = exemplar.height, exemplar.width
            for side in range(4):
                top = component.bbox.top - int(side == 0)
                left = component.bbox.left - int(side == 2)
                bottom, right = top + h - 1, left + w - 1
                if top < 0 or left < 0 or bottom >= frame.height or right >= frame.width:
                    continue
                missing = ({(top, c) for c in range(left, right + 1)},
                           {(bottom, c) for c in range(left, right + 1)},
                           {(r, left) for r in range(top, bottom + 1)},
                           {(r, right) for r in range(top, bottom + 1)})[side]
                border = {(r, c) for r in range(top, bottom + 1) for c in range(left, right + 1)
                          if r in (top, bottom) or c in (left, right)}
                if pixels != border - missing:
                    continue
                line = (frame.rows[top] if side == 0 else frame.rows[bottom] if side == 1
                        else tuple(row[left if side == 2 else right] for row in frame.rows))
                if len(set(line)) == 1:
                    result.append(RasterBox(top, left, h, w, component.value, 3))
    return tuple(result) if len(result) <= MAX_BOXES else ()


def _rotate(rows: tuple[tuple[bool, ...], ...]) -> tuple[tuple[bool, ...], ...]:
    return tuple(tuple(column) for column in zip(*rows[::-1]))


def _mask_matches(
    known: tuple[tuple[bool, ...], ...], support: tuple[tuple[bool, ...], ...],
    target: tuple[tuple[bool, ...], ...], origin: tuple[int, int],
) -> tuple[tuple[MaskPose, ...], int]:
    """Measure four orientations and every bounded raster-window translation.

Retain only the maximal observed agreement count and up to three occurrences.
The caller receives ambiguity explicitly, never a silently truncated choice.
There are no paths, operator sequences, or simulated hidden cells here.
"""
    h, w = len(target), len(target[0])
    mask = (1 << w) - 1
    target_bits = tuple(sum(int(x) << c for c, x in enumerate(row)) for row in target)
    maximum, count, occurrences = -1, 0, []
    for q in range(4):
        raster_bits = (0,) * (h - 1) + tuple(sum(int(x) << (c + w - 1) for c, x in enumerate(row)) for row in support) + (0,) * (h - 1)
        visible_bits = (0,) * (h - 1) + tuple(sum(int(x) << (c + w - 1) for c, x in enumerate(row)) for row in known) + (0,) * (h - 1)
        for r in range(len(support) + h - 1):
            for c in range(len(support[0]) + w - 1):
                evidence, positive, negative = 0, False, False
                for i, target_row in enumerate(target_bits):
                    visible = (visible_bits[r + i] >> c) & mask
                    source = (raster_bits[r + i] >> c) & mask
                    if (source ^ target_row) & visible:
                        break
                    evidence += visible.bit_count()
                    positive |= bool(visible & target_row)
                    negative |= bool(visible & ~target_row)
                else:
                    if not positive or not negative or evidence < maximum:
                        continue
                    if evidence > maximum:
                        maximum, count, occurrences = evidence, 0, []
                    count += 1
                    if len(occurrences) < MAX_MATCHES:
                        rr, cc = r - h + 1, c - w + 1
                        occurrences.append(MaskPose(q, rr, cc, evidence, origin[0] - rr, origin[1] - cc))
        support, known = _rotate(support), _rotate(known)
    return (tuple(occurrences) if count <= MAX_MATCHES else ()), count


def measure_layer_fields(frame: FrameGrid) -> tuple[LayerField, ...]:
    boxes = rectangular_outline_measurements(frame)
    if not boxes:
        return ()
    raster = frame.rows
    fields = []
    for first, second in combinations(boxes, 2):
        if (first.height, first.width) != (second.height, second.width):
            continue
        # Both ordered geometric descriptions remain possible. Only a unique
        # occurrence may be consumed by an external declarative rule.
        for a, b in ((first, second), (second, first)):
            ar, ac, h, w = a.interior
            br, bc, _, _ = b.interior
            if not (ar + h <= br or br + h <= ar or ac + w <= bc or bc + w <= ac):
                continue
            target = tuple(row[bc:bc + w] for row in raster[br:br + h])
            counts = Counter(x for row in target for x in row)
            if len(counts) != 4:
                continue
            exterior = tuple(tuple(not (b.top <= r < b.top + b.height and b.left <= c < b.left + b.width)
                                  for c in range(frame.width)) for r in range(frame.height))
            outside = Counter(x for r, row in enumerate(raster) for c, x in enumerate(row) if exterior[r][c])
            # Exterior connected supports must exceed the enclosed patch area.
            # A value spanning all four raster edges is recorded separately.
            sides = (raster[0], raster[-1], tuple(row[0] for row in raster), tuple(row[-1] for row in raster))
            spanning_edges = [v for v in counts if sum(v in side for side in sides) >= 3]
            if len(spanning_edges) != 1:
                continue
            background = spanning_edges[0]
            values = [v for v in counts if v != background and outside[v] > h * w]
            if len(values) != 2:
                continue
            remaining = set(counts) - {background, *values}
            if len(remaining) != 1:
                continue
            extra = remaining.pop()
            known = tuple(tuple(x in counts and exterior[r][c] for c, x in enumerate(row)) for r, row in enumerate(raster))
            layers = []
            for v in sorted(values):
                support = tuple(tuple(x in (v, extra) for x in row) for row in raster)
                target_mask = tuple(tuple(x in (v, extra) for x in row) for row in target)
                poses, count = _mask_matches(known, support, target_mask, (ar, ac))
                difference = sum(support[ar+r][ac+c] != target_mask[r][c] for r in range(h) for c in range(w))
                layers.append(LayerCorrespondence(v, extra, outside[v], poses, count, difference))
            difference = sum(raster[ar+r][ac+c] != target[r][c] for r in range(h) for c in range(w))
            fields.append(LayerField(a, b, background, extra, tuple(layers), difference))
            if len(fields) > MAX_PAIR_DESCRIPTIONS:
                return ()
    return tuple(fields)


@lru_cache(maxsize=4)
def measure_layer_sample(frame: FrameGrid) -> LayerRasterSample:
    fields = measure_layer_fields(frame)
    boxes = tuple(dict.fromkeys(box for field in fields for box in (field.first_box, field.second_box)))
    hashes = []
    for box in boxes:
        r, c, h, w = box.interior
        patch = tuple(tuple(row[c:c+w]) for row in frame.rows[r:r+h])
        hashes.append((box, hashlib.sha256(json.dumps(patch, sort_keys=True, separators=(",", ":")).encode()).hexdigest()))
    return LayerRasterSample(fields, tuple(hashes))


def measure_layer_transition(
    before: FrameGrid, after: FrameGrid, action_ref: str,
    point: tuple[int, int] | None,
) -> LayerTransitionMeasure | None:
    previous = measure_layer_sample(before)
    if not previous.fields:
        return None
    current = measure_layer_sample(after)
    changes = tuple((r, c, a, b) for r, (ra, rb) in enumerate(zip(before.rows, after.rows))
                    for c, (a, b) in enumerate(zip(ra, rb)) if a != b)
    point_value = before.rows[point[0]][point[1]] if point is not None and 0 <= point[0] < before.height and 0 <= point[1] < before.width else None
    # A transferred marker can reuse a colour seen in the preceding raster.
    palette = {v for f in previous.fields for v in (f.background_value, f.extra_value, f.first_box.value, *(x.value for x in f.layers))}
    added = tuple(b for _, _, a, b in changes if b not in palette and a in palette)
    marker = added[0] if point is not None and len(added) == 1 and len(changes) <= 2 else None
    motions = []
    intersection_count, mismatch_count = 0, 0
    for first in previous.fields:
        if dict(previous.patch_hashes).get(first.second_box) != dict(current.patch_hashes).get(first.second_box):
            continue
        peers = tuple(x for x in current.fields if x.first_box == first.first_box and x.second_box == first.second_box)
        if len(peers) != 1:
            continue
        second = peers[0]
        for a, b in zip(first.layers, second.layers):
            if a.value != b.value or len(a.poses) != 1 or len(b.poses) != 1:
                continue
            p, q = a.poses[0], b.poses[0]
            turns = (p.clockwise_quarters - q.clockwise_quarters) % 4
            dr = p.row_delta - q.row_delta if p.clockwise_quarters == q.clockwise_quarters == 0 else 0
            dc = p.col_delta - q.col_delta if p.clockwise_quarters == q.clockwise_quarters == 0 else 0
            if not (turns or dr or dc):
                continue
            motions.append((a.value, turns, dr, dc))
            if turns or not (dr or dc):
                continue
            other_values = tuple(x.value for x in first.layers if x.value != a.value)
            if len(other_values) != 1:
                continue
            other = other_values[0]
            base_values = (first.background_value, a.value, other, first.extra_value)
            for r in range(after.height):
                for c in range(after.width):
                    sr, sc = r - dr, c - dc
                    if not (0 <= sr < before.height and 0 <= sc < before.width):
                        continue
                    box = first.second_box
                    if any(box.top <= rr < box.top + box.height and box.left <= cc < box.left + box.width for rr, cc in ((r, c), (sr, sc))):
                        continue
                    moving, stationary, observed = before.rows[sr][sc], before.rows[r][c], after.rows[r][c]
                    if moving not in base_values or stationary not in base_values or observed not in base_values:
                        continue
                    overlap = moving in (a.value, first.extra_value) and stationary in (other, first.extra_value)
                    if overlap:
                        intersection_count += 1
                        mismatch_count += observed != first.extra_value
                    elif observed == first.extra_value:
                        mismatch_count += 1
    return LayerTransitionMeasure(action_ref, previous, current, point_value, marker, len(changes), tuple(motions), intersection_count, mismatch_count)


def measure_layer_probe_context(value: object) -> tuple[dict, object]:
    """Return descriptive context and a typed unique stable-patch occurrence."""
    sample = measure_layer_sample(value.current_observed_frame or value.frame)
    history = tuple(value.layer_transition_measurements)
    initial = dict(history[0].before.patch_hashes) if history else dict(sample.patch_hashes)
    hashes = dict(sample.patch_hashes)
    equal_patch_fields = tuple(f for f in sample.fields if initial.get(f.second_box) == hashes.get(f.second_box))
    unique = equal_patch_fields[0] if len(equal_patch_fields) == 1 else None
    return {
        "layer_pair_measurement_count": len(equal_patch_fields),
        "layer_pair_unique": unique is not None,
        "layer_intersection_observed_count": sum(x.intersection_observed_count for x in history),
        "layer_intersection_mismatch_count": sum(x.intersection_mismatch_count for x in history),
    }, unique


def measure_layer_candidate(value: object, field: LayerField | None, candidate: object) -> dict:
    if field is None or len(value.layer_transition_measurements) >= 32:
        return {}
    history = tuple(value.layer_transition_measurements)
    marked = tuple(x for x in history if x.point_value is not None and x.added_marker_value is not None)
    control_value = marked[-1].point_value if marked else None
    point_value = None
    frame = value.current_observed_frame or value.frame
    if candidate.point is not None:
        r, c = candidate.point
        point_value = frame.rows[r][c]
        box = field.second_box
        if box.top <= r < box.top + box.height and box.left <= c < box.left + box.width:
            return {}
    layer_value = point_value if candidate.point is not None else control_value
    matches = tuple(x for x in field.layers if x.value == layer_value)
    if len(matches) != 1 or len(matches[0].poses) != 1:
        return {}
    layer = matches[0]
    pose = layer.poses[0]
    moves = tuple(m for x in history for m in x.motions if m[0] == layer.value)
    rotation_actions = tuple(dict.fromkeys(x.action_ref for x in history if any(m[1] for m in x.motions)))
    translation = dict((ref, (dr, dc)) for ref, dr, dc in value.interface_action_translation_deltas)
    delta = translation.get(candidate.action_ref, (0, 0))
    reduces = pose.clockwise_quarters == 0 and (
        (delta[0] != 0 and delta[1] == 0 and pose.row_delta * delta[0] > 0) or
        (delta[1] != 0 and delta[0] == 0 and pose.col_delta * delta[1] > 0)
    )
    quanta = tuple((dr, dc) for x in history if x.action_ref == candidate.action_ref for v, turn, dr, dc in x.motions if turn == 0 and (dr or dc))
    no_overshoot = all(abs(dr) <= abs(pose.row_delta) and abs(dc) <= abs(pose.col_delta) for dr, dc in quanta)
    controlled_open = any(x.value == control_value and x.current_patch_difference_count > 0 for x in field.layers)
    return {
        "layer_support_measurement_present": True,
        "layer_palette_value": layer.value,
        "layer_extra_palette_value": field.extra_value,
        "layer_pose_observed_cell_count": pose.observed_cell_count,
        "layer_pose_quarters": pose.clockwise_quarters,
        "layer_row_delta": pose.row_delta,
        "layer_col_delta": pose.col_delta,
        "layer_patch_difference_count": layer.current_patch_difference_count,
        "layer_pattern_difference_count": field.exact_current_pattern_difference_count,
        "layer_marker_after_point_observed": bool(marked),
        "layer_candidate_controls_current_value": layer.value == control_value,
        "layer_current_marked_value_has_residual": controlled_open,
        "layer_candidate_translation_reduces_residual": reduces and no_overshoot,
        "layer_candidate_rotation_previously_observed": candidate.action_ref in rotation_actions,
        "layer_observed_rotation_action_count": len(rotation_actions),
        "layer_observed_motion_count": len(moves),
        "layer_candidate_is_point": candidate.point is not None,
    }
