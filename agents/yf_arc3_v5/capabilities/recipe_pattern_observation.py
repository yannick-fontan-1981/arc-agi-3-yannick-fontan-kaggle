"""Exact example/input correspondences; interpretation belongs to DRM.

The supported facet is a complete uniform input lattice with a displayed cue
and a smaller, same-shaped set of filled squares. No game, coordinates, input
order, or world-operation meaning is supplied here.
"""
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from math import gcd
from functools import reduce

from agents.yf_arc3_v5.capabilities.components import detect_components
from agents.yf_arc3_v5.capabilities.contracts import ComponentExtractionInput, FrameGrid
from agents.yf_arc3_v5.capabilities.periodic_correspondence import measure_uniform_lattices
from agents.yf_arc3_v5.logos.types import stable_digest


@dataclass(frozen=True)
class ExampleInputMatch:
    pattern_ref: str
    lattice_rows: tuple[int, ...]
    lattice_columns: tuple[int, ...]
    extent: int
    target_slots: tuple[tuple[int, int], ...]
    cue_slots: tuple[tuple[int, int], ...]
    entered_slots: tuple[tuple[int, int], ...]
    example_value: int
    entered_value: int | None
    example_bbox: tuple[int, int, int, int]

    def slot_at(self, point):
        if point is None:
            return None
        return next(((r, c) for r, top in enumerate(self.lattice_rows)
                     for c, left in enumerate(self.lattice_columns)
                     if top <= point[0] < top + self.extent
                     and left <= point[1] < left + self.extent), None)


def _normalized(points):
    rows, cols = sorted({r for r, _ in points}), sorted({c for _, c in points})
    if len(rows) < 2 or len(cols) < 2:
        return None
    row_pitch = reduce(gcd, (b-a for a, b in zip(rows, rows[1:])))
    col_pitch = reduce(gcd, (b-a for a, b in zip(cols, cols[1:])))
    return tuple(sorted(((r-rows[0])//row_pitch, (c-cols[0])//col_pitch) for r, c in points))


@lru_cache(maxsize=16)
def measure_example_input_matches(frame: FrameGrid, cue_value: int):
    components = detect_components(ComponentExtractionInput(frame=frame)).components
    groups = defaultdict(list)
    for component in components:
        box = component.bbox
        if box.height == box.width and component.area == box.height * box.width:
            groups[(box.height, frame.rows[box.top][box.left])].append(component)
    matches = []
    for field in measure_uniform_lattices(frame):
        for (extent, value), members in sorted(groups.items()):
            if extent >= field.extent or value == cue_value or len(members) < 3:
                continue
            if any(field.contains(member.bbox) for member in members):
                continue
            example = tuple((member.bbox.top, member.bbox.left) for member in members)
            normalized = _normalized(example)
            if normalized is None:
                continue
            # The complete target lattice, not just the normalized motif, is
            # retained. A translated motif within that lattice is a different
            # recipe. The palette is an observed example aspect, not its role.
            entered_values = (None, *sorted({v for row in field.values for v in row if v != cue_value}))
            for entered_value in entered_values:
                target = tuple((r, c) for r, row in enumerate(field.values)
                               for c, cell in enumerate(row) if cell in (cue_value, entered_value))
                cues = tuple((r, c) for r, c in target if field.values[r][c] == cue_value)
                if not cues or len(target) != len(members) or normalized != _normalized(target):
                    continue
                pattern_ref = 'pattern.example_input.' + stable_digest(
                    (len(field.rows), len(field.columns), target, value))[:24]
                matches.append(ExampleInputMatch(
                    pattern_ref, field.rows, field.columns, field.extent, target,
                    cues, tuple(slot for slot in target if slot not in cues), value, entered_value,
                    (min(r for r, _ in example), min(c for _, c in example),
                     max(r for r, _ in example)+extent-1, max(c for _, c in example)+extent-1)))
    if len(matches) > 16:
        raise ValueError('example/input correspondence bound exceeded; no alternatives truncated')
    return tuple(matches)


def _input_context(match):
    return stable_digest((match.pattern_ref, match.cue_slots, match.entered_slots, match.entered_value))[:24]


def measure_retained_recipe_widget_components(frame, components, retained_models):
    """Relocate a witnessed example/lattice pair even after the input clears."""
    groups = defaultdict(list)
    for component in components:
        box = component.bbox
        if box.height == box.width and component.area == box.height * box.width:
            groups[(box.height, component.value)].append(component)
    fields = measure_uniform_lattices(frame)
    refs, proofs = set(), set()
    pair_count = 0
    for model in retained_models:
        target = tuple(tuple(slot) for slot in model.get('target_slots', ()))
        shape = tuple(model.get('input_shape', ()))
        normalized = _normalized(target)
        if normalized is None or len(shape) != 2:
            continue
        candidates = []
        for (extent, value), members in sorted(groups.items()):
            if value != model.get('example_palette_value') or len(members) != len(target):
                continue
            if _normalized(tuple((m.bbox.top, m.bbox.left) for m in members)) != normalized:
                continue
            for field in fields:
                if ((len(field.rows), len(field.columns)) != shape or extent >= field.extent
                        or any(field.contains(m.bbox) for m in members)):
                    continue
                candidates.append((members, field))
        if len(candidates) != 1:
            continue  # No ambiguous prototype location becomes a world exclusion.
        members, field = candidates[0]
        refs.update(m.component_id for m in members)
        refs.update(c.component_id for c in components if field.contains(c.bbox))
        proofs.add(model['claim_ref'])
        pair_count += 1
    return {'retained_recipe_widget_pair_count': pair_count,
            'retained_recipe_widget_component_refs': tuple(sorted(refs)),
            'retained_recipe_widget_claim_refs': tuple(sorted(proofs))}


def measure_recipe_edit_context(before, action_ref, action_data, cue_value):
    if action_ref != 'ACTION6' or 'x' not in action_data or 'y' not in action_data:
        return {}
    matches = tuple(m for m in measure_example_input_matches(before, cue_value)
                    if m.slot_at((action_data['y'], action_data['x'])) in m.cue_slots)
    if len(matches) != 1:
        return {}
    match = matches[0]
    return {'recipe_edit_pattern_ref': match.pattern_ref,
            'recipe_edit_input_context_ref': _input_context(match),
            'recipe_edit_slot': match.slot_at((action_data['y'], action_data['x']))}


def measure_recipe_candidate(matches, candidate, retained_models=(), retained_edits=()):
    matching = tuple(match for match in matches if match.slot_at(candidate.point) in match.cue_slots)
    facts = {'example_input_candidate_match_count': len(matching)}
    if len(matching) == 1:
        match = matching[0]
        slot = match.slot_at(candidate.point)
        prior = tuple(edit for edit in retained_edits
                      if edit.get('pattern_ref') == match.pattern_ref and edit.get('slot') == slot)
        facts.update({
            'example_input_prior_slot_context_count': len(prior),
            'example_input_same_slot_context_count': sum(edit.get('input_context_ref') == _input_context(match) for edit in prior),
            'example_input_changed_context_claim_refs': tuple(edit['claim_ref'] for edit in prior if edit.get('input_context_ref') != _input_context(match)),
            'example_input_pattern_ref': match.pattern_ref,
            'example_input_target_slots': match.target_slots,
            'example_input_cue_count': len(match.cue_slots),
            'example_input_entered_count': len(match.entered_slots),
            'example_input_example_value': match.example_value,
            'example_input_edit_row': slot[0],
            'example_input_edit_column': slot[1],
            'example_input_retained_model_count': sum(
                model.get('pattern_ref') == match.pattern_ref for model in retained_models),
            'example_input_retained_model_claim_refs': tuple(
                model['claim_ref'] for model in retained_models
                if model.get('pattern_ref') == match.pattern_ref),
            'example_input_other_recipe_model_count': sum(
                model.get('pattern_ref') != match.pattern_ref for model in retained_models),
            'example_input_other_recipe_claim_refs': tuple(
                model['claim_ref'] for model in retained_models
                if model.get('pattern_ref') != match.pattern_ref),
        })
    return facts


def _changed_rectangles(before, after, excluded):
    def rectangles(frame):
        result = []
        for component in detect_components(ComponentExtractionInput(frame=frame)).components:
            box = component.bbox
            if component.area != box.height * box.width:
                continue
            if box.top == 0 or box.left == 0 or box.bottom == frame.height-1 or box.right == frame.width-1:
                continue
            if not (box.bottom < excluded[0] or box.top > excluded[2]
                    or box.right < excluded[1] or box.left > excluded[3]):
                continue
            result.append((frame.rows[box.top][box.left], box.top, box.left, box.height, box.width))
        return tuple(result)
    old, new = rectangles(before), rectangles(after)
    common = set(old) & set(new)
    return tuple(x for x in old if x not in common), tuple(x for x in new if x not in common)


def measure_recipe_transition(before, after, action_ref, action_data, cue_value):
    """Report input clearing and exact co-transforming rectangles in one packet.

    A translation describes endpoints only; this does not infer a landing law
    or the absence of an unseen path. Ambiguous component matches remain absent.
    """
    if action_ref != 'ACTION6' or 'x' not in action_data or 'y' not in action_data:
        return {}
    matches = tuple(m for m in measure_example_input_matches(before, cue_value)
                    if m.slot_at((action_data['y'], action_data['x'])) in m.cue_slots)
    if len(matches) != 1:
        return {}
    match = matches[0]
    if len(match.cue_slots) != 1 or len(match.entered_slots) != len(match.target_slots)-1:
        return {}
    values_after = {after.rows[r][c] for r in match.lattice_rows for c in match.lattice_columns}
    cleared = len(values_after) == 1 and match.entered_value not in values_after and cue_value not in values_after
    if not cleared:
        return {}
    excluded = (match.lattice_rows[0], match.lattice_columns[0],
                match.lattice_rows[-1]+match.extent-1, match.lattice_columns[-1]+match.extent-1)
    old, new = _changed_rectangles(before, after, excluded)
    pairs = []
    for first in old:
        possible = tuple(second for second in new
                         if first[0] == second[0] and first[3]*second[4] == first[4]*second[3])
        if len(possible) == 1 and sum(
            other[0] == possible[0][0] and other[3]*possible[0][4] == other[4]*possible[0][3]
            for other in old) == 1:
            pairs.append((first, possible[0]))
    if len(pairs) < 2:
        return {}
    # Every member of the measured group must share one exact uniform scale.
    from fractions import Fraction
    scales = {Fraction(second[3], first[3]) for first, second in pairs}
    if len(scales) != 1:
        return {}
    def box(which):
        cells = tuple(pair[which] for pair in pairs)
        top, left = min(c[1] for c in cells), min(c[2] for c in cells)
        bottom, right = max(c[1]+c[3] for c in cells), max(c[2]+c[4] for c in cells)
        if sum(c[3]*c[4] for c in cells) != (bottom-top)*(right-left):
            return None
        return top, left, bottom-top, right-left
    old_box, new_box = box(0), box(1)
    if old_box is None or new_box is None:
        return {}
    scale = next(iter(scales))
    if old_box[2]*scale != new_box[2] or old_box[3]*scale != new_box[3]:
        return {}
    if any((first[1]-old_box[0])*scale != second[1]-new_box[0]
           or (first[2]-old_box[1])*scale != second[2]-new_box[1] for first, second in pairs):
        return {}
    delta = (new_box[0]-old_box[0], new_box[1]-old_box[1])
    return {
        'recipe_application_packet_measured': True,
        'recipe_pattern_ref': match.pattern_ref,
        'recipe_target_slots': match.target_slots,
        'recipe_example_value': match.example_value,
        'recipe_entered_value': match.entered_value,
        'recipe_input_shape': (len(match.lattice_rows), len(match.lattice_columns)),
        'recipe_input_auto_clear_observed': True,
        'recipe_body_scale_numerator': scale.numerator,
        'recipe_body_scale_denominator': scale.denominator,
        'recipe_body_before_extent': old_box[2:],
        'recipe_body_after_extent': new_box[2:],
        'recipe_body_anchor_delta': delta,
        'recipe_body_palette_signature': tuple(sorted((first[0], Fraction(first[3]*first[4], old_box[2]*old_box[3]).numerator,
                                                       Fraction(first[3]*first[4], old_box[2]*old_box[3]).denominator)
                                                     for first, _ in pairs)),
        'recipe_body_disjoint_endpoints': (new_box[0]+new_box[2] <= old_box[0]
            or old_box[0]+old_box[2] <= new_box[0] or new_box[1]+new_box[3] <= old_box[1]
            or old_box[1]+old_box[3] <= new_box[1]),
    }
