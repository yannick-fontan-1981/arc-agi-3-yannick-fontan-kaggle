"""Exact bounded raster relations around a three-branch symbol.

Left/right groups, concentric pairs and action-conditioned changes only.
Meaning as a recipe, inventory, selection or request belongs to DRM.
"""

from dataclasses import dataclass
from functools import lru_cache
from collections import Counter

from agents.yf_arc3_v5.capabilities.components import detect_components
from agents.yf_arc3_v5.capabilities.concentric_frames import measure_concentric_frames
from agents.yf_arc3_v5.capabilities.contracts import ComponentExtractionInput, FrameGrid

MAX_AREA = 4096
MAX_COMPONENTS = 512
MAX_ROWS = 16
MAX_GROUP_ITEMS = 4


@dataclass(frozen=True)
class DiagramRow:
    left_values: tuple[int, ...]
    right_values: tuple[int, ...]
    branch_value: int
    symbol_extent: int


@lru_cache(maxsize=16)
def _components(frame):
    if frame.height * frame.width > MAX_AREA:
        raise ValueError("symbol diagram area bound exceeded")
    result = detect_components(ComponentExtractionInput(frame=frame)).components
    if len(result) > MAX_COMPONENTS:
        raise ValueError("symbol diagram component bound exceeded")
    return result


def _rectangular_perimeter(component):
    b = component.bbox
    return (min(b.height, b.width) >= 3
            and component.area == 2 * (b.height + b.width) - 4
            and all(r in (0, b.height - 1) or c in (0, b.width - 1)
                    for r, c in component.relative_pixels))


def _solid_square(component):
    return (component.bbox.height == component.bbox.width > 1
            and component.area == component.bbox.height ** 2)


def _three_branch_to_left(component):
    b = component.bbox
    if b.height < 3 or b.height % 2 == 0 or b.width < 2:
        return False
    expected = {(r, b.width - 1) for r in range(b.height)}
    expected.update((b.height // 2, c) for c in range(b.width))
    return set(component.relative_pixels) == expected


@lru_cache(maxsize=8)
def measure_diagram_rows(frame: FrameGrid) -> tuple[DiagramRow, ...]:
    components = _components(frame)
    borders = tuple(c for c in components if min(c.bbox.height, c.bbox.width) >= 3
        and all(frame.rows[r][col] == c.value
                for r in range(c.bbox.top, c.bbox.bottom + 1)
                for col in range(c.bbox.left, c.bbox.right + 1)
                if r in (c.bbox.top, c.bbox.bottom) or col in (c.bbox.left, c.bbox.right)))
    result = []
    for border in borders:
        box = border.bbox
        # Removing the measured perimeter separates a touching same-value glyph
        # without interpreting that value as active, inactive or a control role.
        interior = FrameGrid(rows=tuple(tuple(row[box.left + 1:box.right]) for row in frame.rows[box.top + 1:box.bottom]))
        local = _components(interior)
        squares = tuple(c for c in local if _solid_square(c))
        for branch in local:
            if not _three_branch_to_left(branch):
                continue
            bb = branch.bbox
            top, left_pos = box.top + 1 + bb.top, box.left + 1 + bb.left
            bottom, right_pos = box.top + 1 + bb.bottom, box.left + 1 + bb.right
            if any(d.component_id != border.component_id and box.top <= d.bbox.top
                   and box.bottom >= d.bbox.bottom and box.left <= d.bbox.left
                   and box.right >= d.bbox.right and d.bbox.top < top < bottom < d.bbox.bottom
                   and d.bbox.left < left_pos < right_pos < d.bbox.right for d in borders):
                continue
            aligned = tuple(c for c in squares if c.bbox.top == bb.top and c.bbox.bottom == bb.bottom)
            left = tuple(sorted((c for c in aligned if c.bbox.right < bb.left), key=lambda c: c.bbox.left))
            right = tuple(sorted((c for c in aligned if c.bbox.left > bb.right), key=lambda c: c.bbox.left))
            if not left or not right:
                continue
            if max(len(left), len(right)) > MAX_GROUP_ITEMS:
                raise ValueError("symbol diagram group bound exceeded")
            result.append(DiagramRow(tuple(c.value for c in left), tuple(c.value for c in right), branch.value, bb.height))
            if len(result) > MAX_ROWS:
                raise ValueError("symbol diagram row bound exceeded")
    return tuple(result)


@lru_cache(maxsize=16)
def measure_equal_row_pairs(frame):
    groups = {}
    for f in measure_concentric_frames(frame):
        if f.maximum_equal_padding != 1:
            continue
        key = (f.top, f.bottom, f.right - f.left, f.outer_value)
        groups.setdefault(key, []).append(f)
    return tuple(tuple(sorted(groups[key], key=lambda f: f.left)) for key in sorted(groups) if len(groups[key]) == 2)


def measure_diagram_transition(before, after):
    rows = measure_diagram_rows(before)
    squares_after = tuple(c for c in _components(after) if _solid_square(c))
    squares_before = tuple(c for c in _components(before) if _solid_square(c))
    changed = tuple(c for c in squares_after if not any(c.bbox == d.bbox and c.value == d.value for d in squares_before))
    matches = 0
    for pair in measure_equal_row_pairs(before):
        values = tuple(before.rows[f.center[0]][f.center[1]] for f in pair)
        after_values = tuple(after.rows[f.center[0]][f.center[1]] for f in pair)
        for row in rows:
            appeared = Counter(c.value for c in changed if c.bbox.height == row.symbol_extent
                and not any(f.top <= c.bbox.top <= c.bbox.bottom <= f.bottom
                            and f.left <= c.bbox.left <= c.bbox.right <= f.right for f in pair))
            required = Counter(row.right_values)
            if (values == row.left_values and len(set(after_values)) == 1
                    and all(a != b for a, b in zip(values, after_values))
                    and all(appeared[v] >= required[v] for v in sorted(required))):
                matches += 1
    return matches


def measure_diagram_context(frame, transition_match_count):
    rows = measure_diagram_rows(frame)
    return {
        "three_branch_diagram_row_count": len(rows),
        "three_branch_pair_left_row_count": sum(len(r.left_values) == 2 for r in rows),
        "diagram_left_removal_right_appearance_count": transition_match_count,
    }


def measure_diagram_candidate(frame, candidate, white_value):
    rows = measure_diagram_rows(frame)
    point = candidate.point
    if point is None:
        return {}
    components = _components(frame)
    at_center = tuple(c for c in components if ((c.bbox.top + c.bbox.bottom) // 2,
        (c.bbox.left + c.bbox.right) // 2) == point)
    extents = {r.symbol_extent for r in rows}
    solids = tuple(c for c in at_center if _solid_square(c) and c.bbox.height in extents)
    tight = tuple(f for f in measure_concentric_frames(frame) if f.outer_value == white_value and f.maximum_equal_padding == 1)
    tight_center_values = tuple(frame.rows[f.center[0]][f.center[1]] for f in tight)
    hollow = tuple(c for c in at_center if _rectangular_perimeter(c) and c.bbox.height == c.bbox.width and c.bbox.height in extents)
    pairs = measure_equal_row_pairs(frame)
    matching_pairs = tuple(pair for pair in pairs if any(tuple(frame.rows[f.center[0]][f.center[1]] for f in pair) == row.left_values for row in rows))
    return {
        "diagram_point_full_symbol_count": len(solids),
        "diagram_point_branch_value_square_count": sum(any(c.value == r.branch_value and c.bbox.height == r.symbol_extent for r in rows) for c in solids),
        "diagram_point_equal_pair_center_count": sum(any(f.center == point for f in pair) for pair in pairs),
        "diagram_equal_pair_matches_left_group_count": len(matching_pairs),
        "diagram_tight_white_center_count": len(tight),
        "diagram_point_hollow_matches_tight_value_count": sum(c.value in tight_center_values for c in hollow),
    }
