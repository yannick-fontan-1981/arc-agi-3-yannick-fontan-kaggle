"""Bounded raster contours, opposite boundary patches and replacement deltas.

All returned fields are geometric counts/coordinates. DRM owns compatibility,
selection, transformation and destination interpretations.
"""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from agents.yf_arc3_v5.capabilities.components import detect_components
from agents.yf_arc3_v5.capabilities.contracts import ComponentExtractionInput

MAX_AREA = 4096
MAX_PARTS = 512
MAX_BOXES = 32


def component_has_complete_perimeter(component):
    """Measure an outer perimeter independently of attached interior pixels."""
    h, w = component.bbox.height, component.bbox.width
    if min(h, w) < 3:
        return False
    pixels = set(component.relative_pixels)
    return len(pixels) < h * w and all(
        (y, x) in pixels for y in range(h) for x in range(w)
        if y in (0, h-1) or x in (0, w-1))


@dataclass(frozen=True)
class RasterContour:
    box: tuple[int, int, int, int]
    value: int
    normal: tuple[int, int] = (0, 0)
    patch_value: int = -1
    gap: tuple[int, ...] = ()

    @property
    def center(self):
        t, l, b, r = self.box
        return ((t + b) // 2, (l + r) // 2)


@dataclass(frozen=True)
class ContourDelta:
    two_removed_one_added_boxes: tuple = ()
    one_removed_many_added_boxes: tuple = ()
    two_removed_one_added_count: int = 0
    one_removed_many_added_count: int = 0
    added_closed_contour_identities: tuple = ()


def _inside(inner, outer):
    return (outer[0] < inner[0] and outer[1] < inner[1]
            and inner[2] < outer[2] and inner[3] < outer[3])


def _contains(box, point):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


@lru_cache(maxsize=8)
def measure_contours(frame):
    if frame.height * frame.width > MAX_AREA:
        raise ValueError("contour raster bound exceeded")
    parts = detect_components(ComponentExtractionInput(frame=frame)).components
    if len(parts) > MAX_PARTS:
        raise ValueError("contour component bound exceeded")
    result = []
    for part in parts:
        q = part.bbox
        h, w = q.height, q.width
        if min(h, w) < 3:
            continue
        pixels = set(part.relative_pixels)
        perimeter = {(y, x) for y in range(h) for x in range(w)
                     if y in (0, h - 1) or x in (0, w - 1)}
        if component_has_complete_perimeter(part):
            result.append(RasterContour((q.top, q.left, q.bottom, q.right), part.value))
            continue
        if not pixels <= perimeter:
            continue
        box = (q.top, q.left, q.bottom, q.right)
        missing = perimeter - pixels
        if not missing:
            result.append(RasterContour(box, part.value))
            continue
        sides = [((0, -1), [(y, x) for y, x in missing if x == 0 and 0 < y < h - 1]),
                 ((0, 1), [(y, x) for y, x in missing if x == w - 1 and 0 < y < h - 1]),
                 ((-1, 0), [(y, x) for y, x in missing if y == 0 and 0 < x < w - 1]),
                 ((1, 0), [(y, x) for y, x in missing if y == h - 1 and 0 < x < w - 1])]
        for normal, side in sides:
            if set(side) != missing:
                continue
            gap = tuple(sorted(y if normal[1] else x for y, x in side))
            if gap != tuple(range(gap[0], gap[-1] + 1)):
                continue
            positions = [(q.top + y, q.left + x) for y, x in side]
            outside = [(y + normal[0], x + normal[1]) for y, x in positions]
            if not all(0 <= y < frame.height and 0 <= x < frame.width for y, x in outside):
                continue
            values = {frame.rows[y][x] for y, x in positions + outside}
            if len(values) == 1:
                result.append(RasterContour(box, part.value, normal, next(iter(values)), gap))
    if len(result) > MAX_BOXES:
        return ()  # abstain on the whole relation; never expose a partial inventory
    return tuple(result)


def opposite(a, b):
    return (a.normal != (0, 0) and a.normal == tuple(-x for x in b.normal)
            and a.value == b.value and a.patch_value == b.patch_value and a.gap == b.gap
            and a.box[2] - a.box[0] == b.box[2] - b.box[0]
            and a.box[3] - a.box[1] == b.box[3] - b.box[1])


def same_partitioned_enclosure(frame, a, b, closed):
    """Exact two-cell enclosure geometry; no control or operand role is inferred."""
    for left in closed:
        if not _inside(a.box, left.box):
            continue
        for right in closed:
            if left.value != right.value or not _inside(b.box, right.box):
                continue
            lt, ll, lb, lr = left.box
            rt, rl, rb, rr = right.box
            horizontal = lt == rt and lb == rb and rl == lr + 2 and lr - ll == rr - rl
            vertical = ll == rl and lr == rr and rt == lb + 2 and lb - lt == rb - rt
            if not (horizontal or vertical):
                continue
            outer = (min(lt, rt) - 1, min(ll, rl) - 1,
                     max(lb, rb) + 1, max(lr, rr) + 1)
            for enclosure in closed:
                if enclosure.box != outer:
                    continue
                divider = (
                    ((y, lr + 1) for y in range(lt, lb + 1)) if horizontal
                    else ((lb + 1, x) for x in range(ll, lr + 1))
                )
                values = {frame.rows[y][x] for y, x in divider}
                if len(values) == 1 and left.value not in values:
                    return True
    return False


@lru_cache(maxsize=8)
def measure_multivalue_boxes(frame):
    """Exact 8-connected non-modal regions; containment preserves distinctness."""
    if frame.height * frame.width > MAX_AREA:
        raise ValueError("contour raster bound exceeded")
    counts = Counter(v for row in frame.rows for v in row)
    maximum = max(counts[v] for v in sorted(counts))
    modes = [v for v, n in sorted(counts.items()) if n == maximum]
    if len(modes) != 1:
        return ()
    background, = modes
    todo = {(y, x) for y, row in enumerate(frame.rows) for x, v in enumerate(row) if v != background}
    boxes = []
    while todo:
        seed = min(todo)
        todo.remove(seed)
        pending = [seed]
        points = []
        while pending:
            y, x = pending.pop()
            points.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    p = y + dy, x + dx
                    if p in todo:
                        todo.remove(p)
                        pending.append(p)
        boxes.append((min(y for y, x in points), min(x for y, x in points),
                      max(y for y, x in points), max(x for y, x in points)))
        if len(boxes) > MAX_PARTS:
            raise ValueError("contour region bound exceeded")
    boxes = tuple(boxes)
    if len(boxes) > MAX_BOXES:
        return ()  # no incomplete region inventory
    return boxes


def advance_contour_delta(before, after, prior=None):
    prior = prior or ContourDelta()
    old, new = measure_contours(before), measure_contours(after)
    old_closed_ids = measure_centered_enclosure_identities(before)
    added_closed_ids = tuple(identity for identity in measure_centered_enclosure_identities(after)
                            if identity not in old_closed_ids
                            or identity in prior.added_closed_contour_identities)
    removed_open = [a for a in old if a.normal != (0, 0) and not any(a.box == b.box for b in new)]
    added_closed = [a for a in new if a.normal == (0, 0) and not any(a.box == b.box for b in old)]
    new_region_boxes = measure_multivalue_boxes(after)
    added_closed = [a for a in added_closed if a.box in new_region_boxes]
    retained_closed = tuple(box for box in prior.two_removed_one_added_boxes
                            if any(a.box == box for a in new))
    pair_delta = (len(removed_open) == 2 and opposite(*removed_open) and len(added_closed) == 1)
    if pair_delta:
        retained_closed += (added_closed[0].box,)
    disappeared = [box for box in prior.two_removed_one_added_boxes if box not in retained_closed]
    old_boxes, new_boxes = measure_multivalue_boxes(before), measure_multivalue_boxes(after)
    added = tuple(b for b in new_boxes if b not in old_boxes)
    multiple_delta = len(disappeared) == 1 and len(added) > 1
    counts = Counter(v for row in after.rows for v in row)
    background = max(sorted(counts), key=counts.get)
    retained_multiple = tuple(b for b in prior.one_removed_many_added_boxes
                              if any(after.rows[y][x] != background
                                     for y in range(b[0],b[2]+1) for x in range(b[1],b[3]+1)))
    if multiple_delta:
        retained_multiple += added
    if max(len(retained_closed), len(retained_multiple)) > MAX_BOXES:
        raise ValueError("contour delta bound exceeded")
    return ContourDelta(retained_closed, retained_multiple,
                        min(MAX_BOXES, prior.two_removed_one_added_count + int(pair_delta)),
                        min(MAX_BOXES, prior.one_removed_many_added_count + int(multiple_delta)),
                        added_closed_ids)


@lru_cache(maxsize=8)
def measure_centered_enclosure_identities(frame):
    """Complete perimeters around a component with equal positive offsets."""
    closed = tuple(c for c in measure_contours(frame) if c.normal == (0, 0))
    parts = detect_components(ComponentExtractionInput(frame=frame)).components
    identities = []
    for c in closed:
        t, l, b, r = c.box
        for p in parts:
            q = p.bbox
            gaps = (q.top-t, q.left-l, b-q.bottom, r-q.right)
            if min(q.height, q.width) >= 2 and min(gaps) > 0 and len(set(gaps)) == 1:
                identities.append((*c.box, c.value))
                break
    return tuple(identities)


def measure_contour_context(frame, delta, additions):
    additions = delta.added_closed_contour_identities
    contours = measure_contours(frame)
    opened = tuple(a for a in contours if a.normal != (0, 0))
    framed = tuple(a for a in contours if a.normal == (0, 0) and (*a.box, a.value) in additions)
    pairs = tuple((a, b) for i, a in enumerate(opened) for b in opened[i+1:] if opposite(a, b))
    return {
        "opposite_or_replaced_contour_present": bool(pairs or delta.two_removed_one_added_boxes or delta.one_removed_many_added_boxes),
        "opposite_contour_pair_count": len(pairs),
        "opposite_contour_added_enclosure_count": sum(any(_inside(a.box, f.box) for a in opened) for f in framed),
        "contour_two_removed_one_added_count": delta.two_removed_one_added_count,
        "contour_one_removed_many_added_count": delta.one_removed_many_added_count,
    }


@lru_cache(maxsize=8)
def _most_frequent_raster_value(frame):
    """Exact frequency measurement; ties retain raster encounter order."""
    counts = Counter(v for row in frame.rows for v in row)
    return max(counts, key=counts.get)


def measure_contour_candidate(frame, delta, additions, candidate):
    additions = delta.added_closed_contour_identities
    contours = measure_contours(frame)
    opened = tuple(a for a in contours if a.normal != (0, 0))
    closed = tuple(a for a in contours if a.normal == (0, 0))
    framed = tuple(a for a in closed if (*a.box, a.value) in additions)
    point = candidate.point
    on_open = tuple(a for a in opened if point is not None and _contains(a.box, point))
    enclosed_open = tuple(a for a in opened if any(_inside(a.box, f.box) for f in framed))
    on_multiple = tuple(b for b in delta.one_removed_many_added_boxes if point is not None and _contains(b, point))
    enclosed_multiple = tuple(b for b in delta.one_removed_many_added_boxes if any(_inside(b, f.box) for f in framed))
    # Compare exact non-background colour occurrences under translation. Values
    # occurring in multiple products remain alternative aspects, not fixed roles.
    background = (
        _most_frequent_raster_value(frame)
        if frame.height * frame.width <= MAX_AREA
        else _most_frequent_raster_value.__wrapped__(frame)
    )
    def raster_values(box):
        t, l, b, r = box
        return {frame.rows[y][x] for y in range(t, b+1) for x in range(l, r+1)} - {background}
    product_values = [raster_values(b) for b in delta.one_removed_many_added_boxes]
    occurrence = Counter(v for vs in product_values for v in vs)
    def matches(box, enclosure):
        cy, cx = (box[0]+box[2])//2, (box[1]+box[3])//2
        ty, tx = enclosure.center
        unique_values = {v for v in raster_values(box) if occurrence[v] == 1}
        for v in sorted(unique_values):
            retained = {(y-cy,x-cx) for y in range(box[0],box[2]+1) for x in range(box[1],box[3]+1)
                        if frame.rows[y][x] == v}
            other = {(y-cy,x-cx) for y in range(box[0],box[2]+1) for x in range(box[1],box[3]+1)
                     if frame.rows[y][x] not in (v,background)}
            interior = {(y-ty,x-tx) for y in range(enclosure.box[0]+1,enclosure.box[2])
                        for x in range(enclosure.box[1]+1,enclosure.box[3]) if frame.rows[y][x] == v}
            if retained <= interior <= retained | other:
                return True
        return False
    destinations = tuple(f for f in closed if not any(_inside(b, f.box) for b in delta.one_removed_many_added_boxes)
                         and (*f.box, f.value) not in additions
                         and not any(_inside(f.box, a.box) for a in framed)
                         and f.box not in delta.two_removed_one_added_boxes)
    # Multiple nested contours can describe one candidate enclosure. Keep its
    # outer extent without merging the separate regions inside that extent.
    destinations = tuple(f for f in destinations if not any(_inside(f.box, g.box) for g in destinations))
    matching = tuple(f for f in destinations if len(enclosed_multiple) == 1 and matches(enclosed_multiple[0], f))
    product_match_count = sum(len([f for f in destinations if matches(b, f)]) == 1 for b in on_multiple)
    return {
        "contour_point_opposite_peer_count": sum(sum(opposite(a,b) for b in opened) for a in on_open),
        "contour_point_opposite_partitioned_enclosure_count": sum(
            same_partitioned_enclosure(frame, a, b, closed)
            or same_partitioned_enclosure(frame, b, a, closed)
            for a in on_open for b in opened if opposite(a, b)
        ),
        "contour_point_opposite_enclosed_count": sum(opposite(a,b) for a in on_open for b in enclosed_open),
        "contour_point_enclosed_open_count": sum(a in enclosed_open for a in on_open),
        "contour_point_two_removed_one_added_count": sum(point is not None and _contains(b, point) for b in delta.two_removed_one_added_boxes),
        "contour_point_product_unique_frame_match_count": product_match_count,
        "contour_added_product_enclosure_count": len(enclosed_multiple),
        "contour_point_unique_matching_frame_count": int(len(matching) == 1 and point is not None and _contains(matching[0].box, point)),
    }
