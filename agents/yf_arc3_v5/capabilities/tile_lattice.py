"""Exact raster geometry of repeated inset or edge-adjacent rectangles.

No palette value is a role. A lattice is a measurement, not traversability.
"""

from collections import defaultdict
from math import gcd

from .components import detect_components
from .contracts import ComponentExtractionInput, FrameGrid


def exact_joint_tile_lattices(
    frame: FrameGrid, *, maximum_pitch: int,
) -> tuple[tuple[int, int, int, int], ...]:
    """Measure zero-gap cells from equal full rectangles of several values.

    No individual value defines the lattice: adjacent rectangles of different
    values expose its boundaries. Keep the existing nine-witness, two-axis,
    half-occupied lattice contract. A border, gap or stagger is not a joint.
    """
    groups: dict[tuple[int, int], dict[tuple[int, int], int]] = defaultdict(dict)
    for component in detect_components(ComponentExtractionInput(frame=frame)).components:
        box = component.bbox
        if (2 <= box.height <= maximum_pitch and 2 <= box.width <= maximum_pitch
                and component.area == box.height * box.width):
            groups[(box.height, box.width)][(box.top, box.left)] = component.value
    geometries: set[tuple[int, int, int, int]] = set()
    for (height, width), positions in sorted(groups.items()):
        if len(positions) < 9 or len(set(positions.values())) < 2:
            continue
        rows = sorted({row for row, _ in positions})
        columns = sorted({col for _, col in positions})
        if len(rows) < 3 or len(columns) < 3:
            continue
        if (gcd(*(b - a for a, b in zip(rows, rows[1:]))) != height
                or gcd(*(b - a for a, b in zip(columns, columns[1:]))) != width):
            continue
        capacity = (1 + (rows[-1] - rows[0]) // height) * (
            1 + (columns[-1] - columns[0]) // width
        )
        if len(positions) * 2 < capacity:
            continue
        if not all(any((row + dy, col + dx) in positions for row, col in positions)
                   for dy, dx in ((height, 0), (0, width))):
            continue
        geometries.add((rows[0] % height, columns[0] % width, height, width))
    return tuple(sorted(geometries)) if len(geometries) <= 3 else ()


def exact_bordered_tile_lattices(
    frame: FrameGrid, *, maximum_pitch: int,
) -> tuple[tuple[int, int, int, int], ...]:
    """Return (row offset, column offset, height, width), at most three.

    Reuse the regular multicell lattice contract: nine complete witnesses,
    three rows/columns and at least half of the bounded lattice occupied.
    Every witness is an exact uniform rectangle with a uniform surrounding
    inset. Offsets describe full tiles, including the top/left half-open border.
    Staggered textures, isolated bodies and one-axis indicators do not qualify.
    Ambiguous geometries stay alternatives; more than three fails closed.
    """
    groups: dict[tuple[int, int, int, int], set[tuple[int, int]]] = defaultdict(set)
    for component in detect_components(ComponentExtractionInput(frame=frame)).components:
        box = component.bbox
        if (box.height < 2 or box.width < 2 or
            box.height >= maximum_pitch or box.width >= maximum_pitch or
            component.area != box.height * box.width or
            box.top < 1 or box.left < 1 or
            box.bottom >= frame.height - 1 or box.right >= frame.width - 1):
            continue
        border = (
            {frame.rows[box.top - 1][x] for x in range(box.left, box.right + 1)}
            | {frame.rows[box.bottom + 1][x] for x in range(box.left, box.right + 1)}
            | {frame.rows[y][box.left - 1] for y in range(box.top, box.bottom + 1)}
            | {frame.rows[y][box.right + 1] for y in range(box.top, box.bottom + 1)}
        )
        if len(border) == 1:
            groups[(box.height, box.width, component.value, next(iter(border)))].add(
                (box.top, box.left)
            )
    geometries: set[tuple[int, int, int, int]] = set()
    for (height, width, _interior, border), positions in sorted(groups.items()):
        if len(positions) < 9:
            continue
        rows = sorted({row for row, _ in positions})
        columns = sorted({col for _, col in positions})
        if len(rows) < 3 or len(columns) < 3:
            continue
        row_pitch = gcd(*(b - a for a, b in zip(rows, rows[1:])))
        col_pitch = gcd(*(b - a for a, b in zip(columns, columns[1:])))
        if not (height < row_pitch <= maximum_pitch and width < col_pitch <= maximum_pitch):
            continue
        capacity = (1 + (rows[-1] - rows[0]) // row_pitch) * (
            1 + (columns[-1] - columns[0]) // col_pitch
        )
        if len(positions) * 2 < capacity:
            continue
        top_inset = (row_pitch - height + 1) // 2
        left_inset = (col_pitch - width + 1) // 2
        # Every reconstructed full tile must actually have this uniform inset;
        # equal center spacing alone is insufficient evidence of a tile.
        complete = True
        for row, col in sorted(positions):
            top, left = row - top_inset, col - left_inset
            if top < 0 or left < 0 or top + row_pitch > frame.height or left + col_pitch > frame.width:
                complete = False
                break
            if any(
                frame.rows[y][x] != border
                for y in range(top, top + row_pitch)
                for x in range(left, left + col_pitch)
                if not (row <= y < row + height and col <= x < col + width)
            ):
                complete = False
                break
        if complete:
            geometries.add(((rows[0] - top_inset) % row_pitch,
                            (columns[0] - left_inset) % col_pitch,
                            row_pitch, col_pitch))
    return tuple(sorted(geometries)) if len(geometries) <= 3 else ()
