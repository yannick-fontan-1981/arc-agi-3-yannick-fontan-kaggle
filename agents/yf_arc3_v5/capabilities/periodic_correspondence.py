"""Exact finite lattice/period measurements; no role or action selection.

Omissions are counterfactual measurements. All complete single-valued residue
maps are retained; DRM owns support thresholds and permission to test a value.
"""
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json

from agents.yf_arc3_v5.capabilities.components import detect_components
from agents.yf_arc3_v5.capabilities.contracts import ComponentExtractionInput, FrameGrid

MAX_AREA = 4096
MAX_COMPONENTS = 512
MAX_CELLS = 256
MAX_FIELDS = 4
MAX_DESCRIPTIONS = 256


class PeriodicMeasurementBoundExceeded(ValueError):
    pass


@dataclass(frozen=True)
class UniformLattice:
    rows: tuple[int, ...]
    columns: tuple[int, ...]
    extent: int
    values: tuple[tuple[int, ...], ...]

    def point(self, row, column):
        return (self.rows[row] + (self.extent - 1)//2,
                self.columns[column] + (self.extent - 1)//2)

    def contains(self, box):
        return (box.top >= self.rows[0] and box.bottom < self.rows[-1] + self.extent
                and box.left >= self.columns[0] and box.right < self.columns[-1] + self.extent)


@dataclass(frozen=True)
class ResidueMap:
    omitted_value: int | None
    row_period: int
    column_period: int
    values: tuple[tuple[int, ...], ...]
    minimum_concordance: int
    minimum_count_difference: int


def _digest(values):
    return sha256(json.dumps(values, sort_keys=True, separators=(',', ':')).encode('ascii')).hexdigest()


@lru_cache(maxsize=8)
def measure_uniform_lattices(frame: FrameGrid):
    if frame.height * frame.width > MAX_AREA:
        raise PeriodicMeasurementBoundExceeded('periodic raster area bound')
    components = detect_components(ComponentExtractionInput(frame=frame)).components
    if len(components) > MAX_COMPONENTS:
        raise PeriodicMeasurementBoundExceeded('periodic component bound')
    groups = defaultdict(list)
    for component in components:
        box = component.bbox
        if box.height == box.width and box.height > 1 and component.area == box.height * box.width:
            groups[box.height].append(component)
    fields = []
    for extent, members in sorted(groups.items()):
        rows = tuple(sorted({c.bbox.top for c in members}))
        columns = tuple(sorted({c.bbox.left for c in members}))
        if min(len(rows), len(columns)) < 2 or len(rows)*len(columns) != len(members):
            continue
        if (len({b-a for a,b in zip(rows, rows[1:])}) != 1
                or len({b-a for a,b in zip(columns, columns[1:])}) != 1):
            continue
        if len(members) > MAX_CELLS:
            raise PeriodicMeasurementBoundExceeded('periodic cell bound')
        values = tuple(tuple(frame.rows[r][c] for c in columns) for r in rows)
        fields.append(UniformLattice(rows, columns, extent, values))
        if len(fields) > MAX_FIELDS:
            raise PeriodicMeasurementBoundExceeded('periodic field bound')
    return tuple(fields)


@lru_cache(maxsize=16)
def measure_residue_maps(field: UniformLattice):
    result = []
    attempts = 0
    height, width = len(field.rows), len(field.columns)
    # Preserve exact intact motifs as alternatives to omission, including rare
    # but periodic symbols. Rarity must not erase a structurally repeated value.
    for omitted in (None, *sorted({v for row in field.values for v in row})):
        for ph in range(1, height//2 + 1):
            for pw in range(1, width//2 + 1):
                attempts += 1
                if attempts > MAX_DESCRIPTIONS:
                    raise PeriodicMeasurementBoundExceeded('periodic description bound')
                groups, absent = defaultdict(list), defaultdict(int)
                for r,row in enumerate(field.values):
                    for c,v in enumerate(row):
                        if v == omitted:
                            absent[r%ph,c%pw] += 1
                        else:
                            groups[r%ph,c%pw].append(v)
                # This is an exact partial function, not a majority selection.
                if len(groups) != ph*pw or any(len(set(v)) != 1 for v in (groups[k] for k in sorted(groups))):
                    continue
                mapped = tuple(tuple(groups[r%ph,c%pw][0] for c in range(width)) for r in range(height))
                result.append(ResidueMap(omitted, ph, pw, mapped,
                    min(len(groups[k]) for k in sorted(groups)),
                    min(len(groups[k])-absent[k] for k in sorted(groups))))
    return tuple(result)


def measure_periodic_context(frame, minimum_concordance=None, minimum_count_difference=None, quantity_palette_value=None):
    fields = measure_uniform_lattices(frame)
    all_maps = tuple(m for f in fields for m in measure_residue_maps(f))
    maps = tuple(m for m in all_maps
        if (minimum_concordance is None or m.minimum_concordance >= minimum_concordance)
        and (minimum_count_difference is None or m.minimum_count_difference >= minimum_count_difference))
    mappings = {m.values for m in maps}
    facts = {
        'periodic_field_count': len(fields), 'periodic_inventory_complete': True,
        'periodic_description_count': len(all_maps), 'periodic_distinct_mapping_count': len(mappings),
        'periodic_intact_description_count': sum(m.omitted_value is None for m in all_maps),
        'periodic_minimum_concordance': min((m.minimum_concordance for m in maps), default=0),
        'periodic_minimum_count_difference': min((m.minimum_count_difference for m in maps), default=0),
        'periodic_mapping_hash': _digest(next(iter(mappings))) if len(mappings) == 1 else '',
        'periodic_peripheral_quantity_component_count': len(_outside_rectangles(frame, fields[0], quantity_palette_value)) if len(fields) == 1 else 0,
    }
    return facts, fields, tuple(sorted(mappings))


@dataclass(frozen=True)
class PeriodicDelta:
    field: UniformLattice
    mapped_values: tuple[tuple[int, ...], ...]
    after_frame_hash: str
    point: tuple[int, int]
    before_value: int
    after_value: int
    mapped_value: int
    other_cell_changes: int
    residual_delta: int
    peripheral_quantity_delta: int


def _outside_rectangles(frame, field, palette_value):
    # Exact display-aspect measurement; DRM interprets a revisable error prior.
    return tuple(c for c in detect_components(ComponentExtractionInput(frame=frame)).components
                 if c.value == palette_value and not field.contains(c.bbox)
                 and c.area == c.bbox.height * c.bbox.width
                 and min(c.bbox.height, c.bbox.width) <= field.extent)


def _outside_quantity(frame, field, palette_value):
    return sum(c.area for c in _outside_rectangles(frame, field, palette_value))


def measure_periodic_delta(before, after, point, prior=None, quantity_palette_value=None,
                           minimum_concordance=None, minimum_count_difference=None):
    if point is None:
        return None
    _, fields, mappings = measure_periodic_context(before, minimum_concordance, minimum_count_difference)
    if len(fields) != 1:
        return None
    field, = fields
    # Carry the preceding immutable correspondence through a changed cell.
    if prior is not None and prior.after_frame_hash == _digest(before.rows):
        mapped = prior.mapped_values
    elif len(mappings) == 1:
        mapped, = mappings
    else:
        return None
    if field.values == mapped:
        # An intact correspondence has no residual correction to reconcile.
        return None
    indices = [(r,c) for r in range(len(field.rows)) for c in range(len(field.columns)) if field.point(r,c) == point]
    if len(indices) != 1:
        return None
    r,c = indices[0]
    if len(mapped) != len(field.rows) or len(mapped[0]) != len(field.columns):
        return None
    after_values = tuple(tuple(after.rows[y][x] for x in field.columns) for y in field.rows)
    changes = [(y,x) for y,row in enumerate(field.values) for x,v in enumerate(row) if v != after_values[y][x]]
    residual_delta = sum(after_values[y][x] != mapped[y][x] for y,row in enumerate(mapped) for x,_ in enumerate(row)) - sum(field.values[y][x] != mapped[y][x] for y,row in enumerate(mapped) for x,_ in enumerate(row))
    return PeriodicDelta(field, mapped, _digest(after.rows), point, field.values[r][c], after_values[r][c], mapped[r][c],
        sum(p != (r,c) for p in changes), residual_delta,
        _outside_quantity(after, field, quantity_palette_value)-_outside_quantity(before, field, quantity_palette_value))


def measure_periodic_candidate(frame, fields, mappings, candidate, prior):
    facts = {
        'periodic_point_in_field': any(candidate.point == f.point(r,c)
            for f in fields for r in range(len(f.rows)) for c in range(len(f.columns))),
        'periodic_point_differs_from_mapping': False, 'periodic_point_mapped_value': -1,
        'periodic_transition_present': prior is not None,
        'periodic_point_is_prior_point': prior is not None and candidate.point == prior.point,
        'periodic_prior_point_previously_matched': prior is not None and prior.before_value == prior.mapped_value,
        'periodic_prior_point_reached_mapping': prior is None or prior.after_value == prior.mapped_value,
        'periodic_prior_other_cell_changes': prior.other_cell_changes if prior else 0,
        'periodic_prior_residual_delta': prior.residual_delta if prior else 0,
        'periodic_prior_quantity_delta': prior.peripheral_quantity_delta if prior else 0,
        'periodic_prior_quantity_opposite_delta': prior is not None and prior.residual_delta * prior.peripheral_quantity_delta < 0,
    }
    if len(fields) == 1 and len(mappings) == 1:
        field, = fields
        mapped, = mappings
        for r,row in enumerate(field.values):
            for c,v in enumerate(row):
                if candidate.point == field.point(r,c):
                    facts['periodic_point_differs_from_mapping'] = v != mapped[r][c]
                    facts['periodic_point_mapped_value'] = mapped[r][c]
    if prior is not None and candidate.point == prior.point:
        facts['periodic_point_mapped_value'] = prior.mapped_value
    return facts
