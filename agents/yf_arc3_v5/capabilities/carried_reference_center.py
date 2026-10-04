"""Measure a carried shape and four stationary translated copies around a hole."""
from fractions import Fraction

from agents.yf_arc3_v5.logos.types import FrozenMap


def _signature(component):
    return tuple(sorted((r-component.bbox.top, c-component.bbox.left, component.value)
                        for r, c in component.pixels))


def measure_carried_reference_center(*, observations, tracking, context_epoch):
    if len(observations) > 64 or len(tracking.entities) > 128:
        raise ValueError('carried reference measurement bound exceeded')
    current = {item.entity_id:item.component for item in tracking.entities}
    if len(current) != len(tracking.entities):
        raise ValueError('duplicate carried reference identity')
    rows = tuple(row for row in observations if row['context_epoch'] == context_epoch
                 and row['extent_entity_ref'] in current)
    receivers = frozenset(row['extent_entity_ref'] for row in rows)
    carried = set.intersection(*(set(row['co_translated_entity_refs']) for row in rows)) if rows else set()
    carried = (carried - receivers) & current.keys()
    stationary = set.intersection(*(set(row['stationary_entity_refs']) for row in rows)) if rows else set()
    result = dict(context_epoch=context_epoch, source_term_refs=tuple(sorted(row['term_ref'] for row in rows)),
        observed_receiver_count=len(receivers), unique_carried_marker=len(carried)==1,
        multiple_observed_receivers=len(receivers)>1, four_identical_stationary_references=False,
        reference_is_symmetric_surround=False, center_origin_is_integral=False,
        center_copy_is_disjoint=False)
    if len(carried) != 1:
        return FrozenMap(result)
    marker_ref, = carried
    marker = current[marker_ref]
    signature = _signature(marker)
    references = tuple(sorted(ref for ref in stationary & current.keys()
                              if _signature(current[ref]) == signature))
    result.update(marker_entity_ref=marker_ref, reference_entity_refs=references)
    if len(references) != 4:
        return FrozenMap(result)
    result['four_identical_stationary_references'] = True
    origins = {(current[ref].bbox.top, current[ref].bbox.left) for ref in references}
    row = Fraction(sum(r for r,c in origins), 4)
    col = Fraction(sum(c for r,c in origins), 4)
    vectors = tuple((Fraction(r)-row, Fraction(c)-col) for r,c in sorted(origins))
    surround = (len(origins)==4 and (row,col) not in origins
        and {(2*row-r,2*col-c) for r,c in origins} == origins
        and any(ar*bc-ac*br != 0 for ar,ac in vectors for br,bc in vectors))
    result['reference_is_symmetric_surround'] = surround
    if not surround:
        return FrozenMap(result)
    if row.denominator != 1 or col.denominator != 1:
        return FrozenMap(result)
    result['center_origin_is_integral'] = True
    origin = (int(row),int(col))
    footprint = frozenset((origin[0]+r,origin[1]+c) for r,c,_value in signature)
    reference_pixels = frozenset(pixel for ref in references for pixel in current[ref].pixels)
    result.update(wanted_marker_top_left=origin, wanted_marker_support=tuple(sorted(footprint)),
        marker_relative_valued_pixels=signature, center_copy_is_disjoint=not bool(footprint & reference_pixels))
    return FrozenMap(result)
