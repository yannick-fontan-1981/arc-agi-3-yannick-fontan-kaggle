"""Finite horizontal configuration witnesses under a supplied flow operator.

Morphological inputs and computed traces are conditional, not established roles
or physical laws. No action, semantic commitment or official result is produced.
"""
from collections import Counter
from itertools import product

from agents.yf_arc3_v5.logos.types import stable_digest


def project_parallel_accretion(width, height, start, sign, rectangles, ports, *, limit=1024):
    """Joint ticks: translate, branch around masks, capture or stop on trace.

    This domain has parallel fronts only. Conflicting incidences and unmodelled
    contacts have no certificate. A bound is never a terminal result.
    """
    occupied = {(r, x) for r, left, right in rectangles for x in range(left, right+1)}
    walls = {p for _void, wall in ports for p in wall}
    voids = {p for p, _wall in ports}
    trace = {start}
    active = {start}
    captures = set()
    contacts = set()
    ticks = 0
    while active and ticks < limit:
        ticks += 1
        additions = set()
        for r, x in sorted(active):
            nxt = (r+sign, x)
            if (r, x) in voids:
                captures.add((r, x))
                continue
            if not (0 <= nxt[0] < height and 0 <= nxt[1] < width):
                return None
            if nxt in trace:
                contacts.add(nxt)
                continue
            if nxt in occupied:
                successors = ((r, x-1), (r, x+1))
            elif nxt in walls:
                return None
            else:
                successors = (nxt,)
            for successor in successors:
                if not (0 <= successor[0] < height and 0 <= successor[1] < width):
                    return None
                if successor in trace:
                    contacts.add(successor)
                elif successor not in occupied and successor not in walls:
                    additions.add(successor)
        trace.update(additions)
        active = additions
    if active or captures != voids:
        return None
    return {"trace":tuple(sorted(trace)), "contacts":tuple(sorted(contacts)),
            "captures":tuple(sorted(captures)), "ticks":ticks}


def measure_oriented_contour_network(value):
    contract = value.oriented_contour_network_contract
    empty = {"oriented_network_present":False}
    if contract.get("operator") != "parallel_accretion_around_horizontal_rectangles":
        return empty
    maximum = int(contract.get("maximum_configurations", 0))
    witness_limit = int(contract.get("maximum_witnesses", 0))
    if not (1 <= maximum <= 4096 and 1 <= witness_limit <= 3):
        return empty
    components = value.components.components
    # Full long rectangles of one thickness, independent of palette or length.
    rectangles = tuple(c for c in components if c.bbox.width >= 2*c.bbox.height
                       and c.bbox.height >= 2 and c.area == c.bbox.width*c.bbox.height
                       and not c.touches_frame_boundary)
    rectangle_limit = int(contract.get("maximum_rectangles", 0))
    if not 2 <= rectangle_limit <= 4 or not 2 <= len(rectangles) <= rectangle_limit:
        return empty
    thicknesses = {c.bbox.height for c in rectangles}
    if len(thicknesses) != 1:
        return empty
    q = next(iter(thicknesses))
    rectangles = tuple(sorted(rectangles, key=lambda c:(c.bbox.top,c.bbox.left,c.component_id)))
    frame = value.frame.rows
    height, width = len(frame), len(frame[0])
    if height % q or width % q or max(height//q,width//q) > 32:
        return empty
    if any(c.bbox.top % q or c.bbox.left % q or c.bbox.width % q for c in rectangles):
        return empty
    value_counts = Counter(v for row in frame for v in row).most_common(2)
    if len(value_counts)>1 and value_counts[0][1]==value_counts[1][1]:
        return empty
    background = value_counts[0][0]
    contours = []
    for c in components:
        b = c.bbox
        if b.width != 3*q or b.height != 2*q or c.area != 5*q*q:
            continue
        missing = {(r,x) for r in range(b.top,b.bottom+1) for x in range(b.left,b.right+1)
                   if (r,x) not in frozenset(c.pixels)}
        if len(missing) != q*q:
            continue
        top, left = min(r for r,_ in missing), min(x for _,x in missing)
        if missing != {(r,x) for r in range(top,top+q) for x in range(left,left+q)}:
            continue
        if left != b.left+q or top not in (b.top,b.top+q) or top%q or left%q:
            continue
        if any(frame[r][x] != background for r,x in missing):
            continue
        sign = 1 if top == b.top else -1
        cells = frozenset((r//q,x//q) for r,x in c.pixels)
        contours.append(((top//q,left//q),cells,sign,c.component_id))
    if not 2 <= len(contours) <= 8 or len({c[2] for c in contours}) != 1:
        return empty
    sign = contours[0][2]
    # A boundary cap may be clipped by a thin interface overlay. The adjacent
    # interior square is a conditional starting cell, directed away from cap.
    pairs = []
    pair_components = set()
    for square in components:
        b = square.bbox
        if b.width != q or b.height != q or square.area != q*q or square.value==background:
            continue
        for cap in components:
            cb = cap.bbox
            if cb.width != q or not 1 <= cb.height <= q or cap.area != cb.width*cb.height:
                continue
            if cap.value in (background,square.value) or cb.left != b.left:
                continue
            if sign == -1 and cb.top == b.bottom+1 and cb.bottom >= height-q:
                pairs.append((b.top//q,b.left//q))
                pair_components.update((square.component_id,cap.component_id))
            elif sign == 1 and cb.bottom == b.top-1 and cb.top < q:
                pairs.append((b.top//q,b.left//q))
                pair_components.update((square.component_id,cap.component_id))
    if len(set(pairs)) != 1:
        return empty
    start = pairs[0]
    accounted = pair_components | {c.component_id for c in rectangles} | {c[3] for c in contours}
    for c in components:
        if c.value==background or c.component_id in accounted:
            continue
        b=c.bbox
        if contract.get('interface_exclusion')=='outermost_pixel_line_only' and (
            (b.height==1 and b.top in (0,height-1)) or
            (b.width==1 and b.left in (0,width-1))):
            continue
        if c.touches_frame_boundary and ((b.width==width and b.height<=q)
                                        or (b.height==height and b.width<=q)):
            continue
        return {**empty,'oriented_network_unmodelled_component_present':True}
    ports = tuple((void,walls) for void,walls,_sign,_ref in contours)
    masks = tuple((c.bbox.top//q,c.bbox.left//q,c.bbox.right//q) for c in rectangles)
    domains = tuple(range(width//q-(right-left)) for _r,left,right in masks)
    count = 1
    for domain in domains:
        count *= len(domain)
    if count > maximum:
        return {**empty,"oriented_network_configuration_bound_exhausted":True}
    if contract.get('optimization') != 'minimum_horizontal_primitives_and_focus_changes':
        return empty
    values = Counter(c.value for c in rectangles)
    unique = tuple(i for i,c in enumerate(rectangles) if values[c.value]==1)
    selected = unique[0] if len(unique)==1 and len(values)==2 else -1
    witnesses = []
    minimum_cost = None
    minimum_count = 0
    feasible_count = 0
    for lefts in product(*domains):
        poses = tuple((r,left,left+(right-original))
                      for (r,original,right),left in zip(masks,lefts))
        if any(r==rr and max(l,ll)<=min(h,hh) for i,(r,l,h) in enumerate(poses)
               for rr,ll,hh in poses[i+1:]):
            continue
        result = project_parallel_accretion(width//q,height//q,start,sign,poses,ports)
        if result is not None:
            feasible_count += 1
            distances = tuple(abs(left-original) for (_r,original,_right),left in zip(masks,lefts))
            cost = sum(distances) + sum(bool(d) and i != selected for i,d in enumerate(distances))
            if minimum_cost is None or cost < minimum_cost:
                minimum_cost, minimum_count, witnesses = cost, 1, [(lefts,result)]
            elif cost == minimum_cost:
                minimum_count += 1
                if len(witnesses) < witness_limit:
                    witnesses.append((lefts,result))
    # Ambiguity is retained rather than selecting a pose by Python ranking.
    if minimum_count != 1:
        return {**empty,"oriented_network_witness_count":minimum_count,
                "oriented_network_feasible_configuration_count":feasible_count}
    lefts, result = witnesses[0]
    residuals = tuple((left-current_left)*q for (_r,current_left,_right),left in zip(masks,lefts))
    return {"oriented_network_present":True,
            "oriented_network_witness_count":1,
            "oriented_network_configurations_examined":count,
            "oriented_network_feasible_configuration_count":feasible_count,
            "oriented_network_minimum_primitive_cost":minimum_cost,
            "oriented_network_quantum":q,
            "oriented_network_emission_sign":sign,
            "oriented_network_port_count":len(ports),
            "oriented_network_rectangles":tuple((c.bbox.top,c.bbox.left,c.bbox.bottom,c.bbox.right) for c in rectangles),
            "oriented_network_target_lefts":tuple(left*q for left in lefts),
            "oriented_network_residuals":residuals,
            "oriented_network_contrast_index":selected,
            "oriented_network_all_residuals_zero":not any(residuals),
            "oriented_network_trace_contact_count":len(result['contacts']),
            "oriented_network_trace_digest":stable_digest(result),
            "oriented_network_witness_digest":stable_digest((masks,lefts,start,
                tuple((void,tuple(sorted(walls))) for void,walls in ports))),
            "oriented_network_conditional_operator":contract['operator']}


def measure_oriented_network_point(facts, row, col):
    boxes = facts.get('oriented_network_rectangles',())
    residuals = facts.get('oriented_network_residuals',())
    contrast = int(facts.get('oriented_network_contrast_index',-1))
    current_open = contrast >= 0 and bool(residuals[contrast])
    return {'candidate_oriented_network_focus_present': bool(
        facts.get('oriented_network_present') and not current_open and any(
            i != contrast and residuals[i] and t <= row <= b and l <= col <= r
            for i,(t,l,b,r) in enumerate(boxes)))}


def measure_oriented_network_discrete(facts, action_ref, directional_refs, horizontal_signs):
    contrast = int(facts.get('oriented_network_contrast_index',-1))
    residuals = facts.get('oriented_network_residuals',())
    residual = residuals[contrast] if 0 <= contrast < len(residuals) else 0
    sign = horizontal_signs.get(action_ref,0)
    return {
        'candidate_oriented_network_translation_present': bool(
            facts.get('oriented_network_present') and residual*sign > 0),
        'candidate_oriented_network_translation_residual':residual,
        'candidate_oriented_network_evaluation_present':bool(
            facts.get('oriented_network_present') and facts.get('oriented_network_all_residuals_zero')
            and action_ref not in directional_refs),
    }
