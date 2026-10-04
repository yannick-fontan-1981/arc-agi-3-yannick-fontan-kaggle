"""Bounded scene-local cell render witnesses for a committed two-layer route.

No layer order is assumed. At intersections, either input value is compatible;
new mixed values require another measured render law and remain unsupported.
"""
from agents.yf_arc3_v5.logos.types import FrozenMap


def _samples(tracking):
    result = {}
    for entity in tracking.entities:
        if entity.current_frame_ref == tracking.frame_ref:
            for point in entity.component.pixels:
                if point in result:
                    raise ValueError("route layer tracking overlaps samples")
                result[point] = entity.component.value
                if len(result) > 4096:
                    raise ValueError("route layer scene bound exceeded")
    return result


def _points(cells, geometry):
    _, ro, co, h, w, rp, cp, _, _ = geometry
    if len(cells)*h*w > 4096:
        raise ValueError("route layer sample bound exceeded")
    return tuple((ro+r*rp+dr, co+c*cp+dc) for r,c in cells
                 for dr in range(h) for dc in range(w))


def capture_cell_layer(tracking, geometry, support):
    samples = _samples(tracking)
    top,left = support['cell_bbox'][:2]
    cells = tuple((top+r,left+c) for r,c in support['normalized_cells'])
    points = _points(cells,geometry)
    if not points or any(p not in samples for p in points):
        return FrozenMap()
    _,ro,co,_,_,rp,cp,_,_ = geometry
    return FrozenMap({'top':top,'left':left,'cells':support['normalized_cells'],
        'samples':tuple((r-ro-top*rp,c-co-left*cp,samples[r,c]) for r,c in points)})


def measure_route_cell_layers(tracking, geometry, fixed, moving, moving_box):
    if not fixed or not moving or len(moving_box)!=4:
        return FrozenMap({'available':False,'matches':False})
    _,ro,co,_,_,rp,cp,_,_ = geometry
    layers=[]
    for layer,top,left in ((fixed,fixed['top'],fixed['left']), (moving,*moving_box[:2])):
        if len(layer['samples'])>4096:
            raise ValueError('route layer witness bound exceeded')
        layers.append({(ro+top*rp+r,co+left*cp+c):v for r,c,v in layer['samples']})
    a,b=layers
    current=_samples(tracking)
    overlap=set(a)&set(b)
    outside=set(a)-set(b)
    mismatches=sum(current.get(p) not in {layer[p] for layer in layers if p in layer}
                   for p in set(a)|set(b))
    return FrozenMap({'available':True,'matches':mismatches==0 and bool(outside),
        'uncovered_fixed_sample_count':len(outside),'overlap_sample_count':len(overlap),
        'mismatch_count':mismatches})
