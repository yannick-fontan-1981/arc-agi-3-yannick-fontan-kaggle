"""Bounded rectangular support/perimeter comparisons, without role selection."""
from agents.yf_arc3_v5.logos.types import stable_digest
from dataclasses import dataclass
from agents.yf_arc3_v5.capabilities.contracts import BoundingBox


@dataclass(frozen=True)
class MeasuredHollowSupport:
    """Local logical contour plus observed contact overlay; not raw perception."""
    component_id: str
    value: int
    bbox: BoundingBox
    pixels: tuple
    relative_pixels: tuple
    area: int
    touches_frame_boundary: bool
    source_control_entity_ref: str
    overlay_pixels: tuple


def measure_contact_overlay_supports(components, frame, bindings, *, prior_sprites=(), point=None, support_sizes=(), current_box_support=False, control_boxes=()):
    if frame is None or len(components)>128 or len(bindings)>8:
        return ()
    variants=list(bindings)
    if current_box_support and len(support_sizes)<=8 and len(control_boxes)<=8:
        for binding in bindings:
            box=tuple(binding.get(k) for k in ('bbox_top','bbox_left','bbox_height','bbox_width'))
            source_ref=binding.get('entity_ref') or binding.get('term_ref')
            if not source_ref or not binding.get('role_premise_claim_refs'):
                continue
            if not all(isinstance(n,int) for n in box):
                boxes={(b.top,b.left,b.height,b.width) for b in control_boxes
                    if (b.height,b.width)==(binding.get('bbox_height'),binding.get('bbox_width'))}
                if len(boxes)!=1: continue
                box,=boxes
            bt,bl,bh,bw=box
            for h,w in support_sizes:
                if (h-bh<0 or w-bw<0 or h-bh+w-bw>1 or min(h,w)<3 or h*w>256): continue
                center={(r,c) for r in range((h-1)//2,h//2+1) for c in range((w-1)//2,w//2+1)}
                relative=tuple((r,c) for r in range(h) for c in range(w) if (r,c) not in center)
                for top in range(bt+bh-h,bt+1):
                    for left in range(bl+bw-w,bl+1):
                        if top<0 or left<0 or top+h>frame.height or left+w>frame.width: continue
                        marks={frame.rows[top+r][left+c] for r,c in center}
                        colors={frame.rows[top+r][left+c] for r,c in relative}-marks
                        if len(marks)!=1 or len(colors)!=1: continue
                        color,=colors
                        if binding.get('observed_value') is not None and binding['observed_value']!=color: continue
                        variants.append({**binding,'observed_value':color,'bbox_height':h,'bbox_width':w,
                            'shape_digest':stable_digest(relative)[:16],
                            'entity_ref':str(source_ref),'exact_scene_box':(top,left,h,w)})
    # A selected prior raster may contain several touching bodies. Recover
    # only its unique complete centered patch around the actual click, using
    # independently observed perimeter dimensions and a canonical role premise.
    if point is not None and len(support_sizes)<=8 and len(prior_sprites)<=128:
        for binding in bindings:
            color=binding.get('observed_value')
            if color is None or not binding.get('entity_ref') or not binding.get('role_premise_claim_refs'): continue
            patches=set()
            for h,w in support_sizes:
                if (min(h,w)<3 or h*w>256
                        or int(binding.get('bbox_height',0)) not in (h,h-1)
                        or int(binding.get('bbox_width',0)) not in (w,w-1)): continue
                if (h,w)==(binding.get('bbox_height'),binding.get('bbox_width')): continue
                center={(r,c) for r in range((h-1)//2,h//2+1) for c in range((w-1)//2,w//2+1)}
                relative=tuple((r,c) for r in range(h) for c in range(w) if (r,c) not in center)
                for sprite in prior_sprites:
                    b=sprite.bbox
                    if not (b.top<=point[0]<=b.bottom and b.left<=point[1]<=b.right): continue
                    for top in range(max(b.top,point[0]-h+1),min(point[0],b.bottom-h+1)+1):
                        for left in range(max(b.left,point[1]-w+1),min(point[1],b.right-w+1)+1):
                            if any(sprite.pattern[top-b.top+r][left-b.left+c]!=color for r,c in relative): continue
                            marks={sprite.pattern[top-b.top+r][left-b.left+c] for r,c in center}
                            if len(marks)==1 and color not in marks:
                                patches.add((top,left,h,w,relative))
            if len(patches)==1:
                _,_,h,w,relative=next(iter(patches))
                variants.append({**binding,'bbox_height':h,'bbox_width':w,
                    'shape_digest':stable_digest(relative)[:16]})
    if point is not None and len(prior_sprites)<=128:
        for binding in bindings:
            for sprite in prior_sprites:
                b=sprite.bbox;h,w=b.height,b.width;color=binding.get('observed_value')
                if not (b.top<=point[0]<=b.bottom and b.left<=point[1]<=b.right): continue
                if min(h,w)<3 or h*w>256: continue
                if not (int(binding.get('bbox_height',0)) in (h,h-1)
                        and int(binding.get('bbox_width',0)) in (w,w-1)): continue
                center={(r,c) for r in range((h-1)//2,h//2+1) for c in range((w-1)//2,w//2+1)}
                relative=tuple((r,c) for r in range(h) for c in range(w) if (r,c) not in center)
                # A canonical compound binding can retain selection without a
                # single entity/color digest. Its exact scene box and the
                # pre-click raster supply the same geometric witness locally.
                compound_box=(binding.get('bbox_top'),binding.get('bbox_left'),
                    binding.get('bbox_height'),binding.get('bbox_width'))
                if (color is None and binding.get('term_ref') and binding.get('role_premise_claim_refs')
                        and compound_box[2:]==(h,w)
                        and all(isinstance(n,int) for n in compound_box)):
                    colors={sprite.pattern[r][c] for r,c in relative}
                    marks={sprite.pattern[r][c] for r,c in center}
                    if len(colors)==1 and len(marks)==1 and colors.isdisjoint(marks):
                        variants.append({**binding,'observed_value':next(iter(colors)),
                            'shape_digest':stable_digest(relative)[:16],
                            'entity_ref':binding['term_ref'],'exact_scene_box':compound_box})
                    continue
                if any(sprite.pattern[r][c]!=color for r,c in relative): continue
                marks={sprite.pattern[r][c] for r,c in center}
                if len(marks)!=1 or color in marks: continue
                if (h,w)==(binding.get('bbox_height'),binding.get('bbox_width')): continue
                variants.append({**binding,'bbox_height':h,'bbox_width':w,
                    'shape_digest':stable_digest(relative)[:16]})
    found=[]
    for binding in variants:
        h,w=int(binding.get('bbox_height',0)),int(binding.get('bbox_width',0))
        color=binding.get('observed_value')
        if not binding.get('role_premise_claim_refs') or min(h,w)<3 or h*w>256:
            continue
        center={(r,c) for r in range((h-1)//2,h//2+1) for c in range((w-1)//2,w//2+1)}
        relative=tuple((r,c) for r in range(h) for c in range(w) if (r,c) not in center)
        if not binding.get('shape_digest') or not stable_digest(relative).startswith(str(binding['shape_digest'])):
            continue
        candidates=[]
        for raw in components:
            if raw.value!=color: continue
            b=raw.bbox
            if not (h-1<=b.height<=h and w-1<=b.width<=w): continue
            for top in range(b.bottom-h+1,b.top+1):
                for left in range(b.right-w+1,b.left+1):
                    if binding.get('exact_scene_box') and (top,left,h,w)!=binding['exact_scene_box']: continue
                    if top<0 or left<0 or top+h>frame.height or left+w>frame.width: continue
                    marks={frame.rows[top+r][left+c] for r,c in center}
                    if len(marks)!=1 or color in marks: continue
                    marker,=marks
                    missing={(r,c) for r,c in relative if frame.rows[top+r][left+c]!=color}
                    if not missing or any(frame.rows[top+r][left+c]!=marker for r,c in missing): continue
                    edges=( {(0,c) for c in range(w)}, {(h-1,c) for c in range(w)},
                            {(r,0) for r in range(h)}, {(r,w-1) for r in range(h)} )
                    sides=tuple(i for i,e in enumerate(edges) if missing<=e)
                    if len(sides)!=1: continue
                    side,=sides
                    dy,dx=((-1,0),(1,0),(0,-1),(0,1))[side]
                    if not any(0<=top+r+dy<frame.height and 0<=left+c+dx<frame.width
                        and frame.rows[top+r+dy][left+c+dx]==color for r,c in missing): continue
                    logical=tuple((top+r,left+c) for r,c in relative)
                    if not set(raw.pixels)<=set(logical): continue
                    candidates.append(MeasuredHollowSupport(raw.component_id,color,
                        BoundingBox(top=top,left=left,bottom=top+h-1,right=left+w-1),logical,
                        relative,len(relative),False,str(binding['entity_ref']),
                        tuple(sorted((top+r,left+c) for r,c in missing))))
        if len(candidates)==1: found.extend(candidates)
    # Several active canonical aliases can attest the same observed support.
    # Count that geometry once; retain a valid source witness without changing
    # or discarding any of the canonical bindings supplied to this measurement.
    unique={}
    for body in found:
        unique.setdefault((body.bbox,body.value,body.relative_pixels,body.overlay_pixels),body)
    return tuple(unique.values())


def partition_touching_hollow_supports(components, perimeters):
    """Unique disjoint complete cover by centered rectangles with observed sizes."""
    sizes=tuple(sorted({(p.bbox.height-2,p.bbox.width-2) for p in perimeters
        if min(p.bbox.height,p.bbox.width)>=5 and (p.bbox.height-2)*(p.bbox.width-2)<=256}))
    if len(sizes)>8 or len(components)>128: return ()
    result=[]
    for raw in components:
        if raw.touches_frame_boundary or raw.area>2048: continue
        raw_pixels=set(raw.pixels)
        candidates=[]
        for core in components:
            b=core.bbox
            if (core.value==raw.value or core.area!=b.height*b.width
                    or not (raw.bbox.top<b.top<=b.bottom<raw.bbox.bottom
                    and raw.bbox.left<b.left<=b.right<raw.bbox.right)): continue
            for h,w in sizes:
                if (b.height,b.width)!=(2-h%2,2-w%2): continue
                top,left=b.top-(h-1)//2,b.left-(w-1)//2
                footprint={(r,c) for r in range(top,top+h) for c in range(left,left+w)}
                pixels=footprint-set(core.pixels)
                if not pixels<=raw_pixels: continue
                candidates.append((top,left,h,w,pixels,footprint))
        if not 2<=len(candidates)<=8: continue
        if set().union(*(row[4] for row in candidates))!=raw_pixels: continue
        if any(a[5]&b[5] for i,a in enumerate(candidates) for b in candidates[i+1:]): continue
        for top,left,h,w,pixels,_ in candidates:
            ordered=tuple(sorted(pixels));relative=tuple((r-top,c-left) for r,c in ordered)
            result.append(MeasuredHollowSupport(f'{raw.component_id}.part.r{top}.c{left}',raw.value,
                BoundingBox(top=top,left=left,bottom=top+h-1,right=left+w-1),ordered,
                relative,len(ordered),False,'',()))
    return tuple(result)


def rectangular_hollow_families(components, *, maximum_components=128, frame=None, bindings=(), prior_sprites=(), point=None, partition_touching=False, current_box_support=False, control_boxes=()):
    if len(components) > maximum_components:
        return ()
    perimeters = []
    hollow = []
    for component in components:
        box = component.bbox
        if box.height < 3 or box.width < 3 or component.touches_frame_boundary:
            continue
        border = {(r,c) for r in range(box.top,box.bottom+1)
                  for c in range(box.left,box.right+1)
                  if r in (box.top,box.bottom) or c in (box.left,box.right)}
        pixels = set(component.pixels)
        if pixels == border:
            perimeters.append(component)
        if border <= pixels and component.area < box.height * box.width:
            missing = {(r,c) for r in range(box.top,box.bottom+1)
                       for c in range(box.left,box.right+1)} - pixels
            top,bottom = min(r for r,c in missing),max(r for r,c in missing)
            left,right = min(c for r,c in missing),max(c for r,c in missing)
            if (len(missing) == (bottom-top+1)*(right-left+1)
                    and top+bottom == box.top+box.bottom
                    and left+right == box.left+box.right):
                hollow.append(component)
    reconstructed=measure_contact_overlay_supports(components,frame,bindings,
        prior_sprites=prior_sprites,point=point,current_box_support=current_box_support,control_boxes=control_boxes,
        support_sizes=tuple(sorted({(p.bbox.height-2,p.bbox.width-2) for p in perimeters})) if partition_touching else ())
    if partition_touching:
        hollow.extend(partition_touching_hollow_supports(components,perimeters))
    replaced={body.component_id for body in reconstructed}
    hollow=[body for body in hollow if body.component_id not in replaced]+list(reconstructed)
    matched = tuple(body for body in hollow if any(
        frame.value != body.value
        and frame.bbox.height == body.bbox.height+2
        and frame.bbox.width == body.bbox.width+2 for frame in perimeters))
    values = tuple(sorted({body.value for body in matched}))
    return tuple(body for value in values
                 if sum(body.value == value for body in matched) >= 2
                 for body in hollow if body.value == value)


def measure_recalled_control_steps(*, frame, components, bindings, boxes,
                                  observed_boxes, assignments, deltas):
    matches = tuple(body for body in components if body.component_id in assignments
                    and body.bbox in boxes)
    if len(matches) != 1 or matches[0].bbox in observed_boxes:
        return {}
    body, = matches
    shape_digest = stable_digest(body.relative_pixels)
    compatible = tuple(binding for binding in bindings
        if binding.get('entity_ref') == body.component_id or (
            binding.get('bbox_height') == body.bbox.height
            and binding.get('bbox_width') == body.bbox.width
            and binding.get('observed_value') == body.value
            and binding.get('shape_digest')
            and shape_digest.startswith(str(binding['shape_digest']))))
    if len({binding.get('entity_ref') for binding in compatible}) != 1:
        return {}
    proofs = tuple(sorted({str(ref) for binding in compatible
        for ref in binding.get('role_premise_claim_refs', ())}))
    if not proofs:
        return {}
    row = assignments[body.component_id]
    dr = int(row['nearest_exact_enclosure_residual_row_signed'])
    dc = int(row['nearest_exact_enclosure_residual_col_signed'])
    actions = []
    for action,dy,dx in deltas:
        top,left=body.bbox.top+dy,body.bbox.left+dx
        if (top<0 or left<0 or top+body.bbox.height>frame.height
                or left+body.bbox.width>frame.width
                or abs(dr-2*dy)+abs(dc-2*dx)>=abs(dr)+abs(dc)):
            continue
        exterior = {frame.rows[r][c] for r in range(top,top+body.bbox.height)
                    for c in range(left,left+body.bbox.width)
                    if not (body.bbox.top<=r<=body.bbox.bottom
                            and body.bbox.left<=c<=body.bbox.right)}
        if len(exterior)==1:
            actions.append(action)
    return {'recalled_control_enclosure_step_present':bool(actions),
            'recalled_control_enclosure_step_action_refs':tuple(sorted(set(actions))),
            'recalled_control_enclosure_premise_refs':proofs}
