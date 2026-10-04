"""Exact enclosing raster witnesses across merged component boundaries."""
from agents.yf_arc3_v5.logos.types import stable_digest


def measure_stationary_control_contact_rasters(*, before, after, components,
                                             translations, before_entities,
                                             after_entities, control_boxes,
                                             expected_delta, control_bindings=(), prior_translation_facts=()):
    """Recover a translated complete patch from its independently tracked core.

    No segmentation identity is invented: the before patch can be a subset of
    a connected component. Both full colored patches and the stationary
    control patch must agree exactly. Multiple compatible witnesses fail closed.
    """
    if len(control_boxes) != 1 or expected_delta is None or not any(expected_delta):
        return {}
    if len(components) > 128 or len(translations) > 32:
        return {}
    control, = control_boxes
    rebound = tuple((binding, after_entities.get(binding.get('entity_ref')))
                    for binding in control_bindings if binding.get('role_premise_claim_refs'))
    rebound = tuple((binding, tracked) for binding,tracked in rebound
        if tracked is not None
        and tracked.component.value == binding.get('observed_value')
        and tracked.component.bbox.height == binding.get('bbox_height')
        and tracked.component.bbox.width == binding.get('bbox_width')
        and binding.get('shape_digest')
        and stable_digest(tracked.component.relative_pixels).startswith(str(binding['shape_digest'])))
    if len(rebound) == 1:
        control = rebound[0][1].component.bbox
    if control.height * control.width > 256:
        return {}

    def patch(frame, top, left, height, width):
        if top < 0 or left < 0 or top+height > frame.height or left+width > frame.width:
            return None
        return tuple(tuple(row[left:left+width]) for row in frame.rows[top:top+height])

    original = patch(before, control.top, control.left, control.height, control.width)
    current = patch(after, control.top, control.left, control.height, control.width)
    if original is None or current is None:
        return {}
    if original != current:
        if len(rebound) != 1 or control.height < 3 or control.width < 3:
            return {}
        body_value = rebound[0][1].component.value
        interior_values = {current[r][c] for r in range(1,control.height-1)
                           for c in range(1,control.width-1)} - {body_value}
        if len(interior_values) != 1:
            return {}
        marker, = interior_values
        changed = {(r,c) for r in range(control.height) for c in range(control.width)
                   if original[r][c] != current[r][c]}
        edge = {(r,c) for r in range(control.height) for c in range(control.width)
                if (expected_delta[1]>0 and c==control.width-1)
                or (expected_delta[1]<0 and c==0)
                or (expected_delta[0]>0 and r==control.height-1)
                or (expected_delta[0]<0 and r==0)}
        if changed != edge or any(original[r][c]!=marker or current[r][c]!=body_value for r,c in changed):
            return {}
    witnesses = []
    for motion in translations:
        dy, dx = motion.delta_row, motion.delta_col
        if bool(dy) == bool(dx) or dy*expected_delta[0]+dx*expected_delta[1] <= 0:
            continue
        old = before_entities.get(motion.entity_ref)
        new = after_entities.get(motion.entity_ref)
        if old is None or new is None:
            continue
        core = new.component.bbox
        enclosing = tuple(c for c in components
            if c.bbox.top < core.top <= core.bottom < c.bbox.bottom
            and c.bbox.left < core.left <= core.right < c.bbox.right
            and c.bbox.height*c.bbox.width <= 256
            and not c.touches_frame_boundary)
        valid = []
        for body in enclosing:
            b = body.bbox
            # A solid rectangle except for the independently tracked core.
            expected = {(r,c) for r in range(b.top,b.bottom+1) for c in range(b.left,b.right+1)
                        if not (core.top <= r <= core.bottom and core.left <= c <= core.right)}
            if set(body.pixels) != expected:
                continue
            top,left = b.top-dy,b.left-dx
            before_patch = patch(before,top,left,b.height,b.width)
            if before_patch is None or before_patch != patch(after,b.top,b.left,b.height,b.width):
                continue
            if (old.component.bbox.top != core.top-dy or old.component.bbox.left != core.left-dx):
                continue
            # Oriented edge contact, allowing partial overlap of different sizes.
            touching = ((dx > 0 and control.right+1 == left or dx < 0 and left+b.width == control.left)
                        and max(control.top,top) <= min(control.bottom,top+b.height-1)
                        or (dy > 0 and control.bottom+1 == top or dy < 0 and top+b.height == control.top)
                        and max(control.left,left) <= min(control.right,left+b.width-1))
            if touching:
                valid.append((body,dy,dx))
        if len(valid) == 1:
            witnesses.extend(valid)
    if len(witnesses) != 1:
        return {}
    body,dy,dx = witnesses[0]
    b=body.bbox
    top,left=b.top-dy,b.left-dx
    transit_values=tuple(sorted({before.rows[r][c]
        for r in range(min(top,b.top),max(top,b.top)+b.height)
        for c in range(min(left,b.left),max(left,b.left)+b.width)
        if not (top<=r<top+b.height and left<=c<left+b.width)
        and not (b.top<=r<=b.bottom and b.left<=c<=b.right)}))
    quanta=tuple(abs(int(row.get('delta_row') or 0))+abs(int(row.get('delta_col') or 0))
        for row in prior_translation_facts
        if bool(row.get('delta_row')) != bool(row.get('delta_col')))
    return {
        'unique_stationary_control_enclosing_raster_contact_translation': True,
        'stationary_control_enclosing_raster_contact_pairs': ((
            ':'.join(map(str,(control.top,control.left,control.bottom,control.right))), body.component_id),),
        'contact_raster_translation_delta_row': dy,
        'contact_raster_translation_delta_col': dx,
        'contact_raster_observed_value': body.value,
        'contact_raster_bbox_height': body.bbox.height,
        'contact_raster_bbox_width': body.bbox.width,
        'contact_raster_transit_values': transit_values,
        'contact_raster_ordinary_quantum': min(quanta,default=0),
    }
