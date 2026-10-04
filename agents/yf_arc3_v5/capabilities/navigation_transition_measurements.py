"""Exact cell deltas and rigid fits from observed navigation transitions.

No palette, role, support permission or transfer decision is inferred here.
All candidate persistent component centres are tested; ambiguous fits remain.
"""
from collections import defaultdict

from agents.yf_arc3_v5.capabilities.contracts import (
    ReconfigurablePivotCandidate, ReconfigurableRigidGeometryInput,
    ReconfigurableSupportLayer,
)
from agents.yf_arc3_v5.capabilities.reconfigurable_support import measure_reconfigurable_rigid_geometry
from agents.yf_arc3_v5.logos.types import FrozenMap


def measure_current_translation_underlays(*, before, after):
    """Bind prior pixels to unique exact translated components in this pair.

    No controlled role or interface pitch is selected. Retained geometry is
    deliberately absent: only the supplied consecutive observations contribute.
    """
    from agents.yf_arc3_v5.capabilities import ComponentExtractionInput, FrameGrid, detect_components
    if before is None:return ()
    before=getattr(before,'rows',before);after=getattr(after,'rows',after)
    if not before or not after or len(before)!=len(after):return ()
    width=len(after[0])
    if len(after)*width>4096 or any(len(row)!=width for row in (*before,*after)):return ()
    groups=[]
    for frame in (before,after):
        components=detect_components(ComponentExtractionInput(frame=FrameGrid(rows=frame),connectivity=4)).components
        if len(components)>256:return ()
        grouped=defaultdict(list)
        for c in components:
            signature=(c.value,tuple(sorted((r-c.bbox.top,s-c.bbox.left) for r,s in c.pixels)))
            grouped[signature].append(c)
        groups.append(grouped)
    rows=[]
    for signature,current in sorted(groups[1].items()):
        prior=groups[0].get(signature,())
        if len(current)!=1 or len(prior)!=1:continue
        a,b=prior[0],current[0]
        delta=(b.bbox.top-a.bbox.top,b.bbox.left-a.bbox.left)
        if delta==(0,0):continue
        box=b.bbox
        rows.append(FrozenMap({'component_ref':b.component_id,
            'bbox':(box.top,box.left,box.bottom,box.right),'translation_delta':delta,
            'underlay_values':tuple(sorted({before[r][c] for r in range(box.top,box.bottom+1)
                for c in range(box.left,box.right+1)}))}))
    return tuple(rows) if len(rows)<=64 else ()


def measure_remote_navigation_transition(*, before, after, before_components,
        after_components, control, actor, maximum_pixels=4096, maximum_pivots=32):
    height, width = len(before), len(before[0])
    if height * width > maximum_pixels or len(after) != height or any(
            len(row) != width for row in (*before, *after)):
        raise ValueError("navigation transition frame bound/shape mismatch")
    h, w = actor.bbox.height, actor.bbox.width
    phase_r, phase_c = actor.bbox.top % h, actor.bbox.left % w
    excluded = frozenset(actor.pixels) | frozenset(
        (r, c) for r in range(control.bbox.top, control.bbox.bottom + 1)
        for c in range(control.bbox.left, control.bbox.right + 1))
    groups = defaultdict(list)
    for r in range(phase_r, height-h+1, h):
        for c in range(phase_c, width-w+1, w):
            pixels = tuple((r+dr, c+dc) for dr in range(h) for dc in range(w))
            if any(p in excluded for p in pixels):
                continue
            a = tuple(before[rr][cc] for rr, cc in pixels)
            b = tuple(after[rr][cc] for rr, cc in pixels)
            if a != b:
                groups[a, b].append((r, c))
    cell_rows = tuple(FrozenMap({"before_pattern": a, "after_pattern": b,
        "cell_height": h, "cell_width": w, "cell_positions": tuple(groups[a, b]),
        "cell_count": len(groups[a, b])}) for a, b in sorted(groups))

    changed = frozenset((r, c) for r in range(height) for c in range(width)
        if before[r][c] != after[r][c] and (r, c) not in excluded)
    after_keys = {(c.value, tuple(c.pixels)) for c in after_components}
    actor_pixels = frozenset(actor.pixels)
    persistent = tuple(c for c in before_components
        if (c.value, tuple(c.pixels)) in after_keys and c.area > 1
        and not set(c.pixels) & excluded
        and c.area + sum(c.bbox.top <= r <= c.bbox.bottom and
            c.bbox.left <= col <= c.bbox.right for r, col in actor_pixels)
            == c.bbox.height*c.bbox.width)
    # Oversized domains are explicitly incomplete, never silently truncated.
    fits_complete = len(persistent) <= maximum_pivots
    fit_rows = []
    if persistent and fits_complete:
        candidates = tuple(ReconfigurablePivotCandidate(
            candidate_ref=f"component:{i}",
            position=(c.bbox.top+c.bbox.bottom, c.bbox.left+c.bbox.right),
            staging_actor_positions=tuple((2*r, 2*s) for r, s in actor.pixels))
            for i, c in enumerate(persistent))
        # No support ownership is established by these appearance measurements.
        # Keep the geometry API's support checks false rather than assuming
        # the actor's visible pixels constitute their own supporting layer.
        layers = (ReconfigurableSupportLayer(owner_ref="unbound.support", positions=()),)
        values = sorted({before[r][c] for r, c in changed} | {after[r][c] for r, c in changed})
        for value in values:
            masks = []
            for components in (before_components, after_components):
                masks.append(frozenset(p for c in components if c.value == value
                    and set(c.pixels) & changed and not set(c.pixels) & excluded
                    for p in c.pixels))
            a, b = masks
            if not a or not b or a == b or len(a) != len(b):
                continue
            result = measure_reconfigurable_rigid_geometry(ReconfigurableRigidGeometryInput(
                before_positions=tuple(sorted((2*r, 2*c) for r, c in a)),
                after_positions=tuple(sorted((2*r, 2*c) for r, c in b)),
                pivot_candidates=candidates, before_support_layers=layers, after_support_layers=layers))
            for fit in result.exact_quarter_turn_fits:
                index = int(fit.candidate_ref.split(":")[1])
                landmark = persistent[index]
                centre = candidates[index].position
                fit_rows.append(FrozenMap({"changed_value": value,
                    "control_value_equals_changed_value": value == control.value,
                    "quarter_turns_clockwise": fit.quarter_turns_clockwise,
                    "pivot_centre_twice": centre,
                    "pivot_value": landmark.value,
                    "pivot_height": landmark.bbox.height, "pivot_width": landmark.bbox.width,
                    "pivot_occluded_pixel_count": landmark.bbox.height*landmark.bbox.width-landmark.area,
                    "pivot_outside_transformed_masks": fit.pivot_outside_transformed_masks,
                    "before_relative_pixels_twice": tuple(sorted((2*r-centre[0], 2*c-centre[1]) for r, c in a)),
                    "after_relative_pixels_twice": tuple(sorted((2*r-centre[0], 2*c-centre[1]) for r, c in b))}))
    return FrozenMap({"cell_transition_rows": cell_rows,
        "rigid_fit_rows": tuple(fit_rows), "rigid_fit_enumeration_complete": fits_complete,
        "persistent_component_count": len(persistent), "coordinate_scale": 2})
