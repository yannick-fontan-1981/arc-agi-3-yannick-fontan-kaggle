"""Bounded geometric comparisons; all inferred meanings remain source-owned."""
from collections import deque


def half_surface_positions(height, width, normal, inclusive):
    if not 0 < height * width <= 4096 or normal == (0, 0):
        raise ValueError('invalid bounded half-surface geometry')
    dy, dx = normal
    return tuple((y, x) for y in range(height) for x in range(width)
        if (dy * (2*y + 1-height) * width + dx * (2*x + 1-width) * height
            >= (0 if inclusive else 1)))


def external_span_peers(components, output, values, excluded=()):
    """Enumerate exterior monochrome supports spanning a supplied surface axis."""
    height, width = output['height'], output['width']
    result = []
    for component in components:
        b = component.bbox
        if component.value not in values or component.touches_frame_boundary:
            continue
        if any(any(r['bbox_top'] <= y <= r['bbox_bottom'] and r['bbox_left'] <= x <= r['bbox_right']
                   for y, x in component.pixels) for r in excluded):
            continue
        dy = max(output['bbox_top'] - b.bottom - 1, b.top - output['bbox_bottom'] - 1, 0)
        dx = max(output['bbox_left'] - b.right - 1, b.left - output['bbox_right'] - 1, 0)
        outside = not any(output['bbox_top'] <= y <= output['bbox_bottom']
                          and output['bbox_left'] <= x <= output['bbox_right']
                          for y, x in component.pixels)
        if not outside or max(dy, dx) > max(height, width):
            continue
        if b.height < height and b.width < width:
            continue
        cy = b.top + b.bottom - output['bbox_top'] - output['bbox_bottom']
        cx = b.left + b.right - output['bbox_left'] - output['bbox_right']
        normal = ((cy > 0) - (cy < 0), (cx > 0) - (cx < 0))
        if normal != (0, 0):
            result.append((component.value, normal))
    return tuple(result)


def measure_half_surface_support(*, scene, reference, output, changed, written_values, before):
    if len(written_values) != 1:
        return False
    value = written_values[0]
    if any(before[y][x] == value for y in range(output['bbox_top'], output['bbox_bottom']+1)
           for x in range(output['bbox_left'], output['bbox_right']+1)):
        return False
    peers = external_span_peers(scene.blocks.components, output, (value,), (reference,))
    if len(peers) != 1:
        return False
    _, normal = peers[0]
    return tuple(changed) in tuple(half_surface_positions(output['height'], output['width'], normal, inclusive)
                                  for inclusive in (False, True))


def measure_conditional_half_surface(*, frame, scene, reference, output, methods, steps):
    absent = {'conditional_surface_write_present': False,
              'conditional_surface_write_at_pose': False,
              'conditional_surface_write_first_action_ref': '',
              'conditional_surface_write_variant_count': 0}
    if not any(m.get('observed_oriented_half_surface') is True for m in methods):
        return absent
    peers = external_span_peers(scene.blocks.components, output, reference['values'], (reference,))
    if len(peers) != 1:
        return absent
    value, current = peers[0]
    h, w = output['height'], output['width']
    target = tuple(tuple(frame.rows[reference['bbox_top']+y][reference['bbox_left']+x] for x in range(w)) for y in range(h))
    canvas = tuple(tuple(frame.rows[output['bbox_top']+y][output['bbox_left']+x] for x in range(w)) for y in range(h))
    if target == canvas:
        return absent
    variants = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if (dy, dx) == (0, 0):
                continue
            distinct_supports = set()
            for inclusive in (False, True):
                mask = half_surface_positions(h, w, (dy, dx), inclusive)
                if mask in distinct_supports:
                    continue
                distinct_supports.add(mask)
                positions = frozenset(mask)
                if all((value if (y,x) in positions else canvas[y][x]) == target[y][x]
                       for y in range(h) for x in range(w)):
                    variants.append(((dy, dx), inclusive))
    if len(variants) != 1:
        return dict(absent, conditional_surface_write_variant_count=len(variants))
    destination, inclusive = variants[0]
    # One deterministic shortest witness on a finite counterfactual pose ring.
    # Source must open the ontology extension before any of its edges is used.
    queue = deque([(current, ())]); visited = {current}; route = ()
    while queue:
        position, path = queue.popleft()
        if position == destination:
            route = path
            break
        for action, dy, dx in sorted(steps):
            if abs(dy) + abs(dx) != 1:
                continue
            neighbor = (position[0]+dy, position[1]+dx)
            if neighbor == (0,0) or max(abs(v) for v in neighbor) > 1 or neighbor in visited:
                continue
            visited.add(neighbor); queue.append((neighbor, (*path, action)))
    return {
        'conditional_surface_write_present': current == destination or bool(route),
        'conditional_surface_write_at_pose': current == destination,
        'conditional_surface_write_first_action_ref': route[0] if route else '',
        'conditional_surface_write_variant_count': 1,
        'conditional_surface_write_boundary_inclusive': inclusive,
        'conditional_surface_write_target_normal': destination,
        'conditional_surface_write_current_normal': current,
    }
