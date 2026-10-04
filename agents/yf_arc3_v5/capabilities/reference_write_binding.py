"""Exact current raster joins for declared, revisable surface-writing memory."""
from agents.yf_arc3_v5.capabilities.contracts import VisualSceneInput, MultiResolutionViewInput
from agents.yf_arc3_v5.capabilities.sprites import describe_visual_scene
from agents.yf_arc3_v5.capabilities.views import measure_multi_resolution_views
from agents.yf_arc3_v5.capabilities.surface_write_geometry import measure_conditional_half_surface


def unique_current_output(elements, goal):
    output = elements.get(goal.get('output_candidate_ref'))
    if output is None and goal.get('output_identity_preserved_across_palette_change') is True:
        bounds = tuple(goal.get('output_bbox_' + key) for key in ('top', 'left', 'bottom', 'right'))
        if all(type(value) is int for value in bounds):
            matches = tuple(row for _, row in sorted(elements.items())
                if tuple(row['bbox_' + key] for key in ('top', 'left', 'bottom', 'right')) == bounds
                and row['area'] == row['bbox_area'])
            if len(matches) == 1:
                output = matches[0]
    return output


def measure_reference_write_binding(*, frame, methods, goals, poses, steps=()):
    absent = {'reference_write_binding_present': False,
              'conditional_surface_write_present': False,
              'reference_write_value_absent_from_output': False,
              'reference_write_pose_palette_matches': False,
              'reference_write_mismatch_count': 0,
              'reference_write_unique_action_ref': '',
              'reference_write_unique_goal_ref': '',
              'reference_write_premise_refs': (),
              'reference_write_missing_value_count': 0,
              'reference_write_missing_value_peer_refs': ()}
    if not methods or not goals:
        return absent
    if len(methods) > 16 or len(goals) > 64 or len(poses) > 64:
        raise ValueError('reference write binding input bound exceeded')
    descriptions = {(m['application_action_ref'], m['written_palette_value']) for m in methods}
    if len(descriptions) != 1:
        return absent
    action, palette = next(iter(descriptions))
    scene = describe_visual_scene(VisualSceneInput(frame=frame))
    measurements = measure_multi_resolution_views(MultiResolutionViewInput(
        scene=scene, observation_ref='measurement:current-reference-write'))
    elements = {r['element_ref']: r for r in
                measurements.descriptive_facts['distinct_visual_element_measurements']}
    bindings = []
    for goal in goals:
        reference = elements.get(goal.get('reference_candidate_ref'))
        output = unique_current_output(elements, goal)
        if reference is None or output is None or reference is output:
            continue
        if any(r['area'] != r['bbox_area'] for r in (reference, output)):
            continue
        if (reference['height'], reference['width']) != (output['height'], output['width']):
            continue
        if palette not in reference['values']:
            continue
        bindings.append((goal, reference, output))
    if len(bindings) != 1:
        return absent
    goal, reference, output = bindings[0]
    current_poses = tuple(p for p in poses
        if p.get('surface_candidate_ref') in (output['element_ref'], goal.get('output_candidate_ref'))
        and p.get('cursor_candidate_ref') in elements
        and palette in elements[p['cursor_candidate_ref']]['values'])
    height, width = output['height'], output['width']
    mismatch = sum(frame.rows[reference['bbox_top']+y][reference['bbox_left']+x]
                   != frame.rows[output['bbox_top']+y][output['bbox_left']+x]
                   for y in range(height) for x in range(width))
    missing_values = set(reference['values']) - set(output['values'])
    peers_by_shape = {}
    for component in scene.blocks.components:
        box = component.bbox
        if component.area != box.height * box.width or component.value not in reference['values']:
            continue
        if any(not (box.bottom < surface['bbox_top'] or box.top > surface['bbox_bottom']
                    or box.right < surface['bbox_left'] or box.left > surface['bbox_right'])
               for surface in (reference, output)):
            continue
        peers_by_shape.setdefault((box.height, box.width), []).append(component)
    missing_peers = tuple(c.component_id for _, peers in sorted(peers_by_shape.items())
        if palette in {c.value for c in peers}
        for c in peers if c.value in missing_values)
    conditional = measure_conditional_half_surface(frame=frame, scene=scene,
        reference=reference, output=output, methods=methods, steps=steps)
    return {
        **conditional,
        'reference_write_binding_present': True,
        'reference_write_value_absent_from_output': palette not in output['values'],
        'reference_write_pose_palette_matches': bool(current_poses),
        'reference_write_mismatch_count': mismatch,
        'reference_write_unique_action_ref': action,
        'reference_write_unique_goal_ref': goal['term_ref'],
        'reference_write_premise_refs': tuple(dict.fromkeys(
            ref for m in methods for ref in m['role_premise_claim_refs'])),
        'reference_write_missing_value_count': len(missing_values),
        'reference_write_missing_value_peer_refs': missing_peers,
    }
