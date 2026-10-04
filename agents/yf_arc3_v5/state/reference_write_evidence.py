"""Exact joins from a declared surface goal to its recorded terminal packet."""
from agents.yf_arc3_v5.capabilities.contracts import MultiResolutionViewInput, VisualSceneDescription
from agents.yf_arc3_v5.capabilities.surface_transition import measure_surface_transition
from agents.yf_arc3_v5.capabilities.views import measure_multi_resolution_views
from agents.yf_arc3_v5.capabilities.surface_write_geometry import measure_half_surface_support
from agents.yf_arc3_v5.capabilities.reference_write_binding import unique_current_output
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest


def measure_reference_write_evidence(*, snapshot, repository, scene, outcome, evidence_refs, profile):
    facts = FrozenMap({
        'reference_write_evidence_present': False, 'reference_unchanged': False,
        'before_mismatch_count': 0, 'terminal_mismatch_count': None,
        'matching_packet_suffix_count': 0, 'changed_pixel_count': 0,
        'changed_to_value_count': 0, 'prior_reference_unchanged': False,
        'prior_matching_observation_count': None, 'prior_observation_index_complete': False,
        'changed_support_matches_oriented_half_surface': False,
    })

    def result(values=facts, refs=evidence_refs):
        return FrozenMap({'descriptive_facts': values, 'evidence_refs': refs,
                          'state_revision': snapshot.revision})

    executed = outcome.get('executed_action_contract') or FrozenMap()
    goal_ref = executed.get(profile['goal_contract_field'])
    goal = snapshot.term(goal_ref) if goal_ref else None
    if scene is None or goal is None or goal.attributes.get('goal_kind') != profile['goal_kind']:
        return result()
    history = outcome['observation_history']
    if len(history['prior_observation_rows']) > profile['maximum_frames']:
        raise ValueError('reference write history bound exceeded')
    if len(profile['surface_binding_fields']) != 2:
        raise ValueError('reference write requires exactly two surface bindings')
    try:
        before = repository.get(history['history_before_raw_input_ref'])
        after = repository.get(history['history_after_raw_input_ref'])
        prior = tuple(repository.get(row['raw_input_ref']).frame for row in history['prior_observation_rows'])
    except KeyError:
        return result()
    typed_scene = scene if isinstance(scene, VisualSceneDescription) else VisualSceneDescription.model_validate(scene)
    if typed_scene.frame.rows != before.frame:
        return result()
    views = measure_multi_resolution_views(MultiResolutionViewInput(
        scene=typed_scene, observation_ref=before.frame_id,
        maximum_distinct_visual_elements=profile['maximum_elements']))
    elements = views.descriptive_facts['distinct_visual_element_measurements']
    elements_by_ref = {row['element_ref']: row for row in elements}
    bound = []
    for field in profile['surface_binding_fields']:
        rows = tuple(row for row in elements if row['element_ref'] == goal.attributes.get(field))
        if not rows and field == profile['surface_binding_fields'][1]:
            current_output = unique_current_output(elements_by_ref, goal.attributes)
            rows = (current_output,) if current_output is not None else ()
        if len(rows) != 1 or rows[0]['area'] != rows[0]['bbox_area']:
            return result()
        bound.append(rows[0])
    if len(bound) != 2 or bound[0]['element_ref'] == bound[1]['element_ref']:
        return result()
    boxes = tuple(tuple(row[key] for key in ('bbox_top', 'bbox_left', 'bbox_bottom', 'bbox_right')) for row in bound)
    if (bound[0]['height'], bound[0]['width']) != (bound[1]['height'], bound[1]['width']):
        return result()
    measured = measure_surface_transition(before=before.frame,
        packet=after.metadata.get('transition_frames', ()), prior_frames=prior,
        reference_box=boxes[0], output_box=boxes[1],
        maximum_pixels=profile['maximum_pixels'], maximum_frames=profile['maximum_frames'])
    values = measured['changed_to_values']
    facts = FrozenMap.overlay(measured, FrozenMap({
        'reference_write_evidence_present': True,
        'official_boundary_present': outcome['official_boundary_present'],
        'transition_ref': outcome['transition_ref'],
        'source_goal_ref': goal.id, 'application_action_ref': executed['action_ref'],
        'source_intent_ref': executed['action_intent_id'],
        'unique_written_value': values[0] if len(values) == 1 else None,
        'changed_to_value_count': len(values),
        'changed_support_matches_oriented_half_surface': measure_half_surface_support(
            scene=typed_scene, reference=bound[0], output=bound[1], changed=measured['changed_support_lower_bound'],
            written_values=values, before=before.frame),
        'prior_observation_index_complete': history['prior_observation_index_complete'],
        'observation_evidence_digest': stable_digest((outcome['transition_ref'], measured, goal.id)),
    }))
    return result(facts, tuple(dict.fromkeys((*evidence_refs, goal.id))))
