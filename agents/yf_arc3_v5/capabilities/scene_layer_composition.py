"""Exact counterfactual object masks, without role or action authorization.

The supplied actors move under supplied translations. Cover components are
also measured under an explicitly requested zero-displacement counterfactual;
their actual last displacement and observation count remain separate facts.
No drawing order is invented: each overlap retains every possible pixel value.
"""

from collections.abc import Mapping

from agents.yf_arc3_v5.logos.types import FrozenMap


def _field(value, name):
    return value[name] if isinstance(value, Mapping) else getattr(value, name)


def measure_scene_layer_composition(*, before_scene, before_tracking, after_scene,
                                   entity_refs, transformations, render_account,
                                   action_ref, maximum_layers,
                                   maximum_support_pixels):
    if not 2 <= maximum_layers <= 16 or not 1 <= maximum_support_pixels <= 8192:
        raise ValueError("invalid scene layer composition bound")
    unknown = FrozenMap({
        "scene_layer_composition_available": False,
        "scene_layer_composition_within_bound": True,
        "scene_layer_rows": None,
        "scene_layer_overlap_render_samples": None,
        "scene_layer_total_support_count": None,
        "scene_layer_unexplained_pixel_count": None,
        "scene_layer_unexplained_extension_count": None,
        "scene_layer_outside_pixel_count": None,
        "scene_layer_cover_min_observation_count": None,
        "scene_layer_cover_nonzero_last_delta_count": None,
        "scene_layer_cover_untracked_count": None,
        "scene_layer_same_last_action": _field(before_tracking, "action_ref") == action_ref,
    })

    def bounded_unknown():
        return FrozenMap.from_frozen_items(dict((*unknown.items(),
            ("scene_layer_composition_within_bound", False))))

    if not render_account["translation_render_available"]:
        return unknown
    entities = _field(before_tracking, "entities")
    prior_components = _field(_field(before_scene, "blocks"), "components")
    current_components = _field(_field(after_scene, "blocks"), "components")
    if any(len(items) > 128 for items in (entities, prior_components, current_components)):
        return bounded_unknown()
    by_entity = {_field(item, "entity_id"): item for item in entities}
    by_component = {_field(item, "component_id"): item for item in prior_components}
    tracking_by_component = {_field(_field(item, "component"), "component_id"): item for item in entities}
    if (len(entity_refs) != 2 or len(set(entity_refs)) != 2 or len(transformations) != 2
            or any(ref not in by_entity for ref in entity_refs)):
        return unknown
    if len(by_entity) != len(entities) or len(by_component) != len(prior_components):
        return unknown
    # The render account has already checked comparability of each rigid
    # descriptor with this exact before support. Never accept a different pair.
    from agents.yf_arc3_v5.capabilities.coupling_render_accounting import measure_translation_render
    measured = measure_translation_render(before_scene=before_scene, before_tracking=before_tracking,
        after_scene=after_scene, entity_refs=entity_refs, transformations=transformations,
        maximum_steps=64, maximum_support_pixels=maximum_support_pixels)
    if not measured["translation_render_within_bound"]:
        return bounded_unknown()
    if measured != render_account:
        return unknown
    own_refs = tuple(_field(_field(by_entity[ref], "component"), "component_id") for ref in entity_refs)
    cover_refs = tuple(sorted({row["component_ref"]
        for rows in render_account["replaced_pixel_prior_component_rows"] for row in rows
        if not row["frame_enclosing"] and row["component_ref"] not in own_refs}))
    if len(entity_refs) + len(cover_refs) > maximum_layers:
        return bounded_unknown()
    if any(ref not in by_component for ref in cover_refs):
        return unknown
    rows = _field(_field(after_scene, "frame"), "rows")
    height, width = len(rows), len(rows[0])
    layer_inputs = tuple((_field(by_entity[ref], "component"),
                          descriptor["delta_row"], descriptor["delta_col"], ref)
                         for ref, descriptor in zip(entity_refs, transformations)) + tuple(
        (by_component[ref], 0, 0,
         _field(tracking_by_component[ref], "entity_id") if ref in tracking_by_component else None)
        for ref in cover_refs)
    total = sum(_field(component, "area") for component, _, _, _ in layer_inputs)
    if total > maximum_support_pixels:
        return bounded_unknown()
    projected = []
    possible_values = {}
    support_multiplicity = {}
    outside = 0
    for component, dr, dc, entity_ref in layer_inputs:
        pixels = tuple((row + dr, col + dc) for row, col in _field(component, "pixels"))
        bbox = _field(component, "bbox")
        top, left = _field(bbox, "top") + dr, _field(bbox, "left") + dc
        bottom, right = _field(bbox, "bottom") + dr, _field(bbox, "right") + dc
        value = _field(component, "value")
        geometry = FrozenMap({
            "component_id": _field(component, "component_id"), "value": value,
            "pixels": pixels, "relative_pixels": _field(component, "relative_pixels"),
            "bbox": FrozenMap({"top": top, "left": left, "bottom": bottom, "right": right}),
            "area": len(pixels), "touches_frame_boundary":
                top <= 0 or left <= 0 or bottom >= height - 1 or right >= width - 1,
            "schema_version": "yf_arc3_v5.component_description.v1",
        })
        projected.append((component, geometry, frozenset(pixels), entity_ref, dr, dc))
        for row, col in pixels:
            if not (0 <= row < height and 0 <= col < width):
                outside += 1
                continue
            possible_values.setdefault((row, col), set()).add(value)
            support_multiplicity[(row, col)] = support_multiplicity.get((row, col), 0) + 1
    union = frozenset(possible_values)
    unexplained = sum(rows[row][col] not in values
                      for (row, col), values in sorted(possible_values.items()))
    layer_rows, extension_pixels = [], set()
    for component, geometry, support, entity_ref, dr, dc in projected:
        observation_refs = []
        for current in current_components:
            if _field(current, "value") != geometry["value"]:
                continue
            current_pixels = frozenset(_field(current, "pixels"))
            if support & current_pixels:
                observation_refs.append(_field(current, "component_id"))
                extension_pixels.update(current_pixels - union)
        covered = sum(support_multiplicity.get(pixel, 0) > 1 for pixel in support)
        prior = tracking_by_component.get(_field(component, "component_id"))
        layer_rows.append(FrozenMap({
            "prior_entity_ref": entity_ref, "component_geometry": geometry,
            "prediction_delta_row": dr, "prediction_delta_col": dc,
            "observation_component_refs": tuple(observation_refs),
            "overlap_pixel_count": covered,
            "last_observed_delta_row": _field(prior, "delta_row") if prior is not None else None,
            "last_observed_delta_col": _field(prior, "delta_col") if prior is not None else None,
            "observation_count": _field(prior, "observation_count") if prior is not None else 0,
        }))
    covers = layer_rows[len(entity_refs):]
    from agents.yf_arc3_v5.capabilities.overlap_render_samples import measure_overlap_render_samples
    try:
        samples = measure_overlap_render_samples(layers=layer_rows, rendered_rows=rows,
            maximum_regions=maximum_layers * maximum_layers,
            maximum_support_pixels=maximum_support_pixels)
    except ValueError:
        return bounded_unknown()
    return FrozenMap({
        "scene_layer_composition_available": True,
        "scene_layer_composition_within_bound": True,
        "scene_layer_rows": tuple(layer_rows),
        "scene_layer_overlap_render_samples": samples,
        "scene_layer_total_support_count": total,
        "scene_layer_unexplained_pixel_count": unexplained,
        "scene_layer_unexplained_extension_count": len(extension_pixels),
        "scene_layer_outside_pixel_count": outside,
        "scene_layer_cover_min_observation_count": min((row["observation_count"] for row in covers), default=0),
        "scene_layer_cover_nonzero_last_delta_count": sum(
            row["last_observed_delta_row"] not in (None, 0)
            or row["last_observed_delta_col"] not in (None, 0) for row in covers),
        "scene_layer_cover_untracked_count": sum(row["prior_entity_ref"] is None for row in covers),
        "scene_layer_same_last_action": unknown["scene_layer_same_last_action"],
    })


def materialize_component_layer_view(*, observed_components, layer_rows,
                                     maximum_layers, maximum_support_pixels):
    """Mechanically substitute only Source-supplied masks in an alternative.

    The original extraction is never changed. This function does not decide
    whether the alternative can ground an action; unexplained paints, identity,
    bounds, selection and evidence have to be interpreted by DRM/SRC first.
    """
    if not 2 <= maximum_layers <= 16 or not 1 <= maximum_support_pixels <= 8192:
        raise ValueError("invalid component layer view bound")
    components = _field(observed_components, "components")
    if len(components) > 128 or len(layer_rows) > maximum_layers:
        raise ValueError("component layer view exceeds component bound")
    if sum(_field(row["component_geometry"], "area") for row in layer_rows) > maximum_support_pixels:
        raise ValueError("component layer view exceeds support bound")
    removed = frozenset(ref for row in layer_rows for ref in row["observation_component_refs"])
    observed_refs = frozenset(_field(component, "component_id") for component in components)
    if not removed <= observed_refs:
        raise ValueError("component layer view cites a missing observed component")
    replacements = tuple(row["component_geometry"] for row in layer_rows)
    geometry_fields = ("component_id", "value", "pixels", "relative_pixels", "bbox", "area",
                       "touches_frame_boundary", "schema_version")
    retained = tuple(component if isinstance(component, Mapping) else
        FrozenMap.from_frozen_items({name: _field(component, name) for name in geometry_fields})
        for component in components if _field(component, "component_id") not in removed)
    combined = (*retained, *replacements)
    refs = tuple(_field(component, "component_id") for component in combined)
    if len(refs) != len(set(refs)):
        raise ValueError("component layer view has unresolved component reference collisions")
    # Return interned descriptive views, not a revalidated whole scene. Typed
    # capability adapters may validate this small extraction at their boundary.
    return FrozenMap({"components": tuple(sorted(combined, key=lambda component: _field(component, "component_id"))),
                      "schema_version": "yf_arc3_v5.component_extraction_result.v1"})
