"""Transport one typed input geometry through every agenda leaf, without choice."""

from agents.yf_arc3_v5.logos.types import FrozenMap


def transport_control_geometry(value, agenda):
    boxes = tuple((box.top, box.left, box.bottom, box.right) for box in value.input_aligned_entity_bboxes)
    delta = FrozenMap({"input_aligned_entity_bbox_tuples": boxes,
        "input_aligned_entity_bbox_count": len(boxes), "input_aligned_entity_bbox_bound": 64,
        "input_aligned_entity_bbox_bound_reached": len(boxes) >= 64})
    if len(boxes) > 64:
        raise ValueError("current control geometry bound exceeded")
    def bind(facts):
        for key, expected in sorted(delta.items()):
            if key in facts and facts[key] != expected:
                raise ValueError("agenda changed current control geometry: " + key)
        return FrozenMap.overlay(delta, facts)
    return agenda.model_copy(update={
        "context_facts": bind(agenda.context_facts),
        "alternative_facts": FrozenMap({ref: bind(facts) for ref, facts in sorted(agenda.alternative_facts.items())}),
        "descriptive_delta_facts": FrozenMap({ref: bind(facts) for ref, facts in sorted(agenda.descriptive_delta_facts.items())}),
    })
