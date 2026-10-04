"""Exact historical geometry joined to the two actually dispatched commands."""

from types import SimpleNamespace

from agents.yf_arc3_v5.capabilities import BoundingBox, FrameGrid
from agents.yf_arc3_v5.capabilities.palette_support_routes import measure_palette_support_routes
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest
from agents.yf_arc3_v5.state.terminal_executed_evidence import measure_executed_action_contract


def measure_terminal_entry_geometry(*, snapshot, repository, store, before_ref,
                                    before_frame, executed, history, profile):
    contract = profile.get("entry_geometry_contract")
    empty = FrozenMap({"entry_geometry_present": False})
    if not contract or not executed or repository is None or store is None:
        return empty
    if any(executed.get(field) != expected for field, expected in sorted(contract["action_equalities"].items())):
        return empty
    rows = history.get("prior_observation_rows", ())
    if len(rows) < 2 or not history.get("prior_observation_index_complete"):
        return empty
    previous = measure_executed_action_contract(store=store, after_ref=before_ref, profile=profile)
    if not previous or previous.get("action_ref") != executed.get("action_ref"):
        return empty
    if rows[-1]["frame_id"] != executed["frame_id"] or rows[-2]["frame_id"] != previous["frame_id"]:
        return empty

    def support(action):
        terms = snapshot.terms_for_literal_attribute("operational_scene_ref", action["frame_id"])
        if len(terms) > contract["maximum_scene_terms"]:
            return None
        candidates = {}
        for term in terms:
            attrs = term.attributes
            if attrs.get(contract["role_attribute"]) != contract["role_value"]:
                continue
            refs = tuple(attrs.get("role_premise_claim_refs", ()))
            if not refs or not set(refs).issubset(action.get("canonical_premise_claim_refs", ())):
                continue
            values = tuple(attrs.get(field) for field in ("bbox_top", "bbox_left", "bbox_height", "bbox_width"))
            if any(type(value) is not int for value in values) or not attrs.get("support_digest"):
                continue
            top, left, height, width = values
            if min(top, left) < 0 or min(height, width) < 1:
                continue
            box = BoundingBox(top=top, left=left, bottom=top + height - 1, right=left + width - 1)
            candidates[(values, attrs["support_digest"])] = (box, attrs, refs)
        return next(iter(candidates.values())) if len(candidates) == 1 else None

    current, prior = support(executed), support(previous)
    if current is None or prior is None or before_frame is None:
        return empty
    body, attrs, refs = current
    old, old_attrs, old_refs = prior
    if body.height != old.height or body.width != old.width or body.height != body.width:
        return empty
    dr, dc = body.top - old.top, body.left - old.left
    if abs(dr) + abs(dc) != body.width:
        return empty
    prior_frame = repository.get(rows[-2]["raw_input_ref"]).frame

    def pixels(frame, box):
        if box.bottom >= len(frame) or box.right >= len(frame[0]):
            return None
        return tuple((r - box.top, c - box.left, frame[r][c])
                     for r in range(box.top, box.bottom + 1) for c in range(box.left, box.right + 1))

    current_pixels, previous_pixels = pixels(before_frame, body), pixels(prior_frame, old)
    if current_pixels is None or previous_pixels is None:
        return empty
    if stable_digest(current_pixels) != attrs["support_digest"] or stable_digest(previous_pixels) != old_attrs["support_digest"]:
        return empty
    palette = tuple(sorted({p[2] for p in current_pixels}))
    if palette != tuple(sorted({p[2] for p in previous_pixels})):
        return empty
    action = executed["action_ref"]
    measured = measure_palette_support_routes(SimpleNamespace(
        route_generation_policy={"palette_support_route_measurement": contract["route_measurement"]},
        canonical_provisional_control_geometry_bboxes=(body,),
        canonical_provisional_control_term_facts=(FrozenMap({"support_digest": attrs["support_digest"]}),),
        frame=FrameGrid(rows=before_frame), current_transition_change_bboxes=(),
        current_input_aligned_underlay_values=(), visited_input_aligned_entity_bboxes=(old, body),
        interface_action_translation_deltas=((action, dr // body.height, dc // body.width),),
        available_action_refs=(action,), last_action_ref=action,
    ))
    if len(measured) != 1 or measured[0]["action_refs"] != (action,):
        return empty
    target = measured[0]["region_bbox"]
    top, left, bottom, right = target
    if any(before_frame[r][c] != prior_frame[r][c] for r in range(top, bottom + 1) for c in range(left, right + 1)):
        return empty
    return FrozenMap({
        "entry_geometry_present": True,
        "entry_palette_values": palette,
        "entry_relation_token": stable_digest(palette),
        "entry_previous_delta": (dr, dc),
        "entry_before_bbox": (body.top, body.left, body.bottom, body.right),
        "entry_region_bbox": target,
        "entry_predicted_bbox": (body.top + dr, body.left + dc, body.bottom + dr, body.right + dc),
        "entry_geometry_premise_refs": tuple(dict.fromkeys((*refs, *old_refs))),
        "entry_executed_intent_ref": executed["action_intent_id"],
        "entry_previous_intent_ref": previous["action_intent_id"],
    })
