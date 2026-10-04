"""Current geometric prerequisites, not a retrospective verdict or permission."""

from collections.abc import Mapping

from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest
from agents.yf_arc3_v5.capabilities.coupling_context_geometry import measure_translation_context


def _field(value, name):
    return value[name] if isinstance(value, Mapping) else getattr(value, name)


def measure_current_coupling_context(*, review, scene, tracking, context_epoch,
                                    focus_signatures, return_rows,
                                    maximum_steps, maximum_support_pixels):
    """Read only supplied canonical rows and this exact current observation.

    Preserve every matching return receipt. Missing evidence is unknown, not
    a false geometry or a supposed inverse. No status or eligibility is chosen.
    """
    if len(return_rows) > 64 or len(focus_signatures) > 64:
        raise ValueError("current coupling context evidence bound exceeded")
    frame_ref = _field(tracking, "frame_ref")
    refs = review["entity_refs"]
    entities = _field(tracking, "entities")
    if len(entities) > 128 or len(refs) != 2 or len(set(refs)) != 2:
        raise ValueError("current coupling context entity bound exceeded")
    by_ref = {_field(entity, "entity_id"): entity for entity in entities}
    if len(by_ref) != len(entities):
        raise ValueError("duplicate current coupling context entity")
    same_epoch = review["context_epoch"] == context_epoch
    old_focus = review.get("current_before_focus_signatures")
    focus_available = bool(old_focus) and bool(focus_signatures)
    focus_equal = tuple(old_focus) == tuple(focus_signatures) if focus_available else None
    receipt_refs = tuple(sorted(row["term_ref"] for row in return_rows
        if same_epoch and row["context_epoch"] == context_epoch and row["entity_refs"] == refs
        and bool(focus_signatures) and row["before_focus_signatures"] == focus_signatures
        and row["after_focus_signatures"] == focus_signatures))
    if len(receipt_refs) != len(set(receipt_refs)):
        raise ValueError("duplicate current coupling return receipt")
    geometry = measure_translation_context(scene=scene, tracking=tracking,
        entity_refs=refs, transformations=review["expected_transformations"] if same_epoch else (),
        maximum_steps=maximum_steps, maximum_support_pixels=maximum_support_pixels,
        entity_member_refs=review.get("entity_member_refs"), include_support_digests=True)
    outside = geometry["translated_support_outside_frame_counts"]
    contacts = geometry["swept_support_other_component_refs"]
    return FrozenMap.from_frozen_items(dict((*geometry.items(),
        ("current_context_token", stable_digest((review["term_ref"], frame_ref, context_epoch))),
        ("review_ref", review["term_ref"]), ("mapping_token", review["mapping_token"]),
        ("review_status", review["status"]),
        ("entity_refs", refs), ("action_ref", review["action_ref"]),
        ("context_epoch", context_epoch), ("current_frame_ref", frame_ref),
        ("current_focus_signatures", focus_signatures),
        ("current_focus_comparison_available", focus_available),
        ("current_focus_comparison_equal", focus_equal),
        ("current_focus_candidate_count", len(focus_signatures)),
        ("current_return_receipt_refs", receipt_refs),
        ("current_return_evidence_claim_refs", tuple(sorted({claim_ref for row in return_rows
            if row["term_ref"] in receipt_refs for claim_ref in row.get("evidence_claim_refs", ())}))),
        ("current_return_receipt_count", len(receipt_refs)),
        ("current_context_epoch_equal", same_epoch),
        ("current_outside_pixel_count", sum(outside) if outside is not None else None),
        ("current_intersected_component_count", len(contacts) if contacts is not None else None))))
