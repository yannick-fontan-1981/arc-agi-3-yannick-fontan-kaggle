"""Exact scene-free recognition keys bound to each observed effect slot.

No roles or control are inferred. Whitelists and bounds are declared in DRM.
The relative effect/appearance pairing is retained when actor order changes.
"""

from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest
from agents.yf_arc3_v5.capabilities.shared_transformation_pattern import effect_descriptor_fields


def index_effect_bearers(entities, pair_rows, maximum_bearers):
    """Dereference only the bounded measured pair inputs, not unrelated scenery."""
    refs = frozenset(ref for row in pair_rows for ref in row["entity_refs"])
    if len(refs) > maximum_bearers:
        raise ValueError("shared effect bearer bound exceeded")
    return {entity.entity_id: entity for entity in entities if entity.entity_id in refs}


def measure_shared_effect_slots(*, descriptors, appearances, contexts, profile):
    if len(descriptors) != profile["required_effect_count"]:
        return ()
    if len(appearances) != len(descriptors) or len(contexts) != len(descriptors):
        raise ValueError("effect appearance arity mismatch")
    rows = []
    for descriptor, appearance, context_rows in zip(descriptors, appearances, contexts):
        if appearance is None:
            # Losing an actor does not license another actor's signature.
            return ()
        if len(appearance.keys) > profile["maximum_signature_keys"]:
            raise ValueError("effect appearance key bound exceeded")
        effect = FrozenMap({field: descriptor[field] for field in effect_descriptor_fields(descriptor, profile)})
        values = tuple(FrozenMap({field: item["values"][field]
            for field in profile["context_fields"] if field in item["values"]})
            for item in context_rows)
        rows.append(FrozenMap({"effect": effect,
            "signature_keys": tuple(FrozenMap(key.model_dump()) for key in appearance.keys),
            "appearance_contract_source_hash": appearance.contract_source_hash,
            "context_alternatives": tuple(sorted(set(values), key=stable_digest))}))
    return tuple(sorted(rows, key=stable_digest))
