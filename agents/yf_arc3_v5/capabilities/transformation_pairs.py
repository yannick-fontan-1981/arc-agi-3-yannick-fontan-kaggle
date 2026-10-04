"""Exact, bounded transition descriptors and recurrence joins; no role inference."""

from itertools import combinations

from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest
from agents.yf_arc3_v5.capabilities.pair_observation_history import PairObservationHistory


def measure_entity_changes(before, after):
    previous = {item.entity_id: item for item in before.entities}
    rows = []
    if len(after.entities) > 128:
        return (), True
    for current in after.entities:
        old = previous.get(current.entity_id)
        if old is None:
            continue  # Identity is not invented for a newly segmented component.
        first, second = old.component, current.component
        descriptor = FrozenMap({
            "delta_row": current.delta_row, "delta_col": current.delta_col,
            "delta_height": second.bbox.height - first.bbox.height,
            "delta_width": second.bbox.width - first.bbox.width,
            "delta_area": second.area - first.area,
            "value_before": first.value, "value_after": second.value,
            "shape_before": stable_digest(first.relative_pixels),
            "shape_after": stable_digest(second.relative_pixels),
        })
        changed = (descriptor["delta_row"] != 0 or descriptor["delta_col"] != 0
                   or descriptor["delta_height"] != 0 or descriptor["delta_width"] != 0
                   or descriptor["delta_area"] != 0 or descriptor["value_before"] != descriptor["value_after"]
                   or descriptor["shape_before"] != descriptor["shape_after"])
        rows.append(FrozenMap({"entity_ref": current.entity_id, "descriptor": descriptor, "changed": changed,
            "before_support_digest": stable_digest((first.value, first.pixels)),
            "after_support_digest": stable_digest((second.value, second.pixels))}))
    return tuple(rows), False


def measure_pair_recurrence(*, entity_rows, previous_rows, action_ref, transition_ref, context_epoch, max_pairs=64,
                            operator_scope_ref=None):
    """Join exact pairs requested by Source, retaining incompatible records.

    Repetition is not a causal verdict. The returned booleans/counts are only
    equality measurements; DRM alone assigns an epistemic status.
    """
    if not 1 <= max_pairs <= 64 or len(entity_rows) > 128:
        raise ValueError("transformation pair bound exceeded")
    history = PairObservationHistory.from_rows(previous_rows)
    current = {item["entity_ref"]: item for item in entity_rows}
    if len(current) != len(entity_rows):
        raise ValueError("duplicate transformation entity")
    changed = sorted(ref for ref, row in current.items() if row["changed"])
    def members(ref):
        return frozenset(current[ref].get("member_refs") or (ref,))
    # Overlapping descriptions are not distinct physical partners.
    pairs = {(first, second) for first, second in combinations(changed, 2)
             if members(first).isdisjoint(members(second))}
    for old in history.iter_rows():
        if (old["action_ref"] != action_ref or old["context_epoch"] != context_epoch
                or old.get("operator_scope_ref") != operator_scope_ref):
            continue
        first, second = old["entity_refs"]
        # An observed lack of both effects is still comparable evidence. It is
        # not the same as losing an entity during segmentation or occlusion.
        if first in current and second in current and members(first).isdisjoint(members(second)):
            pairs.add((first, second))
    if len(pairs) > max_pairs:
        return FrozenMap({"pair_rows": (), "pair_row_count": 0, "pair_enumeration_truncated": True, "measurement_available": True})
    result = []
    for first, second in sorted(pairs):
        pair_start = len(result)
        descriptors = (current[first]["descriptor"], current[second]["descriptor"])
        pair_basis = (context_epoch, action_ref, first, second)
        pair_token = stable_digest(pair_basis if operator_scope_ref is None else (*pair_basis, operator_scope_ref))
        mapping_token = stable_digest((pair_token, descriptors))
        comparable = tuple(row for row in history.rows_for((first, second)) if row["pair_token"] == pair_token)
        exact = tuple(row for row in comparable if row["mapping_token"] == mapping_token)
        previous_transitions = tuple(dict.fromkeys(ref for row in exact for ref in row["sample_transition_refs"]))
        if transition_ref in previous_transitions:
            continue  # Re-reading one observation never raises confidence.
        if len(previous_transitions) >= 64:
            raise ValueError("transformation pair observation bound exceeded")
        result.append(FrozenMap({
            "pair_token": pair_token, "mapping_token": mapping_token,
            "entity_refs": (first, second), "observed_transformations": descriptors,
            "entity_member_refs": tuple(tuple(sorted(members(ref))) for ref in (first, second)),
            "entity_collective_observations": tuple(current[ref].get("collective_observation") for ref in (first, second)),
            "action_ref": action_ref, "context_epoch": context_epoch,
            "operator_scope_ref": operator_scope_ref,
            "sample_transition_refs": previous_transitions + (transition_ref,),
            "contrary_transition_refs": tuple(dict.fromkeys(ref for row in exact for ref in row["contrary_transition_refs"])),
            "prior_exact_match_present": bool(exact),
            "prior_incompatible_match_present": any(row["mapping_token"] != mapping_token for row in comparable),
        }))
        for old in comparable:
            if old["mapping_token"] == mapping_token or transition_ref in old["contrary_transition_refs"]:
                continue
            contrary_refs = old["contrary_transition_refs"] + (transition_ref,)
            if len(contrary_refs) > 64:
                raise ValueError("transformation pair contradiction bound exceeded")
            result.append(FrozenMap({
                "pair_token": pair_token, "mapping_token": old["mapping_token"],
                "entity_refs": old["entity_refs"], "observed_transformations": old["observed_transformations"],
                "entity_member_refs": old.get("entity_member_refs") or tuple((ref,) for ref in old["entity_refs"]),
                "entity_collective_observations": old.get("entity_collective_observations", (None, None)),
                "action_ref": action_ref, "context_epoch": context_epoch,
                "operator_scope_ref": operator_scope_ref,
                "sample_transition_refs": old["sample_transition_refs"], "contrary_transition_refs": contrary_refs,
                "prior_exact_match_present": True, "prior_incompatible_match_present": True,
            }))
        # Pair count and contradictory mapping variants are distinct bounds.
        # Preserve all updates for a pair; never truncate to a global row prefix.
        if len(result) - pair_start > 64:
            return FrozenMap({"pair_rows": (), "pair_row_count": 0, "pair_enumeration_truncated": True, "measurement_available": True})
    return FrozenMap({"pair_rows": tuple(result), "pair_row_count": len(result), "pair_enumeration_truncated": False, "measurement_available": True})
