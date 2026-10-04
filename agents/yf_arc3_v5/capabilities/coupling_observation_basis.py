"""Exact support-return and decoration equality, never a control decision.

The observations are bounded to the supplied scene epoch. A geometric return
is not a global inverse operator and does not exclude obstacles or boundaries.
"""

from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest


def _rigid_translation(descriptor):
    required = ("delta_row", "delta_col", "delta_height", "delta_width", "delta_area",
                "value_before", "value_after", "shape_before", "shape_after")
    return (all(key in descriptor for key in required)
            and all(descriptor[key] == 0 for key in ("delta_height", "delta_width", "delta_area"))
            and descriptor["value_before"] == descriptor["value_after"]
            and descriptor["shape_before"] == descriptor["shape_after"]
            and (descriptor["delta_row"] != 0 or descriptor["delta_col"] != 0))


def measure_pair_observation_basis(*, pair_row, entity_rows, previous_rows,
                                   before_focus_signatures, after_focus_signatures,
                                   transition_ref, observed_return_rows=()):
    if len(entity_rows) > 128 or len(previous_rows) > 64:
        raise ValueError("pair observation basis bound exceeded")
    if len(before_focus_signatures) > 64 or len(after_focus_signatures) > 64:
        raise ValueError("decoration observation bound exceeded; no ties truncated")
    if len(observed_return_rows) > 64:
        raise ValueError("canonical support return row bound exceeded")
    current = {row["entity_ref"]: row for row in entity_rows}
    if len(current) != len(entity_rows):
        raise ValueError("duplicate observation basis entity")
    refs = pair_row["entity_refs"]
    current_rows = tuple(current.get(ref) for ref in refs)
    current_match = (all(row is not None for row in current_rows)
                     and tuple(row["descriptor"] for row in current_rows) == pair_row["observed_transformations"])
    old_exact = tuple(row for row in previous_rows if row["mapping_token"] == pair_row["mapping_token"])
    if len(old_exact) > 1:
        raise ValueError("duplicate canonical mapping basis")
    old = old_exact[0] if old_exact else FrozenMap()
    before_digests = tuple(row.get("before_support_digest") for row in current_rows) if current_match else None
    after_digests = tuple(row.get("after_support_digest") for row in current_rows) if current_match else None
    inverse_refs, same_focus_inverse_refs, same_focus_mapping_tokens = [], [], []
    if (current_match and all(before_digests) and all(after_digests)
            and all(_rigid_translation(row["descriptor"]) for row in current_rows)):
        for prior in previous_rows:
            if (prior["entity_refs"] != refs or prior["context_epoch"] != pair_row["context_epoch"]
                    or prior.get("basis_after_support_digests") != before_digests
                    or prior.get("basis_before_support_digests") != after_digests
                    or not prior.get("basis_transition_ref")
                    or prior["basis_transition_ref"] == transition_ref):
                continue
            descriptors = prior["observed_transformations"]
            if not all(_rigid_translation(desc) for desc in descriptors):
                continue
            if any((old_desc["delta_row"] + new_row["descriptor"]["delta_row"] != 0
                    or old_desc["delta_col"] + new_row["descriptor"]["delta_col"] != 0)
                   for old_desc, new_row in zip(descriptors, current_rows)):
                continue
            inverse_refs.append(prior["basis_transition_ref"])
            if (len(before_focus_signatures) == 1
                    and before_focus_signatures == after_focus_signatures
                    and prior.get("basis_before_focus_signatures") == before_focus_signatures
                    and prior.get("basis_after_focus_signatures") == before_focus_signatures):
                same_focus_inverse_refs.append(prior["basis_transition_ref"])
                same_focus_mapping_tokens.append(prior["mapping_token"])
    # On an incompatible observation the original mapping retains its own basis,
    # not the footprint/selection of the newly observed alternative.
    basis = FrozenMap({
        "basis_transition_ref": transition_ref if current_match else old.get("basis_transition_ref"),
        "basis_before_support_digests": before_digests if current_match else old.get("basis_before_support_digests"),
        "basis_after_support_digests": after_digests if current_match else old.get("basis_after_support_digests"),
        "basis_before_focus_signatures": before_focus_signatures if current_match else old.get("basis_before_focus_signatures", ()),
        "basis_after_focus_signatures": after_focus_signatures if current_match else old.get("basis_after_focus_signatures", ()),
    })
    expected_focus = old.get("basis_before_focus_signatures") or ()
    expected_after_focus = old.get("basis_after_focus_signatures") or ()
    singleton_current = len(before_focus_signatures) == len(after_focus_signatures) == 1
    singleton_expected = len(expected_focus) == len(expected_after_focus) == 1
    equal_focus = singleton_current and singleton_expected and before_focus_signatures == after_focus_signatures == expected_focus == expected_after_focus
    return_refs = tuple(sorted(set(row["term_ref"] for row in observed_return_rows
        if row["entity_refs"] == refs and row["context_epoch"] == pair_row["context_epoch"]
        and len(before_focus_signatures) == 1 and before_focus_signatures == after_focus_signatures
        and row["before_focus_signatures"] == before_focus_signatures
        and row["after_focus_signatures"] == before_focus_signatures)))
    return FrozenMap.from_frozen_items(dict((*basis.items(),
        ("current_before_focus_signatures", before_focus_signatures),
        ("current_after_focus_signatures", after_focus_signatures),
        ("focus_comparison_available", singleton_current and singleton_expected),
        ("focus_comparison_equal", equal_focus),
        ("inverse_translation_observation_refs", tuple(sorted(set(inverse_refs)))),
        ("inverse_translation_same_focus_observation_refs", tuple(sorted(set(same_focus_inverse_refs)))),
        ("inverse_translation_same_focus_mapping_tokens", tuple(sorted(set(same_focus_mapping_tokens)))),
        ("inverse_translation_same_focus_match_present", bool(same_focus_inverse_refs)),
        ("observed_pair_return_term_refs", return_refs),
        ("observed_pair_return_match_present", bool(return_refs)),
        ("pair_return_basis_token", stable_digest((pair_row["mapping_token"], tuple(sorted(set(same_focus_mapping_tokens))), before_focus_signatures))),
    )))
