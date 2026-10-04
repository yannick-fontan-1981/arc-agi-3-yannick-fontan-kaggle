"""Bounded reference index over recorded action results, without reinterpreting pixels."""

from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest


def measure_terminal_observation_history(*, repository, before_ref, after_ref, profile):
    """Index the contiguous score partition ending just before the winning input.

    Transport ordinals, not lexicographic frame IDs, establish order. Missing or
    ambiguous ordinals never prove that a configuration did not occur. No raster
    is copied, and no predicate is evaluated by this inventory.
    """
    empty = {
        "prior_observation_rows": (), "prior_observation_count": 0,
        "prior_observation_index_complete": False,
        "prior_observation_index_digest": stable_digest(()),
        "history_before_raw_input_ref": before_ref,
        "history_after_raw_input_ref": after_ref,
        "history_index_issue": "endpoint_unavailable",
    }
    if repository is None or before_ref is None or after_ref is None:
        return FrozenMap(empty)
    def unavailable(issue):
        empty["history_index_issue"] = issue
        return FrozenMap(empty)
    try:
        before, after = repository.get(before_ref), repository.get(after_ref)
    except KeyError:
        return FrozenMap(empty)
    maximum = profile["maximum_history_inputs"]
    # The repository checks the bound before allocating a reference tuple.
    try:
        inputs = repository.bounded_values(maximum)
    except ValueError:
        return unavailable("history_input_bound_exceeded")
    ordinal_field = profile["history_ordinal_field"]
    scope_fields = profile["history_scope_fields"]
    scope = tuple(before.metadata.get(key) for key in scope_fields)
    end = after.metadata.get(ordinal_field)
    prior = before.metadata.get(ordinal_field)
    if (type(end) is not int or type(prior) is not int or end < 1 or prior != end - 1
            or tuple(after.metadata.get(key) for key in scope_fields) != scope
            or after.score <= before.score):
        return unavailable("endpoint_sequence_unverified")
    by_ordinal = {}
    for value in inputs:
        if tuple(value.metadata.get(key) for key in scope_fields) != scope:
            continue
        ordinal = value.metadata.get(ordinal_field)
        if type(ordinal) is not int or ordinal < 0:
            return unavailable("ordinal_unavailable")
        if ordinal <= end:
            by_ordinal.setdefault(ordinal, []).append(value)
    if any(len(by_ordinal[key]) != 1 for key in sorted(by_ordinal)):
        return unavailable("ordinal_ambiguous")
    indexed = {ordinal: group[0] for ordinal, group in sorted(by_ordinal.items())}
    rows, complete, issue = [], False, "ordinal_gap"
    for ordinal in range(end - 1, -1, -1):
        value = indexed.get(ordinal)
        if value is None:
            break
        if value.score != before.score:
            complete, issue = True, None
            break
        rows.append(FrozenMap({"raw_input_ref": value.raw_input_ref,
            "frame_id": value.frame_id, "step_index": ordinal,
            "score": value.score, "state": value.state}))
        if len(rows) > maximum:
            raise ValueError("terminal observation history bound exceeded")
        if ordinal == 0:
            complete, issue = True, None
    ordered = tuple(reversed(rows))
    return FrozenMap({
        "prior_observation_rows": ordered, "prior_observation_count": len(ordered),
        "prior_observation_index_complete": complete,
        "prior_observation_index_digest": stable_digest(ordered),
        "history_before_raw_input_ref": before_ref,
        "history_after_raw_input_ref": after_ref,
        "history_index_issue": issue,
    })
