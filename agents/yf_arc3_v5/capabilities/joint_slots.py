"""Bounded slot geometry, exact permutation matching and requested finite search.

No role, goal, operator or route is selected here. Callers supply constraints
and traversal orders; the result preserves bounds and ambiguity explicitly.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence

from agents.yf_arc3_v5.logos.types import stable_digest
from .contracts import ComponentDescription
from .distributed_square_tracks import measure_distributed_square_structure


def measure_joint_slot_geometry(
    components: Sequence[ComponentDescription],
) -> dict[str, object]:
    if len(components) > 256:
        return {}
    # Bound the family before the rectangular description enumeration.
    counts: dict[int, int] = {}
    for c in components:
        if c.bbox.height == c.bbox.width and c.area == c.bbox.height ** 2:
            counts[c.area] = counts.get(c.area, 0) + 1
    if any(count > 64 for count in sorted(counts.values())):
        return {}
    measured = measure_distributed_square_structure(None, components)
    structures = tuple(measured.get("structures", ()))
    markers = tuple(measured.get("markers", ()))
    associations = tuple(measured.get("associations", ()))
    if not (2 <= len(structures) <= 3 and 2 <= len(markers) <= 8
            and 2 <= len(associations) <= 16
            and measured.get("shared_slot_centers_twice")):
        return {}
    if any(m["component_count"] != 4 for m in markers):
        return {}
    slots = {tuple(p) for s in structures for p in s["slot_centers_twice"]}
    if len(slots) != measured["carrier_count"] or len(slots) > 64:
        return {}
    # A morphology correspondence is reported only when the same number of
    # complete marked positions and square occurrences carries each value.
    values = {tuple(p): v for s in structures
              for p, v in zip(s["slot_centers_twice"], s["slot_values"])}
    marker_values = {m["visual_value"] for m in markers}
    if any(sum(v == value for v in sorted(values.values())) !=
           sum(m["visual_value"] == value for m in markers) for value in marker_values):
        return {}
    return {
        **measured,
        "geometry_ref": "measurement.joint_slots." + stable_digest(
            (structures, markers, associations)
        )[:16],
        "slot_values_by_position": tuple((p, values[p]) for p in sorted(values)),
    }


def measure_supported_slot_rotation(
    before: Mapping[str, object], after: Mapping[str, object],
) -> dict[str, object]:
    """Match a nonzero unit rotation and unchanged complement on all slots."""
    old = {tuple(p): v for p, v in before.get("slot_values_by_position", ())}
    new = {tuple(p): v for p, v in after.get("slot_values_by_position", ())}
    if not old or set(old) != set(new):
        return {}
    matches = []
    for item in before.get("structures", ()):
        slots = tuple(tuple(p) for p in item["slot_centers_twice"])
        if any(old[p] != new[p] for p in old if p not in slots):
            continue
        if all(old[p] == new[p] for p in slots):
            continue
        for offset in (-1, 1):
            if all(new[slots[(i + offset) % len(slots)]] == old[p]
                   for i, p in enumerate(slots)):
                matches.append((slots, offset))
    if len(matches) != 1:
        return {"closed_support_permutation_match_count": len(matches)}
    slots, offset = matches[0]
    prior = tuple(old[p] for p in slots)
    current = tuple(new[p] for p in slots)
    return {
        "closed_support_permutation_match_count": 1,
        "unique_closed_support_permutation": True,
        "unique_closed_support_signed_slot_quantum": offset,
        "closed_support_one_slot_quantum": True,
        "closed_support_nonzero_permutation": True,
        "closed_support_slot_count": len(slots),
        "closed_support_slot_centers_twice": slots,
        "closed_support_before_value_sequence": prior,
        "closed_support_after_value_sequence": current,
        "closed_support_before_state_digest": stable_digest(prior)[:16],
        "closed_support_after_state_digest": stable_digest(current)[:16],
    }


def search_slot_permutations(
    *, slot_values_by_position: Sequence, requirements: Sequence,
    operators: Sequence[Mapping], traversal_orders: Sequence[Sequence[str]],
    maximum_depth: int, maximum_states: int,
    state_equivalence: str = "exact_values",
) -> dict[str, object]:
    """One shortest witness per supplied order, with at most three witnesses."""
    if not (0 < maximum_depth <= 40 and 0 < maximum_states <= 16384
            and 0 < len(traversal_orders) <= 3 and 0 < len(operators) <= 16):
        raise ValueError("slot search requires explicit finite bounds")
    values = {tuple(p): int(v) for p, v in slot_values_by_position}
    slots = tuple(sorted(values))
    index = {p: i for i, p in enumerate(slots)}
    if not 0 < len(slots) <= 64 or not 0 < len(requirements) <= 8:
        raise ValueError("slot search domain exceeds its bound")
    constraints = tuple((index[tuple(p)], int(v)) for p, v in requirements)
    maps = {}
    for op in operators:
        domain = tuple(index[tuple(p)] for p in op["slot_centers_twice"])
        if len(set(domain)) != len(domain) or not domain:
            raise ValueError("slot operator must have a bijective domain")
        mapping = list(range(len(slots)))
        for i, source in enumerate(domain):
            mapping[source] = domain[(i + int(op["signed_slot_quantum"])) % len(domain)]
        ref = str(op["candidate_ref"])
        if ref in maps:
            raise ValueError("slot operators must have distinct identities")
        maps[ref] = tuple(mapping)
    routes = []
    counts = []
    if state_equivalence not in {"exact_values", "terminal_value_membership"}:
        raise ValueError("slot search requires a declared state equivalence")
    full_initial = tuple(values[p] for p in slots)
    required_values = frozenset(v for _, v in constraints)
    initial = (tuple(v if v in required_values else None for v in full_initial)
               if state_equivalence == "terminal_value_membership" else full_initial)
    for order in traversal_orders:
        if len(order) != len(maps) or set(order) != set(maps):
            raise ValueError("traversal must list each declared operator once")
        queue = deque([(initial, ())])
        visited = {initial}
        found = None
        while queue:
            state, path = queue.popleft()
            if all(state[i] == v for i, v in constraints):
                found = path
                break
            if len(path) == maximum_depth:
                continue
            for ref in order:
                successor = list(state)
                for source, destination in enumerate(maps[ref]):
                    successor[destination] = state[source]
                successor = tuple(successor)
                if successor in visited:
                    continue
                if len(visited) >= maximum_states:
                    return {"routes": (), "bound_reached": True,
                            "visited_counts": (*counts, len(visited))}
                visited.add(successor)
                queue.append((successor, (*path, ref)))
        counts.append(len(visited))
        if found is not None and found not in routes:
            # Check the witness on the complete original field. The quotient
            # never weakens a terminal requirement or invents an operator.
            verified = full_initial
            for ref in found:
                successor = list(verified)
                for source, destination in enumerate(maps[ref]):
                    successor[destination] = verified[source]
                verified = tuple(successor)
            if not all(verified[i] == v for i, v in constraints):
                raise ValueError("projected slot witness fails full-state requirements")
            routes.append(found)
    return {"routes": tuple(routes), "bound_reached": False,
            "visited_counts": tuple(counts)}
