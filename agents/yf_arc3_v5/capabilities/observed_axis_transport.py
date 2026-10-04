"""Bounded comparisons of sealed intervention memberships, without action release."""
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest


def measure_observed_axis_transport(*, observations, present_entity_refs, context_epoch):
    if len(observations) > 64 or len(present_entity_refs) > 128:
        raise ValueError("axis transport observation bound exceeded")
    present = frozenset(present_entity_refs)
    if len(present) != len(present_entity_refs):
        raise ValueError("duplicate current axis transport identity")
    grouped = {}
    sources = set()
    for row in observations:
        ref = row['extent_entity_ref']
        if row['context_epoch'] != context_epoch or ref not in present:
            continue
        moving = frozenset(row['co_translated_entity_refs']) & present
        fixed = frozenset(row['stationary_entity_refs']) & present
        if moving & fixed or ref in moving:
            raise ValueError("inconsistent sealed intervention memberships")
        grouped.setdefault(ref, []).append((moving, fixed))
        sources.add(row['term_ref'])
    refs = tuple(sorted(grouped))
    if len(refs) > 32:
        raise ValueError("axis transport receiver bound exceeded")
    asymmetric, contrary, one_sided = set(), set(), set()
    for first in refs:
        for second in refs:
            if first == second:
                continue
            transported = any(second in moving for moving, _ in grouped[first])
            preserved = any(first in fixed for _, fixed in grouped[second])
            contradicted = (any(second in fixed for _, fixed in grouped[first])
                            or any(first in moving for moving, _ in grouped[second]))
            if transported and preserved:
                (contrary if contradicted else asymmetric).add((first, second))
            elif transported:
                one_sided.add((first, second))
    reach = {ref: {b for a, b in asymmetric if a == ref} for ref in refs}
    for middle in refs:
        for first in refs:
            if middle in reach[first]:
                reach[first].update(reach[middle])
    cyclic = any(ref in reach[ref] for ref in refs)
    covering = tuple(sorted((a, b) for a, b in asymmetric
        if not any(mid != a and mid != b and mid in reach[a] and b in reach[mid] for mid in refs)))
    unique_parents = all(sum(b == ref for _, b in covering) <= 1 for ref in refs)
    passive = tuple(sorted(set().union(*(moving for rows in grouped.values() for moving, _ in rows)) - set(refs))) if refs else ()
    return FrozenMap({
        'observation_token': stable_digest((context_epoch, tuple(sorted(sources)))),
        'context_epoch': context_epoch, 'source_term_refs': tuple(sorted(sources)),
        'receiver_refs': refs, 'receiver_count': len(refs), 'multiple_receiver_refs': len(refs) > 1,
        'asymmetric_effect_pairs': tuple(sorted(asymmetric)), 'covering_effect_pairs': covering,
        'contrary_effect_pairs': tuple(sorted(contrary)), 'one_sided_effect_pairs': tuple(sorted(one_sided)),
        'passive_entity_refs': passive, 'is_acyclic': not cyclic,
        'has_unique_covering_parent': unique_parents,
        'has_asymmetric_effect': bool(asymmetric), 'has_contrary_effect': bool(contrary),
    })
