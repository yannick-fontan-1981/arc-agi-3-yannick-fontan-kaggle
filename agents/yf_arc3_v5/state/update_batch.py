"""Flatten an SRC-declared batch of disjoint prepared deltas, without selection."""
from collections.abc import Mapping

from agents.yf_arc3_v5.logos.types import FrozenMap


def _field(value, name):
    return value[name] if isinstance(value, Mapping) else getattr(value, name)


def link_disjoint_update_batch(previous, items, evidence):
    """Immutable O(1) link; do not copy or hash any previous groups."""
    if not items:
        if previous is not None or evidence:
            raise ValueError('empty update batch link must initialize a batch')
        return FrozenMap()
    if not isinstance(previous, FrozenMap):
        raise ValueError('update batch requires its immutable previous link')
    return FrozenMap({'previous': previous, 'items': items, 'evidence': evidence})


def flatten_disjoint_update_batch(chunks, maximum_items):
    """Flatten immutable prepared-delta links once.

    No reads, reordering of updates, deduplication of targets or conflict
    resolution: overlap rejects the entire batch. The reducer still checks
    freshness and publishes atomically. All input evidence is retained.
    """
    if not 1 <= maximum_items <= 8192:
        raise ValueError('invalid disjoint update batch bound')
    groups, count = [], 0
    while chunks:
        if not isinstance(chunks, Mapping) or set(chunks) != {'previous', 'items', 'evidence'}:
            raise ValueError('invalid update batch cons cell')
        chunks, items, evidence = chunks['previous'], chunks['items'], chunks['evidence']
        if not isinstance(items, (tuple, list)) or not items:
            raise ValueError('empty or invalid prepared update group')
        if not isinstance(evidence, (tuple, list)) or len(evidence) > 8192:
            raise ValueError('invalid update batch evidence')
        count += len(items)
        if count > maximum_items:
            raise ValueError('disjoint update batch bound exceeded')
        groups.append((items, evidence))
    if not isinstance(chunks, Mapping):
        raise ValueError('invalid update batch terminator')
    updates, seen, evidence_refs = [], set(), set()
    for group, evidence in reversed(groups):
        evidence_refs.update(evidence)
        if len(evidence_refs) > 8192:
            raise ValueError('update batch evidence bound exceeded')
        for item in group:
            mutation = _field(item, 'mutation')
            kind = _field(mutation, 'kind')
            if kind not in ('put_term', 'put_claim'):
                raise ValueError('batch requires prepared cognitive updates')
            after = _field(mutation, 'after')
            identity = _field(after, 'id')
            if kind == 'put_claim':
                attributes = _field(after, 'attributes')
                identity = attributes.get('revision_family_ref') or identity
            key = (kind, identity)
            if key in seen:
                raise ValueError('overlapping update batch target')
            seen.add(key)
            updates.append(item)
    return FrozenMap({'has_updates': bool(updates), 'update_items': tuple(updates),
                      'evidence_refs': tuple(sorted(evidence_refs))})
