"""The sole deterministic event-to-meaning-state reducer."""

from __future__ import annotations

from collections.abc import Set as AbstractSet

from itertools import groupby

from agents.yf_arc3_v5.logos.claims import Claim
from agents.yf_arc3_v5.logos.terms import Term
from agents.yf_arc3_v5.state.events import (
    CognitiveEvent,
    PutClaim,
    PutTerm,
    RecordArtifact,
)
from agents.yf_arc3_v5.state.snapshot import CognitiveSnapshot


class ReductionError(ValueError):
    """Raised when an event sequence violates canonical state authority."""


def _current_claim_ids(claims: dict[str, Claim]) -> set[str]:
    superseded = {claim.supersedes for claim in (claims[__yf_order_key] for __yf_order_key in sorted(claims)) if claim.supersedes}
    return {claim_id for claim_id in claims if claim_id not in superseded}


def _apply_term(terms: dict[str, Term], mutation: PutTerm) -> None:
    current = terms.get(mutation.after.id)
    if mutation.before is None:
        if current is not None:
            raise ReductionError(f"term already exists: {mutation.after.id}")
    elif current != mutation.before:
        raise ReductionError(f"stale term before-image: {mutation.after.id}")
    terms[mutation.after.id] = mutation.after


def _apply_claim(
    claims: dict[str, Claim],
    mutation: PutClaim,
    *,
    current_ids: set[str] | None = None,
    retain_superseded: bool = True,
) -> None:
    if mutation.before is None:
        if mutation.after.id in claims:
            raise ReductionError(f"claim already exists: {mutation.after.id}")
    else:
        effective_current_ids = (
            current_ids if current_ids is not None else _current_claim_ids(claims)
        )
        if mutation.before.id not in effective_current_ids:
            raise ReductionError(f"claim is no longer current: {mutation.before.id}")
        if claims.get(mutation.before.id) != mutation.before:
            raise ReductionError(f"stale claim before-image: {mutation.before.id}")
        if mutation.after.id in claims:
            raise ReductionError(
                f"claim revision id already exists: {mutation.after.id}"
            )
    if mutation.before is not None and not retain_superseded:
        del claims[mutation.before.id]
    claims[mutation.after.id] = mutation.after
    if current_ids is not None:
        if mutation.before is not None:
            current_ids.discard(mutation.before.id)
        current_ids.add(mutation.after.id)


def reduce_event_delta(
    snapshot: CognitiveSnapshot,
    events: tuple[CognitiveEvent, ...],
    *,
    existing_event_ids: AbstractSet[str] = frozenset(),
    existing_artifact_ids: AbstractSet[str] = frozenset(),
    existing_claim_ids: AbstractSet[str] = frozenset(),
) -> CognitiveSnapshot:
    """Apply one new transaction without replaying the historical journal.

    The caller retains the append-only journal and its identity indexes.  Only
    the new transaction and the current immutable meaning snapshot cross this
    boundary. Rebuilding indexes and the snapshot is proportional to retained
    meaning, not just the delta. V4 excludes superseded claims from that work;
    legacy formats deliberately keep their historical snapshot semantics.
    """

    if not events:
        raise ReductionError("cannot reduce an empty event delta")
    transaction_id = events[0].transaction_id
    expected_sequence = snapshot.last_sequence + 1
    seen_new_event_ids: set[str] = set()
    for event in events:
        expected_schema = {
            "yf_arc3_v5.cognitive_event.v1": "yf_arc3_v5.cognitive_snapshot.v2",
            "yf_arc3_v5.cognitive_event.v2": "yf_arc3_v5.cognitive_snapshot.v2",
            "yf_arc3_v5.cognitive_event.v3": "yf_arc3_v5.cognitive_snapshot.v3",
            "yf_arc3_v5.cognitive_event.v4": "yf_arc3_v5.cognitive_snapshot.v4",
        }.get(event.schema_version)
        if expected_schema != snapshot.schema_version:
            raise ReductionError("event/snapshot hash versions differ")
        if event.sequence != expected_sequence:
            raise ReductionError(
                "non-contiguous event sequence: "
                f"expected {expected_sequence}, got {event.sequence}"
            )
        if event.run_id != snapshot.run_id:
            raise ReductionError("event delta cannot change run identity")
        if event.transaction_id != transaction_id:
            raise ReductionError("event delta must contain one transaction")
        if event.event_id in existing_event_ids or event.event_id in seen_new_event_ids:
            raise ReductionError(f"duplicate event identity: {event.event_id}")
        seen_new_event_ids.add(event.event_id)
        expected_sequence += 1

    before = events[0].state_revision_before
    after = events[0].state_revision_after
    has_state_mutation = any(
        isinstance(event.mutation, (PutTerm, PutClaim)) for event in events
    )
    expected_after = before + 1 if has_state_mutation else before
    if before != snapshot.revision or after != expected_after:
        raise ReductionError(
            f"invalid transaction revision {transaction_id}: "
            f"{before}->{after}, current={snapshot.revision}"
        )
    if any(
        event.state_revision_before != before or event.state_revision_after != after
        for event in events
    ):
        raise ReductionError(f"mixed revisions inside transaction: {transaction_id}")

    # Workflow checkpoints and operator audit records are journal deltas only:
    # they cannot alter the canonical TERM/CLAIM meaning state.  Validate their
    # identities and lineage, then advance only the journal cursor.  Rebuilding
    # and re-hashing the complete semantic snapshot here made every declarative
    # CONTINUE statement progressively slower as the meaning graph grew.
    if not has_state_mutation:
        new_artifact_ids: set[str] = set()
        for event in events:
            mutation = event.mutation
            if not isinstance(mutation, RecordArtifact):  # pragma: no cover
                raise ReductionError(
                    f"unsupported audit mutation: {type(mutation).__name__}"
                )
            artifact = mutation.artifact
            if (
                artifact.id in existing_artifact_ids
                or artifact.id in new_artifact_ids
            ):
                raise ReductionError(f"duplicate artifact identity: {artifact.id}")
            new_artifact_ids.add(artifact.id)
            if artifact.state_revision != after:
                raise ReductionError(f"artifact has stale state revision: {artifact.id}")
            if artifact.timeline_id and artifact.timeline_id != event.timeline_id:
                raise ReductionError(f"artifact timeline mismatch: {artifact.id}")
            if artifact.frame_id and artifact.frame_id != event.frame_id:
                raise ReductionError(f"artifact frame mismatch: {artifact.id}")
        # Audit-only transactions cannot change TERM/CLAIM meaning.  Reusing
        # the already validated immutable semantic fields avoids invoking the
        # snapshot validator and re-hashing the complete meaning graph on
        # every workflow checkpoint.  Imported snapshots still go through
        # normal validation; this is only the append-only hot delta path.
        # Exact unchanged meaning also preserves its derived lookup views.
        # A shallow metadata copy shares immutable TERM/CLAIM objects and
        # bounded query caches; a cognitive mutation takes the fresh build
        # branch below and cannot inherit stale derived state.
        return snapshot.model_copy(update={
            "event_count": snapshot.event_count + len(events),
            "last_sequence": events[-1].sequence,
        }, deep=False)

    changes_terms = any(isinstance(event.mutation, PutTerm) for event in events)
    changes_claims = any(isinstance(event.mutation, PutClaim) for event in events)
    terms = dict(snapshot.term_index) if changes_terms else snapshot.term_index
    claims = dict(snapshot.claim_index) if changes_claims else snapshot.claim_index
    current_claim_ids = set(snapshot.current_claim_ids) if changes_claims else snapshot.current_claim_ids
    # Carry only already-demanded exact indexes through the validated delta.
    # No parent snapshot is retained and no canonical payload is changed.
    prior_families = snapshot.__dict__.get("current_claims_by_revision_family")
    families = dict(prior_families) if prior_families is not None and changes_claims else prior_families
    new_artifact_ids: set[str] = set()
    new_claim_ids: set[str] = set()
    for event in events:
        mutation = event.mutation
        if isinstance(mutation, RecordArtifact):
            artifact = mutation.artifact
            if (
                artifact.id in existing_artifact_ids
                or artifact.id in new_artifact_ids
            ):
                raise ReductionError(f"duplicate artifact identity: {artifact.id}")
            new_artifact_ids.add(artifact.id)
            if artifact.state_revision != after:
                raise ReductionError(f"artifact has stale state revision: {artifact.id}")
            if artifact.timeline_id and artifact.timeline_id != event.timeline_id:
                raise ReductionError(f"artifact timeline mismatch: {artifact.id}")
            if artifact.frame_id and artifact.frame_id != event.frame_id:
                raise ReductionError(f"artifact frame mismatch: {artifact.id}")
        elif isinstance(mutation, PutTerm):
            _apply_term(terms, mutation)
        elif isinstance(mutation, PutClaim):
            if mutation.after.id in existing_claim_ids or mutation.after.id in new_claim_ids:
                raise ReductionError(f"claim revision id already exists: {mutation.after.id}")
            _apply_claim(
                claims,
                mutation,
                current_ids=current_claim_ids,
                retain_superseded=snapshot.schema_version != "yf_arc3_v5.cognitive_snapshot.v4",
            )
            new_claim_ids.add(mutation.after.id)
            if families is not None:
                if mutation.before is not None:
                    old_family = str(mutation.before.attributes.get("revision_family_ref") or mutation.before.id)
                    remaining = tuple(item for item in families[old_family] if item.id != mutation.before.id)
                    if remaining:
                        families[old_family] = remaining
                    else:
                        del families[old_family]
                new_family = str(mutation.after.attributes.get("revision_family_ref") or mutation.after.id)
                families[new_family] = tuple(sorted(
                    (*families.get(new_family, ()), mutation.after), key=lambda item: item.id,
                ))
        else:  # pragma: no cover - discriminated union is closed
            raise ReductionError(f"unsupported mutation: {type(mutation).__name__}")

    result = CognitiveSnapshot.build(
        run_id=snapshot.run_id,
        revision=after,
        prediction_revision=after,
        event_count=snapshot.event_count + len(events),
        last_sequence=events[-1].sequence,
        terms=tuple(terms.values()) if changes_terms else snapshot.terms,
        claims=tuple(claims.values()) if changes_claims else snapshot.claims,
        schema_version=snapshot.schema_version,
        _previous=snapshot,
    )
    # These are the same cached_property views as a cold snapshot would build.
    # Preserve sorted iteration and isolate each mutable index from its parent.
    result.__dict__.update({
        "term_index": dict(sorted(terms.items())) if changes_terms else terms,
        "claim_index": dict(sorted(claims.items())) if changes_claims else claims,
        "current_claim_ids": frozenset(current_claim_ids),
    })
    if families is not None:
        result.__dict__["current_claims_by_revision_family"] = families
    return result


def reduce_events(events: tuple[CognitiveEvent, ...]) -> CognitiveSnapshot:
    if not events:
        raise ReductionError("cannot infer run identity from an empty event sequence")

    if any(event.schema_version == "yf_arc3_v5.cognitive_event.v4" for event in events):
        if any(event.schema_version != "yf_arc3_v5.cognitive_event.v4" for event in events):
            raise ReductionError("mixed snapshot hash versions in journal")
        # Restoration is explicitly an archive operation. Normal commits only
        # apply their delta to the current view; they never replay this journal.
        snapshot = CognitiveSnapshot.empty(events[0].run_id,
            schema_version="yf_arc3_v5.cognitive_snapshot.v4")
        event_ids, artifact_ids, claim_ids, transaction_ids = set(), set(), set(), set()
        for transaction_id, group in groupby(events, key=lambda item: item.transaction_id):
            if transaction_id in transaction_ids:
                raise ReductionError(f"transaction is not contiguous: {transaction_id}")
            batch = tuple(group)
            snapshot = reduce_event_delta(snapshot, batch, existing_event_ids=event_ids,
                existing_artifact_ids=artifact_ids, existing_claim_ids=claim_ids)
            transaction_ids.add(transaction_id)
            event_ids.update(event.event_id for event in batch)
            artifact_ids.update(event.mutation.artifact.id for event in batch
                if isinstance(event.mutation, RecordArtifact))
            claim_ids.update(event.mutation.after.id for event in batch
                if isinstance(event.mutation, PutClaim))
        return snapshot

    merkle_events = sum(event.schema_version == "yf_arc3_v5.cognitive_event.v3" for event in events)
    if merkle_events not in (0, len(events)):
        raise ReductionError("mixed snapshot hash versions in journal")
    snapshot_schema_version = ("yf_arc3_v5.cognitive_snapshot.v3" if merkle_events
        else "yf_arc3_v5.cognitive_snapshot.v2")

    expected_sequence = 1
    run_id = events[0].run_id
    seen_event_ids: set[str] = set()
    for event in events:
        if event.sequence != expected_sequence:
            raise ReductionError(
                f"non-contiguous event sequence: expected {expected_sequence}, got {event.sequence}"
            )
        if event.run_id != run_id:
            raise ReductionError("one event store cannot mix run identities")
        if event.event_id in seen_event_ids:
            raise ReductionError(f"duplicate event identity: {event.event_id}")
        seen_event_ids.add(event.event_id)
        expected_sequence += 1

    terms: dict[str, Term] = {}
    claims: dict[str, Claim] = {}
    revision = 0
    seen_transactions: set[str] = set()
    seen_artifact_ids: set[str] = set()

    for transaction_id, event_group in groupby(
        events, key=lambda item: item.transaction_id
    ):
        group = tuple(event_group)
        if transaction_id in seen_transactions:
            raise ReductionError(f"transaction is not contiguous: {transaction_id}")
        seen_transactions.add(transaction_id)
        before = group[0].state_revision_before
        after = group[0].state_revision_after
        has_state_mutation = any(
            isinstance(event.mutation, (PutTerm, PutClaim)) for event in group
        )
        expected_after = before + 1 if has_state_mutation else before
        if before != revision or after != expected_after:
            raise ReductionError(
                f"invalid transaction revision {transaction_id}: {before}->{after}, current={revision}"
            )
        if any(
            event.state_revision_before != before or event.state_revision_after != after
            for event in group
        ):
            raise ReductionError(
                f"mixed revisions inside transaction: {transaction_id}"
            )

        trial_terms = dict(terms)
        trial_claims = dict(claims)
        for event in group:
            if isinstance(event.mutation, RecordArtifact):
                artifact = event.mutation.artifact
                if artifact.id in seen_artifact_ids:
                    raise ReductionError(f"duplicate artifact identity: {artifact.id}")
                seen_artifact_ids.add(artifact.id)
                if artifact.state_revision != after:
                    raise ReductionError(
                        f"artifact has stale state revision: {artifact.id}"
                    )
                if artifact.timeline_id and artifact.timeline_id != event.timeline_id:
                    raise ReductionError(f"artifact timeline mismatch: {artifact.id}")
                if artifact.frame_id and artifact.frame_id != event.frame_id:
                    raise ReductionError(f"artifact frame mismatch: {artifact.id}")
            if isinstance(event.mutation, PutTerm):
                _apply_term(trial_terms, event.mutation)
            elif isinstance(event.mutation, PutClaim):
                _apply_claim(trial_claims, event.mutation)
            elif isinstance(event.mutation, RecordArtifact):
                # Lifecycle records are part of the authoritative journal but
                # cannot write the TERM/CLAIM snapshot directly.
                continue
            else:  # pragma: no cover - discriminated union is closed
                raise ReductionError(
                    f"unsupported mutation: {type(event.mutation).__name__}"
                )
        terms = trial_terms
        claims = trial_claims
        revision = after

    return CognitiveSnapshot.build(
        run_id=run_id,
        revision=revision,
        prediction_revision=revision,
        event_count=len(events),
        last_sequence=events[-1].sequence,
        terms=tuple(terms.values()),
        claims=tuple(claims.values()),
        schema_version=snapshot_schema_version,
    )
