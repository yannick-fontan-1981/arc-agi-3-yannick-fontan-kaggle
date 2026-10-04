"""Immutable reducer output and revision-keyed semantic views."""

from __future__ import annotations

from functools import cached_property, lru_cache
import hashlib
import math
import weakref

from pydantic import model_validator

from agents.yf_arc3_v5.logos.claims import Claim
from agents.yf_arc3_v5.logos.terms import Term
from agents.yf_arc3_v5.logos.types import (
    FrozenModel,
    NonNegativeRevision,
    Ref,
    canonical_json,
)


_SNAPSHOT_SCHEMA_V1 = "yf_arc3_v5.cognitive_snapshot.v1"
_SNAPSHOT_SCHEMA_VERSION = "yf_arc3_v5.cognitive_snapshot.v2"
_SNAPSHOT_SCHEMA_V3 = "yf_arc3_v5.cognitive_snapshot.v3"
_SNAPSHOT_SCHEMA_V4 = "yf_arc3_v5.cognitive_snapshot.v4"


_LIVE_ITEM_BYTES: dict[int, tuple[weakref.ReferenceType[Term | Claim], bytes]] = {}
_LIVE_ITEM_DIGESTS: dict[int, tuple[weakref.ReferenceType[Term | Claim], bytes]] = {}


def _canonical_item_digest(value: Term | Claim) -> bytes:
    """Content-address one immutable leaf; never carry the cache into copies."""
    identity = id(value)
    cached = _LIVE_ITEM_DIGESTS.get(identity)
    if cached is not None and cached[0]() is value:
        return cached[1]
    domain = b"term\0" if isinstance(value, Term) else b"claim\0"
    digest = hashlib.sha256(domain + _canonical_cognitive_item_bytes(value)).digest()

    def discard(reference, key=identity):
        current = _LIVE_ITEM_DIGESTS.get(key)
        if current is not None and current[0] is reference:
            _LIVE_ITEM_DIGESTS.pop(key, None)

    _LIVE_ITEM_DIGESTS[identity] = (weakref.ref(value, discard), digest)
    return digest


def _canonical_cognitive_item_bytes(value: Term | Claim) -> bytes:
    """Serialize once per live immutable item, without retaining that item.

    A capacity-limited LRU scan thrashes when the current state is larger than
    its capacity. Like the pure-capability input-digest cache, this mechanical
    cache is lifetime-bounded by its weakly referenced input objects instead.
    No entry is copied by model_copy and no identity enters serialized bytes.
    """
    identity = id(value)
    entry = _LIVE_ITEM_BYTES.get(identity)
    if entry is not None and entry[0]() is value:
        return entry[1]
    encoded = canonical_json(value).encode("utf-8")

    def discard(reference, key=identity):
        current = _LIVE_ITEM_BYTES.get(key)
        if current is not None and current[0] is reference:
            _LIVE_ITEM_BYTES.pop(key, None)

    _LIVE_ITEM_BYTES[identity] = (weakref.ref(value, discard), encoded)
    return encoded


class _IdentityChunk:
    """Bounded cache key retaining immutable items, without recursive field hashes.

    Strong references prevent address reuse while cached. Identity affects only
    cache hits: the canonical bytes and public state hash remain content-based.
    """

    __slots__ = ("values", "_hash")

    def __init__(self, values: tuple[Term | Claim, ...]):
        if len(values) > 64:
            raise ValueError("canonical serialization chunk exceeds 64 items")
        self.values = values
        # Only a private cache bucket; equality checks every item in order.
        # No process-randomized content hash enters canonical serialization.
        self._hash = sum(map(id, values))

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, _IdentityChunk):
            return NotImplemented
        return len(self.values) == len(other.values) and all(
            left is right for left, right in zip(self.values, other.values)
        )


@lru_cache(maxsize=256)
def _canonical_chunk_digest(chunk: _IdentityChunk) -> bytes:
    return hashlib.sha256(b"chunk\0" + len(chunk.values).to_bytes(8, "big")
        + b"".join(_canonical_item_digest(value) for value in chunk.values)).digest()


def _merkle_collection_payload(domain: bytes, values: tuple[Term | Claim, ...]) -> bytes:
    """Exact existing ordered chunk bytes; no journal or parent is retained."""
    return domain + len(values).to_bytes(8, "big") + b"".join(
        _canonical_chunk_digest(_IdentityChunk(values[offset:offset + 64]))
        for offset in range(0, len(values), 64)
    )


def _merkle_state_digest(*, revision, prediction_revision, terms, claims,
                         schema_version=_SNAPSHOT_SCHEMA_V3, _payloads=None) -> str:
    """Versioned, domain-separated ordered chunks; only changed leaves rehash.

    Counts delimit collections and chunks. Revision counters remain part of
    identity; run/journal cursors remain excluded exactly as in V2. Legacy
    saves keep their original V2 hashing for the whole resumed session.
    """
    digest = hashlib.sha256((schema_version + "\0").encode("ascii"))
    digest.update(canonical_json((revision, prediction_revision)).encode("utf-8"))
    payloads = _payloads if _payloads is not None else (
        _merkle_collection_payload(b"terms\0", terms),
        _merkle_collection_payload(b"claims\0", claims),
    )
    for payload in payloads:
        digest.update(payload)
    return digest.hexdigest()


@lru_cache(maxsize=256)
def _canonical_cognitive_chunk_bytes(values: tuple[Term | Claim, ...] | _IdentityChunk) -> bytes:
    """Reuse bounded immutable serialization blocks, not whole snapshots."""
    if isinstance(values, _IdentityChunk):
        values = values.values
    if len(values) > 64:
        raise ValueError("canonical serialization chunk exceeds 64 items")
    return b",".join(_canonical_cognitive_item_bytes(value) for value in values)


def _update_canonical_items(digest, values: tuple[Term, ...] | tuple[Claim, ...]) -> None:
    """Stream exactly the former item/comma bytes with fewer hash calls."""
    for offset in range(0, len(values), 64):
        if offset:
            digest.update(b",")
        digest.update(_canonical_cognitive_chunk_bytes(_IdentityChunk(values[offset:offset + 64])))


# These contexts are private, copy-only checkpoints, never cognitive state.
# CLAIM bytes precede revision counters in the historical canonical encoding;
# unchanged prefixes can therefore be reused even when TERM/revision data change.
_CLAIM_DIGEST_ROOT = hashlib.sha256(b'{"claims":[')
_CLAIM_PREFIX_CHECKPOINT_LIMIT = 128


@lru_cache(maxsize=_CLAIM_PREFIX_CHECKPOINT_LIMIT)
def _extend_canonical_claim_prefix(previous, chunk: bytes, separator: bool):
    digest = previous.copy()
    if separator:
        digest.update(b",")
    digest.update(chunk)
    return digest


def _canonical_claim_prefix(claims: tuple[Claim, ...]):
    digest = _CLAIM_DIGEST_ROOT
    for offset in range(0, len(claims), 64):
        chunk = _canonical_cognitive_chunk_bytes(_IdentityChunk(claims[offset:offset + 64]))
        if offset // 64 < _CLAIM_PREFIX_CHECKPOINT_LIMIT:
            digest = _extend_canonical_claim_prefix(digest, chunk, bool(offset))
        else:
            # A sequential scan longer than the LRU evicts its own first
            # checkpoints and misses forever on unchanged input. Retain only
            # the bounded leading checkpoints; stream the uncached suffix.
            # Copy before updating: the preceding context may be cached.
            digest = digest.copy()
            digest.update(b",")
            digest.update(chunk)
    # Callers append revision/TERM bytes only to their own copy. Cached contexts
    # must stay immutable, including across failed validation or interleaved runs.
    return digest.copy()


def _semantic_state_digest(
    *,
    run_id: Ref,
    revision: int,
    prediction_revision: int,
    terms: tuple[Term, ...],
    claims: tuple[Claim, ...],
    schema_version: Ref = _SNAPSHOT_SCHEMA_VERSION,
    _merkle_payloads: tuple[bytes, bytes] | None = None,
) -> str:
    """Produce the existing canonical SHA-256 without a giant JSON copy."""

    if schema_version in (_SNAPSHOT_SCHEMA_V3, _SNAPSHOT_SCHEMA_V4):
        if schema_version == _SNAPSHOT_SCHEMA_V4:
            ids = {claim.id for claim in claims}
            if len(ids) != len(claims):
                raise ValueError("current snapshot has duplicate claim identities")
            if any(claim.supersedes in ids for claim in claims):
                raise ValueError("current snapshot cannot contain superseded claims")
        return _merkle_state_digest(revision=revision, prediction_revision=prediction_revision,
            terms=terms, claims=claims, schema_version=schema_version, _payloads=_merkle_payloads)
    if schema_version not in (_SNAPSHOT_SCHEMA_V1, _SNAPSHOT_SCHEMA_VERSION):
        raise ValueError("unsupported cognitive snapshot schema")

    # Keep byte-for-byte parity with the historical canonical JSON layout, but
    # stream its already-cached immutable TERM/CLAIM encodings directly into
    # SHA-256.  Snapshot identity therefore remains unchanged while large
    # semantic graphs no longer require repeated string concatenation.
    digest = _canonical_claim_prefix(claims)
    digest.update(b'],"prediction_revision":')
    digest.update(str(prediction_revision).encode("ascii"))
    digest.update(b',"revision":')
    digest.update(str(revision).encode("ascii"))
    if schema_version == _SNAPSHOT_SCHEMA_V1:
        digest.update(b',"run_id":')
        digest.update(canonical_json(run_id).encode("utf-8"))
    digest.update(b',"schema_version":')
    digest.update(canonical_json(schema_version).encode("utf-8"))
    digest.update(b',"terms":[')
    _update_canonical_items(digest, terms)
    digest.update(b"]}")
    return digest.hexdigest()


class CognitiveSnapshot(FrozenModel):
    run_id: Ref
    revision: NonNegativeRevision = 0
    prediction_revision: NonNegativeRevision = 0
    event_count: NonNegativeRevision = 0
    last_sequence: NonNegativeRevision = 0
    terms: tuple[Term, ...] = ()
    claims: tuple[Claim, ...] = ()
    state_hash: Ref
    schema_version: Ref = _SNAPSHOT_SCHEMA_VERSION

    def semantic_state_payload(self) -> dict[str, object]:
        """Return only cognitive meaning, excluding journal position metadata."""

        return {
            "revision": self.revision,
            "prediction_revision": self.prediction_revision,
            "terms": self.terms,
            "claims": self.claims,
            "schema_version": self.schema_version,
        }

    @model_validator(mode="after")
    def validate_state_hash(self) -> "CognitiveSnapshot":
        expected = _semantic_state_digest(
            run_id=self.run_id,
            revision=self.revision,
            prediction_revision=self.prediction_revision,
            terms=self.terms,
            claims=self.claims,
            schema_version=self.schema_version,
        )
        if self.state_hash != expected:
            raise ValueError("snapshot state hash mismatch")
        return self

    @classmethod
    def build(
        cls,
        *,
        run_id: Ref,
        revision: int = 0,
        prediction_revision: int = 0,
        event_count: int = 0,
        last_sequence: int = 0,
        terms: tuple[Term, ...] = (),
        claims: tuple[Claim, ...] = (),
        schema_version: Ref = _SNAPSHOT_SCHEMA_VERSION,
        _previous: "CognitiveSnapshot | None" = None,
    ) -> "CognitiveSnapshot":
        # The reducer passes the exact previous immutable tuple only when the
        # transaction has no mutation of that collection. Other inputs retain
        # the cold canonical sort, validation and hash path.
        sorted_terms = terms if _previous is not None and _previous.__dict__.get("_canonical_terms") is terms else tuple(sorted(terms, key=lambda item: item.id))
        sorted_claims = claims if _previous is not None and _previous.__dict__.get("_canonical_claims") is claims else tuple(sorted(claims, key=lambda item: item.id))
        payloads = None
        payload_cache = {}
        if schema_version in (_SNAPSHOT_SCHEMA_V3, _SNAPSHOT_SCHEMA_V4):
            parts = []
            for key, domain, values in (("_merkle_terms", b"terms\0", sorted_terms),
                                        ("_merkle_claims", b"claims\0", sorted_claims)):
                prior = _previous.__dict__.get(key) if _previous is not None else None
                payload = prior[1] if prior is not None and prior[0] is values else _merkle_collection_payload(domain, values)
                payload_cache[key] = (values, payload)
                parts.append(payload)
            payloads = tuple(parts)
        state_hash = _semantic_state_digest(
            run_id=run_id,
            revision=revision,
            prediction_revision=prediction_revision,
            terms=sorted_terms,
            claims=sorted_claims,
            schema_version=schema_version,
            _merkle_payloads=payloads,
        )
        # Every item and the digest above were built from already validated
        # canonical TERM/CLAIM values.  Imported snapshots still pass through
        # the full model validator; the reducer hot path must not hash the same
        # immutable state twice.
        result = cls.model_construct(
            run_id=run_id,
            revision=revision,
            prediction_revision=prediction_revision,
            event_count=event_count,
            last_sequence=last_sequence,
            terms=sorted_terms,
            claims=sorted_claims,
            state_hash=state_hash,
            schema_version=schema_version,
        )
        result.__dict__.update(payload_cache)
        result.__dict__.update({"_canonical_terms": sorted_terms, "_canonical_claims": sorted_claims})
        return result

    @property
    def reasoning_state_hash(self) -> Ref:
        """Canonical TERM/CLAIM hash, independent of journal/session identity."""

        return self.state_hash

    @classmethod
    def empty(cls, run_id: Ref, *, schema_version: Ref = _SNAPSHOT_SCHEMA_VERSION) -> "CognitiveSnapshot":
        return cls.build(run_id=run_id, schema_version=schema_version)

    @cached_property
    def term_index(self) -> dict[Ref, Term]:
        return {term.id: term for term in self.terms}

    @cached_property
    def claim_index(self) -> dict[Ref, Claim]:
        return {claim.id: claim for claim in self.claims}

    @cached_property
    def current_claim_ids(self) -> frozenset[Ref]:
        superseded = {claim.supersedes for claim in self.claims if claim.supersedes}
        return frozenset(claim.id for claim in self.claims if claim.id not in superseded)

    @cached_property
    def current_claims(self) -> tuple[Claim, ...]:
        return tuple(claim for claim in self.claims if claim.id in self.current_claim_ids)

    @cached_property
    def current_claims_by_revision_family(self) -> dict[str, tuple[Claim, ...]]:
        """Exact ordered lookup; family collisions retain every current claim."""
        grouped: dict[str, list[Claim]] = {}
        for claim in self.current_claims:
            family = str(claim.attributes.get("revision_family_ref") or claim.id)
            grouped.setdefault(family, []).append(claim)
        return {key: tuple(grouped[key]) for key in sorted(grouped)}

    @cached_property
    def terms_by_projection_contract(self) -> dict[Ref, tuple[Term, ...]]:
        """Index immutable mechanical projection contracts once per snapshot."""

        grouped: dict[Ref, list[Term]] = {}
        for term in self.terms:
            contract = term.attributes.get("projection_contract")
            if isinstance(contract, str) and contract:
                grouped.setdefault(contract, []).append(term)
        return {key: tuple(grouped[key]) for key in sorted(grouped)}

    @cached_property
    def current_claims_by_predicate(self) -> dict[Ref, tuple[Claim, ...]]:
        """Index current canonical CLAIMs once per snapshot."""

        grouped: dict[Ref, list[Claim]] = {}
        for claim in self.current_claims:
            grouped.setdefault(claim.predicate, []).append(claim)
        return {key: tuple(grouped[key]) for key in sorted(grouped)}

    def terms_for_projection_contract(self, contract: Ref) -> tuple[Term, ...]:
        return self.terms_by_projection_contract.get(contract, ())

    @cached_property
    def literal_term_query_cache(self) -> dict[tuple[str, object], tuple[Term, ...]]:
        """Derived equality views only; never part of canonical state."""
        return {}

    def terms_for_literal_attribute(self, name: str, value: object) -> tuple[Term, ...]:
        """Preserve exact equality and order, scanning once per immutable view.

        No role vocabulary or status interpretation is encoded in this index.
        Complex values and NaN keep ordinary equality semantics without a
        dictionary identity shortcut. Revisions own independent derived caches.
        """
        cacheable = value is None or isinstance(value, (str, bool, int, float))
        if isinstance(value, float) and math.isnan(value):
            cacheable = False
        if not cacheable:
            return tuple(term for term in self.terms if term.attributes.get(name) == value)
        key = (name, value)
        cache = self.literal_term_query_cache
        if key not in cache:
            matches = tuple(term for term in self.terms if term.attributes.get(name) == value)
            if len(cache) >= 128:
                return matches
            cache[key] = matches
        return cache[key]

    def claims_for_predicate(self, predicate: Ref) -> tuple[Claim, ...]:
        return self.current_claims_by_predicate.get(predicate, ())

    def term(self, term_id: Ref) -> Term | None:
        return self.term_index.get(term_id)

    def claim(self, claim_id: Ref) -> Claim | None:
        return self.claim_index.get(claim_id)
