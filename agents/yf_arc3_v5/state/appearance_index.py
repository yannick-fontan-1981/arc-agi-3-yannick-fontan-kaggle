"""Derived exact index, containing canonical entry references, never meaning.

The owning SRC transaction supplies immutable entries/deltas. This index has
no level, TTL, role attribution, cognitive identifier generator or eviction.
It can be rebuilt from the canonical TERM/CLAIM projection after save/load.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from agents.yf_arc3_v5.capabilities.appearance import AppearanceSignatureKey
from agents.yf_arc3_v5.logos.types import FrozenMap, FrozenModel, Ref


def exact_context_is_compatible(requirements: FrozenMap, values: FrozenMap) -> bool:
    """An absent fact is unknown; an unequal observed literal is incompatible."""
    return all(name not in values or values[name] == expected
               for name, expected in sorted(requirements.items()))


class AppearanceIndexEntry(FrozenModel):
    entry_ref: Ref
    revision: int = Field(ge=0)
    keys: tuple[AppearanceSignatureKey, ...] = Field(min_length=1)
    contract_source_hash: Ref
    context_requirements: FrozenMap = Field(default_factory=FrozenMap)

    @model_validator(mode="after")
    def validate_keys(self) -> "AppearanceIndexEntry":
        if len({key.code for key in self.keys}) != len(self.keys):
            raise ValueError("entry repeats a tier key")
        if len({key.ordinal for key in self.keys}) != len(self.keys):
            raise ValueError("entry repeats a tier ordinal")
        return self


class AppearanceIndexMatch(FrozenModel):
    entry_ref: Ref
    entry_revision: int
    key: AppearanceSignatureKey


class AppearanceIndexLookup(FrozenModel):
    matches: tuple[AppearanceIndexMatch, ...]
    maximal_ordinal: int | None
    maximal_matches: tuple[AppearanceIndexMatch, ...]


class ExactAppearanceIndex:
    """Incremental bucket lookup; dominance is a declared ordinal operation."""

    def __init__(self, contract_source_hash: str) -> None:
        self.contract_source_hash = contract_source_hash
        self._entries: dict[str, AppearanceIndexEntry] = {}
        self._buckets: dict[tuple[str, int, str], set[str]] = {}

    def apply_delta(
        self,
        *,
        upserts: tuple[AppearanceIndexEntry, ...] = (),
        removals: tuple[str, ...] = (),
    ) -> None:
        """Validate the whole delta before changing any index bucket."""
        references = tuple(entry.entry_ref for entry in upserts)
        if len(set(references)) != len(references) or len(set(removals)) != len(removals):
            raise ValueError("duplicate entry in appearance index delta")
        if set(references) & set(removals):
            raise ValueError("entry cannot be upserted and removed in the same delta")
        for entry in upserts:
            if entry.contract_source_hash != self.contract_source_hash:
                raise ValueError("appearance index contract source mismatch")
            previous = self._entries.get(entry.entry_ref)
            if previous is not None and (
                entry.revision < previous.revision
                or (entry.revision == previous.revision and entry != previous)
            ):
                raise ValueError("appearance entry revision is stale or divergent")
        for reference in removals + references:
            previous = self._entries.pop(reference, None)
            if previous is not None:
                for key in previous.keys:
                    address = (key.code, key.ordinal, key.digest)
                    bucket = self._buckets[address]
                    bucket.remove(reference)
                    if not bucket:
                        del self._buckets[address]
        for entry in upserts:
            self._entries[entry.entry_ref] = entry
            for key in entry.keys:
                self._buckets.setdefault((key.code, key.ordinal, key.digest), set()).add(entry.entry_ref)

    def lookup(
        self, keys: tuple[AppearanceSignatureKey, ...], *, contract_source_hash: str,
        max_matches: int,
        current_contexts: tuple[FrozenMap, ...] = (),
    ) -> AppearanceIndexLookup:
        if contract_source_hash != self.contract_source_hash:
            raise ValueError("appearance lookup contract source mismatch")
        if max_matches < 1:
            raise ValueError("appearance lookup requires a positive explicit bound")
        matches: list[AppearanceIndexMatch] = []
        visited = 0
        for key in keys:
            bucket = self._buckets.get((key.code, key.ordinal, key.digest), ())
            visited += len(bucket)
            if visited > max_matches:
                raise ValueError("appearance lookup match bound exceeded; no truncated dominance")
            if not bucket:
                continue
            # Empty and singleton buckets have no ordering choice.  Sort only
            # genuine alternatives, preserving exact dominance tie order.
            references = bucket if len(bucket) == 1 else sorted(bucket)
            for reference in references:
                entry = self._entries[reference]
                if current_contexts and not any(exact_context_is_compatible(entry.context_requirements, values)
                                               for values in current_contexts):
                    continue
                matches.append(AppearanceIndexMatch(
                    entry_ref=reference, entry_revision=entry.revision, key=key,
                ))
        maximum = max((item.key.ordinal for item in matches), default=None)
        return AppearanceIndexLookup(
            matches=tuple(matches), maximal_ordinal=maximum,
            maximal_matches=tuple(item for item in matches if item.key.ordinal == maximum),
        )

    def canonical_entries(self) -> tuple[AppearanceIndexEntry, ...]:
        return tuple(self._entries[reference] for reference in sorted(self._entries))
