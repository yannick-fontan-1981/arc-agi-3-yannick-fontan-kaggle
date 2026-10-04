"""Incremental projection of source-declared canonical memory entries.

Only immutable TERM/CLAIM references and derived buckets are retained. Query
labels, link statuses and attribute bindings all come from the DRM contract.
"""

from __future__ import annotations

from agents.yf_arc3_v5.capabilities.appearance import AppearanceSignatureKey, BearerAppearanceMeasurement
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest
from agents.yf_arc3_v5.state.appearance_index import AppearanceIndexEntry, ExactAppearanceIndex, exact_context_is_compatible
from agents.yf_arc3_v5.state.event_store import EventStore
from agents.yf_arc3_v5.state.events import PutClaim, PutTerm


class CanonicalAppearanceIndex:
    def __init__(self, *, store: EventStore, contract_source_hash: str, projection: FrozenMap) -> None:
        self._store = store
        self._projection = projection
        self._contract_source_hash = contract_source_hash
        self._cursor = 0
        self._terms = {}
        self._links = {}
        self._link_refs_by_term: dict[str, set[str]] = {}
        self._index = ExactAppearanceIndex(contract_source_hash)
        self._indexed_refs: set[str] = set()
        self._binding_links = {}
        self._binding_link_refs_by_term: dict[str, set[str]] = {}

    @property
    def entry_count(self) -> int:
        return len(self._indexed_refs)

    def synchronize(self) -> int:
        affected: set[str] = set()
        events = self._store.event_view
        for offset in range(self._cursor, len(events)):
            mutation = events[offset].mutation
            if isinstance(mutation, PutTerm):
                term = mutation.after
                if term.label == self._projection["entry_label"]:
                    self._terms[term.id] = term
                    affected.add(term.id)
            elif isinstance(mutation, PutClaim):
                claim = mutation.after
                if claim.predicate == self._projection.get("current_binding_predicate"):
                    if claim.supersedes:
                        previous = self._binding_links.pop(claim.supersedes, None)
                        if previous is not None:
                            for argument in previous.arguments:
                                self._binding_link_refs_by_term.get(argument, set()).discard(previous.id)
                    self._binding_links[claim.id] = claim
                    for argument in claim.arguments:
                        self._binding_link_refs_by_term.setdefault(argument, set()).add(claim.id)
                if claim.predicate == self._projection["link_predicate"]:
                    if claim.supersedes:
                        previous = self._links.pop(claim.supersedes, None)
                        if previous is not None:
                            affected.update(previous.arguments)
                            for argument in previous.arguments:
                                self._link_refs_by_term.get(argument, set()).discard(previous.id)
                    self._links[claim.id] = claim
                    affected.update(claim.arguments)
                    for argument in claim.arguments:
                        self._link_refs_by_term.setdefault(argument, set()).add(claim.id)
        upserts = []
        removals = []
        for reference in sorted(affected):
            term = self._terms.get(reference)
            links = tuple(self._links[ref] for ref in sorted(self._link_refs_by_term.get(reference, ())))
            comparable_links = tuple(claim for claim in links if (
                claim.epistemic_status.value in self._projection["allowed_link_statuses"]
                and claim.disposition.value == self._projection["link_disposition"]
                and claim.polarity.value == self._projection["link_polarity"]
                and not claim.active_contradictions and not claim.defeated_by
            ))
            if term is not None and term.attributes.get("active") is True and comparable_links:
                retractions = term.attributes.get(self._projection["key_retractions_field"], ())
                keys = tuple(AppearanceSignatureKey.model_validate(key) for key in term.attributes[self._projection["key_field"]])
                retained_keys = tuple(key for key in keys if key.digest not in retractions)
                if retained_keys:
                    upserts.append(AppearanceIndexEntry(
                        entry_ref=reference, revision=term.last_changed_state_revision,
                        keys=retained_keys,
                        contract_source_hash=term.attributes[self._projection["contract_field"]],
                        context_requirements=term.attributes.get(self._projection["context_requirements_field"], FrozenMap()),
                    ))
                elif reference in self._indexed_refs:
                    removals.append(reference)
            elif reference in self._indexed_refs:
                removals.append(reference)
        self._index.apply_delta(upserts=tuple(upserts), removals=tuple(removals))
        self._indexed_refs.difference_update(removals)
        self._indexed_refs.update(entry.entry_ref for entry in upserts)
        consumed = len(events) - self._cursor
        self._cursor = len(events)
        return consumed

    def recognition_rows(
        self, *, measurements: tuple[BearerAppearanceMeasurement, ...],
        observation_ref: str, max_matches_per_bearer: int, max_rows: int,
        bearer_contexts: FrozenMap = FrozenMap(),
    ) -> tuple[FrozenMap, ...]:
        if max_rows < 1 or len(measurements) > max_rows:
            raise ValueError("recognition bearer/row bound exceeded")
        if len({item.bearer_ref for item in measurements}) != len(measurements):
            raise ValueError("recognition repeats a current bearer")
        self.synchronize()
        rows = []
        measured_lookups = []
        strongest_by_association: dict[str, int] = {}
        candidate_count = 0
        for measurement in measurements:
            contexts = bearer_contexts.get(measurement.bearer_ref, ())
            lookup = self._index.lookup(
                measurement.keys, contract_source_hash=measurement.contract_source_hash,
                max_matches=max_matches_per_bearer,
                current_contexts=tuple(item["values"] for item in contexts),
            )
            candidate_count += len(lookup.maximal_matches)
            if candidate_count > max_rows:
                raise ValueError("recognition row bound exceeded; no ties dropped")
            measured_lookups.append((measurement, contexts, lookup))
            for match in lookup.maximal_matches:
                strongest_by_association[match.entry_ref] = max(
                    strongest_by_association.get(match.entry_ref, 0), match.key.ordinal)
        for measurement, contexts, lookup in measured_lookups:
            for match in lookup.maximal_matches:
                if (self._projection.get("association_wide_dominance") is True
                        and match.key.ordinal < strongest_by_association[match.entry_ref]):
                    continue
                term = self._terms[match.entry_ref]
                requirements = term.attributes.get(self._projection["context_requirements_field"], FrozenMap())
                compatible_contexts = tuple(item for item in contexts if exact_context_is_compatible(requirements, item["values"])) if requirements else ()
                unresolved_fields = set(requirements)
                for item in compatible_contexts:
                    unresolved_fields.intersection_update(set(requirements) - set(item["values"]))
                context_term_dependencies = tuple(item["term_dependency"] for item in compatible_contexts)
                dependencies_by_ref = {binding["claim_ref"]: binding
                    for item in compatible_contexts for binding in item["claim_dependencies"]}
                context_claim_dependencies = tuple(binding for _ref, binding in sorted(dependencies_by_ref.items()))
                attributes = self._projection["recognition_attribute_bindings"]
                bound_values = {target: term.attributes[source] for target, source in sorted(attributes.items())}
                link_bindings = tuple(FrozenMap({"claim_ref": self._links[ref].id, "content_digest": stable_digest(self._links[ref])}) for ref in sorted(self._link_refs_by_term[term.id])
                    if self._links[ref].epistemic_status.value in self._projection["allowed_link_statuses"]
                    and self._links[ref].disposition.value == self._projection["link_disposition"]
                    and self._links[ref].polarity.value == self._projection["link_polarity"]
                    and not self._links[ref].active_contradictions and not self._links[ref].defeated_by)
                binding_values = FrozenMap({
                    "bearer_ref": measurement.bearer_ref, "association_ref": term.id,
                    "association_revision": match.entry_revision, "match_code": match.key.code,
                    "match_rank": match.key.ordinal,
                    "match_digest": match.key.digest, "observation_ref": observation_ref,
                    "canonical_claim_dependency_digests": link_bindings + context_claim_dependencies,
                    "canonical_term_dependency_revisions": (FrozenMap({"term_ref": term.id, "revision": match.entry_revision}),) + context_term_dependencies,
                    "role_context_requirements": requirements,
                    "unresolved_context_fields": tuple(sorted(unresolved_fields)),
                })
                token = stable_digest(tuple((field, binding_values[field]) for field in self._projection["binding_token_fields"]))
                binding_ref = self._projection["binding_id_pattern"].format(binding_token=token)
                binding_links = tuple(self._binding_links[ref] for ref in sorted(self._binding_link_refs_by_term.get(binding_ref, ())))
                incompatible_count = sum(
                    claim.epistemic_status.value not in self._projection.get("current_binding_allowed_statuses", ())
                    or claim.disposition.value != self._projection["link_disposition"]
                    or claim.polarity.value != self._projection["link_polarity"]
                    or bool(claim.active_contradictions) or bool(claim.defeated_by)
                    for claim in binding_links
                )
                rows.append(FrozenMap({
                    **self._projection["neutral_current_bearer_measurements"],
                    "binding_token": token,
                    "current_binding_incompatible_claim_count": incompatible_count,
                    "current_binding_revision_claim_refs": tuple(claim.id for claim in binding_links),
                    **binding_values,
                    "role_premise_claim_refs": (self._projection["binding_claim_id_pattern"].format(binding_token=token),),
                    **bound_values,
                }))
        return tuple(rows)

    def relocation_templates(self, *, max_templates: int) -> tuple[tuple[tuple[int, int, int], ...], ...]:
        """Project only rasters of currently admitted canonical entries."""
        self.synchronize()
        field = self._projection["relocation_template_field"]
        templates = set()
        for reference in sorted(self._indexed_refs):
            term = self._terms[reference]
            if term.attributes[self._projection["contract_field"]] != self._contract_source_hash:
                continue
            pixels = term.attributes.get(field)
            if pixels:
                templates.add(tuple(tuple(point) for point in pixels))
                if len(templates) > max_templates:
                    raise ValueError("canonical relocation template bound exceeded; no template truncated")
        return tuple(sorted(templates))
