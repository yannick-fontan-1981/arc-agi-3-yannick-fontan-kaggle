"""Mechanical validity of source-committed canonical reference bindings.

No role, priority, lifetime or evidence status is inferred. Declarations freeze
exact TERM revisions, current CLAIM content digests and explicit link
constraints; this checks identities and compares the declared values.
"""

from agents.yf_arc3_v5.logos.terms import Term
from agents.yf_arc3_v5.logos.types import stable_digest
from agents.yf_arc3_v5.state.snapshot import CognitiveSnapshot


def canonical_reference_bindings_current(
    snapshot: CognitiveSnapshot, term: Term, *, frame_ref: str | None,
    observation_ref: str | None = None,
    require_projectable: bool = True,
) -> bool:
    if require_projectable and term.attributes.get("canonical_fact_projection_enabled") is False:
        return False
    scope = term.attributes.get("operational_scene_ref")
    if scope is not None and scope != frame_ref:
        return False
    observation_scope = term.attributes.get("operational_observation_ref")
    observation_binding = term.attributes.get("canonical_observation_scope_binding")
    if observation_binding is not None:
        dependency = snapshot.term(observation_binding["term_ref"])
        if dependency is None:
            return False
        observation_ref = dependency.attributes.get(observation_binding["attribute"])
    if observation_scope is not None and observation_scope != observation_ref:
        return False
    for binding in term.attributes.get("canonical_term_dependency_revisions", ()):
        dependency = snapshot.term(binding["term_ref"])
        if dependency is None or dependency.last_changed_state_revision != binding["revision"]:
            return False
    for binding in term.attributes.get("canonical_claim_dependency_digests", ()):
        dependency = snapshot.claim(binding["claim_ref"])
        if dependency is None or stable_digest(dependency) != binding["content_digest"]:
            return False
        if not any(current.id == dependency.id for current in snapshot.claims_for_predicate(dependency.predicate)):
            return False
    constraints = term.attributes.get("canonical_role_link_constraints")
    if constraints is not None:
        premise_refs = term.attributes.get("role_premise_claim_refs", ())
        if not premise_refs:
            return False
        for ref in premise_refs:
            premise = snapshot.claim(ref)
            if premise is None or not any(current.id == ref for current in snapshot.claims_for_predicate(premise.predicate)):
                return False
            if (premise.epistemic_status.value not in constraints["allowed_statuses"]
                    or premise.disposition.value != constraints["disposition"]
                    or premise.polarity.value != constraints["polarity"]
                    or (constraints["no_contradictions"] and premise.active_contradictions)
                    or (constraints["no_defeaters"] and premise.defeated_by)):
                return False
    return True
