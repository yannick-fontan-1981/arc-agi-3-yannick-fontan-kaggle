"""Read source-declared extent witnesses and transport their dependencies."""
from agents.yf_arc3_v5.capabilities.observed_extent_templates import measured_extent_variants
from agents.yf_arc3_v5.capabilities.bearer_relocation import relocate_exact_bearer_templates
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest


def measured_extent_supports(*, snapshot, tracking, templates, profile, relocation_contract, support_contract):
    claims = snapshot.claims_for_predicate(profile['predicate'])
    if len(claims) > profile['maximum_records']:
        raise ValueError('extent witness record bound exceeded')
    witnesses = {}
    for claim in claims:
        if (claim.epistemic_status.value not in profile['allowed_statuses']
                or claim.disposition.value != profile['disposition']
                or claim.polarity.value != profile['polarity']
                or claim.active_contradictions or claim.defeated_by
                or any(claim.attributes.get(k) != v for k, v in sorted(profile['equalities'].items()))):
            continue
        fields = [claim.attributes.get(profile[k]) for k in ('before_field', 'after_field', 'palette_field')]
        if any(value is None for value in fields):
            continue
        for template in templates:
            for variant in measured_extent_variants(template, *fields, max_pixels=support_contract.max_pixels):
                witnesses.setdefault(variant, {})[claim.id] = claim
                if len(witnesses) > relocation_contract.max_templates:
                    raise ValueError('extent variant bound exceeded; nothing dropped')
    requests, dependencies = {}, {}
    for variant, proofs in sorted(witnesses.items()):
        for request in relocate_exact_bearer_templates(tracking=tracking, templates=(variant,),
                contract=relocation_contract, support_contract=support_contract):
            requests[request.bearer_ref] = request
            dependencies.setdefault(request.bearer_ref, {}).update(proofs)
            if len(requests) > support_contract.max_supports:
                raise ValueError('extent support bound exceeded')
    bindings = FrozenMap({ref: FrozenMap({'canonical_claim_dependency_digests': tuple(
        FrozenMap({'claim_ref': claim.id, 'content_digest': stable_digest(claim)})
        for _, claim in sorted(proofs.items()))}) for ref, proofs in sorted(dependencies.items())})
    return tuple(requests[ref] for ref in sorted(requests)), bindings
