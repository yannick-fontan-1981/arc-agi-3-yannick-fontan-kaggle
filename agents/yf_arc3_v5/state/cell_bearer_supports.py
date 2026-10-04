"""Join current declared cell unions to complete tracked raster components.

This is an exact measurement, not a role classifier. Neither historical shape
nor historical cardinality participates. Incomplete members are never trimmed.
"""

from agents.yf_arc3_v5.capabilities.bearer_support import CurrentBearerSupportRequest
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest
from agents.yf_arc3_v5.state.reference_bindings import canonical_reference_bindings_current


def current_cell_bearer_requests(*, snapshot, tracking, profile, support_contract):
    grids = tuple(term for term in snapshot.terms_for_literal_attribute("operational_scene_ref", tracking.frame_ref)
        if term.attributes.get("projection_contract") == profile["grid_contract"]
        and canonical_reference_bindings_current(snapshot, term, frame_ref=tracking.frame_ref))
    if len(grids) > profile["maximum_grids"]:
        raise ValueError("cell bearer grid bound exceeded")
    if not grids:
        return (), FrozenMap()
    entities = {item.entity_id: item for item in tracking.entities
        if item.current_frame_ref == tracking.frame_ref}
    if len(entities) > profile["maximum_entities"]:
        raise ValueError("cell bearer entity bound exceeded")
    owners = {}
    for ref, entity in sorted(entities.items()):
        for point in entity.component.pixels:
            if point in owners:
                raise ValueError("cell bearer tracking overlaps pixels")
            owners[point] = ref
            if len(owners) > profile["maximum_scene_pixels"]:
                raise ValueError("cell bearer scene pixel bound exceeded")
    memberships = set()
    dependencies = {}
    checks = 0
    for grid in grids:
        _, row_offset, col_offset, height, width, row_pitch, col_pitch, rows, cols = grid.attributes["grid_geometry"]
        if min(height, width, row_pitch, col_pitch, rows, cols) <= 0 or height > row_pitch or width > col_pitch:
            raise ValueError("cell bearer calibration invalid")
        supports = grid.attributes["cell_support_rows"]
        if len(supports) > profile["maximum_supports_per_grid"]:
            raise ValueError("cell bearer source support bound exceeded")
        for support in supports:
            cells = tuple((support["top"] + r, support["left"] + c) for r, c in support["normalized_cells"])
            if len(set(cells)) != len(cells) or any(not (0 <= r < rows and 0 <= c < cols) for r, c in cells):
                raise ValueError("cell bearer cells repeated or outside lattice")
            size = len(cells) * height * width
            checks += size
            if checks > profile["maximum_pixel_checks"]:
                raise ValueError("cell bearer pixel check bound exceeded")
            if size > support_contract.max_pixels:
                raise ValueError("cell bearer support pixel bound exceeded")
            pixels = frozenset((row_offset + r * row_pitch + dr, col_offset + c * col_pitch + dc)
                for r, c in cells for dr in range(height) for dc in range(width))
            if not pixels or not pixels.issubset(owners):
                continue
            members = tuple(sorted({owners[point] for point in pixels}))
            if len(members) < 2:
                continue  # Already enumerated as one tracked component.
            if len(members) > support_contract.max_members:
                raise ValueError("cell bearer member bound exceeded")
            if sum(len(entities[ref].component.pixels) for ref in members) != len(pixels):
                continue  # A component extending outside the union is not a part.
            memberships.add(members)
            dependencies.setdefault(members, {})[grid.id] = grid
            if len(memberships) > support_contract.max_supports:
                raise ValueError("cell bearer output bound exceeded")
    requests = tuple(CurrentBearerSupportRequest(
        bearer_ref="measurement.cell_union." + stable_digest((tracking.frame_ref, members)),
        member_entity_refs=members, observation_ref=tracking.frame_ref,
    ) for members in sorted(memberships))
    bindings = {}
    for request in requests:
        witnesses = tuple(grid for _, grid in sorted(dependencies[request.member_entity_refs].items()))
        refs = tuple(sorted({ref for grid in witnesses for ref in grid.attributes["role_premise_claim_refs"]}))
        bindings[request.bearer_ref] = FrozenMap({
            "canonical_term_dependency_revisions": tuple(FrozenMap({"term_ref": grid.id,
                "revision": grid.last_changed_state_revision}) for grid in witnesses),
            "canonical_claim_dependency_digests": tuple(FrozenMap({"claim_ref": ref,
                "content_digest": stable_digest(snapshot.claim(ref))}) for ref in refs),
        })
    return requests, FrozenMap(bindings)
