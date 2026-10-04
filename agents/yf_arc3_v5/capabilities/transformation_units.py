"""Bounded compound-support measurements under a source-declared contract.

Primitive observations are immutable inputs. A compound keeps its actual
multicolour support and member observations, never a fabricated base colour.
Old membership can be measured after separation, without asserting persistence.
"""

from agents.yf_arc3_v5.capabilities.bearer_support import measure_adjacent_translation_groups
from agents.yf_arc3_v5.capabilities.coupling_observation_basis import _rigid_translation
from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest
from agents.yf_arc3_v5.capabilities.pair_observation_history import PairObservationHistory


def measure_transformation_units(*, before, after, entity_rows, previous_rows, profile, decoration_carrier_refs=(), registered_delta=(0, 0), collective_measurements=()):
    if profile["schema_version"] != "yf_arc3_v5.adjacent_cotranslation_units.v1":
        raise ValueError("unsupported compound comparison contract")
    if len(entity_rows) > profile["maximum_units"]:
        raise ValueError("compound comparison row bound exceeded")
    if isinstance(previous_rows, PairObservationHistory):
        prior_rows = previous_rows.iter_rows()
    else:
        # Membership-only fixtures/callers need not carry pair references.
        if len(previous_rows) > 64:
            raise ValueError("compound comparison row bound exceeded")
        prior_rows = previous_rows
    rows = {row["entity_ref"]: row for row in entity_rows}
    if len(rows) != len(entity_rows):
        raise ValueError("duplicate compound comparison member")
    translations = tuple((ref, row["descriptor"]["delta_row"], row["descriptor"]["delta_col"])
        for ref, row in sorted(rows.items()) if _rigid_translation(row["descriptor"]))
    groups = measure_adjacent_translation_groups(tracking=after, member_translations=translations,
        connectivity=profile["connectivity"], max_members=profile["maximum_members"],
        max_pixels=profile["maximum_pixels"])
    translation_groups = {group for group in groups if len(group) > 1}
    visual_groups = measure_enclosed_recolor_groups(before=before, after=after,
        rows=rows, carrier_refs=decoration_carrier_refs, profile=profile)
    if len(collective_measurements) > 1:
        raise ValueError("collective comparison measurement bound exceeded")
    collective_groups = {}
    for observation in collective_measurements:
        members = tuple(sorted(observation["member_refs"]))
        if (not 2 <= len(members) <= profile["maximum_members"]
                or len(set(members)) != len(members) or not all(ref in rows for ref in members)):
            raise ValueError("incomplete collective comparison membership")
        count = observation["slot_count"]
        first_values, second_values = observation["before_values"], observation["after_values"]
        quantum = observation["signed_slot_quantum"]
        if (count != len(members) or len(first_values) != count or len(second_values) != count
                or type(quantum) is not int or abs(quantum) != 1
                or tuple(first_values[(index - quantum) % count] for index in range(count)) != second_values):
            raise ValueError("collective permutation observation is not exact")
        collective_groups[members] = observation
    current_groups = translation_groups | set(visual_groups) | set(collective_groups)
    grouped_members = frozenset(ref for group in current_groups for ref in group)
    prior_groups = set()
    for prior in prior_rows:
        for members in prior.get("entity_member_refs") or ():
            if len(members) > profile["maximum_members"] or len(set(members)) != len(members):
                raise ValueError("invalid previous compound membership")
            if len(members) > 1 and all(ref in rows for ref in members):
                prior_groups.add(tuple(sorted(members)))
    # A separated former unit is measured as an old membership hypothesis,
    # alongside the current observations. Pair enumeration must exclude shared
    # support members; these are not additional independent physical objects.
    all_groups = sorted(current_groups | prior_groups)
    before_by_ref = {item.entity_id: item for item in before.entities}
    after_by_ref = {item.entity_id: item for item in after.entities}
    output = [row for ref, row in sorted(rows.items()) if ref not in grouped_members]
    member_refs, valued_supports = {}, {}
    for members in all_groups:
        supports = []
        for tracking in (before_by_ref, after_by_ref):
            if sum(len(tracking[ref].component.pixels) for ref in members) > profile["maximum_pixels"]:
                raise ValueError("compound support pixel bound exceeded")
            pixels = tuple(sorted((r, c, tracking[ref].component.value)
                for ref in members for r, c in tracking[ref].component.pixels))
            if len(pixels) > profile["maximum_pixels"]:
                raise ValueError("compound support pixel bound exceeded")
            if len({(r, c) for r, c, _ in pixels}) != len(pixels):
                raise ValueError("compound member supports overlap")
            supports.append(pixels)
        first, second = supports
        def geometry(pixels):
            top = min(r for r, c, v in pixels)
            left = min(c for r, c, v in pixels)
            return (top, left, max(r for r, c, v in pixels) - top + 1,
                max(c for r, c, v in pixels) - left + 1,
                stable_digest(tuple((r - top, c - left) for r, c, v in pixels)))
        a, b = geometry(first), geometry(second)
        reference = "measurement.compound." + stable_digest(members)
        descriptor = FrozenMap({"delta_row": b[0] - a[0] - registered_delta[0], "delta_col": b[1] - a[1] - registered_delta[1],
            "delta_height": b[2] - a[2], "delta_width": b[3] - a[3],
            "delta_area": len(second) - len(first),
            "value_before": tuple(sorted({v for r, c, v in first})),
            "value_after": tuple(sorted({v for r, c, v in second})),
            "shape_before": a[4], "shape_after": b[4],
            "member_change_digests": tuple(stable_digest(rows[ref]["descriptor"]) for ref in members),
            "current_adjacent_cotranslation": members in translation_groups,
            "current_enclosed_visual_mode": members in visual_groups})
        collective = collective_groups.get(members)
        if collective is not None:
            # The operation recurs even when different slot colours change.
            # Keep the exact member observations outside the operation token.
            descriptor = FrozenMap({**{key: item for key, item in sorted(descriptor.items())
                if key not in ("member_change_digests", "current_adjacent_cotranslation")},
                "collective_slot_quantum": collective["signed_slot_quantum"],
                "collective_slot_count": collective["slot_count"]})
        collective_observation = (FrozenMap({"measurement": collective,
            "member_observations": tuple(rows[ref] for ref in members)})
            if collective is not None else None)
        output.append(FrozenMap({"entity_ref": reference, "member_refs": members,
            "descriptor": descriptor, "changed": any(rows[ref]["changed"] for ref in members),
            "collective_observation": collective_observation,
            "before_support_digest": stable_digest(first), "after_support_digest": stable_digest(second)}))
        member_refs[reference] = members
        valued_supports[reference] = first
    if len(output) > profile["maximum_units"]:
        raise ValueError("compound output unit bound exceeded")
    return FrozenMap({"entity_rows": tuple(sorted(output, key=lambda row: row["entity_ref"])),
        "member_refs_by_unit": FrozenMap(member_refs),
        "before_valued_pixels_by_unit": FrozenMap(valued_supports),
        "current_compound_count": len(current_groups), "prior_compound_count": len(prior_groups)})


def measure_enclosed_recolor_groups(*, before, after, rows, carrier_refs, profile):
    """Exact stationary recolour of holes inside a declared marker carrier.

    The carrier is supplied by the existing DRM marker-ownership measurement,
    not guessed from a palette or from a large surrounding background.
    """
    if not profile.get("include_declared_carrier_enclosed_recoloring", False):
        return ()
    if len(carrier_refs) > 64:
        raise ValueError("decoration carrier bound exceeded")
    before_entities = {item.entity_id: item for item in before.entities}
    after_entities = {item.entity_id: item for item in after.entities}
    recolored = {}
    for ref, row in sorted(rows.items()):
        descriptor = row["descriptor"]
        if (all(descriptor.get(key) == 0 for key in ("delta_row", "delta_col", "delta_height", "delta_width", "delta_area"))
            and descriptor.get("shape_before") == descriptor.get("shape_after")
            and descriptor.get("value_before") != descriptor.get("value_after")):
            first, second = before_entities[ref].component, after_entities[ref].component
            if first.pixels == second.pixels:
                pixels = frozenset(first.pixels)
                if len(pixels) > profile["maximum_pixels"]:
                    raise ValueError("decoration support bound exceeded")
                boundary = frozenset((r+dr,c+dc) for r,c in pixels
                    for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)) if (r+dr,c+dc) not in pixels)
                recolored[ref] = boundary
    groups = []
    for carrier in sorted(set(carrier_refs)):
        if carrier not in rows or rows[carrier]["changed"]:
            continue
        first, second = before_entities[carrier].component, after_entities[carrier].component
        if first.pixels != second.pixels:
            continue
        pixels = frozenset(first.pixels)
        members = tuple(ref for ref, boundary in sorted(recolored.items()) if boundary and boundary <= pixels)
        if not members:
            continue
        group = tuple(sorted((carrier, *members)))
        if len(group) > profile["maximum_members"]:
            raise ValueError("decorated compound member bound exceeded")
        groups.append(group)
    if len({ref for group in groups for ref in group}) != sum(map(len, groups)):
        raise ValueError("ambiguous overlapping decorated compounds")
    return tuple(groups)
