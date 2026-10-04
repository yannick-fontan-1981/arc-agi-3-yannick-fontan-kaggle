"""Exact description equality, independent of the number of supporting proofs.

Canonical rows remain untouched: transport may factor exact descriptions,
but equality of a bearer does not merge or select evidence.
An absent entity identity is unknown unless Source supplied the complete
bounded current member-union description. No temporal identity is invented.
"""

from collections.abc import Iterable

from agents.yf_arc3_v5.logos.types import FrozenMap, stable_digest


def compact_exact_entity_descriptions(
    rows: Iterable[FrozenMap], *, maximum_descriptions: int = 64,
    maximum_proofs_per_description: int = 64,
) -> tuple[FrozenMap, ...]:
    """Lossless transport factoring, not evidence selection or role merging.

    Only byte-equal descriptions share storage. Each original TERM identity
    retains its own premise list; unknown descriptions never establish aliases.
    Neither distinct descriptions nor support proofs may be truncated.
    """
    groups: dict[FrozenMap, list[FrozenMap]] = {}
    group_order: list[FrozenMap] = []
    for row in rows:
        if "source_term_rows" in row:
            raise ValueError("description transport already factored")
        description = FrozenMap({key: value for key, value in sorted(row.items())
                                 if key not in ("term_ref", "role_premise_claim_refs")})
        key = description if exact_entity_description_count((row,)) == 1 else row
        if key not in groups:
            group_order.append(key)
        group = groups.setdefault(key, [])
        group.append(row)
        if len(groups) > maximum_descriptions:
            raise ValueError("canonical description bound exceeded; no alternatives truncated")
        if len(group) > maximum_proofs_per_description:
            raise ValueError("canonical description proof bound exceeded; no proofs truncated")
    result = []
    for key in group_order:
        group = groups[key]
        if len(group) == 1:
            result.append(group[0])
            continue
        common = {key: value for key, value in sorted(group[0].items())
                  if key not in ("term_ref", "role_premise_claim_refs")}
        common["source_term_rows"] = tuple(FrozenMap({
            "term_ref": row["term_ref"],
            "role_premise_claim_refs": row.get("role_premise_claim_refs", ()),
        }) for row in group)
        common["role_premise_claim_refs"] = tuple(dict.fromkeys(
            ref for row in group for ref in row.get("role_premise_claim_refs", ())))
        result.append(FrozenMap(common))
    return tuple(result)


def exact_entity_description_count(rows: tuple[FrozenMap, ...]) -> int | None:
    for row in rows:
        if row.get("entity_ref"):
            continue
        members = row.get("member_refs", ())
        if (not isinstance(members, (tuple, list)) or not 2 <= len(members) <= 64
                or any(not isinstance(ref, str) or not ref for ref in members)
                or len(set(members)) != len(members)
                or not isinstance(row.get("support_digest"), str) or not row["support_digest"]
                or not isinstance(row.get("operational_scene_ref"), str) or not row["operational_scene_ref"]
                or type(row.get("bbox_height")) is not int or row["bbox_height"] <= 0
                or type(row.get("bbox_width")) is not int or row["bbox_width"] <= 0
                or row["bbox_height"] * row["bbox_width"] > 4096):
            return None
    return len({stable_digest(FrozenMap({
        key: tuple(sorted(value)) if key == "member_refs" else value
        for key, value in sorted(row.items())
        if key not in ("term_ref", "role_premise_claim_refs", "source_term_rows")
    })) for row in rows})


def exact_entity_descriptions_are_aliases(rows: tuple[FrozenMap, ...]) -> bool:
    return len(rows) > 1 and exact_entity_description_count(rows) == 1
