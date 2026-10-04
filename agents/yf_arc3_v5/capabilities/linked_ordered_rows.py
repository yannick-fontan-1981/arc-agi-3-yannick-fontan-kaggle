"""Bounded raster/graph measurements under a declared substitution operator.

The graph is a revisable interpretation of connected contours. This calculation
does not establish call semantics, edit permission, or official acceptance.
"""
from collections import Counter

from agents.yf_arc3_v5.capabilities.contracts import (
    TypedInstructionResource, TypedInterpreterInput, TypedProgramDefinition,
)
from agents.yf_arc3_v5.capabilities.typed_recursive_program import measure_typed_interpreter
from agents.yf_arc3_v5.logos.types import stable_digest


def measure_linked_ordered_rows(value, hollow_families, components):
    contract = value.linked_ordered_family_measurement_contract
    if contract.get("operator") != "substitute_connected_contour_row_then_resume":
        return ()
    maximum_rows = int(contract.get("maximum_rows", 0))
    maximum_slots = int(contract.get("maximum_slots", 0))
    maximum_records = int(contract.get("maximum_records", 0))
    if not (2 <= maximum_rows <= 16 and 1 <= maximum_slots <= 32 and 1 <= maximum_records <= 3):
        return ()
    raster = value.frame.rows
    records = []
    for hollow in hollow_families:
        aspects = tuple(item.value for item in hollow)
        aspect_set = frozenset(aspects)
        resources = tuple(c for c in components if c.value in aspect_set
                          and c.area == c.bbox.width * c.bbox.height
                          and c.bbox.top > max(h.bbox.bottom for h in hollow)
                          and c.bbox.width < hollow[0].bbox.width
                          and c.bbox.height < hollow[0].bbox.height)
        if not resources:
            continue
        shapes = Counter((c.bbox.width, c.bbox.height) for c in resources)
        largest_count = max(shapes.values())
        dominant = tuple(s for s, count in sorted(shapes.items()) if count == largest_count)
        if len(dominant) != 1:
            continue
        width, height = dominant[0]
        regions = []
        for region in components:
            box = region.bbox
            if (box.width <= 2 * width or box.height <= height
                    or region.area * 2 < box.width * box.height):
                continue
            members = tuple(sorted((c for c in components
                                    if c.component_id != region.component_id
                                    and c.bbox.width <= width and c.bbox.height <= height
                                    and c.bbox.top + c.bbox.bottom == box.top + box.bottom
                                    and box.left <= c.bbox.left <= c.bbox.right <= box.right
                                    and box.top <= c.bbox.top <= c.bbox.bottom <= box.bottom),
                                   key=lambda c: c.bbox.left + c.bbox.right))
            if not 2 <= len(members) <= maximum_slots:
                continue
            # An ordered container does not require a regular cell lattice.
            if any(first.bbox.right >= second.bbox.left
                   for first, second in zip(members, members[1:])):
                continue
            if box.top < 1 or box.left < 1 or box.bottom + 1 >= len(raster) or box.right + 1 >= len(raster[0]):
                continue
            perimeter = [(box.top-1, x) for x in range(box.left-1, box.right+2)]
            perimeter += [(box.bottom+1, x) for x in range(box.left-1, box.right+2)]
            perimeter += [(y, box.left-1) for y in range(box.top, box.bottom+1)]
            perimeter += [(y, box.right+1) for y in range(box.top, box.bottom+1)]
            counts = Counter(raster[y][x] for y, x in perimeter)
            shell, count = counts.most_common(1)[0]
            if count * 4 < len(perimeter) * 3 or shell == region.value:
                continue
            regions.append((region, members, shell, tuple(perimeter)))
        if not 2 <= len(regions) <= maximum_rows or sum(len(r[1]) for r in regions) > maximum_slots:
            continue
        reserve = tuple(c for c in resources if (c.bbox.width, c.bbox.height) == (width, height)
                        and not any(region.bbox.left <= c.bbox.left <= c.bbox.right <= region.bbox.right
                                    and region.bbox.top <= c.bbox.top <= c.bbox.bottom <= region.bbox.bottom
                                    for region, _, _, _ in regions))
        if (len({c.value for c in reserve}) != len(reserve)
                or len({c.bbox.top + c.bbox.bottom for c in reserve}) > 1):
            continue
        links = {}
        ambiguous = False
        for row_index, (region, members, _, _) in enumerate(regions):
            for ordinal, cell in enumerate(members):
                cx = cell.bbox.left + cell.bbox.right
                cy = cell.bbox.top + cell.bbox.bottom
                if (cx - width + 1) % 2 or (cy - height + 1) % 2:
                    continue
                left, top = (cx-width+1)//2, (cy-height+1)//2
                right, bottom = left+width-1, top+height-1
                edge = [(top, x) for x in range(left, right+1)]
                edge += [(bottom, x) for x in range(left, right+1)]
                edge += [(y, left) for y in range(top+1, bottom)]
                edge += [(y, right) for y in range(top+1, bottom)]
                edge_values = {raster[y][x] for y, x in edge}
                if len(edge_values) != 1 or cell.value != region.value:
                    continue
                shell = next(iter(edge_values))
                carriers = tuple(c for c in components if c.value == shell
                                 and all((y-c.bbox.top, x-c.bbox.left) in c.relative_pixels for y, x in edge))
                targets = tuple(i for i, (_, _, target_shell, perimeter) in enumerate(regions)
                                if i != row_index and shell == target_shell
                                and any(sum((y-c.bbox.top, x-c.bbox.left) in c.relative_pixels
                                            for y, x in perimeter) * 4 >= len(perimeter) * 3
                                        for c in carriers))
                if len(targets) > 1:
                    ambiguous = True
                elif targets:
                    links[row_index, ordinal] = targets[0]
        if ambiguous or not links:
            continue
        incoming = set(links.values())
        roots = tuple(i for i in range(len(regions)) if i not in incoming)
        if len(roots) != 1:
            continue
        root = roots[0]
        expanded = []
        active = set()
        calls = []
        def visit(index):
            if index in active or len(expanded) > maximum_slots:
                return False
            active.add(index)
            for ordinal, cell in enumerate(regions[index][1]):
                target = links.get((index, ordinal))
                if target is None:
                    expanded.append((index, ordinal, cell))
                else:
                    calls.append((index, ordinal, target))
                    if not visit(target):
                        return False
            active.remove(index)
            return True
        if not visit(root) or len(expanded) != len(aspects):
            continue
        expected = {}
        for aspect, (row_index, ordinal, _) in zip(aspects, expanded):
            address = (row_index, ordinal)
            if address in expected and expected[address] != aspect:
                ambiguous = True
            expected[address] = aspect
        if ambiguous:
            continue
        tokens, definitions = [], []
        for row_index, (_, members, _, _) in enumerate(regions):
            refs = []
            for ordinal, cell in enumerate(members):
                ref = cell.component_id
                refs.append(ref)
                target = links.get((row_index, ordinal))
                if target is None:
                    aspect = expected.get((row_index, ordinal))
                    if aspect is None:
                        ambiguous = True
                        break
                    tokens.append(TypedInstructionResource(token_ref=ref, instruction_kind="literal", value_ref=f"aspect:{aspect}"))
                else:
                    tokens.append(TypedInstructionResource(token_ref=ref, instruction_kind="call", target_definition_ref=regions[target][0].component_id))
            definitions.append(TypedProgramDefinition(definition_ref=regions[row_index][0].component_id, instruction_token_refs=tuple(refs)))
        if ambiguous:
            continue
        result = measure_typed_interpreter(TypedInterpreterInput(
            tokens=tuple(tokens), definitions=tuple(definitions),
            entry_definition_ref=regions[root][0].component_id,
            reference_values=tuple(f"aspect:{a}" for a in aspects),
            boundary_kind="full_termination", official_acceptance_evidence=False,
            maximum_steps=128, maximum_calls=32, maximum_stack_depth=16,
            maximum_output=maximum_slots,
        ))
        if not result.normal_termination or not result.prefix_match:
            continue
        loci = tuple(c for _, _, c in expanded)
        open_ordinals = tuple(i for i, (a, c) in enumerate(zip(aspects, loci)) if a != c.value)
        open_values = frozenset(loci[i].value for i in open_ordinals)
        sources = {c.value: c for c in reserve}
        if open_ordinals and (len(open_values) != 1 or not open_values.isdisjoint(aspect_set)
                              or any(aspects[i] not in sources for i in open_ordinals)):
            continue
        records.append({
            "digest": stable_digest((tuple(c.component_id for c in hollow), tuple(c.component_id for c in loci), tuple(calls)))[:16],
            "family_size": len(aspects), "hollow_members": hollow,
            "locus_members": loci, "source_by_aspect": sources,
            "aspect_by_ordinal": aspects, "open_ordinals": open_ordinals,
            "closed_ordinals": tuple(i for i in range(len(aspects)) if i not in open_ordinals),
            "linked_row_count": len(regions), "linked_row_call_count": result.calls_executed,
            "linked_row_returns_complete": not result.pending_returns,
            "linked_row_provenance": tuple((e.output_index, e.definition_ref, e.instruction_index, e.call_site_chain) for e in result.emissions),
        })
        if len(records) >= maximum_records:
            break
    return tuple(records)
