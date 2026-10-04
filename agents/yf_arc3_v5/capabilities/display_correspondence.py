"""Bounded exact repetition, containment, interior matching and relocation.

All roles, epistemic status and use of a marker convention belong to DRM/SRC.
Supplied body/part descriptors come from canonical terms, never a palette rule.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pydantic import Field

from agents.yf_arc3_v5.logos.types import FrozenMap, FrozenModel, Ref, stable_digest
from .components import detect_components
from .contracts import FrameGrid, ComponentDescription, ComponentExtractionInput, PeriodicCellGridInput
from .frame import enumerate_periodic_cell_grids
from .tile_lattice import exact_bordered_tile_lattices, exact_joint_tile_lattices


class DisplayCorrespondenceInput(FrozenModel):
    schema_version: Ref = "yf_arc3_v5.display_correspondence_input.v1"
    before: FrameGrid
    after: FrameGrid
    observation_ref: Ref
    previous_observation_ref: Ref
    scope_ref: Ref
    action_ref: Ref
    announced_count: int = Field(ge=0)
    before_completed: int = Field(ge=0)
    after_completed: int = Field(ge=0)
    direction: tuple[int, int]
    body_rows: tuple[FrozenMap, ...] = Field(max_length=8)
    convention_rows: tuple[FrozenMap, ...] = Field(max_length=8)
    contact_rows: tuple[FrozenMap, ...] = Field(default=(), max_length=128)
    max_components: int = Field(default=512, ge=1, le=1024)
    max_rows: int = Field(default=128, ge=1, le=256)
    boundary_inventory_only: bool = False
    lattice_geometry: FrozenMap = Field(default_factory=FrozenMap)


class DisplayCorrespondenceMeasurements(FrozenModel):
    schema_version: Ref = "yf_arc3_v5.display_correspondence_measurements.v1"
    descriptive_facts: FrozenMap
    evidence_refs: tuple[Ref, ...]


def _ref(kind, scope, value):
    return f"measurement.{kind}:" + stable_digest((scope, value))[:20]


def _box(component):
    b = component.bbox
    return (b.top, b.left, b.bottom, b.right)


def _inside(inner, outer):
    return (outer[0] <= inner[0] <= inner[2] <= outer[2]
            and outer[1] <= inner[1] <= inner[3] <= outer[3])


def _interior(frame, box):
    t, l, b, r = box
    return tuple(tuple(frame.rows[y][x] for x in range(l+1, r))
                 for y in range(t+1, b))


def _side(row, col, height, width):
    sides = []
    if row == 0: sides.append((-1, 0))
    if row == height-1: sides.append((1, 0))
    if col == 0: sides.append((0, -1))
    if col == width-1: sides.append((0, 1))
    return sides[0] if len(sides) == 1 else None


def _equality_partition(matrix):
    """Canonicalize only pixel-equality structure, never palette identity."""

    labels = {}
    return tuple(
        tuple(labels.setdefault(pixel, len(labels)) for pixel in row)
        for row in matrix
    )


def regular_boundary_component_groups(
    frame: FrameGrid,
    components: tuple[ComponentDescription, ...],
) -> tuple[tuple[ComponentDescription, ...], ...]:
    """Exact edge repetition independent of segment colour and fill state.

    Solid rectangles share their dimensions; nonrectangular glyphs must share
    their exact pixel support. No semantic role or movement exclusion is made.
    """

    groups: dict[tuple[object, ...], list[ComponentDescription]] = defaultdict(list)
    for component in components:
        box = component.bbox
        edge_gap = min(
            box.top, box.left,
            frame.height - 1 - box.bottom,
            frame.width - 1 - box.right,
        )
        if edge_gap > max(box.height, box.width):
            continue
        support = () if component.area == box.height * box.width else component.relative_pixels
        groups[("row", box.top, box.bottom, box.width, support)].append(component)
        groups[("col", box.left, box.right, box.height, support)].append(component)
    result = []
    for key in sorted(groups):
        ordered = tuple(sorted(groups[key], key=lambda c: c.bbox.left if key[0] == "row" else c.bbox.top))
        if len(ordered) < 2:
            continue
        coords = [c.bbox.left if key[0] == "row" else c.bbox.top for c in ordered]
        pitches = {b - a for a, b in zip(coords, coords[1:])}
        if len(pitches) == 1 and next(iter(pitches)) > key[3]:
            result.append(ordered)
    return tuple(result)


def regular_boundary_component_group_sizes(
    frame: FrameGrid, components: tuple[ComponentDescription, ...],
) -> tuple[int, ...]:
    return tuple(sorted(len(group) for group in regular_boundary_component_groups(frame, components)))


def measure_display_correspondence(value: DisplayCorrespondenceInput) -> DisplayCorrespondenceMeasurements:
    prior_mismatches = {r.get("marker_value") for r in value.convention_rows
                        if r.get("status")=="rejected" and r.get("scope_ref")==value.scope_ref}
    facts = {"measurement_complete": True, "boundary_rows": [], "containment_rows": [],
             "interior_rows": [], "entry_rows": [], "marker_rows": [], "pose_rows": [],
             "stationary_partition_change_rows": [], "spanning_surface_rows": []}
    evidence = (value.previous_observation_ref, value.observation_ref)
    def result():
        if any(len(facts[k])>value.max_rows for k in sorted(facts) if isinstance(facts[k],list)):
            for k in tuple(sorted(facts)):
                v=facts[k]
                if isinstance(v,list): facts[k] = []
            facts["measurement_complete"] = False
        facts.update(observation_ref=value.observation_ref, scope_ref=value.scope_ref,
                     marker_contradiction_present=bool(prior_mismatches) or any(not r["marker_matches"] for r in facts["entry_rows"]),
                     action_ref=value.action_ref,
                     measured_row_count=sum(len(facts[k]) for k in sorted(facts) if isinstance(facts[k],list)))
        return DisplayCorrespondenceMeasurements(descriptive_facts=FrozenMap(facts), evidence_refs=evidence)
    if (value.before.height, value.before.width) != (value.after.height, value.after.width):
        facts["measurement_complete"] = False
        return result()
    components = detect_components(ComponentExtractionInput(frame=value.after)).components
    old_components = detect_components(ComponentExtractionInput(frame=value.before)).components
    if max(len(components), len(old_components)) > value.max_components:
        facts["measurement_complete"] = False
        return result()
    height, width = value.after.height, value.after.width
    bands = [c for c in components if max(c.bbox.height,c.bbox.width) >= 2*min(c.bbox.height,c.bbox.width) and (
             (c.bbox.width == width and c.bbox.height < height and c.touches_frame_boundary)
             or (c.bbox.height == height and c.bbox.width < width and c.touches_frame_boundary))]
    groups = defaultdict(list)
    for c in components:
        b = c.bbox
        edge_gap = min(b.top, b.left, height-1-b.bottom, width-1-b.right)
        if edge_gap <= max(b.height, b.width) and c.area == b.height*b.width:
            groups[("row", b.top, b.bottom, b.width)].append(c)
            groups[("col", b.left, b.right, b.height)].append(c)
        containers = [p for p in bands if p != c and _inside(_box(c), _box(p))]
        if len(containers) == 1:
            parent = containers[0]
            facts["containment_rows"].append({
                "part_ref": _ref("extent", value.scope_ref, _box(c)),
                "band_ref": _ref("extent", value.scope_ref, _box(parent)),
                "bbox": _box(c), "band_bbox": _box(parent)})
    for members in regular_boundary_component_groups(value.after, components):
        boxes = tuple(_box(c) for c in members)
        changed = sum(any(value.before.rows[y][x] != value.after.rows[y][x]
                          for y,x in c.pixels) for c in members)
        facts["boundary_rows"].append({"group_ref": _ref("aligned_boundary", value.scope_ref, boxes),
            "member_count": len(members), "announced_count": value.announced_count,
            "member_bboxes": boxes, "changed_member_count": changed,
            "completed_delta": value.after_completed-value.before_completed})

    geometry = value.lattice_geometry
    rp, cp = int(geometry.get("row_pitch") or 0), int(geometry.get("col_pitch") or 0)
    ro, co = int(geometry.get("row_offset") or 0), int(geometry.get("col_offset") or 0)
    if rp > 1 and cp > 1:
        for c in components:
            b = c.bbox
            if (b.top, b.left, b.bottom, b.right) != (0, 0, height-1, width-1):
                continue
            if not all(any(value.after.rows[y][x] == c.value for y,x in edge) for edge in (
                tuple((0,x) for x in range(width)), tuple((height-1,x) for x in range(width)),
                tuple((y,0) for y in range(height)), tuple((y,width-1) for y in range(height)))):
                continue
            off_grid = tuple(_box(p) for p in components if p != c
                and min(p.bbox.top,p.bbox.left,height-1-p.bbox.bottom,width-1-p.bbox.right) <= max(p.bbox.height,p.bbox.width)
                and ((p.bbox.top-ro)%rp or (p.bbox.left-co)%cp)
                and p.bbox.height < rp and p.bbox.width < cp)
            if off_grid:
                facts["spanning_surface_rows"].append({"surface_ref":c.component_id,
                    "surface_value":c.value, "bbox":_box(c), "off_grid_member_bboxes":off_grid,
                    "off_grid_member_count":len(off_grid)})
    if value.boundary_inventory_only:
        return result()

    grid_result = enumerate_periodic_cell_grids(PeriodicCellGridInput(frame=value.after))
    if len(grid_result.candidates) != 1:
        return result()
    grid = grid_result.candidates[0]
    if grid_result.enumeration_truncated:
        exact = set(exact_bordered_tile_lattices(value.after,maximum_pitch=16))
        exact.update(exact_joint_tile_lattices(value.after,maximum_pitch=16))
        if exact != {(grid.row_offset,grid.col_offset,grid.cell_height,grid.cell_width)}:
            return result()
    outlines = []
    for c in components:
        b = c.bbox
        if b.height < 3 or b.width < 3 or not any(_inside(_box(c), _box(p)) for p in bands):
            continue
        border = {(y,x) for y in range(b.top,b.bottom+1) for x in range(b.left,b.right+1)
                  if y in (b.top,b.bottom) or x in (b.left,b.right)}
        if set(c.pixels) == border:
            outlines.append((c, _interior(value.after, _box(c))))
    tiles = []
    for cell in grid.cells:
        b = cell.bbox; box = (b.top,b.left,b.bottom,b.right)
        if min(b.height,b.width) < 3 or any(_inside(box,_box(p)) for p in bands): continue
        core = _interior(value.after, box)
        if len({v for row in core for v in row}) < 2: continue
        border = [(y,x,value.after.rows[y][x]) for y in range(b.top,b.bottom+1)
                  for x in range(b.left,b.right+1) if y in (b.top,b.bottom) or x in (b.left,b.right)]
        counts = Counter(v for _,_,v in border)
        single = [v for v in sorted(counts) if counts[v] == 1]
        if len(counts) != 2 or len(single) != 1: continue
        accent = single[0]; base = next(v for v in sorted(counts) if v != accent)
        y,x,_ = next(p for p in border if p[2] == accent)
        tiles.append({"box":box, "core":core, "base":base, "accent":accent,
                      "side":_side(y-b.top,x-b.left,b.height,b.width),
                      "cell_ref":_ref("extent",value.scope_ref,box), "cell":(cell.row,cell.col)})
    families = defaultdict(list)
    for tile in tiles: families[(tile["core"],tile["base"])].append(tile)
    for core,base in sorted(families):
        members = families[(core,base)]
        refs = tuple(t["cell_ref"] for t in members)
        family_ref = _ref("interior_family",value.scope_ref,(core,base,refs))
        for tile in members: tile["family_ref"] = family_ref
        peers = [(c,pattern) for c,pattern in outlines if pattern == core]
        matching_families = sum(1 for c,b in sorted(families) if c == core)
        for c,_ in peers:
            bbox = _box(c)
            old_inner = _interior(value.before,bbox)
            changed = sum(value.before.rows[y][x] != value.after.rows[y][x] for y,x in c.pixels)
            before_values = tuple(sorted({value.before.rows[y][x] for y,x in c.pixels}))
            facts["interior_rows"].append({"family_ref":family_ref,
                "indicator_ref":_ref("extent",value.scope_ref,bbox),
                "object_refs":refs, "object_cells":tuple(t["cell"] for t in members),
                "member_count":len(members), "indicator_count":len(peers),
                "matching_family_count":matching_families, "inner_digest":stable_digest(core),
                "inner_preserved":old_inner == core, "changed_border_pixels":changed,
                "border_before_values":before_values, "border_after_value":c.value,
                "border_pixel_count":c.area})

    # The source supplies one already interpreted body descriptor; measurements
    # never choose a controlled identity from the visual population.
    if len(value.body_rows) == 1:
        body = value.body_rows[0]
        pattern = tuple(tuple(p) for p in body.get("body_relative_pixels", ()))
        def matches(pool):
            return [c for c in pool if c.value == body.get("body_value") and c.relative_pixels == pattern]
        old, new = matches(old_components), matches(components)
        if len(old) == len(new) == 1:
            a,b = old[0].bbox,new[0].bbox
            dr,dc = b.top-a.top,b.left-a.left
            adjacent = [p for p in components if p.value == body.get("part_value") and p.area == 1
                        and b.top-1<=p.bbox.top<=b.bottom+1 and b.left-1<=p.bbox.left<=b.right+1]
            if len(adjacent) == 1:
                p = adjacent[0].bbox
                normals = [(rr,cc) for rr,cc,ok in [(-1,0,p.bottom<b.top),(1,0,p.top>b.bottom),
                           (0,-1,p.right<b.left),(0,1,p.left>b.right)] if ok]
                if len(normals) == 1:
                    facts["pose_rows"].append({"body_ref":body["body_entity_ref"],
                        "bbox":_box(new[0]), "side":normals[0], "delta_row":dr,"delta_col":dc})
            step = (value.direction[0]*grid.row_pitch,value.direction[1]*grid.col_pitch)
            touched = [t for t in tiles if _inside((a.top+step[0],a.left+step[1],a.bottom+step[0],a.right+step[1]),t["box"])]
            if step != (0,0) and (dr,dc) != step and (dr,dc) != (0,0) and len(touched)==1:
                entry = touched[0]
                exits=[]
                for t in families[(entry["core"],entry["base"])]:
                    if t is entry: continue
                    rt,ct,rb,cb=t["box"]
                    for normal in [(-1,0),(1,0),(0,-1),(0,1)]:
                        rr,cc=normal[0]*grid.row_pitch,normal[1]*grid.col_pitch
                        if _inside(_box(new[0]),(rt+rr,ct+cc,rb+rr,cb+cc)): exits.append((t,normal))
                if len(exits)==1:
                    exit_tile,normal=exits[0]
                    facts["entry_rows"].append({"entry_ref":entry["cell_ref"],"exit_ref":exit_tile["cell_ref"],
                        "family_ref":entry["family_ref"],"body_ref":body["body_entity_ref"],
                        "arrival_bbox":_box(new[0]),"entry_cell":entry["cell"],"exit_cell":exit_tile["cell"],
                        "side":normal,"marker_matches":exit_tile["side"]==normal,
                        "marker_ref":_ref("marker",value.scope_ref,exit_tile["accent"]),
                        "previous_marker_mismatch":exit_tile["accent"] in prior_mismatches,
                        "marker_value":exit_tile["accent"],"part_value_equal":exit_tile["accent"]==body.get("part_value")
                        and tuple(tuple(p) for p in body.get("part_relative_pixels",())) == ((0,0),)})
    observed_values = {r["marker_value"] for r in facts["entry_rows"] if r["marker_matches"] and r["part_value_equal"]}
    retained_values = {r.get("marker_value") for r in value.convention_rows
                       if r.get("status")=="supported" and r.get("scope_ref")==value.scope_ref}
    contradicted_values = prior_mismatches | {r["marker_value"] for r in facts["entry_rows"] if not r["marker_matches"]}
    for tile in tiles:
        if tile["side"] is not None:
            facts["marker_rows"].append({"cell_ref":tile["cell_ref"],"family_ref":tile["family_ref"],
                "cell":tile["cell"],"side":tile["side"],"marker_value":tile["accent"],
                "arrival_cell":(tile["cell"][0]+tile["side"][0],tile["cell"][1]+tile["side"][1]),
                "analogy_contradicted":tile["accent"] in contradicted_values,
                "analogy_supported":tile["accent"] in (observed_values|retained_values)-contradicted_values})
    for row in facts["interior_rows"]:
        row["entry_family_observed"] = any(r["family_ref"] == row["family_ref"] for r in facts["entry_rows"])
        row["changed_family_indicator_count"] = sum(
            r["family_ref"]==row["family_ref"] and r["inner_preserved"]
            and r["changed_border_pixels"]==r["border_pixel_count"]
            for r in facts["interior_rows"])
    pairs={(tuple(r["border_before_values"]),r["border_after_value"])
           for r in facts["interior_rows"] if r["entry_family_observed"]
           and r["changed_family_indicator_count"]>=2 and r["inner_preserved"]
           and r["changed_border_pixels"]==r["border_pixel_count"]}
    pairs.update((tuple(r["border_before_values"]),r["border_after_value"])
                 for r in value.contact_rows if r.get("scope_ref")==value.scope_ref)
    for row in facts["interior_rows"]:
        row["current_border_equals_observed_before"] = len(pairs)==1 and any(
            len(before)==1 and row["border_after_value"]==before[0] and after!=before[0]
            for before,after in sorted(pairs))
        row["current_border_equals_observed_after"] = len(pairs)==1 and any(
            len(before)==1 and row["border_after_value"]==after and after!=before[0]
            for before,after in sorted(pairs))
    linked_indicator_rows = tuple(
        row for row in facts["interior_rows"]
        if row["current_border_equals_observed_before"]
        or row["current_border_equals_observed_after"]
    )
    remaining_indicator_count = sum(
        row["current_border_equals_observed_before"]
        for row in linked_indicator_rows
    )
    changed_to_satisfied_count = sum(
        row["current_border_equals_observed_after"]
        and row["changed_border_pixels"] == row["border_pixel_count"]
        for row in linked_indicator_rows
    )
    action_linked_completion_observed = bool(
        linked_indicator_rows
        and remaining_indicator_count == 0
        and changed_to_satisfied_count > 0
        and any(row["entry_family_observed"] for row in linked_indicator_rows)
    )

    # Exact same-cell equality partitions describe a stationary object's shape
    # independently of colour names.  The capability does not call such an
    # object an access point or terminal; DRM may do so only when the separately
    # measured action-linked indicator family has just become complete.
    partition_changes = []
    for cell in grid.cells:
        bbox = (cell.bbox.top, cell.bbox.left, cell.bbox.bottom, cell.bbox.right)
        if any(_inside(bbox, _box(band)) for band in bands):
            continue
        before_cell = tuple(
            tuple(value.before.rows[y][x] for x in range(cell.bbox.left, cell.bbox.right + 1))
            for y in range(cell.bbox.top, cell.bbox.bottom + 1)
        )
        after_cell = tuple(
            tuple(value.after.rows[y][x] for x in range(cell.bbox.left, cell.bbox.right + 1))
            for y in range(cell.bbox.top, cell.bbox.bottom + 1)
        )
        pixel_count = cell.bbox.height * cell.bbox.width
        changed_pixel_count = sum(
            before_pixel != after_pixel
            for before_row, after_row in zip(before_cell, after_cell)
            for before_pixel, after_pixel in zip(before_row, after_row)
        )
        before_partition = _equality_partition(before_cell)
        after_partition = _equality_partition(after_cell)
        if (
            changed_pixel_count != pixel_count
            or len({pixel for row in before_cell for pixel in row}) < 2
            or before_partition != after_partition
        ):
            continue
        partition_changes.append({
            "cell_ref": _ref("extent", value.scope_ref, bbox),
            "cell": (cell.row, cell.col),
            "bbox": bbox,
            "before_partition_digest": stable_digest(before_partition),
            "after_partition_digest": stable_digest(after_partition),
            "changed_pixel_count": changed_pixel_count,
            "pixel_count": pixel_count,
            "linked_indicator_member_count": len(linked_indicator_rows),
            "remaining_linked_indicator_member_count": remaining_indicator_count,
            "changed_to_satisfied_indicator_member_count": changed_to_satisfied_count,
            "action_linked_indicator_completion_observed": action_linked_completion_observed,
        })
    candidate_count = len(partition_changes)
    for row in partition_changes:
        row["stationary_partition_change_candidate_count"] = candidate_count
        facts["stationary_partition_change_rows"].append(row)
    return result()
