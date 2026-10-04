"""Bounded cell observations and disjoint changes, without service meanings."""
from agents.yf_arc3_v5.capabilities.contracts import BoundingBox, FrameGrid, FrozenMap
from agents.yf_arc3_v5.logos.types import stable_digest


def samples_from_current_observed_scene(samples):
    """Keep physical temporal comparisons inside one observed scene identity."""
    if not samples:
        return ()
    last_index=max(sample.frame_index for sample in samples)
    values={sample.scene_observation_value for sample in samples if sample.frame_index==last_index}
    if len(values)!=1:
        return ()
    value=next(iter(values))
    return tuple(sample for sample in samples if sample.scene_observation_value==value)


def _rigid_pattern_orbit_ref(pixels,height,width):
    """Exact colour-preserving rectangle isometries; no role interpretation."""
    variants=[]
    for reflected in (False,True):
        current=tuple((r,width-1-c if reflected else c,value) for r,c,value in pixels)
        h,w=height,width
        for _ in range(4):
            variants.append((h,w,tuple(sorted(current))))
            current=tuple((c,h-1-r,value) for r,c,value in current)
            h,w=w,h
    return stable_digest(min(variants))


def _boundary_slot_rows(frame, grid, components, body_values):
    groups = {}
    rects=tuple((BoundingBox.model_validate(component['bbox']),component.get('value'))
                for component in components if component.get('area') ==
                BoundingBox.model_validate(component['bbox']).height*BoundingBox.model_validate(component['bbox']).width)
    merged_rows={}
    # Adjacent equal-valued slots may be one component. A measured marker unit
    # supplies a geometric divisor; exact solid neighbours are split by it.
    for seed, seed_value in rects:
        if seed_value not in body_values or seed.height>grid.cell_height or seed.width>grid.cell_width:
            continue
        for axis, anchor, thickness, unit, lo, hi, extent in (
            ('row',seed.top,seed.height,seed.width,seed.left,seed.right,len(frame.rows)),
            ('column',seed.left,seed.width,seed.height,seed.top,seed.bottom,len(frame.rows[0]))):
            if min(anchor,extent-anchor-thickness)>min(grid.cell_height,grid.cell_width):
                continue
            intervals=tuple((box.left,box.right,palette) if axis=='row' else (box.top,box.bottom,palette)
                for box,palette in rects if (box.top==anchor and box.height==thickness if axis=='row'
                    else box.left==anchor and box.width==thickness)
                and (box.width if axis=='row' else box.height)%unit==0)
            connected={(lo,hi,seed_value)}
            for _ in range(16):
                added={row for row in intervals if any(row[0]==old[1]+1 or row[1]+1==old[0] for old in connected)}-connected
                if not added: break
                connected.update(added)
            start=min(row[0] for row in connected); end=max(row[1] for row in connected)
            count=(end-start+1)//unit
            if not 2<=count<=16 or sum(b-a+1 for a,b,_ in connected)!=end-start+1:
                continue
            positions=tuple(range(start,end+1,unit))
            values=tuple(next(palette for a,b,palette in connected if a<=position<=b) for position in positions)
            key=(axis,anchor,thickness,unit,positions)
            merged_rows[key]=FrozenMap({'group_ref':'measurement.boundary_slots.'+stable_digest(key)[:20],
                'member_values':values,'member_count':count,'member_bounds':(),
                'unit_descriptor':(axis,seed.height,seed.width,unit),
                'unit_origin':(anchor,start) if axis=='row' else (start,anchor)})
    grid_boxes = frozenset(cell.bbox for cell in grid.cells)
    for component in components:
        box = BoundingBox.model_validate(component['bbox'])
        if box in grid_boxes or box.height > grid.cell_height or box.width > grid.cell_width:
            continue
        for axis, distance, anchor, ordinal in (
            ('row', box.top, box.top, box.left),
            ('row', len(frame.rows)-box.bottom-1, box.top, box.left),
            ('column', box.left, box.left, box.top),
            ('column', len(frame.rows[0])-box.right-1, box.left, box.top),
        ):
            if distance <= min(grid.cell_height, grid.cell_width):
                groups.setdefault((axis, anchor, box.height, box.width), {})[ordinal] = (box, component.get('value'))
    rows = []
    for key, members in sorted(groups.items()):
        ordinals = tuple(sorted(members))
        if not 2 <= len(ordinals) <= 16 or len({b-a for a,b in zip(ordinals,ordinals[1:])}) != 1:
            continue
        values = tuple(members[position][1] for position in ordinals)
        if not set(values).intersection(body_values):
            continue
        rows.append(FrozenMap({'group_ref': 'measurement.boundary_slots.'+stable_digest((key,ordinals))[:20],
                              'member_values': values, 'member_bounds': tuple(members[p][0].model_dump(mode='python') for p in ordinals),
                              'member_count': len(ordinals),
                              'unit_descriptor':(key[0],key[2],key[3],ordinals[1]-ordinals[0]),
                              'unit_origin':(members[ordinals[0]][0].top,members[ordinals[0]][0].left)}))
    rows.extend(value for _,value in sorted(merged_rows.items()))
    return tuple(rows) if len(rows) <= 8 else ()


def measure_temporal_cells(frame: FrameGrid, grid, instances, components=()) -> FrozenMap:
    if grid is None or len(grid.cells) > 256 or len(instances) > 3:
        return FrozenMap()
    boxes = {ref: BoundingBox.model_validate(component['bbox']) for ref, component in sorted(instances.items())}
    if not boxes or any(box.height != grid.cell_height or box.width != grid.cell_width
                        or instances[ref].get('area') + 1 != box.height * box.width
                        for ref, box in sorted(boxes.items())):
        return FrozenMap()
    embedded_values = {frame.rows[(box.top + box.bottom)//2][(box.left + box.right)//2] for _, box in sorted(boxes.items())}
    if len(embedded_values) != 1:
        return FrozenMap()
    embedded = next(iter(embedded_values))
    refs = {(cell.row, cell.col): f'measurement.cell.{cell.row}.{cell.col}' for cell in grid.cells}
    bounds = {refs[cell.row, cell.col]: cell.bbox for cell in grid.cells}
    centers = {refs[cell.row, cell.col]: frame.rows[(cell.bbox.top + cell.bbox.bottom)//2][(cell.bbox.left + cell.bbox.right)//2]
               for cell in grid.cells}
    pattern_pixels = {refs[cell.row, cell.col]: tuple(
        (r-cell.bbox.top, c-cell.bbox.left, frame.rows[r][c])
        for r in range(cell.bbox.top, cell.bbox.bottom+1)
        for c in range(cell.bbox.left, cell.bbox.right+1)
        if frame.rows[r][c] != embedded) for cell in grid.cells}
    patterns={ref:stable_digest(pixels) for ref,pixels in sorted(pattern_pixels.items())}
    orbits={ref:_rigid_pattern_orbit_ref(pixels,bounds[ref].height,bounds[ref].width)
            for ref,pixels in sorted(pattern_pixels.items())}
    instance_cells = {ref: cell_ref for ref, box in sorted(boxes.items()) for cell_ref, cell_box in sorted(bounds.items())
                      if box == cell_box}
    if len(instance_cells) != len(boxes):
        return FrozenMap()
    return FrozenMap({'cell_refs': tuple(refs[key] for key in sorted(refs)), 'cell_bounds': FrozenMap({ref: box.model_dump(mode='python') for ref, box in sorted(bounds.items())}),
                      'center_equals_embedded': FrozenMap({ref: value == embedded for ref, value in sorted(centers.items())}),
                      'cell_pattern_refs': FrozenMap(patterns), 'cell_pattern_orbit_refs':FrozenMap(orbits),
                      'instance_cell_refs': FrozenMap(instance_cells),
                      'grid_geometry': (grid.row_offset, grid.col_offset, grid.cell_height, grid.cell_width,
                                        grid.row_pitch, grid.col_pitch),
                      'boundary_slot_rows': _boundary_slot_rows(frame, grid, components,
                          {component.get('value') for _, component in sorted(instances.items())}),
                      'embedded_value': embedded})


def measure_disjoint_cell_changes(samples, translation_action_refs) -> FrozenMap:
    """Entry/change pairs and ordinary departures; phase commands never prove departure.

    Every row is a measurement candidate. Source declarations must decide which
    relation and response laws its evidence supports.
    """
    rows = {}
    slot_transitions = []
    visits = 0
    motion_refs = frozenset(translation_action_refs)
    samples=samples_from_current_observed_scene(samples)
    for before, after in zip(samples, samples[1:]):
        a, b = before.observed_cell_facts, after.observed_cell_facts
        if (not a or not b or after.frame_index != before.frame_index+1
                or a['grid_geometry'] != b['grid_geometry']):
            continue
        if after.preceding_action_ref not in motion_refs:
            for old_group in a.get('boundary_slot_rows', ()):
                new_groups = tuple(row for row in b.get('boundary_slot_rows', ()) if row['group_ref'] == old_group['group_ref'])
                if len(new_groups) != 1:
                    continue
                new_group = new_groups[0]
                # Exact repeated palette-marker displacement, not a phase meaning.
                for value in frozenset(old_group['member_values']).intersection(new_group['member_values']):
                    old_indices = tuple(i for i,v in enumerate(old_group['member_values']) if v == value)
                    new_indices = tuple(i for i,v in enumerate(new_group['member_values']) if v == value)
                    if len(old_indices) == len(new_indices) == 1 and old_indices != new_indices:
                        slot_transitions.append(FrozenMap({'group_ref': old_group['group_ref'], 'value': value,
                            'unit_descriptor':old_group.get('unit_descriptor'),
                            'index_delta': new_indices[0]-old_indices[0], 'action_ref': after.preceding_action_ref,
                            'frame_ref': after.frame_ref}))
            continue
        old_cells, new_cells = a['instance_cell_refs'], b['instance_cell_refs']
        if set(old_cells) != set(new_cells):
            continue
        occupied = frozenset((*old_cells.values(), *new_cells.values()))
        entries = frozenset(new_cells[ref] for ref in new_cells if new_cells[ref] != old_cells[ref]
                            and a['center_equals_embedded'].get(new_cells[ref]) is False)
        visits += len(a['cell_refs'])
        if visits > 16384:
            return FrozenMap({'contact_cell_measurement_bound_reached': True, 'contact_remote_cell_rows': ()})
        changed = tuple(ref for ref in a['cell_refs'] if ref not in occupied
                        and a['cell_pattern_refs'][ref] != b['cell_pattern_refs'][ref]
                        and a['center_equals_embedded'][ref] != b['center_equals_embedded'][ref])
        if len(entries) == 1 and changed:
            source = next(iter(entries))
            inactive = tuple(ref for ref in changed if not a['center_equals_embedded'][ref])
            active = tuple(ref for ref in changed if not b['center_equals_embedded'][ref])
            key = (source, inactive, active)
            rows.setdefault(key, {'source_cell_ref': source, 'changed_cell_refs': changed,
                                  'source_pre_entry_pattern_ref': a['cell_pattern_refs'][source],
                                  'source_pre_entry_pattern_orbit_ref':a.get('cell_pattern_orbit_refs',{}).get(source),
                                  'inactive_nonembedded_center_refs': inactive,
                                  'active_nonembedded_center_refs': active,
                                  'entry_frame_ref': after.frame_ref,
                                  'entry_frame_index': after.frame_index,
                                  'ordinary_departure_count': 0, 'reversed_after_departure_count': 0,
                                  'preserved_after_departure_count': 0})
        for _, row in sorted(rows.items()):
            source = row['source_cell_ref']
            if source not in old_cells.values() or source in new_cells.values():
                continue
            row['ordinary_departure_count'] += 1
            active_cells = row['active_nonembedded_center_refs']
            inactive_cells = row['inactive_nonembedded_center_refs']
            row['reversed_after_departure_count'] += int(all(
                b['center_equals_embedded'].get(ref) is False for ref in inactive_cells) and all(
                b['center_equals_embedded'].get(ref) is True for ref in active_cells))
            row['preserved_after_departure_count'] += int(all(
                b['center_equals_embedded'].get(ref) is True for ref in inactive_cells) and all(
                b['center_equals_embedded'].get(ref) is False for ref in active_cells))
        if len(rows) > 3:
            return FrozenMap({'contact_cell_measurement_bound_reached': True, 'contact_remote_cell_rows': ()})
    if len(slot_transitions) > 16:
        return FrozenMap({'contact_cell_measurement_bound_reached': True, 'contact_remote_cell_rows': ()})
    return FrozenMap({'contact_cell_measurement_bound_reached': False,
                      'contact_cell_visit_count': visits,
                      'boundary_slot_transition_rows': tuple(slot_transitions),
                      'contact_remote_cell_rows': tuple(FrozenMap(row) for _, row in sorted(rows.items()))})
