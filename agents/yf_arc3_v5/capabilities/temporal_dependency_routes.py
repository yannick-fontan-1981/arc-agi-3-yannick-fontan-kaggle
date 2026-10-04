"""Finite joint search under supplied response and clock alternatives.

This module measures witnesses. It assigns no role, chooses no model, and never
authorizes execution. Traversal orders and every law come from source inputs.
"""
from collections import deque
from itertools import product
from typing import Literal

from pydantic import Field, model_validator
from agents.yf_arc3_v5.capabilities.contracts import FrozenModel, FrozenMap, Ref


class RecordedCellTape(FrozenModel):
    recording_ref: Ref
    cell_refs: tuple[Ref, ...] = Field(min_length=2, max_length=65)
    read_index: int = Field(ge=0, le=128)


class ContactCellResponse(FrozenModel):
    source_cell_ref: Ref
    inactive_blocked_refs: tuple[Ref, ...] = Field(default=(), max_length=256)
    active_blocked_refs: tuple[Ref, ...] = Field(default=(), max_length=256)
    initial_active_value: bool
    response_kinds: tuple[Literal['while_occupied', 'latched_on_entry'], ...] = Field(min_length=1, max_length=2)


class JointCellSearchInput(FrozenModel):
    node_refs: tuple[Ref, ...] = Field(min_length=1, max_length=256)
    edges: tuple[tuple[Ref, Ref, Ref], ...] = Field(min_length=1, max_length=1024)
    initial_cell_ref: Ref
    terminal_cell_ref: Ref
    terminal_cell_refs: tuple[Ref, ...] = Field(default=(), max_length=8)
    recordings: tuple[RecordedCellTape, ...] = Field(max_length=3)
    responses: tuple[ContactCellResponse, ...] = Field(max_length=3)
    fixed_blocked_refs: tuple[Ref, ...] = Field(default=(), max_length=256)
    clock_kinds: tuple[Literal['before_motion_every_attempt', 'after_successful_motion'], ...] = Field(min_length=1, max_length=2)
    endpoint_behavior: Literal['retain', 'remove', 'unknown']
    # Alternative identities and complete action orders are supplied by DRM.
    traversal_orders: tuple[tuple[Ref, tuple[Ref, ...]], ...] = Field(min_length=1, max_length=3)
    premise_refs: tuple[Ref, ...] = Field(min_length=1, max_length=32)
    depth_bound: int = Field(default=64, ge=1, le=64)
    expansion_bound: int = Field(default=65536, ge=1, le=65536)
    state_bound: int = Field(default=4096, ge=1, le=4096)

    @model_validator(mode='after')
    def validate_domain(self):
        nodes = frozenset(self.node_refs)
        if len(nodes) != len(self.node_refs):
            raise ValueError('cell nodes must be unique')
        referenced = [self.initial_cell_ref, self.terminal_cell_ref, *self.terminal_cell_refs, *self.fixed_blocked_refs]
        for tape in self.recordings:
            referenced.extend(tape.cell_refs)
        for response in self.responses:
            referenced.extend((response.source_cell_ref, *response.inactive_blocked_refs, *response.active_blocked_refs))
            if len(set(response.response_kinds)) != len(response.response_kinds):
                raise ValueError('response alternatives must be unique')
        if not set(referenced).issubset(nodes):
            raise ValueError('all referenced cells must belong to the finite domain')
        if len({(origin, action) for origin, action, dest in self.edges}) != len(self.edges):
            raise ValueError('each cell/action must have one destination')
        if any(origin not in nodes or dest not in nodes for origin, action, dest in self.edges):
            raise ValueError('edge leaves the supplied domain')
        actions = frozenset(action for origin, action, dest in self.edges)
        if any(len(set(order)) != len(order) or set(order) != actions for ref, order in self.traversal_orders):
            raise ValueError('each traversal order must cover the exact action domain')
        if len({ref for ref, order in self.traversal_orders}) != len(self.traversal_orders):
            raise ValueError('traversal alternative refs must be unique')
        model_count = len(self.clock_kinds)
        for response in self.responses:
            model_count *= len(response.response_kinds)
        if model_count > 8:
            raise ValueError('joint response/clock alternatives exceed eight models')
        return self


def measure_joint_cell_routes(value: JointCellSearchInput) -> FrozenMap:
    edges = {(origin, action): dest for origin, action, dest in value.edges}
    models = tuple((clock, kinds) for clock in value.clock_kinds
                   for kinds in product(*(response.response_kinds for response in value.responses)))
    fixed = frozenset(value.fixed_blocked_refs)

    def tape_cell(tape, index):
        if index < len(tape.cell_refs):
            return tape.cell_refs[index], True
        if value.endpoint_behavior == 'retain':
            return tape.cell_refs[-1], True
        if value.endpoint_behavior == 'remove':
            return None, True
        return None, False

    def presence(cell, indices):
        occupants = {cell}
        for tape, index in zip(value.recordings, indices):
            replay_cell, known = tape_cell(tape, index)
            if not known:
                return None
            if replay_cell is not None:
                occupants.add(replay_cell)
        return tuple(response.source_cell_ref in occupants for response in value.responses)

    def advance(indices):
        return tuple(min(index+1, len(tape.cell_refs)) for tape, index in zip(value.recordings, indices))

    def outputs(occupied, latches, kinds):
        return tuple(occ if kind == 'while_occupied' else latch for occ, latch, kind in zip(occupied, latches, kinds))

    def transition(state, action, model):
        cell, indices, latches = state
        clock, kinds = model
        old_presence = presence(cell, indices)
        if old_presence is None:
            return None
        if clock == 'before_motion_every_attempt':
            indices = advance(indices)
        current_presence = presence(cell, indices)
        if current_presence is None:
            return None
        latches = tuple(latch or (now and not old) for latch, now, old in zip(latches, current_presence, old_presence))
        active_values = outputs(current_presence, latches, kinds)
        blocked = set(fixed)
        for response, active in zip(value.responses, active_values):
            blocked.update(response.active_blocked_refs if active else response.inactive_blocked_refs)
        dest = edges.get((cell, action))
        if dest is None or dest == cell or dest in blocked:
            return None  # No invented WAIT and no blocked clock tick as progress.
        if clock == 'after_successful_motion':
            indices = advance(indices)
        next_presence = presence(dest, indices)
        if next_presence is None:
            return None
        latches = tuple(latch or (now and not old) for latch, now, old in zip(latches, next_presence, current_presence))
        return dest, indices, latches

    initial = (value.initial_cell_ref, tuple(tape.read_index for tape in value.recordings),
               tuple(response.initial_active_value for response in value.responses))
    initial_states = tuple(initial for model in models)
    facts = {}
    expanded = 0
    constructed = 0
    bound_reached = False
    signatures = set()
    for alternative_ref, order in value.traversal_orders:
        frontier = deque(((initial_states, ()),))
        visited = {initial_states}
        constructed += 1
        route = None
        first_states = None
        while frontier:
            states, prefix = frontier.popleft()
            if prefix and all(state[0] in (value.terminal_cell_refs or (value.terminal_cell_ref,)) for state in states):
                route = prefix
                break
            if len(prefix) >= value.depth_bound:
                continue
            for action in order:
                successors = []
                for state, model in zip(states, models):
                    if expanded >= value.expansion_bound:
                        bound_reached = True
                        break
                    expanded += 1
                    successor = transition(state, action, model)
                    if successor is None:
                        break
                    successors.append(successor)
                if bound_reached:
                    break
                if len(successors) != len(models):
                    continue
                key = tuple(successors)
                if key in visited:
                    continue
                if constructed >= value.state_bound:
                    bound_reached = True
                    break
                constructed += 1
                visited.add(key)
                frontier.append((key, (*prefix, action)))
            if bound_reached:
                break
        if bound_reached:
            return FrozenMap({'joint_search_bound_reached': True, 'expanded_transition_count': expanded,
                              'joint_route_rows': (), 'model_count': len(models), 'constructed_state_count': constructed})
        if route is not None and route not in signatures:
            signatures.add(route)
            first_states = tuple(transition(state, route[0], model) for state, model in zip(initial_states, models))
            facts[alternative_ref] = FrozenMap({'alternative_ref': alternative_ref, 'action_refs': route,
                'first_action_ref': route[0], 'route_length': len(route), 'premise_refs': value.premise_refs,
                'first_expected_joint_states': first_states, 'all_models_reach_terminal': True,
                'response_source_cell_refs': tuple(response.source_cell_ref for response in value.responses)})
    return FrozenMap({'joint_search_bound_reached': False, 'expanded_transition_count': expanded,
                      'joint_route_rows': tuple(facts[key] for key in sorted(facts)), 'model_count': len(models), 'constructed_state_count': constructed})


def measure_unmatched_current_instance_bounds(value) -> tuple:
    """Measure the exact complement of current prefix-matched instance boxes."""
    from agents.yf_arc3_v5.capabilities.contracts import BoundingBox
    contract=value.route_generation_policy.get('temporal_joint_cell_route_measurement_contract', {})
    sequences=value.canonical_instance_sequence_facts
    grids=value.periodic_cell_grids.candidates
    if not contract.get('measure_unmatched_current_instances') or len(grids)!=1 or not sequences:
        return ()
    if any(row.get('observation_frame_ref')!=value.current_observation_frame_ref
           or row.get('has_divergence') is not False
           or row.get('independent_command_observed') is not True
           or not row.get('candidate_instance_bounds') for row in sequences):
        return ()
    excluded=frozenset(BoundingBox.model_validate(row['candidate_instance_bounds'][-1]) for row in sequences)
    grid=grids[0]
    return tuple(component.bbox for component in value.components.components
                 if component.bbox.height==grid.cell_height and component.bbox.width==grid.cell_width
                 and component.area+1==grid.cell_height*grid.cell_width and component.bbox not in excluded)


def measure_current_nonrecorded_input_bounds(value) -> tuple:
    """Intersect independent observed scopes; never prefer a conflicting scope."""
    aligned = value.input_aligned_entity_bboxes
    unmatched = measure_unmatched_current_instance_bounds(value)
    if not aligned:
        return unmatched
    if not unmatched:
        return aligned
    unmatched_set = frozenset(unmatched)
    return tuple(box for box in aligned if box in unmatched_set)


def _measure_fresh_return_tape_pack(value, contract, inventory):
    """Keep raw recorded spans separate from any observed replay binding."""
    from agents.yf_arc3_v5.capabilities.contracts import BoundingBox
    if not contract.get('measure_fresh_return_conditional_tape_pack') or not inventory.get('return_recording_span_packs'):
        return (), (), ()
    capacity = measure_phase_capacity_from_cells(value)
    if capacity.get('phase_slot_domain_description_count') != 1:
        return (), (), ()
    cells = inventory.get('current_observed_cell_facts') or {}
    bounds = frozenset(BoundingBox.model_validate(row) for _,row in sorted(cells.get('cell_bounds', {}).items()))
    matching = []
    for pack in inventory.get('return_recording_span_packs', ()):
        spans = pack.get('recording_span_rows', ())
        box = BoundingBox.model_validate(pack['current_original_bounds'])
        methods = tuple(row for row in value.canonical_phase_slot_method_facts
                        if pack['phase_action_ref'] in row.get('measured_return_action_refs', ()))
        if (pack.get('observation_frame_ref') != inventory.get('observation_frame_ref')
                or not 1 <= len(spans) <= 3 or box not in bounds or not methods
                or len(spans) != capacity['phase_slot_measured_member_count'] - 1
                    - capacity['phase_slot_remaining_in_observed_direction']):
            continue
        if any(not 2 <= len(row['recorded_instance_bounds']) <= 65
               or any(BoundingBox.model_validate(raw) not in bounds for raw in row['recorded_instance_bounds'])
               for row in spans):
            continue
        matching.append((spans, box, tuple(row['claim_ref'] for row in methods)))
    if len(matching) != 1:
        return (), (), ()
    spans, box, premises = matching[0]
    rows = tuple(FrozenMap({'recorded_instance_bounds': row['recorded_instance_bounds'],
        'claim_ref': inventory['claim_ref'], 'recording_measurement_ref': row['recording_frame_ref'],
        'candidate_observed_state_count': 0}) for row in spans)
    return rows, (box,), premises


def measure_temporal_routes_from_cells(value, contract) -> FrozenMap:
    """Bind exact current measurements to source-supplied finite laws and orders."""
    from agents.yf_arc3_v5.capabilities.contracts import BoundingBox
    inventories = value.canonical_temporal_contact_inventory_facts
    sequences = value.canonical_instance_sequence_facts
    if len(inventories) != 1 or any(row.get('has_divergence') is not False for row in sequences):
        return FrozenMap()
    inventory = inventories[0]
    conditional_boxes, conditional_premises = (), ()
    if not sequences:
        sequences, conditional_boxes, conditional_premises = _measure_fresh_return_tape_pack(value, contract, inventory)
    if not sequences:
        return FrozenMap()
    cells = inventory.get('current_observed_cell_facts')
    contacts = inventory.get('contact_remote_cell_rows')
    if not cells or not contacts or any(not row.get('recorded_instance_bounds') for row in sequences):
        return FrozenMap()
    bounds = {ref: BoundingBox.model_validate(box) for ref, box in sorted(cells['cell_bounds'].items())}
    boxes=conditional_boxes or measure_current_nonrecorded_input_bounds(value)
    controlled_refs = tuple(ref for ref, box in sorted(bounds.items()) if box in frozenset(boxes))
    if len(controlled_refs) != 1:
        return FrozenMap()
    initial_ref = controlled_refs[0]
    initial_box = bounds[initial_ref]
    enclosures = tuple(component for component in value.components.components
                       if initial_box.height < component.bbox.height <= 2*initial_box.height
                       and initial_box.width < component.bbox.width <= 2*initial_box.width
                       and component.area < component.bbox.height*component.bbox.width)
    target_refs = set()
    for component in enclosures:
        for ref, box in sorted(bounds.items()):
            if not (component.bbox.top <= box.top <= box.bottom <= component.bbox.bottom
                    and component.bbox.left <= box.left <= box.right <= component.bbox.right):
                continue
            center = ((box.top+box.bottom)//2, (box.left+box.right)//2)
            if all(value.frame.rows[r][c] == cells['embedded_value']
                   or ((r,c) == center and value.frame.rows[r][c] == component.value)
                   for r in range(box.top,box.bottom+1) for c in range(box.left,box.right+1)):
                target_refs.add(ref)
    if len(target_refs) != 1:
        return FrozenMap()
    tapes = []
    for row in sequences:
        tape_cells = []
        for raw_box in row['recorded_instance_bounds']:
            box = BoundingBox.model_validate(raw_box)
            matches = tuple(ref for ref, cell_box in sorted(bounds.items()) if box == cell_box)
            if len(matches) != 1:
                return FrozenMap()
            tape_cells.append(matches[0])
        tapes.append(RecordedCellTape(recording_ref=row.get('recording_measurement_ref', row['claim_ref']), cell_refs=tuple(tape_cells),
                                     read_index=row['candidate_observed_state_count']))
        if row.get('candidate_instance_bounds'):
            actual_box = BoundingBox.model_validate(row['candidate_instance_bounds'][-1])
            expected_box = bounds[tape_cells[min(row['candidate_observed_state_count'], len(tape_cells)-1)]]
            if actual_box != expected_box:
                return FrozenMap({'temporal_current_tape_endpoint_contradiction': True})
    dynamic = set()
    sources = set()
    responses = []
    center_flags = cells['center_equals_embedded']
    for row in contacts:
        inactive = tuple(row['inactive_nonembedded_center_refs'])
        active = tuple(row['active_nonembedded_center_refs'])
        active_pattern = all(center_flags.get(ref) is True for ref in inactive) and all(
            center_flags.get(ref) is False for ref in active)
        inactive_pattern = all(center_flags.get(ref) is False for ref in inactive) and all(
            center_flags.get(ref) is True for ref in active)
        if active_pattern == inactive_pattern:
            return FrozenMap()  # Neither ambiguous nor unknown masks get a default.
        current_occupants = {initial_ref, *(tape.cell_refs[min(tape.read_index, len(tape.cell_refs)-1)] for tape in tapes)}
        if row['source_cell_ref'] in current_occupants and not active_pattern:
            return FrozenMap({'temporal_current_contact_output_contradiction': True})
        dynamic.update((*inactive, *active))
        sources.add(row['source_cell_ref'])
        response_kinds = tuple(kind for kind in contract['response_kinds'] if not any(
            row.get(counter, 0) > 0 for counter in contract.get('response_exclusion_counters', {}).get(kind, ())))
        if not response_kinds:
            return FrozenMap({'temporal_contact_law_domain_empty': True})
        responses.append(ContactCellResponse(source_cell_ref=row['source_cell_ref'],
            inactive_blocked_refs=inactive, active_blocked_refs=active,
            initial_active_value=active_pattern, response_kinds=response_kinds))
    by_top_left = {(box.top,box.left): ref for ref,box in sorted(bounds.items())}
    pitch_r, pitch_c = cells['grid_geometry'][4:6]
    edges = tuple((ref, str(action), dest) for ref,box in sorted(bounds.items())
                  for action,dr,dc in value.interface_action_translation_deltas
                  if (dest := by_top_left.get((box.top+dr*pitch_r, box.left+dc*pitch_c))) is not None)
    actions = frozenset(action for origin,action,dest in edges)
    orders = tuple((ref, tuple(action for action in order if action in actions))
                   for ref,order in contract['traversal_orders'])
    occupied = frozenset(cells['instance_cell_refs'].values())
    target_ref = next(iter(target_refs))
    fixed = tuple(ref for ref in bounds if center_flags.get(ref) is False
                  and ref not in dynamic and ref not in sources and ref not in occupied and ref != target_ref)
    premises = tuple(dict.fromkeys((inventory['claim_ref'], *conditional_premises, *(row['claim_ref'] for row in sequences))))
    if len(contract['clock_kinds']) * len(contract['response_kinds']) ** len(responses) > 8:
        return FrozenMap({'joint_search_bound_reached': True, 'joint_route_rows': (), 'model_domain_bound_reached': True})
    request = JointCellSearchInput(node_refs=tuple(bounds), edges=edges, initial_cell_ref=initial_ref,
        terminal_cell_ref=target_ref, recordings=tuple(tapes), responses=tuple(responses),
        fixed_blocked_refs=fixed, clock_kinds=tuple(contract['clock_kinds']),
        endpoint_behavior=contract['endpoint_behavior'], traversal_orders=orders, premise_refs=premises,
        depth_bound=contract['depth_bound'], expansion_bound=contract['expansion_bound'], state_bound=contract['state_bound'])
    measured = measure_joint_cell_routes(request)
    if conditional_boxes:
        rows = tuple(FrozenMap.overlay(FrozenMap({
            'first_step_contributions_pairwise_distinct_in_every_model': all(
                len(frozenset((state[0], *(tape.cell_refs[min(index,len(tape.cell_refs)-1)]
                    for tape,index in zip(tapes,state[1]))))) == len(tapes)+1
                for state in row['first_expected_joint_states'])}), row)
            for row in measured.get('joint_route_rows', ()))
        measured = FrozenMap.overlay(FrozenMap({'conditional_return_tape_pack_measured': True,
            'joint_route_rows':rows}), measured)
    if measured['joint_search_bound_reached'] or measured['joint_route_rows']:
        return measured
    if not contract.get('same_pattern_dependency_target_measurement_enabled', False):
        return measured
    source_patterns = frozenset(row.get('source_pre_entry_pattern_ref') for row in contacts
                                if row.get('source_pre_entry_pattern_ref'))
    source_orbits=frozenset(row.get('source_pre_entry_pattern_orbit_ref') for row in contacts
        if row.get('source_pre_entry_pattern_orbit_ref')) if contract.get('allow_rigid_contact_pattern_correspondence') else frozenset()
    endpoint_refs = frozenset(tape.cell_refs[-1] for tape in tapes)
    candidates = tuple(ref for ref in bounds if ref not in sources and ref not in endpoint_refs
                       and ref != initial_ref and ref != target_ref
                       and (cells['cell_pattern_refs'].get(ref) in source_patterns
                            or cells.get('cell_pattern_orbit_refs',{}).get(ref) in source_orbits))
    if len(candidates) != 1:
        return FrozenMap({**dict(measured), 'same_pattern_dependency_target_count': len(candidates)})
    dependency_ref = candidates[0]
    dependency_request = request.model_copy(update={
        'terminal_cell_ref': dependency_ref,
        'fixed_blocked_refs': tuple(ref for ref in fixed if ref != dependency_ref),
        'state_bound': max(1, request.state_bound-measured['constructed_state_count']),
        'expansion_bound': max(1, request.expansion_bound-measured['expanded_transition_count'])})
    if (measured['constructed_state_count'] >= request.state_bound
            or measured['expanded_transition_count'] >= request.expansion_bound):
        return FrozenMap({'joint_search_bound_reached': True, 'joint_route_rows': ()})
    dependent = measure_joint_cell_routes(dependency_request)
    return FrozenMap({'joint_search_bound_reached': dependent['joint_search_bound_reached'],
        'conditional_return_tape_pack_measured': bool(conditional_boxes),
        'joint_route_rows': tuple(FrozenMap({**dict(row), 'same_pattern_dependency_target': True,
                                           'terminal_cell_ref': dependency_ref})
                                 for row in dependent['joint_route_rows']),
        'constructed_state_count': measured['constructed_state_count']+dependent['constructed_state_count'],
        'expanded_transition_count': measured['expanded_transition_count']+dependent['expanded_transition_count'],
        'model_count': dependent['model_count'], 'same_pattern_dependency_target_count': 1})


def measure_contact_pattern_routes(value, contract) -> FrozenMap:
    """Shortest supplied-order witnesses to current measured pattern matches.

    Only pattern identities are transported across scenes. No old cell, route or
    target handle is rebound. DRM decides whether such a witness is a useful probe.
    """
    from agents.yf_arc3_v5.capabilities.temporal_cells import measure_temporal_cells
    from agents.yf_arc3_v5.capabilities.contracts import BoundingBox
    grids=value.periodic_cell_grids.candidates
    if len(grids)!=1 or value.canonical_instance_sequence_facts:
        return FrozenMap()
    current_boxes=frozenset(value.input_aligned_entity_bboxes)
    bodies=tuple(component for component in value.components.components
                 if component.bbox.height==grids[0].cell_height and component.bbox.width==grids[0].cell_width
                 and component.area+1==component.bbox.height*component.bbox.width
                 and (not current_boxes or component.bbox in current_boxes))
    if len(bodies)!=1:
        return FrozenMap()
    measured=measure_temporal_cells(value.frame, grids[0], {'measurement:controlled':bodies[0].model_dump(mode='python')})
    if not measured:
        return FrozenMap()
    patterns=frozenset(pattern for method in value.canonical_contact_source_pattern_method_facts
                      for pattern in method.get('contact_source_pattern_refs', ()))
    orbits=frozenset(pattern for method in value.canonical_contact_source_pattern_method_facts
        for pattern in method.get('contact_source_pattern_orbit_refs',())) if contract.get('allow_rigid_contact_pattern_correspondence') else frozenset()
    initial=measured['instance_cell_refs']['measurement:controlled']
    targets=tuple(ref for ref in measured['cell_refs'] if ref!=initial
                  and (measured['cell_pattern_refs'][ref] in patterns
                       or measured.get('cell_pattern_orbit_refs',{}).get(ref) in orbits))
    if not targets or len(targets)>8:
        return FrozenMap()
    bounds={ref:BoundingBox.model_validate(box) for ref,box in sorted(measured['cell_bounds'].items())}
    positions={(box.top,box.left):ref for ref,box in sorted(bounds.items())}
    grid=grids[0]
    edges=tuple((ref,str(action),dest) for ref,box in sorted(bounds.items())
        for action,dr,dc in value.interface_action_translation_deltas
        if (dest:=positions.get((box.top+dr*grid.row_pitch,box.left+dc*grid.col_pitch))) is not None)
    if not edges:
        return FrozenMap()
    actions=frozenset(action for _,action,_ in edges)
    orders=tuple((ref,tuple(action for action in order if action in actions)) for ref,order in contract['traversal_orders'])
    premises=tuple(method['claim_ref'] for method in value.canonical_contact_source_pattern_method_facts
                   if method.get('claim_ref') and (set(method.get('contact_source_pattern_refs', ())).intersection(patterns)
                       or set(method.get('contact_source_pattern_orbit_refs',())).intersection(orbits)))
    if not premises:
        return FrozenMap()
    request=JointCellSearchInput(node_refs=measured['cell_refs'],edges=edges,initial_cell_ref=initial,
        terminal_cell_ref=initial,terminal_cell_refs=targets,recordings=(),responses=(),
        fixed_blocked_refs=tuple(ref for ref in measured['cell_refs'] if ref!=initial and ref not in targets
                                 and measured['center_equals_embedded'][ref] is False),
        clock_kinds=tuple(contract['clock_kinds']),endpoint_behavior=contract['endpoint_behavior'],traversal_orders=orders,
        premise_refs=premises,depth_bound=contract['depth_bound'],state_bound=contract['state_bound'],
        expansion_bound=contract['expansion_bound'])
    result=measure_joint_cell_routes(request)
    return FrozenMap({'joint_search_bound_reached':result['joint_search_bound_reached'],
        'joint_route_rows':result['joint_route_rows'], 'body_candidate_count':len(bodies),
        'expanded_transition_count':result['expanded_transition_count'],
        'constructed_state_count':result['constructed_state_count']})


def measure_phase_capacity_from_cells(value) -> FrozenMap:
    """Measure a repeated marker domain against observed return-operator shifts."""
    from agents.yf_arc3_v5.capabilities.contracts import BoundingBox
    inventories = value.canonical_temporal_contact_inventory_facts
    if len(inventories) > 1:
        return FrozenMap()
    contract=value.route_generation_policy.get('temporal_joint_cell_route_measurement_contract', {})
    if inventories:
        inventory=inventories[0]
    else:
        if not contract.get('measure_current_phase_cells_before_contact_inventory'):
            return FrozenMap()
        grids=value.periodic_cell_grids.candidates
        if len(grids)!=1:
            return FrozenMap()
        grid=grids[0]
        current_bodies=tuple(component for component in value.components.components
            if component.bbox.height==grid.cell_height and component.bbox.width==grid.cell_width
            and component.area+1==component.bbox.height*component.bbox.width)
        if len(current_bodies)!=1:
            return FrozenMap()
        from agents.yf_arc3_v5.capabilities.temporal_cells import measure_temporal_cells
        cells=measure_temporal_cells(value.frame,grid,
            {'measurement:current-body':current_bodies[0].model_dump(mode='python')},
            tuple(component.model_dump(mode='python') for component in value.components.components))
        inventory=FrozenMap({'current_observed_cell_facts':cells,'contact_remote_cell_rows':()})
    cells = inventory.get('current_observed_cell_facts')
    if not cells:
        return FrozenMap()
    bboxes = frozenset(value.input_aligned_entity_bboxes or measure_unmatched_current_instance_bounds(value))
    bodies = tuple(component for component in value.components.components if (not bboxes or component.bbox in bboxes)
                   and component.bbox.height == cells['grid_geometry'][2]
                   and component.bbox.width == cells['grid_geometry'][3]
                   and (bboxes or component.area+1==component.bbox.height*component.bbox.width))
    if len(bodies) != 1:
        return FrozenMap()
    body = bodies[0]
    cell_matches = tuple(ref for ref, box in sorted(cells['cell_bounds'].items())
                         if BoundingBox.model_validate(box) == body.bbox)
    if len(cell_matches) != 1:
        return FrozenMap()
    active_cell = cell_matches[0]
    endpoints = tuple(BoundingBox.model_validate(row['recorded_instance_bounds'][-1])
                      for row in value.canonical_instance_sequence_facts if row.get('recorded_instance_bounds'))
    observed_return_refs = frozenset(row.get('phase_action_ref') for row in value.canonical_instance_sequence_facts
                                    if row.get('phase_action_ref'))
    shifts = tuple(shift for shift in inventory.get('boundary_slot_transition_rows', ())
                   if shift.get('action_ref') in observed_return_refs) + tuple(
        shift for method in value.canonical_phase_slot_method_facts
        for shift in method.get('boundary_slot_transition_rows', ())
        if shift.get('action_ref') in method.get('measured_return_action_refs', ()))
    shifts = tuple(shift for shift in shifts if shift.get('value') == body.value and abs(shift.get('index_delta', 0)) == 1)
    descriptions = set()
    for slots in cells.get('boundary_slot_rows', ()):
        indices = tuple(i for i, palette in enumerate(slots['member_values']) if palette == body.value)
        if len(indices) != 1:
            continue
        for shift in shifts:
            compatible_unit=bool(contract.get('allow_slot_unit_geometry_rebinding')
                and shift.get('unit_descriptor') is not None
                and shift.get('unit_descriptor')==slots.get('unit_descriptor'))
            if shift['group_ref'] != slots['group_ref'] and not compatible_unit:
                continue
            remaining = slots['member_count']-1-indices[0] if shift['index_delta'] > 0 else indices[0]
            descriptions.add((slots['member_count'], remaining, shift['action_ref'],
                              slots.get('unit_descriptor'),slots.get('unit_origin')))
    if len(descriptions) != 1:
        return FrozenMap({'phase_slot_domain_description_count': len(descriptions)})
    count, remaining, operator, _unit, _origin = next(iter(descriptions))
    method_premises = tuple(method['claim_ref'] for method in value.canonical_phase_slot_method_facts
                            if method.get('claim_ref') and any(shift.get('value') == body.value
                                and abs(shift.get('index_delta', 0)) == 1
                                and shift.get('action_ref') in method.get('measured_return_action_refs', ())
                                for shift in method.get('boundary_slot_transition_rows', ())))
    return FrozenMap({'phase_slot_domain_description_count': 1,
        'phase_current_observed_cell_facts':cells,
        'phase_slot_measured_member_count': count, 'phase_slot_remaining_in_observed_direction': remaining,
        'phase_slot_observed_return_action_ref': operator,
        'active_body_matches_unrecorded_contact_source': any(row['source_cell_ref'] == active_cell
            for row in inventory.get('contact_remote_cell_rows', ())) and body.bbox not in endpoints,
        'phase_slot_inventory_premise_refs': tuple(dict.fromkeys((*(
            (inventory['claim_ref'],) if inventory.get('claim_ref') else ()), *method_premises,
            *(row['claim_ref'] for row in value.canonical_instance_sequence_facts))))})


def measure_temporal_allocation_cells(value, capacity) -> FrozenMap:
    """Current measured pattern set, enclosure set and marker cardinalities.

    No role or allocation is selected here. An inaccessible matching region is
    retained in the set just like an accessible one. DRM interprets the facts.
    """
    from agents.yf_arc3_v5.capabilities.contracts import BoundingBox
    inventories=value.canonical_temporal_contact_inventory_facts
    if len(inventories)>1 or capacity.get('phase_slot_domain_description_count')!=1:
        return FrozenMap()
    inventory=inventories[0] if inventories else FrozenMap()
    cells=inventory.get('current_observed_cell_facts') or capacity.get('phase_current_observed_cell_facts')
    if not cells:
        return FrozenMap()
    bounds={ref:BoundingBox.model_validate(box) for ref,box in sorted(cells['cell_bounds'].items())}
    patterns=frozenset(pattern for method in value.canonical_contact_source_pattern_method_facts
                      for pattern in method.get('contact_source_pattern_refs', ()))
    contract=value.route_generation_policy.get('temporal_joint_cell_route_measurement_contract',{})
    orbits=frozenset(pattern for method in value.canonical_contact_source_pattern_method_facts
        for pattern in method.get('contact_source_pattern_orbit_refs',())) if contract.get('allow_rigid_contact_pattern_correspondence') else frozenset()
    matched=frozenset(ref for ref in cells['cell_refs'] if cells['cell_pattern_refs'][ref] in patterns
                     or cells.get('cell_pattern_orbit_refs',{}).get(ref) in orbits)
    entered=frozenset(row['source_cell_ref'] for row in inventory.get('contact_remote_cell_rows', ()))
    pattern_refs=tuple(sorted(matched|entered))
    if not pattern_refs or len(pattern_refs)>3:
        return FrozenMap()
    receiving=set()
    height,width=cells['grid_geometry'][2:4]
    for component in value.components.components:
        if not (height<component.bbox.height<=2*height and width<component.bbox.width<=2*width
                and component.area<component.bbox.height*component.bbox.width):
            continue
        for ref,box in sorted(bounds.items()):
            if not (component.bbox.top<=box.top<=box.bottom<=component.bbox.bottom
                    and component.bbox.left<=box.left<=box.right<=component.bbox.right):
                continue
            center=((box.top+box.bottom)//2,(box.left+box.right)//2)
            if all(value.frame.rows[r][c]==cells['embedded_value']
                   or ((r,c)==center and value.frame.rows[r][c]==component.value)
                   for r in range(box.top,box.bottom+1) for c in range(box.left,box.right+1)):
                receiving.add(ref)
    return FrozenMap({'current_contact_pattern_cell_refs':pattern_refs,
        'current_contact_pattern_cell_count':len(pattern_refs),
        'current_receiving_enclosure_cell_refs':tuple(sorted(receiving)),
        'current_receiving_enclosure_cell_count':len(receiving),
        'phase_member_count_equals_contact_and_receiving_count':
            capacity['phase_slot_measured_member_count']==len(pattern_refs)+len(receiving),
        'current_allocation_measurement_premise_refs':tuple(dict.fromkeys((
            *capacity.get('phase_slot_inventory_premise_refs',()),
            *(method['claim_ref'] for method in value.canonical_contact_source_pattern_method_facts
              if method.get('claim_ref')))))})
