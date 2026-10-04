"""Compile observed fixed-axis transport into a bounded, revisable body model.

No palette, game, target coordinate or action witness is supplied here. Geometry
comes from the current tracking, interventions and an independently proposed goal.
This module measures routes; DRM/SRC must select and release their first action.
"""
from fractions import Fraction
from heapq import heappop, heappush
from itertools import count

from agents.yf_arc3_v5.capabilities.articulated_channel_kinematics import (
    measure_articulated_kinematics, measure_articulated_shared_snapshot,
)
from agents.yf_arc3_v5.capabilities.contracts import (
    ArticulatedBodyState, ArticulatedKinematicsInput, ArticulatedLocalOperation,
    ArticulatedSharedSnapshotInput,
)
from agents.yf_arc3_v5.logos.types import FrozenMap


def _rational(point):
    return tuple((Fraction(x).numerator, Fraction(x).denominator) for x in point)


def _point(point):
    return tuple(Fraction(n, d) for n, d in point)


def _root(component, axis):
    b = component.bbox
    if axis[0] == 0:
        return (Fraction(b.top+b.bottom, 2), Fraction(b.left if axis[1]>0 else b.right+1))
    return (Fraction(b.top if axis[0]>0 else b.bottom+1), Fraction(b.left+b.right, 2))


def compile_observed_axis_configuration(*, observations, graph, goal, tracking,
                                        frame, context_epoch):
    """Measure current roots, parent offsets, supports and scoped commands."""
    if len(observations)>64 or len(tracking.entities)>128:
        raise ValueError('observed axis configuration bound exceeded')
    current = {e.entity_id:e.component for e in tracking.entities}
    if len(current)!=len(tracking.entities):
        raise ValueError('duplicate axis configuration identity')
    if (graph.get('context_epoch')!=context_epoch or goal.get('context_epoch')!=context_epoch
            or graph.get('graph_status')!='provisional_directed_transport_forest'):
        return None
    refs = tuple(graph['receiver_refs'])
    if not 1<len(refs)<=16 or any(ref not in current for ref in refs):
        return None
    rows = tuple(o for o in observations if o['context_epoch']==context_epoch
                 and o['extent_entity_ref'] in refs)
    parents = {b:a for a,b in graph['covering_effect_pairs']}
    if len(parents)!=len(graph['covering_effect_pairs']):
        return None
    axes, roots, extents, commands = {}, {}, {}, []
    for ref in refs:
        evidence = tuple(o for o in rows if o['extent_entity_ref']==ref)
        directions = {tuple(o['axis_vector']) for o in evidence}
        if len(directions)!=1:
            return None
        axis, = directions
        if axis not in ((1,0),(-1,0),(0,1),(0,-1)):
            return None
        axes[ref], roots[ref] = axis, _root(current[ref], axis)
        box = current[ref].bbox
        extents[ref] = box.height if axis[0] else box.width
        # Overdraw is allowed only by a currently tracked transported part.
        rectangle = {(r,c) for r in range(box.top,box.bottom+1)
                     for c in range(box.left,box.right+1)}
        movable = {part for o in evidence for part in o['co_translated_entity_refs']}
        occluders = {p for part in movable if part in current for p in current[part].pixels}
        if rectangle-set(current[ref].pixels)-occluders:
            return None
        for o in evidence:
            data = o['observed_action_data']
            x,y = data.get('x'),data.get('y')
            delta = o['delta_extent']
            if (type(x) is not int or type(y) is not int or type(delta) is not int
                    or delta==0 or not 0<=y<len(frame) or not 0<=x<len(frame[y])
                    or frame[y][x]!=current[ref].value):
                continue
            command = (o['operator_scope_ref'],o['observed_action_ref'],x,y,ref,delta)
            if command not in commands:
                commands.append(command)
    if not commands:
        return None
    passive = tuple(ref for ref in graph['passive_entity_refs'] if ref in current)
    for part in passive:
        carriers = {ref for ref in refs if any(part in o['co_translated_entity_refs']
                    for o in rows if o['extent_entity_ref']==ref)}
        deepest = tuple(ref for ref in carriers
                        if not any(a==ref and b in carriers for a,b in graph['asymmetric_effect_pairs']))
        if len(deepest)!=1:
            return None
        parents[part] = deepest[0]
        roots[part] = (Fraction(current[part].bbox.top),Fraction(current[part].bbox.left))
    marker = goal['marker_entity_ref']
    if marker not in passive or len(refs)+len(passive)>32:
        return None
    bodies = []
    offsets = {}
    for ref in (*refs,*passive):
        parent = parents.get(ref)
        origin = roots[ref]
        local = tuple(a-b for a,b in zip(origin, roots[parent])) if parent else origin
        offsets[ref] = tuple((Fraction(r)-origin[0],Fraction(c)-origin[1])
                             for r,c in current[ref].pixels)
        bodies.append(ArticulatedBodyState(body_ref=ref,parent_ref=parent,
            local_translation=_rational(local),local_axis=_rational(axes.get(ref,(0,0))),
            local_extent=(extents.get(ref,0),1)))
    moving = {p for ref in (*refs,*passive) for p in current[ref].pixels}
    # Background is a revisable scene-plane hypothesis, not a palette binding.
    outside = tuple(c for ref,c in sorted(current.items()) if ref not in refs and ref not in passive)
    max_area = max((c.area for c in outside),default=0)
    planes = tuple(c for c in outside if c.area==max_area)
    if len(planes)!=1 or max_area*2<=sum(len(row) for row in frame):
        return None
    references = set(goal['reference_entity_refs'])
    fixed = {p for ref,c in sorted(current.items()) if ref not in refs and ref not in passive
             and ref not in references and c.value!=planes[0].value for p in c.pixels}
    return dict(bodies=tuple(bodies),refs=refs,parents=parents,axes=axes,
        initial_extents=tuple(extents[ref] for ref in refs),offsets=offsets,
        commands=tuple(commands),marker=marker,target=tuple(goal['wanted_marker_top_left']),
        fixed=frozenset(fixed-moving),height=len(frame),width=len(frame[0]),
        source_term_refs=tuple(sorted(o['term_ref'] for o in rows)),
        goal_ref=goal['goal_ref'],context_epoch=context_epoch)


def measure_configuration_state(model, extents):
    """Compose all supplied changes from one shared snapshot, once per cause."""
    differences = {ref:new-old for ref,new,old in zip(model['refs'],extents,model['initial_extents'])}
    operations = []
    for body in model['bodies']:
        delta = differences.get(body.body_ref,0)
        if delta:
            operations.append(ArticulatedLocalOperation(
                operation_ref=f'extent:{body.body_ref}',cause_ref=f'extent:{body.body_ref}',
                receiver_ref=body.body_ref,order_index=len(operations),
                composition_side='after_local',extent_delta=(delta,1)))
        parent_delta = differences.get(body.parent_ref,0)
        if parent_delta:
            shift = tuple(parent_delta*v for v in model['axes'][body.parent_ref])
            operations.append(ArticulatedLocalOperation(
                operation_ref=f'carrier:{body.body_ref}',cause_ref=f'extent:{body.parent_ref}',
                receiver_ref=body.body_ref,order_index=len(operations),
                composition_side='after_local',translation_delta=_rational(shift)))
    bodies = model['bodies']
    if operations:
        measured = measure_articulated_shared_snapshot(ArticulatedSharedSnapshotInput(
            bodies=bodies,operations=tuple(operations)))
        if measured.status!='applied':
            raise ValueError('unordered observed axis snapshot')
        bodies = measured.resulting_bodies
    geometry = measure_articulated_kinematics(ArticulatedKinematicsInput(bodies=bodies))
    origins = {g.body_ref:_point(g.root_position) for g in geometry.body_geometries}
    supports = {}
    for body in bodies:
        origin = origins[body.body_ref]
        if body.body_ref in model['axes']:
            axis = model['axes'][body.body_ref]
            length = body.local_extent[0]//body.local_extent[1]
            cross = sorted({offset[1] if axis[0] else offset[0]
                            for offset in model['offsets'][body.body_ref]})
            forward = range(length) if max(axis)>0 else range(-length,0)
            pixels = {(origin[0]+along,origin[1]+side) if axis[0]
                      else (origin[0]+side,origin[1]+along) for along in forward for side in cross}
        else:
            pixels = {(origin[0]+r,origin[1]+c) for r,c in model['offsets'][body.body_ref]}
        if any(r.denominator!=1 or c.denominator!=1 for r,c in pixels):
            raise ValueError('nonintegral observed body footprint')
        supports[body.body_ref] = frozenset((int(r),int(c)) for r,c in pixels)
    marker_origin = origins[model['marker']]
    return tuple(int(x) for x in marker_origin),supports


def measure_observed_axis_route(model, *, max_states, max_actions):
    """Measure one shortest route with primitive cost and a Manhattan lower bound.

    Command order is supplied by DRM/SRC. No path family is enumerated. The
    complete forest is checked against fixed foreground and viewport bounds at
    every successor. Exhaustion leaves the goal open, without releasing an action.
    """
    if not 1<=max_states<=100000 or not 1<=max_actions<=64:
        raise ValueError('observed axis route limits outside declared contract')
    if model is None:
        return FrozenMap(dict(model_available=False,route_found=False,route_status='missing_model'))
    initial = model['initial_extents']
    commands = model['commands']
    quantum = max(abs(c[5]) for c in commands)
    index = {ref:i for i,ref in enumerate(model['refs'])}
    target = model['target']
    serial = count()
    start,_ = measure_configuration_state(model,initial)
    def lower_bound(point):
        return (sum(abs(a-b) for a,b in zip(point,target))+quantum-1)//quantum
    agenda = [(lower_bound(start),0,next(serial),initial)]
    cost,previous = {initial:0},{}
    expanded,rejected = 0,0
    while agenda and expanded<max_states:
        _bound,g,_serial,state = heappop(agenda)
        if cost[state]!=g:
            continue
        expanded+=1
        point,_ = measure_configuration_state(model,state)
        if point==target:
            route=[]
            cursor=state
            while cursor in previous:
                cursor,command = previous[cursor]
                route.append(command)
            route.reverse()
            return FrozenMap(dict(model_available=True,route_found=bool(route),
                route_status='goal_already_occupied' if not route else 'measured_complete_route',
                primitive_cost=len(route),first_command=route[0] if route else (),
                command_route=tuple(route),expanded_states=expanded,rejected_clearance_count=rejected,
                goal_ref=model['goal_ref'],source_term_refs=model['source_term_refs'],
                current_marker=start,wanted_marker=target,
                collision_policy='all_observed_bodies_against_provisional_fixed_foreground'))
        if g>=max_actions:
            continue
        for command in commands:
            next_state=list(state)
            i=index[command[4]]
            next_state[i]+=command[5]
            if next_state[i]<1 or next_state[i]>max(model['height'],model['width']):
                continue
            next_state=tuple(next_state)
            if cost.get(next_state,max_actions+1)<=g+1:
                continue
            next_point,supports=measure_configuration_state(model,next_state)
            pixels = set().union(*supports.values())
            if (pixels & model['fixed'] or any(not 0<=r<model['height'] or not 0<=c<model['width']
                                             for r,c in pixels)):
                rejected+=1
                continue
            cost[next_state]=g+1
            previous[next_state]=(state,command)
            heappush(agenda,(g+1+lower_bound(next_point),g+1,next(serial),next_state))
    return FrozenMap(dict(model_available=True,route_found=False,
        route_status='state_bound_exhausted' if agenda else 'no_route_under_observed_commands',
        expanded_states=expanded,rejected_clearance_count=rejected,
        current_marker=start,wanted_marker=target))


def measure_axis_frontier(*, observations, graphs, goals, tracking, frame, agenda, profile):
    """Preserve the agenda and add measurements for a DRM-owned causal frontier."""
    current = {e.entity_id:e.component for e in tracking.entities}
    pairs = tuple((g,t) for g in graphs for t in goals
                  if g['context_epoch']==t['context_epoch'] and t['marker_entity_ref'] in current
                  and all(ref in current for ref in (*g['receiver_refs'],*t['reference_entity_refs'])))
    if len(pairs)!=1:
        return FrozenMap(dict(available=False,agenda=agenda.model_dump(mode='json'),measurement_status='no_unique_current_model_goal'))
    graph,goal = pairs[0]
    rows = tuple(o for o in observations if o['context_epoch']==graph['context_epoch'])
    premises = tuple(sorted(set(graph.get('premise_claim_refs',()))
                            |set(goal.get('premise_claim_refs',()))))
    if not graph.get('premise_claim_refs') or not goal.get('premise_claim_refs'):
        return FrozenMap(dict(available=False,agenda=agenda.model_dump(mode='json'),
            measurement_status='missing_canonical_model_or_goal_claim'))
    if profile['observation_support_transport']!='canonical_graph_source_terms':
        raise ValueError('unsupported source-declared axis evidence transport')
    observation_refs = tuple(sorted({o['term_ref'] for o in rows}))
    # The canonical graph retains its complete observation lineage. An action
    # cites that supported parent rather than flattening every ancestor claim.
    # Missing lineage or an unsupported observation blocks the whole model;
    # neither historical evidence nor contradictory observations are truncated.
    if (not set(observation_refs).issubset(graph.get('source_term_refs',()))
            or any(not o.get('premise_claim_refs') for o in rows)):
        return FrozenMap(dict(available=False,agenda=agenda.model_dump(mode='json'),
            measurement_status='incomplete_canonical_graph_observation_lineage'))
    model = compile_observed_axis_configuration(observations=rows,graph=graph,goal=goal,
        tracking=tracking,frame=frame,context_epoch=graph['context_epoch'])
    if profile['command_traversal_order']!='point_row_column_scope':
        raise ValueError('unsupported source-declared command traversal order')
    if model is not None:
        model['commands'] = tuple(sorted(model['commands'],key=lambda c:(c[3],c[2],c[0])))
    measured = measure_observed_axis_route(model,max_states=profile['maximum_states'],
        max_actions=profile['maximum_actions'])
    first = measured.get('first_command')
    known_points = {(o['observed_action_ref'],o['observed_action_data']['x'],o['observed_action_data']['y'])
                    for o in rows}
    known_palettes = {current[o['extent_entity_ref']].value for o in rows if o['extent_entity_ref'] in current}
    def component_at(x,y):
        found = tuple(c for _ref,c in sorted(current.items()) if (y,x) in c.pixels)
        return found[0] if len(found)==1 else None
    def signature(c):
        return tuple(sorted((r-c.bbox.top,col-c.bbox.left) for r,col in c.pixels))
    known_signatures = set()
    for _action,x,y in sorted(known_points):
        c = component_at(x,y)
        if c is not None:
            known_signatures.add(signature(c))
    facts = dict(agenda.alternative_facts)
    available = False
    for candidate in agenda.candidates:
        row = dict(facts.get(candidate.candidate_ref,{}))
        data = candidate.action_data
        x,y = data.get('x'),data.get('y')
        point = (candidate.action_ref,x,y)
        kind = ''
        is_first = bool(first and point==tuple(first[1:4]))
        # Command tuples store x,y; the traversal order is declared separately.
        if is_first:
            kind = 'complete_route'
        elif (not measured['route_found'] and type(x) is int and type(y) is int
                and point not in known_points
                and candidate.candidate_ref not in agenda.current_context_no_effect_candidate_refs
                and row.get('contour_point_opposite_partitioned_enclosure_count')==1
                and row.get('partitioned_enclosure_retained_equal_translation_occurrence_count',0)
                    >=profile['minimum_retained_cohort']):
            c = component_at(x,y)
            value = row.get('candidate_palette_value')
            remote_rectangles = tuple(other for _ref,other in sorted(current.items())
                if c is not None and other.value==value and other.component_id!=c.component_id
                and other.area==other.bbox.height*other.bbox.width
                and not other.bbox.top<=y<=other.bbox.bottom)
            if c is not None and (value in known_palettes or len(remote_rectangles)==1):
                kind = ('missing_receiver' if value not in known_palettes
                        and signature(c) in known_signatures else 'missing_operator')
        row.update(axis_frontier_available=bool(kind),axis_frontier_kind=kind,
            premise_claim_refs=premises,canonical_premise_claim_refs=premises,
            axis_configuration_supporting_observation_term_refs=observation_refs,
            point_row=y if type(y) is int else -1,point_col=x if type(x) is int else -1,
            axis_configuration_model_present=model is not None,
            axis_configuration_route_status=measured['route_status'],
            axis_configuration_route_primitive_cost=measured.get('primitive_cost'),
            axis_configuration_body_count=len(model['bodies']) if model else 0,
            axis_configuration_goal_term_ref=goal['term_ref'],
            axis_configuration_graph_term_ref=graph['term_ref'],
            axis_configuration_wanted_marker=goal['wanted_marker_top_left'])
        if is_first:
            row.update(axis_configuration_predicted_marker=measure_configuration_state(model,
                tuple(length+(first[5] if ref==first[4] else 0)
                      for ref,length in zip(model['refs'],model['initial_extents'])))[0])
        facts[candidate.candidate_ref] = FrozenMap(row)
        available = available or bool(kind)
    updated = agenda.model_copy(update={'alternative_facts':FrozenMap(facts)})
    return FrozenMap(dict(available=available,agenda=updated.model_dump(mode='json'),measurement_status=measured['route_status'],
        expanded_states=measured.get('expanded_states',0)))
