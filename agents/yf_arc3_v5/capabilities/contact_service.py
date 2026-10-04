"""Bounded geometric contact-service witnesses under a declared experiment model."""
from collections import deque


def measure_contact_services(value, bodies, assignments, contract, *, point_action_ref=None):
    empty={'contact_service_witness_present':False}
    if not contract or not (2 <= len(bodies) <= int(contract['max_bodies'])):
        return empty
    memories=tuple(m for m in value.canonical_contact_transport_facts
        if m.get('stationary_initiator_observed') and int(m.get('ordinary_quantum',0))>0
        and m.get('memory_claim_ref') and all(b.value==m.get('observed_value') for b in bodies))
    floors=frozenset((*value.episode_input_aligned_underlay_values,*value.current_input_aligned_underlay_values))
    point=getattr(value,'last_point_position',None)
    point_test=bool(contract.get('point_selection_requires_one_current_directional_test')
        and point_action_ref and getattr(value,'last_action_ref',None)==point_action_ref and point is not None)
    quanta={abs(dy)+abs(dx) for _,dy,dx in value.known_action_translation_deltas if bool(dy)!=bool(dx)}
    if not memories or not floors or not quanta:
        return empty
    q=min(quanta)
    distances={((abs(int(m.get('delta_row',0)))+abs(int(m.get('delta_col',0))))*q)//int(m['ordinary_quantum'])
        for m in memories if (abs(int(m.get('delta_row',0)))+abs(int(m.get('delta_col',0))))%int(m['ordinary_quantum'])==0}
    if len(distances)!=1:
        return empty
    distance,=distances
    transit=frozenset(v for m in memories for v in m.get('transit_values',()))|floors
    steps=tuple(sorted((a,dy*q,dx*q) for a,dy,dx in value.interface_action_translation_deltas
        if abs(dy)+abs(dx)==1))
    if not steps:
        return empty
    frame=value.current_observed_frame or value.frame
    bodies=tuple(sorted(bodies,key=lambda b:b.component_id))
    starts=tuple((b.bbox.top,b.bbox.left) for b in bodies)
    sizes=tuple((b.bbox.height,b.bbox.width) for b in bodies)
    goals=tuple((int(assignments[b.component_id]['assigned_zone_cell_middle_row'])-(b.bbox.height-1)//2,
                 int(assignments[b.component_id]['assigned_zone_cell_middle_col'])-(b.bbox.width-1)//2) for b in bodies)
    def pixels(pos,size):
        return {(r,c) for r in range(pos[0],pos[0]+size[0]) for c in range(pos[1],pos[1]+size[1])}
    occupied=tuple(pixels(pos,size) for pos,size in zip(starts,sizes))
    # Exact target interiors and their enclosing outlines are already assigned.
    goal_regions=set().union(*(pixels((p[0]-1,p[1]-1),(s[0]+2,s[1]+2)) for p,s in zip(goals,sizes)))
    def valid(index,pos,*,crossing=False,ignore=frozenset(),obstacles=occupied):
        h,w=sizes[index];r,c=pos
        if r<0 or c<0 or r+h>frame.height or c+w>frame.width: return False
        footprint=pixels(pos,sizes[index])
        if any(footprint & pts for j,pts in enumerate(obstacles) if j!=index and j not in ignore): return False
        allowed=transit if crossing else floors
        return all(p in occupied[index] or p in goal_regions or frame.rows[p[0]][p[1]] in allowed for p in footprint)
    expanded=0
    def reachable(index,start,obstacles=occupied):
        nonlocal expanded
        queue=deque((start,)); paths={start:()}
        while queue:
            pos=queue.popleft()
            expanded+=1
            if expanded>int(contract.get('max_total_state_expansions',131072)): return None
            for action,dy,dx in steps:
                dest=(pos[0]+dy,pos[1]+dx)
                if dest in paths or not valid(index,dest,obstacles=obstacles): continue
                if len(paths)>=int(contract['max_states_per_body']): return None
                if len(paths[pos])>=int(contract['max_route_length']): continue
                paths[dest]=(*paths[pos],action);queue.append(dest)
        return paths
    reach=tuple(reachable(i,p) for i,p in enumerate(starts))
    if any(row is None for row in reach): return empty
    inaccessible=tuple(i for i in range(len(bodies)) if goals[i] not in reach[i])
    if not inaccessible: return empty
    controlled=tuple(i for i,b in enumerate(bodies) if b.bbox in value.input_aligned_entity_bboxes)
    if contract.get('use_unique_canonical_control_geometry'):
        canonical_boxes=getattr(value,'canonical_provisional_control_geometry_bboxes',())
        canonical=tuple(i for i,b in enumerate(bodies) if b.bbox in canonical_boxes)
        if len(canonical)==1:
            controlled=canonical
    reconstructed=tuple(i for i,b in enumerate(bodies) if getattr(b,'source_control_entity_ref',None))
    if len(reconstructed)==1:
        controlled=reconstructed
    control_test=False
    if point_test:
        pointed=tuple(i for i,b in enumerate(bodies)
            if b.bbox.top<=point[0]<=b.bbox.bottom and b.bbox.left<=point[1]<=b.bbox.right)
        if len(pointed)==1:
            controlled=pointed
            control_test=True
    if len(controlled)!=1: return empty
    witnesses=[]
    structural_count=0
    witness_bound=int(contract['max_witnesses'])
    for charge in inaccessible:
        ch,cw=sizes[charge]
        staged=sorted(reach[charge].items(),key=lambda item:(len(item[1]),item[1],item[0])) if contract.get('allow_reachable_charge_preparation') else [(starts[charge],())]
        actor_reach_cache={}
        for actor in range(len(bodies)):
            if actor==charge: continue
            for action,dy,dx in steps:
                structural_count+=1
                if structural_count>int(contract.get('max_structural_descriptions',128)): return empty
                ah,aw=sizes[actor]
                uy,ux=dy//q,dx//q
                first_witness=None
                for (cr,cc),preparation in staged:
                    # Lower bound includes the impulse and both control changes.
                    lower=len(preparation)+1+(charge!=controlled[0])+1 if preparation else 1+(actor!=controlled[0])
                    if first_witness is not None and lower>first_witness[0]: continue
                    landing=(cr+uy*distance,cc+ux*distance)
                    ignore=frozenset((actor,))
                    if not all(valid(charge,(cr+uy*n,cc+ux*n),crossing=True,ignore=ignore)
                               for n in range(q,distance+q,q)): continue
                    for _ in range(int(contract['max_continuation_steps'])):
                        if valid(charge,landing,ignore=ignore): break
                        landing=(landing[0]+dy,landing[1]+dx)
                        if not valid(charge,landing,crossing=True,ignore=ignore): break
                    if not valid(charge,landing,ignore=ignore) or landing in reach[charge]: continue
                    if sum(abs(a-b) for a,b in zip(landing,goals[charge]))>=sum(abs(a-b) for a,b in zip(starts[charge],goals[charge])): continue
                    key=(actor,cr,cc)
                    if key not in actor_reach_cache:
                        obstacles=tuple(pixels((cr,cc),sizes[j]) if j==charge else pts for j,pts in enumerate(occupied))
                        actor_reach_cache[key]=reach[actor] if not preparation else reachable(actor,starts[actor],obstacles)
                    available=actor_reach_cache[key]
                    if available is None: return {'contact_service_witness_present':False,'contact_service_bound_reached':True}
                    contacts=[]
                    for pos,path in sorted(available.items()):
                        ar,ac=pos
                        touches=((ux>0 and ac+aw==cc or ux<0 and cc+cw==ac) and max(ar,cr)<min(ar+ah,cr+ch)
                            or (uy>0 and ar+ah==cr or uy<0 and cr+ch==ar) and max(ac,cc)<min(ac+aw,cc+cw))
                        if touches: contacts.append((len(path),path,pos))
                    if not contacts: continue
                    _,path,pos=min(contacts)
                    route=(*path,action)
                    first=charge if preparation else actor
                    cost=(len(preparation)+(charge!=controlled[0])+1+len(route)) if preparation else len(route)+(actor!=controlled[0])
                    witness=(cost,first,charge,preparation or route,landing,pos,actor,preparation,route)
                    if first_witness is None or witness<first_witness: first_witness=witness
                if first_witness is not None:
                    witnesses.append(first_witness)
                    witnesses.sort()
                    del witnesses[witness_bound:]
    if not witnesses: return empty
    witnesses.sort()
    # Declared shortest primitive cost; retain at most three structural witnesses.
    witnesses=tuple(w for w in witnesses if w[0]==witnesses[0][0])
    directional=tuple(sorted({w[3][0] for w in witnesses if w[1]==controlled[0]}))
    points=tuple(sorted({bodies[w[1]].component_id for w in witnesses if w[1]!=controlled[0]}))
    return {'contact_service_witness_present':True,
        'contact_service_requires_current_control_test':control_test,
        'contact_service_directional_action_refs':directional,
        'contact_service_point_component_refs':points,
        'contact_service_point_support_bboxes':tuple(sorted({
            (bodies[w[1]].bbox.top,bodies[w[1]].bbox.left,bodies[w[1]].bbox.bottom,bodies[w[1]].bbox.right)
            for w in witnesses if w[1]!=controlled[0]})),
        'contact_service_premise_claim_refs':tuple(sorted({str(m['memory_claim_ref']) for m in memories})),
        'contact_service_witness_rows':tuple((bodies[a].component_id,bodies[c].component_id,route,landing,pos,
            bodies[actor].component_id,preparation,impulse_route) for _,a,c,route,landing,pos,actor,preparation,impulse_route in witnesses),
        'contact_service_state_expansions':expanded,
        'contact_service_minimum_primitive_cost':witnesses[0][0],
        'contact_service_inaccessible_body_count':len(inaccessible)}
