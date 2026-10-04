"""Finite geometric witnesses under explicitly supplied provisional models.

DRM owns model admission and epistemic meaning. This module measures one
shortest witness, counterfactual reachability and current candidate equality;
it neither assigns a role nor grants permission to execute the witness.
"""
from collections import deque
from agents.yf_arc3_v5.capabilities.appearance import measure_orthogonal_bands
from agents.yf_arc3_v5.logos.types import FrozenMap


def measure_support_navigation_domain(*, frame, components, memory, associations, contract, available_actions,
        current_input_aligned_underlay_values=(), input_aligned_entity_bboxes=(), current_translation_underlay_rows=()):
    empty = FrozenMap({'complete': False, 'witness_present': False})
    height,width=len(frame),len(frame[0])
    if height*width>contract['maximum_pixels'] or len(components)>contract['maximum_components']:
        raise ValueError('support dependency observation bound exceeded')
    if any(len(row)!=width for row in frame):raise ValueError('ragged observation')
    byref={r['term_ref']:r for r in associations}
    moving=memory.get('moving_association_refs',());fixed=memory.get('fixed_association_refs',())
    if len(moving)!=1 or len(fixed)!=1 or any(ref not in byref for ref in (*moving,*fixed,*memory['control_association_refs'])):return empty
    def sprite(c):return tuple(sorted((r-c.bbox.top,s-c.bbox.left,c.value) for r,s in c.pixels))
    def shape(c):return tuple((r,s) for r,s,v in sprite(c))
    def matches(row):return tuple(c for c in components if sprite(c)==tuple(map(tuple,row['normalized_valued_pixels'])))
    actors,targets=matches(byref[moving[0]]),matches(byref[fixed[0]])
    if len(actors)!=1 or len(targets)!=1:return empty
    actor,target=actors[0],targets[0];h,w=actor.bbox.height,actor.bbox.width
    calibrations=memory.get('observed_lattice_calibration_rows',())
    if len(calibrations)!=1:return empty
    cal=calibrations[0]
    if (cal['row_pitch'],cal['col_pitch'],cal['row_phase'],cal['col_phase'])!=(h,w,actor.bbox.top%h,actor.bbox.left%w):return empty
    ro,co=cal['row_phase'],cal['col_phase']
    if (target.bbox.top-ro)%h or (target.bbox.left-co)%w:return empty
    learned_controls=tuple(byref[ref] for ref in memory['control_association_refs'])
    shapes={tuple((r,c) for r,c,v in row['normalized_valued_pixels']) for row in learned_controls}
    controls=tuple(c for c in components if shape(c) in shapes)
    excluded={p for c in controls for p in c.pixels}
    points_by_cell={(r,c):tuple((ro+r*h+dr,co+c*w+dc) for dr in range(h) for dc in range(w))
        for r in range((height-ro)//h) for c in range((width-co)//w)}
    patterns={p:tuple(frame[r][c] for r,c in points) for p,points in sorted(points_by_cell.items())}
    entered={tuple(row['before_pattern']) for row in memory.get('observed_entered_surface_rows',())}
    models=contract['surface_models'];actor_pixels=set(actor.pixels)
    support_components=components
    # A current measured destination underlay can recover a completely hidden
    # tip. Re-extract its layer so rotations/swaps carry this support with them;
    # adding the actor cell to static floor would incorrectly survive a turn.
    underlay=tuple(current_input_aligned_underlay_values)
    underlay_bound=tuple(input_aligned_entity_bboxes)==(actor.bbox,)
    if 'current_exact_translation_actor_underlay' in models:
        actor_box=(actor.bbox.top,actor.bbox.left,actor.bbox.bottom,actor.bbox.right)
        exact=tuple(row for row in current_translation_underlay_rows
            if row['component_ref']==actor.component_id and tuple(row['bbox'])==actor_box)
        if len(exact)==1:
            underlay=tuple(exact[0]['underlay_values']);underlay_bound=True
    if ('current_measured_homogeneous_actor_underlay' in models and len(underlay)==1
            and underlay_bound
            and underlay[0]!=actor.value and (underlay[0],)*(h*w) in entered):
        from agents.yf_arc3_v5.capabilities import ComponentExtractionInput, FrameGrid, detect_components
        restored=[list(row) for row in frame]
        for r,c in actor.pixels:restored[r][c]=underlay[0]
        frame=restored
        support_components=detect_components(ComponentExtractionInput(
            frame=FrameGrid(rows=restored),connectivity=4)).components
        if len(support_components)>contract['maximum_components']:return empty
        patterns={p:tuple(frame[r][c] for r,c in points) for p,points in sorted(points_by_cell.items())}
    rectangles=[]
    for c in support_components:
        if set(c.pixels)==actor_pixels:continue
        if c.bbox.height%h or c.bbox.width%w or (c.bbox.top-ro)%h or (c.bbox.left-co)%w or set(c.pixels)&excluded:continue
        box={(r,s) for r in range(c.bbox.top,c.bbox.bottom+1) for s in range(c.bbox.left,c.bbox.right+1)}
        full=c.area==len(box) and 'complete_rectangular_cell_union' in models
        occluded=bool(box-set(c.pixels)) and box-set(c.pixels)<=actor_pixels and 'actor_occluded_rectangle' in models
        if full or occluded:rectangles.append((c,box))
    # Restore only a uniquely witnessed rectangular layer beneath the actor.
    # The actor sprite must never become a static supporting layer of its own.
    for p,points in sorted(points_by_cell.items()):
        if not set(points)&actor_pixels:continue
        values={c.value for c,box in rectangles if set(points)<=box}
        if len(values)==1:patterns[p]=tuple([next(iter(values))]*(h*w))
    rectangle_pixels={p for _,box in rectangles for p in box}
    surface={p for p,pat in sorted(patterns.items()) if ('observed_entered_pattern' in models and pat in entered)
        or all(point in rectangle_pixels for point in points_by_cell[p])}
    start=((actor.bbox.top-ro)//h,(actor.bbox.left-co)//w)
    terminal=((target.bbox.top-ro)//h,(target.bbox.left-co)//w)
    surface.add(terminal)
    proxies=[]
    if 'missing_control_smaller_same_palette_band_homologue_contact' in contract['permission_models']:
        for row in learned_controls:
            if matches(row):continue
            pixels=tuple((r,c) for r,c,v in row['normalized_valued_pixels'])
            signature=measure_orthogonal_bands(pixels)['oriented_signature']
            values={v for r,c,v in row['normalized_valued_pixels']}
            if len(values)!=1:continue
            for c in components:
                if c.value in values and len(c.pixels)<len(pixels) and measure_orthogonal_bands(c.pixels)['oriented_signature']==signature:
                    proxies.append((row['term_ref'],c))
    proxy_cells={}
    for ref,c in proxies:
        cells=set();item=set(c.pixels)
        for p,points in sorted(points_by_cell.items()):
            overlap=item&set(points)
            if not overlap:continue
            remaining={frame[r][s] for r,s in set(points)-item}
            if len(remaining)==1 and tuple([next(iter(remaining))]*(h*w)) in entered and 'entered_pattern_occluded_by_homologous_item' in models:
                surface.add(p);cells.add(p)
        proxy_cells.setdefault(ref,set()).update(cells)
    operators=[];effect_models=contract['effect_models']
    for row in learned_controls:
        effects=[e['typed_transition_measurements'] for e in memory.get('observed_remote_effect_rows',())
            if e['control_association_ref']==row['term_ref'] and 'typed_transition_measurements' in e]
        fits={tuple((k,f[k]) for k in ('changed_value','pivot_value','pivot_height','pivot_width','quarter_turns_clockwise')):f
            for e in effects if e['rigid_fit_enumeration_complete'] for f in e['rigid_fit_rows']}
        present=matches(row)
        if len(present)>1:return empty
        for fit_key in sorted(fits):
            f=fits[fit_key]
            if 'observed_quarter_turn_fit' not in effect_models:continue
            for pivot,box in rectangles:
                if (pivot.value,pivot.bbox.height,pivot.bbox.width)!=(f['pivot_value'],f['pivot_height'],f['pivot_width']):continue
                arms=[(c,b) for c,b in rectangles if c.value==f['changed_value']]
                pixels={p for c,b in arms for p in b}
                if pixels:operators.append({'kind':'rotation','pivot':pivot,'pixels':pixels,'step':f['quarter_turns_clockwise'],
                    'control':present[0] if present else None,'required_ref':None if present else row['term_ref'],
                    'period':4,'model_ref':'observed_quarter_turn_fit','unobserved':False})
        swaps={tuple(t['before_pattern']):tuple(t['after_pattern']) for e in effects for t in e['cell_transition_rows']
            if (tuple(t['before_pattern']) in entered or tuple(t['after_pattern']) in entered)
            and len(set(t['before_pattern']))!=len(set(t['after_pattern']))}
        positions={p for p,pat in sorted(patterns.items()) if pat in swaps and not any(point in excluded for point in points_by_cell[p])}
        if swaps and positions and len(present)==1 and 'observed_cell_pattern_exchange' in effect_models:
            if any(swaps.get(swaps[p])!=p for p in swaps):continue
            operators.append({'kind':'pattern_swap','positions':positions,'swaps':swaps,'control':present[0],
                'required_ref':None,'period':2,'model_ref':'observed_cell_pattern_exchange','unobserved':False})
    known={tuple(map(tuple,r['normalized_valued_pixels'])) for r in learned_controls}
    if 'control_shape_and_pivot_palette_assembly_quarter_turn' in effect_models:
        for control in controls:
            if sprite(control) in known:continue
            for pivot,box in rectangles:
                if pivot.value!=control.value:continue
                arms=[]
                for c,b in rectangles:
                    horizontal=c.bbox.top==pivot.bbox.top and c.bbox.height==pivot.bbox.height and (c.bbox.right+1==pivot.bbox.left or pivot.bbox.right+1==c.bbox.left)
                    vertical=c.bbox.left==pivot.bbox.left and c.bbox.width==pivot.bbox.width and (c.bbox.bottom+1==pivot.bbox.top or pivot.bbox.bottom+1==c.bbox.top)
                    if horizontal or vertical:arms.append((c,b))
                if len(arms)==2 and arms[0][0].value==arms[1][0].value:
                    operators.append({'kind':'rotation','pivot':pivot,'pixels':{p for c,b in arms for p in b},'step':1,
                        'control':control,'required_ref':None,'period':4,
                        'model_ref':'control_shape_and_pivot_palette_assembly_quarter_turn','unobserved':True})
    if not operators or len(operators)>contract['maximum_operators']:return empty
    required_refs=tuple(sorted(proxy_cells));cache={}
    def supports(config):
        if config in cache:return cache[config]
        cells=set(surface)
        for op in operators:
            cells-=op['positions'] if op['kind']=='pattern_swap' else {((r-ro)//h,(c-co)//w) for r,c in op['pixels']}
        for op,q in zip(operators,config):
            if op['kind']=='pattern_swap':
                cells|={p for p in op['positions'] if (patterns[p] if q==0 else op['swaps'][patterns[p]]) in entered}
            else:
                pivot=op['pivot'];pr=pivot.bbox.top+pivot.bbox.bottom;pc=pivot.bbox.left+pivot.bbox.right;pixels=set()
                for r,c in op['pixels']:
                    rr,cc=2*r-pr,2*c-pc
                    for _ in range(q):rr,cc=cc,-rr
                    if (rr+pr)%2 or (cc+pc)%2:continue
                    pixels.add(((rr+pr)//2,(cc+pc)//2))
                cells|={p for p,points in sorted(points_by_cell.items()) if all(point in pixels for point in points)}
        cache[config]=frozenset(cells);return cache[config]
    initial=(start,tuple(0 for _ in operators),0)
    if start not in supports(initial[1]):return empty
    def search(disabled=None,keep_path=False):
        pending=deque([initial]);parents={initial:None};end=None
        while pending:
            state=pending.popleft();pos,config,acquired=state
            if pos==terminal:end=state;break
            successors=[]
            for action,delta in sorted(contract['interface_steps'].items()):
                if action not in available_actions:continue
                p=(pos[0]+delta[0],pos[1]+delta[1])
                if p in supports(config):
                    flags=acquired
                    for bit,ref in enumerate(required_refs):
                        if p in proxy_cells[ref]:flags|=1<<bit
                    successors.append(((p,config,flags),('step',action,p)))
            for i,op in enumerate(operators):
                if disabled==i or contract['point_action_ref'] not in available_actions:continue
                ref=op['required_ref']
                if ref and (ref not in required_refs or not acquired&(1<<required_refs.index(ref))):continue
                following=list(config);following[i]=(following[i]+op.get('step',1))%op['period'];following=tuple(following)
                if pos in supports(config) and pos in supports(following):successors.append(((pos,following,acquired),('transform',i,pos)))
            for successor,action in successors:
                if successor in parents:continue
                if len(parents)>=contract['maximum_states']:return False,False,(),len(parents)
                parents[successor]=(state,action) if keep_path else True;pending.append(successor)
        path=[]
        if keep_path:
            while end and parents[end]:prior,action=parents[end];path.append(action);end=prior
            path.reverse()
        return True,end is not None,tuple(path),len(parents)
    complete,found,path,states=search(keep_path=True)
    if not complete or not found or not path:return FrozenMap({'complete':complete,'witness_present':False,'tested_states':states})
    counterfactual=[]
    for i in range(len(operators)):
        done,reached,_,count=search(disabled=i)
        counterfactual.append(FrozenMap({'operation_index':i,'complete':done,'terminal_reachable_without_operation':reached,'tested_states':count}))
    first=path[0];point=None;action_ref=first[1]
    if first[0]=='transform':
        c=operators[first[1]]['control']
        if c is None:return empty
        point=(c.bbox.left+c.bbox.width//2,c.bbox.top+c.bbox.height//2);action_ref=contract['point_action_ref']
    witness_required_refs=tuple(sorted({operators[row[1]]['required_ref'] for row in path
        if row[0]=='transform' and operators[row[1]]['required_ref']}))
    return FrozenMap({'complete':complete,'witness_present':True,'tested_states':states,
        'configuration_count':len(cache),'path_length':len(path),'path_rows':path,
        'first_action_ref':action_ref,'first_point':point,'first_operation_index':first[1] if first[0]=='transform' else None,
        'first_destination_cell':first[2],'actor_start_cell':start,'terminal_cell':terminal,
        'counterfactual_rows':tuple(counterfactual),'proxy_count':len(proxies),
        'witness_required_missing_control_refs':witness_required_refs,
        'operation_rows':tuple(FrozenMap({'model_ref':o['model_ref'],'unobserved':o['unobserved'],
            'required_association_ref':o['required_ref'],'currently_bound':o['control'] is not None}) for o in operators)})


def annotate_support_navigation_domain(value, agenda):
    memories=value.canonical_navigation_method_facts
    terminals=value.canonical_terminal_inventory_facts
    if len(memories)!=1 or len(terminals)!=1 or not value.navigation_dependency_contract:return agenda
    memory,terminal=memories[0],terminals[0]
    if not memory.get('memory_claim_ref') or not memory.get('observed_entered_surface_rows'):return agenda
    from agents.yf_arc3_v5.capabilities import ComponentExtractionInput, detect_components
    frame=value.current_observed_frame or value.frame
    components=detect_components(ComponentExtractionInput(frame=frame,connectivity=4)).components
    measured=measure_support_navigation_domain(frame=frame.rows,components=components,memory=memory,
        associations=value.canonical_navigation_association_facts,contract=value.navigation_dependency_contract,
        available_actions=value.available_action_refs,
        current_input_aligned_underlay_values=value.current_input_aligned_underlay_values,
        input_aligned_entity_bboxes=value.input_aligned_entity_bboxes,
        current_translation_underlay_rows=value.current_translation_underlay_rows)
    refs=tuple(sorted({memory['memory_claim_ref'],terminal['consultation_claim_ref']}))
    deltas={}
    for candidate in agenda.candidates:
        matches=bool(measured.get('witness_present') and candidate.action_ref==measured['first_action_ref'])
        if matches:
            point=measured['first_point']
            matches=(candidate.point is None) if point is None else (
                candidate.point==(point[1],point[0]) and (not candidate.action_data or (
                    candidate.action_data.get('x')==point[0] and candidate.action_data.get('y')==point[1])))
        delta={'navigation_dependency_witness_candidate':matches}
        if matches:
            delta.update({'navigation_dependency_measurement':measured,
                'navigation_dependency_complete':measured['complete'],
                'navigation_dependency_goal_ref':terminal['term_ref'],
                'navigation_dependency_premise_refs':refs,
                'navigation_dependency_historical_goal_proof':terminal.get('historical_proof_present',False),
                'navigation_dependency_contexts_resolved':terminal.get('current_contexts_resolved',False),
                'navigation_dependency_counterexamples':terminal.get('historical_counterexample_count',-1),
                'navigation_dependency_proxy_count':len(measured['witness_required_missing_control_refs']),
                'navigation_dependency_first_is_transform':measured['first_operation_index'] is not None})
        deltas[candidate.candidate_ref]=FrozenMap(delta)
    return agenda.model_copy(update={
        'alternative_facts':FrozenMap({ref:FrozenMap.overlay(deltas.get(ref,FrozenMap()),facts) for ref,facts in sorted(agenda.alternative_facts.items())}),
        'descriptive_delta_facts':FrozenMap({ref:FrozenMap.overlay(deltas.get(ref,FrozenMap()),facts) for ref,facts in sorted(agenda.descriptive_delta_facts.items())}),
        'context_facts':FrozenMap.overlay(FrozenMap({'navigation_dependency_domain_complete':measured.get('complete',False),
            'navigation_dependency_path_length':measured.get('path_length',0)}),agenda.context_facts)})
