"""A suffix beam that compares direct transfers with explicit midcourse burns."""
import campaign as C
from large_block_kernel import expand,schedule_score
from dsm_kernel import dsm_expand
BR=C.np.array([[m,r,s] for r in range(3) for m in range(2) for s in range(1 if r==0 else 2)],dtype=C.np.int64)
from ctoc.kernels.adaptive_scp import trajectory,branch_near
from ctoc.kernels.external_search import regularize
from scipy.optimize import brentq

def screen(task):
    grid_step=900.;blended=False;warm_epochs=True;settings={}
    if isinstance(task,int):n,width,penalty,tag=task,320,.35,'dsm_tail'
    else:
        n,width,penalty,tag=task[:4]
        if len(task)>4:grid_step=task[4]
        if len(task)>5:blended=task[5]
        if len(task)>6:warm_epochs=task[6]
        if len(task)>7:settings=task[7]
    pressure_hours=float(settings.get('pressure_hours',0.));pressure_weight=float(settings.get('pressure_weight',0.));lookahead_weight=float(settings.get('lookahead_weight',0.))
    refinement=float(settings.get('continuous_refinement_s',0.))
    max_transfer=float(settings.get('max_transfer_s',0.))
    free_launch=bool(settings.get('free_launch',False))
    reserve_fraction=float(settings.get('refined_reserve_fraction',0.))
    protected_key=settings.get('protected_lineage_key','refined_lineage')
    free_midcourse=bool(settings.get('free_midcourse',False))
    label=f'{tag}_{n:02}';beg=C.time.monotonic()
    base=C.to_pole_j2(C.load(C.OUT/(settings.get('source_file','global_escape_04')+'.json')),C.P);vis=C.np.flatnonzero(C.np.array(base['node_ids'])>=0);left=0 if free_launch else int(vis[-n-1]);indices=vis[-n:]
    if free_launch:assert n==35
    ids=C.np.array(base['node_ids'])[indices];tt=C.np.array(base['times']);rr=C.np.array(base['positions']);vi,vf,_,base_pulses,_,_=trajectory(base)
    vin=vf[left-1];tl=tt[left];tr=864000.
    normals=C.np.cross(rr[left:-1],vi[left:]);normals/=C.np.linalg.norm(normals,axis=1)[:,None];normal=normals.mean(axis=0);normal/=C.np.linalg.norm(normal)
    angle=C.np.deg2rad(float(settings.get('plane_rotation_deg',0.)))
    if angle:
        axis=rr[0]/C.np.linalg.norm(rr[0]);normal=normal*C.np.cos(angle)+C.np.cross(axis,normal)*C.np.sin(angle)+axis*(axis@normal)*(1-C.np.cos(angle));normal/=C.np.linalg.norm(normal)
    grids=[]
    for j,k in enumerate(ids):
        grid=C.np.arange(tl+240,tr-180,grid_step);keep=[]
        for tm in grid:
            r=C.target_state(C.A.STATES,int(k),float(tm))[0]
            if abs(r@normal)/C.np.linalg.norm(r)<C.np.sin(C.np.deg2rad(15)):keep.append(float(tm))
        if warm_epochs:keep+=list(tt[indices[j]]+C.np.array([-1800.,-900.,0.,900.,1800.]))
        else:
            probe=C.np.r_[grid,tr-180.]
            residual=[float(C.target_state(C.A.STATES,int(k),float(tm))[0]@normal) for tm in probe]
            for a in range(len(probe)-1):
                if residual[a]*residual[a+1]>=0:continue
                center=brentq(lambda tm:float(C.target_state(C.A.STATES,int(k),float(tm))[0]@normal),float(probe[a]),float(probe[a+1]),xtol=.05)
                for delta in (-grid_step/2,0.,grid_step/2):
                    tm=center+delta
                    if not tl+180<tm<tr-180:continue
                    pos=C.target_state(C.A.STATES,int(k),float(tm))[0]
                    if abs(pos@normal)/C.np.linalg.norm(pos)<C.np.sin(C.np.deg2rad(15)):keep.append(float(tm))
        if settings.get('warm_launch_epochs',False) and tt[indices[j]]<=64800.:keep.append(float(tt[indices[j]]))
        grids.append(C.np.unique([t for t in keep if tl+180<t<tr-180]))
    counts=C.np.array([len(g) for g in grids],dtype=C.np.int64);times=C.np.zeros((n,int(max(counts))));positions=C.np.zeros((*times.shape,3))
    for k,g in enumerate(grids):
        times[k,:len(g)]=g
        for a,tm in enumerate(g):positions[k,a]=C.target_state(C.A.STATES,int(ids[k]),float(tm))[0]
    beam=[dict(cost=0.,mask=0,t=tl,r=rr[left],v=vin,order=[],times=[],segments=[])]
    if free_launch:
        from launch_population import first_seeds
        beam=[]
        direct_launch=first_seeds(times,positions,counts,normal,tr,BR,12,float(settings.get('launch_time_penalty',.5)))
        for q in direct_launch:
            k,a,b=int(q[0]),int(q[1]),int(q[3]);segment=dict(target=int(ids[k]),position=positions[k,a].tolist(),time_s=float(q[2]),branch=BR[b].tolist(),midcourse=False)
            beam.append(dict(cost=float(q[4]),mask=1<<k,t=float(q[2]),r=positions[k,a],v=q[5:8].copy(),order=[k],times=[float(q[2])],segments=[segment],initial=q[8:14].tolist()))
        pass  # Disabled historical experiment removed.
        print('GLOBAL_LAUNCH_SEEDS',label,len(beam),flush=True)
    guiding=bool(settings.get('incumbent_guidance',False))
    if guiding:
        assert free_launch and n==35
        from beam_prefix_joint import incumbent
        beam.append(incumbent(base,vis,vf,base_pulses,1))
    for depth in range(1 if free_launch else 0,n):
        candidates=[]
        threaded=int(C.os.environ.get('CTOC_BEAM_THREADS','1'))>1
        if threaded:
            args=(C.np.array([p['r'] for p in beam]),C.np.array([p['v'] for p in beam]),C.np.array([p['t'] for p in beam]),C.np.array([p['mask'] for p in beam],dtype=C.np.int64),depth,times,positions,counts,BR,tl,tr,penalty,blended,pressure_hours,pressure_weight)
            if refinement:
                from refined_beam_kernel import batch
                ad,am,dc,mc=batch(*args,C.A.STATES,ids,normal,refinement,max_transfer,free_midcourse)
            else:
                from parallel_beam_kernel import batch
                ad,am,dc,mc=batch(*args,max_transfer)
        for pi,p in enumerate(beam):
            if threaded:direct=ad[pi,:dc[pi]];staged=am[pi,:mc[pi]]
            else:
                direct=expand(p['r'],p['v'],p['t'],p['mask'],depth,times,positions,counts,BR,tl,tr,4,penalty,pressure_hours,pressure_weight,max_transfer)
                staged=dsm_expand(p['r'],p['v'],p['t'],p['mask'],depth,times,positions,counts,BR,tl,tr,direct,penalty,blended,pressure_hours,pressure_weight,max_transfer,free_midcourse)
                if refinement:
                    from refined_beam_kernel import polish
                    args=(p['r'],p['v'],p['t'],C.A.STATES,ids,BR,n-depth-1,tl,tr,penalty,pressure_hours,pressure_weight,normal,refinement,max_transfer)
                    direct=polish(direct,*args);staged=polish(staged,*args,free_midcourse)
            for q in list(direct)+list(staged):
                k,a,b=int(q[0]),int(q[1]),int(q[3]);cost=p['cost']+q[4]
                arrival=C.target_state(C.A.STATES,int(ids[k]),float(q[2]))[0] if refinement else positions[k,a]
                segment=dict(target=int(ids[k]),position=arrival.tolist(),time_s=float(q[2]),branch=BR[b].tolist(),midcourse=len(q)>8)
                if len(q)>8:segment.update(mid_time_s=float(q[8]),mid_position=q[9:12].tolist(),coast_velocity=q[12:15].tolist())
                child=dict(cost=float(cost),mask=p['mask']|(1<<k),t=float(q[2]),r=arrival,v=q[5:8].copy(),order=p['order']+[k],times=p['times']+[float(q[2])],last_branch=b,segments=p['segments']+[segment])
                if free_launch:child['initial']=p['initial']
                if 'prefix_seed' in p:child['prefix_seed']=p['prefix_seed']
                if p.get('refined_lineage'):child['refined_lineage']=True
                if p.get('multi_launch'):child['multi_launch']=True
                rank=cost+(0 if depth==n-1 else schedule_score(q[2],n-depth-1,tl,tr,penalty,pressure_hours,pressure_weight));candidates.append((float(rank),child))
        if lookahead_weight and depth<n-1:
            candidates.sort(key=lambda q:q[0]);short=[];buckets=set()
            for rank,p in candidates:
                key=(p['mask'],p['order'][-1],int(p['t']//1800),tuple(C.np.rint(p['v']/.1).astype(int)))
                if key in buckets:continue
                buckets.add(key);short.append((rank,p))
                if len(short)>=width*3:break
            pass  # Disabled historical experiment removed.
            if short:
                from beam_lookahead import next_cost
                pp=[q[1] for q in short]
                ahead=next_cost(C.np.array([p['r'] for p in pp]),C.np.array([p['v'] for p in pp]),C.np.array([p['t'] for p in pp]),C.np.array([p['mask'] for p in pp],dtype=C.np.int64),times,positions,counts,tr)
                candidates=[(rank+lookahead_weight*float(ahead[i]),p) for i,(rank,p) in enumerate(short)]
        candidates.sort(key=lambda q:q[0]);beam=[];seen=set();quota={}
        pass  # Disabled historical experiment removed.
        for rank,p in candidates:
            pass  # Disabled historical experiment removed.
            key=(p['mask'],p['order'][-1],int(p['t']//1800),tuple(C.np.rint(p['v']/.1).astype(int)));order=tuple(p['order']);cap=width if depth==0 else (24 if depth==1 else 6)
            if key in seen or quota.get(order,0)>=cap:continue
            seen.add(key);quota[order]=quota.get(order,0)+1;beam.append(p)
            if len(beam)>=width:break
        prefix_every=int(settings.get('prefix_refine_every',0))
        pass  # Disabled historical experiment removed.
        if guiding:
            from beam_prefix_joint import incumbent
            beam=beam[:width-1]+[incumbent(base,vis,vf,base_pulses,depth+1)]
        print('DSM_TAIL_BEAM',n,depth+1,len(beam),len(quota),flush=True)
        C.save(label+'_progress.json',dict(depth=depth+1,target_count=n,beam_count=len(beam),order_count=len(quota),refined_lineage_count=sum(bool(p.get('refined_lineage')) for p in beam),multi_launch_count=sum(bool(p.get('multi_launch')) for p in beam),seconds=C.time.monotonic()-beg,beam_threads=int(C.os.environ.get('CTOC_BEAM_THREADS','1'))))
        if not beam:raise ValueError('No full free-terminal suffix')
    selected=[];orders={};last_counts={}
    ranked=sorted(beam,key=lambda q:q['cost'])
    pass  # Disabled historical experiment removed.
    physical_seen=set()
    for p in ranked:
        order=tuple(p['order']);last=order[-1]
        if orders.get(order,0)>=2 or last_counts.get(last,0)>=int(settings.get('max_per_last_target',3)):continue
        if settings.get('require_midcourse',True) and not any(s['midcourse'] for s in p['segments']):continue
        from beam_prefix_joint import build
        try:d=build(base,left,p,free_launch)
        except ValueError:continue
        import hashlib,json
        signature=hashlib.sha256(json.dumps({k:d[k] for k in ['initial','positions','times','node_ids','branches']},sort_keys=True).encode()).hexdigest()
        if signature in physical_seen:continue
        physical_seen.add(signature)
        orders[order]=orders.get(order,0)+1;last_counts[last]=last_counts.get(last,0)+1;d.pop('encounter_offsets_km',None)
        assert sorted(C.visits(d))==list(range(35))
        name=f'{label}_c{len(selected)}';C.save(name+'_seed_pole.json',d);selected.append(dict(label=name,surrogate=float(p['cost']),order=(ids[list(order)]+1).tolist(),last_time_hours=float(p['t']/3600),midcourse_count=sum(s['midcourse'] for s in p['segments']),incumbent=bool(p.get('guide',False)),refined_lineage=bool(p.get('refined_lineage')),multi_launch=bool(p.get('multi_launch')),prefix_retained_nodes=len(p.get('prefix_seed',{}).get('times',[]))))
        if len(selected)>=int(settings.get('final_count',8)):break
    result=dict(label=label,choices=selected,beam_width=width,time_penalty=penalty,grid_step_s=grid_step,blended_initial_burns=blended,warm_start_epochs=warm_epochs,ranking_settings=settings,seconds=C.time.monotonic()-beg);C.save(label+'_screen.json',result);return result

def optimize(label):
    beg=C.time.monotonic()
    try:
        d=C.load(C.OUT/(label+'_seed_pole.json'));d=regularize(d);C.ensure_offsets(d,C.A.STATES);raw=d['cost'];d=C.refine(d,C.A.STATES,80/60,max_iterations=1200);d=C.complete(d,label)
        result=dict(label=label,cost=d['cost'],raw_j2_cost=raw,last_target=C.visits(d)[-1]+1,seconds=C.time.monotonic()-beg)
    except Exception as exc:
        C.traceback.print_exc();result=dict(label=label,error=repr(exc),seconds=C.time.monotonic()-beg)
    C.save(label+'_receipt.json',result);return result

