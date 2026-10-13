"""Incumbent prefix injection and bounded J2 prefix refinement inside a beam."""
import campaign as C
from ctoc.kernels.adaptive_scp import trajectory,branch_near
from ctoc.kernels.external_search import regularize
from large_block_kernel import schedule_score
from beam_lookahead import next_cost

def build(base,left,p,free_launch):
    if 'prefix_seed' in p:d=C.copy.deepcopy(p['prefix_seed'])
    else:
        d=C.copy.deepcopy(base)
        if free_launch:d.update(initial=p['initial'],positions=[p['initial'][:3]],times=[0.],node_ids=[-1],branches=[])
        else:
            for key in ('positions','times','node_ids'):d[key]=d[key][:left+1]
            d['branches']=d['branches'][:left]
    d.pop('reduced_scp_history',None);d.pop('encounter_offsets_km',None)
    for s in p['segments']:
        if s['midcourse']:
            from dsm_kernel import elliptic_coast
            from ctoc.kernels.physics import MU
            r0=C.np.array(d['positions'][-1]);v0=C.np.array(s['coast_velocity']);t0=d['times'][-1];duration=s['mid_time_s']-t0
            axis_a=1/(2/C.np.linalg.norm(r0)-C.np.dot(v0,v0)/MU)
            period=2*C.np.pi*C.np.sqrt(axis_a**3/MU)
            pieces=max(1,int(C.np.ceil(duration/(.2*period))))
            previous=0.
            for j in range(1,pieces+1):
                elapsed=duration*j/pieces
                position=C.np.array(s['mid_position']) if j==pieces else elliptic_coast(r0,v0,elapsed)[0]
                expected=v0 if j==1 else elliptic_coast(r0,v0,previous)[1]
                branch,_,_=branch_near(C.np.array(d['positions'][-1]),position,elapsed-previous,expected)
                d['positions'].append(position.tolist());d['times'].append(t0+elapsed);d['node_ids'].append(-1);d['branches'].append(branch)
                previous=elapsed
        d['positions'].append(s['position']);d['times'].append(s['time_s']);d['node_ids'].append(s['target']);d['branches'].append(s['branch'])
    return d

def incumbent(base,visits,vf,pulses,count):
    node=len(base['times'])-1 if count==len(visits) else int(visits[count-1]);d=C.copy.deepcopy(base)
    for key in ('positions','times','node_ids'):d[key]=d[key][:node+1]
    d['branches']=d['branches'][:node];d.pop('reduced_scp_history',None);d.pop('encounter_offsets_km',None)
    cost=float(C.np.linalg.norm(pulses[:node],axis=1).sum());d['cost']=cost
    return dict(cost=cost,mask=(1<<count)-1,t=float(d['times'][-1]),r=C.np.array(d['positions'][-1]),v=vf[node-1].copy(),order=list(range(count)),times=[float(d['times'][i]) for i,k in enumerate(d['node_ids']) if k>=0],segments=[],initial=d['initial'],prefix_seed=d,guide=True)
