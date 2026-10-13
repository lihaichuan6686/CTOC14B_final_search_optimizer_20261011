"""Relax the whole carrier, reassign all 35 encounters, then restore actual constraints."""
import campaign as C
from scipy.optimize import linear_sum_assignment
def attempt(task):
    rotated=len(task)==3
    if rotated:
        number,angle,axis_name=task;radius=None;label=f'rotated_{axis_name}_{number:02}';extra=dict(angle_degrees=angle,rotation_axis=axis_name)
    else:number,radius=task;label=f'reassignment_{number:02}';extra={}
    t0=C.time.monotonic();trace=[]
    try:
        base=C.load(C.OUT/'rotation_parent.json');d=C.to_pole_j2(base,C.P)
        if rotated:
            from ctoc.kernels.external_search import regularize
            axis=C.np.array([0.,0.,1.]) if axis_name=='z' else C.np.array(d['initial'][:3])/C.np.linalg.norm(d['initial'][:3])
            ax,ay,az=axis;K=C.np.array([[0.,-az,ay],[az,0.,-ax],[-ay,ax,0.]])
            a=C.np.deg2rad(angle);Q=C.np.eye(3)*C.np.cos(a)+(1-C.np.cos(a))*C.np.outer(axis,axis)+C.np.sin(a)*K
            d['positions']=(C.np.array(d['positions'])@Q.T).tolist();d['initial']=(C.np.array(d['initial']).reshape(2,3)@Q.T).ravel().tolist()
            d['node_ids']=[-1]*len(d['node_ids']);d.pop('encounter_offsets_km',None);d=regularize(d)
        else:d=C.repair(d,C.A.STATES,list(range(35)),radius,.0001,seconds=40)
        C.save(label+'_relaxed_pole.json',d);relaxed=d['cost']
        d['node_ids']=[-1]*len(d['node_ids']);C.ensure_offsets(d,C.A.STATES)
        choices={};matrix=C.np.full((35,len(d['branches'])),1e12)
        for k in range(35):
            options=C.A.alternatives(d,k,-999,per_arc=True);choices[k]={q['arc']:q for q in options}
            for q in options:matrix[k,q['arc']]=q['distance_km']
        rows,cols=linear_sum_assignment(matrix)
        if len(rows)!=35 or matrix[rows,cols].max()>1e11:raise ValueError('No full geometric assignment')
        entries=sorted([(int(k),choices[int(k)][int(a)]) for k,a in zip(rows,cols)],key=lambda q:q[1]['time_s'])
        for k,q in entries:
            moment=q['time_s'];ts=C.np.array(d['times']);arc=int(C.np.searchsorted(ts,moment)-1)
            d=C.insert_coast(d,arc,(moment-ts[arc])/(ts[arc+1]-ts[arc]));d['node_ids'][arc+1]=k
        new_order=[k+1 for k in C.visits(d)];C.save(label+'_assignment.json',dict(entries=entries,order=new_order,relaxed_cost_NOT_ELIGIBLE=relaxed))
        for penalty in (.0001,.001,.01,.1):
            d=C.repair(d,C.A.STATES,list(range(35)),.995,penalty,seconds=30)
            trace.append(dict(penalty=penalty,cost=d['cost'],excess_km=d['elastic_excess_km']))
            C.save(label+'_repair_state_pole.json',d)
            if d['elastic_excess_km']<1e-5:break
        if d['elastic_excess_km']>.5:raise ValueError('Global reassignment incomplete: '+str(d['elastic_excess_km']))
        for i,k in enumerate(d['node_ids']):
            if k<0:continue
            center=C.target_state(C.A.STATES,k,d['times'][i])[0];offset=C.np.array(d['positions'][i])-center
            if C.np.linalg.norm(offset)>.995:d['positions'][i]=(center+offset*.995/C.np.linalg.norm(offset)).tolist()
        C.ensure_offsets(d,C.A.STATES);d=C.refine(d,C.A.STATES,55/60,max_iterations=850);d=C.complete(d,label)
        result=dict(label=label,cost=d['cost'],radius_km=radius,new_order=new_order,changed_order=new_order!=[k+1 for k in C.visits(base)],trace=trace,seconds=C.time.monotonic()-t0,**extra)
    except Exception as exc:
        C.traceback.print_exc();result=dict(label=label,radius_km=radius,error=repr(exc),trace=trace,seconds=C.time.monotonic()-t0,**extra)
    C.save(label+'_receipt.json',result);return result
