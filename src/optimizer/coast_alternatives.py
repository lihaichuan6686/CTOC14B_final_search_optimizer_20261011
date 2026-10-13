import numpy as np
from scipy.optimize import minimize_scalar
from ctoc.kernels.j2_refine import shoot,propagate
from ctoc.kernels.physics import target_state
STATES=None

def alternatives(d,target,old_slot,per_arc=False):
    """One continuously refined geometric minimum per different visit slot."""
    t=np.array(d['times']);r=np.array(d['positions']);slots={}
    for arc in range(len(t)-1):
        slot=sum(k>=0 for k in d['node_ids'][1:arc+1])
        if slot==old_slot:continue
        duration=t[arc+1]-t[arc]
        if duration<50:continue
        vi,_,_,_,err=shoot(r[arc],r[arc+1],duration,np.array(d['branches'][arc]))
        if err>1e-4:continue
        def distance(dt):
            s,_,_=propagate(r[arc],vi,float(dt))
            return float(np.linalg.norm(s[:3]-target_state(STATES,target,float(t[arc]+dt))[0]))
        grid=np.linspace(22,duration-22,max(4,int(duration/600)+1))
        values=np.array([distance(x) for x in grid]);best=int(np.argmin(values))
        lo=grid[max(0,best-1)];hi=grid[min(len(grid)-1,best+1)]
        fit=minimize_scalar(distance,bounds=(lo,hi),method='bounded',options=dict(xatol=.2,maxiter=30))
        dt=float(fit.x);miss=float(fit.fun)
        entry=dict(arc=arc,fraction=dt/duration,time_s=float(t[arc]+dt),distance_km=miss,new_slot=slot)
        key=arc if per_arc else slot
        if key not in slots or miss<slots[key]['distance_km']:slots[key]=entry
    return sorted(slots.values(),key=lambda x:x['distance_km'])
