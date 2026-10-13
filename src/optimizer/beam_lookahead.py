"""One-step ranking heuristic, never a constraint or a trajectory score."""
import numpy as np
from numba import njit,prange
from ctoc.kernels.physics import lambert,arc_min_radius,RE

@njit(cache=True,parallel=True)
def next_cost(r,v,t,masks,times,positions,counts,tright):
    costs=np.zeros(len(t))
    for i in prange(len(t)):
        if masks[i]==(1<<len(counts))-1:continue
        best=3.
        for k in range(len(counts)):
            if masks[i]&(1<<k):continue
            lo=np.searchsorted(times[k,:counts[k]],t[i]+300.)
            hi=np.searchsorted(times[k,:counts[k]],min(tright,t[i]+86400.),side='right')
            for a in range(lo,hi,2):
                dt=times[k,a]-t[i]
                for major in range(2):
                    va,vz=lambert(r[i],positions[k,a],dt,major,0,0)
                    if not np.isfinite(va[0]):continue
                    value=np.linalg.norm(va-v[i])
                    if value<best and arc_min_radius(r[i],va,positions[k,a],vz,dt)>=RE+203:best=value
        costs[i]=best
    return costs
