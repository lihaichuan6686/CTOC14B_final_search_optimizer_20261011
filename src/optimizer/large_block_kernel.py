import numpy as np
from numba import njit
from ctoc.kernels.physics import lambert,arc_min_radius,RE

@njit(cache=True)
def schedule_score(tm,remaining,tleft,tright,penalty,pressure_hours=0.,pressure_weight=0.):
    pressure=remaining*pressure_hours*3600/max(tright-tm,180.)
    return penalty*(tm-tleft)/(tright-tleft)+pressure_weight*max(0.,pressure-.7)**2

@njit(cache=True)
def expand(r,v,t,mask,depth,times,positions,counts,branches,tleft,tright,nkeep,penalty,pressure_hours=0.,pressure_weight=0.,max_transfer_s=0.):
    n=len(counts);out=np.full((n*nkeep,8),np.nan);row=0
    for k in range(n):
        if mask & (1<<k):continue
        best=np.full(nkeep,1e12);data=np.zeros((nkeep,8))
        for a in range(counts[k]):
            tm=times[k,a]
            if tm-t<180 or tright-tm<(n-depth)*180:continue
            if max_transfer_s>0 and tm-t>max_transfer_s:continue
            for b in range(len(branches)):
                va,vz=lambert(r,positions[k,a],tm-t,branches[b,0],branches[b,1],branches[b,2])
                if not np.isfinite(va[0]):continue
                dv=np.linalg.norm(va-v);rank=dv+schedule_score(tm,n-depth-1,tleft,tright,penalty,pressure_hours,pressure_weight)
                worst=np.argmax(best)
                if rank>=best[worst]:continue
                if arc_min_radius(r,va,positions[k,a],vz,tm-t)<RE+203:continue
                # Different arrival epochs/branches can have different futures.
                duplicate=-1
                for q in range(nkeep):
                    if best[q]<1e11 and abs(data[q,2]-tm)<600 and int(data[q,3])==b:duplicate=q
                if duplicate>=0:
                    if best[duplicate]<=rank:continue
                    worst=duplicate
                best[worst]=rank;data[worst]=np.array([float(k),float(a),tm,float(b),dv,vz[0],vz[1],vz[2]])
        for q in range(nkeep):
            if best[q]<1e11:out[row]=data[q];row+=1
    return out[:row]

@njit(cache=True)
def close(r,v,t,rend,vout,tend,branches):
    best=1e12;which=-1
    for b in range(len(branches)):
        a,z=lambert(r,rend,tend-t,branches[b,0],branches[b,1],branches[b,2])
        if not np.isfinite(a[0]):continue
        cost=np.linalg.norm(a-v)+np.linalg.norm(vout-z)
        if cost<best and arc_min_radius(r,a,rend,z,tend-t)>=RE+203:best=cost;which=b
    return best,which
