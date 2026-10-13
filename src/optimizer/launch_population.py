"""Circular LEO launch seeds with free phase and several nearby plane tilts."""
import numpy as np
from numba import njit
from ctoc.kernels.physics import lambert,arc_min_radius,RE,MU

@njit(cache=True)
def first_seeds(times,positions,counts,normal,end,branches,keep,time_penalty=.5):
    n=len(counts);out=np.full((n*keep,14),np.nan);cursor=0
    for k in range(n):
        best=np.full(keep,1e12);rows=np.zeros((keep,14))
        for a in range(counts[k]):
            tm=times[k,a]
            if tm<900. or tm>min(64800.,end*.65):continue
            target=positions[k,a];radial=target/np.linalg.norm(target)
            h=normal-np.dot(normal,radial)*radial;h/=np.linalg.norm(h)
            tangent=np.cross(h,radial)
            for tilt in (-.14,0.,.14):
                tau=np.cos(tilt)*tangent+np.sin(tilt)*h
                for j in range(48):
                    angle=2*np.pi*(j+.5)/48
                    r=(RE+605)*(np.cos(angle)*radial-np.sin(angle)*tau)
                    v=np.sqrt(MU/(RE+605))*(np.sin(angle)*radial+np.cos(angle)*tau)
                    for b in range(len(branches)):
                        va,vz=lambert(r,target,tm,branches[b,0],branches[b,1],branches[b,2])
                        if not np.isfinite(va[0]):continue
                        dv=np.linalg.norm(va-v);rank=dv+time_penalty*tm/end;w=np.argmax(best)
                        if rank>=best[w]:continue
                        if arc_min_radius(r,va,target,vz,tm)<RE+203:continue
                        duplicate=-1
                        for z in range(keep):
                            if best[z]<1e11 and abs(rows[z,2]-tm)<1200 and np.linalg.norm(rows[z,5:8]-vz)<.1:duplicate=z;break
                        if duplicate>=0:
                            if best[duplicate]<=rank:continue
                            w=duplicate
                        best[w]=rank;rows[w]=np.array([float(k),float(a),tm,float(b),dv,vz[0],vz[1],vz[2],r[0],r[1],r[2],v[0],v[1],v[2]])
        for j in range(keep):
            if best[j]<1e11:out[cursor]=rows[j];cursor+=1
    return out[:cursor]
