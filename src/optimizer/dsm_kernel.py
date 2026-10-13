"""Two-body multi-impulse screening; never used as a verified trajectory score."""
import numpy as np
from numba import njit
from ctoc.kernels.physics import MU,RE,lambert,arc_min_radius
from large_block_kernel import schedule_score

@njit(cache=True)
def elliptic_coast(r,v,dt):
    rn=np.linalg.norm(r);vv=np.dot(v,v);a=1/(2/rn-vv/MU)
    if a<=0:return np.full(3,np.nan),np.full(3,np.nan),-1.
    h=np.cross(r,v);hn=np.linalg.norm(h)
    if hn<1e-10:return np.full(3,np.nan),np.full(3,np.nan),-1.
    ev=((vv-MU/rn)*r-np.dot(r,v)*v)/MU;e=np.linalg.norm(ev)
    if e>=1:return np.full(3,np.nan),np.full(3,np.nan),-1.
    if e<1e-9:p=r/rn;E0=0.
    else:
        p=ev/e;E0=np.arctan2(np.dot(r,v)/(e*np.sqrt(MU*a)),(1-rn/a)/e)
    q=np.cross(h/hn,p);M=(E0-e*np.sin(E0)+np.sqrt(MU/a**3)*dt+np.pi)%(2*np.pi)-np.pi
    E=M
    for _ in range(35):
        delta=(E-e*np.sin(E)-M)/(1-e*np.cos(E));E-=delta
        if abs(delta)<1e-13:break
    root=np.sqrt(1-e*e);pos=a*((np.cos(E)-e)*p+root*np.sin(E)*q)
    vel=np.sqrt(MU*a)/np.linalg.norm(pos)*(-np.sin(E)*p+root*np.cos(E)*q)
    return pos,vel,a*(1-e)

@njit(cache=True)
def dsm_expand(r,v,t,mask,depth,times,positions,counts,branches,tleft,tright,direct,penalty,blended=False,pressure_hours=0.,pressure_weight=0.,max_transfer_s=0.,free_midcourse=False):
    n=len(counts);out=np.full((n*3,16),np.nan);row=0
    for k in range(n):
        if mask&(1<<k):continue
        options=np.zeros(counts[k],np.bool_)
        for q in direct:
            if int(q[0])==k:options[int(q[1])]=True
        # Also inspect two epochs found by unforced coasting, independent of
        # the direct Lambert ranking, to reduce its screening bias.
        close=np.full(2,1e12);indices=np.full(2,-1,np.int64)
        for a in range(counts[k]):
            tm=times[k,a]
            if tm-t<600 or tright-tm<(n-depth)*180:continue
            if max_transfer_s>0 and tm-t>max_transfer_s:continue
            pos,vel,rad=elliptic_coast(r,v,tm-t)
            if not np.isfinite(pos[0]):continue
            value=np.linalg.norm(pos-positions[k,a])/(tm-t)
            w=np.argmax(close)
            if value<close[w]:close[w]=value;indices[w]=a
        for a in indices:
            if a>=0:options[a]=True
        best=np.full(3,1e12);data=np.zeros((3,16))
        for a in range(counts[k]):
            if not options[a]:continue
            tm=times[k,a];duration=tm-t
            if duration<600 or tright-tm<(n-depth)*180:continue
            if max_transfer_s>0 and duration>max_transfer_s:continue
            radial=r/np.linalg.norm(r);tangent=v-np.dot(v,radial)*radial
            reference=np.full(3,np.nan);reference_cost=1e12
            if blended:
                for branch in branches:
                    aa,zz=lambert(r,positions[k,a],duration,branch[0],branch[1],branch[2])
                    if np.isfinite(aa[0]):
                        c=np.linalg.norm(aa-v)
                        if c<reference_cost:reference_cost=c;reference=aa
            for mode in range(5 if blended else 3):
                if mode<3:launch=v+(mode-1)*.03*tangent
                elif np.isfinite(reference[0]):launch=v+(.5 if mode==3 else 1.25)*(reference-v)
                else:continue
                burn=np.linalg.norm(launch-v)
                fractions=np.array([.1,.25,.35,.5,.65,.75,.9]) if free_midcourse else np.array([.35,.65])
                for fraction in fractions:
                    mid=t+fraction*duration;rm,vm,rp=elliptic_coast(r,launch,mid-t)
                    if not np.isfinite(rm[0]):continue
                    if rp<RE+203 and arc_min_radius(r,launch,rm,vm,mid-t)<RE+203:continue
                    for b in range(len(branches)):
                        aa,zz=lambert(rm,positions[k,a],tm-mid,branches[b,0],branches[b,1],branches[b,2])
                        if not np.isfinite(aa[0]):continue
                        cost=burn+np.linalg.norm(aa-vm);rank=cost+schedule_score(tm,n-depth-1,tleft,tright,penalty,pressure_hours,pressure_weight);w=np.argmax(best)
                        if rank>=best[w]:continue
                        if arc_min_radius(rm,aa,positions[k,a],zz,tm-mid)<RE+203:continue
                        duplicate=-1
                        for j in range(3):
                            if best[j]<1e11 and abs(data[j,2]-tm)<600 and int(data[j,3])==b:duplicate=j
                        if duplicate>=0:
                            if best[duplicate]<=rank:continue
                            w=duplicate
                        best[w]=rank;data[w]=np.array([float(k),float(a),tm,float(b),cost,zz[0],zz[1],zz[2],mid,rm[0],rm[1],rm[2],launch[0],launch[1],launch[2],fraction])
        for j in range(3):
            if best[j]<1e11:out[row]=data[j];row+=1
    return out[:row]
