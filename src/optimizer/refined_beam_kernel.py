"""Continuous arrival-time polishing inside direct/DSM beam expansion."""
import numpy as np
from numba import njit,prange
from ctoc.kernels.physics import lambert,target_state,arc_min_radius,RE
from large_block_kernel import expand,schedule_score
from dsm_kernel import dsm_expand,elliptic_coast

@njit(cache=True)
def trial(tm,row,r,v,t,states,ids,branches,nremaining,tleft,tright,penalty,hours,weight,normal):
    k=int(row[0]);branch=branches[int(row[3])];pos=target_state(states,ids[k],tm)[0]
    result=row.copy();result[2]=tm
    if abs(np.dot(pos,normal))/np.linalg.norm(pos)>np.sin(np.pi/12):return 1e12,result
    if len(row)==8:
        va,vz=lambert(r,pos,tm-t,branch[0],branch[1],branch[2])
        if not np.isfinite(va[0]) or arc_min_radius(r,va,pos,vz,tm-t)<RE+203:return 1e12,result
        dv=np.linalg.norm(va-v)
    else:
        middle=t+row[15]*(tm-t);rm,vm,rad=elliptic_coast(r,row[12:15],middle-t)
        if not np.isfinite(rm[0]):return 1e12,result
        if rad<RE+203 and arc_min_radius(r,row[12:15],rm,vm,middle-t)<RE+203:return 1e12,result
        va,vz=lambert(rm,pos,tm-middle,branch[0],branch[1],branch[2])
        if not np.isfinite(va[0]) or arc_min_radius(rm,va,pos,vz,tm-middle)<RE+203:return 1e12,result
        dv=np.linalg.norm(row[12:15]-v)+np.linalg.norm(va-vm);result[8]=middle;result[9:12]=rm
    result[4]=dv;result[5:8]=vz
    return dv+schedule_score(tm,nremaining,tleft,tright,penalty,hours,weight),result

@njit(cache=True)
def fraction_polish(row,r,v,t,states,ids,branches,nremaining,tleft,tright,penalty,hours,weight,normal):
    duration=row[2]-t;lower=max(.01,30./duration);upper=min(.99,1.-30./duration)
    grid=np.linspace(lower,upper,11);values=np.full(11,1e12)
    best=row[4]+schedule_score(row[2],nremaining,tleft,tright,penalty,hours,weight);chosen=row.copy()
    for j in range(11):
        probe=row.copy();probe[15]=grid[j]
        value,q=trial(row[2],probe,r,v,t,states,ids,branches,nremaining,tleft,tright,penalty,hours,weight,normal)
        values[j]=value
        if value<best:best=value;chosen=q.copy()
    j=int(np.argmin(values));lo=grid[max(0,j-1)];hi=grid[min(10,j+1)]
    x=hi-.61803398875*(hi-lo);y=lo+.61803398875*(hi-lo)
    for _ in range(14):
        probe=row.copy();probe[15]=x
        cx,qx=trial(row[2],probe,r,v,t,states,ids,branches,nremaining,tleft,tright,penalty,hours,weight,normal)
        probe[15]=y
        cy,qy=trial(row[2],probe,r,v,t,states,ids,branches,nremaining,tleft,tright,penalty,hours,weight,normal)
        if cx<best:best=cx;chosen=qx.copy()
        if cy<best:best=cy;chosen=qy.copy()
        if cx<cy:hi=y
        else:lo=x
        x=hi-.61803398875*(hi-lo);y=lo+.61803398875*(hi-lo)
    return chosen

@njit(cache=True)
def polish(rows,r,v,t,states,ids,branches,nremaining,tleft,tright,penalty,hours,weight,normal,step,max_transfer_s=0.,free_midcourse=False):
    out=rows.copy()
    for i in range(len(rows)):
        row=rows[i];best=row[4]+schedule_score(row[2],nremaining,tleft,tright,penalty,hours,weight);chosen=row.copy()
        lo=max(t+(180 if len(row)==8 else 600),row[2]-step/2);hi=min(tright-(nremaining+1)*180,row[2]+step/2)
        if max_transfer_s>0:hi=min(hi,t+max_transfer_s)
        if hi-lo<1:continue
        x=hi-.61803398875*(hi-lo);y=lo+.61803398875*(hi-lo)
        cx,qx=trial(x,row,r,v,t,states,ids,branches,nremaining,tleft,tright,penalty,hours,weight,normal)
        cy,qy=trial(y,row,r,v,t,states,ids,branches,nremaining,tleft,tright,penalty,hours,weight,normal)
        for _ in range(14):
            if cx<best:best=cx;chosen=qx.copy()
            if cy<best:best=cy;chosen=qy.copy()
            if cx<cy:
                hi=y;y=x;cy=cx;qy=qx.copy();x=hi-.61803398875*(hi-lo)
                cx,qx=trial(x,row,r,v,t,states,ids,branches,nremaining,tleft,tright,penalty,hours,weight,normal)
            else:
                lo=x;x=y;cx=cy;qx=qy.copy();y=lo+.61803398875*(hi-lo)
                cy,qy=trial(y,row,r,v,t,states,ids,branches,nremaining,tleft,tright,penalty,hours,weight,normal)
        if cx<best:best=cx;chosen=qx.copy()
        if cy<best:chosen=qy.copy()
        if free_midcourse and len(row)>8:chosen=fraction_polish(chosen,r,v,t,states,ids,branches,nremaining,tleft,tright,penalty,hours,weight,normal)
        out[i]=chosen
    return out

@njit(cache=True,parallel=True)
def batch(r,v,t,masks,depth,times,positions,counts,branches,tleft,tright,penalty,blended,hours,weight,states,ids,normal,step,max_transfer_s=0.,free_midcourse=False):
    parents=len(t);n=len(counts);direct=np.full((parents,n*4,8),np.nan);staged=np.full((parents,n*3,16),np.nan)
    dc=np.zeros(parents,np.int64);sc=np.zeros(parents,np.int64)
    for i in prange(parents):
        a=expand(r[i],v[i],t[i],masks[i],depth,times,positions,counts,branches,tleft,tright,4,penalty,hours,weight,max_transfer_s)
        b=dsm_expand(r[i],v[i],t[i],masks[i],depth,times,positions,counts,branches,tleft,tright,a,penalty,blended,hours,weight,max_transfer_s,free_midcourse)
        a=polish(a,r[i],v[i],t[i],states,ids,branches,n-depth-1,tleft,tright,penalty,hours,weight,normal,step,max_transfer_s)
        b=polish(b,r[i],v[i],t[i],states,ids,branches,n-depth-1,tleft,tright,penalty,hours,weight,normal,step,max_transfer_s,free_midcourse)
        dc[i]=len(a);sc[i]=len(b);direct[i,:len(a)]=a;staged[i,:len(b)]=b
    return direct,staged,dc,sc
