"""Parallelize independent parent expansions without changing candidate arithmetic."""
import numpy as np
from numba import njit,prange
from large_block_kernel import expand
from dsm_kernel import dsm_expand

@njit(cache=True,parallel=True)
def batch(r,v,t,masks,depth,times,positions,counts,branches,tleft,tright,penalty,blended,pressure_hours=0.,pressure_weight=0.,max_transfer_s=0.):
    parents=len(t);n=len(counts);direct=np.full((parents,n*4,8),np.nan);staged=np.full((parents,n*3,16),np.nan)
    dc=np.zeros(parents,np.int64);sc=np.zeros(parents,np.int64)
    for i in prange(parents):
        a=expand(r[i],v[i],t[i],masks[i],depth,times,positions,counts,branches,tleft,tright,4,penalty,pressure_hours,pressure_weight,max_transfer_s)
        b=dsm_expand(r[i],v[i],t[i],masks[i],depth,times,positions,counts,branches,tleft,tright,a,penalty,blended,pressure_hours,pressure_weight,max_transfer_s)
        dc[i]=len(a);sc[i]=len(b);direct[i,:len(a)]=a;staged[i,:len(b)]=b
    return direct,staged,dc,sc
