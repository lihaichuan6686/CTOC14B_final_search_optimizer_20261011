"""J2 repair with exact target-relative offsets and an L1 feasibility penalty.

Selected encounter constraints may be violated during search. The merit
function can accept a fuel increase when it reduces encounter infeasibility.
This is an engineering SCP adaptation, not a global convergence certificate.
"""
import bootstrap
import json
import numpy as np
import copy
import time
import cvxpy as cp
import scipy.sparse as sp
from ctoc.kernels.elliptic_reduced_scp import pack,evaluate
from ctoc.kernels.elliptic_joint_scp import initial_elliptic
from ctoc.kernels.physics import RE


def repair(data,st,targets,radius_km,penalty,seconds=50):
    d=copy.deepcopy(data);ids=np.array(d['node_ids']);branches=np.array(d['branches'])
    n=len(branches);off=6+3*n;z=pack(d,st)
    selected=np.array([j for j,k in enumerate(ids[1:]) if k in targets],dtype=int)
    held=np.array([j for j,k in enumerate(ids[1:]) if k>=0 and k not in targets],dtype=int)
    scales=np.r_[[.04,.04,.04,.03,.05,.05],np.full(3*n,.015),np.full(n,.2)]
    for j in held:scales[6+3*j:9+3*j]=.25
    for j in selected:scales[6+3*j:9+3*j]=500.
    trust=.5;deadline=time.monotonic()+seconds;history=[]
    def excess(q):
        offsets=q[6:off].reshape(n,3)[selected]
        return np.maximum(np.linalg.norm(offsets,axis=1)-radius_km,0)
    for iteration in range(300):
        if time.monotonic()>deadline:break
        dv,J,rad,_=evaluate(z,ids,branches,st)
        old_cost=float(np.linalg.norm(dv,axis=1).sum());old=old_cost+penalty*float(excess(z).sum())
        step=cp.Variable(len(z));epigraph=cp.Variable(n);slack=cp.Variable(len(selected),nonneg=True)
        dx=cp.multiply(scales,step);q=z+dx;rr=cp.reshape(q[6:off],(n,3),order='C');tt=q[off:]
        impulse=cp.reshape(dv.ravel()+sp.csc_matrix(J)@dx,(n,3),order='C')
        constraints=[cp.norm(impulse,axis=1)<=epigraph,cp.abs(step)<=trust,
                     q[1]>=-1.5707,q[1]<=1.5707,q[3]>=-.999999,q[3]<=.999999,
                     cp.norm(q[4:6])<=.9985,tt[0]>=.05,tt[-1]<=240,
                     tt[1:]-tt[:-1]>=d.get('minimum_gap_hours',.005),
                     cp.norm(rr[selected],axis=1)<=radius_km+slack]
        if len(held):constraints.append(cp.norm(rr[held],axis=1)<=.995)
        model=cp.Problem(cp.Minimize(cp.sum(epigraph)+penalty*cp.sum(slack)+1e-8*cp.sum_squares(step)),constraints)
        try:model.solve(solver='CLARABEL',tol_gap_abs=1e-8,tol_gap_rel=1e-8,tol_feas=1e-9,max_iter=200)
        except cp.error.SolverError:trust*=.4;continue
        if step.value is None or not np.all(np.isfinite(step.value)):trust*=.4;continue
        predicted=old-float(model.value);accepted=False;new_cost=old_cost
        for fraction in (1.,.5,.25,.125,.0625):
            trial=z+np.asarray(step.value)*scales*fraction
            try:
                pulses,_,minimum,_=evaluate(trial,ids,branches,st,False)
                cost=float(np.linalg.norm(pulses,axis=1).sum())
            except (ValueError,np.linalg.LinAlgError,ZeroDivisionError):continue
            merit=cost+penalty*float(excess(trial).sum())
            if minimum>=RE+202 and merit<old-1e-9:
                z=trial;accepted=True;new_cost=cost
                ratio=(old-merit)/max(predicted,1e-12)
                trust=min(10,trust*1.4) if ratio>.6 else max(1e-5,trust*.6) if ratio<.1 else trust
                break
        if not accepted:trust*=.4
        item=dict(iteration=iteration,cost=new_cost,excess_km=float(excess(z).sum()),
                  accepted=accepted,trust=trust,penalty=penalty,radius_km=np.asarray(radius_km).tolist())
        history.append(item)
        if iteration%10==0:print('ELASTIC',json.dumps(item),flush=True)
        if trust<1e-7:break
    dv,_,rad,x=evaluate(z,ids,branches,st,False)
    init=initial_elliptic(z[:6]);positions=np.vstack([init[:3],x[6:off].reshape(n,3)*30000])
    d.update(initial=init.tolist(),positions=positions.tolist(),times=np.r_[0,z[off:]*3600].tolist(),
             cost=float(np.linalg.norm(dv,axis=1).sum()),minimum_arc_radius_km=float(rad),
             elastic_excess_km=float(excess(z).sum()),elastic_history=history)
    d.pop('encounter_offsets_km',None)
    return d
