"""Numerical operators extracted from the successful optimization path.

Partial and penalized objectives are never admitted to the feasible archive.
Each worker executes one isolated, restartable task at a time.
"""
import contextlib
import hashlib
import json
import os
import signal
import multiprocessing
import time
import traceback
from pathlib import Path

CONTEXT=None

def context():
    global CONTEXT
    if CONTEXT is None:
        import bootstrap
        import numba
        numba.set_num_threads(1)
        import campaign
        CONTEXT=campaign
    return CONTEXT

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def atomic(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temp.replace(p)

def full_pole(d,label='candidate'):
    C=context()
    if sorted(C.visits(d))!=list(range(35)):raise ValueError('Candidate must visit all 35 targets exactly once')
    return C.complete(d,label)

def baseline(task):
    C=context();d=read(task['source']);cfg=task['settings'];rng=C.np.random.default_rng(task['seed'])
    trace=[];incumbent=None;incumbent_report=None
    if d['model']=='j2':
        checked=C.evaluate(d,C.P,strict=True)
        if checked['complete'] and not checked['errors']:
            incumbent=C.copy.deepcopy(d);incumbent['cost']=checked['total_dv_km_s'];incumbent_report=checked
    if d['model']=='two_body':
        from ctoc.kernels.free_impulse import joint_polish,conditioned_nodes
        from ctoc.kernels.basin_escape import compact,node_branch_dp
        initial_report=C.evaluate(d,C.P,strict=False)
        best=C.copy.deepcopy(d) if initial_report['complete'] and not initial_report['errors'] else None
        if best is not None:best['cost']=initial_report['total_dv_km_s']
        for phase in ('joint','free_nodes','branch_compact'):
            q=C.copy.deepcopy(d)
            if phase=='free_nodes':q=conditioned_nodes(q,max_angle=105)
            if phase=='branch_compact':q=compact(node_branch_dp(q,2),threshold=.0001)
            q=C.normalize(joint_polish(q,C.P.states,cfg['joint_seconds']/60,None,
                smoothings=(.01,.001,.00003),maxiter=cfg['joint_iterations'],min_gap_hours=10/3600))
            report=C.evaluate(q,C.P,strict=False)
            if report['complete'] and not report['errors']:
                q['cost']=report['total_dv_km_s'];d=q
                if best is None or q['cost']<best['cost']:best=q
                trace.append(dict(phase=phase,cost=q['cost']))
        if best is None:raise ValueError('No feasible two-body refinement')
        d=best
    from ctoc.algorithms import refine_j2
    try:
        d,meta=refine_j2(d,C.P,C.A.STATES,dict(local_iterations=cfg['scp_iterations']),rng,cfg['j2_seconds'])
        # refine_j2 returns ICRF; strict replay alone determines the published score.
        report=C.evaluate(d,C.P,strict=True)
        if report['errors'] or not report['complete']:raise ValueError(str(report['errors']))
    except Exception:
        if incumbent is None:raise
        traceback.print_exc();d=incumbent;report=incumbent_report;meta=dict(kept_valid_incumbent=True)
    if incumbent is not None and report['total_dv_km_s']>=incumbent['cost']:
        d=incumbent;report=incumbent_report;meta=dict(kept_valid_incumbent=True)
    d['cost']=report['total_dv_km_s'];C.save('candidate.json',d);C.save('candidate_validation.json',report)
    return dict(cost=d['cost'],candidate=str(C.OUT/'candidate.json'),trace=trace,metadata=meta)

def release(task):
    C=context();base=read(task['source']);k=int(task['target']);cfg=task['settings']
    d=C.to_pole_j2(base,C.P);index=d['node_ids'].index(k);old_slot=C.visits(base).index(k)
    d['node_ids'][index]=-1;C.ensure_offsets(d,C.A.STATES)
    d=C.refine(d,C.A.STATES,cfg['release_seconds']/60,max_iterations=cfg['release_iterations'])
    C.save('released_pole.json',d)
    choices=C.A.alternatives(d,k,old_slot)[:cfg['opportunities_per_target']]
    # A large geometric miss is not rejected here: successful historical repairs
    # included starting misses of hundreds of kilometres.
    return dict(target=k,parent=task['source'],released=str(C.OUT/'released_pole.json'),
        partial_cost_NOT_ELIGIBLE=d['cost'],release_gain=base['cost']-d['cost'],opportunities=choices)

def repair(task):
    C=context();cfg=task['settings'];k=int(task['target']);d=read(task['released']);e=task['opportunity'];trace=[]
    moment=e['time_s'];times=C.np.array(d['times']);arc=int(C.np.searchsorted(times,moment)-1)
    if arc<0 or arc>=len(times)-1 or min(moment-times[arc],times[arc+1]-moment)<10:
        raise ValueError('Encounter too close to a node')
    d=C.insert_coast(d,arc,(moment-times[arc])/(times[arc+1]-times[arc]));d['node_ids'][arc+1]=k
    for penalty in (.0002,.002,.02,.2):
        d=C.repair(d,C.A.STATES,[k],.995,penalty,seconds=cfg['repair_seconds'])
        trace.append(dict(penalty=penalty,cost_NOT_ELIGIBLE=d['cost'],excess_km=d['elastic_excess_km']))
        if d['elastic_excess_km']<1e-5:break
    if d['elastic_excess_km']>.5:raise ValueError('Elastic repair did not restore feasibility')
    i=d['node_ids'].index(k);center=C.target_state(C.A.STATES,k,d['times'][i])[0];offset=C.np.array(d['positions'][i])-center
    if C.np.linalg.norm(offset)>.995:d['positions'][i]=(center+offset*.995/C.np.linalg.norm(offset)).tolist()
    C.ensure_offsets(d,C.A.STATES)
    d=C.refine(d,C.A.STATES,cfg['polish_seconds']/60,max_iterations=cfg['scp_iterations'])
    d=full_pole(d)
    return dict(cost=d['cost'],candidate=str(C.OUT/'candidate.json'),target=k,trace=trace)

def topology(task):
    C=context();cfg=task['settings'];d=read(task['source'])
    from ctoc.algorithms import refine_j2
    q,meta=refine_j2(d,C.P,C.A.STATES,dict(local_iterations=cfg['scp_iterations'],primer_samples=cfg['primer_samples'],_topology_mode=task['mode']),
        C.np.random.default_rng(task['seed']),cfg['topology_seconds'],True,task['mode'])
    report=C.evaluate(q,C.P,strict=True)
    if report['errors'] or not report['complete']:raise ValueError(str(report['errors']))
    q['cost']=report['total_dv_km_s'];C.save('candidate.json',q);C.save('candidate_validation.json',report)
    return dict(cost=q['cost'],candidate=str(C.OUT/'candidate.json'),metadata=meta)

def rotation(task):
    C=context();C.save('rotation_parent.json',read(task['source']))
    from global_reassignment import attempt
    out=attempt((0,task['angle'],task['axis']))
    if 'error' in out:raise ValueError(out['error'])
    return dict(cost=out['cost'],candidate=str(C.OUT/(out['label']+'.json')))

def beam(task):
    C=context();import numba
    threads=int(task['threads']);numba.set_num_threads(threads);os.environ['CTOC_BEAM_THREADS']=str(threads)
    C.save('beam_parent.json',read(task['source']))
    from dsm_tail_search import screen,optimize
    cfg=task['settings'];settings=dict(source_file='beam_parent',lookahead_weight=.6,continuous_refinement_s=1800.,
        free_launch=True,require_midcourse=False,max_transfer_s=72*3600.,launch_time_penalty=12.,incumbent_guidance=True,
        max_per_last_target=12,final_count=cfg['beam_final_count'],free_midcourse=task['free_midcourse'])
    result=screen((35,cfg['beam_width'],12.,'beam',1800.,True,False,settings))
    outputs=[]
    for q in result['choices']:
        if q.get('incumbent'):continue
        r=optimize(q['label'])
        if 'error' not in r:outputs.append(dict(cost=r['cost'],candidate=str(C.OUT/(r['label']+'.json'))))
    numba.set_num_threads(1)
    return dict(candidates=outputs,screen=result)

def execute(task):
    # The main program handles Ctrl+C and stops dispatching; finish active work.
    if multiprocessing.current_process().name!='MainProcess':
        signal.signal(signal.SIGINT,signal.SIG_IGN)
    folder=Path(task['folder']);folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'receipt.json'
    inputs={k:hashlib.sha256(Path(task[k]).read_bytes()).hexdigest() for k in ('source','released') if task.get(k)}
    signature=hashlib.sha256(json.dumps(dict(task=task,input_hashes=inputs),sort_keys=True).encode()).hexdigest()
    if receipt.exists():
        old=read(receipt)
        files=[old[k] for k in ('candidate','released') if old.get(k)]
        files += [q['candidate'] for q in old.get('candidates',[])]
        if old.get('signature')==signature and old.get('ok') and all(Path(p).is_file() for p in files):return old
    started=time.monotonic()
    with (folder/'task.log').open('a',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            C=context();C.OUT=folder
            atomic(folder/'request.json',task)
            out=globals()[task['operator']](task)
            result=dict(ok=True,signature=signature,operator=task['operator'],task=task,seconds=time.monotonic()-started,**out)
        except Exception as exc:
            traceback.print_exc();result=dict(ok=False,signature=signature,operator=task['operator'],task=task,error=repr(exc),seconds=time.monotonic()-started)
    atomic(receipt,result)
    return result
