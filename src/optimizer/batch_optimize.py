"""Batch multistart free-node optimization with automatic encounter relocation.

No preferred target, order, plane or archived final answer enters a normal run.
"""
import argparse
import concurrent.futures as futures
import hashlib
import heapq
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import signal
import sys
import time
import bootstrap
from batch_ops import atomic,read,execute

STOP=False
def stop(*_):
    global STOP
    STOP=True

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def physical(d):return {k:d[k] for k in ('initial','positions','times','node_ids','branches','model','frame')}
def family(d):
    encounters=[(int(k),float(t)) for k,t in zip(d['node_ids'],d['times']) if k>=0]
    return digest(([k for k,t in encounters],[int(t//14400) for k,t in encounters]))

def diverse(items,limit,rng,explore=.25):
    """Cost-first family representatives, plus seeded exploration of other families."""
    ordered=sorted(items,key=lambda q:(q['cost'],q['id']))
    representatives=[];seen=set()
    for q in ordered:
        if q['family'] not in seen:representatives.append(q);seen.add(q['family'])
    n=min(limit,len(representatives));quota=max(1,math.ceil(n*(1-explore))) if n else 0
    picked=representatives[:quota];rest=representatives[quota:];rng.shuffle(rest);picked+=rest[:n-len(picked)]
    # Fill only after each distinct family had a chance; identical physical inputs dedup earlier.
    ids={q['id'] for q in picked}
    for q in ordered:
        if len(picked)>=limit:break
        if q['id'] not in ids:picked.append(q);ids.add(q['id'])
    return picked

def repair_choices(releases,limit,rng):
    """Separate rankings avoid equating fuel relief with a geometric proxy."""
    items=[]
    for release in releases:
        if not release['ok']:continue
        for rank,opportunity in enumerate(release['opportunities']):
            items.append(dict(target=release['target'],released=release['released'],opportunity=opportunity,
                gain=release['release_gain'],rank=rank))
    items.sort(key=lambda q:(q['target'],q['rank'],q['opportunity']['time_s']))
    by_gain=sorted(items,key=lambda q:(-q['gain'],q['rank'],q['target']))
    by_distance=sorted(items,key=lambda q:(q['opportunity']['distance_km'],q['target']))
    shuffled=list(items);rng.shuffle(shuffled)
    picked=[];seen=set();target_counts={}
    # At most two repair slots per target; reserve one third for exploration.
    quota=max(1,limit//3)
    for stream,count in ((by_gain,quota),(by_distance,quota),(shuffled,limit)):
        added=0
        for q in stream:
            key=(q['target'],q['rank'])
            if key in seen or target_counts.get(q['target'],0)>=2:continue
            picked.append(q);seen.add(key);target_counts[q['target']]=target_counts.get(q['target'],0)+1;added+=1
            if len(picked)>=limit:return picked
            if added>=count:break
    return picked

class Runner:
    def __init__(self,args,cfg):
        self.args=args;self.cfg=cfg;self.work=args.work.resolve();self.archive={};self.start=time.monotonic();self.stats={}
        self.work.mkdir(parents=True,exist_ok=True)
    def stopped(self):return STOP or (self.work/'STOP').exists() or (self.args.hours>0 and time.monotonic()-self.start>=self.args.hours*3600)
    def task(self,operator,source,tag,**kwargs):
        return dict(operator=operator,source=source,settings=self.cfg,seed=self.cfg['random_seed']+int(digest(tag)[:8],16),
            folder=str(self.work/'tasks'/tag),**kwargs)
    def wave(self,pool,tasks):
        queue=iter(tasks);pending={};results=[]
        while True:
            while len(pending)<self.args.workers and not self.stopped():
                task=next(queue,None)
                if task is None:break
                pending[pool.submit(execute,task)]=task
            if not pending:break
            done,_=futures.wait(pending,return_when=futures.FIRST_COMPLETED)
            for future in done:
                task=pending.pop(future);result=future.result();results.append(result)
                name=task['operator'];stats=self.stats.setdefault(name,dict(done=0,failed=0));stats['done']+=1
                if not result['ok']:stats['failed']+=1
                else:self.admit(result)
                print(json.dumps(dict(event='task_done',operator=name,ok=result['ok'],cost=result.get('cost'),
                    best=min((q['cost'] for q in self.archive.values()),default=None))),flush=True)
                atomic(self.work/'status.json',dict(stats=self.stats,elapsed_s=time.monotonic()-self.start,archive_count=len(self.archive)))
        return results
    def admit(self,result):
        candidates=result.get('candidates',[]) if 'candidate' not in result else [result]
        for item in candidates:
            p=Path(item['candidate']);d=read(p)
            report_path=p.with_name(p.stem+'_validation.json')
            report=read(report_path)
            if (not report.get('strict') or not report.get('continuous_replay') or not report.get('complete')
                    or report.get('errors') or report.get('covered')!=35 or not math.isfinite(report['total_dv_km_s'])):
                raise ValueError('Attempt to admit an unverified candidate')
            d['cost']=report['total_dv_km_s'];key=digest(physical(d));dest=self.work/'archive'/f'{key}.json'
            atomic(dest,d)
            self.archive[key]=dict(id=key,family=family(d),cost=d['cost'],file=str(dest),origin=str(p),operator=result['operator'])
            best=min(self.archive.values(),key=lambda q:(q['cost'],q['id']))
            atomic(self.work/'BEST_J2.json',read(best['file']))
            if best['id']==key:atomic(self.work/'BEST_J2_validation.json',report)
            atomic(self.work/'archive.json',sorted(self.archive.values(),key=lambda q:q['cost']))
    def parents(self,round_id):
        return diverse(list(self.archive.values()),self.cfg['active_parents'],random.Random(self.cfg['random_seed']+round_id),self.cfg['exploration_fraction'])

def intake(paths,problem,pool_limit):
    from route_io import check_source,discover,convert
    # Bound RAM when a 2200-run search contains hundreds of thousands of routes.
    # Keep a cheap-cost pool and an independent reproducible hash reservoir.
    cheap=[];reservoir=[];errors=[];seen_count=0
    for path in paths:
        path=path.resolve();check_source(path,problem)
        data=read(path) if path.is_file() and path.suffix=='.json' else None
        records=[(path.name,data)] if data is not None and ('order' in data or 'node_ids' in data) else discover(path)
        for origin,record in records:
            try:
                source_cost=float(record.get('cost',record.get('cost_km_s',math.nan)))
                if not math.isfinite(source_cost):raise ValueError('Non-finite intake cost')
                sample_key=digest(record);priority=int(sample_key,16)
                take_cost=len(cheap)<pool_limit or source_cost < -cheap[0][0]
                take_sample=len(reservoir)<pool_limit or priority < -reservoir[0][0]
                seen_count+=1
                if not take_cost and not take_sample:continue
                key,d,cost=convert(record,problem)
                if d['frame']!='ICRF' or sorted(k for k in d['node_ids'] if k>=0)!=list(range(35)) or not math.isfinite(cost):
                    raise ValueError('Only complete finite ICRF seeds are accepted')
                item=dict(id=key,family=family(d),cost=cost,data=d,origin=origin)
                for heap,rank,take in ((cheap,cost,take_cost),(reservoir,priority,take_sample)):
                    if take:
                        entry=(-rank,seen_count,item)
                        if len(heap)<pool_limit:heapq.heappush(heap,entry)
                        else:heapq.heapreplace(heap,entry)
            except (KeyError,TypeError,ValueError) as exc:errors.append(dict(origin=origin,error=str(exc)))
    items={q['id']:q for _,_,q in cheap+reservoir}
    if not items:raise ValueError('No complete seeds found')
    return list(items.values()),errors,seen_count

def locked_run(args,cfg):
    import numba,psutil
    from ctoc.problem import Problem
    problem=Problem.load(bootstrap.ensure_data())
    items,errors,seen_count=intake(args.input,problem,max(32,cfg['max_seeds']*8))
    chosen=diverse(items,cfg['max_seeds'],random.Random(cfg['random_seed']),cfg['exploration_fraction'])
    runner=Runner(args,cfg)
    code_hash=digest({p.relative_to(bootstrap.ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in bootstrap.ROOT.rglob('*.py') if not any(x in p.parts for x in ('.venv','work','vendor')) and not p.is_relative_to(args.work.resolve())})
    signature=digest(dict(config=cfg,code=code_hash,inputs=sorted((q['id'],q['cost']) for q in chosen)))
    manifest=runner.work/'run_manifest.json'
    if manifest.exists() and read(manifest)['signature']!=signature:
        raise ValueError('Inputs/config/code changed: use a new --work directory')
    import importlib.metadata as metadata
    versions={n:metadata.version(n) for n in ('numpy','scipy','numba','cvxpy','clarabel','psutil')}
    if manifest.exists() and read(manifest)['versions']!=versions:
        raise ValueError('Runtime versions changed: use a new --work directory')
    atomic(manifest,dict(signature=signature,config=cfg,code_hash=code_hash,versions=versions,workers=args.workers,
        selected_ids=[q['id'] for q in chosen],intake_errors=errors,source_count=seen_count,retained_intake_pool=len(items),
        started_from_two_body_only=all(q['data']['model']=='two_body' for q in chosen),
        starting_best_by_model={model:min(q['cost'] for q in chosen if q['data']['model']==model) for model in {q['data']['model'] for q in chosen}},
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    tasks=[]
    for q in chosen:
        path=runner.work/'inputs'/f"{q['id']}.json";atomic(path,q['data'])
        tasks.append(runner.task('baseline',str(path),'baseline_'+q['id'][:20]))
    for sig in ('SIGTERM','SIGINT','SIGUSR1'):
        if hasattr(signal,sig):signal.signal(getattr(signal,sig),stop)
    print(f'CPU_CHECK allocated={args.cpu_budget} process_workers={args.workers} numerical_threads_per_worker=1',flush=True)
    with futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        runner.wave(pool,tasks)
        if not runner.archive and not runner.stopped():raise RuntimeError('All baseline tasks failed; inspect task.log files')
        for iteration in range(cfg['relocation_rounds']):
            if runner.stopped():break
            parents=runner.parents(iteration);tasks=[]
            for p in parents:
                for target in range(35):
                    tasks.append(runner.task('release',p['file'],f"round{iteration}_release_{p['id'][:16]}_{target:02}",target=target,parent_id=p['id']))
            released=runner.wave(pool,tasks);repairs=[]
            for p in parents:
                subset=[q for q in released if q['task']['parent_id']==p['id']]
                selected=repair_choices(subset,cfg['repairs_per_parent'],random.Random(cfg['random_seed']+iteration+int(p['id'][:8],16)))
                atomic(runner.work/'selection'/f"round{iteration}_{p['id']}.json",selected)
                for index,q in enumerate(selected):
                    repairs.append(runner.task('repair',p['file'],f"round{iteration}_repair_{p['id'][:16]}_{index:02}",
                        target=q['target'],released=q['released'],opportunity=q['opportunity']))
            runner.wave(pool,repairs)
            # Add and remove impulse nodes on the accepted full routes.
            topology=[]
            for p in runner.parents(iteration+100):
                for mode in (0,3):
                    topology.append(runner.task('topology',p['file'],f"round{iteration}_topology_{p['id'][:16]}_{mode}",mode=mode))
            runner.wave(pool,topology)
        rotations=[]
        if not runner.stopped():
            for p in runner.parents(999)[:cfg['rotation_parents']]:
                for axis in ('r','z'):
                    for angle in (-3.,3.):
                        rotations.append(runner.task('rotation',p['file'],f"rotate_{p['id'][:16]}_{axis}_{angle}",axis=axis,angle=angle))
            runner.wave(pool,rotations)
    # All process workers have exited. Deep beam can use the full CPU allocation.
    if not runner.stopped():
        for p in runner.parents(1000)[:cfg['deep_beam_parents']]:
            for free in (False,True):
                if runner.stopped():break
                task=runner.task('beam',p['file'],f"beam_{p['id'][:16]}_{int(free)}",threads=min(args.cpu_budget,numba.config.NUMBA_NUM_THREADS),free_midcourse=free)
                result=execute(task)
                if result['ok']:runner.admit(result)
                else:print('BEAM_FAILED',result['error'],flush=True)
    final=dict(status='stopped' if runner.stopped() else 'completed',stats=runner.stats,
        best_j2_km_s=min((q['cost'] for q in runner.archive.values()),default=None),elapsed_s=time.monotonic()-runner.start,
        started_from_two_body_only=all(q['data']['model']=='two_body' for q in chosen))
    atomic(runner.work/'summary.json',final);print(json.dumps(final),flush=True)

def run(input_paths,work,workers,cpu_budget,cfg,hours=0):

    from types import SimpleNamespace
    args=SimpleNamespace(input=[Path(p) for p in input_paths],work=Path(work),workers=workers,cpu_budget=cpu_budget,hours=hours)
    def invalid(message):raise ValueError(message)

    if args.cpu_budget<1 or args.hours<0:invalid('Invalid resource limit')
    for key in ('max_seeds','active_parents','repairs_per_parent','opportunities_per_target','joint_iterations','scp_iterations','release_iterations','beam_width','beam_final_count'):
        if type(cfg[key]) is not int or cfg[key]<1:invalid('Invalid configuration: '+key)
    for key in ('relocation_rounds','rotation_parents','deep_beam_parents'):
        if type(cfg[key]) is not int or cfg[key]<0:invalid('Invalid configuration: '+key)
    if not 0<=cfg['exploration_fraction']<=1:invalid('Invalid exploration_fraction')
    import psutil
    if cfg['memory_gb_per_worker']<=0:invalid('memory_gb_per_worker must be positive')
    for key in ('joint_seconds','j2_seconds','release_seconds','repair_seconds','polish_seconds','topology_seconds'):
        if not math.isfinite(cfg[key]) or cfg[key]<=0:invalid('Invalid time budget: '+key)
    available=psutil.virtual_memory().available/1024**3
    memory_workers=max(1,int(available/cfg['memory_gb_per_worker']))
    args.workers=min(args.workers or args.cpu_budget,args.cpu_budget,memory_workers)
    if args.workers<1:invalid('workers must be positive')

    args.work.mkdir(parents=True,exist_ok=True)
    if (args.work/'STOP').exists():invalid('STOP exists in the work directory; remove it only when resuming intentionally')
    guard=(args.work/'batch.guard').open('a+b')
    try:
        if os.name=='nt':
            import msvcrt
            if guard.seek(0,2)==0:guard.write(b'0');guard.flush()
            guard.seek(0);msvcrt.locking(guard.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(guard.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        locked_run(args,cfg)
    finally:guard.close()
