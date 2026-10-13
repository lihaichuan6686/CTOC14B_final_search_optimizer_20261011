"""Local plane-window beam implementation; use the root search.py entry point."""
import os, signal, json, gzip, hashlib, time, platform, sys
from pathlib import Path
from datetime import datetime
PROCESS_STARTED=time.monotonic()
from run_budget import RunBudget
import numpy as np
import numba
import config as C
# Numba specializes module constants. Config/source changes get a separate cache.
_cache_base=Path(os.environ.get('NUMBA_CACHE_DIR',str(C.ROOT/'.numba_cache')))
_cache_digest=hashlib.sha256()
for _source in sorted(Path(__file__).parent.glob('*.py')):
    _cache_digest.update(_source.name.encode());_cache_digest.update(_source.read_bytes())
os.environ['NUMBA_CACHE_DIR']=str(_cache_base/_cache_digest.hexdigest()[:16])
numba.config.reload_config()
from beam_population import continuous_beam_run

STOP=False

def stop(*_):
    global STOP
    STOP=True

def atomic(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)



def main():
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    if hasattr(signal,"SIGUSR1"):
        signal.signal(signal.SIGUSR1,stop)
    budget=RunBudget(PROCESS_STARTED,C.SERVER_BUDGET_SECONDS,
        C.SERVER_MIN_RUN_RESERVE_SECONDS,C.SERVER_RUN_RESERVE_FACTOR)
    stop_reason=None
    def should_stop():
        nonlocal stop_reason
        if STOP or (C.ROOT/"STOP").exists():
            stop_reason="requested_stop"
            return True
        if budget.expired():
            stop_reason="time_budget"
            return True
        return False
    numba.set_num_threads(min(C.WORKERS,numba.config.NUMBA_NUM_THREADS))
    print(f'CPU_CHECK workers={C.WORKERS} numba_threads={numba.get_num_threads()}',flush=True)
    if not (0 < C.PLANE_TOLERANCE_DEG < 90 and C.WINDOW_CELL_MAX_H > 0
            and C.FIRST_PHASE_SAMPLES >= 4 and C.FIRST_TIME_SAMPLES_PER_CELL >= 3
            and C.FIRST_REFINE_PER_TARGET >= 1 and C.FIRST_COORDINATE_PASSES >= 1
            and C.MIN_TARGET_GAP_S > 0 and C.MAX_TRANSFER_H > 0
            and C.FIRST_MAX_ARRIVAL_H > 0
            and C.FIRST_ALTITUDE_SAMPLES>=1 and C.FIRST_KEEP_PER_CELL_BRANCH>=1
            and C.FIRST_CELL_MAX_H>0 and C.LOOKAHEAD_CELL_MAX_H>0
            and (C.EXPANSION_TIME_POINTS==1 or C.EXPANSION_TIME_POINTS>=3)
            and C.EXPANSION_MINIMA_PER_CELL>=1
            and 590 <= C.INITIAL_ALTITUDE_MIN_KM <= C.INITIAL_ALTITUDE_MAX_KM <= 610):
        raise ValueError('Invalid plane-window search configuration')
    shard_index,shard_count=0,1
    run_indices=range(shard_index,C.BEAM_RUNS,shard_count)
    name=datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+str(os.getpid())+f'_s{shard_index:04d}'
    out=C.ROOT/'results/search'/name;out.mkdir(parents=True,exist_ok=False)
    parts=out/'routes';parts.mkdir()
    latest='latest_run.txt' if shard_count==1 else f'latest_run_shard_{shard_index:04d}.txt'
    (C.ROOT/'results'/latest).write_text(str(out),encoding='utf-8')
    checksum=hashlib.sha256(C.TARGET_DATA_FILE.read_bytes()).hexdigest()
    states=np.ascontiguousarray(np.load(C.TARGET_DATA_FILE,allow_pickle=False),dtype=float)
    assert states.shape==(35,14401,6) and np.isfinite(states).all()
    meta=dict(schema='ctoc14b.stage1.v6',run_id=name,status='running',data_sha256=checksum,
              units='km,s,km/s',epoch_utc='2035-01-01T12:00:00Z',frame='ICRF',model='two_body',
              mission_end_s=864000,completed_runs=0,complete_routes=0,best_screening_cost_km_s=None,
              python=sys.version,platform=platform.platform(),threads=numba.get_num_threads(),
              configuration={k:v for k,v in vars(C).items() if k.isupper() and isinstance(v,(int,float,tuple,str))},
              source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
              algorithm='v6_plane_window_beam',plane_sampling='uniform_oriented_sphere_sector_no_prescreen',
              search_profile=C.SEARCH_PROFILE,shard_index=shard_index,shard_count=shard_count,
              assigned_runs=len(run_indices),total_plane_runs=C.BEAM_RUNS)
    meta.update(budget_seconds=C.SERVER_BUDGET_SECONDS,stop_reason=None)
    atomic(out/'run_metadata.json',meta);best=[];started=PROCESS_STARTED
    print(f"TIME_BUDGET seconds={C.SERVER_BUDGET_SECONDS} runs_cap={C.BEAM_RUNS}",flush=True)
    try:
        for i in run_indices:
            if should_stop():break
            if not budget.can_start():
                stop_reason="time_reserve"
                break
            parameter_rng=np.random.default_rng(C.RANDOM_SEED+1009*i+7919)
            params=dict(width=C.BEAM_WIDTHS[i%len(C.BEAM_WIDTHS)],seed=C.RANDOM_SEED+1009*i,
                        penalty=float(parameter_rng.choice(C.TIME_PENALTIES)),
                        maxrev=int(parameter_rng.choice(C.MAX_REVOLUTIONS_BY_RUN)),mode=int(parameter_rng.integers(2)))
            diagnostics={}
            t=time.monotonic();cpu_started=time.process_time()
            routes=continuous_beam_run(states,**params,diagnostics=diagnostics,should_stop=should_stop)
            if diagnostics.get("aborted"):
                meta.update(interrupted_global_run=i+1,interrupted_diagnostics=diagnostics)
                break
            diagnostics["search_average_cpu_cores"]=(time.process_time()-cpu_started)/max(time.monotonic()-t,1e-9)
            part=parts/f'run_{i+1:06d}.jsonl.gz';tmp=part.with_suffix('.gz.tmp')
            with gzip.open(tmp,'wt',encoding='utf-8') as f:
                for route in routes:
                    assert len(route.order)==35 and set(route.order)==set(range(35))
                    d=route.to_dict();d.update(schema='ctoc14b.route.v6',data_sha256=checksum,epoch_utc=meta['epoch_utc'],frame='ICRF',mission_end_s=864000)
                    f.write(json.dumps(d,ensure_ascii=False,allow_nan=False)+'\n')
                    best.append(d)
            tmp.replace(part)
            # Shard receipt is the commit marker; stage2 never ingests a half-written shard.
            atomic(part.with_suffix('.receipt.json'),dict(file=part.name,sha256=hashlib.sha256(part.read_bytes()).hexdigest(),count=len(routes),run=i+1))
            best=sorted(best,key=lambda d:d['cost_km_s'])[:200]
            meta.update(completed_runs=meta['completed_runs']+1,last_global_run=i+1,complete_routes=meta['complete_routes']+len(routes),elapsed_s=time.monotonic()-started,
                        best_screening_cost_km_s=best[0]['cost_km_s'] if best else None)
            atomic(out/'best_routes.json',dict(routes=best));atomic(out/'run_metadata.json',meta)
            with (out/'per_run.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(run=i+1,seconds=time.monotonic()-t,count=len(routes),parameters=params,diagnostics=diagnostics,best=min((r.cost_km_s for r in routes),default=None)))+'\n')
            budget.observe(time.monotonic()-t)
            print(f'RUN {meta["completed_runs"]}/{len(run_indices)} global={i+1}/{C.BEAM_RUNS} shard={shard_index}/{shard_count} complete={len(routes)} best={meta["best_screening_cost_km_s"]}',flush=True)
        meta["status"]="stopped" if stop_reason else "completed"
        meta["stop_reason"]=stop_reason
        print(f"FINISH status={meta['status']} reason={stop_reason} completed_runs={meta['completed_runs']} elapsed_s={time.monotonic()-started:.1f}",flush=True)
    except BaseException as e:
        meta.update(status='failed',error=repr(e));raise
    finally:
        meta["elapsed_s"]=time.monotonic()-started
        atomic(out/'run_metadata.json',meta)

if __name__=='__main__':main()
