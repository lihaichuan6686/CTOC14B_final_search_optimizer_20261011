"""Shared numerical context for the automated batch optimizer."""
import os,sys,json,time,copy,traceback,ctypes,argparse
from pathlib import Path
import bootstrap
import numpy as np
from ctoc.problem import Problem
from ctoc.algorithms import configure_branches,to_pole_j2
from ctoc.trajectory import rotate,normalize
from ctoc.evaluation import evaluate
from ctoc.kernels.elliptic_reduced_scp import refine
from ctoc.kernels.multi_impulse_round8 import ensure_offsets,insert_coast
from ctoc.kernels.physics import target_state
from elastic_visit_scp import repair
import coast_alternatives as A
P=Problem.load(bootstrap.ensure_data());configure_branches(2)
A.STATES=P.pole_states();OUT=bootstrap.ROOT/'work'
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(name,d):
    p=OUT/name;p.parent.mkdir(exist_ok=True,parents=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');tmp.replace(p)
def visits(d):return [k for k in d['node_ids'] if k>=0]
def complete(d,label):
    d=rotate(d,P.pole.T);d.update(model='j2',frame='ICRF');d=normalize(d)
    report=evaluate(d,P,strict=True)
    if report['errors'] or not report['complete']:raise ValueError(str(report['errors']))
    d['cost']=report['total_dv_km_s'];save(label+'.json',d);save(label+'_validation.json',report)
    return d
