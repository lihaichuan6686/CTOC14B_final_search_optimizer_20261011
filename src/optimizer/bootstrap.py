"""Shared local paths and numerical thread limits."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ.setdefault('NUMBA_NUM_THREADS',str(min(8,os.cpu_count() or 1)))
os.environ.setdefault('NUMBA_CACHE_DIR',str(PROJECT/'.cache/optimizer'))
sys.path.insert(0,str(ROOT/'engine'))
def ensure_data():return PROJECT/'src/data'
