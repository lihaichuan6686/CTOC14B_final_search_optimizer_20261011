"""束搜索主程序：直接运行本文件。"""
import argparse
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description='本地轨道面窗口束搜索；无参数即可运行。')
    parser.add_argument('--runs',type=int,help='本次搜索平面数量，默认读取 src/settings.py')
    parser.add_argument('--threads',type=int,help='本次计算线程数')
    parser.add_argument('--seed',type=int,help='随机种子')
    parser.add_argument('--check',action='store_true',help='只检查环境和输入数据，不开始搜索')
    args=parser.parse_args()
    python=ROOT/'venv/Scripts/python.exe'
    if Path(sys.prefix).resolve()!=ROOT/'venv':
        if not python.is_file():raise SystemExit('找不到本项目 venv，请查看 README.md。')
        os.execv(str(python),[str(python),str(Path(__file__).resolve()),*sys.argv[1:]])
    sys.dont_write_bytecode=True
    sys.path[:0]=[str(ROOT/'src'),str(ROOT/'src/search')]
    import settings
    threads=args.threads if args.threads is not None else settings.SEARCH_THREADS
    if threads<1 or threads>(os.cpu_count() or 1):parser.error('threads 必须介于 1 和本机逻辑核心数之间')
    os.environ['NUMBA_NUM_THREADS']=str(threads)
    os.environ['NUMBA_CACHE_DIR']=str(ROOT/'.cache/search')
    for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
    import config
    config.WORKERS=threads
    if args.runs is not None:config.BEAM_RUNS=args.runs
    if args.seed is not None:config.RANDOM_SEED=args.seed
    if config.BEAM_RUNS<1:parser.error('runs 必须大于零')
    if args.check:
        import hashlib,json,numpy as np,numba
        manifest=json.loads((ROOT/'src/data/manifest.json').read_text(encoding='utf-8'))
        assert hashlib.sha256(config.TARGET_DATA_FILE.read_bytes()).hexdigest()==manifest['files']['states.npy']
        states=np.load(config.TARGET_DATA_FILE,mmap_mode='r',allow_pickle=False)
        assert states.shape==(35,14401,6) and np.isfinite(states).all()
        print(f'检查通过：35 个目标，{config.BEAM_RUNS} 个平面，{threads} 个线程。');return
    if (ROOT/'STOP').exists():raise SystemExit('检测到 STOP 文件。确认继续后请删除它。')
    from generate_initial_routes import main as search_main
    search_main()

if __name__=='__main__':main()
