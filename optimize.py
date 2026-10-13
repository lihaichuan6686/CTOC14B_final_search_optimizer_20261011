"""批量优化主程序：默认读取最近一次束搜索结果。"""
import argparse
import multiprocessing
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description='本地自动批量优化；无参数时优化最近一次搜索结果。')
    parser.add_argument('--input',nargs='+',type=Path,help='可选：已有搜索批次目录或完整轨迹 JSON')
    parser.add_argument('--work',type=Path,default=ROOT/'results/optimization',help='优化结果和接续目录')
    parser.add_argument('--workers',type=int,help='优化工作进程数')
    parser.add_argument('--hours',type=float,default=0,help='停止派发新任务的小时预算；0 表示按配置完成')
    parser.add_argument('--check',action='store_true',help='只检查环境，不开始优化')
    args=parser.parse_args()
    python=ROOT/'venv/Scripts/python.exe'
    if Path(sys.prefix).resolve()!=ROOT/'venv':
        if not python.is_file():raise SystemExit('找不到本项目 venv，请查看 README.md。')
        os.execv(str(python),[str(python),str(Path(__file__).resolve()),*sys.argv[1:]])
    sys.dont_write_bytecode=True
    sys.path[:0]=[str(ROOT/'src'),str(ROOT/'src/optimizer')]
    import settings
    os.environ['NUMBA_NUM_THREADS']=str(settings.SEARCH_THREADS)
    os.environ['NUMBA_CACHE_DIR']=str(ROOT/'.cache/optimizer')
    if args.check:
        import bootstrap,cvxpy
        from ctoc.problem import Problem
        p=Problem.load(bootstrap.ensure_data())
        assert 'CLARABEL' in cvxpy.installed_solvers()
        print(f'检查通过：J2 星历 {p.states.shape}，CLARABEL 求解器可用。');return
    paths=args.input
    if not paths:
        latest=ROOT/'results/latest_run.txt'
        if not latest.is_file():raise SystemExit('还没有束搜索结果。请先运行 search.py，或用 --input 指定已有结果。')
        paths=[Path(latest.read_text(encoding='utf-8').strip())]
    from batch_optimize import run
    run(paths,args.work,args.workers if args.workers is not None else settings.OPTIMIZER_WORKERS,
        settings.SEARCH_THREADS,dict(settings.OPTIMIZER),args.hours)

if __name__=='__main__':
    multiprocessing.freeze_support()
    main()
