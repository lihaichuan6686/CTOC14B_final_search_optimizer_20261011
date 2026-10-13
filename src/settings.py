"""只需修改这里的计算规模；所有时刻单位为秒，速度增量为 km/s。"""
import os

# 本机默认最多用 8 个线程，优化同时处理 4 个独立任务。
SEARCH_THREADS=max(1,min(8,(os.cpu_count() or 1)-2))
SEARCH_RUNS=128
SEARCH_SEED=20261011
OPTIMIZER_WORKERS=min(4,SEARCH_THREADS)

OPTIMIZER={
    'random_seed':20261011,
    'max_seeds':8,               # 从束搜索结果中选取的初解数
    'active_parents':4,          # 每轮继续探索的父解数
    'exploration_fraction':.25,  # 给不同访问顺序/时窗的探索名额
    'relocation_rounds':2,       # 目标换位轮数；每轮评估全部 35 个目标
    'opportunities_per_target':3,
    'repairs_per_parent':12,
    'joint_seconds':90,
    'joint_iterations':1200,
    'j2_seconds':120,
    'scp_iterations':1200,
    'release_seconds':12,
    'release_iterations':170,
    'repair_seconds':18,
    'polish_seconds':80,
    'topology_seconds':90,
    'primer_samples':61,
    'rotation_parents':1,
    'deep_beam_parents':1,       # 少量精英做中途脉冲重搜；设为 0 可缩短计算
    'beam_width':96,
    'beam_final_count':8,
    'memory_gb_per_worker':1.1,
}
