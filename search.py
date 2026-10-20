from pathlib import Path
import gzip
import json
import time
import numpy as np
from numba import set_num_threads
from threadpoolctl import threadpool_limits
from beam_search import SearchParameters, continuous_beam_run

PROJECT = Path(__file__).resolve().parent
EPHEMERIS_FILE = PROJECT / "data/states.npy"
OUTPUT_DIRECTORY = PROJECT / "results/search"
PLANE_COUNT = 128
THREAD_COUNT = 8
RANDOM_SEED = 20261011
BEAM_WIDTHS = (96, 192, 192, 288)
MAX_REVOLUTIONS = (0, 0, 0, 1)
TIME_PENALTIES = (0.4, 0.8, 1.2, 1.6)

# 所有数值参数直接在这里修改；具名元组只负责将参数明确传给 Numba 内核。
PARAMETERS = SearchParameters(
    plane_tolerance_deg=6.0,
    plane_normal_z_min=-1.0,  # cos(i) 均匀抽样，覆盖整个有向法向球
    plane_normal_z_max=1.0,
    window_edge_tolerance_s=0.05, #二分查找分辨率
    window_cell_max_h=6.0, #每个时窗切成最长6个小时的小格
    max_transfer_h=72.0,
    min_target_gap_s=30.0, #防止零时长lambert！！
    first_max_arrival_h=72.0,
    first_phase_samples=64, #延初始轨道相位角采64个点
    first_altitude_samples=2,
    initial_altitude_min_km=590.1,
    initial_altitude_max_km=609.9,
    first_time_samples_per_cell=3,
    first_keep_per_cell_branch=4,
    first_refine_per_target=4,
    first_golden_iterations=8,
    first_coordinate_passes=1,
    first_cell_max_h=6.0,
    expansion_time_points=1,
    expansion_minima_per_cell=1,
    representatives_per_branch=3,
    local_refinements_per_branch=1,
    golden_iterations=5,
    time_distance_s=600.0,
    velocity_distance_km_s=0.06,
    state_bucket_cap=8,
    exploration_fraction=0.30,
    lookahead_pool_multiplier=2,
    lookahead_weight=0.60,
    lookahead_cell_max_h=6.0,
    search_cost_ceiling_km_s=24.0,
)


def main():
    # 星历形状 (35,14401,6)：ICRF 中的位置/km和速度/km/s，每60秒一行。
    # t=0 对应2035-01-01 12:00:00 UTC；任务总长864000秒。
    states = np.load(EPHEMERIS_FILE)
    set_num_threads(THREAD_COUNT)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for run_index in range(PLANE_COUNT):
        started = time.perf_counter()
        seed = RANDOM_SEED + 1009 * run_index
        rng = np.random.default_rng(seed + 7919)
        width = BEAM_WIDTHS[run_index % len(BEAM_WIDTHS)]
        time_penalty = float(rng.choice(TIME_PENALTIES))
        max_revolutions = int(rng.choice(MAX_REVOLUTIONS))
        time_mode = int(rng.integers(2))

        # 函数内依次：抽平面、解平面带时窗、搜索首访、扩展剩余34层。
        diagnostics = {}
        routes = continuous_beam_run(
            PARAMETERS,
            states,
            width,
            seed,
            time_penalty,
            max_revolutions,
            mode=time_mode,
            diagnostics=diagnostics,
        )

        # 每轮保存全部完整链。优化器直接读取这些文件，不需要预载转换包。
        filename = OUTPUT_DIRECTORY / f"run_{run_index + 1:06d}.jsonl.gz"
        with gzip.open(filename, "wt", encoding="utf-8") as output:
            for route in routes:
                output.write(json.dumps(route.to_dict(), ensure_ascii=False) + "\n")
        best = min((route.cost_km_s for route in routes), default=None)
        print(
            f"平面 {run_index+1}/{PLANE_COUNT}，完整链 {len(routes)}，"
            f"最小Δv {best} km/s，用时 {time.perf_counter()-started:.1f} s"
        )


if __name__ == "__main__":
    # Numba 负责束扩展并行，单个线性代数调用使用一条 BLAS 线程。
    with threadpool_limits(limits=1):
        main()
