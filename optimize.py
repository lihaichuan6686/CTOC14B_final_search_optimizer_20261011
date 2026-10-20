"""运行本文件：筛选初解 → 二体联调 → J₂ SCP → 换序修复 → 脉冲增删 → 精英重搜。

每次提交给进程池的都是一个具体数学步骤，轨迹和星历作为参数传递。
没有字符串任务派发、隐藏星历、任务文件中转或自动接续。
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import time
import numpy as np
from numba import set_num_threads
from threadpoolctl import threadpool_limits
from candidate_selection import (
    read_search_routes,
    diverse_candidates,
    select_repairs,
    save_trajectory,
)
from two_body_optimization import joint_polish, conditioned_nodes, node_branch_dp, compact
from trajectory_evaluation import evaluate, rotate
from impulse_nodes import regularize, coarsen, add_zero_nodes, ensure_offsets
from j2_optimization import refine
from primer_insertion import scan
from target_relocation import release_target, relocate_target, rotate_and_reassign
from midcourse_search import search_midcourse_routes

# -------------------- 直接修改这里 --------------------
PROJECT = Path(__file__).resolve().parent
EPHEMERIS_FILE = PROJECT / "data/states.npy"
POLE_FILE = PROJECT / "data/calibration.json"
SEARCH_DIRECTORY = PROJECT / "results/search"  # 优化哪个搜索目录，一眼在这里看到
TRAJECTORY_FILES = []  # 也可填 [Path('某条完整轨迹.json')]
OUTPUT_DIRECTORY = PROJECT / "results/optimization"
WORKERS = 4
BEAM_THREADS = 8
RANDOM_SEED = 20261011
INITIAL_COUNT = 8
ACTIVE_PARENTS = 4
EXPLORATION_FRACTION = 0.25
RELOCATION_ROUNDS = 2
OPPORTUNITIES_PER_TARGET = 3
REPAIRS_PER_PARENT = 12
JOINT_SECONDS = 90
JOINT_ITERATIONS = 1200
J2_SECONDS = 120
SCP_ITERATIONS = 1200
RELEASE_SECONDS = 12
RELEASE_ITERATIONS = 170
REPAIR_SECONDS = 18
POLISH_SECONDS = 80
TOPOLOGY_SECONDS = 90
PRIMER_SAMPLES = 61
ROTATION_PARENTS = 1
ROTATION_ANGLES_DEG = (-3.0, 3.0)
ROTATION_REPAIR_SECONDS = 30
ROTATION_POLISH_SECONDS = 55
ROTATION_ITERATIONS = 850
DEEP_BEAM_PARENTS = 1
DEEP_BEAM_WIDTH = 96
DEEP_BEAM_FINAL_COUNT = 8
DEEP_BEAM_GRID_SECONDS = 1800.0


def set_worker_threads():
    # 只控制真正的数值并行：每个工作进程一条Numba/BLAS线程。
    set_num_threads(1)
    threadpool_limits(limits=1)


def initial_refinement(
    trajectory, states, pole, joint_seconds, joint_iterations, j2_seconds, scp_iterations
):
    """单条初解的连续优化；这里直接列出原优化器的三个二体阶段。"""
    set_num_threads(1)
    # 每个进程一个 BLAS 线程，避免4进程再各开几十个线性代数线程。
    with threadpool_limits(limits=1):
        current = copy.deepcopy(trajectory)
        if current["model"] == "two_body":
            best = None
            report = evaluate(current, states, pole)
            if report["complete"] and not report["errors"]:
                current["cost"] = report["total_dv_km_s"]
                best = copy.deepcopy(current)
            for phase in range(3):
                candidate = copy.deepcopy(current)
                if phase == 1:
                    candidate = conditioned_nodes(candidate, max_angle=105.0)
                if phase == 2:
                    candidate = compact(node_branch_dp(candidate, 2), threshold=0.0001)
                candidate = joint_polish(
                    candidate,
                    states,
                    joint_seconds / 60,
                    None,
                    smoothings=(0.01, 0.001, 0.00003),
                    maxiter=joint_iterations,
                    min_gap_hours=10 / 3600,
                )
                report = evaluate(candidate, states, pole)
                if report["complete"] and not report["errors"]:
                    candidate["cost"] = report["total_dv_km_s"]
                    current = candidate
                    if best is None or candidate["cost"] < best["cost"]:
                        best = copy.deepcopy(candidate)
            if best is None:
                raise ValueError("三阶段二体优化未得到完整可行链")
            current = conditioned_nodes(best, max_angle=105.0)

        # 行向量右乘 pole：ICRF → 固定极轴系。目标星历和航天器必须一起旋转。
        current = rotate(current, pole)
        current.update(model="j2", frame="fixed_pole_surrogate")
        current = regularize(current)
        pole_states = np.concatenate((states[:, :, :3] @ pole, states[:, :, 3:] @ pole), axis=2)
        ensure_offsets(current, pole_states)
        current = refine(current, pole_states, j2_seconds / 60, max_iterations=scp_iterations)
        current = rotate(current, pole.T)
        current.update(model="j2", frame="ICRF")
        return current


def insert_impulses(trajectory, states, seconds, iterations, samples):
    """全链 primer 找峰，在互不相邻的最多4条弧上插零脉冲节点，再联调。"""
    started = time.monotonic()
    peaks, report = scan(trajectory, states, samples)
    selected = []
    for peak in peaks:
        if peak["primer_norm"] < 1.0005:
            break
        if all(abs(peak["arc"] - other["arc"]) > 1 for other in selected):
            selected.append(peak)
        if len(selected) == 4:
            break
    candidate = add_zero_nodes(trajectory, selected) if selected else copy.deepcopy(trajectory)
    remaining = max(0.006, seconds - (time.monotonic() - started))
    return refine(candidate, states, remaining / 60, max_iterations=iterations)


def remove_weak_impulses(trajectory, states, seconds, iterations):
    started = time.monotonic()
    candidate = coarsen(trajectory)
    remaining = max(0.006, seconds - (time.monotonic() - started))
    return refine(candidate, states, remaining / 60, max_iterations=iterations)


def admit_candidate(candidate, archive, states, pole, output_directory):
    """仅把完整、连续传播通过的轨迹纳入候选集；费用使用实际脉冲总和。"""
    if candidate["frame"] == "fixed_pole_surrogate":
        candidate = rotate(candidate, pole.T)
        candidate["frame"] = "ICRF"
    report = evaluate(candidate, states, pole, strict=True)
    if not report["complete"] or report["errors"]:
        print("候选未通过物理复验：", report["errors"])
        return
    candidate["cost"] = report["total_dv_km_s"]
    # 避免同一物理轨迹反复占据探索名额。
    for previous in archive:
        if all(
            candidate[key] == previous[key]
            for key in ("initial", "positions", "times", "node_ids", "branches")
        ):
            return
    archive.append(candidate)
    save_trajectory(output_directory / f"candidate_{len(archive):04d}.json", candidate)
    if candidate["cost"] <= min(item["cost"] for item in archive):
        save_trajectory(output_directory / "BEST_J2.json", candidate)
        (output_directory / "BEST_J2_validation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f'有效候选 {len(archive)}：Δv={candidate["cost"]:.9f} km/s')


def main():
    states = np.load(EPHEMERIS_FILE)  # ICRF，35×14401×6，km和km/s
    pole = np.array(json.loads(POLE_FILE.read_text(encoding="utf-8"))["pole_frame_to_icrf"])
    pole_states = np.concatenate((states[:, :, :3] @ pole, states[:, :, 3:] @ pole), axis=2)
    rng = np.random.default_rng(RANDOM_SEED)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # 1. 选择初解。给定 TRAJECTORY_FILES 时直接读；否则读取上方指定的搜索目录。
    if TRAJECTORY_FILES:
        candidates = [json.loads(path.read_text(encoding="utf-8")) for path in TRAJECTORY_FILES]
    else:
        candidates = read_search_routes(SEARCH_DIRECTORY, states, max(32, INITIAL_COUNT * 8), rng)
    seeds = diverse_candidates(candidates, INITIAL_COUNT, rng, EXPLORATION_FRACTION)
    archive = []
    for seed in seeds:
        if seed["model"] == "j2":
            admit_candidate(seed, archive, states, pole, OUTPUT_DIRECTORY)

    # 不同候选并行；同一候选的数学阶段依次执行。线性代数线程限制在计算上下文内。
    with threadpool_limits(limits=1), ProcessPoolExecutor(
        max_workers=WORKERS, initializer=set_worker_threads
    ) as pool:
        # 2. 二体联合优化 → 增自由节点 → 分支动态规划/压缩 → J₂联合优化。
        futures = [
            pool.submit(
                initial_refinement,
                seed,
                states,
                pole,
                JOINT_SECONDS,
                JOINT_ITERATIONS,
                J2_SECONDS,
                SCP_ITERATIONS,
            )
            for seed in seeds
        ]
        for future in futures:
            try:
                admit_candidate(future.result(), archive, states, pole, OUTPUT_DIRECTORY)
            except (ValueError, np.linalg.LinAlgError) as error:
                print("初解的数值优化失败：", error)

        # 3. 逐个释放35个目标，找替代时刻，择机会修复；完整解才进入下一轮。
        for round_index in range(RELOCATION_ROUNDS):
            parents = diverse_candidates(archive, ACTIVE_PARENTS, rng, EXPLORATION_FRACTION)
            repair_futures = []
            for parent in parents:
                carrier = regularize(rotate(parent, pole))
                carrier["frame"] = "fixed_pole_surrogate"
                release_futures = [
                    pool.submit(
                        release_target,
                        carrier,
                        pole_states,
                        target,
                        RELEASE_SECONDS,
                        RELEASE_ITERATIONS,
                        OPPORTUNITIES_PER_TARGET,
                    )
                    for target in range(35)
                ]
                releases = []
                for future in release_futures:
                    try:
                        releases.append(future.result())
                    except (ValueError, np.linalg.LinAlgError) as error:
                        print("目标释放不可行：", error)
                choices = select_repairs(releases, REPAIRS_PER_PARENT, rng)
                for released, target, gain, rank, encounter in choices:
                    repair_futures.append(
                        pool.submit(
                            relocate_target,
                            released,
                            pole_states,
                            target,
                            encounter,
                            REPAIR_SECONDS,
                            POLISH_SECONDS,
                            SCP_ITERATIONS,
                        )
                    )
            for future in repair_futures:
                try:
                    admit_candidate(future.result(), archive, states, pole, OUTPUT_DIRECTORY)
                except (ValueError, np.linalg.LinAlgError) as error:
                    print("目标换位未形成可行解：", error)

            # 4. 对有效父解分别尝试增加和删除脉冲节点。
            parents = diverse_candidates(archive, ACTIVE_PARENTS, rng, EXPLORATION_FRACTION)
            topology_futures = []
            for parent in parents:
                carrier = regularize(rotate(parent, pole))
                carrier["frame"] = "fixed_pole_surrogate"
                topology_futures.append(
                    pool.submit(
                        insert_impulses,
                        carrier,
                        pole_states,
                        TOPOLOGY_SECONDS,
                        SCP_ITERATIONS,
                        PRIMER_SAMPLES,
                    )
                )
                topology_futures.append(
                    pool.submit(
                        remove_weak_impulses, carrier, pole_states, TOPOLOGY_SECONDS, SCP_ITERATIONS
                    )
                )
            for future in topology_futures:
                try:
                    admit_candidate(future.result(), archive, states, pole, OUTPUT_DIRECTORY)
                except (ValueError, np.linalg.LinAlgError) as error:
                    print("脉冲结构候选不可行：", error)
            print(f"完成目标换位与脉冲调整第 {round_index+1} 轮")

        # 5. 少量精英整体旋转，重新分配所有目标，跳出原访问顺序。
        elites = diverse_candidates(archive, ACTIVE_PARENTS, rng, EXPLORATION_FRACTION)
        rotation_futures = []
        for parent in elites[:ROTATION_PARENTS]:
            carrier = regularize(rotate(parent, pole))
            carrier["frame"] = "fixed_pole_surrogate"
            for axis in ("r", "z"):
                for angle in ROTATION_ANGLES_DEG:
                    rotation_futures.append(
                        pool.submit(
                            rotate_and_reassign,
                            carrier,
                            pole_states,
                            angle,
                            axis,
                            ROTATION_REPAIR_SECONDS,
                            ROTATION_POLISH_SECONDS,
                            ROTATION_ITERATIONS,
                        )
                    )
        for future in rotation_futures:
            try:
                admit_candidate(future.result(), archive, states, pole, OUTPUT_DIRECTORY)
            except (ValueError, np.linalg.LinAlgError) as error:
                print("整体重分配未恢复可行性：", error)

    # 6. 进程池结束后，全部数值线程用于精英的中途脉冲束搜索。
    set_num_threads(BEAM_THREADS)
    elites = diverse_candidates(archive, ACTIVE_PARENTS, rng, EXPLORATION_FRACTION)
    with threadpool_limits(limits=1):
        for parent in elites[:DEEP_BEAM_PARENTS]:
            carrier = regularize(rotate(parent, pole))
            carrier["frame"] = "fixed_pole_surrogate"
            for free_midcourse in (False, True):
                proposals = search_midcourse_routes(
                    carrier,
                    pole_states,
                    DEEP_BEAM_WIDTH,
                    DEEP_BEAM_FINAL_COUNT,
                    DEEP_BEAM_GRID_SECONDS,
                    free_midcourse,
                )
                for proposal in proposals:
                    try:
                        candidate = regularize(proposal)
                        candidate = refine(
                            candidate,
                            pole_states,
                            POLISH_SECONDS / 60,
                            max_iterations=SCP_ITERATIONS,
                        )
                        admit_candidate(candidate, archive, states, pole, OUTPUT_DIRECTORY)
                    except (ValueError, np.linalg.LinAlgError) as error:
                        print("中途脉冲候选不可行：", error)
    if archive:
        print("本次最好Δv：", min(item["cost"] for item in archive), "km/s")
    else:
        print("本次预算内没有通过完整J₂复验的解，请查看上面的数值失败原因。")


if __name__ == "__main__":
    main()
