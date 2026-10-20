"""读取搜索链，按费用、访问顺序和时窗保留不同候选。

保留低费用池和随机蓄水池，避免只把最低粗搜费用送去优化。
这是候选选择算法，不是文件格式兼容层。只读本项目 jsonl.gz 和完整轨迹 JSON。
"""

import copy
import gzip
import heapq
import json
import math
from orbit_dynamics import target_state


def read_search_routes(directory, states, pool_size, rng):
    cheap = []
    reservoir = []
    count = 0
    for filename in sorted(directory.glob("run_*.jsonl.gz")):
        with gzip.open(filename, "rt", encoding="utf-8") as stream:
            for line in stream:
                route = json.loads(line)
                cost = route["cost_km_s"]
                priority = float(rng.random())
                # 堆顶存最差候选；只把入池的记录展开成完整节点轨迹，控制内存。
                take_cheap = len(cheap) < pool_size or cost < -cheap[0][0]
                take_random = len(reservoir) < pool_size or priority < -reservoir[0][0]
                if take_cheap or take_random:
                    initial = route["initial_state_km_km_s"]
                    positions = [initial[:3]]
                    for target, epoch in zip(route["order"], route["times_s"]):
                        positions.append(target_state(states, target, epoch)[0].tolist())
                    trajectory = dict(
                        initial=initial,
                        positions=positions,
                        times=[0.0, *route["times_s"]],
                        node_ids=[-1, *route["order"]],
                        branches=route["branches"],
                        cost=cost,
                        model="two_body",
                        frame="ICRF",
                        origin=f"{filename.name}:{count}",
                    )
                    for heap, rank, take in (
                        (cheap, cost, take_cheap),
                        (reservoir, priority, take_random),
                    ):
                        if take:
                            item = (-rank, count, trajectory)
                            if len(heap) < pool_size:
                                heapq.heappush(heap, item)
                            else:
                                heapq.heapreplace(heap, item)
                count += 1
    unique = {index: trajectory for _, index, trajectory in cheap + reservoir}
    print(f"读取 {count} 条搜索链，保留候选池 {len(unique)} 条")
    return list(unique.values())


def family(trajectory):
    # 相同顺序但跨越不同4小时访问窗的轨迹，也视为不同家族。
    visits = [
        (target, int(epoch // 14400))
        for target, epoch in zip(trajectory["node_ids"], trajectory["times"])
        if target >= 0
    ]
    return tuple(visits)


def diverse_candidates(candidates, count, rng, exploration_fraction=0.25):
    ordered = sorted(candidates, key=lambda trajectory: trajectory["cost"])
    representatives = []
    seen = set()
    for trajectory in ordered:
        key = family(trajectory)
        if key not in seen:
            representatives.append(trajectory)
            seen.add(key)
    limit = min(count, len(representatives))
    quality_count = max(1, math.ceil(limit * (1 - exploration_fraction))) if limit else 0
    chosen = representatives[:quality_count]
    rest = representatives[quality_count:]
    rng.shuffle(rest)
    chosen.extend(rest[: limit - len(chosen)])
    for trajectory in ordered:
        if len(chosen) >= count:
            break
        if not any(trajectory is item for item in chosen):
            chosen.append(trajectory)
    return chosen


def select_repairs(releases, limit, rng):
    # 三路名额：释放收益、几何距离、随机探索。每个目标最多占两个名额。
    opportunities = []
    for released, target, gain, choices in releases:
        for rank, encounter in enumerate(choices):
            opportunities.append((released, target, gain, rank, encounter))
    by_gain = sorted(opportunities, key=lambda item: (-item[2], item[3], item[1]))
    by_distance = sorted(opportunities, key=lambda item: (item[4]["distance_km"], item[1]))
    random_order = list(opportunities)
    rng.shuffle(random_order)
    chosen = []
    seen = set()
    target_count = {}
    quota = max(1, limit // 3)
    for stream, count in ((by_gain, quota), (by_distance, quota), (random_order, limit)):
        added = 0
        for item in stream:
            target, rank = item[1], item[3]
            if (target, rank) in seen or target_count.get(target, 0) >= 2:
                continue
            chosen.append(item)
            seen.add((target, rank))
            target_count[target] = target_count.get(target, 0) + 1
            added += 1
            if len(chosen) >= limit:
                return chosen
            if added >= count:
                break
    return chosen


def save_trajectory(filename, trajectory):
    # 除去迭代历史，输出直接保留计算所需的物理量。
    result = copy.deepcopy(trajectory)
    for key in list(result):
        if key.endswith("_history"):
            result.pop(key)
    result.update(
        units="km,s,km/s",
        epoch_utc="2035-01-01T12:00:00Z",
        mission_end_s=864000.0,
        format_version=1,
    )
    filename.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
