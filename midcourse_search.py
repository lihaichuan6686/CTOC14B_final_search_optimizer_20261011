"""围绕优秀载体轨迹，重新搜索35目标顺序，直接比较两体直接弧和中途脉冲弧。

这是优化末级的重搜，不替代 search.py 的广泛平面搜索。
到达网格、零点补点、连续细化、状态去重和父解前缀保留均来自原最终版本。
"""

import copy
import numpy as np
from scipy.optimize import brentq
from orbit_dynamics import MU, target_state
from impulse_nodes import trajectory, branch_near
from midcourse_transfers import batch, first_seeds, next_cost, schedule_score, elliptic_coast


def search_midcourse_routes(base, states, width, final_count, grid_step, free_midcourse):
    visits = np.flatnonzero(np.array(base["node_ids"]) >= 0)
    ids = np.array(base["node_ids"])[visits]
    positions_base = np.array(base["positions"])
    departing, arriving, _, pulses, _, _ = trajectory(base)
    normals = np.cross(positions_base[:-1], departing)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    normal = normals.mean(axis=0)
    normal /= np.linalg.norm(normal)
    branches = np.array(
        [
            [major, rev, side]
            for rev in range(3)
            for major in range(2)
            for side in range(1 if rev == 0 else 2)
        ],
        dtype=np.int64,
    )

    # 在载体平均平面±15°范围取网格，并在跨面根附近补点。
    grids = []
    for target in ids:
        grid = np.arange(240.0, 864000.0 - 180.0, grid_step)
        chosen = []
        for epoch in grid:
            position = target_state(states, int(target), epoch)[0]
            if abs(position @ normal) / np.linalg.norm(position) < np.sin(np.pi / 12):
                chosen.append(float(epoch))
        probe = np.r_[grid, 864000.0 - 180.0]
        residual = [float(target_state(states, int(target), epoch)[0] @ normal) for epoch in probe]
        for index in range(len(probe) - 1):
            if residual[index] * residual[index + 1] >= 0:
                continue
            center = brentq(
                lambda epoch: float(target_state(states, int(target), epoch)[0] @ normal),
                probe[index],
                probe[index + 1],
                xtol=0.05,
            )
            for delta in (-grid_step / 2, 0.0, grid_step / 2):
                epoch = center + delta
                if not 180.0 < epoch < 864000.0 - 180.0:
                    continue
                position = target_state(states, int(target), epoch)[0]
                if abs(position @ normal) / np.linalg.norm(position) < np.sin(np.pi / 12):
                    chosen.append(float(epoch))
        grids.append(np.unique(chosen))
    counts = np.array([len(grid) for grid in grids], dtype=np.int64)
    times = np.zeros((35, int(max(counts))))
    positions = np.zeros((*times.shape, 3))
    for target_index, grid in enumerate(grids):
        times[target_index, : len(grid)] = grid
        for index, epoch in enumerate(grid):
            positions[target_index, index] = target_state(states, int(ids[target_index]), epoch)[0]

    # 首访允许重选LEO相位和附近倾角；原父解前缀单独保留一个名额。
    beam = []
    launches = first_seeds(times, positions, counts, normal, 864000.0, branches, 12, 12.0)
    for row in launches:
        target, sample, branch = int(row[0]), int(row[1]), int(row[3])
        segment = dict(
            target=int(ids[target]),
            position=positions[target, sample].tolist(),
            time_s=float(row[2]),
            branch=branches[branch].tolist(),
            midcourse=False,
        )
        beam.append(
            dict(
                cost=float(row[4]),
                mask=1 << target,
                t=float(row[2]),
                r=positions[target, sample],
                v=row[5:8].copy(),
                order=[target],
                times=[float(row[2])],
                segments=[segment],
                initial=row[8:14].tolist(),
            )
        )
    beam.append(incumbent_prefix(base, visits, arriving, pulses, 1))

    for depth in range(1, 35):
        direct, midcourse, direct_counts, midcourse_counts = batch(
            np.array([item["r"] for item in beam]),
            np.array([item["v"] for item in beam]),
            np.array([item["t"] for item in beam]),
            np.array([item["mask"] for item in beam], dtype=np.int64),
            depth,
            times,
            positions,
            counts,
            branches,
            0.0,
            864000.0,
            12.0,
            True,
            0.0,
            0.0,
            states,
            ids,
            normal,
            1800.0,
            72 * 3600.0,
            free_midcourse,
        )
        candidates = []
        for parent_index, parent in enumerate(beam):
            rows = list(direct[parent_index, : direct_counts[parent_index]])
            rows += list(midcourse[parent_index, : midcourse_counts[parent_index]])
            for row in rows:
                target, branch = int(row[0]), int(row[3])
                cost = parent["cost"] + row[4]
                position = target_state(states, int(ids[target]), float(row[2]))[0]
                segment = dict(
                    target=int(ids[target]),
                    position=position.tolist(),
                    time_s=float(row[2]),
                    branch=branches[branch].tolist(),
                    midcourse=len(row) > 8,
                )
                if len(row) > 8:
                    segment.update(
                        mid_time_s=float(row[8]),
                        mid_position=row[9:12].tolist(),
                        coast_velocity=row[12:15].tolist(),
                    )
                child = dict(
                    cost=float(cost),
                    mask=parent["mask"] | (1 << target),
                    t=float(row[2]),
                    r=position,
                    v=row[5:8].copy(),
                    order=parent["order"] + [target],
                    times=parent["times"] + [float(row[2])],
                    initial=parent["initial"],
                    segments=parent["segments"] + [segment],
                )
                if "prefix_seed" in parent:
                    child["prefix_seed"] = parent["prefix_seed"]
                rank = cost + (
                    0 if depth == 34 else schedule_score(row[2], 34 - depth, 0.0, 864000.0, 12.0)
                )
                candidates.append((float(rank), child))

        if depth < 34:
            shortlist = []
            seen = set()
            for rank, child in sorted(candidates, key=lambda item: item[0]):
                key = (
                    child["mask"],
                    child["order"][-1],
                    int(child["t"] // 1800),
                    tuple(np.rint(child["v"] / 0.1).astype(int)),
                )
                if key in seen:
                    continue
                seen.add(key)
                shortlist.append((rank, child))
                if len(shortlist) >= width * 3:
                    break
            if shortlist:
                children = [child for _, child in shortlist]
                ahead = next_cost(
                    np.array([p["r"] for p in children]),
                    np.array([p["v"] for p in children]),
                    np.array([p["t"] for p in children]),
                    np.array([p["mask"] for p in children], dtype=np.int64),
                    times,
                    positions,
                    counts,
                    864000.0,
                )
                candidates = [
                    (rank + 0.6 * float(ahead[index]), child)
                    for index, (rank, child) in enumerate(shortlist)
                ]

        beam = []
        seen = set()
        order_count = {}
        for rank, child in sorted(candidates, key=lambda item: item[0]):
            key = (
                child["mask"],
                child["order"][-1],
                int(child["t"] // 1800),
                tuple(np.rint(child["v"] / 0.1).astype(int)),
            )
            order = tuple(child["order"])
            cap = 24 if depth == 1 else 6
            if key in seen or order_count.get(order, 0) >= cap:
                continue
            seen.add(key)
            order_count[order] = order_count.get(order, 0) + 1
            beam.append(child)
            if len(beam) >= width:
                break
        beam = beam[: width - 1] + [incumbent_prefix(base, visits, arriving, pulses, depth + 1)]
        print(f"中途脉冲重搜：第{depth+1}层，保留{len(beam)}链")

    selected = []
    order_count = {}
    last_count = {}
    seen = set()
    for child in sorted(beam, key=lambda item: item["cost"]):
        order = tuple(child["order"])
        if order_count.get(order, 0) >= 2 or last_count.get(order[-1], 0) >= 12:
            continue
        try:
            candidate = build_trajectory(base, child)
        except ValueError:
            continue
        signature = (
            tuple(candidate["initial"]),
            tuple(candidate["times"]),
            tuple(candidate["node_ids"]),
            tuple(map(tuple, candidate["positions"])),
            tuple(map(tuple, candidate["branches"])),
        )
        if signature in seen:
            continue
        seen.add(signature)
        order_count[order] = order_count.get(order, 0) + 1
        last_count[order[-1]] = last_count.get(order[-1], 0) + 1
        # 保留前缀用于搜索引导，但不重复联调完整的父解本身。
        if not child.get("guide", False):
            selected.append(candidate)
        if sum(order_count.values()) >= final_count:
            break
    return selected


def incumbent_prefix(base, visits, arriving, pulses, count):
    node = len(base["times"]) - 1 if count == len(visits) else int(visits[count - 1])
    prefix = copy.deepcopy(base)
    for key in ("positions", "times", "node_ids"):
        prefix[key] = prefix[key][: node + 1]
    prefix["branches"] = prefix["branches"][:node]
    prefix.pop("encounter_offsets_km", None)
    cost = float(np.linalg.norm(pulses[:node], axis=1).sum())
    prefix["cost"] = cost
    return dict(
        cost=cost,
        mask=(1 << count) - 1,
        t=float(prefix["times"][-1]),
        r=np.array(prefix["positions"][-1]),
        v=arriving[node - 1].copy(),
        order=list(range(count)),
        times=[prefix["times"][i] for i in visits[:count]],
        segments=[],
        initial=prefix["initial"],
        prefix_seed=prefix,
        guide=True,
    )


def build_trajectory(base, child):
    if "prefix_seed" in child:
        candidate = copy.deepcopy(child["prefix_seed"])
    else:
        candidate = copy.deepcopy(base)
        initial = child["initial"]
        candidate.update(
            initial=initial, positions=[initial[:3]], times=[0.0], node_ids=[-1], branches=[]
        )
    candidate.pop("encounter_offsets_km", None)
    for segment in child["segments"]:
        if segment["midcourse"]:
            r0 = np.array(candidate["positions"][-1])
            v0 = np.array(segment["coast_velocity"])
            t0 = candidate["times"][-1]
            duration = segment["mid_time_s"] - t0
            semimajor = 1 / (2 / np.linalg.norm(r0) - v0 @ v0 / MU)
            period = 2 * np.pi * np.sqrt(semimajor**3 / MU)
            pieces = max(1, int(np.ceil(duration / (0.2 * period))))
            previous = 0.0
            for index in range(1, pieces + 1):
                elapsed = duration * index / pieces
                position = (
                    np.array(segment["mid_position"])
                    if index == pieces
                    else elliptic_coast(r0, v0, elapsed)[0]
                )
                expected = v0 if index == 1 else elliptic_coast(r0, v0, previous)[1]
                branch, _, _ = branch_near(
                    np.array(candidate["positions"][-1]), position, elapsed - previous, expected
                )
                candidate["positions"].append(position.tolist())
                candidate["times"].append(t0 + elapsed)
                candidate["node_ids"].append(-1)
                candidate["branches"].append(branch)
                previous = elapsed
        candidate["positions"].append(segment["position"])
        candidate["times"].append(segment["time_s"])
        candidate["node_ids"].append(segment["target"])
        candidate["branches"].append(segment["branch"])
    return candidate
