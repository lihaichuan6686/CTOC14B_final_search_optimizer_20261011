"""平面窗口束搜索：首访采样 → 逐层 Lambert 扩展 → 前瞻评分 → 保留多样候选。

states[目标, 时刻, 分量] 为 ICRF 星历；每 60 s 一行，共 35×14401×6。
扩展记录每行 13 列：[累计Δv, 到达时刻, 父链序号, 目标号, 分支号,
到达速度(3), 到达位置(3), 历史最小半径, 本段Δv]。
窗口每行 [起点秒, 终点秒, 原窗口号]；访问编号 0–34。
parameters 是显式传入的数值参数，不会在导入时修改或读取配置文件。
"""

import math
import time
import numpy as np
from numba import njit, prange
from typing import NamedTuple, Any
from dataclasses import dataclass, field, asdict
from lambert import MU, RE, norm, lambert, target_state, arc_min_radius
from plane_windows import build_windows, sample_plane, window_cells

END = 864000.0
RMIN = RE + 205.0


class SearchParameters(NamedTuple):
    expansion_minima_per_cell: int = 1
    expansion_time_points: int = 1
    exploration_fraction: float = 0.3
    first_altitude_samples: int = 2
    first_cell_max_h: float = 6.0
    first_coordinate_passes: int = 1
    first_golden_iterations: int = 8
    first_keep_per_cell_branch: int = 4
    first_max_arrival_h: float = 72.0
    first_phase_samples: int = 64
    first_refine_per_target: int = 4
    first_time_samples_per_cell: int = 3
    golden_iterations: int = 5
    initial_altitude_max_km: float = 609.9
    initial_altitude_min_km: float = 590.1
    local_refinements_per_branch: int = 1
    lookahead_cell_max_h: float = 6.0
    lookahead_pool_multiplier: int = 2
    lookahead_weight: float = 0.6
    max_transfer_h: float = 72.0
    min_target_gap_s: float = 30.0
    plane_normal_z_max: float = 1.0
    plane_normal_z_min: float = -1.0
    plane_sector_count: int = 1
    plane_sector_index: int = 1
    plane_tolerance_deg: float = 6.0
    representatives_per_branch: int = 3
    search_cost_ceiling_km_s: float = 24.0
    search_profile: str = "local_plane_fast"
    state_bucket_cap: int = 8
    time_distance_s: float = 600.0
    velocity_distance_km_s: float = 0.06
    window_cell_max_h: float = 6.0
    window_edge_tolerance_s: float = 0.05


@dataclass
class Route:
    order: list[int]
    times_s: list[float]
    branches: list[list[int]]
    initial_state_km_km_s: list[float]
    cost_km_s: float
    arrival_velocity_km_s: list[float]
    minimum_radius_km: float
    origin: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            order_1based=[target + 1 for target in self.order],
            target_count=len(set(self.order)),
            end_time_days=self.times_s[-1] / 86400.0 if self.times_s else None,
            units="km,s,km/s",
            status="two_body_screening_only",
        )
        return value


@njit(cache=True)
def time_score(parameters, epoch, remaining, penalty, mode):
    if mode == 0:
        return penalty * epoch / 86400.0
    pressure = remaining * 5.0 * 3600 / max(END - epoch, float(parameters.min_target_gap_s))
    return penalty * max(0.0, pressure - 0.7) ** 2 + 0.03 * penalty * epoch / 86400.0


@njit(cache=True)
def leg(parameters, states, k, t, r, v, dt, branch, normal, sine):
    """Delta-v, endpoint, arrival velocity, minimum radius, actual Lambert calls."""
    if dt < float(parameters.min_target_gap_s) - 1e-08 or t + dt > END + 1e-08:
        return (1e100, np.zeros(3), np.zeros(3), 0.0, 0)
    r1, _ = target_state(states, k, t + dt)
    if abs(np.dot(r1, normal)) > (sine + 1e-12) * norm(r1):
        return (1e100, r1, np.zeros(3), 0.0, 0)
    va, vz = lambert(r, r1, dt, branch[0], branch[1], branch[2])
    if not np.isfinite(va[0]):
        return (1e100, r1, vz, 0.0, 1)
    rad = arc_min_radius(r, va, r1, vz, dt)
    if rad < RMIN:
        return (1e100, r1, vz, rad, 1)
    return (norm(va - v), r1, vz, rad, 1)


@njit(cache=True)
def state_representatives(values, durations, velocities, count):
    selected = np.full(count, -1, np.int64)
    valid = np.flatnonzero(values < 1e90)
    if not len(valid):
        return selected
    selected[0] = valid[np.argmin(values[valid])]
    used = np.zeros(len(values), np.bool_)
    used[selected[0]] = True
    nearest = np.full(len(values), 1e100)
    for n in range(1, count):
        best, best_merit = (-1, -1e100)
        x = selected[n - 1]
        for j in valid:
            if used[j]:
                continue
            d = ((durations[j] - durations[x]) / 1800.0) ** 2
            velocity_distance = 0.0
            for axis in range(3):
                delta = velocities[j, axis] - velocities[x, axis]
                velocity_distance += delta * delta
            d += velocity_distance / 0.15**2
            nearest[j] = min(nearest[j], d)
            distance = nearest[j]
            merit = math.log1p(distance) - 0.15 * (values[j] - values[selected[0]])
            if merit > best_merit:
                best, best_merit = (j, merit)
        if best < 0:
            break
        selected[n] = best
        used[best] = True
    return selected


@njit(cache=True)
def _cell_probe(
    parameters,
    states,
    k,
    t,
    r,
    v,
    lo,
    hi,
    branch,
    normal,
    sine,
    remaining,
    penalty,
    mode,
    probe_limit=3,
):
    """Midpoint first; at most two interior fallback probes. Rollout uses only midpoint."""
    best, best_dt, best_v, best_r = (1e100, (lo + hi) * 0.5 - t, np.zeros(3), 0.0)
    calls = 0
    fractions = np.array([0.5, 0.25, 0.75])
    for p in range(probe_limit):
        epoch = lo + (hi - lo) * fractions[p]
        dv, _, vz, rad, used = leg(parameters, states, k, t, r, v, epoch - t, branch, normal, sine)
        calls += used
        if dv < 1e90:
            value = dv + time_score(parameters, epoch, remaining, penalty, mode)
            if value < best:
                best, best_dt, best_v, best_r = (value, epoch - t, vz, rad)
            if p == 0:
                break
    return (best, best_dt, best_v, best_r, calls)


@njit(cache=True)
def local_minima_indices(values, limit):
    """Rank separate sampled valleys; include one-sided boundary minima."""
    result = np.full(limit, -1, np.int64)
    count = 0
    for j in np.argsort(values):
        if values[j] >= 1e90:
            break
        if j > 0 and values[j] > values[j - 1]:
            continue
        if j + 1 < len(values) and values[j] > values[j + 1]:
            continue
        if j > 0 and values[j] == values[j - 1]:
            continue
        result[count] = j
        count += 1
        if count == limit:
            break
    return result[:count]


@njit(cache=True)
def _refine_bracket(
    parameters,
    states,
    k,
    t,
    r,
    v,
    a,
    b,
    branch,
    normal,
    sine,
    remaining,
    penalty,
    mode,
    best,
    best_dt,
    best_v,
):
    calls = 0
    if b - a < 1e-07:
        return (best, best_dt, best_v, calls)
    x, y = (b - 0.61803398875 * (b - a), a + 0.61803398875 * (b - a))
    dx, _, vx, _, used = leg(parameters, states, k, t, r, v, x, branch, normal, sine)
    calls += used
    cx = dx + time_score(parameters, t + x, remaining, penalty, mode)
    dy, _, vy, _, used = leg(parameters, states, k, t, r, v, y, branch, normal, sine)
    calls += used
    cy = dy + time_score(parameters, t + y, remaining, penalty, mode)
    for iteration in range(int(parameters.golden_iterations) + 1):
        if cx < best:
            best, best_dt, best_v = (cx, x, vx)
        if cy < best:
            best, best_dt, best_v = (cy, y, vy)
        if iteration == int(parameters.golden_iterations):
            break
        if cx < cy:
            b, y, cy, vy = (y, x, cx, vx)
            x = b - 0.61803398875 * (b - a)
            dx, _, vx, _, used = leg(parameters, states, k, t, r, v, x, branch, normal, sine)
            calls += used
            cx = dx + time_score(parameters, t + x, remaining, penalty, mode)
        else:
            a, x, cx, vx = (x, y, cy, vy)
            y = a + 0.61803398875 * (b - a)
            dy, _, vy, _, used = leg(parameters, states, k, t, r, v, y, branch, normal, sine)
            calls += used
            cy = dy + time_score(parameters, t + y, remaining, penalty, mode)
    return (best, best_dt, best_v, calls)


@njit(cache=True)
def branch_samples(
    parameters, states, k, t, r, v, cells, branch, normal, sine, remaining, penalty, mode
):
    """Dense profiles refine valleys in EVERY cell before inter-window pruning."""
    capacity = (
        len(cells)
        * (int(parameters.expansion_time_points) + int(parameters.expansion_minima_per_cell))
        if int(parameters.expansion_time_points) > 1
        else len(cells)
    )
    vals = np.full(capacity, 1e100)
    durations = np.zeros(capacity)
    vv = np.zeros((capacity, 3))
    calls = 0
    cursor = 0
    for j in range(len(cells)):
        lo, hi = (cells[j, 0], cells[j, 1])
        if int(parameters.expansion_time_points) == 1:
            value, dt, vz, _, used = _cell_probe(
                parameters,
                states,
                k,
                t,
                r,
                v,
                lo,
                hi,
                branch,
                normal,
                sine,
                remaining,
                penalty,
                mode,
            )
            vals[cursor] = value
            durations[cursor] = dt
            vv[cursor] = vz
            calls += used
            cursor += 1
            continue
        margin = min(float(parameters.window_edge_tolerance_s) * 1.1, (hi - lo) * 0.25)
        grid = np.linspace(lo + margin, hi - margin, int(parameters.expansion_time_points)) - t
        offset = cursor
        for s in range(int(parameters.expansion_time_points)):
            dv, _, vz, _, used = leg(parameters, states, k, t, r, v, grid[s], branch, normal, sine)
            calls += used
            vals[cursor] = dv + time_score(parameters, t + grid[s], remaining, penalty, mode)
            durations[cursor] = grid[s]
            vv[cursor] = vz
            cursor += 1
        for s in local_minima_indices(
            vals[offset : offset + int(parameters.expansion_time_points)],
            int(parameters.expansion_minima_per_cell),
        ):
            a = lo - t if s == 0 else grid[s - 1]
            b = hi - t if s == int(parameters.expansion_time_points) - 1 else grid[s + 1]
            value, dt, vz, used = _refine_bracket(
                parameters,
                states,
                k,
                t,
                r,
                v,
                a,
                b,
                branch,
                normal,
                sine,
                remaining,
                penalty,
                mode,
                vals[offset + s],
                grid[s],
                vv[offset + s],
            )
            vals[cursor] = value
            durations[cursor] = dt
            vv[cursor] = vz
            calls += used
            cursor += 1
    return (vals[:cursor], durations[:cursor], vv[:cursor], calls)


@njit(cache=True, parallel=True)
def expand(
    parameters,
    states,
    masks,
    positions,
    velocities,
    epochs,
    costs,
    radii,
    windows,
    counts,
    normal,
    sine,
    branches,
    remaining,
    penalty,
    mode,
):
    per_parent = 35 * len(branches) * int(parameters.representatives_per_branch)
    out = np.full((len(masks) * per_parent, 13), 1e100)
    call_counts = np.zeros(len(masks), np.int64)
    cell_counts = np.zeros(len(masks), np.int64)
    for p in prange(len(masks)):
        cursor = p * per_parent
        latest = min(
            END - remaining * float(parameters.min_target_gap_s),
            epochs[p] + float(parameters.max_transfer_h) * 3600.0,
        )
        for k in range(35):
            if masks[p] & np.int64(1) << k:
                continue
            cells = window_cells(
                windows[k],
                counts[k],
                epochs[p] + float(parameters.min_target_gap_s),
                latest,
                float(parameters.window_cell_max_h) * 3600.0,
            )
            cell_counts[p] += len(cells)
            if not len(cells):
                continue
            for bi in range(len(branches)):
                vals, durations, vv, calls = branch_samples(
                    parameters,
                    states,
                    k,
                    epochs[p],
                    positions[p],
                    velocities[p],
                    cells,
                    branches[bi],
                    normal,
                    sine,
                    remaining,
                    penalty,
                    mode,
                )
                call_counts[p] += calls
                selected = state_representatives(
                    vals, durations, vv, int(parameters.representatives_per_branch)
                )
                for rank in range(int(parameters.representatives_per_branch)):
                    j = selected[rank]
                    if j < 0:
                        continue
                    dt = durations[j]
                    incumbent = vals[j]
                    if int(parameters.expansion_time_points) == 1 and rank < int(
                        parameters.local_refinements_per_branch
                    ):
                        a, b = (cells[j, 0] - epochs[p], cells[j, 1] - epochs[p])
                        if b - a > 1e-06:
                            x, y = (b - 0.61803398875 * (b - a), a + 0.61803398875 * (b - a))
                            dx, _, _, _, used = leg(
                                parameters,
                                states,
                                k,
                                epochs[p],
                                positions[p],
                                velocities[p],
                                x,
                                branches[bi],
                                normal,
                                sine,
                            )
                            call_counts[p] += used
                            cx = dx + time_score(
                                parameters, epochs[p] + x, remaining, penalty, mode
                            )
                            dy, _, _, _, used = leg(
                                parameters,
                                states,
                                k,
                                epochs[p],
                                positions[p],
                                velocities[p],
                                y,
                                branches[bi],
                                normal,
                                sine,
                            )
                            call_counts[p] += used
                            cy = dy + time_score(
                                parameters, epochs[p] + y, remaining, penalty, mode
                            )
                            for _ in range(int(parameters.golden_iterations)):
                                if cx < incumbent:
                                    dt, incumbent = (x, cx)
                                if cy < incumbent:
                                    dt, incumbent = (y, cy)
                                if cx < cy:
                                    b, y, cy = (y, x, cx)
                                    x = b - 0.61803398875 * (b - a)
                                    dx, _, _, _, used = leg(
                                        parameters,
                                        states,
                                        k,
                                        epochs[p],
                                        positions[p],
                                        velocities[p],
                                        x,
                                        branches[bi],
                                        normal,
                                        sine,
                                    )
                                    call_counts[p] += used
                                    cx = dx + time_score(
                                        parameters, epochs[p] + x, remaining, penalty, mode
                                    )
                                else:
                                    a, x, cx = (x, y, cy)
                                    y = a + 0.61803398875 * (b - a)
                                    dy, _, _, _, used = leg(
                                        parameters,
                                        states,
                                        k,
                                        epochs[p],
                                        positions[p],
                                        velocities[p],
                                        y,
                                        branches[bi],
                                        normal,
                                        sine,
                                    )
                                    call_counts[p] += used
                                    cy = dy + time_score(
                                        parameters, epochs[p] + y, remaining, penalty, mode
                                    )
                            if cx < incumbent:
                                dt, incumbent = (x, cx)
                            if cy < incumbent:
                                dt, incumbent = (y, cy)
                    dv, r1, vz, rad, used = leg(
                        parameters,
                        states,
                        k,
                        epochs[p],
                        positions[p],
                        velocities[p],
                        dt,
                        branches[bi],
                        normal,
                        sine,
                    )
                    call_counts[p] += used
                    if dv >= 1e90:
                        continue
                    out[cursor, 0] = costs[p] + dv
                    out[cursor, 1] = epochs[p] + dt
                    out[cursor, 2] = p
                    out[cursor, 3] = k
                    out[cursor, 4] = bi
                    for axis in range(3):
                        out[cursor, 5 + axis] = vz[axis]
                        out[cursor, 8 + axis] = r1[axis]
                    out[cursor, 11] = min(radii[p], rad)
                    out[cursor, 12] = dv
                    cursor += 1
    return (out[out[:, 0] < 1e90], int(call_counts.sum()), int(cell_counts.sum()))


@njit(cache=True, parallel=True)
def rollout(parameters, states, records, masks, windows, counts, normal, sine):
    result = np.zeros(len(records))
    call_counts = np.zeros(len(records), np.int64)
    for i in prange(len(records)):
        rec = records[i]
        mask = masks[int(rec[2])] | np.int64(1) << int(rec[3])
        if mask == (np.int64(1) << 35) - 1:
            continue
        best = 1e100
        for k in range(35):
            if mask & np.int64(1) << k:
                continue
            cells = window_cells(
                windows[k],
                counts[k],
                rec[1] + float(parameters.min_target_gap_s),
                min(END, rec[1] + 86400.0),
                float(parameters.lookahead_cell_max_h) * 3600.0,
            )
            for j in range(len(cells)):
                for major in range(2):
                    value, _, _, _, calls = _cell_probe(
                        parameters,
                        states,
                        k,
                        rec[1],
                        rec[8:11],
                        rec[5:8],
                        cells[j, 0],
                        cells[j, 1],
                        np.array([major, 0, 0]),
                        normal,
                        sine,
                        0,
                        0.0,
                        0,
                        1,
                    )
                    call_counts[i] += calls
                    best = min(best, value)
        result[i] = min(best, 3.0)
    return (result, int(call_counts.sum()))


def dynamic_cell(rec, first):
    r, v = (rec[8:11], rec[5:8])
    h = np.cross(r, v)
    h /= max(np.linalg.norm(h), 1e-12)
    energy = float(v @ v / 2 - MU / np.linalg.norm(r))
    return (
        first,
        int(rec[3]),
        int(rec[1] // 21600),
        int((h[2] + 1) * 3),
        int((math.atan2(h[1], h[0]) + math.pi) / 0.8),
        int(energy // 2),
    )


def select_records(parameters, records, scores, masks, routes, width):
    order = np.argsort(scores)
    picked = []
    used = set()
    cells = set()
    buckets = {}

    def add(i, require_new=False):
        if i in used:
            return False
        row = records[i]
        p = int(row[2])
        k = int(row[3])
        cell = dynamic_cell(row, routes[p].order[0])
        if require_new and cell in cells:
            return False
        key = (int(masks[p]) | 1 << k, k, int(row[1] // 1800))
        prior = buckets.setdefault(key, [])
        for t, v in prior:
            if ((t - row[1]) / parameters.time_distance_s) ** 2 + np.sum(
                (v - row[5:8]) ** 2
            ) / parameters.velocity_distance_km_s**2 < 1:
                return False
        if len(prior) >= parameters.state_bucket_cap:
            return False
        prior.append((row[1], row[5:8].copy()))
        used.add(i)
        cells.add(cell)
        picked.append(i)
        return True

    quality = max(1, int(width * (1 - parameters.exploration_fraction)))
    for i in order:
        add(int(i))
        if len(picked) >= quality:
            break
    for i in order:
        add(int(i), True)
        if len(picked) >= width:
            break
    for i in order:
        if len(picked) >= width:
            break
        add(int(i))
    return picked


@njit(cache=True)
def _first_value(
    parameters, states, k, epoch, phase, radius, u, q, major, normal, sine, penalty, mode
):
    direction = u * math.cos(phase) + q * math.sin(phase)
    r0 = direction * radius
    v0 = (-u * math.sin(phase) + q * math.cos(phase)) * math.sqrt(MU / radius)
    dv, _, vz, rad, calls = leg(
        parameters, states, k, 0.0, r0, v0, epoch, np.array([major, 0, 0]), normal, sine
    )
    return (dv + time_score(parameters, epoch, 34, penalty, mode), dv, r0, v0, vz, rad, calls)


@njit(cache=True, parallel=True)
def _first_grid(parameters, states, requests, phases, radius, u, q, normal, sine, penalty, mode):
    per = len(phases) * 2
    records = np.full((len(requests) * per, 18), 1e100)
    calls = np.zeros(len(requests) * per, np.int64)
    for cursor in prange(len(requests) * per):
        i = cursor // per
        j = cursor % per // 2
        major = cursor % 2
        epoch, k, window, lo, hi = requests[i]
        _, dv, r0, v0, vz, rad, used = _first_value(
            parameters,
            states,
            int(k),
            epoch,
            phases[j],
            radius,
            u,
            q,
            major,
            normal,
            sine,
            penalty,
            mode,
        )
        calls[cursor] = used
        if dv >= float(parameters.search_cost_ceiling_km_s):
            continue
        records[cursor, 0] = dv
        records[cursor, 1] = epoch
        records[cursor, 2] = k
        records[cursor, 3] = major
        records[cursor, 4] = window
        for axis in range(3):
            records[cursor, 5 + axis] = r0[axis]
            records[cursor, 8 + axis] = v0[axis]
            records[cursor, 11 + axis] = vz[axis]
        records[cursor, 14] = rad
        records[cursor, 15] = phases[j]
        records[cursor, 16] = lo
        records[cursor, 17] = hi
    return (records[records[:, 0] < 1e90], int(calls.sum()))


@njit(cache=True, parallel=True)
def _first_refine(parameters, states, rows, phase_step, u, q, normal, sine, penalty, mode):
    result = rows.copy()
    calls = np.zeros(len(rows), np.int64)
    for i in prange(len(rows)):
        row = rows[i]
        k = int(row[2])
        major = int(row[3])
        epoch = row[1]
        phase = row[15]
        radius = norm(row[5:8])
        best = row[0] + time_score(parameters, epoch, 34, penalty, mode)
        phase_lo, phase_hi = (phase - phase_step, phase + phase_step)
        for _ in range(int(parameters.first_coordinate_passes)):
            for coordinate in range(2):
                a, b = (row[16], row[17]) if coordinate == 0 else (phase_lo, phase_hi)
                fixed_epoch, fixed_phase = (epoch, phase)
                if b - a < 1e-09:
                    continue
                x, y = (b - 0.61803398875 * (b - a), a + 0.61803398875 * (b - a))
                cx = _first_value(
                    parameters,
                    states,
                    k,
                    x if coordinate == 0 else fixed_epoch,
                    fixed_phase if coordinate == 0 else x,
                    radius,
                    u,
                    q,
                    major,
                    normal,
                    sine,
                    penalty,
                    mode,
                )
                cy = _first_value(
                    parameters,
                    states,
                    k,
                    y if coordinate == 0 else fixed_epoch,
                    fixed_phase if coordinate == 0 else y,
                    radius,
                    u,
                    q,
                    major,
                    normal,
                    sine,
                    penalty,
                    mode,
                )
                calls[i] += cx[6] + cy[6]
                for iteration in range(int(parameters.first_golden_iterations) + 1):
                    if cx[0] < best:
                        best = cx[0]
                        if coordinate == 0:
                            epoch = x
                        else:
                            phase = x
                    if cy[0] < best:
                        best = cy[0]
                        if coordinate == 0:
                            epoch = y
                        else:
                            phase = y
                    if iteration == int(parameters.first_golden_iterations):
                        break
                    if cx[0] < cy[0]:
                        b, y, cy = (y, x, cx)
                        x = b - 0.61803398875 * (b - a)
                        cx = _first_value(
                            parameters,
                            states,
                            k,
                            x if coordinate == 0 else fixed_epoch,
                            fixed_phase if coordinate == 0 else x,
                            radius,
                            u,
                            q,
                            major,
                            normal,
                            sine,
                            penalty,
                            mode,
                        )
                        calls[i] += cx[6]
                    else:
                        a, x, cx = (x, y, cy)
                        y = a + 0.61803398875 * (b - a)
                        cy = _first_value(
                            parameters,
                            states,
                            k,
                            y if coordinate == 0 else fixed_epoch,
                            fixed_phase if coordinate == 0 else y,
                            radius,
                            u,
                            q,
                            major,
                            normal,
                            sine,
                            penalty,
                            mode,
                        )
                        calls[i] += cy[6]
        _, dv, r0, v0, vz, rad, used = _first_value(
            parameters, states, k, epoch, phase, radius, u, q, major, normal, sine, penalty, mode
        )
        calls[i] += used
        if dv < float(parameters.search_cost_ceiling_km_s):
            result[i, 0] = dv
            result[i, 1] = epoch
            for axis in range(3):
                result[i, 5 + axis] = r0[axis]
                result[i, 8 + axis] = v0[axis]
                result[i, 11 + axis] = vz[axis]
            result[i, 14] = rad
            result[i, 15] = phase
    return (result, int(calls.sum()))


def compact_first_rows(parameters, rows, penalty, mode):
    """Bound memory per altitude/target, retaining each cell and branch separately."""
    if not len(rows):
        return rows
    scores = rows[:, 0] + np.array(
        [time_score(parameters, t, 34, penalty, mode) for t in rows[:, 1]]
    )
    buckets = {}
    selected = []
    for i in np.argsort(scores):
        row = rows[i]
        key = (int(row[3]), float(row[16]), float(row[17]))
        if buckets.get(key, 0) >= parameters.first_keep_per_cell_branch:
            continue
        buckets[key] = buckets.get(key, 0) + 1
        selected.append(int(i))
    return rows[selected]


def seed_routes(
    parameters,
    states,
    width,
    penalty,
    rng,
    mode,
    windows,
    counts,
    normal,
    u,
    q,
    sine,
    plane_info,
    diagnostics,
):
    latest = min(
        END - 34 * float(parameters.min_target_gap_s),
        float(parameters.first_max_arrival_h) * 3600.0,
    )
    phase_step = 2 * np.pi / parameters.first_phase_samples
    phases = np.arange(parameters.first_phase_samples) * phase_step + rng.uniform(0, phase_step)
    altitude_count = int(parameters.first_altitude_samples)
    altitudes = parameters.initial_altitude_min_km + (
        np.arange(altitude_count) + rng.random(altitude_count)
    ) / altitude_count * (parameters.initial_altitude_max_km - parameters.initial_altitude_min_km)
    retained = []
    calls = 0
    time_samples = 0
    feasible_count = 0
    for k in range(35):
        requests = []
        for lo, hi, j in window_cells(
            windows[k],
            counts[k],
            float(parameters.min_target_gap_s),
            latest,
            float(parameters.first_cell_max_h) * 3600.0,
        ):
            margin = min(float(parameters.window_edge_tolerance_s) * 1.1, (hi - lo) * 0.25)
            for epoch in np.linspace(
                lo + margin, hi - margin, parameters.first_time_samples_per_cell
            ):
                requests.append((epoch, k, j, lo, hi))
        if not requests:
            continue
        time_samples += len(requests)
        for altitude in altitudes:
            rows, used = _first_grid(
                parameters,
                states,
                np.asarray(requests),
                phases,
                RE + altitude,
                u,
                q,
                normal,
                sine,
                penalty,
                mode,
            )
            calls += used
            feasible_count += len(rows)
            if len(rows):
                retained.append(compact_first_rows(parameters, rows, penalty, mode))
    diagnostics["first_grid_lambert_calls"] = calls
    diagnostics["first_time_samples"] = time_samples
    diagnostics["initial_altitudes_km"] = altitudes.tolist()
    diagnostics["first_grid_feasible_candidates"] = feasible_count
    if not retained:
        return []
    rows = np.vstack(retained)
    diagnostics["first_grid_retained_candidates"] = len(rows)
    scores = rows[:, 0] + np.array(
        [time_score(parameters, t, 34, penalty, mode) for t in rows[:, 1]]
    )
    ranked = np.argsort(scores)
    refine_indices = []
    refine_used = set()
    per_target = {}
    seen = set()
    for pass_no in range(2):
        for i in ranked:
            k = int(rows[i, 2])
            key = (
                k,
                int(rows[i, 4]),
                round(float(np.linalg.norm(rows[i, 5:8])), 6),
                int(rows[i, 3]),
            )
            if per_target.get(k, 0) >= parameters.first_refine_per_target or int(i) in refine_used:
                continue
            if pass_no == 0 and key in seen:
                continue
            refine_indices.append(int(i))
            refine_used.add(int(i))
            seen.add(key)
            per_target[k] = per_target.get(k, 0) + 1
    refined, used = _first_refine(
        parameters, states, rows[refine_indices], phase_step, u, q, normal, sine, penalty, mode
    )
    diagnostics["first_refine_lambert_calls"] = used
    rows = np.vstack([rows, refined])
    scores = rows[:, 0] + np.array(
        [time_score(parameters, t, 34, penalty, mode) for t in rows[:, 1]]
    )
    ranked = np.argsort(scores)
    selected = []
    used = set()
    counts_target = {}
    covered_targets = set()
    covered_windows = set()
    near = {}

    def add(i):
        if i in used:
            return False
        row = rows[i]
        k = int(row[2])
        key = (k, int(row[3]), int(row[1] // 600))
        if counts_target.get(k, 0) >= max(6, width // 12):
            return False
        for old in near.get(key, []):
            if abs(row[1] - old[1]) < 1.0 and np.linalg.norm(row[5:11] - old[5:11]) < 0.0001:
                return False
        near.setdefault(key, []).append(row)
        used.add(i)
        selected.append(i)
        counts_target[k] = counts_target.get(k, 0) + 1
        covered_targets.add(k)
        covered_windows.add((k, int(row[4]), round(float(np.linalg.norm(row[5:8])), 6)))
        return True

    for i in ranked:
        if int(rows[i, 2]) not in covered_targets:
            add(int(i))
        if len(selected) >= width:
            break
    for i in ranked:
        if len(selected) >= width:
            break
        if (
            int(rows[i, 2]),
            int(rows[i, 4]),
            round(float(np.linalg.norm(rows[i, 5:8])), 6),
        ) not in covered_windows:
            add(int(i))
    for i in ranked:
        if len(selected) >= width:
            break
        add(int(i))
    diagnostics["first_targets_feasible"] = sorted((int(k + 1) for k in np.unique(rows[:, 2])))
    diagnostics["first_targets_retained"] = sorted((k + 1 for k in covered_targets))
    diagnostics["first_candidates_evaluated"] = len(rows)
    return [
        Route(
            [int(rows[i, 2])],
            [float(rows[i, 1])],
            [[int(rows[i, 3]), 0, 0]],
            rows[i, 5:11].tolist(),
            float(rows[i, 0]),
            rows[i, 11:14].tolist(),
            float(rows[i, 14]),
            dict(plane_info, initial_source="sampled_plane", first_window_index=int(rows[i, 4])),
        )
        for i in selected
    ]


def continuous_beam_run(
    parameters,
    states,
    width,
    seed,
    penalty,
    maxrev,
    mode=0,
    depth_limit=35,
    diagnostics=None,
):
    if diagnostics is None:
        diagnostics = {}
    started = time.monotonic()
    rng = np.random.default_rng(seed)
    normal, u, q = sample_plane(rng, parameters.plane_normal_z_min, parameters.plane_normal_z_max)
    sine = math.sin(math.radians(parameters.plane_tolerance_deg))
    plane_info = dict(
        plane_normal_icrf=normal.tolist(),
        plane_tolerance_deg=float(parameters.plane_tolerance_deg),
        plane_inclination_deg=float(np.degrees(np.arccos(normal[2]))),
        plane_raan_deg=float(np.degrees(np.arctan2(normal[0], -normal[1])) % 360),
    )
    plane_info.update(
        plane_sector_index=parameters.plane_sector_index,
        plane_sector_count=parameters.plane_sector_count,
        plane_normal_z_min=parameters.plane_normal_z_min,
        plane_normal_z_max=parameters.plane_normal_z_max,
    )
    diagnostics.update(
        plane_info,
        expansion_lambert_calls=0,
        lookahead_lambert_calls=0,
        expansion_window_cells=0,
        depth_reached=0,
    )
    diagnostics["search_profile"] = parameters.search_profile
    print(
        f"PLANE seed={seed} inclination={plane_info['plane_inclination_deg']:.4f} raan={plane_info['plane_raan_deg']:.4f} tolerance={parameters.plane_tolerance_deg}",
        flush=True,
    )
    windows, counts = build_windows(
        states, normal, parameters.plane_tolerance_deg, parameters.window_edge_tolerance_s
    )
    diagnostics["window_counts"] = counts.tolist()
    diagnostics["window_seconds"] = time.monotonic() - started
    first_started = time.monotonic()
    """
    铺网格:每个目标,把它的门票窗口切成 ≤6 小时的小格,每格取 3 个到达时刻;初始轨道取 2 个高度(590–610 km 带内)、沿轨道 64 个相位角;对"时刻×相位"的每种组合,从初始位置向目标做一次 Lambert,算 Δv。
    粗筛:每个(窗口格×长/短弧)只留 4 条,控制内存。
    精化:每个目标挑 4 条最好的,对(到达时刻、相位)两个坐标轮流做黄金分割,把 Δv 磨低一点。
    定束:再按分数排一遍,尽量保证 35 个目标各有代表,选出 width 条。
    """
    routes = seed_routes(
        parameters,
        states,
        width,
        penalty,
        rng,
        mode,
        windows,
        counts,
        normal,
        u,
        q,
        sine,
        plane_info,
        diagnostics,
    )
    diagnostics["first_search_seconds"] = time.monotonic() - first_started
    diagnostics["depth_reached"] = 1 if routes else 0
    print(
        f"FIRST kept={len(routes)} targets={len(diagnostics.get('first_targets_retained', []))} Lambert={diagnostics.get('first_grid_lambert_calls', 0) + diagnostics.get('first_refine_lambert_calls', 0)} seconds={diagnostics['first_search_seconds']:.2f}",
        flush=True,
    )
    if not routes:
        return []
    branches = np.array(
        [[m, r, s] for r in range(maxrev + 1) for m in range(2) for s in range(1 if r == 0 else 2)],
        np.int64,
    )
    rewards = rng.uniform(-0.12, 0.12, 35)
    for depth in range(1, depth_limit):
        masks = np.array([sum((1 << k for k in r.order)) for r in routes], np.int64)
        positions = np.array([target_state(states, r.order[-1], r.times_s[-1])[0] for r in routes])
        rec, calls, cells = expand(
            parameters,
            states,
            masks,
            positions,
            np.array([r.arrival_velocity_km_s for r in routes]),
            np.array([r.times_s[-1] for r in routes]),
            np.array([r.cost_km_s for r in routes]),
            np.array([r.minimum_radius_km for r in routes]),
            windows,
            counts,
            normal,
            sine,
            branches,
            34 - depth,
            penalty,
            mode,
        )
        diagnostics["expansion_lambert_calls"] += calls
        diagnostics["expansion_window_cells"] += cells
        rec = rec[rec[:, 0] <= float(parameters.search_cost_ceiling_km_s)]
        if not len(rec):
            return []
        base = rec[:, 0] + np.array(
            [time_score(parameters, t, 34 - depth, penalty, mode) for t in rec[:, 1]]
        )
        accumulated = np.array([sum(rewards[r.order]) for r in routes])
        base -= accumulated[rec[:, 2].astype(int)] + rewards[rec[:, 3].astype(int)]
        shortlist = select_records(
            parameters, rec, base, masks, routes, width * parameters.lookahead_pool_multiplier
        )
        small = rec[shortlist]
        score = base[shortlist]
        if depth < 34:
            ahead, used = rollout(parameters, states, small, masks, windows, counts, normal, sine)
            score = score + parameters.lookahead_weight * ahead
            diagnostics["lookahead_lambert_calls"] += used
        chosen = select_records(parameters, small, score, masks, routes, width)
        following = []
        for i in chosen:
            row = small[i]
            parent = routes[int(row[2])]
            k = int(row[3])
            following.append(
                Route(
                    parent.order + [k],
                    parent.times_s + [float(row[1])],
                    parent.branches + [branches[int(row[4])].tolist()],
                    parent.initial_state_km_km_s,
                    float(row[0]),
                    row[5:8].tolist(),
                    float(row[11]),
                    dict(
                        parent.origin,
                        kind="v6_plane_window_beam",
                        seed=seed,
                        max_revolutions=maxrev,
                        time_mode=mode,
                    ),
                )
            )
        routes = following
        if not routes:
            return []
        diagnostics["depth_reached"] = depth + 1
        print(
            f"BEAM depth={depth + 1:02d} kept={len(routes)} best={min((r.cost_km_s for r in routes)):.6f} Lambert={calls} cells={cells}",
            flush=True,
        )
    diagnostics["elapsed_seconds"] = time.monotonic() - started
    return routes
