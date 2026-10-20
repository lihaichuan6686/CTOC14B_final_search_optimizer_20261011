"""二体全链联调：初始圆轨道四变量、自由脉冲位置、全部节点时刻。

joint_polish 用 SLSQP + 逐级平滑的脉冲范数；每个变量只重算受影响的相邻弧。
node_branch_dp 用动态规划选整条链的 Lambert 分支；compact 删除可合并的自由节点。
所有时间变量以小时缩放，位置以 30000 km 缩放；计算脉冲仍用 km/s。
"""

import copy
import json
import math
import time
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, LinearConstraint
from orbit_dynamics import MU, RE, norm, target_state, lambert, arc_min_radius

BRANCHES = np.array(
    [
        [major, rev, side]
        for rev in range(3)
        for major in range(2)
        for side in range(1 if rev == 0 else 2)
    ],
    dtype=np.int64,
)


@njit(cache=True)
def initial_from_x(x):
    lon, lat, heading, altitude = (x[0], x[1], x[2], 600 + 10 * x[3])
    unit = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])
    east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    north = np.cross(unit, east)
    v = math.sqrt(MU / (RE + altitude)) * (math.cos(heading) * east + math.sin(heading) * north)
    return np.concatenate(((RE + altitude) * unit, v))


def pack_circular_initial(initial):
    r = np.array(initial[:3])
    v = np.array(initial[3:])
    radius = norm(r)
    lon = np.arctan2(r[1], r[0])
    lat = np.arcsin(r[2] / radius)
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.cross(r / radius, east)
    heading = np.arctan2(np.dot(v, north), np.dot(v, east))
    return np.r_[lon, lat, heading, (radius - RE - 600) / 10]


@njit(cache=True)
def arc_choices(r1, r2, dt, maxrev):
    vi = np.full((len(BRANCHES), 3), np.nan)
    vf = vi.copy()
    for b in range(len(BRANCHES)):
        major, rev, side = BRANCHES[b]
        if rev > maxrev:
            continue
        a, z = lambert(r1, r2, dt, major, rev, side)
        if np.isfinite(a[0]) and arc_min_radius(r1, a, r2, z, dt) >= RE + 205:
            vi[b] = a
            vf[b] = z
    return (vi, vf)


def cache(data):
    r = np.array(data["positions"])
    t = np.array(data["times"])
    b = np.array(data["branches"])
    vin = [np.array(data["initial"][3:])]
    vout = []
    for i in range(len(b)):
        a, z = lambert(r[i], r[i + 1], t[i + 1] - t[i], *b[i])
        vout.append(a)
        vin.append(z)
    return (r, np.array(vin), np.array(vout))


def actual_cost(data):
    _, vi, vo = cache(data)
    return float(np.linalg.norm(vo - vi[:-1], axis=1).sum())


@njit(cache=True)
def segment(r1, r2, dt, branch):
    a, z = lambert(r1, r2, dt, branch[0], branch[1], branch[2])
    if not np.isfinite(a[0]):
        return (np.zeros(3), np.zeros(3), 100000000.0)
    rad = arc_min_radius(r1, a, r2, z, dt)
    sine = norm(np.cross(r1, r2)) / (norm(r1) * norm(r2))
    ill = max(0.0, 0.035 - sine) if np.dot(r1, r2) < 0 else 0.0
    return (a, z, max(0.0, RE + 205 - rad) * 0.1 + 1000 * ill * ill)


@njit(cache=True)
def impulse_cost(v, smooth):
    return math.sqrt(np.dot(v, v) + smooth * smooth)


@njit(cache=True)
def node_local(k, rr, tt, r, t, vi, vf, branches, vstart, smooth):
    n = len(t) - 1
    if k == 0:
        a, z, p = segment(rr, r[1], t[1], branches[0])
        c = impulse_cost(a - vstart, smooth) + p
        if n > 1:
            c += impulse_cost(vi[1] - z, smooth)
        return c
    if tt - t[k - 1] < 0.5 or (k < n and t[k + 1] - tt < 0.5):
        return 1000000000.0
    a, z, p = segment(r[k - 1], rr, tt - t[k - 1], branches[k - 1])
    c = impulse_cost(a - (vstart if k == 1 else vf[k - 2]), smooth) + p
    if k < n:
        b, w, q = segment(rr, r[k + 1], t[k + 1] - tt, branches[k])
        c += impulse_cost(b - z, smooth) + q
        if k + 1 < n:
            c += impulse_cost(vi[k + 1] - w, smooth)
    return c


@njit(cache=True)
def joint_value_gradient(x, states, node_ids, branches, free, smooth, want_gradient=True):
    """由变量重建位置/时间，逐弧Lambert计算Δv，中心差分只重算相邻受影响项。"""
    n = len(node_ids) - 1
    free_count = len(free)
    time_offset = 4 + 3 * free_count
    initial = initial_from_x(x)
    vstart = initial[3:]
    t = np.zeros(n + 1)
    t[1:] = x[time_offset:] * 3600
    r = np.zeros((n + 1, 3))
    r[0] = initial[:3]
    for i in range(free_count):
        r[free[i]] = x[4 + 3 * i : 7 + 3 * i] * 30000
    for k in range(1, n + 1):
        if node_ids[k] >= 0:
            r[k] = target_state(states, node_ids[k], t[k])[0]
    vi = np.empty((n, 3))
    vf = np.empty((n, 3))
    pen = np.empty(n)
    cost = 0.0
    for i in range(n):
        if t[i + 1] - t[i] < 0.5:
            return (1000000000.0, np.zeros_like(x))
        vi[i], vf[i], pen[i] = segment(r[i], r[i + 1], t[i + 1] - t[i], branches[i])
    for i in range(n):
        cost += impulse_cost(vi[i] - (vstart if i == 0 else vf[i - 1]), smooth) + pen[i]
    gradient = np.zeros_like(x)
    if not want_gradient:
        return (cost, gradient)
    for j in range(len(x)):
        h = 1e-06 if j >= time_offset else 1e-07
        values = np.zeros(2)
        for si in range(2):
            delta = h * (1 if si == 0 else -1)
            if j < 4:
                xx = x[:4].copy()
                xx[j] += delta
                init = initial_from_x(xx)
                values[si] = node_local(0, init[:3], 0.0, r, t, vi, vf, branches, init[3:], smooth)
            elif j < time_offset:
                k = free[(j - 4) // 3]
                rr = r[k].copy()
                rr[(j - 4) % 3] += delta * 30000
                values[si] = node_local(k, rr, t[k], r, t, vi, vf, branches, vstart, smooth)
            else:
                k = j - time_offset + 1
                tt = t[k] + delta * 3600
                rr = target_state(states, node_ids[k], tt)[0] if node_ids[k] >= 0 else r[k]
                values[si] = node_local(k, rr, tt, r, t, vi, vf, branches, vstart, smooth)
        gradient[j] = (values[0] - values[1]) / (2 * h)
    return (cost, gradient)


def joint_polish(
    data,
    states,
    minutes,
    output,
    method="SLSQP",
    smoothings=(0.03, 0.003, 0.0002),
    maxiter=350,
    min_gap_hours=0.05,
):
    """二体全链SLSQP：初轨4变量 + 自由节点3坐标 + 所有节点时刻。

    不匹配目标速度；每个目标位置随其访问时刻由Hermite星历更新。
    平滑目标采用sqrt(Δv²+ε²)，按smoothings递减ε；局部中心差分形成稀疏影响的梯度。
    时间线性约束要求相邻节点至少min_gap_hours，任务结束不超过240小时。
    best保存时间可行且非平滑目标更低的迭代点；输出cost重新累加实际脉冲。"""
    for key in ["encounter_offsets_km", "encounter_ball_history", "encounter_ball_radius_km"]:
        data.pop(key, None)
    node_ids = np.array(data["node_ids"], dtype=np.int64)
    free = np.where(node_ids[1:] < 0)[0] + 1
    branches = np.array(data["branches"], dtype=np.int64)
    x = np.r_[
        pack_circular_initial(data["initial"]),
        np.array(data["positions"])[free].ravel() / 30000,
        np.array(data["times"])[1:] / 3600,
    ]
    time_offset = 4 + 3 * len(free)
    n = len(node_ids) - 1
    matrix = np.zeros((n, len(x)))
    for k in range(n):
        matrix[k, time_offset + k] = 1
        if k:
            matrix[k, time_offset + k - 1] = -1
    bounds = (
        [(None, None), (-1.5707, 1.5707), (None, None), (-0.999999, 0.999999)]
        + [(None, None)] * (3 * len(free))
        + [(min_gap_hours, 240)] * n
    )
    best = [joint_value_gradient(x, states, node_ids, branches, free, 0.0)[0], x.copy()]
    start = time.monotonic()
    deadline = start + minutes * 60
    calls = [0]
    for smoothing in smoothings:

        def fun(z):
            if time.monotonic() > deadline:
                raise TimeoutError
            c, g = joint_value_gradient(z, states, node_ids, branches, free, smoothing)
            calls[0] += 1
            if calls[0] % 100 == 0:
                print("JOINT", calls[0], c, "smooth", smoothing, flush=True)
            raw = joint_value_gradient(z, states, node_ids, branches, free, 0.0, False)[0]
            if (
                np.min(np.diff(np.r_[0, z[time_offset:]])) >= min_gap_hours - 1e-09
                and raw < best[0]
            ):
                best[:] = [raw, z.copy()]
            return (c, g)

        try:
            if method == "L-BFGS-B":
                epochs = np.r_[0, x[time_offset:], 240.05]
                temporal = [
                    (
                        max(0.05, (epochs[k - 1] + epochs[k]) / 2 + 0.025),
                        min(240.0, (epochs[k] + epochs[k + 1]) / 2 - 0.025),
                    )
                    for k in range(1, n + 1)
                ]
                res = minimize(
                    fun,
                    x,
                    method=method,
                    jac=True,
                    bounds=bounds[:time_offset] + temporal,
                    options={
                        "maxiter": maxiter,
                        "ftol": 1e-12,
                        "gtol": 1e-07,
                        "maxcor": 40,
                        "maxls": 30,
                    },
                )
            else:
                res = minimize(
                    fun,
                    x,
                    method="SLSQP",
                    jac=True,
                    bounds=bounds,
                    constraints=[LinearConstraint(matrix, min_gap_hours, np.inf)],
                    options={"maxiter": maxiter, "ftol": 1e-08},
                )
            x = res.x
            print("JOINT_PHASE", smoothing, res.fun, res.message, flush=True)
        except TimeoutError:
            break
    x = best[1]
    times = np.r_[0, x[time_offset:] * 3600]
    initial = initial_from_x(x)
    pos = np.array(data["positions"])
    pos[0] = initial[:3]
    pos[free] = x[4:time_offset].reshape(-1, 3) * 30000
    for k in range(1, len(node_ids)):
        if node_ids[k] >= 0:
            pos[k] = target_state(states, node_ids[k], times[k])[0]
    actual = 0.0
    v = initial[3:]
    minradius = 1e30
    for k in range(n):
        a, z = lambert(pos[k], pos[k + 1], times[k + 1] - times[k], *branches[k])
        actual += norm(a - v)
        v = z
        minradius = min(
            minradius, arc_min_radius(pos[k], a, pos[k + 1], z, times[k + 1] - times[k])
        )
    data.update(
        initial=initial.tolist(),
        positions=pos.tolist(),
        times=times.tolist(),
        cost=float(actual),
        search_objective=best[0],
        minimum_arc_radius_km=minradius,
        joint_elapsed_s=time.monotonic() - start,
        refinement_model="two_body",
        minimum_gap_hours=min_gap_hours,
    )
    if output:
        output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("JOINT_RESULT", actual, "regularized", best[0], flush=True)
    return data


def conditioned_nodes(data, max_angle=140.0):
    """沿同一Kepler弧二分长转角段，插入零脉冲自由节点。

    插点位置由DOP853两体传播得到；重新选取能复现原速度的Lambert分支。
    不是凭空增加燃料；这些节点随后可自由移动，以改变脉冲结构。"""
    import copy

    data = copy.deepcopy(data)
    pos = [data["positions"][0]]
    times = [0.0]
    node_ids = [-1]
    branches = []

    def rhs(t, y):
        return np.r_[y[3:], -MU * y[:3] / np.linalg.norm(y[:3]) ** 3]

    def append(r1, r2, t1, t2, b, k, depth=0):
        angle = np.degrees(np.arccos(np.clip(np.dot(r1, r2) / norm(r1) / norm(r2), -1, 1)))
        if b[0]:
            angle = 360 - angle
        angle += 360 * b[1]
        if angle <= max_angle or depth >= 9:
            pos.append(r2.tolist())
            times.append(t2)
            node_ids.append(k)
            branches.append(b)
            return
        a, z = lambert(r1, r2, t2 - t1, *b)
        dt = (t2 - t1) / 2
        middle = solve_ivp(rhs, [0, dt], np.r_[r1, a], method="DOP853", rtol=2e-12, atol=1e-11).y[
            :, -1
        ]
        for ra, rb, tt1, tt2, v, k2 in [
            (r1, middle[:3], t1, t1 + dt, a, -1),
            (middle[:3], r2, t1 + dt, t2, middle[3:], k),
        ]:
            choices = []
            for rev in range(b[1] + 1):
                for major in range(2):
                    for side in range(1 if rev == 0 else 2):
                        av, zv = lambert(ra, rb, tt2 - tt1, major, rev, side)
                        if np.isfinite(av[0]):
                            choices.append((norm(av - v), [major, rev, side]))
            error, branch = min(choices, key=lambda x: x[0])
            if error > 1e-05:
                raise ValueError("Subdivision changed the Kepler arc")
            append(ra, rb, tt1, tt2, branch, k2, depth + 1)

    r = np.array(data["positions"])
    tt = data["times"]
    for i, b in enumerate(data["branches"]):
        append(r[i], r[i + 1], tt[i], tt[i + 1], b, data["node_ids"][i + 1])
    data.update(
        positions=pos,
        times=times,
        node_ids=node_ids,
        branches=branches,
        subdivision_max_angle_deg=max_angle,
    )
    return data


def node_branch_dp(data, maxrev=2):
    """对固定节点位置和时刻，用动态规划选择全链Lambert分支。

    递推费用=前段累计费用+||本段出发速度−前段到达速度||。
    第一段同步选初始圆轨道切向速度，末端不追加速度匹配脉冲。"""
    trajectory_data = copy.deepcopy(data)
    r = np.array(trajectory_data["positions"])
    t = np.array(trajectory_data["times"])
    n = len(t) - 1
    m = len(BRANCHES)
    vi = []
    vf = []
    for i in range(n):
        a, z = arc_choices(r[i], r[i + 1], float(t[i + 1] - t[i]), maxrev)
        vi.append(a)
        vf.append(z)
    vi = np.array(vi)
    vf = np.array(vf)
    cost = np.full(m, np.inf)
    parents = np.full((n, m), -1, int)
    unit = r[0] / norm(r[0])
    circle = np.full((m, 3), np.nan)
    speed = np.sqrt(MU / norm(r[0]))
    for b in range(m):
        if not np.all(np.isfinite(vi[0, b])):
            continue
        tangent = vi[0, b] - vi[0, b] @ unit * unit
        if norm(tangent) < 1e-12:
            continue
        circle[b] = speed * tangent / norm(tangent)
        cost[b] = norm(vi[0, b] - circle[b])
    for i in range(1, n):
        nxt = np.full(m, np.inf)
        for b in range(m):
            if not np.all(np.isfinite(vi[i, b])):
                continue
            c = np.nan_to_num(
                cost + np.linalg.norm(vf[i - 1] - vi[i, b], axis=1), nan=np.inf, posinf=np.inf
            )
            p = int(np.argmin(c))
            parents[i, b] = p
            nxt[b] = c[p]
        cost = nxt
    b = int(np.argmin(cost))
    total = float(cost[b])
    if not np.isfinite(total):
        raise ValueError("No feasible full node branch chain")
    path = [b]
    for i in range(n - 1, 0, -1):
        b = parents[i, b]
        path.append(int(b))
    path.reverse()
    trajectory_data["initial"][3:] = circle[path[0]].tolist()
    trajectory_data["branches"] = BRANCHES[path].tolist()
    trajectory_data["cost"] = total
    return trajectory_data


def compact(data, threshold=0.0005):
    """逐个尝试合并自由节点，只有重算后的全链费用变化不超过threshold才删除。"""
    trajectory_data = copy.deepcopy(data)
    old = actual_cost(trajectory_data)
    i = 1
    while i < len(trajectory_data["times"]) - 1:
        if trajectory_data["node_ids"][i] >= 0:
            i += 1
            continue
        candidates = []
        for branch in BRANCHES:
            trial = copy.deepcopy(trajectory_data)
            for key in ["positions", "times", "node_ids"]:
                trial[key].pop(i)
            trial["branches"][i - 1 : i + 1] = [branch.tolist()]
            a, z = lambert(
                np.array(trial["positions"][i - 1]),
                np.array(trial["positions"][i]),
                trial["times"][i] - trial["times"][i - 1],
                *branch
            )
            if (
                not np.all(np.isfinite(a))
                or arc_min_radius(
                    np.array(trial["positions"][i - 1]),
                    a,
                    np.array(trial["positions"][i]),
                    z,
                    trial["times"][i] - trial["times"][i - 1],
                )
                < RE + 205
            ):
                continue
            value = actual_cost(trial)
            if np.isfinite(value):
                candidates.append((value, trial))
        if candidates and min(candidates, key=lambda p: p[0])[0] <= old + threshold:
            old, trajectory_data = min(candidates, key=lambda p: p[0])
        else:
            i += 1
    trajectory_data["cost"] = old
    return trajectory_data
