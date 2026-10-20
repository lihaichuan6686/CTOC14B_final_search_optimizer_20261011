"""用独立 DOP853 连续传播复验：脉冲实际累加，访问点不重置状态。

这部分是物理可行性检查：35 个目标、飞越半径、初轨、200 km 高度和10天任务窗口。
不检查环境或通用文件格式。states 和轨迹文件为 ICRF，pole 为极轴系到 ICRF 的矩阵。
"""

import copy
import numpy as np
from scipy.integrate import solve_ivp
from orbit_dynamics import MU, RE, lambert, arc_min_radius, target_state
from orbit_dynamics import shoot, acceleration_hessian, variational_rhs


def rotate(data, matrix):
    d = copy.deepcopy(data)
    d["positions"] = (np.asarray(d["positions"]) @ matrix).tolist()
    initial = np.array(d["initial"])
    d["initial"] = np.r_[initial[:3] @ matrix, initial[3:] @ matrix].tolist()
    d.pop("encounter_offsets_km", None)
    return d


def precise_shoot(r0, r1, duration, velocity):
    """Correct the RK4 search seed with independently integrated DOP853 + STM."""
    velocity = np.array(velocity, dtype=float, copy=True)
    for _ in range(7):
        sol = solve_ivp(
            lambda t, y: variational_rhs(y),
            (0.0, float(duration)),
            np.r_[r0, velocity, np.eye(6).ravel()],
            method="DOP853",
            rtol=8e-13,
            atol=1e-12,
            max_step=120.0,
        )
        if not sol.success or not np.isfinite(sol.y[:, -1]).all():
            raise ValueError("High precision J2 integration failed")
        end = sol.y[:, -1]
        defect = np.asarray(r1) - end[:3]
        residual = float(np.linalg.norm(defect))
        if residual <= 5e-09:
            return (velocity, end[3:6], residual)
        velocity += np.linalg.solve(end[6:].reshape(6, 6)[:3, 3:], defect)
    raise ValueError(f"High precision J2 shooting failed: {residual} km")


def coast(r, v, duration, model, strict=True):
    if duration <= 0:
        return (np.r_[r, v], float(np.linalg.norm(r)), [])

    def rhs(t, y):
        acceleration = (
            acceleration_hessian(y[:3])[0]
            if model == "j2"
            else -MU * y[:3] / np.linalg.norm(y[:3]) ** 3
        )
        return np.r_[y[3:], acceleration]

    def radial(t, y):
        return float(y[:3] @ y[3:])

    def ground(t, y):
        return float(np.linalg.norm(y[:3]) - RE - 199.0)

    ground.terminal = True
    ground.direction = -1
    sol = solve_ivp(
        rhs,
        (0.0, float(duration)),
        np.r_[r, v],
        method="DOP853",
        rtol=8e-13 if strict else 1e-09,
        atol=1e-12 if strict else 1e-08,
        events=(radial, ground),
        max_step=120.0 if strict else 300.0,
    )
    if not sol.success or sol.t[-1] < duration - 1e-05 or (not np.isfinite(sol.y).all()):
        raise ValueError("Coast integration failed or crossed the altitude limit")
    radii = [np.linalg.norm(sol.y[:3, 0]), np.linalg.norm(sol.y[:3, -1])]
    radii.extend((np.linalg.norm(row[:3]) for row in sol.y_events[0]))
    return (sol.y[:, -1], float(min(radii)), sol.t_events[0].tolist())


def evaluate(data, states, pole, strict=False):
    d = copy.deepcopy(data)
    model = d["model"]
    q = rotate(d, pole) if model == "j2" else d
    r = np.array(q["positions"])
    t = np.array(q["times"])
    b = np.array(q["branches"], dtype=np.int64)
    # 这些是竞赛物理约束，不能随输入格式检查一起删掉。
    if t[0] != 0 or t[-1] > 864000 or np.any(np.diff(t) <= 0):
        raise ValueError("节点时刻不满足任务窗口或时间顺序")
    if sorted(k for k in d["node_ids"] if k >= 0) != list(range(35)):
        raise ValueError("轨迹未恰好访问全部35个目标")
    if np.linalg.norm(r[0] - np.array(q["initial"][:3])) > 1e-9:
        raise ValueError("第一段位置与初始轨道不连续")
    errors = []
    vi = []
    vf = []
    minimum = float("inf")
    endpoint_errors = []
    extrema = []
    a = 1 / (2 / np.linalg.norm(r[0]) - np.dot(q["initial"][3:], q["initial"][3:]) / MU)
    ecc = np.linalg.norm(
        np.cross(q["initial"][3:], np.cross(r[0], q["initial"][3:])) / MU
        - r[0] / np.linalg.norm(r[0])
    )
    if not RE + 590 - 1e-08 <= a <= RE + 610 + 1e-08 or not 0 <= ecc < 0.001:
        errors.append("initial_orbit")
    for i in range(len(b)):
        dt = t[i + 1] - t[i]
        if model == "j2":
            va, vz, _, rad, residual = shoot(r[i], r[i + 1], dt, b[i])
            if residual > 0.0001:
                raise ValueError(f"Unconverged J2 arc {i}: {residual} km")
        else:
            va, vz = lambert(r[i], r[i + 1], dt, *b[i])
            residual = 0.0
            rad = arc_min_radius(r[i], va, r[i + 1], vz, dt)
        if not np.isfinite(va).all() or not np.isfinite(vz).all():
            raise ValueError(f"Invalid Lambert arc {i}")
        if strict and model == "j2":
            va, vz, residual = precise_shoot(r[i], r[i + 1], dt, va)
        elif strict:
            end, rad, roots = coast(r[i], va, dt, model, True)
            residual = float(np.linalg.norm(end[:3] - r[i + 1]))
            extrema.extend((t[i] + np.array(roots)).tolist())
            if residual > 0.0005:
                errors.append(f"propagation_endpoint_{i}")
            vz = end[3:]
        vi.append(va)
        vf.append(vz)
        minimum = min(minimum, float(rad))
        endpoint_errors.append(float(residual) * 1000)
    vi = np.array(vi)
    vf = np.array(vf)
    pulses = vi - np.vstack((np.array(q["initial"][3:]), vf[:-1]))
    cost = float(np.linalg.norm(pulses, axis=1).sum())
    actual_positions = r.copy()
    replay_errors = []
    if strict:
        state = np.array(q["initial"], dtype=float)
        minimum = float(np.linalg.norm(state[:3]))
        extrema = []
        for i in range(len(b)):
            state[3:] += pulses[i]
            state, rad, roots = coast(state[:3], state[3:], t[i + 1] - t[i], model, True)
            minimum = min(minimum, rad)
            extrema.extend((t[i] + np.array(roots)).tolist())
            actual_positions[i + 1] = state[:3]
            replay_errors.append(float(np.linalg.norm(state[:3] - r[i + 1])) * 1000)
    else:
        state = np.r_[r[-1], vf[-1]]
    end, final_rad, roots = coast(state[:3], state[3:], 864000.0 - t[-1], model, strict)
    minimum = min(minimum, final_rad)
    extrema.extend((t[-1] + np.array(roots)).tolist())
    if minimum < RE + 200:
        errors.append("altitude")
    if model == "j2":
        actual_positions = actual_positions @ pole.T
    misses = []
    valid = []
    for i, k in enumerate(d["node_ids"]):
        if k >= 0:
            miss = float(
                np.linalg.norm(actual_positions[i] - target_state(states, k, float(t[i]))[0]) * 1000
            )
            misses.append(miss)
            if miss <= 1000:
                valid.append(int(k))
            else:
                errors.append(f"flyby_{k + 1}")
    if model == "j2":
        vi = vi @ pole.T
        vf = vf @ pole.T
        pulses = pulses @ pole.T
    report = dict(
        model=model,
        status=(
            ("numerical_passed" if strict else "numerical_screened")
            if not errors
            else "numerical_failed"
        ),
        strict=bool(strict),
        covered=len(set(valid)),
        planned=len(misses),
        complete=len(set(valid)) == 35,
        errors=errors,
        total_dv_km_s=cost,
        min_alt_km=minimum - RE,
        max_miss_m=max(misses, default=0.0),
        max_endpoint_error_m=max(endpoint_errors),
        continuous_replay=bool(strict),
        max_replay_node_error_m=max(replay_errors, default=0.0),
        initial_semimajor_axis_km=float(a),
        initial_eccentricity=float(ecc),
        mission_end_s=864000.0,
        final_coast_start_s=float(t[-1]),
        radial_extrema_s=extrema,
        vi_km_s=vi.tolist(),
        vf_km_s=vf.tolist(),
        impulses_km_s=pulses.tolist(),
        disclaimer="Numerical surrogate evaluation; official ATK validation is separate.",
    )
    return report
