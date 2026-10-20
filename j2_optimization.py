"""J₂ 多重打靶 + 序列凸优化（SCP），在一个文件中展开变量、目标、约束和更新。

变量 z = [初轨六参数, 后续节点三坐标, 后续节点时刻]。
初轨参数：[经度, 纬度, 航向, (a−RE−600)/10, e·cos(f)/0.001, e·sin(f)/0.001]。
访问节点三坐标是相对目标的位置偏移/km；自由节点是绝对位置/30000km；时刻单位 h。
目标 Σ||Δv_i||，目标速度不需要匹配。约束：初轨范围、访问球、时间顺序和信赖域。
通过每段 STM 计算脉冲对端点/时间的导数；非线性回代检查高度和实际下降量。
"""

import copy
import json
import math
import time
import numpy as np
import scipy.sparse as sp
import cvxpy as cp
from numba import njit
from orbit_dynamics import MU, RE, target_state, shoot, acceleration_hessian
from impulse_nodes import ensure_offsets


@njit(cache=True)
def initial_elliptic(x):
    """把6个无量纲初轨变量变成[r0,v0]；用e·cos(f)、e·sin(f)避免近圆轨道的角度奇异。"""
    lon, lat, heading = (x[0], x[1], x[2])
    a = RE + 600 + 10 * x[3]
    ec = 0.001 * x[4]
    es = 0.001 * x[5]
    unit = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])
    east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    north = np.cross(unit, east)
    tangent = math.cos(heading) * east + math.sin(heading) * north
    p = a * (1 - ec * ec - es * es)
    return np.concatenate(
        (p / (1 + ec) * unit, math.sqrt(MU / p) * (es * unit + (1 + ec) * tangent))
    )


def pack_initial(initial):
    """从初始状态反求经纬度、航向、半长轴及两个偏心率分量。"""
    r, v = np.array(initial).reshape(2, 3)
    rn = np.linalg.norm(r)
    h = np.cross(r, v)
    p = h @ h / MU
    a = 1 / (2 / rn - v @ v / MU)
    lon = np.arctan2(r[1], r[0])
    lat = np.arcsin(r[2] / rn)
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.cross(r / rn, east)
    heading = np.arctan2(v @ north, v @ east)
    return np.array(
        [
            lon,
            lat,
            heading,
            (a - RE - 600) / 10,
            (p / rn - 1) / 0.001,
            r @ v / rn / np.sqrt(MU / p) / 0.001,
        ]
    )


@njit(cache=True)
def pulse_model(x, branches, want_jac=True):
    """给定绝对节点变量，计算全链脉冲与Jacobian。

    每段先用J₂打靶求vi,vf和STM Φ。由末端位置约束微分得
    ∂vi/∂r0 = −Φ_rv⁻¹ Φ_rr，∂vi/∂r1 = Φ_rv⁻¹，∂vi/∂T = −Φ_rv⁻¹ vf。
    再传递到达速度导数，相邻弧首尾相减得到每次脉冲的导数。
    自由节点位置以30000 km缩放，时间以3600 s缩放，导数包含这些比例。"""
    arc_count = len(branches)
    time_offset = 6 + 3 * arc_count
    dim = len(x)
    init = initial_elliptic(x)
    r = np.empty((arc_count + 1, 3))
    r[0] = init[:3]
    r[1:] = x[6:time_offset].reshape(arc_count, 3) * 30000
    t = np.zeros(arc_count + 1)
    t[1:] = x[time_offset:] * 3600
    delta_v = np.zeros((arc_count, 3))
    jac = np.zeros((arc_count * 3, dim))
    previous = init[3:]
    minimum = 1000000000.0
    dr = np.zeros((arc_count + 1, 3, dim))
    previous_jac = np.zeros((3, dim))
    if want_jac:
        for j in range(6):
            xp = x[:6].copy()
            xm = x[:6].copy()
            xp[j] += 1e-06
            xm[j] -= 1e-06
            derivative = (initial_elliptic(xp) - initial_elliptic(xm)) / 2e-06
            dr[0, :, j] = derivative[:3]
            previous_jac[:, j] = derivative[3:]
        for j in range(arc_count):
            for axis in range(3):
                dr[j + 1, axis, 6 + 3 * j + axis] = 30000
    for i in range(arc_count):
        vi, vf, phi, rad, error = shoot(r[i], r[i + 1], t[i + 1] - t[i], branches[i])
        if error > 0.001 or not np.isfinite(vi[0]):
            raise ValueError("Elliptic J2 shooting failed")
        delta_v[i] = vi - previous
        previous = vf
        minimum = min(minimum, rad)
        if want_jac:
            B = np.linalg.inv(phi[:3, 3:])
            A = -B @ phi[:3, :3]
            C = phi[3:, :3] + phi[3:, 3:] @ A
            D = phi[3:, 3:] @ B
            vt = -B @ vf
            wt = acceleration_hessian(r[i + 1])[0] + phi[3:, 3:] @ vt
            startjac = A @ dr[i] + B @ dr[i + 1]
            endjac = C @ dr[i] + D @ dr[i + 1]
            startjac[:, time_offset + i] += 3600 * vt
            endjac[:, time_offset + i] += 3600 * wt
            if i:
                startjac[:, time_offset + i - 1] -= 3600 * vt
                endjac[:, time_offset + i - 1] -= 3600 * wt
            jac[3 * i : 3 * i + 3] = startjac - previous_jac
            previous_jac = endjac
    return (delta_v, jac, minimum)


def pack(trajectory_data, states):
    """排列优化变量：[初轨6个，节点坐标3N个，节点时间N个]。

    访问节点用相对目标的km偏移，自由节点用绝对位置/30000；时间用小时。
    访问位置=目标位置(t)+偏移，因此目标运动不是固定点近似。"""
    r = np.array(trajectory_data["positions"])[1:] / 30000
    t = np.array(trajectory_data["times"])[1:] / 3600
    for j, k in enumerate(trajectory_data["node_ids"][1:]):
        if k >= 0:
            r[j] = (
                np.array(trajectory_data["positions"][j + 1])
                - target_state(states, k, t[j] * 3600)[0]
            )
    return np.r_[pack_initial(trajectory_data["initial"]), r.ravel(), t]


def evaluate(variables, node_ids, branches, states, jac=True):
    """还原绝对节点，计算脉冲并把Jacobian转到目标相对坐标。

    访问时刻的链式法则：dΔv/dt += (dΔv/dr)·v_target；h→s再乘3600。"""
    arc_count = len(branches)
    time_offset = 6 + 3 * arc_count
    x = variables.copy()
    velocities = {}
    for j, k in enumerate(node_ids[1:]):
        if k >= 0:
            r, v = target_state(states, k, variables[time_offset + j] * 3600)
            x[6 + 3 * j : 9 + 3 * j] = (r + variables[6 + 3 * j : 9 + 3 * j]) / 30000
            velocities[j] = v
    delta_v, impulse_jacobian, rad = pulse_model(x, branches, jac)
    if jac:
        for j, v in velocities.items():
            columns = impulse_jacobian[:, 6 + 3 * j : 9 + 3 * j].copy() / 30000
            impulse_jacobian[:, time_offset + j] += columns @ v * 3600
            impulse_jacobian[:, 6 + 3 * j : 9 + 3 * j] = columns
    return (delta_v, impulse_jacobian, rad, x)


def refine(
    data,
    states,
    minutes,
    output=None,
    weights=None,
    max_iterations=1500,
    encounter_radius=0.995,
    terminal_time_limit_hours=240.0,
):
    """J₂全链SCP：线性化脉冲→解SOCP→非线性回代→更新信赖域。

    每个脉冲的二阶锥上界形成Σ||Δv||目标，访问偏移限制在0.995 km球内。
    初轨a位于590–610 km高度范围，偏心率小于0.001，时刻严格递增。
    step为缩放后的步长，radius为信赖域；CLARABEL求解当前凸子问题。
    用1、1/2、1/4…回溯检查真实费用下降和202 km高度裕量。
    实际/预测下降比决定扩张或缩短信赖域；停滞、步长或改善量达到阈值后停止。"""
    if not np.isfinite(encounter_radius) or not 0 < encounter_radius <= 10000:
        raise ValueError("Invalid continuation radius")
    trajectory_data = copy.deepcopy(data)
    node_ids = np.array(trajectory_data["node_ids"])
    branches = np.array(trajectory_data["branches"])
    arc_count = len(branches)
    time_offset = 6 + 3 * arc_count
    visits = np.flatnonzero(node_ids[1:] >= 0)
    variables = pack(trajectory_data, states)
    scales = np.r_[
        [0.04, 0.04, 0.04, 0.03, 0.05, 0.05],
        np.full(3 * arc_count, 0.015),
        np.full(arc_count, 0.08),
    ]
    for j in visits:
        scales[6 + 3 * j : 9 + 3 * j] = 0.25
    weights = np.ones(arc_count) if weights is None else np.array(weights)
    if weights.shape != (arc_count,) or np.any(weights <= 0) or (not np.all(np.isfinite(weights))):
        raise ValueError("Bad weights")
    radius = 0.3
    deadline = time.monotonic() + minutes * 60
    history = []
    stalls = 0
    for iteration in range(max_iterations):
        if time.monotonic() > deadline:
            break
        delta_v, impulse_jacobian, _, _ = evaluate(variables, node_ids, branches, states)
        old = float(weights @ np.linalg.norm(delta_v, axis=1))
        step = cp.Variable(len(variables))
        bound = cp.Variable(arc_count)
        dx = cp.multiply(scales, step)
        zz = variables + dx
        impulses = cp.reshape(
            delta_v.ravel() + sp.csc_matrix(impulse_jacobian) @ dx, (arc_count, 3), order="C"
        )
        node_coordinates = cp.reshape(zz[6:time_offset], (arc_count, 3), order="C")
        node_times = zz[time_offset:]
        cons = [
            cp.norm(impulses, axis=1) <= bound,
            cp.abs(step) <= radius,
            zz[1] >= -1.5707,
            zz[1] <= 1.5707,
            zz[3] >= -0.999999,
            zz[3] <= 0.999999,
            cp.norm(zz[4:6]) <= 0.9985,
            node_times[0] >= trajectory_data.get("minimum_gap_hours", 10.0 / 3600),
            node_times[-1] <= min(240.0, float(terminal_time_limit_hours)),
            node_times[1:] - node_times[:-1]
            >= trajectory_data.get("minimum_gap_hours", 10.0 / 3600),
            cp.norm(node_coordinates[visits], axis=1) <= encounter_radius,
        ]
        problem = cp.Problem(cp.Minimize(weights @ bound + 1e-09 * cp.sum_squares(step)), cons)
        try:
            problem.solve(
                solver="CLARABEL",
                tol_gap_abs=1e-10,
                tol_feas=1e-10,
                tol_gap_rel=1e-10,
                max_iter=200,
            )
        except cp.error.SolverError:
            radius *= 0.4
            continue
        if step.value is None:
            radius *= 0.4
            continue
        if (
            not np.all(np.isfinite(step.value))
            or bound.value is None
            or (not np.all(np.isfinite(bound.value)))
        ):
            radius *= 0.4
            continue
        predicted = old - float(weights @ bound.value)
        accepted = False
        value = old
        for fraction in [1.0, 0.5, 0.25, 0.125, 0.0625]:
            trial = variables + np.array(step.value) * scales * fraction
            if np.linalg.norm(trial[4:6]) > 0.99850001:
                continue
            try:
                p, _, rad, _ = evaluate(trial, node_ids, branches, states, False)
                value = float(weights @ np.linalg.norm(p, axis=1))
            except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
                continue
            if rad >= RE + 202 and value < old - 1e-10:
                variables = trial
                accepted = True
                break
        if accepted:
            stalls = 0
            ratio = (old - value) / max(predicted, 1e-15)
            if ratio > 0.65:
                radius = min(6.0, radius * 1.5)
            elif ratio < 0.15:
                radius = max(1e-05, radius * 0.6)
        else:
            stalls += 1
            radius *= 0.4
            value = old
        rec = dict(
            iteration=iteration,
            objective=value,
            predicted=predicted,
            radius=radius,
            accepted=accepted,
        )
        history.append(rec)
        if iteration % 10 == 0 or not accepted:
            print("REDUCED_SCP", json.dumps(rec), flush=True)
        if stalls >= 10 or radius < 1e-07 or (predicted < 5e-10 and old - value < 5e-10):
            break
    delta_v, _, rad, x = evaluate(variables, node_ids, branches, states, False)
    init = initial_elliptic(x)
    trajectory_data.update(
        initial=init.tolist(),
        positions=np.vstack([init[:3], x[6:time_offset].reshape(arc_count, 3) * 30000]).tolist(),
        times=np.r_[0, x[time_offset:] * 3600].tolist(),
        cost=float(np.linalg.norm(delta_v, axis=1).sum()),
        reduced_scp_history=history,
        minimum_arc_radius_km=float(rad),
    )
    ensure_offsets(trajectory_data, states)
    if output is not None:
        output.write_text(json.dumps(trajectory_data, indent=2), encoding="utf-8")
    print("REDUCED_RESULT", trajectory_data["cost"], flush=True)
    return trajectory_data


def repair(data, states, targets, radius_km, penalty, seconds=50):
    """弹性SCP：允许指定目标暂时超出访问球，以L1距离违约惩罚引导修复。

    目标=真实脉冲范数之和 + penalty·Σmax(||offset||−radius_km,0)。
    未释放目标仍严格在0.995 km访问球内。该惩罚目标不能作为有效解成绩。
    外层逐级增大penalty，恢复可行性后再调用refine优化真实费用。"""
    trajectory_data = copy.deepcopy(data)
    node_ids = np.array(trajectory_data["node_ids"])
    branches = np.array(trajectory_data["branches"])
    arc_count = len(branches)
    time_offset = 6 + 3 * arc_count
    variables = pack(trajectory_data, states)
    selected = np.array([j for j, k in enumerate(node_ids[1:]) if k in targets], dtype=int)
    held = np.array(
        [j for j, k in enumerate(node_ids[1:]) if k >= 0 and k not in targets], dtype=int
    )
    scales = np.r_[
        [0.04, 0.04, 0.04, 0.03, 0.05, 0.05], np.full(3 * arc_count, 0.015), np.full(arc_count, 0.2)
    ]
    for j in held:
        scales[6 + 3 * j : 9 + 3 * j] = 0.25
    for j in selected:
        scales[6 + 3 * j : 9 + 3 * j] = 500.0
    trust = 0.5
    deadline = time.monotonic() + seconds
    history = []

    def excess(q):
        offsets = q[6:time_offset].reshape(arc_count, 3)[selected]
        return np.maximum(np.linalg.norm(offsets, axis=1) - radius_km, 0)

    for iteration in range(300):
        if time.monotonic() > deadline:
            break
        delta_v, impulse_jacobian, rad, _ = evaluate(variables, node_ids, branches, states)
        old_cost = float(np.linalg.norm(delta_v, axis=1).sum())
        old = old_cost + penalty * float(excess(variables).sum())
        step = cp.Variable(len(variables))
        epigraph = cp.Variable(arc_count)
        slack = cp.Variable(len(selected), nonneg=True)
        dx = cp.multiply(scales, step)
        q = variables + dx
        node_coordinates = cp.reshape(q[6:time_offset], (arc_count, 3), order="C")
        node_times = q[time_offset:]
        impulse = cp.reshape(
            delta_v.ravel() + sp.csc_matrix(impulse_jacobian) @ dx, (arc_count, 3), order="C"
        )
        constraints = [
            cp.norm(impulse, axis=1) <= epigraph,
            cp.abs(step) <= trust,
            q[1] >= -1.5707,
            q[1] <= 1.5707,
            q[3] >= -0.999999,
            q[3] <= 0.999999,
            cp.norm(q[4:6]) <= 0.9985,
            node_times[0] >= 0.05,
            node_times[-1] <= 240,
            node_times[1:] - node_times[:-1] >= trajectory_data.get("minimum_gap_hours", 0.005),
            cp.norm(node_coordinates[selected], axis=1) <= radius_km + slack,
        ]
        if len(held):
            constraints.append(cp.norm(node_coordinates[held], axis=1) <= 0.995)
        model = cp.Problem(
            cp.Minimize(cp.sum(epigraph) + penalty * cp.sum(slack) + 1e-08 * cp.sum_squares(step)),
            constraints,
        )
        try:
            model.solve(
                solver="CLARABEL",
                tol_gap_abs=1e-08,
                tol_gap_rel=1e-08,
                tol_feas=1e-09,
                max_iter=200,
            )
        except cp.error.SolverError:
            trust *= 0.4
            continue
        if step.value is None or not np.all(np.isfinite(step.value)):
            trust *= 0.4
            continue
        predicted = old - float(model.value)
        accepted = False
        new_cost = old_cost
        for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial = variables + np.asarray(step.value) * scales * fraction
            try:
                pulses, _, minimum, _ = evaluate(trial, node_ids, branches, states, False)
                cost = float(np.linalg.norm(pulses, axis=1).sum())
            except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
                continue
            merit = cost + penalty * float(excess(trial).sum())
            if minimum >= RE + 202 and merit < old - 1e-09:
                variables = trial
                accepted = True
                new_cost = cost
                ratio = (old - merit) / max(predicted, 1e-12)
                trust = (
                    min(10, trust * 1.4)
                    if ratio > 0.6
                    else max(1e-05, trust * 0.6) if ratio < 0.1 else trust
                )
                break
        if not accepted:
            trust *= 0.4
        item = dict(
            iteration=iteration,
            cost=new_cost,
            excess_km=float(excess(variables).sum()),
            accepted=accepted,
            trust=trust,
            penalty=penalty,
            radius_km=np.asarray(radius_km).tolist(),
        )
        history.append(item)
        if iteration % 10 == 0:
            print("ELASTIC", json.dumps(item), flush=True)
        if trust < 1e-07:
            break
    delta_v, _, rad, x = evaluate(variables, node_ids, branches, states, False)
    init = initial_elliptic(variables[:6])
    positions = np.vstack([init[:3], x[6:time_offset].reshape(arc_count, 3) * 30000])
    trajectory_data.update(
        initial=init.tolist(),
        positions=positions.tolist(),
        times=np.r_[0, variables[time_offset:] * 3600].tolist(),
        cost=float(np.linalg.norm(delta_v, axis=1).sum()),
        minimum_arc_radius_km=float(rad),
        elastic_excess_km=float(excess(variables).sum()),
        elastic_history=history,
    )
    trajectory_data.pop("encounter_offsets_km", None)
    return trajectory_data
