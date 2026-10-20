"""J₂ 轨迹的弧段与自由节点操作，所有位置在极轴坐标系中。

trajectory 返回每段出发/到达速度、STM、各节点脉冲、总Δv、最小半径。
regularize 分段缓解长弧打靶病态；insert_coast 插入零脉冲节点；coarsen 合并弱脉冲。
这些是动力学和节点结构计算，不读文件、不持有星历全局变量。
"""

import copy
import numpy as np
from orbit_dynamics import RE, norm, lambert, target_state, shoot, propagate


def trajectory(data):
    r = np.array(data["positions"])
    t = np.array(data["times"])
    b = np.array(data["branches"])
    n = len(b)
    vi = []
    vf = []
    phi = []
    minimum = np.inf
    for i in range(n):
        a, z, P, rad, error = shoot(r[i], r[i + 1], t[i + 1] - t[i], b[i])
        if error > 0.0001 or not np.all(np.isfinite(a)):
            raise ValueError("Invalid J2 arc")
        vi.append(a)
        vf.append(z)
        phi.append(P)
        minimum = min(minimum, rad)
    vi = np.array(vi)
    vf = np.array(vf)
    pulses = vi - np.vstack([np.array(data["initial"][3:]), vf[:-1]])
    return (vi, vf, np.array(phi), pulses, float(np.linalg.norm(pulses, axis=1).sum()), minimum)


def insertion_maps(phi_total, phi_partial):
    onward = np.linalg.solve(phi_partial.T, phi_total.T).T
    Q = np.linalg.solve(phi_total[:3, 3:], onward[:3, 3:])
    D = onward[3:, 3:] - phi_total[3:, 3:] @ Q
    return (Q, D)


def branch_near(r0, r1, dt, expected, maxrev=3):
    choices = []
    for rev in range(maxrev + 1):
        for major in range(2):
            for side in range(1 if rev == 0 else 2):
                b = np.array([major, rev, side])
                a, _ = lambert(r0, r1, dt, *b)
                if np.all(np.isfinite(a)):
                    choices.append((norm(a - expected), b))
    for _, b in sorted(choices, key=lambda q: q[0])[:3]:
        a, z, P, rad, error = shoot(r0, r1, dt, b)
        if error < 0.0001 and norm(a - expected) < 0.2 and (rad >= RE + 202):
            return (b.tolist(), a, z)
    raise ValueError("Cannot identify the continuous J2 branch")


def coarsen(data, threshold=2e-05):
    d = copy.deepcopy(data)
    removed = 0
    for _ in range(10):
        vi, vf, _, pulses, baseline, _ = trajectory(d)
        accepted = False
        for k in np.argsort(np.linalg.norm(pulses, axis=1)):
            if k == 0 or d["node_ids"][k] >= 0 or norm(pulses[k]) > threshold:
                continue
            q = copy.deepcopy(d)
            r = np.array(q["positions"])
            t = np.array(q["times"])
            try:
                b, _, _ = branch_near(r[k - 1], r[k + 1], t[k + 1] - t[k - 1], vi[k - 1])
            except ValueError:
                continue
            for key in ["positions", "times", "node_ids"]:
                q[key].pop(int(k))
            q["branches"][k - 1 : k + 1] = [b]
            try:
                _, _, _, _, cost, rad = trajectory(q)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if rad >= RE + 202 and cost <= baseline + 1e-06:
                q["cost"] = cost
                d = q
                removed += 1
                accepted = True
                break
        if not accepted:
            break
    print("COARSEN", removed, "nodes removed", flush=True)
    return d


def regularize(d, max_angle=110.0):
    """Repair ill-conditioned long arcs by shorter, strictly converged BVPs.

    A long arc's unconverged endpoint is used only to initialize a midpoint.
    It is not accepted as a valid coast: both new BVPs must independently
    converge and the resulting entire trajectory is reevaluated strictly.
    """
    d = copy.deepcopy(d)
    i, inserted, records = (0, 0, [])
    while i < len(d["branches"]):
        r = np.array(d["positions"])
        t = np.array(d["times"])
        b = np.array(d["branches"][i])
        angle = np.degrees(
            np.arccos(
                np.clip(r[i] @ r[i + 1] / np.linalg.norm(r[i]) / np.linalg.norm(r[i + 1]), -1, 1)
            )
        )
        angle = (360 - angle if b[0] else angle) + 360 * b[1]
        vi, _, _, _, err = shoot(r[i], r[i + 1], t[i + 1] - t[i], b)
        if angle <= max_angle and err < 0.0001:
            i += 1
            continue
        if not np.all(np.isfinite(vi)) or err > 0.02:
            raise ValueError(f"Long-arc seed residual too large: arc={i}, error_km={err}")
        dt = (t[i + 1] - t[i]) / 2
        if dt < 20:
            raise ValueError("Cannot split a short invalid coast")
        midpoint, _, _ = propagate(r[i], vi, dt)
        b0, _, _ = branch_near(r[i], midpoint[:3], dt, vi)
        b1, _, _ = branch_near(midpoint[:3], r[i + 1], dt, midpoint[3:])
        records.append(
            dict(arc=i, seed_endpoint_residual_m=float(err * 1000), angle_deg=float(angle))
        )
        d["positions"].insert(i + 1, midpoint[:3].tolist())
        d["times"].insert(i + 1, float(t[i] + dt))
        d["node_ids"].insert(i + 1, -1)
        d["branches"][i : i + 1] = [b0, b1]
        inserted += 1
        if inserted > 250:
            raise ValueError("Excessive mesh regularization")
    _, _, _, _, d["cost"], rad = trajectory(d)
    d["minimum_arc_radius_km"] = rad
    d["external_regularization"] = records
    print("EXTERNAL_REGULARIZE", inserted, d["cost"], flush=True)
    return d


def insert_coast(data, arc, fraction):
    """Insert a dynamically continuous zero-burn node, preserving every visit."""
    d = copy.deepcopy(data)
    r = np.array(d["positions"])
    t = np.array(d["times"])
    vi, _, _, _, err = shoot(r[arc], r[arc + 1], t[arc + 1] - t[arc], np.array(d["branches"][arc]))
    if err > 0.0001:
        raise ValueError("Invalid coast to subdivide")
    duration = (t[arc + 1] - t[arc]) * fraction
    if min(duration, t[arc + 1] - t[arc] - duration) < 20:
        raise ValueError("Too short a new coast")
    state, _, _ = propagate(r[arc], vi, duration)
    b0, _, _ = branch_near(r[arc], state[:3], duration, vi)
    b1, _, _ = branch_near(state[:3], r[arc + 1], t[arc + 1] - t[arc] - duration, state[3:])
    d["positions"].insert(arc + 1, state[:3].tolist())
    d["times"].insert(arc + 1, float(t[arc] + duration))
    d["node_ids"].insert(arc + 1, -1)
    d["branches"][arc : arc + 1] = [b0, b1]
    d.pop("encounter_offsets_km", None)
    return d


def remove_burn(data, k):
    """Bridge across a free burn, allowing a real finite fuel increase."""
    if k <= 0 or k >= len(data["times"]) - 1 or data["node_ids"][k] >= 0:
        raise ValueError("Only an interior free node can be removed")
    vi, vf, _, _, _, _ = trajectory(data)
    r = np.array(data["positions"])
    t = np.array(data["times"])
    incoming = np.array(data["initial"][3:]) if k == 1 else vf[k - 2]
    outgoing = vi[k + 1] if k + 1 < len(vi) else None
    choices = []
    for rev in range(3):
        for major in range(2):
            for side in range(1 if rev == 0 else 2):
                b = np.array([major, rev, side])
                a, z = lambert(r[k - 1], r[k + 1], t[k + 1] - t[k - 1], *b)
                if not np.all(np.isfinite(a)):
                    continue
                screening = np.linalg.norm(a - incoming) + (
                    np.linalg.norm(outgoing - z) if outgoing is not None else 0
                )
                choices.append((screening, b))
    best = None
    for _, b in sorted(choices, key=lambda q: q[0])[:5]:
        try:
            a, z, _, rad, err = shoot(r[k - 1], r[k + 1], t[k + 1] - t[k - 1], b)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if err > 0.0001 or rad < RE + 202 or (not np.all(np.isfinite(a))):
            continue
        value = np.linalg.norm(a - incoming) + (
            np.linalg.norm(outgoing - z) if outgoing is not None else 0
        )
        if best is None or value < best[0]:
            best = (value, b.tolist())
    if best is None:
        raise ValueError("No feasible J2 bridge after burn removal")
    d = copy.deepcopy(data)
    for key in ("positions", "times", "node_ids"):
        d[key].pop(k)
    d["branches"][k - 1 : k + 1] = [best[1]]
    d.pop("encounter_offsets_km", None)
    return d


def ensure_offsets(d, states):
    offsets = np.zeros_like(np.array(d["positions"]))
    for j, k in enumerate(d["node_ids"]):
        if k >= 0:
            offsets[j] = np.array(d["positions"][j]) - target_state(states, k, d["times"][j])[0]
    d["encounter_offsets_km"] = offsets.tolist()
    return d


def add_zero_nodes(data, peaks):
    d = copy.deepcopy(data)
    for candidate in sorted(peaks, key=lambda v: v["arc"], reverse=True):
        i = candidate["arc"]
        r = np.array(d["positions"])
        t = d["times"]
        tm = candidate["time_s"]
        rm = candidate["position"]
        b0, _, _ = branch_near(r[i], rm, tm - t[i], candidate["v1"])
        b1, _, _ = branch_near(rm, r[i + 1], t[i + 1] - tm, candidate["v2"])
        d["positions"].insert(i + 1, rm.tolist())
        d["times"].insert(i + 1, tm)
        d["node_ids"].insert(i + 1, -1)
        d["branches"][i : i + 1] = [b0, b1]
    d["minimum_gap_hours"] = 10.0 / 3600
    d["cost"] = trajectory(d)[4]
    return d
