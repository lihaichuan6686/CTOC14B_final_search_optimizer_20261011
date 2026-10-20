"""由全链脉冲的次梯度构造 primer，寻找值得增加自由脉冲的弧段。

primer 范数超过 1 是局部插入机会；插入后的收益仍由全链优化及实际传播决定。
"""

import numpy as np
import scipy.sparse as sp
import cvxpy as cp
from orbit_dynamics import target_state, norm, propagate
from j2_optimization import pulse_model, pack_initial
from impulse_nodes import trajectory, insertion_maps


def subgradients(data, states):
    ids = np.array(data["node_ids"])
    n = len(ids) - 1
    off = 6 + 3 * n
    visits = np.flatnonzero(ids >= 0)
    x = np.r_[
        pack_initial(data["initial"]),
        np.array(data["positions"])[1:].ravel() / 30000,
        np.array(data["times"])[1:] / 3600,
    ]
    dv, J, _ = pulse_model(x, np.array(data["branches"]))
    centers = []
    velocities = []
    for j in visits:
        r, v = target_state(states, ids[j], x[off + j - 1] * 3600)
        centers.append(r)
        velocities.append(v)
    scales = np.r_[[0.02, 0.02, 0.02, 0.02, 0.03, 0.03], np.full(3 * n, 0.001), np.full(n, 0.02)]
    delta = cp.Variable(len(x))
    bound = cp.Variable(n)
    dx = cp.multiply(scales, delta)
    xx = x + dx
    pulse = cp.reshape(dv.ravel() + sp.csc_matrix(J) @ dx, (n, 3), order="C")
    cone = cp.SOC(bound, pulse, axis=1)
    tt = xx[off:]
    rr = cp.reshape(xx[6:off], (n, 3), order="C") * 30000
    moving = np.array(centers) + cp.multiply(
        np.array(velocities),
        cp.reshape((tt[visits - 1] - x[off:][visits - 1]) * 3600, (len(visits), 1), order="C"),
    )
    constraints = [
        cone,
        cp.abs(delta) <= 0.01,
        xx[1] >= -1.5707,
        xx[1] <= 1.5707,
        xx[3] >= -0.999999,
        xx[3] <= 0.999999,
        cp.norm(xx[4:6]) <= 0.9985,
        tt[0] >= data.get("minimum_gap_hours", 10.0 / 3600),
        tt[-1] <= 240,
        tt[1:] - tt[:-1] >= data.get("minimum_gap_hours", 10.0 / 3600),
        cp.norm(rr[visits - 1] - moving, axis=1) <= 0.995,
    ]
    problem = cp.Problem(cp.Minimize(cp.sum(bound) + 1e-09 * cp.sum_squares(delta)), constraints)
    problem.solve(
        solver="CLARABEL", tol_gap_abs=1e-10, tol_feas=1e-10, tol_gap_rel=1e-10, max_iter=200
    )
    if (
        problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        or cone.dual_value is None
        or cone.dual_value[1] is None
    ):
        raise ValueError(f"No coupled cone dual: solver status={problem.status}")
    p = -np.array(cone.dual_value[1])
    if p.shape == (3, n):
        p = p.T
    if p.shape != (n, 3) or not np.isfinite(p).all() or np.max(np.linalg.norm(p, axis=1)) > 1.001:
        raise ValueError("Invalid subgradients")
    return (p, float(np.linalg.norm(dv, axis=1).sum() - problem.value))


def scan(data, states, samples=61):
    p, gain = subgradients(data, states)
    vi, vf, phi, pulses, cost, rad = trajectory(data)
    r = np.array(data["positions"])
    t = np.array(data["times"])
    records = []
    peaks = []
    for i in range(len(vi)):
        duration = t[i + 1] - t[i]
        nextp = p[i + 1] if i + 1 < len(vi) else np.zeros(3)
        best = None
        for fraction in np.linspace(0.03, 0.97, samples):
            if min(fraction, 1 - fraction) * duration < 20:
                continue
            state, P, _ = propagate(r[i], vi[i], duration * fraction)
            Q, D = insertion_maps(phi[i], P)
            magnitude = float(norm(Q.T @ p[i] + D.T @ nextp))
            item = dict(
                arc=i,
                time_s=float(t[i] + fraction * duration),
                fraction=float(fraction),
                primer_norm=magnitude,
            )
            records.append(item)
            if best is None or magnitude > best["primer_norm"]:
                best = {
                    **item,
                    "position": state[:3].copy(),
                    "v1": vi[i].copy(),
                    "v2": state[3:].copy(),
                }
        if best:
            peaks.append(best)
    peaks.sort(key=lambda item: item["primer_norm"], reverse=True)
    return (
        peaks,
        dict(
            cost_km_s=cost,
            full_mesh_linearized_gain_km_s=gain,
            samples=records,
            max_primer_norm=peaks[0]["primer_norm"] if peaks else 0.0,
            interpretation="Local coupled single-insertion check, not a global optimality certificate.",
        ),
    )
