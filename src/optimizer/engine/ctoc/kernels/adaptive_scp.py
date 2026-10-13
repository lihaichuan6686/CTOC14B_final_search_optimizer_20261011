"""J2 adaptive impulse mesh: STM insertion tests, coarsening, and SOCP.

The inserted impulse w changes the endpoint burns by -Q w and -D w,
where Q=Phi_rv(T,0)^-1 Phi_rv(T,t) and
D=Phi_vv(T,t)-Phi_vv(T,0) Q. This preserves both end states to first
order. We minimize the three exact vector norms, including zero burns.
Candidates are accepted only after nonlinear J2 shooting lowers the cost.
Inspired by Beauregard, Izzo & Acciarini (arXiv:2405.18903); this is a
local insertion test, not a global optimality certificate.
"""

import copy
import numpy as np
from scipy.optimize import minimize
from .physics import lambert, norm, RE
from .j2_refine import shoot, propagate


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


def insertion_step(d0, d1, Q, D, limit=0.04):

    def fun(w):
        u = d0 - Q @ w
        v = d1 - D @ w
        a = np.sqrt(u @ u + 1e-16)
        z = np.sqrt(v @ v + 1e-16)
        m = np.sqrt(w @ w + 1e-16)
        return (a + z + m, -Q.T @ u / a - D.T @ v / z + w / m)

    result = minimize(
        fun,
        np.zeros(3),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-limit, limit)] * 3,
        options={"maxiter": 100, "ftol": 1e-13, "gtol": 1e-08, "maxls": 35},
    )
    w = result.x
    gain = norm(d0) + norm(d1) - norm(d0 - Q @ w) - norm(d1 - D @ w) - norm(w)
    return (float(gain), w)


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


def scan(data):
    vi, vf, phi, pulses, cost, rad = trajectory(data)
    r = np.array(data["positions"])
    t = np.array(data["times"])
    candidates = []
    for i in range(len(vi)):
        duration = t[i + 1] - t[i]
        if duration < 300:
            continue
        d1 = pulses[i + 1] if i + 1 < len(vi) else np.zeros(3)
        if norm(pulses[i]) + norm(d1) < 1e-05:
            continue
        for fraction in [0.15, 0.3, 0.5, 0.7, 0.85]:
            if min(fraction, 1 - fraction) * duration < 45:
                continue
            state, P, _ = propagate(r[i], vi[i], duration * fraction)
            try:
                Q, D = insertion_maps(phi[i], P)
                gain, w = insertion_step(pulses[i], d1, Q, D)
            except np.linalg.LinAlgError:
                continue
            if gain > 1e-06:
                candidates.append(
                    {
                        "arc": i,
                        "fraction": fraction,
                        "predicted_gain": gain,
                        "w": w,
                        "position": state[:3] - P[:3, 3:] @ Q @ w,
                        "v1": vi[i] - Q @ w,
                        "v2": state[3:] + w,
                    }
                )
    candidates.sort(key=lambda d: d["predicted_gain"], reverse=True)
    return (candidates, cost)


def insert(data, candidate):
    d = copy.deepcopy(data)
    i = candidate["arc"]
    r = np.array(d["positions"])
    t = np.array(d["times"])
    tm = t[i] + candidate["fraction"] * (t[i + 1] - t[i])
    rm = candidate["position"]
    b0, _, _ = branch_near(r[i], rm, tm - t[i], candidate["v1"])
    b1, _, _ = branch_near(rm, r[i + 1], t[i + 1] - tm, candidate["v2"])
    d["positions"].insert(i + 1, rm.tolist())
    d["times"].insert(i + 1, float(tm))
    d["node_ids"].insert(i + 1, -1)
    d["branches"][i : i + 1] = [b0, b1]
    d["minimum_gap_hours"] = 10.0 / 3600
    _, _, _, _, cost, rad = trajectory(d)
    if rad < RE + 202:
        raise ValueError("Insertion height margin violated")
    d["cost"] = cost
    return d


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


def condition_j2(data, max_angle=120.0):
    """Split the same J2 coast before its endpoint shooting gets singular."""
    d = copy.deepcopy(data)
    i = 0
    added = 0
    while i < len(d["branches"]):
        r = np.array(d["positions"])
        t = np.array(d["times"])
        b = d["branches"][i]
        angle = np.degrees(
            np.arccos(np.clip(np.dot(r[i], r[i + 1]) / norm(r[i]) / norm(r[i + 1]), -1, 1))
        )
        if b[0]:
            angle = 360 - angle
        angle += 360 * b[1]
        if angle <= max_angle:
            i += 1
            continue
        vi, vf, _, _, error = shoot(r[i], r[i + 1], t[i + 1] - t[i], np.array(b))
        if error > 0.001:
            raise ValueError("Cannot subdivide invalid J2 coast")
        dt = (t[i + 1] - t[i]) / 2
        mid, _, _ = propagate(r[i], vi, dt)
        b0, _, _ = branch_near(r[i], mid[:3], dt, vi)
        b1, _, _ = branch_near(mid[:3], r[i + 1], dt, mid[3:])
        d["positions"].insert(i + 1, mid[:3].tolist())
        d["times"].insert(i + 1, float(t[i] + dt))
        d["node_ids"].insert(i + 1, -1)
        d["branches"][i : i + 1] = [b0, b1]
        added += 1
        if added > 250:
            raise ValueError("Excessive subdivision")
    d["cost"] = trajectory(d)[4]
    print("J2_SUBDIVIDE", added, "cost", d["cost"], flush=True)
    return d
